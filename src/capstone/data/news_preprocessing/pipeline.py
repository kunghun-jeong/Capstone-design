"""전처리 파이프라인 조립: raw JSONL → interim/processed parquet + 실행 로그."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .aggregate import aggregate_ticker_day
from .config import Config
from .dedup import dedup_cross_source, explode_by_ticker
from .normalize import normalize, read_jsonl_tree
from .trading_calendar import attribute_trading_day, trading_sessions


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_tables(alpaca: list[dict], massive: list[dict], cfg: Config) -> dict[str, pd.DataFrame]:
    """원본 레코드 → {alpaca_normalized, massive_normalized, articles, article_ticker, ticker_day}. 디스크 접근 없음."""
    universe = set(cfg.tickers)
    frames = {"alpaca": normalize(alpaca, "alpaca"), "massive": normalize(massive, "massive")}
    combined = pd.concat(list(frames.values()), ignore_index=True)
    combined = combined[combined["tickers"].map(lambda ts: bool(universe & set(ts)))].reset_index(drop=True)
    if combined.empty:
        raise ValueError("유니버스 종목이 태그된 기사가 없습니다. 입력 경로/티커를 확인하세요.")

    combined["attributed_day"] = attribute_trading_day(combined["published_utc"], cfg.cutoff)
    sessions = trading_sessions(cfg.warmup_start, cfg.end)
    in_range = combined["attributed_day"].between(sessions[0], sessions[-1])
    combined = combined[in_range].reset_index(drop=True)

    articles = dedup_cross_source(combined)
    article_ticker = explode_by_ticker(articles, cfg.tickers, cfg.aliases, cfg.broad_tag_threshold)
    ticker_day = aggregate_ticker_day(
        article_ticker, cfg.tickers, cfg.warmup_start, cfg.end,
        window=cfg.window, min_periods=cfg.min_periods, scale_floor=cfg.scale_floor,
    )
    return {
        "alpaca_normalized": frames["alpaca"], "massive_normalized": frames["massive"],
        "articles": articles, "article_ticker": article_ticker, "ticker_day": ticker_day,
    }


def run(cfg: Config, root: Path, config_path: Path) -> dict:
    inputs = {name: root / cfg.paths[name] for name in ("alpaca_raw", "massive_raw")}
    files = sorted(p for d in inputs.values() for p in d.glob("*/[0-9][0-9][0-9][0-9].jsonl"))
    if not files:
        raise FileNotFoundError(f"입력 JSONL이 없습니다: {[str(p) for p in inputs.values()]}")
    tables = build_tables(read_jsonl_tree(inputs["alpaca_raw"]), read_jsonl_tree(inputs["massive_raw"]), cfg)

    interim, processed = root / cfg.paths["interim"], root / cfg.paths["processed"]
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    targets = {
        "alpaca_normalized": interim, "massive_normalized": interim, "articles": interim,
        "article_ticker": processed, "ticker_day": processed,
    }
    for name, directory in targets.items():
        tables[name].to_parquet(directory / f"{name}.parquet", index=False)

    log = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(root),
        "config_sha256": _sha256(config_path),
        "inputs": {str(p.relative_to(root)): _sha256(p) for p in files},
        "rows": {name: len(df) for name, df in tables.items()},
    }
    (processed / "_run.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return log
