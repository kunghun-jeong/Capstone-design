"""전처리 파이프라인 조립.

1단계: raw JSONL → ``interim/posts_comments.parquet``(언급 글, 플래그 포함, 필터 전) + ``interim/daily_activity.parquet``
2단계: interim → ``processed/article_ticker.parquet``, ``ticker_day.parquet``, ``ticker_day_by_subreddit.parquet``
실행 로그(``_run_stage1.json``, ``_run.json``)에 config 해시·입력 파일 해시·git sha·단계별 건수를 남긴다.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..news_preprocessing.pipeline import _git_sha, _sha256
from .aggregate import aggregate_by_subreddit, aggregate_reddit
from .clean import explode_by_ticker, merge_copies, remove_noise
from .config import Config
from .normalize import ROW_COLUMNS, TickerMatcher, list_raw_files, process_file


def _process(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    return process_file(*args)


def build_stage1(files: list[tuple[Path, str, str]], cfg: Config, workers: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(path, sub, kind, cfg) for path, sub, kind in files]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_process, jobs))
    else:
        results = [_process(j) for j in jobs]
    rows = pd.concat([r for r, _ in results if not r.empty], ignore_index=True) if results else pd.DataFrame(columns=ROW_COLUMNS)
    if rows.empty:
        raise ValueError("유니버스 종목이 언급된 글이 없습니다. 입력 경로/티커를 확인하세요.")
    dup = rows["article_id"].duplicated()
    if dup.any():   # 원본 폴더 기간이 겹치면 같은 글이 두 번 들어온다
        raise ValueError(f"article_id 중복 {int(dup.sum())}건 — paths.raw 폴더들의 수집 기간이 겹치는지 확인하세요.")
    activity = pd.concat([a for _, a in results if not a.empty], ignore_index=True)
    activity = activity.groupby(["subreddit", "kind", "day"], as_index=False).sum()
    rows = rows.sort_values(["published_utc", "article_id"]).reset_index(drop=True)
    return rows, activity


def build_stage2(rows: pd.DataFrame, activity: pd.DataFrame, cfg: Config) -> tuple[dict[str, pd.DataFrame], dict]:
    """디스크 접근 없음. 반환: (테이블들, 단계별 건수)."""
    lo, hi = pd.Timestamp(cfg.baseline_start), pd.Timestamp(cfg.end)
    rows = rows[rows["attributed_day"].between(lo, hi)].reset_index(drop=True)
    stats: dict = {"stage1_rows_in_range": len(rows)}

    mention = rows[rows["n_tags"] > 0] if not cfg.include_name_only else rows
    stats["name_only_excluded"] = int(len(rows) - len(mention))
    clean, removed = remove_noise(mention, cfg)
    stats["removed"] = removed
    articles = merge_copies(clean)
    stats["merged_copies"] = int(len(clean) - len(articles))
    stats["articles"] = len(articles)

    matcher = TickerMatcher(cfg.tickers, cfg.ticker_patterns, cfg.aliases, cfg.other_tickers)
    article_ticker = explode_by_ticker(articles, cfg, lambda title: matcher.match(title)[0])
    ticker_day = aggregate_reddit(article_ticker, activity, cfg)
    by_sub = aggregate_by_subreddit(article_ticker, activity, cfg)

    in_period = article_ticker["attributed_day"].between(pd.Timestamp(cfg.start), hi)
    tables = {
        "article_ticker": article_ticker[in_period].reset_index(drop=True),
        "ticker_day": ticker_day,
        "ticker_day_by_subreddit": by_sub,
    }
    stats["rows"] = {name: len(df) for name, df in tables.items()}
    return tables, stats


def _base_log(cfg_path: Path, root: Path) -> dict:
    return {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(root),
        "config_sha256": _sha256(cfg_path),
    }


def run_stage1(cfg: Config, root: Path, config_path: Path, workers: int | None = None) -> dict:
    files = list_raw_files([root / p for p in cfg.paths["raw"]])
    if not files:
        raise FileNotFoundError(f"입력 JSONL이 없습니다: {cfg.paths['raw']}")
    rows, activity = build_stage1(files, cfg, workers or max(1, (os.cpu_count() or 2) - 2))
    interim = root / cfg.paths["interim"]
    interim.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(interim / "posts_comments.parquet", index=False)
    activity.to_parquet(interim / "daily_activity.parquet", index=False)
    log = {
        **_base_log(config_path, root),
        "inputs": {str(p.relative_to(root)): _sha256(p) for p, _, _ in files},
        "rows": {"posts_comments": len(rows), "daily_activity": len(activity), "raw_items": int(activity["n_items"].sum())},
    }
    (interim / "_run_stage1.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return log


def run_stage2(cfg: Config, root: Path, config_path: Path) -> dict:
    interim, processed = root / cfg.paths["interim"], root / cfg.paths["processed"]
    rows = pd.read_parquet(interim / "posts_comments.parquet")
    activity = pd.read_parquet(interim / "daily_activity.parquet")
    tables, stats = build_stage2(rows, activity, cfg)
    processed.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(processed / f"{name}.parquet", index=False)
    stage1 = json.loads((interim / "_run_stage1.json").read_text(encoding="utf-8"))
    log = {**_base_log(config_path, root), "stage1": stage1, **stats}
    (processed / "_run.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return log
