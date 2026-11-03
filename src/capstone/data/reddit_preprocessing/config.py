"""``configs/reddit_preprocessing.yaml`` 로더."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    tickers: list[str]
    baseline_start: date
    start: date
    end: date
    cutoff: str                      # "close" 또는 "HH:MM"(America/New_York)
    window: int
    min_periods: int
    scale_floor: float
    ratio_scale_floor: float
    broad_tag_threshold: int
    min_chars: int
    drop_removed_posts: bool
    include_name_only: bool
    author_salt: str
    ticker_patterns: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    other_tickers: list[str] = field(default_factory=list)
    bots: list[str] = field(default_factory=list)
    paths: dict = field(default_factory=dict)


def _as_date(value, name: str) -> date:
    if not isinstance(value, date):
        raise ValueError(f"{name}는 YYYY-MM-DD 날짜여야 합니다: {value!r}")
    return value


def _str_list(values, name: str) -> list[str]:
    values = list(values or [])
    # 따옴표 없는 ON/NO 등이 불리언으로 파싱되는 사고를 막는다 (alpaca 리포트 9절 참고).
    if not all(isinstance(v, str) for v in values):
        raise ValueError(f"{name}는 전부 문자열이어야 합니다 (따옴표 확인): {values!r}")
    return values


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    period = raw["period"]
    base = raw["baseline"]
    paths = dict(raw.get("paths") or {})
    if isinstance(paths.get("raw"), str):
        paths["raw"] = [paths["raw"]]
    cfg = Config(
        tickers=_str_list(raw["tickers"], "tickers"),
        baseline_start=_as_date(period["baseline_start"], "period.baseline_start"),
        start=_as_date(period["start"], "period.start"),
        end=_as_date(period["end"], "period.end"),
        cutoff=str(raw.get("cutoff", "close")),
        window=int(base["window"]),
        min_periods=int(base["min_periods"]),
        scale_floor=float(base["scale_floor"]),
        ratio_scale_floor=float(base["ratio_scale_floor"]),
        broad_tag_threshold=int(raw["broad_tag_threshold"]),
        min_chars=int(raw["min_chars"]),
        drop_removed_posts=bool(raw.get("drop_removed_posts", False)),
        include_name_only=bool(raw.get("include_name_only", False)),
        author_salt=str(raw.get("author_salt", "")),
        ticker_patterns={str(k): str(v) for k, v in (raw.get("ticker_patterns") or {}).items()},
        aliases={k: _str_list(v, f"aliases.{k}") for k, v in (raw.get("aliases") or {}).items()},
        other_tickers=_str_list(raw.get("other_tickers"), "other_tickers"),
        bots=_str_list(raw.get("bots"), "bots"),
        paths=paths,
    )
    if not cfg.baseline_start <= cfg.start <= cfg.end:
        raise ValueError("period는 baseline_start <= start <= end 여야 합니다.")
    unknown = set(cfg.ticker_patterns) - set(cfg.tickers)
    if unknown:
        raise ValueError(f"ticker_patterns에 유니버스 밖 종목이 있습니다: {sorted(unknown)}")
    return cfg
