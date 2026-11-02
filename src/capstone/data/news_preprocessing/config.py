"""``configs/news_preprocessing.yaml`` 로더."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    tickers: list[str]
    warmup_start: date
    start: date
    end: date
    cutoff: str                      # "close" 또는 "HH:MM"(America/New_York)
    window: int
    min_periods: int
    scale_floor: float
    broad_tag_threshold: int
    aliases: dict[str, list[str]] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)


def _as_date(value, name: str) -> date:
    if not isinstance(value, date):
        raise ValueError(f"{name}는 YYYY-MM-DD 날짜여야 합니다: {value!r}")
    return value


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tickers = raw["tickers"]
    # 따옴표 없는 ON/NO 등이 불리언으로 파싱되는 사고를 막는다 (alpaca 리포트 9절 참고).
    if not all(isinstance(t, str) for t in tickers):
        raise ValueError(f"tickers는 전부 문자열이어야 합니다 (따옴표 확인): {tickers!r}")
    period = raw["period"]
    base = raw["baseline"]
    cfg = Config(
        tickers=list(tickers),
        warmup_start=_as_date(period["warmup_start"], "period.warmup_start"),
        start=_as_date(period["start"], "period.start"),
        end=_as_date(period["end"], "period.end"),
        cutoff=str(raw.get("cutoff", "close")),
        window=int(base["window"]),
        min_periods=int(base["min_periods"]),
        scale_floor=float(base["scale_floor"]),
        broad_tag_threshold=int(raw["broad_tag_threshold"]),
        aliases={k: list(v) for k, v in (raw.get("aliases") or {}).items()},
        paths=dict(raw.get("paths") or {}),
    )
    if not cfg.warmup_start <= cfg.start <= cfg.end:
        raise ValueError("period는 warmup_start <= start <= end 여야 합니다.")
    return cfg
