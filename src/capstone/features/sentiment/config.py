"""``configs/reddit_sentiment.yaml`` 로더."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SentimentConfig:
    preprocessing_config: str
    model_name: str
    revision: str
    max_length: int
    batch_size: int
    device: str
    half_precision: str | bool
    exclude_short: bool
    sentiment_scale_floor: float
    paths: dict = field(default_factory=dict)


def load_sentiment_config(path: Path) -> SentimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model = raw["model"]
    if not str(model.get("revision") or "").strip():
        raise ValueError("model.revision을 커밋 해시로 고정하세요(재현용).")
    return SentimentConfig(
        preprocessing_config=str(raw["preprocessing_config"]),
        model_name=str(model["name"]),
        revision=str(model["revision"]),
        max_length=int(model.get("max_length", 512)),
        batch_size=int(model.get("batch_size", 64)),
        device=str(model.get("device", "auto")),
        half_precision=model.get("half_precision", "auto"),
        exclude_short=bool(raw.get("exclude_short", True)),
        sentiment_scale_floor=float(raw["sentiment_scale_floor"]),
        paths=dict(raw.get("paths") or {}),
    )
