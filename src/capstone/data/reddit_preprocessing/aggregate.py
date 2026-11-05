"""(ticker, 귀속 거래일) 집계 — 뉴스 aggregate_ticker_day를 그대로 쓰고, Reddit용 비율 버전만 덧붙인다.

비율 = 그 종목 언급 글 수(n_rows, 복붙 합치기 전 원본 수) ÷ 그날 노이즈 제거 후 전체 글 수 × 10,000.
WSB 하루 글 수가 크게 변해서 원 건수에는 커뮤니티 활동량이 섞이기 때문이다(계획 ② 2번).
분자·분모 모두 "복붙 합치기 전" 건수라 단위가 같다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..news_preprocessing.aggregate import _robust_z, aggregate_ticker_day
from .config import Config

RATE_PER = 10_000


def denominator_column(cfg: Config) -> str:
    return "n_clean_excl_removed" if cfg.drop_removed_posts else "n_clean"


def aggregate_reddit(article_ticker: pd.DataFrame, activity: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """baseline_start~end로 베이스라인을 계산한 뒤 start~end만 돌려준다."""
    lo, hi = pd.Timestamp(cfg.baseline_start), pd.Timestamp(cfg.end)
    at = article_ticker[article_ticker["attributed_day"].between(lo, hi)]
    out = aggregate_ticker_day(
        at, cfg.tickers, cfg.baseline_start, cfg.end,
        window=cfg.window, min_periods=cfg.min_periods, scale_floor=cfg.scale_floor,
    )
    denom = activity.groupby("day")[denominator_column(cfg)].sum()
    out["n_total_items"] = out["day"].map(denom).fillna(0).astype(int)
    out["mention_rate"] = out["n_rows"] / out["n_total_items"].replace(0, np.nan) * RATE_PER
    out["mention_rate_z"] = out.groupby("ticker")["mention_rate"].transform(
        lambda s: _robust_z(s, cfg.window, cfg.min_periods, cfg.ratio_scale_floor)
    )
    keep = out["day"].between(pd.Timestamp(cfg.start), hi)
    return out[keep].reset_index(drop=True)


def aggregate_by_subreddit(article_ticker: pd.DataFrame, activity: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    parts = []
    for sub in sorted(activity["subreddit"].unique()):
        part = aggregate_reddit(article_ticker[article_ticker["subreddit"] == sub], activity[activity["subreddit"] == sub], cfg)
        part.insert(0, "subreddit", sub)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
