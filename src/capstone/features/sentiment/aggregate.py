"""글 단위 감성 → (ticker, 귀속 거래일) 감성 피처 + 언급량과의 상호작용.

- 한 글이 여러 종목을 언급하면 같은 점수가 각 종목에 들어간다. 그래서 단일 종목 글(is_primary)만 쓴 버전도 둔다.
- 감성 글이 0건인 날의 평균은 결측(NaN)이다. 0(중립)으로 채우지 않는다 — 채울지는 모델 단계에서 정한다.
- "평소 대비"는 언급량과 같은 robust z(t 이전 window 거래일, 당일 제외)를 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from capstone.data.news_preprocessing.aggregate import _robust_z
from capstone.data.news_preprocessing.trading_calendar import trading_sessions
from capstone.data.reddit_preprocessing.config import Config

SENTIMENT_COLUMNS = [
    "n_scored", "sent_mean", "sent_pos_share", "sent_neg_share", "sent_net",
    "n_scored_primary", "sent_mean_primary", "sent_mean_z", "sent_mean_primary_z",
]


def _daily(use: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    g = use.groupby(["ticker", "attributed_day"])
    out = pd.DataFrame({
        f"n_scored{suffix}": g.size(),
        f"sent_mean{suffix}": g["sent_score"].mean(),
    })
    if not suffix:
        out["sent_pos_share"] = g["sent_label"].agg(lambda s: (s == "positive").mean())
        out["sent_neg_share"] = g["sent_label"].agg(lambda s: (s == "negative").mean())
    return out


def sentiment_ticker_day(article_ticker: pd.DataFrame, scores: pd.DataFrame, cfg: Config, *,
                         scale_floor: float, exclude_short: bool = True) -> pd.DataFrame:
    """article_ticker는 baseline_start~end 전체여야 z 워밍업이 된다. 반환은 start~end만."""
    lo, hi = pd.Timestamp(cfg.baseline_start), pd.Timestamp(cfg.end)
    at = article_ticker[article_ticker["attributed_day"].between(lo, hi)]
    if exclude_short:
        at = at[~at["is_short"]]
    use = at.merge(scores[["article_id", "sent_score", "sent_label"]], on="article_id", how="inner")
    missing = at["article_id"].nunique() - use["article_id"].nunique()
    if missing:
        raise ValueError(f"감성 점수가 없는 글 {missing}건 — score 단계를 먼저 실행하세요.")

    grid = pd.MultiIndex.from_product([cfg.tickers, trading_sessions(cfg.baseline_start, cfg.end)],
                                      names=["ticker", "attributed_day"])
    out = pd.DataFrame(index=grid).join(_daily(use)).join(_daily(use[use["is_primary"]], "_primary")).reset_index()
    for c in ["n_scored", "n_scored_primary"]:
        out[c] = out[c].fillna(0).astype(int)
    out["sent_net"] = out["sent_pos_share"] - out["sent_neg_share"]
    out = out.rename(columns={"attributed_day": "day"}).sort_values(["ticker", "day"]).reset_index(drop=True)
    for c in ["sent_mean", "sent_mean_primary"]:
        out[f"{c}_z"] = out.groupby("ticker")[c].transform(lambda s: _robust_z(s, cfg.window, cfg.min_periods, scale_floor))
    keep = out["day"].between(pd.Timestamp(cfg.start), hi)
    return out.loc[keep, ["ticker", "day", *SENTIMENT_COLUMNS]].reset_index(drop=True)


def join_with_mentions(ticker_day: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """전처리 ticker_day(언급량) + 감성 + 상호작용(언급 비율 z × 감성 z)."""
    out = ticker_day.merge(sentiment, on=["ticker", "day"], how="left", validate="one_to_one")
    if len(out) != len(ticker_day) or out["n_scored"].isna().any():
        raise ValueError("ticker_day와 감성 테이블의 (ticker, day)가 맞지 않습니다.")
    out["mention_x_sent_z"] = out["mention_rate_z"] * out["sent_mean_z"]
    return out
