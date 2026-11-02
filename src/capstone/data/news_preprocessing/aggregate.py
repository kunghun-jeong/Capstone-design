"""(ticker, 귀속 거래일) 집계 + "평소 대비 편차"(robust z).

모든 (ticker, 거래일) 조합을 만들고 기사 없는 날은 0으로 채운다 (결측이 아니라 0건이라는 정보).
베이스라인은 t 이전 window 거래일의 중앙값/MAD이며 t 당일은 포함하지 않는다 (shift 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trading_calendar import trading_sessions

COUNT_COLS = ["n_articles", "n_primary"]


def _robust_z(counts: pd.Series, window: int, min_periods: int, scale_floor: float) -> pd.Series:
    past = counts.shift(1)   # 당일 제외
    med = past.rolling(window, min_periods=min_periods).median()
    mad = past.rolling(window, min_periods=min_periods).apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
    scale = (1.4826 * mad).clip(lower=scale_floor)
    return (counts - med) / scale


def aggregate_ticker_day(
    article_ticker: pd.DataFrame, tickers: list[str], warmup_start, end, *, window: int, min_periods: int, scale_floor: float,
) -> pd.DataFrame:
    sessions = trading_sessions(warmup_start, end)
    grid = pd.MultiIndex.from_product([tickers, sessions], names=["ticker", "day"]).to_frame(index=False)

    at = article_ticker.rename(columns={"attributed_day": "day"})
    g = at.groupby(["ticker", "day"])
    agg = pd.DataFrame({
        "n_articles": g["article_id"].nunique(),
        "n_rows": g["n_rows"].sum(),
        "n_primary": g["is_primary"].sum(),
        "n_broad": g["is_broad"].sum(),
        "n_publishers": g["publisher"].nunique(),
        "article_ids": g["article_id"].agg(lambda s: sorted(set(s))),
    }).reset_index()
    out = grid.merge(agg, on=["ticker", "day"], how="left")
    for c in ["n_articles", "n_rows", "n_primary", "n_broad", "n_publishers"]:
        out[c] = out[c].fillna(0).astype(int)
    out["article_ids"] = out["article_ids"].map(lambda v: v if isinstance(v, list) else [])
    out = out.sort_values(["ticker", "day"]).reset_index(drop=True)

    for c in COUNT_COLS:
        out[f"{c}_z"] = out.groupby("ticker")[c].transform(lambda s: _robust_z(s, window, min_periods, scale_floor))
    return out
