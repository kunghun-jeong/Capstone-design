"""발행 시각 → 귀속 거래일.

규칙: 기사는 "컷오프가 발행 시각보다 뒤인 첫 NYSE 거래일"에 귀속된다.
 - cutoff="close": 각 거래일의 실제 장 마감(조기 폐장일은 13:00 ET). 장 마감 전 기사는 당일, 이후·주말·휴장일 기사는 다음 거래일.
 - cutoff="HH:MM": 모든 거래일에 America/New_York 고정 시각(예: 09:30이면 개장 전 기사만 당일 귀속).
귀속일 d의 피처는 d의 종가 시점까지의 정보로 만든 것이고, 라벨 close(d)→close(d+3)과 같은 정보 집합이다.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pandas_market_calendars as mcal

NY = "America/New_York"


def session_cutoffs(start: date, end: date, cutoff: str = "close") -> pd.Series:
    """[start, end] 거래일(index, naive 날짜) → 컷오프 시각(UTC tz-aware)."""
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    if cutoff == "close":
        cuts = sched["market_close"].dt.tz_convert("UTC")
    else:
        hh, mm = (int(x) for x in cutoff.split(":"))
        cuts = pd.Series(
            [pd.Timestamp(d.date()).tz_localize(NY) + pd.Timedelta(hours=hh, minutes=mm) for d in sched.index],
            index=sched.index,
        ).dt.tz_convert("UTC")
    cuts.index = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
    return cuts


def trading_sessions(start: date, end: date) -> pd.DatetimeIndex:
    return session_cutoffs(start, end, "close").index


def attribute_trading_day(published_utc: pd.Series, cutoff: str = "close") -> pd.Series:
    """발행 시각(UTC, tz-aware) → 귀속 거래일(naive Timestamp, 자정). 범위를 벗어나면 NaT."""
    ts = pd.to_datetime(published_utc, utc=True)
    if ts.isna().any():
        raise ValueError("published_utc에 결측/파싱 불가 값이 있습니다.")
    lo = (ts.min() - timedelta(days=7)).date()
    hi = (ts.max() + timedelta(days=10)).date()   # 연휴가 겹쳐도 다음 거래일이 잡히게 여유를 둔다
    cuts = session_cutoffs(lo, hi, cutoff)
    # 정렬된 컷오프 중 ts보다 엄격히 뒤인 첫 값의 위치
    cut_ns = cuts.dt.tz_convert(None).to_numpy().astype("datetime64[ns]")
    ts_ns = ts.dt.tz_convert(None).to_numpy().astype("datetime64[ns]")
    pos = cut_ns.searchsorted(ts_ns, side="right")
    days = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    ok = pos < len(cuts)
    days[ok] = cuts.index[pos[ok]].values
    return days
