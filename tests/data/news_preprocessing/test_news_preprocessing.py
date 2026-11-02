from datetime import date

import pandas as pd
import pytest

from capstone.data.news_preprocessing.aggregate import aggregate_ticker_day
from capstone.data.news_preprocessing.config import Config, load_config
from capstone.data.news_preprocessing.dedup import dedup_cross_source, explode_by_ticker, normalize_title, normalize_url
from capstone.data.news_preprocessing.normalize import clean_text, normalize
from capstone.data.news_preprocessing.pipeline import build_tables
from capstone.data.news_preprocessing.trading_calendar import attribute_trading_day, trading_sessions


def _attr(*stamps, cutoff="close"):
    s = pd.Series(pd.to_datetime(list(stamps), utc=True))
    return [None if pd.isna(d) else d.strftime("%Y-%m-%d") for d in attribute_trading_day(s, cutoff)]


# ---- 시점 정합 / 누수 ----

def test_before_close_attributed_same_day_after_close_next_day():
    # 2026-04-01(수) 15:59 ET = 19:59 UTC, 16:01 ET = 20:01 UTC
    assert _attr("2026-04-01T19:59:00Z", "2026-04-01T20:01:00Z") == ["2026-04-01", "2026-04-02"]


def test_weekend_and_holiday_roll_to_next_session():
    # 금 장후 / 토 / 일 → 월. 2026-04-03(성금요일 휴장) 낮 기사 → 04-06(월)
    assert _attr("2026-04-10T21:00:00Z", "2026-04-11T15:00:00Z", "2026-04-12T15:00:00Z") == ["2026-04-13"] * 3
    assert _attr("2026-04-03T15:00:00Z") == ["2026-04-06"]


def test_early_close_day_uses_actual_close():
    # 2026-11-27(금, 추수감사절 다음날) 13:00 ET 조기 폐장 = 18:00 UTC
    assert _attr("2026-11-27T17:59:00Z", "2026-11-27T18:01:00Z") == ["2026-11-27", "2026-11-30"]


def test_fixed_cutoff_0930():
    # 09:30 ET = 13:30 UTC. 개장 전 기사만 당일 귀속
    assert _attr("2026-04-01T13:29:00Z", "2026-04-01T13:31:00Z", cutoff="09:30") == ["2026-04-01", "2026-04-02"]


def test_attribution_never_precedes_publication_date_et():
    stamps = pd.date_range("2026-03-25", "2026-05-05", freq="97min", tz="UTC")
    days = attribute_trading_day(pd.Series(stamps))
    et_dates = pd.Series(stamps).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    assert (days.values >= et_dates.values).all()
    assert days.isin(trading_sessions(date(2026, 3, 20), date(2026, 5, 31))).all()


def test_session_counts_match_plan():
    assert len(trading_sessions(date(2026, 4, 1), date(2026, 10, 31))) == 148
    assert len(trading_sessions(date(2026, 1, 1), date(2026, 3, 31))) == 61


# ---- 정규화 ----

def test_clean_text_strips_html_and_whitespace():
    assert clean_text('<p>Hello&nbsp;<a class="x">AMD</a>\n  rises</p>') == "Hello AMD rises"
    assert clean_text("   ") is None and clean_text(None) is None


def _alpaca(i, created="2026-04-01T15:00:00Z", symbols=("NVDA",), headline="NVDA beats", content="<p>body</p>", url="https://www.benzinga.com/a/1"):
    return {"id": i, "headline": headline, "summary": "", "content": content, "source": "benzinga",
            "created_at": created, "updated_at": created, "symbols": list(symbols), "url": url, "author": "x"}


def _massive(i, published="2026-04-01T15:00:00Z", tickers=("NVDA",), title="NVDA beats", url="https://fool.com/x"):
    return {"id": str(i), "published_utc": published, "title": title, "author": "y", "article_url": url,
            "publisher": {"name": "The Motley Fool"}, "tickers": list(tickers), "description": "desc"}


def test_normalize_schema_nulls_and_tag_union():
    df = normalize([_alpaca(1, symbols=["NVDA", "AMD"]), _alpaca(1, symbols=["AMD", "MU"])], "alpaca")
    assert len(df) == 1                                  # 종목별 폴더 반복 → id 중복 제거
    row = df.iloc[0]
    assert row["tickers"] == ["AMD", "MU", "NVDA"] and row["n_tags"] == 3
    assert row["summary"] is None and row["body"] == "body"   # 빈 summary는 None
    assert str(df["published_utc"].dt.tz) == "UTC"
    m = normalize([_massive(1)], "massive").iloc[0]
    assert m["body"] is None and m["publisher"] == "The Motley Fool" and m["summary"] == "desc"


# ---- 중복 제거 / explode ----

def test_url_and_title_normalization():
    assert normalize_url("HTTPS://www.Example.com/a/?utm_source=x&id=2") == "example.com/a?id=2"
    assert normalize_url("") is None
    assert normalize_title("NVDA: Beats, Again!") == "nvda beats again"


def _combined(rows):
    df = pd.concat([normalize(r, s) for s, r in rows], ignore_index=True)
    df["attributed_day"] = attribute_trading_day(df["published_utc"])
    return df


def test_cross_source_dedup_by_title_and_day_keeps_spread_info():
    df = _combined([("alpaca", [_alpaca(1)]), ("massive", [_massive(9)]),
                    ("massive", [_massive(10, title="Different story", url="https://fool.com/y")])])
    out = dedup_cross_source(df)
    assert len(out) == 2
    merged = out[out["n_rows"] == 2].iloc[0]
    assert merged["source"] == "alpaca" and merged["sources"] == ["alpaca", "massive"] and merged["n_sources"] == 2
    assert merged["publishers"] == ["The Motley Fool", "benzinga"]


def test_same_title_on_different_days_is_not_merged():
    df = _combined([("alpaca", [_alpaca(1, created="2026-04-01T15:00:00Z")]),
                    ("massive", [_massive(2, published="2026-04-08T15:00:00Z")])])
    assert len(dedup_cross_source(df)) == 2


def test_explode_keeps_rows_and_flags():
    df = _combined([("alpaca", [_alpaca(1, symbols=["NVDA", "AMD", "XYZ"], headline="Nvidia and AMD rally"),
                                _alpaca(2, symbols=[f"T{i}" for i in range(12)] + ["MU"], headline="Market wrap", url="u2")])])
    art = dedup_cross_source(df)
    out = explode_by_ticker(art, ["NVDA", "AMD", "MU"], {"NVDA": ["Nvidia"]}, broad_threshold=10)
    assert sorted(zip(out["article_id"], out["ticker"])) == [("alpaca:1", "AMD"), ("alpaca:1", "NVDA"), ("alpaca:2", "MU")]
    flags = out.set_index(["article_id", "ticker"])
    assert flags.loc[("alpaca:1", "NVDA"), "is_primary"] and flags.loc[("alpaca:1", "AMD"), "is_primary"]
    assert not flags.loc[("alpaca:2", "MU"), "is_primary"] and flags.loc[("alpaca:2", "MU"), "is_broad"]


# ---- 집계 / 베이스라인 ----

def _at(ticker, day, n, primary=True):
    return [{"article_id": f"{ticker}{day}{k}", "ticker": ticker, "attributed_day": pd.Timestamp(day), "n_rows": 1,
             "is_primary": primary, "is_broad": False, "publisher": "p"} for k in range(n)]


def _agg(rows, **kw):
    df = pd.DataFrame(rows)
    return aggregate_ticker_day(df, ["NVDA", "MU"], date(2026, 4, 1), date(2026, 4, 9),
                                window=3, min_periods=3, scale_floor=1.0, **kw)


def test_zero_filled_grid_and_robust_z_excludes_today():
    rows = []
    for d, n in zip(["2026-04-01", "2026-04-02", "2026-04-06", "2026-04-07", "2026-04-08"], [1, 1, 1, 1, 10]):
        rows += _at("NVDA", d, n)
    out = _agg(rows)
    nvda = out[out["ticker"] == "NVDA"].set_index("day")
    assert len(out) == 2 * len(trading_sessions(date(2026, 4, 1), date(2026, 4, 9)))   # 04-03 휴장 제외, 전 조합 존재
    assert (out[out["ticker"] == "MU"]["n_articles"] == 0).all()                        # 기사 없는 날은 0
    assert nvda["n_articles_z"].iloc[:3].isna().all()                                   # 워밍업 부족 → NaN
    assert nvda.loc["2026-04-07", "n_articles_z"] == pytest.approx(0.0)
    assert nvda.loc["2026-04-08", "n_articles_z"] == pytest.approx(9.0)                 # (10-1)/max(0,1)


def test_baseline_ignores_current_and_future_days():
    base = []
    for d in ["2026-04-01", "2026-04-02", "2026-04-06", "2026-04-07"]:
        base += _at("NVDA", d, 2)
    a = _agg(base + _at("NVDA", "2026-04-08", 5))
    b = _agg(base + _at("NVDA", "2026-04-08", 5) + _at("NVDA", "2026-04-09", 50))     # 미래 급증
    day = pd.Timestamp("2026-04-08")
    za = a[(a.ticker == "NVDA") & (a.day == day)]["n_articles_z"].iloc[0]
    zb = b[(b.ticker == "NVDA") & (b.day == day)]["n_articles_z"].iloc[0]
    assert za == zb == pytest.approx(3.0)                                              # (5-2)/1


# ---- 설정 / 파이프라인 ----

def test_config_rejects_boolean_ticker(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tickers: [NVDA, ON]\nperiod: {warmup_start: 2026-01-01, start: 2026-04-01, end: 2026-10-31}\n"
                 "baseline: {window: 60, min_periods: 40, scale_floor: 1.0}\nbroad_tag_threshold: 10\n")
    with pytest.raises(ValueError, match="문자열"):
        load_config(p)


def test_repo_config_loads():
    cfg = load_config(__import__("pathlib").Path(__file__).resolve().parents[3] / "configs" / "news_preprocessing.yaml")
    assert "SNDK" not in cfg.tickers and cfg.start == date(2026, 4, 1)


def test_build_tables_end_to_end():
    cfg = Config(tickers=["NVDA", "AMD"], warmup_start=date(2026, 3, 2), start=date(2026, 4, 1), end=date(2026, 4, 10),
                 cutoff="close", window=3, min_periods=3, scale_floor=1.0, broad_tag_threshold=10, aliases={})
    alpaca = [_alpaca(1, created="2026-04-01T19:59:00Z"), _alpaca(2, created="2026-04-01T20:01:00Z", headline="AMD up", symbols=["AMD"], url="u2"),
              _alpaca(3, symbols=["XYZ"], url="u3"),                           # 유니버스 밖 → 제외
              _alpaca(4, created="2026-05-20T15:00:00Z", url="u4")]            # 기간 밖 → 제외
    t = build_tables(alpaca, [_massive(7, published="2026-04-01T19:59:30Z")], cfg)
    at = t["article_ticker"]
    assert sorted(at["article_id"]) == ["alpaca:1", "alpaca:2"] and at.loc[at.article_id == "alpaca:1", "n_rows"].iloc[0] == 2
    td = t["ticker_day"].set_index(["ticker", "day"])
    assert td.loc[("NVDA", pd.Timestamp("2026-04-01")), "n_articles"] == 1
    assert td.loc[("AMD", pd.Timestamp("2026-04-02")), "n_articles"] == 1
