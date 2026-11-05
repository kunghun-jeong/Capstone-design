from datetime import date

import numpy as np
import pandas as pd
import pytest

from capstone.data.news_preprocessing.noise import flag_noise, sentiment_text
from capstone.data.news_preprocessing.sentiment import aggregate_sentiment, score_articles


def _arts(rows):
    return pd.DataFrame(rows, columns=["article_id", "title", "summary", "n_tags"])


def test_flag_noise_drops_templates_and_broad_but_keeps_real_news():
    df = flag_noise(_arts([
        ("a1", "Information Technology Stocks Whale Activity In Today's Session", None, 3),
        ("a2", "Will The S&P 500 Open Up Or Down Today?", None, 5),
        ("a3", "Nvidia Rallies", "In today's rapidly changing and fiercely competitive landscape, investors", 2),
        ("a4", "Market wrap", "stocks moved", 25),
        ("a5", "Nvidia beats estimates, raises guidance", "Shares jumped after hours.", 2),
        ("a6", "Analyst raises Micron price target to $200", None, 1),
        ("a7", "3 Stocks To Buy Right Now", None, 3),
    ]), broad_tag_threshold=10).set_index("article_id")
    reasons = df["drop_reason"]
    assert reasons[reasons.notna()].to_dict() == {
        "a1": "template_whale", "a2": "template_open_up_down", "a3": "template_boilerplate", "a4": "broad",
    }
    assert reasons[["a5", "a6", "a7"]].isna().all()
    assert df.loc["a6", "is_analyst_action"] and df.loc["a7", "is_list_promo"] and not df.loc["a5", "is_list_promo"]


def test_flag_noise_does_not_drop_rows():
    df = _arts([("a1", "Whale activity today", None, 1), ("a2", "Fine", "ok", 1)])
    assert len(flag_noise(df, 10)) == 2


def test_sentiment_text_title_plus_summary_or_title_only():
    df = _arts([("a", "Title", "Summary text.", 1), ("b", "Only title", None, 1), ("c", "Empty summary", "  ", 1)])
    assert sentiment_text(df).tolist() == ["Title. Summary text.", "Only title", "Empty summary"]


class _Dummy:
    """모델 없이 쓰는 점수기: 텍스트에 'good'이 있으면 긍정, 'bad'가 있으면 부정."""
    def __init__(self): self.calls = 0
    def score(self, texts):
        self.calls += len(texts)
        return np.array([[.8, .1, .1] if "good" in t else [.1, .8, .1] if "bad" in t else [.1, .1, .8] for t in texts], dtype="float32")


def test_score_articles_uses_cache_and_resumes(tmp_path):
    cache, scorer = tmp_path / "s.parquet", _Dummy()
    out = score_articles(["1", "2"], ["good news", "bad news"], scorer, cache, log=lambda *_: None)
    assert scorer.calls == 2 and out.set_index("article_id").loc["1", "sent_score"] == pytest.approx(0.7)
    out = score_articles(["1", "2", "3"], ["good news", "bad news", "flat"], scorer, cache, log=lambda *_: None)
    assert scorer.calls == 3                       # 새 기사 1건만 계산
    assert sorted(out["article_id"]) == ["1", "2", "3"]


def _scored(day, ticker, score, aid, primary=True, drop=None):
    return {"article_id": aid, "ticker": ticker, "attributed_day": pd.Timestamp(day), "is_primary": primary,
            "drop_reason": drop, "sent_score": score, "p_pos": max(score, 0), "p_neg": max(-score, 0)}


def _agg(rows, **kw):
    return aggregate_sentiment(pd.DataFrame(rows), ["NVDA", "MU"], date(2026, 4, 1), date(2026, 4, 9),
                               window=3, min_periods=3, **kw)


def test_aggregate_excludes_dropped_and_leaves_empty_days_nan():
    rows = [_scored("2026-04-01", "NVDA", 0.5, "a"), _scored("2026-04-01", "NVDA", -0.1, "b"),
            _scored("2026-04-01", "NVDA", -0.9, "c", drop="broad")]
    out = _agg(rows).set_index(["ticker", "day"])
    day1 = out.loc[("NVDA", pd.Timestamp("2026-04-01"))]
    assert day1["n_scored"] == 2 and day1["sent_mean"] == pytest.approx(0.2)
    assert not out.loc[("NVDA", pd.Timestamp("2026-04-02")), "has_sent"]
    assert np.isnan(out.loc[("NVDA", pd.Timestamp("2026-04-02")), "sent_mean"])    # 0으로 채우지 않는다
    assert out.loc[("MU", pd.Timestamp("2026-04-01")), "n_scored"] == 0


def test_sent_dev_baseline_excludes_today_and_future():
    days = ["2026-04-01", "2026-04-02", "2026-04-06", "2026-04-07"]
    base = [_scored(d, "NVDA", 0.2, f"x{i}") for i, d in enumerate(days)]
    a = _agg(base + [_scored("2026-04-08", "NVDA", 0.8, "t")])
    b = _agg(base + [_scored("2026-04-08", "NVDA", 0.8, "t"), _scored("2026-04-09", "NVDA", -1.0, "f")])   # 미래 급변
    pick = lambda df: df[(df.ticker == "NVDA") & (df.day == pd.Timestamp("2026-04-08"))].iloc[0]
    assert pick(a)["sent_dev"] == pytest.approx(0.6)       # 0.8 - 과거 평균 0.2
    assert pick(a)["sent_dev"] == pick(b)["sent_dev"]
    assert pick(a)["sent_z"] == pytest.approx(0.6 / 0.05)  # 과거 표준편차 0 → 하한 0.05
