from datetime import date

import numpy as np
import pandas as pd
import pytest

from capstone.data.news_preprocessing.aggregate import _robust_z
from capstone.data.reddit_preprocessing.config import Config
from capstone.features.sentiment.aggregate import join_with_mentions, sentiment_ticker_day
from capstone.features.sentiment.finbert import label_order, score_articles
from capstone.features.sentiment.pipeline import articles_to_score

# 2026-04-01(수) ~ 04-10(금): 03(성금요일) 휴장 → 거래일 1,2,6,7,8,9,10
DAYS = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10"])


def _cfg(**kw) -> Config:
    base = dict(
        tickers=["NVDA", "MU"], baseline_start=date(2026, 4, 1), start=date(2026, 4, 7), end=date(2026, 4, 10),
        cutoff="close", window=3, min_periods=2, scale_floor=1.0, ratio_scale_floor=1.0, broad_tag_threshold=5,
        min_chars=20, drop_removed_posts=False, include_name_only=False, author_salt="s",
    )
    base.update(kw)
    return Config(**base)


class FakeScorer:
    """텍스트 안의 'up'/'down'으로 확률을 정하는 가짜 모델. 호출된 텍스트를 기록한다."""
    name, revision = "fake/finbert", "r1"

    def __init__(self):
        self.seen: list[str] = []

    def score(self, texts):
        self.seen += texts
        rows = [[0.8, 0.1, 0.1] if "up" in t else [0.1, 0.7, 0.2] if "down" in t else [0.1, 0.1, 0.8] for t in texts]
        return np.array(rows, dtype=np.float32)


def _at(rows):
    """(article_id, ticker, day, score, label, is_primary, is_short) → article_ticker, scores"""
    at = pd.DataFrame([{"article_id": a, "ticker": t, "attributed_day": pd.Timestamp(d), "is_primary": p, "is_short": sh,
                        "text_for_sentiment": f"text {a}"} for a, t, d, _, _, p, sh in rows])
    sc = pd.DataFrame([{"article_id": a, "sent_score": s, "sent_label": lab} for a, _, _, s, lab, _, _ in rows]).drop_duplicates("article_id")
    return at, sc


def test_label_order_matches_by_name():
    # ProsusAI/finbert 순서와 다른 모델 순서 모두 LABELS(positive, negative, neutral)로 맞춘다
    assert label_order({0: "positive", 1: "negative", 2: "neutral"}) == [0, 1, 2]
    assert label_order({0: "Neutral", 1: "Positive", 2: "Negative"}) == [1, 2, 0]
    with pytest.raises(ValueError):
        label_order({0: "pos", 1: "neg", 2: "neu"})


def test_score_articles_scores_once_and_resumes(tmp_path):
    out = tmp_path / "scores.parquet"
    arts = pd.DataFrame({"article_id": ["a", "b", "b", "c"], "text": ["NVDA up", "MU down", "MU down", "NVDA flat"]})
    s1 = FakeScorer()
    res = score_articles(arts, s1, out, chunk_size=2, log=lambda m: None)
    assert sorted(s1.seen) == ["MU down", "NVDA flat", "NVDA up"]          # 같은 글은 한 번
    r = res.set_index("article_id")
    assert r.loc["a", "sent_label"] == "positive" and r.loc["b", "sent_label"] == "negative" and r.loc["c", "sent_label"] == "neutral"
    assert r.loc["a", "sent_score"] == pytest.approx(0.7)
    assert np.allclose(res[["p_pos", "p_neg", "p_neu"]].sum(axis=1), 1.0)

    s2 = FakeScorer()   # 이어서 실행: 이미 있는 글은 다시 계산하지 않는다
    score_articles(arts, s2, out, log=lambda m: None)
    assert s2.seen == []

    s3 = FakeScorer()   # 텍스트가 바뀐 글만 다시 계산
    changed = arts.assign(text=lambda d: d["text"].where(d["article_id"] != "c", "NVDA up now"))
    res = score_articles(changed, s3, out, log=lambda m: None)
    assert s3.seen == ["NVDA up now"]
    assert res.set_index("article_id").loc["c", "sent_label"] == "positive"
    assert len(pd.read_parquet(out)) == 3


def test_score_articles_refuses_mixed_revision(tmp_path):
    out = tmp_path / "scores.parquet"
    arts = pd.DataFrame({"article_id": ["a"], "text": ["NVDA up"]})
    score_articles(arts, FakeScorer(), out, log=lambda m: None)
    other = FakeScorer()
    other.revision = "r2"
    with pytest.raises(ValueError, match="리비전"):
        score_articles(arts, other, out, log=lambda m: None)


def test_articles_to_score_drops_short_and_duplicates():
    at, _ = _at([("a", "NVDA", "2026-04-01", 0, "neutral", True, False), ("a", "MU", "2026-04-01", 0, "neutral", False, False),
                 ("b", "MU", "2026-04-01", 0, "neutral", True, True)])
    assert articles_to_score(at, exclude_short=True)["article_id"].tolist() == ["a"]
    assert articles_to_score(at, exclude_short=False)["article_id"].tolist() == ["a", "b"]


def test_sentiment_ticker_day_values():
    at, sc = _at([
        ("a", "NVDA", "2026-04-07", 0.6, "positive", True, False),
        ("b", "NVDA", "2026-04-07", -0.4, "negative", False, False),   # 여러 종목 글
        ("b", "MU", "2026-04-07", -0.4, "negative", False, False),
        ("c", "NVDA", "2026-04-07", 0.9, "positive", True, True),      # 짧은 글 → 제외
        ("d", "NVDA", "2026-04-08", 0.0, "neutral", True, False),
    ])
    out = sentiment_ticker_day(at, sc, _cfg(), scale_floor=0.01).set_index(["ticker", "day"])
    assert len(out) == 2 * 4                                            # start~end 거래일 × 종목, 빈 날 포함
    r = out.loc[("NVDA", pd.Timestamp("2026-04-07"))]
    assert r.n_scored == 2 and r.sent_mean == pytest.approx(0.1)
    assert r.sent_pos_share == 0.5 and r.sent_neg_share == 0.5 and r.sent_net == 0
    assert r.n_scored_primary == 1 and r.sent_mean_primary == pytest.approx(0.6)
    mu = out.loc[("MU", pd.Timestamp("2026-04-07"))]
    assert mu.n_scored == 1 and mu.n_scored_primary == 0 and np.isnan(mu.sent_mean_primary)
    empty = out.loc[("MU", pd.Timestamp("2026-04-09"))]
    assert empty.n_scored == 0 and np.isnan(empty.sent_mean)            # 0으로 채우지 않는다


def test_sentiment_ticker_day_raises_when_scores_missing():
    at, sc = _at([("a", "NVDA", "2026-04-07", 0.6, "positive", True, False), ("b", "NVDA", "2026-04-07", 0.1, "neutral", True, False)])
    with pytest.raises(ValueError, match="점수가 없는 글"):
        sentiment_ticker_day(at, sc[sc.article_id == "a"], _cfg(), scale_floor=0.01)


def test_sentiment_z_uses_only_past_days():
    # NVDA 일별 평균: 0.1, 0.2, 0.3, 0.0, 0.5, -0.2, 0.4 (거래일 순서)
    means = [0.1, 0.2, 0.3, 0.0, 0.5, -0.2, 0.4]
    rows = [(f"n{i}", "NVDA", d, m, "neutral", True, False) for i, (d, m) in enumerate(zip(DAYS, means))]
    at, sc = _at(rows)
    out = sentiment_ticker_day(at, sc, _cfg(), scale_floor=0.01)
    nv = out[out.ticker == "NVDA"].set_index("day")["sent_mean_z"]
    expected = _robust_z(pd.Series(means, index=DAYS), 3, 2, 0.01).loc[nv.index]
    assert np.allclose(nv.values, expected.values)
    # 4/7 z = (0.0 − median(0.1, 0.2, 0.3)) / (1.4826 × MAD 0.1) — 4/7 당일 값은 기준에 안 들어간다
    assert nv.loc["2026-04-07"] == pytest.approx((0.0 - 0.2) / (1.4826 * 0.1))

    # 미래(4/10) 값을 바꿔도 그 이전 z는 그대로
    sc2 = sc.copy()
    sc2.loc[sc2.article_id == "n6", "sent_score"] = -0.9
    out2 = sentiment_ticker_day(at, sc2, _cfg(), scale_floor=0.01)
    nv2 = out2[out2.ticker == "NVDA"].set_index("day")["sent_mean_z"]
    assert np.allclose(nv.loc[:"2026-04-09"], nv2.loc[:"2026-04-09"])


def test_join_with_mentions_interaction_and_alignment():
    at, sc = _at([("a", "NVDA", d, 0.1 * i, "neutral", True, False) for i, d in enumerate(DAYS)])
    sent = sentiment_ticker_day(at, sc, _cfg(), scale_floor=0.01)
    td = sent[["ticker", "day"]].assign(mention_rate_z=2.0)
    out = join_with_mentions(td, sent)
    assert np.allclose(out["mention_x_sent_z"], 2.0 * out["sent_mean_z"], equal_nan=True)
    with pytest.raises(ValueError):
        join_with_mentions(pd.concat([td, td.assign(ticker="AMD")]), sent)
