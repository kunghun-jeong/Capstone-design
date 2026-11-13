import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import zstandard

from capstone.data.reddit_preprocessing.dump import dump_kind, iter_records, process_dump, year_ranges
from capstone.data.reddit_preprocessing.normalize import process_file
from capstone.data.reddit_preprocessing.pipeline import build_stage1, load_parts

from test_reddit_preprocessing import _cfg, _comment, _post, _write

UTC = timezone.utc


def _zst(path, recs):
    path.write_bytes(zstandard.ZstdCompressor().compress("".join(json.dumps(r) + "\n" for r in recs).encode()))
    return path


def test_dump_kind_maps_submissions_to_posts():
    assert dump_kind("wallstreetbets_submissions.zst") == ("wallstreetbets", "posts")
    assert dump_kind("AMD_Stock_comments.zst") == ("AMD_Stock", "comments")
    with pytest.raises(ValueError):
        dump_kind("wallstreetbets.zst")


def test_iter_records_filters_range_and_normalizes_old_fields(tmp_path):
    old = _comment(1, "NVDA up", created="2026-04-06T15:00:00Z")
    old["created_utc"] = str(old["created_utc"])          # 옛 덤프: 문자열 시각
    del old["name"]                                        # 옛 덤프: name 없음
    inside = _comment(2, "MU down", created="2026-04-07T15:00:00Z")
    outside = _comment(3, "NVDA", created="2026-04-09T15:00:00Z")
    path = _zst(tmp_path / "wallstreetbets_comments.zst", [old, inside, outside])
    got = list(iter_records(path, datetime(2026, 4, 6, tzinfo=UTC), datetime(2026, 4, 8, tzinfo=UTC), "comments"))
    assert [r["id"] for r in got] == ["c1", "c2"]
    assert got[0]["created_utc"] == int(old["created_utc"])
    assert isinstance(got[0]["created_utc"], int)
    assert got[0]["name"] == "t1_c1"


def test_process_dump_matches_jsonl_path_even_when_chunked(tmp_path):
    # 같은 레코드를 JSONL(기존 1단계)과 덤프(묶음 2건씩)로 처리하면 결과가 같아야 한다
    cfg = _cfg()
    recs = [_comment(i, b, created=t) for i, (b, t) in enumerate([
        ("NVDA to the moon today", "2026-04-07T15:00:00Z"),
        ("MU earnings look great", "2026-04-07T21:30:00Z"),     # 장 마감 후 → 다음 거래일
        ("nothing here at all really", "2026-04-08T14:00:00Z"),
        ("Micron is cheap", "2026-04-08T15:00:00Z"),            # 회사명만
        ("NVDA", "2026-04-09T15:00:00Z"),                       # 짧은 글
    ])]
    jsonl = _write(tmp_path / "raw", "wallstreetbets", "comments", "2026-04", recs)
    r1, a1 = process_file(jsonl, "wallstreetbets", "comments", cfg)
    zst = _zst(tmp_path / "wallstreetbets_comments.zst", recs)
    r2, a2, n = process_dump(zst, cfg, datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC), chunk=2)
    assert n == 5
    pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))
    a1 = a1.groupby(["subreddit", "kind", "day"], as_index=False).sum()
    pd.testing.assert_frame_equal(a1, a2, check_dtype=False)


def test_parts_merge_into_stage1_and_reject_duplicates(tmp_path):
    cfg = _cfg()
    zst = _zst(tmp_path / "wallstreetbets_submissions.zst", [_post(1, "NVDA DD"), _post(2, "MU earnings")])
    rows, act, _ = process_dump(zst, cfg, datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC))
    parts = tmp_path / "parts"
    parts.mkdir()
    rows.to_parquet(parts / "dump_x.rows.parquet")
    act.to_parquet(parts / "dump_x.activity.parquet")
    jsonl = _write(tmp_path / "raw", "wallstreetbets", "comments", "2026-04", [_comment(9, "AMD and NVDA go")])
    loaded, used = load_parts([parts])
    assert len(used) == 2
    all_rows, all_act = build_stage1([(jsonl, "wallstreetbets", "comments")], cfg, parts=loaded)
    assert set(all_rows["article_id"]) == {"reddit:t3_p1", "reddit:t3_p2", "reddit:t1_c9"}
    assert set(all_act["kind"]) == {"post", "comment"}
    with pytest.raises(ValueError, match="중복"):        # 기간이 겹친 조각을 두 번 넣으면 멈춘다
        build_stage1([], cfg, parts=loaded + loaded)


def test_year_ranges_split_on_utc_year():
    r = year_ranges(datetime(2020, 10, 1, tzinfo=UTC), datetime(2022, 3, 1, tzinfo=UTC))
    assert [(a.date().isoformat(), b.date().isoformat()) for a, b in r] == [
        ("2020-10-01", "2021-01-01"), ("2021-01-01", "2022-01-01"), ("2022-01-01", "2022-03-01")]


# ---- 표본 수집 가중치 ----

def test_sampled_records_are_weighted_in_activity_and_rows(tmp_path):
    from capstone.data.reddit_preprocessing.normalize import process_records
    cfg = _cfg()
    recs = [_comment(1, "NVDA calls are printing", _sample_weight=20.0),
            _comment(2, "nothing about stocks here", _sample_weight=20.0),
            _comment(3, "MU puts", author="AutoModerator", _sample_weight=10.0)]
    rows, act = process_records(recs, "wallstreetbets", "comments", cfg)
    a = act.iloc[0]
    assert a.n_items == 50 and a.n_items_sampled == 3          # 추정 원본 50건, 실제로 받은 글 3건
    assert a.n_bot == 10 and a.n_clean == 40
    assert set(rows["sample_weight"]) == {20.0, 10.0}


def test_weighted_ticker_day_counts_and_rate():
    from capstone.data.reddit_preprocessing.aggregate import aggregate_reddit
    from datetime import date
    cfg = _cfg(baseline_start=date(2026, 4, 6), start=date(2026, 4, 7), end=date(2026, 4, 7), window=1, min_periods=1)
    day = pd.Timestamp("2026-04-07")
    at = pd.DataFrame([
        {"article_id": "a", "ticker": "NVDA", "attributed_day": day, "n_rows": 20.0, "is_primary": True, "is_broad": False,
         "publisher": "wallstreetbets", "sample_weight": 20.0},
        {"article_id": "b", "ticker": "NVDA", "attributed_day": day, "n_rows": 1.0, "is_primary": False, "is_broad": False,
         "publisher": "stocks", "sample_weight": 1.0},
    ])
    act = pd.DataFrame([{"subreddit": "wallstreetbets", "kind": "comment", "day": day, "n_clean": 400.0, "n_clean_excl_removed": 400.0},
                        {"subreddit": "stocks", "kind": "comment", "day": day, "n_clean": 100.0, "n_clean_excl_removed": 100.0}])
    out = aggregate_reddit(at, act, cfg).set_index("ticker").loc["NVDA"]
    assert out.n_articles == 21 and out.n_articles_sampled == 2 and out.n_primary == 20
    assert out.mention_rate == pytest.approx(21 / 500 * 10_000)


def test_weighted_sentiment_mean():
    from capstone.features.sentiment.aggregate import _daily
    day = pd.Timestamp("2026-04-07")
    use = pd.DataFrame([{"ticker": "NVDA", "attributed_day": day, "sent_score": 0.5, "sent_label": "positive", "sample_weight": 3.0},
                        {"ticker": "NVDA", "attributed_day": day, "sent_score": -0.5, "sent_label": "negative", "sample_weight": 1.0}])
    r = _daily(use).iloc[0]
    assert r.n_scored == 2 and r.sent_mean == pytest.approx(0.25) and r.sent_pos_share == pytest.approx(0.75)


def test_hour_estimate():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("sample_reddit", pathlib.Path(__file__).resolve().parents[3] / "scripts" / "sample_reddit.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    h = 1_000_000_000 // 3600 * 3600
    assert mod.hour_estimate(100, h + 600, h + 1000, h + 600, h + 3600) == pytest.approx(100 / 400 * 3600)   # 꽉 찬 페이지: 밀도 × 3600
    assert mod.hour_estimate(30, h + 1900, h + 3500, h + 1800, h + 3600) == pytest.approx(30 * 3600 / 1800)  # 덜 찬 페이지
    assert mod.hour_estimate(0, 0, 0, h, h + 3600) == 0
