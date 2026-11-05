import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from capstone.data.news_preprocessing.aggregate import _robust_z
from capstone.data.news_preprocessing.trading_calendar import trading_sessions
from capstone.data.reddit_preprocessing.aggregate import aggregate_reddit
from capstone.data.reddit_preprocessing.clean import explode_by_ticker, merge_copies, remove_noise
from capstone.data.reddit_preprocessing.config import Config, load_config
from capstone.data.reddit_preprocessing.normalize import TickerMatcher, author_hash, clean_text, list_raw_files, process_file
from capstone.data.reddit_preprocessing.pipeline import build_stage1, build_stage2

REPO = Path(__file__).resolve().parents[3]


def _cfg(**kw) -> Config:
    base = dict(
        tickers=["NVDA", "MU", "GOOGL", "TSM", "META"], baseline_start=date(2026, 4, 1), start=date(2026, 4, 6),
        end=date(2026, 4, 10), cutoff="close", window=2, min_periods=2, scale_floor=1.0, ratio_scale_floor=1.0,
        broad_tag_threshold=3, min_chars=20, drop_removed_posts=False, include_name_only=False, author_salt="s",
        ticker_patterns={"GOOGL": "GOOGL?", "TSM": "TSMC?"},
        aliases={"NVDA": ["Nvidia"], "MU": ["Micron"], "GOOGL": ["Google"], "META": ["Meta"]},
        other_tickers=["SNDK", "INTC"], bots=["AutoModerator", "VisualMod"],
        paths={"raw": ["raw"], "interim": "interim", "processed": "processed"},
    )
    base.update(kw)
    return Config(**base)


def _ts(s: str) -> int:
    return int(pd.Timestamp(s).timestamp())


def _comment(i, body, created="2026-04-07T15:00:00Z", **kw):
    rec = {"id": f"c{i}", "name": f"t1_c{i}", "created_utc": _ts(created), "retrieved_on": _ts(created) + 30,
           "_meta": {"retrieved_2nd_on": _ts(created) + 130000}, "subreddit": "wallstreetbets", "author": f"user{i}",
           "body": body, "link_id": "t3_x", "parent_id": "t3_x", "link_title": "Daily thread NVDA",
           "permalink": f"/r/wallstreetbets/comments/x/y/c{i}/", "score": 42, "stickied": False, "distinguished": None,
           "edited": False}
    rec.update(kw)
    return rec


def _post(i, title, selftext="", created="2026-04-07T15:00:00Z", **kw):
    rec = {"id": f"p{i}", "name": f"t3_p{i}", "created_utc": _ts(created), "retrieved_on": _ts(created) + 30,
           "_meta": {}, "subreddit": "wallstreetbets", "author": f"op{i}", "title": title, "selftext": selftext,
           "permalink": f"/r/wallstreetbets/comments/p{i}/z/", "score": 7, "num_comments": 3, "stickied": False,
           "distinguished": None, "edited": False, "removed_by_category": None}
    rec.update(kw)
    return rec


def _write(root: Path, sub: str, kind: str, month: str, recs: list[dict]) -> Path:
    path = root / sub / kind / f"{month}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


# ---- 종목 언급 판정 ----

@pytest.fixture
def matcher():
    c = _cfg()
    return TickerMatcher(c.tickers, c.ticker_patterns, c.aliases, c.other_tickers)


def test_ticker_is_case_sensitive_with_word_boundaries(matcher):
    assert matcher.match("Bought NVDA and MU calls")[0] == ["MU", "NVDA"]
    assert matcher.match("you must hold, mu is fine, nvda too")[0] == []      # 소문자 단어는 티커가 아니다
    assert matcher.match("MUST SEE NVDAX AMUSE")[0] == []                      # 단어 경계
    assert matcher.match("NVDA's earnings, (MU)")[0] == ["MU", "NVDA"]


def test_dollar_prefix_allows_lowercase(matcher):
    assert matcher.match("loaded $nvda and $Mu")[0] == ["MU", "NVDA"]


def test_goog_and_tsmc_are_merged(matcher):
    assert matcher.match("GOOG vs GOOGL")[0] == ["GOOGL"]
    assert matcher.match("TSMC beat")[0] == ["TSM"]


def test_name_only_and_other_tickers(matcher):
    tickers, names, other = matcher.match("Nvidia and Google are great, also SNDK and INTC. MU too")
    assert tickers == ["MU"] and names == ["GOOGL", "NVDA"] and other == ["INTC", "SNDK"]
    # 티커가 있으면 회사명만 언급으로 치지 않는다
    assert matcher.match("NVDA aka Nvidia")[1] == []


def test_name_match_survives_unicode_case_folding(matcher):
    assert matcher.match("MİCRON İS GREAT")[1] == ["MU"]


def test_clean_text_and_author_hash():
    assert clean_text("a &amp; b\n\n <3") == "a & b <3"            # 마크다운의 '<'는 지우지 않는다
    assert clean_text("[removed]") is None and clean_text("  ") is None
    h = author_hash("someone", "s")
    assert len(h) == 16 and h != author_hash("someone", "t") and author_hash("[deleted]", "s") is None


# ---- 1단계: 파일 처리 ----

def test_process_file_rows_flags_and_activity(tmp_path):
    cfg = _cfg()
    recs = [
        _comment(1, "NVDA to the moon"),
        _comment(2, "no ticker here at all, just vibes"),
        _comment(3, "[removed]"),
        _comment(4, "Nvidia GPUs are nice for games"),
        _comment(5, "MU bet created", author="VisualMod"),
        _comment(6, "https://example.com/MU"),
        _comment(7, "MU", edited=1775000000),
    ]
    path = _write(tmp_path, "wallstreetbets", "comments", "2026-04", recs)
    rows, act = process_file(path, "wallstreetbets", "comments", cfg)

    assert set(rows["article_id"]) == {"reddit:t1_c1", "reddit:t1_c4", "reddit:t1_c5", "reddit:t1_c6", "reddit:t1_c7"}
    r = rows.set_index("article_id")
    assert r.at["reddit:t1_c1", "tickers"] == ["NVDA"] and r.at["reddit:t1_c4", "name_only_tickers"] == ["NVDA"]
    assert r.at["reddit:t1_c5", "is_bot"] and r.at["reddit:t1_c6", "is_link_only"]
    assert r.at["reddit:t1_c7", "is_short"] and r.at["reddit:t1_c7", "is_edited"]
    assert r.at["reddit:t1_c1", "kind"] == "comment" and r.at["reddit:t1_c1", "publisher"] == "wallstreetbets"
    assert r.at["reddit:t1_c1", "url"].startswith("https://www.reddit.com/r/wallstreetbets/")
    # 원본 작성자 아이디와 score 원 이름은 산출물에 없다
    assert "author" not in rows.columns and "score" not in rows.columns
    assert r.at["reddit:t1_c1", "author_hash"] == author_hash("user1", "s")
    assert str(rows["published_utc"].dtype) == "datetime64[ns, UTC]"

    assert act["n_items"].sum() == 7
    assert act["n_deleted_text"].sum() == 1 and act["n_bot"].sum() == 1 and act["n_link_only"].sum() == 1
    assert act["n_clean"].sum() == 4


def test_post_text_and_removed_flag(tmp_path):
    cfg = _cfg()
    recs = [_post(1, "META earnings", "[removed]", removed_by_category="moderator"), _post(2, "Thoughts?", "Long TSMC here")]
    rows, act = process_file(_write(tmp_path, "stocks", "posts", "2026-04", recs), "stocks", "posts", cfg)
    r = rows.set_index("article_id")
    assert r.at["reddit:t3_p1", "text_for_sentiment"] == "META earnings" and r.at["reddit:t3_p1", "is_removed"]
    assert r.at["reddit:t3_p2", "text_for_sentiment"] == "Thoughts?\nLong TSMC here"
    assert r.at["reddit:t3_p2", "num_comments_t36h"] == 3
    assert act["n_removed"].sum() == 1 and act["n_clean_excl_removed"].sum() == 1


def test_list_raw_files_skips_logs_and_state(tmp_path):
    _write(tmp_path, "stocks", "posts", "2026-04", [_post(1, "MU")])
    (tmp_path / "stocks" / "posts" / "2026-04.state.json").write_text("{}")
    _write(tmp_path, "_logs", "posts", "2026-04", [_post(2, "MU")])
    assert [(s, k) for _, s, k in list_raw_files([tmp_path])] == [("stocks", "posts")]


# ---- 2단계: 노이즈·복붙·explode ----

def _rows(**overrides):
    base = dict(
        article_id="reddit:t1_a", subreddit="wallstreetbets", kind="comment", attributed_day=pd.Timestamp("2026-04-07"),
        published_utc=pd.Timestamp("2026-04-07T15:00:00Z"), title=None, summary=None, publisher="wallstreetbets",
        tickers=["NVDA"], n_tags=1, name_only_tickers=[], n_other_tickers=0, text_for_sentiment="NVDA up big today",
        link_title=None, author_hash="h", is_deleted_text=False, is_bot=False, is_mod=False, is_stickied=False,
        is_link_only=False, is_removed=False, is_short=False, is_edited=False,
    )
    base.update(overrides)
    return base


def test_remove_noise_counts_first_reason_once():
    df = pd.DataFrame([_rows(article_id="a", is_bot=True, is_mod=True), _rows(article_id="b", is_stickied=True),
                       _rows(article_id="c", kind="post", is_removed=True), _rows(article_id="d", is_short=True)])
    out, removed = remove_noise(df, _cfg())
    assert removed == {"deleted_text": 0, "bot": 1, "mod": 0, "stickied": 1, "link_only": 0}
    assert list(out["article_id"]) == ["c", "d"]          # 짧은 글과 제거 표시 게시글은 기본 유지
    out2, removed2 = remove_noise(df, _cfg(drop_removed_posts=True))
    assert removed2["removed_post"] == 1 and list(out2["article_id"]) == ["d"]


def test_merge_copies_same_day_same_sub_only():
    t0 = pd.Timestamp("2026-04-07T15:00:00Z")
    df = pd.DataFrame([
        _rows(article_id="a", text_for_sentiment="MU  to 1000", published_utc=t0 + pd.Timedelta(minutes=5)),
        _rows(article_id="b", text_for_sentiment="mu to 1000", published_utc=t0, tickers=["MU"]),
        _rows(article_id="c", text_for_sentiment="MU to 1000", attributed_day=pd.Timestamp("2026-04-08")),
        _rows(article_id="d", text_for_sentiment="MU to 1000", subreddit="stocks"),
    ])
    out = merge_copies(df).set_index("article_id")
    assert set(out.index) == {"b", "c", "d"}                  # 대표 = 가장 먼저 올라온 글
    assert out.at["b", "n_rows"] == 2 and sorted(out.at["b", "source_ids"]) == ["a", "b"]
    assert out.at["c", "n_rows"] == 1


def test_explode_primary_broad_and_name_only():
    cfg = _cfg()
    m = TickerMatcher(cfg.tickers, cfg.ticker_patterns, cfg.aliases, cfg.other_tickers)
    df = merge_copies(pd.DataFrame([
        _rows(article_id="a"),
        _rows(article_id="b", tickers=["MU", "NVDA"], n_tags=2, n_other_tickers=1, text_for_sentiment="MU NVDA SNDK"),
        _rows(article_id="c", kind="post", title="Weekly thoughts", tickers=["META"], text_for_sentiment="Weekly thoughts\nMETA"),
        _rows(article_id="d", kind="post", title="META run", tickers=["META"], text_for_sentiment="META run"),
        _rows(article_id="e", tickers=["MU"], name_only_tickers=["NVDA"], text_for_sentiment="MU and Nvidia"),
    ]))
    at = explode_by_ticker(df, cfg, lambda title: m.match(title)[0])
    key = at.set_index(["article_id", "ticker"])
    assert len(at) == 6                                        # a1 + b2 + c1 + d1 + e1(회사명 NVDA 제외)
    assert key.at[("a", "NVDA"), "is_primary"] and not key.at[("b", "MU"), "is_primary"]
    assert key.at[("b", "MU"), "is_broad"] and not key.at[("a", "NVDA"), "is_broad"]
    assert not key.at[("c", "META"), "is_primary"] and key.at[("d", "META"), "is_primary"]   # 게시글은 제목 기준
    at2 = explode_by_ticker(df, _cfg(include_name_only=True), lambda title: m.match(title)[0])
    assert at2.set_index(["article_id", "ticker"]).at[("e", "NVDA"), "is_name_only"]


# ---- 집계·베이스라인 ----

def _at(day: str, n: int, ticker="NVDA", sub="wallstreetbets", tag="") -> list[dict]:
    return [{"article_id": f"{ticker}{day}{tag}{i}", "ticker": ticker, "attributed_day": pd.Timestamp(day), "n_rows": 1,
             "is_primary": True, "is_broad": False, "publisher": sub, "subreddit": sub} for i in range(n)]


def test_aggregate_zero_fill_rate_and_trim():
    cfg = _cfg(tickers=["NVDA", "MU"])
    at = pd.DataFrame(_at("2026-04-01", 2) + _at("2026-04-02", 4) + _at("2026-04-07", 10))
    days = trading_sessions(date(2026, 4, 1), date(2026, 4, 10))
    act = pd.DataFrame({"subreddit": "wallstreetbets", "kind": "comment", "day": days, "n_clean": 1000, "n_clean_excl_removed": 900})
    td = aggregate_reddit(at, act, cfg)
    assert td["day"].min() == pd.Timestamp("2026-04-06") and td["day"].max() == pd.Timestamp("2026-04-10")
    assert set(td["ticker"]) == {"NVDA", "MU"} and len(td) == 2 * 5
    nv = td[td["ticker"] == "NVDA"].set_index("day")
    assert nv.at[pd.Timestamp("2026-04-07"), "mention_rate"] == pytest.approx(100.0)    # 10 / 1000 × 10,000
    assert nv.at[pd.Timestamp("2026-04-08"), "n_articles"] == 0
    assert td[td["ticker"] == "MU"]["n_articles"].eq(0).all()
    # 제거 표시 게시글을 빼면 분모도 바뀐다
    td2 = aggregate_reddit(at, act, replace(cfg, drop_removed_posts=True))
    assert td2.set_index(["ticker", "day"]).at[("NVDA", pd.Timestamp("2026-04-07")), "n_total_items"] == 900


def test_robust_z_baseline_excludes_today_for_counts_and_rates():
    s = pd.Series([1.0, 2.0, 3.0, 100.0])
    z = _robust_z(s, window=3, min_periods=3, scale_floor=0.0)
    # 당일(100)이 창에 들어갔다면 중앙값이 3이 된다. 과거 [1, 2, 3]만 쓰면 중앙값 2, MAD 1.
    assert z.iloc[3] == pytest.approx((100 - 2) / 1.4826)

    cfg = _cfg(tickers=["NVDA"], window=2, min_periods=2)
    days = trading_sessions(date(2026, 4, 1), date(2026, 4, 10))
    act = pd.DataFrame({"subreddit": "s", "kind": "comment", "day": days, "n_clean": 100, "n_clean_excl_removed": 100})
    base = pd.DataFrame(_at("2026-04-01", 1) + _at("2026-04-02", 3) + _at("2026-04-06", 5))
    spike = pd.concat([base, pd.DataFrame(_at("2026-04-06", 50, tag="x"))], ignore_index=True)
    a = aggregate_reddit(base, act, cfg).set_index("day")
    b = aggregate_reddit(spike, act, cfg).set_index("day")
    d = pd.Timestamp("2026-04-06")   # 과거 2거래일 = 4/1(1건), 4/2(3건). 4/3은 성금요일 휴장
    # 당일 값이 5건이든 55건이든 베이스라인은 과거 [1, 3] → 중앙값 2, MAD 1 그대로
    assert a.at[d, "n_articles_z"] == pytest.approx((5 - 2) / 1.4826)
    assert b.at[d, "n_articles_z"] == pytest.approx((55 - 2) / 1.4826)
    # 비율(만 건당): 과거 [100, 300] → 중앙값 200, MAD 100
    assert a.at[d, "mention_rate_z"] == pytest.approx((500 - 200) / 148.26)
    assert b.at[d, "mention_rate_z"] == pytest.approx((5500 - 200) / 148.26)
    later = pd.Timestamp("2026-04-07")
    assert a.at[later, "n_articles_z"] != b.at[later, "n_articles_z"]   # 다음 날부터는 과거로 반영


# ---- 누수 테스트 (전 구간 end-to-end) ----

@pytest.fixture
def built(tmp_path):
    cfg = _cfg(tickers=["NVDA", "MU"], baseline_start=date(2026, 5, 18), start=date(2026, 5, 20), end=date(2026, 5, 29))
    comments = [
        _comment(1, "NVDA ripping before close", created="2026-05-21T19:59:00Z"),     # 15:59 ET → 5/21
        _comment(2, "NVDA after hours dump", created="2026-05-21T20:01:00Z"),         # 16:01 ET → 5/22
        _comment(3, "MU on memorial day", created="2026-05-25T15:00:00Z"),            # 휴장 → 5/26
        _comment(4, "MU on saturday", created="2026-05-23T15:00:00Z"),                # 토 → 5/26
        _comment(5, "NVDA again", created="2026-05-19T14:00:00Z"),
        _comment(6, "NVDA to 300", created="2026-05-27T14:00:00Z"),
        _comment(7, "NVDA to 300", created="2026-05-27T14:05:00Z"),                   # 복붙 → 합쳐짐
    ]
    posts = [_post(1, "MU earnings play", created="2026-05-26T13:00:00Z", removed_by_category="automod_filtered")]
    files = [(_write(tmp_path, "wallstreetbets", "comments", "2026-05", comments), "wallstreetbets", "comments"),
             (_write(tmp_path, "wallstreetbets", "posts", "2026-05", posts), "wallstreetbets", "posts")]
    rows, act = build_stage1(files, cfg)
    tables, stats = build_stage2(rows, act, cfg)
    return cfg, rows, tables, stats


def test_attribution_after_close_weekend_holiday(built):
    _, rows, _, _ = built
    day = rows.set_index("article_id")["attributed_day"].dt.strftime("%Y-%m-%d")
    assert day["reddit:t1_c1"] == "2026-05-21" and day["reddit:t1_c2"] == "2026-05-22"
    assert day["reddit:t1_c3"] == "2026-05-26" and day["reddit:t1_c4"] == "2026-05-26"


def test_attributed_day_never_before_publication_date_et(built):
    _, rows, tables, _ = built
    for df in (rows, tables["article_ticker"]):
        et = df["published_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        assert (df["attributed_day"] >= et).all()


def test_no_rows_on_holidays_or_outside_period(built):
    cfg, _, tables, _ = built
    sessions = trading_sessions(cfg.start, cfg.end)
    assert tables["article_ticker"]["attributed_day"].isin(sessions).all()
    assert tables["ticker_day"]["day"].isin(sessions).all()
    assert pd.Timestamp("2026-05-25") not in set(tables["ticker_day"]["day"])
    assert tables["article_ticker"]["attributed_day"].min() >= pd.Timestamp(cfg.start)   # 5/19 글은 베이스라인에만


def test_processed_has_no_late_info_columns(built):
    _, _, tables, _ = built
    banned = {"score", "score_t36h", "num_comments", "num_comments_t36h", "was_deleted_later", "has_2nd_retrieval", "author"}
    for df in tables.values():
        assert not banned & set(df.columns)


def test_end_to_end_counts_and_schema(built):
    _, _, tables, stats = built
    assert stats["merged_copies"] == 1
    td = tables["ticker_day"].set_index(["ticker", "day"])
    assert td.at[("NVDA", pd.Timestamp("2026-05-27")), "n_articles"] == 1
    assert td.at[("NVDA", pd.Timestamp("2026-05-27")), "n_rows"] == 2
    assert td.at[("MU", pd.Timestamp("2026-05-26")), "n_articles"] == 3        # 휴장일·주말 글 + 제거 표시 게시글(기본 포함)
    news_td = ["ticker", "day", "n_articles", "n_rows", "n_primary", "n_broad", "n_publishers", "article_ids",
               "n_articles_z", "n_primary_z"]
    news_at = ["article_id", "ticker", "attributed_day", "published_utc", "n_tags", "n_rows", "n_sources",
               "publisher", "is_primary", "is_broad", "title", "summary"]
    assert set(news_td) <= set(tables["ticker_day"].columns)
    assert set(news_at) <= set(tables["article_ticker"].columns)
    assert "subreddit" in tables["ticker_day_by_subreddit"].columns


def test_overlapping_raw_dirs_are_rejected(tmp_path):
    cfg = _cfg()
    recs = [_comment(1, "NVDA")]
    a = _write(tmp_path / "a", "wallstreetbets", "comments", "2026-04", recs)
    b = _write(tmp_path / "b", "wallstreetbets", "comments", "2026-04", recs)
    with pytest.raises(ValueError, match="중복"):
        build_stage1([(a, "wallstreetbets", "comments"), (b, "wallstreetbets", "comments")], cfg)


# ---- 설정 ----

def test_repo_config_loads():
    cfg = load_config(REPO / "configs" / "reddit_preprocessing.yaml")
    assert len(cfg.tickers) == 10 and "GOOG" not in cfg.tickers
    assert cfg.start == date(2026, 5, 1) and cfg.end == date(2026, 9, 30)


def test_config_rejects_boolean_ticker(tmp_path):
    text = (REPO / "configs" / "reddit_preprocessing.yaml").read_text(encoding="utf-8").replace("  - MRVL", "  - ON")
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="문자열"):
        load_config(path)
