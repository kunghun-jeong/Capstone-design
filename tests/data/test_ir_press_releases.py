import pandas as pd
import pytest

from capstone.data.ir_press_releases import (
    CollectStats,
    FeedSource,
    build_page_url,
    clean_html,
    collect,
    fetch_all_pages,
    fetch_feed_bytes,
    load_feeds,
    parse_entries,
    summarize,
)

# "Q4형" 플랫폼(AMD 등)의 실제 응답을 2건으로 줄인 것 — summary가 비어있고 2자리 연도.
Q4_STYLE_FEED = b"""<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <title>Example Corp (EX) Press Releases</title>
    <item>
      <title>Example Corp to Report Fiscal Third Quarter 2026 Financial Results</title>
      <link>https://ir.example.com/news-events/press-releases/detail/1/example-q3-2026</link>
      <pubDate>Tue, 06 Oct 26 16:15:00 -0400</pubDate>
      <guid>https://ir.example.com/news-events/press-releases/detail/1/example-q3-2026</guid>
    </item>
    <item>
      <title>Example Corp Declares Quarterly Dividend Payment</title>
      <link>https://ir.example.com/news-events/press-releases/detail/2/example-dividend</link>
      <pubDate>Fri, 25 Sep 26 16:05:00 -0400</pubDate>
      <guid>https://ir.example.com/news-events/press-releases/detail/2/example-dividend</guid>
    </item>
  </channel>
</rss>
"""

# "그 외" 플랫폼(AVGO/AMAT 등) 스타일 — summary·author 있고 4자리 연도.
OTHER_STYLE_FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>Example Corp News Releases</title>
    <item>
      <title>Example Corp and Partner Collaborate on &lt;b&gt;New&lt;/b&gt; Chips</title>
      <link>https://investors.example.com/news-releases/example-partner-chips</link>
      <description>&lt;p&gt;Collaboration&amp;nbsp;aims to develop breakthroughs.&lt;/p&gt;</description>
      <pubDate>Tue, 06 Oct 2026 09:00:00 -0400</pubDate>
      <dc:creator>Example Corp</dc:creator>
      <guid>12345</guid>
    </item>
  </channel>
</rss>
"""

EMPTY_FEED = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Empty</title></channel></rss>"""


def test_clean_html_strips_tags_and_entities():
    assert clean_html("<p>Collaboration aims</p>") == "Collaboration aims"
    # 이중 이스케이프로 남은 문자 그대로의 "&nbsp;" (feedparser가 한 번만 디코드한 경우, 실제 관찰됨)
    assert clean_html("<p>Collaboration&nbsp;aims</p>") == "Collaboration aims"
    assert clean_html("a\xa0b") == "a b"  # 실제 유니코드 non-breaking space인 경우
    assert clean_html(None) == ""


def test_parse_entries_q4_style_two_digit_year():
    rows = parse_entries(Q4_STYLE_FEED, ticker="EX")
    assert len(rows) == 2
    r = rows[0]
    assert r["ticker"] == "EX"
    assert r["title"] == "Example Corp to Report Fiscal Third Quarter 2026 Financial Results"
    assert r["summary"] == ""  # 이 플랫폼은 summary가 비어있음
    assert r["published_parsed"].startswith("2026-10-06")  # 2자리 연도 "26"이 2026으로 해석됨


def test_parse_entries_other_style_has_summary_and_author():
    rows = parse_entries(OTHER_STYLE_FEED, ticker="EX2")
    assert len(rows) == 1
    r = rows[0]
    assert r["title"] == "Example Corp and Partner Collaborate on New Chips"  # <b> 태그 제거됨
    assert r["summary"] == "Collaboration aims to develop breakthroughs."
    assert r["author"] == "Example Corp"
    assert r["published_parsed"].startswith("2026-10-06")


def test_parse_entries_empty_feed():
    assert parse_entries(EMPTY_FEED, ticker="EX") == []


def _feed_with_guids(guids: list[str]) -> bytes:
    items = "".join(
        f"<item><title>T{g}</title><link>https://x/{g}</link>"
        f"<pubDate>Tue, 06 Oct 2026 09:00:00 -0400</pubDate><guid>{g}</guid></item>"
        for g in guids
    )
    return f"<?xml version='1.0'?><rss version='2.0'><channel><title>X</title>{items}</channel></rss>".encode()


def test_build_page_url():
    assert build_page_url("https://x/rss", 1) == "https://x/rss"  # 1페이지는 그대로
    assert build_page_url("https://x/rss", 2) == "https://x/rss?page=2"
    assert build_page_url("https://x/rss?a=b", 2) == "https://x/rss?a=b&page=2"


def test_fetch_all_pages_stops_on_short_page_when_pagination_supported():
    """실제 페이지네이션 지원 플랫폼: 페이지마다 새 guid가 오다가 빈 페이지에서 멈춘다."""
    pages = {
        "https://x/rss": _feed_with_guids(["a", "b"]),
        "https://x/rss?page=2": _feed_with_guids(["c", "d"]),
        "https://x/rss?page=3": _feed_with_guids([]),  # 빈 페이지 = 끝
    }

    def fake_fetcher(url):
        return pages[url]

    rows, pages_fetched = fetch_all_pages("X", "https://x/rss", user_agent="x", fetcher=fake_fetcher)
    assert pages_fetched == 3
    assert [r["guid"] for r in rows] == ["a", "b", "c", "d"]


def test_fetch_all_pages_stops_when_platform_ignores_page_param():
    """페이지네이션 미지원 플랫폼: ?page=를 무시하고 매번 같은 글을 돌려준다 → 1페이지에서 멈춘다."""
    same_page = _feed_with_guids(["a", "b"])

    def fake_fetcher(url):
        return same_page  # url(페이지 번호)과 무관하게 항상 동일

    rows, pages_fetched = fetch_all_pages("X", "https://x/rss", user_agent="x", fetcher=fake_fetcher)
    assert pages_fetched == 2  # 1페이지 성공 + 2페이지째가 전부 중복임을 확인하고 멈춤
    assert [r["guid"] for r in rows] == ["a", "b"]


def test_fetch_all_pages_respects_max_pages():
    def fake_fetcher(url):
        page_no = url.split("page=")[-1] if "page=" in url else "1"
        return _feed_with_guids([f"{page_no}-{i}" for i in range(2)])

    rows, pages_fetched = fetch_all_pages(
        "X", "https://x/rss", user_agent="x", max_pages=3, fetcher=fake_fetcher
    )
    assert pages_fetched == 3
    assert len(rows) == 6


def test_fetch_feed_bytes_retries_then_raises(monkeypatch):
    calls = []

    class FakeResp:
        status_code = 500
        content = b""

    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            calls.append(url)
            return FakeResp()

    monkeypatch.setattr("time.sleep", lambda s: None)  # 테스트에서 실제로 기다리지 않음
    with pytest.raises(RuntimeError):
        fetch_feed_bytes("https://x", user_agent="test-agent", session=FakeSession(), max_retries=3)
    assert len(calls) == 3


def test_load_feeds_reads_ticker_url_pairs(tmp_path):
    cfg = tmp_path / "f.yaml"
    cfg.write_text("tickers:\n  - ticker: ex\n    url: https://ir.example.com/rss\n", encoding="utf-8")
    feeds = load_feeds(cfg)
    assert feeds == [FeedSource(ticker="EX", url="https://ir.example.com/rss")]


def test_load_feeds_requires_ticker_and_url(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("tickers:\n  - ticker: ex\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_feeds(cfg)


def test_collect_continues_after_one_feed_fails(tmp_path):
    feeds = [FeedSource(ticker="OK", url="https://ok"), FeedSource(ticker="BAD", url="https://bad")]

    def fake_fetcher(url):
        if url == "https://bad":
            raise RuntimeError("network down")
        return Q4_STYLE_FEED

    stats = collect(feeds, tmp_path / "pr.csv", user_agent="x", fetcher=fake_fetcher)
    assert isinstance(stats, CollectStats)
    assert stats.rows == 2
    assert stats.per_ticker_rows == {"OK": 2}
    assert len(stats.errors) == 1 and "BAD" in stats.errors[0]

    df = pd.read_csv(tmp_path / "pr.csv", dtype=str)
    assert len(df) == 2 and set(df["ticker"]) == {"OK"}


def test_summarize_basic_counts():
    df = pd.DataFrame(
        [
            {"ticker": "EX", "published_parsed": "2026-10-06 00:00:00", "summary": ""},
            {"ticker": "EX", "published_parsed": "2026-09-25 00:00:00", "summary": "has text"},
            {"ticker": "EX2", "published_parsed": "2026-10-06 00:00:00", "summary": "has text"},
        ]
    )
    s = summarize(df)
    assert s["rows"] == 3
    assert s["by_ticker"]["EX"] == 2
    assert s["empty_summary_ratio_by_ticker"]["EX"] == pytest.approx(0.5)
    assert s["empty_summary_ratio_by_ticker"]["EX2"] == pytest.approx(0.0)
