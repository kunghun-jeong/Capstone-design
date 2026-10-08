"""회사 공식 IR(투자자 관계) 보도자료 RSS를 feedparser로 수집하는 모듈.

- 종목당 피드 하나(`configs/ir_press_release_feeds.yaml`)라서 종목 태깅이 필요 없다 — 이게 이
  소스를 고른 핵심 이유다. 여러 "일반 뉴스" 애그리게이터(Google News, SeekingAlpha, Nasdaq.com,
  Investing.com)는 기술적으론 접근되지만 "개인·비영리 열람만" 또는 "저장/복제 금지" 약관에 걸려
  제외했다 — 회사가 직접 배포하는 자사 보도자료는 그런 제약이 없다.
- 두 가지 IR 플랫폼이 섞여 있고 필드가 조금 다르다 (`configs/ir_press_release_feeds.yaml` 주석
  참고). `parse_entries`가 둘 다 같은 스키마로 평탄화한다.
- SEC EDGAR에서 겪은 Akamai/Cloudflare 봇 차단과 같은 종류의 차단이 NVDA/QCOM/MU IR 사이트에서도
  나타났다 — 그 세 종목은 우회하지 않고 제외했다 (reports/ir_press_releases_report.md 참고).

조사 결과는 `reports/ir_press_releases_report.md` 참고.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import feedparser
import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 20
MAX_RETRIES = 3
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class FeedSource:
    ticker: str
    url: str


def load_feeds(config_path: Path) -> list[FeedSource]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    rows = raw.get("tickers") if isinstance(raw, dict) else raw
    if not rows:
        raise ValueError(f"{config_path}에 'tickers' 목록이 없음")
    out = []
    for r in rows:
        if "ticker" not in r or "url" not in r:
            raise ValueError(f"ticker/url 필드가 없는 항목: {r}")
        out.append(FeedSource(ticker=str(r["ticker"]).strip().upper(), url=str(r["url"]).strip()))
    return out


def clean_html(text: str | None) -> str:
    """태그를 걷어내고 공백을 정리한다.

    `&nbsp;`는 두 가지 형태로 나타날 수 있다 — 실제 유니코드 non-breaking space(`\\xa0`, feedparser가
    XML 엔티티를 정상적으로 디코드한 경우)이거나, 설명문 자체가 이중으로 이스케이프돼 있어 feedparser가
    한 번만 풀고 남긴 문자 그대로의 `"&nbsp;"` 문자열(실제 IR 피드에서 관찰됨)이다. 둘 다 처리한다.
    """
    if not text:
        return ""
    stripped = _TAG_RE.sub(" ", text).replace("\xa0", " ").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", stripped).strip()


def build_page_url(url: str, page: int) -> str:
    """`?page=N`을 붙인다. 플랫폼에 따라 무시될 수 있다 (실제 동작은 fetch_all_pages가 guid로 확인)."""
    sep = "&" if "?" in url else "?"
    return url if page <= 1 else f"{url}{sep}page={page}"


def fetch_feed_bytes(
    url: str,
    *,
    user_agent: str,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> bytes:
    sess = session or requests
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_exc = e
        else:
            if resp.status_code == 200:
                return resp.content
            last_exc = RuntimeError(f"HTTP {resp.status_code} for {url}")
        if attempt < max_retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"{max_retries}회 재시도 후 실패: {url}") from last_exc


def parse_entries(feed_bytes: bytes, *, ticker: str) -> list[dict]:
    """RSS 바이트를 종목별 평탄한 dict 리스트로 바꾼다. 두 IR 플랫폼의 필드 차이를 흡수한다."""
    d = feedparser.parse(feed_bytes)
    if d.bozo and not d.entries:
        logger.warning("feedparser 파싱 실패 (ticker=%s): %s", ticker, d.get("bozo_exception"))
        return []
    rows = []
    for e in d.entries:
        rows.append(
            {
                "ticker": ticker,
                "title": clean_html(e.get("title", "")),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
                "published_parsed": (
                    time.strftime("%Y-%m-%d %H:%M:%S", e.published_parsed) if e.get("published_parsed") else ""
                ),
                "summary": clean_html(e.get("summary")),
                "author": e.get("author", ""),
                "guid": e.get("id", ""),
            }
        )
    return rows


def fetch_all_pages(
    ticker: str,
    url: str,
    *,
    user_agent: str,
    max_pages: int | None = None,
    sleep_sec: float = 0.5,
    fetcher: Callable[[str], bytes] | None = None,
    progress: Callable[[str], None] = lambda s: None,
) -> tuple[list[dict], int]:
    """`page`를 늘려가며 끝까지(또는 max_pages까지) 받는다.

    일부 IR 플랫폼(AVGO/AMAT 등)은 `?page=`를 무시하고 매번 같은 최신 10건을 돌려준다 — 그래서 "응답이
    비었으면 멈춘다"가 아니라 **"이 페이지의 글이 전부 이미 본 guid면 멈춘다"**로 종료 조건을 잡았다.
    이렇게 하면 페이지네이션을 지원하는 플랫폼(AMD/INTC/KLAC/MRVL 확인됨, 최대 약 130페이지)은 끝까지
    받고, 지원하지 않는 플랫폼은 1페이지만 받고 자연히 멈춘다.
    """
    get_bytes = fetcher or (lambda u: fetch_feed_bytes(u, user_agent=user_agent))
    seen_guids: set[str] = set()
    all_rows: list[dict] = []
    page = 1
    while True:
        raw = get_bytes(build_page_url(url, page))
        rows = parse_entries(raw, ticker=ticker)
        new_rows = [r for r in rows if r["guid"] not in seen_guids]
        progress(f"{ticker} 페이지 {page}: {len(rows)}건 중 신규 {len(new_rows)}건")
        if not rows or not new_rows:
            break
        for r in new_rows:
            seen_guids.add(r["guid"])
        all_rows.extend(new_rows)
        if max_pages is not None and page >= max_pages:
            break
        page += 1
        if fetcher is None:
            time.sleep(sleep_sec)
    return all_rows, page


@dataclass
class CollectStats:
    feeds: int = 0
    rows: int = 0
    pages: int = 0
    errors: list[str] = field(default_factory=list)
    per_ticker_rows: dict = field(default_factory=dict)
    per_ticker_pages: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "started_at"}
        d["elapsed_sec"] = round(time.time() - self.started_at, 1)
        return d


def collect(
    feeds: Iterable[FeedSource],
    out_path: Path,
    *,
    user_agent: str,
    max_pages: int | None = None,
    sleep_sec: float = 0.5,
    fetcher: Callable[[str], bytes] | None = None,
    progress: Callable[[str], None] = print,
) -> CollectStats:
    """피드 전체를 끝까지(또는 max_pages까지) 페이지네이션해 out_path CSV 하나로 합친다.

    한 피드 실패는 나머지를 막지 않는다. 페이지네이션 미지원 플랫폼은 `fetch_all_pages`가 1페이지에서
    자연히 멈춘다 (모듈 docstring 참고).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feeds = list(feeds)
    stats = CollectStats(feeds=len(feeds))
    all_rows: list[dict] = []

    for fs in feeds:
        try:
            rows, pages = fetch_all_pages(
                fs.ticker,
                fs.url,
                user_agent=user_agent,
                max_pages=max_pages,
                sleep_sec=sleep_sec,
                fetcher=fetcher,
                progress=progress,
            )
        except Exception as e:  # noqa: BLE001 — 한 종목 실패가 전체를 막지 않게
            stats.errors.append(f"{fs.ticker}: {e!r}")
            progress(f"[오류] {fs.ticker}: {e!r}")
            continue
        progress(f"{fs.ticker}: 총 {len(rows)}건 ({pages}페이지)")
        all_rows.extend(rows)
        stats.per_ticker_rows[fs.ticker] = len(rows)
        stats.per_ticker_pages[fs.ticker] = pages
        stats.pages += pages

    stats.rows = len(all_rows)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    try:
        pd.DataFrame(all_rows).to_csv(tmp, index=False)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out_path)
    return stats


# ------------------------------------------------------------------ 요약 ----


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"rows": len(df)}
    if df.empty:
        return out
    out["by_ticker"] = df["ticker"].value_counts()
    out["empty_summary_ratio_by_ticker"] = df.groupby("ticker")["summary"].apply(
        lambda s: float((s.fillna("") == "").mean())
    )
    dates = pd.to_datetime(df["published_parsed"], errors="coerce")
    out["date_min"], out["date_max"] = dates.min(), dates.max()
    return out


def print_summary(summary: dict, out: Callable[[str], None] = print) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 200):
        out(f"\n총 {summary['rows']:,}건")
        if "by_ticker" in summary:
            out("\n[종목별 건수]")
            out(summary["by_ticker"].to_string())
            out("\n[종목별 summary 빈칸 비율]")
            out(summary["empty_summary_ratio_by_ticker"].to_string())
            out(f"\n[기간] {summary['date_min']} ~ {summary['date_max']}")
