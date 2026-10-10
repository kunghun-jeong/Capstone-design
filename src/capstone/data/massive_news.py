"""Massive Stocks News API 수집기.

티커별 월 단위로 뉴스를 조회하고 모든 ``next_url`` 페이지를 순회한다.
원본 기사는 ``<out_dir>/<SYMBOL>/<UTC year>.jsonl``에 저장한다.

API 키는 ``MASSIVE_API_KEY`` 환경변수에서만 읽는다. 요청 간격은 분당
최대 5회 제한에 맞춰 항상 12초 이상으로 유지한다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

NEWS_URL = "https://api.massive.com/v2/reference/news"
API_KEY_ENV = "MASSIVE_API_KEY"
MIN_REQUEST_INTERVAL = 12.0
MAX_LIMIT = 1000
MAX_PAGES_PER_MONTH = 10_000
MAX_RETRIES = 4
STATE_FILE = "_state.json"
RUNS_LOG = "_runs.jsonl"


class MassiveCredentialsError(RuntimeError):
    """The required API key is not configured."""


class MassiveAuthError(RuntimeError):
    """The API rejected the key or the account lacks permission."""


class MassiveRequestError(RuntimeError):
    """An HTTP or transport request failed."""


class MassiveResponseError(RuntimeError):
    """The API response was not valid JSON in the expected shape."""


class MassivePaginationError(RuntimeError):
    """Pagination returned an unsafe URL, repeated cursor, or too many pages."""


class MassiveStorageError(RuntimeError):
    """A JSONL or checkpoint file could not be read or written safely."""


def load_api_key() -> str:
    """Read the key from the process environment without exposing its value."""
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise MassiveCredentialsError(f"환경변수 {API_KEY_ENV}가 설정되어 있지 않습니다.")
    return api_key


def load_universe(config_path: Path) -> list[str]:
    """Load ticker strings from a YAML ``tickers`` list."""
    import yaml

    try:
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise ValueError(f"티커 설정 파일을 읽을 수 없습니다: {config_path}") from exc
    tickers = config.get("tickers") if isinstance(config, dict) else None
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("티커 설정 파일에는 비어 있지 않은 tickers 목록이 필요합니다.")
    invalid = [ticker for ticker in tickers if not isinstance(ticker, str) or not ticker.strip()]
    if invalid:
        raise ValueError(f"티커는 비어 있지 않은 문자열이어야 합니다: {invalid}")
    return [ticker.strip().upper() for ticker in tickers]


@dataclass
class FetchStats:
    requests: int = 0
    successful_requests: int = 0
    retries: int = 0
    rate_limited_429: int = 0
    server_errors: int = 0
    connection_errors: int = 0
    articles: int = 0
    months_done: int = 0
    months_skipped: int = 0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        result = self.__dict__.copy()
        result["elapsed_sec"] = round(time.time() - self.started_at, 1)
        return result


class MassiveNewsClient:
    """Small requests-based client with injectable session and clock functions."""

    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        timeout: float = 30.0,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        min_interval: float = MIN_REQUEST_INTERVAL,
        max_pages: int = MAX_PAGES_PER_MONTH,
    ):
        if not api_key:
            raise MassiveCredentialsError(f"환경변수 {API_KEY_ENV}가 설정되어 있지 않습니다.")
        if min_interval < MIN_REQUEST_INTERVAL:
            raise ValueError(f"min_interval은 분당 5회 제한을 위해 {MIN_REQUEST_INTERVAL}초 이상이어야 합니다.")
        if max_retries < 0 or max_pages < 1:
            raise ValueError("max_retries는 0 이상, max_pages는 1 이상이어야 합니다.")

        self._session = session if session is not None else requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._sleep = sleep
        self._monotonic = monotonic
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.min_interval = min_interval
        self.max_pages = max_pages
        self._last_request: float | None = None
        self.stats = FetchStats()

    def _throttle(self) -> None:
        now = self._monotonic()
        if self._last_request is not None:
            wait = self.min_interval - (now - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._monotonic()

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        delay = min(self.backoff_base * (2**attempt), self.backoff_max)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        return delay

    def _get_json(self, url: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats.requests += 1
            retry_after: str | None = None
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                self.stats.connection_errors += 1
                last_error = type(exc).__name__
            else:
                status = response.status_code
                if status == 200:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise MassiveResponseError("Massive API가 올바른 JSON을 반환하지 않았습니다.") from exc
                    if not isinstance(payload, dict):
                        raise MassiveResponseError("Massive API 응답의 최상위 값은 객체여야 합니다.")
                    self.stats.successful_requests += 1
                    return payload
                if status in (401, 403):
                    raise MassiveAuthError(f"Massive API 인증 또는 권한 오류 (HTTP {status}).")
                if status == 429:
                    self.stats.rate_limited_429 += 1
                    last_error = "HTTP 429"
                    retry_after = response.headers.get("Retry-After")
                elif status == 408 or 500 <= status <= 599:
                    if status >= 500:
                        self.stats.server_errors += 1
                    last_error = f"HTTP {status}"
                    retry_after = response.headers.get("Retry-After")
                else:
                    raise MassiveRequestError(f"Massive API 요청 실패 (HTTP {status}).")

            if attempt >= self.max_retries:
                break
            self.stats.retries += 1
            delay = self._backoff(attempt, retry_after)
            logger.warning("%s — %.1f초 후 재시도 (%d/%d)", last_error, delay, attempt + 1, self.max_retries)
            self._sleep(delay)
        raise MassiveRequestError(f"재시도 한도 초과 ({last_error}).")

    def iter_articles(self, ticker: str, start: date, end_exclusive: date) -> Iterator[dict[str, Any]]:
        """Iterate all articles in ``[start, end_exclusive)`` for one ticker."""
        params: dict[str, Any] | None = {
            "ticker": ticker.strip().upper(),
            "published_utc.gte": f"{start.isoformat()}T00:00:00Z",
            "published_utc.lt": f"{end_exclusive.isoformat()}T00:00:00Z",
            "sort": "published_utc",
            "order": "asc",
            "limit": MAX_LIMIT,
        }
        url = NEWS_URL
        seen_urls: set[str] = set()

        for _ in range(self.max_pages):
            if url in seen_urls:
                raise MassivePaginationError("Massive API 페이지네이션 URL이 반복되었습니다.")
            seen_urls.add(url)
            payload = self._get_json(url, params)
            params = None  # next_url already contains the cursor and query parameters.
            results = payload.get("results")
            if not isinstance(results, list):
                raise MassiveResponseError("Massive API 응답에 results 배열이 없습니다.")
            for index, article in enumerate(results):
                if not isinstance(article, dict):
                    raise MassiveResponseError(f"뉴스 결과 {index}번 항목이 객체가 아닙니다.")
                if not isinstance(article.get("id"), str) or not article["id"]:
                    raise MassiveResponseError(f"뉴스 결과 {index}번 항목에 유효한 id가 없습니다.")
                if not isinstance(article.get("published_utc"), str):
                    raise MassiveResponseError(f"뉴스 결과 {index}번 항목에 published_utc가 없습니다.")
                yield article

            next_url = payload.get("next_url")
            if not next_url:
                return
            if not isinstance(next_url, str):
                raise MassivePaginationError("Massive API next_url은 문자열이어야 합니다.")
            url = validate_next_url(next_url)

        raise MassivePaginationError(f"페이지 수가 안전 한도 {self.max_pages}를 초과했습니다.")


def validate_next_url(value: str) -> str:
    """Accept only Massive's HTTPS news endpoint before forwarding auth headers."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise MassivePaginationError("Massive API next_url이 올바른 URL이 아닙니다.") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.massive.com"
        or port not in (None, 443)
        or parsed.path != "/v2/reference/news"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise MassivePaginationError("Massive API next_url의 호스트 또는 경로가 허용되지 않습니다.")
    return value


def month_chunks(start: date, end: date) -> list[tuple[str, date, date]]:
    """Split inclusive dates into calendar months represented as half-open ranges."""
    if start > end:
        raise ValueError("start는 end보다 늦을 수 없습니다.")
    chunks: list[tuple[str, date, date]] = []
    cursor = date(start.year, start.month, 1)
    end_exclusive = end + timedelta(days=1)
    while cursor < end_exclusive:
        next_month = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
        lo = max(cursor, start)
        hi = min(next_month, end_exclusive)
        chunks.append((cursor.strftime("%Y-%m"), lo, hi))
        cursor = next_month
    return chunks


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MassiveStorageError(f"체크포인트 파일을 읽을 수 없습니다: {path}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("completed"), dict):
        raise MassiveStorageError(f"체크포인트 구조가 올바르지 않습니다: {path}")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise MassiveStorageError(f"체크포인트를 저장할 수 없습니다: {path}") from exc


def _article_year(article: Mapping[str, Any]) -> int:
    value = article.get("published_utc")
    if not isinstance(value, str):
        raise MassiveResponseError("뉴스 기사에 published_utc가 없습니다.")
    try:
        published = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MassiveResponseError("뉴스 기사의 published_utc 형식이 올바르지 않습니다.") from exc
    if published.tzinfo is None:
        raise MassiveResponseError("published_utc에는 시간대 정보가 필요합니다.")
    return published.astimezone(timezone.utc).year


def _read_jsonl_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MassiveStorageError(f"손상된 JSONL 레코드: {path}:{line_number}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                    raise MassiveStorageError(f"JSONL 레코드에 유효한 id가 없습니다: {path}:{line_number}")
                ids.add(record["id"])
    except OSError as exc:
        raise MassiveStorageError(f"JSONL 파일을 읽을 수 없습니다: {path}") from exc
    return ids


def _existing_ids(ticker_dir: Path) -> set[str]:
    ids: set[str] = set()
    if ticker_dir.exists():
        for path in ticker_dir.glob("[0-9][0-9][0-9][0-9].jsonl"):
            ids.update(_read_jsonl_ids(path))
    return ids


def _append_article(article: Mapping[str, Any], ticker_dir: Path, seen_ids: set[str]) -> bool:
    article_id = article["id"]
    if article_id in seen_ids:
        return False
    path = ticker_dir / f"{_article_year(article)}.jsonl"
    try:
        with open(path, "a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(dict(article), ensure_ascii=False) + "\n")
    except OSError as exc:
        raise MassiveStorageError(f"뉴스 기사를 저장할 수 없습니다: {path}") from exc
    seen_ids.add(article_id)
    return True


def collect_symbol(
    client: MassiveNewsClient,
    symbol: str,
    start: date,
    end: date,
    out_dir: Path,
    *,
    progress: Callable[[str], None] = print,
) -> int:
    """Collect one symbol, checkpoint completed months, and return new row count."""
    ticker = symbol.strip().upper()
    if not ticker:
        raise ValueError("티커는 비어 있을 수 없습니다.")
    months = month_chunks(start, end)
    ticker_dir = Path(out_dir) / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    state_path = ticker_dir / STATE_FILE
    state = _read_state(state_path)
    seen_ids = _existing_ids(ticker_dir)
    saved_total = 0

    for month, lo, hi in months:
        expected = {"start": lo.isoformat(), "end_exclusive": hi.isoformat()}
        if state["completed"].get(month) == expected:
            client.stats.months_skipped += 1
            continue

        month_saved = 0
        for article in client.iter_articles(ticker, lo, hi):
            if _append_article(article, ticker_dir, seen_ids):
                month_saved += 1
                saved_total += 1
        # State is committed only after every page and article in the month succeeds.
        state["completed"][month] = expected
        _write_state(state_path, state)
        client.stats.articles += month_saved
        client.stats.months_done += 1
        progress(f"[{ticker}] {month}: 신규 {month_saved}건 (누적 요청 {client.stats.requests})")
    return saved_total


def log_run(out_dir: Path, symbols: list[str], start: date, end: date, stats: FetchStats) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbols": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        **stats.as_dict(),
    }
    record.pop("started_at", None)
    try:
        with open(out_dir / RUNS_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise MassiveStorageError(f"실행 통계를 저장할 수 없습니다: {out_dir / RUNS_LOG}") from exc
