"""Massive Stocks News API 수집기.

티커별 요청 기간 전체를 조회하고 모든 ``next_url`` 페이지를 순회한다.
원본 기사는 ``<out_dir>/<SYMBOL>/<YYYY>.jsonl``에 저장한다.
CLI 기본 ``out_dir``는 ``data/raw/massive_news_ticker``이며, 체크포인트와 실행 로그도 같은 구조에 저장한다.

API 키는 ``MASSIVE_API_KEY`` 환경변수에서만 읽는다.
요청 간격은 분당 최대 5회 제한에 맞춰 항상 12초 이상으로 유지한다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

NEWS_URL = "https://api.massive.com/v2/reference/news"
API_KEY_ENV = "MASSIVE_API_KEY"

MIN_REQUEST_INTERVAL = 12.0    # MASSIVE BASIC PLAN은 분당 요청 횟수 최대 5번으로 제한
MAX_LIMIT = 1000               # 문서상 최대값, limit 파라미터의 실제 범위는 10~1000
MAX_PAGES_PER_TICKER = 10_000  # 티커당 최대 요청 페이지 수 (티커당 1000만건/11년)
MAX_RETRIES = 4                # 최대 재시도 횟수
STATE_FILE = "_state.json"     # ticker별 상태 기록 파일
RUNS_LOG = "_runs.jsonl"       # 전체 실행 로그 기록 파일
ENV_FILE = Path(__file__).resolve().parents[3] / ".env" # .env 파일


# --- [1] API 및 접근 제어 관련 에러 ---
class MassiveAPIError(RuntimeError):
    """모든 인증 에러 관련 부모 클래스"""

class MassiveCredentialsError(MassiveAPIError):
    """로컬 환경에 API 키나 시크릿이 설정되어 있지 않음 (요청 전 차단)"""

class MassiveUnauthorizedError(MassiveAPIError):
    """401 Unauthorized (인증 실패)"""

class MassivePermissionError(MassiveAPIError):
    """403 Forbidden (권한 또는 요금제 부족)"""


# --- [2] 네트워크 및 요청 관련 에러 ---
class MassiveRequestError(RuntimeError):
    """HTTP 요청 또는 네트워크 전송 실패"""


# --- [3] 데이터 응답 및 파싱 / 로직 에러 ---
class MassiveResponseError(RuntimeError):
    """API 응답이 예상한 형태의 유효한 JSON 형식이 아님"""

class MassivePaginationError(RuntimeError):
    """페이지네이션 과정에서 안전하지 않은 URL, 중복된 커서, 또는 허용치를 넘는 과도한 페이지 발생"""


# --- [5] 파일 시스템 및 저장소 에러 ---
class MassiveStorageError(RuntimeError):
    """JSONL 또는 체크포인트 파일을 안전하게 읽거나 쓸 수 없음"""


# --- [6] 설정 및 검증 에러 ---
class MassiveConfigError(RuntimeError):
    """YAML 설정 파일 형식 또는 티커 값이 유효하지 않음"""


def load_api_key() -> str:
    """저장소 루트의 .env를 현재 프로세스 환경에 로드한 뒤 API 키를 반환.
    같은 이름의 환경변수가 이미 있으면 기존 환경변수 값 우선.
    """
    load_dotenv(dotenv_path=ENV_FILE)
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise MassiveCredentialsError(f"환경변수 {API_KEY_ENV}가 설정되어 있지 않습니다.")
    return api_key


def load_universe(config_path: Path) -> list[str]:
    """YAML 파일의 ``tickers`` 리스트에서 티커 문자열들을 불러옴"""
    import yaml

    try:
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise MassiveConfigError(f"티커 설정 파일을 읽을 수 없습니다: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise MassiveConfigError(f"티커 설정 파일의 YAML 형식이 올바르지 않습니다: {config_path}") from exc
    tickers = config.get("tickers") if isinstance(config, dict) else None
    if not isinstance(tickers, list) or not tickers:
        raise MassiveConfigError("티커 설정 파일에는 비어 있지 않은 tickers 목록이 필요합니다.")
    invalid = [ticker for ticker in tickers if not isinstance(ticker, str) or not ticker.strip()]
    if invalid:
        raise MassiveConfigError(f"티커는 비어 있지 않은 문자열이어야 합니다: {invalid}")
    return [ticker.strip().upper() for ticker in tickers]


def _as_utc_datetime(value: date | datetime) -> datetime:
    """날짜는 UTC 자정으로, 시간대가 있는 시각은 UTC 시각으로 변환한다."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise MassiveConfigError("날짜·시각에는 시간대 정보가 필요합니다.")
        return value.astimezone(timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def _format_utc_datetime(value: date | datetime) -> str:
    return _as_utc_datetime(value).isoformat(timespec="auto").replace("+00:00", "Z")


def _parse_checkpoint_datetime(value: Any) -> datetime | None:
    """체크포인트의 날짜·시각을 UTC 시각으로 변환하고, 형식이 다르면 None을 반환한다."""
    if not isinstance(value, str):
        return None
    try:
        value_date = date.fromisoformat(value)
    except ValueError:
        value_date = None
    if value_date is not None:
        return datetime(value_date.year, value_date.month, value_date.day, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


@dataclass
class FetchStats:
    requests: int = 0               # API 요청 시도
    successful_requests: int = 0    # 성공한 API 응답 횟수
    retries: int = 0                # 실패 후 재시도 횟수(최대 4)
    rate_limited_429: int = 0       # HTTP 429 응답 횟수
    server_errors: int = 0          # 서버 에러 발생 횟수
    connection_errors: int = 0      # 연결 오류 발생 횟수
    articles: int = 0               # 새로 저장한 기사 수
    tickers_done: int = 0           # 수집 완료 ticker 수
    tickers_skipped: int = 0        # 이미 완료된 것으로 확인해 건너뛴 티커 수
    started_at: float = field(default_factory=time.time) # 객체 생성 시각 기록

    def as_dict(self) -> dict[str, Any]:
        result = self.__dict__.copy()
        result["elapsed_sec"] = round(time.time() - self.started_at, 1)
        return result


class MassiveNewsClient:
    """세션(Session)과 시간(Clock) 함수를 주입받아 사용 가능한, requests 기반 소형 클라이언트."""

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
        max_pages: int = MAX_PAGES_PER_TICKER,
    ):
        if not api_key:
            raise MassiveCredentialsError(f"환경변수 {API_KEY_ENV}가 설정되어 있지 않습니다.")
        if min_interval < MIN_REQUEST_INTERVAL:
            raise ValueError(f"min_interval은 분당 5회 제한을 위해 {MIN_REQUEST_INTERVAL}초 이상이어야 합니다.")

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
                if status == 401:
                    raise MassiveUnauthorizedError(f"Massive API 인증 오류 (HTTP {status}).")
                if status == 403:
                    raise MassivePermissionError(f"Massive API 권한 오류 (HTTP {status}).")
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

    def iter_articles(
        self, ticker: str, start: date | datetime, end_exclusive: date | datetime
    ) -> Iterator[dict[str, Any]]:
        """하나의 티커(종목)에 대해 [start, end_exclusive) 범위 모든 기사 순회(반복)"""
        start_utc = _as_utc_datetime(start)
        end_utc = _as_utc_datetime(end_exclusive)
        if start_utc >= end_utc:
            raise MassiveConfigError("시작 시각은 종료 시각보다 빨라야 합니다.")
        params: dict[str, Any] | None = {
            "ticker": ticker.strip().upper(),
            "published_utc.gte": _format_utc_datetime(start_utc),
            "published_utc.lt": _format_utc_datetime(end_utc),
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
    """인증 헤더 전달 전, 요청하려는 엔드포인트가 Massive 공식 HTTPS 뉴스 엔드포인트가 맞는지 확인하고 수락"""
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


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_successful_end_exclusive": None, "completed_tickers": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MassiveStorageError(f"체크포인트 파일을 읽을 수 없습니다: {path}") from exc
    if not isinstance(state, dict):
        raise MassiveStorageError(f"체크포인트 구조가 올바르지 않습니다: {path}")
    if not isinstance(state.get("completed_tickers", {}), dict):
        raise MassiveStorageError(f"체크포인트 구조가 올바르지 않습니다: {path}")
    watermark = state.get("last_successful_end_exclusive")
    if watermark is not None and _parse_checkpoint_datetime(watermark) is None:
        raise MassiveStorageError(f"체크포인트 종료 시각이 올바르지 않습니다: {path}")
    return {
        "last_successful_end_exclusive": watermark,
        "completed_tickers": state.get("completed_tickers", {}),
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        progress = {"last_successful_end_exclusive": state["last_successful_end_exclusive"]}
        temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
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
    start: date | datetime,
    end_exclusive: date | datetime,
    out_dir: Path,
    *,
    progress: Callable[[str], None] = print,
) -> int:
    """한 Symbol(ticker)의 전체 기간 데이터 수집, pagination이 완료되면 체크포인트 저장"""
    ticker = symbol.strip().upper()
    start_utc = _as_utc_datetime(start)
    end_utc = _as_utc_datetime(end_exclusive)
    if start_utc >= end_utc:
        raise MassiveConfigError("시작 시각은 종료 시각보다 빨라야 합니다.")
    start_text = _format_utc_datetime(start_utc)
    end_text = _format_utc_datetime(end_utc)
    ticker_dir = Path(out_dir) / ticker
    try:
        ticker_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MassiveStorageError(f"티커 저장 디렉터리를 만들 수 없습니다: {ticker_dir}") from exc
    state_path = ticker_dir / STATE_FILE
    state = _read_state(state_path)
    seen_ids = _existing_ids(ticker_dir)
    watermark = _parse_checkpoint_datetime(state["last_successful_end_exclusive"])
    if watermark is None:
        legacy_watermarks = []
        for completed in state["completed_tickers"].values():
            if not isinstance(completed, dict):
                continue
            completed_start = _parse_checkpoint_datetime(completed.get("start"))
            completed_end = _parse_checkpoint_datetime(completed.get("end_exclusive"))
            if completed_start == start_utc and completed_end is not None:
                legacy_watermarks.append(completed_end)
        if legacy_watermarks:
            watermark = max(legacy_watermarks)

    effective_start = max(start_utc, watermark) if watermark is not None else start_utc
    if effective_start >= end_utc:
        client.stats.tickers_skipped += 1
        return 0

    saved_total = 0
    for article in client.iter_articles(ticker, effective_start, end_utc):
        if _append_article(article, ticker_dir, seen_ids):
            saved_total += 1

    # 모든 페이지 수집이 성공한 뒤에만 다음 실행 기준 시각을 갱신한다.
    state["last_successful_end_exclusive"] = end_text
    _write_state(state_path, state)
    client.stats.articles += saved_total
    client.stats.tickers_done += 1
    progress(
        f"[{ticker}] {_format_utc_datetime(effective_start)}~{end_text} (끝 제외): "
        f"신규 {saved_total}건 (누적 요청 {client.stats.requests})"
    )
    return saved_total


def log_run(
    out_dir: Path,
    symbols: list[str],
    start: date | datetime,
    end_exclusive: date | datetime,
    stats: FetchStats,
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbols": symbols,
            "start": _format_utc_datetime(start),
            "end_exclusive": _format_utc_datetime(end_exclusive),
            **stats.as_dict(),
        }
        record.pop("started_at", None)
        with open(out_dir / RUNS_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise MassiveStorageError(f"실행 통계를 저장할 수 없습니다: {out_dir / RUNS_LOG}") from exc
