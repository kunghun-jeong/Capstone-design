"""Arctic Shift API 기반 Reddit 게시글·댓글 수집기.

서브레딧·종류(posts/comments)·UTC 월 단위로 [start, end) 구간을 ``created_utc`` 오름차순으로 순회한다.
원본은 ``<out_dir>/<subreddit>/<kind>/<YYYY-MM>.jsonl``에, 월별 진행 상태는 같은 폴더의
``<YYYY-MM>.state.json``에 저장한다. 실행 로그는 ``<out_dir>/_runs.jsonl``에 남긴다.

Arctic Shift API(2026-10 확인) 동작:
- ``after``·``before`` 둘 다 경계값을 포함하지 않는다(exclusive). 그래서 [a, b) 구간은
  ``after=a-1, before=b``로 요청하고, 다음 페이지는 "마지막 created_utc − 1"부터 다시 요청한 뒤 id로
  중복을 거른다(같은 초에 올라온 글이 페이지 경계에서 빠지지 않게).
- 서버 쪽 처리 시간 초과는 HTTP 422 + ``{"error": "Timeout..."}``로 온다 → 재시도.
  잘못된 파라미터·필드는 HTTP 400 → 즉시 실패.
- 키 없이 쓰는 무료 서비스다. 문서상 초당 1~2회 정도는 문제없다고 하므로 기본 간격은 0.5초.
- 게시 후 약 36시간 동안은 score·num_comments가 1 또는 0인 자리표시 값이다. CLI 기본 종료 시각을
  현재보다 36시간 이전으로 두는 이유다.

응답의 전체 필드(1건당 2~3.5KB)를 다 저장하지 않고 ``KEEP_FIELDS``만 남겨 저장 용량을 줄인다.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://arctic-shift.photon-reddit.com/api"
USER_AGENT = "skku-capstone-design/0.1 (+https://github.com/kunghun-jeong/Capstone-design)"
KINDS = ("posts", "comments")

PAGE_LIMIT = 100                # API 최대값
MIN_REQUEST_INTERVAL = 0.5      # 요청 간 최소 간격(초)
MAX_RETRIES = 6
MAX_PAGES_PER_UNIT = 100_000    # 한 서브레딧·종류·월 단위의 안전 한도 (WSB 댓글 한 달 ≈ 1만 페이지)
RUNS_LOG = "_runs.jsonl"

# 시점 정합(created_utc, retrieved_on), 삭제·제거 판별(_meta, removed_by_category, author),
# 참여도(score, num_comments, upvote_ratio), 스레드 구조(link_id, parent_id)를 남긴다.
KEEP_FIELDS: dict[str, tuple[str, ...]] = {
    "posts": (
        "id", "name", "created_utc", "retrieved_on", "_meta", "subreddit", "author",
        "title", "selftext", "url", "domain", "permalink", "is_self", "link_flair_text",
        "score", "upvote_ratio", "num_comments", "over_18", "stickied", "distinguished",
        "edited", "locked", "removed_by_category",
    ),
    "comments": (
        "id", "name", "created_utc", "retrieved_on", "_meta", "subreddit", "author",
        "body", "link_id", "parent_id", "link_title", "permalink", "score",
        "controversiality", "is_submitter", "stickied", "distinguished", "edited",
    ),
}


# --- 에러 ---
class RedditConfigError(RuntimeError):
    """설정 파일이나 인자가 유효하지 않음"""

class RedditRequestError(RuntimeError):
    """HTTP 요청 실패 또는 재시도 한도 초과"""

class RedditResponseError(RuntimeError):
    """API 응답이 예상한 형태가 아님"""

class RedditPaginationError(RuntimeError):
    """페이지가 더 이상 진행되지 않거나 안전 한도를 넘음"""

class RedditStorageError(RuntimeError):
    """JSONL·상태 파일을 읽거나 쓸 수 없음"""


@dataclass(frozen=True)
class Target:
    subreddit: str
    kind: str


def load_targets(config_path: Path) -> list[Target]:
    """YAML의 ``subreddits`` 목록(``name``, ``kinds``)을 Target 목록으로 불러온다."""
    import yaml

    try:
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise RedditConfigError(f"서브레딧 설정 파일을 읽을 수 없습니다: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise RedditConfigError(f"서브레딧 설정 파일의 YAML 형식이 올바르지 않습니다: {config_path}") from exc
    entries = config.get("subreddits") if isinstance(config, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RedditConfigError("설정 파일에는 비어 있지 않은 subreddits 목록이 필요합니다.")
    targets: list[Target] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise RedditConfigError(f"subreddits 항목에는 name 문자열이 필요합니다: {entry!r}")
        kinds = entry.get("kinds", list(KINDS))
        if not isinstance(kinds, list) or not kinds or any(kind not in KINDS for kind in kinds):
            raise RedditConfigError(f"kinds는 {KINDS} 중에서 골라야 합니다: {entry!r}")
        targets.extend(Target(entry["name"].strip(), kind) for kind in kinds)
    return targets


def _to_ts(value: date | datetime) -> int:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise RedditConfigError("날짜·시각에는 시간대 정보가 필요합니다.")
        return int(value.timestamp())
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp())


def _format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def month_units(start_ts: int, end_ts: int) -> list[tuple[str, int, int]]:
    """[start_ts, end_ts)를 UTC 월 경계로 잘라 ``(YYYY-MM, 구간 시작, 구간 끝)`` 목록으로 반환."""
    if start_ts >= end_ts:
        raise RedditConfigError("시작 시각은 종료 시각보다 빨라야 합니다.")
    units = []
    current = datetime.fromtimestamp(start_ts, timezone.utc)
    month_start = datetime(current.year, current.month, 1, tzinfo=timezone.utc)
    while int(month_start.timestamp()) < end_ts:
        next_month = datetime(
            month_start.year + month_start.month // 12, month_start.month % 12 + 1, 1, tzinfo=timezone.utc
        )
        unit_start = max(start_ts, int(month_start.timestamp()))
        unit_end = min(end_ts, int(next_month.timestamp()))
        units.append((month_start.strftime("%Y-%m"), unit_start, unit_end))
        month_start = next_month
    return units


@dataclass
class FetchStats:
    requests: int = 0
    successful_requests: int = 0
    retries: int = 0
    rate_limited_429: int = 0
    server_timeouts_422: int = 0
    server_errors: int = 0
    connection_errors: int = 0
    items: int = 0              # 새로 저장한 게시글·댓글 수
    units_done: int = 0         # 끝까지 수집한 서브레딧·종류·월 단위 수
    units_skipped: int = 0      # 이미 완료돼 건너뛴 단위 수
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **counts: int) -> None:
        with self._lock:
            for name, value in counts.items():
                setattr(self, name, getattr(self, name) + value)

    def as_dict(self) -> dict[str, Any]:
        result = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        result["elapsed_sec"] = round(time.time() - self.started_at, 1)
        return result


class ArcticShiftClient:
    """세션·시계 함수를 주입할 수 있는 requests 기반 클라이언트. 스레드마다 하나씩 만든다."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        timeout: float = 120.0,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        min_interval: float = MIN_REQUEST_INTERVAL,
        max_pages: int = MAX_PAGES_PER_UNIT,
        stats: FetchStats | None = None,
    ):
        if min_interval < MIN_REQUEST_INTERVAL:
            raise ValueError(f"min_interval은 무료 서비스 부담을 줄이기 위해 {MIN_REQUEST_INTERVAL}초 이상이어야 합니다.")
        self._session = session if session is not None else requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._sleep = sleep
        self._monotonic = monotonic
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.min_interval = min_interval
        self.max_pages = max_pages
        self._last_request: float | None = None
        self.stats = stats if stats is not None else FetchStats()

    def _throttle(self) -> None:
        now = self._monotonic()
        if self._last_request is not None:
            wait = self.min_interval - (now - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._monotonic()

    def _backoff(self, attempt: int, reset_after: str | None) -> float:
        delay = min(self.backoff_base * (2**attempt), self.backoff_max)
        if reset_after:
            try:
                delay = max(delay, float(reset_after) + 1)
            except ValueError:
                pass
        return delay

    def get_page(self, kind: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        if kind not in KINDS:
            raise RedditConfigError(f"kind는 {KINDS} 중 하나여야 합니다: {kind}")
        url = f"{API_BASE}/{kind}/search"
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats.add(requests=1)
            reset_after: str | None = None
            try:
                response = self._session.get(url, params=dict(params), timeout=self.timeout, allow_redirects=False)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                self.stats.add(connection_errors=1)
                last_error = type(exc).__name__
            else:
                status = response.status_code
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                error = payload.get("error") if isinstance(payload, dict) else None
                if status == 200 and isinstance(payload, dict) and not error:
                    data = payload.get("data")
                    if not isinstance(data, list):
                        raise RedditResponseError("Arctic Shift 응답에 data 배열이 없습니다.")
                    self.stats.add(successful_requests=1)
                    return data
                if status == 400:
                    raise RedditRequestError(f"Arctic Shift 요청 파라미터 오류 (HTTP 400): {error}")
                if status == 429:
                    self.stats.add(rate_limited_429=1)
                    reset_after = response.headers.get("x-ratelimit-reset")
                    last_error = "HTTP 429"
                elif status == 422:
                    self.stats.add(server_timeouts_422=1)
                    last_error = f"HTTP 422 ({error})"
                elif status == 200 or status == 408 or 500 <= status <= 599:
                    if status >= 500:
                        self.stats.add(server_errors=1)
                    last_error = f"HTTP {status} ({error})" if error else f"HTTP {status}"
                else:
                    raise RedditRequestError(f"Arctic Shift 요청 실패 (HTTP {status}): {error}")

            if attempt >= self.max_retries:
                break
            self.stats.add(retries=1)
            delay = self._backoff(attempt, reset_after)
            logger.warning("%s — %.1f초 후 재시도 (%d/%d)", last_error, delay, attempt + 1, self.max_retries)
            self._sleep(delay)
        raise RedditRequestError(f"재시도 한도 초과 ({last_error}).")

    def iter_items(
        self, subreddit: str, kind: str, after_exclusive: int, before_exclusive: int
    ) -> Iterator[tuple[dict[str, Any], int]]:
        """(after, before) 구간 항목을 오름차순으로 순회한다. 각 항목과 함께 다음 요청에 쓸 커서를 준다.

        페이지가 꽉 찼으면 "마지막 created_utc − 1"부터 다시 요청하므로 같은 항목이 다시 올 수 있다.
        중복 제거는 호출 쪽(id 기준)에서 한다.
        """
        after = after_exclusive
        previous_ids: set[str] = set()
        for _ in range(self.max_pages):
            page = self.get_page(
                kind,
                {"subreddit": subreddit, "after": after, "before": before_exclusive,
                 "limit": PAGE_LIMIT, "sort": "asc"},
            )
            page_ids: set[str] = set()
            last_ts = after
            for index, item in enumerate(page):
                if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                    raise RedditResponseError(f"{index}번 항목에 유효한 id가 없습니다.")
                created = item.get("created_utc")
                if not isinstance(created, (int, float)) or isinstance(created, bool):
                    raise RedditResponseError(f"{index}번 항목에 유효한 created_utc가 없습니다.")
                page_ids.add(item["id"])
                last_ts = max(last_ts, int(created))
            if len(page) < PAGE_LIMIT:
                for item in page:
                    yield item, before_exclusive - 1
                return
            if page_ids <= previous_ids:
                raise RedditPaginationError(
                    f"같은 초({last_ts})에 {PAGE_LIMIT}건 이상이 있어 페이지가 진행되지 않습니다."
                )
            next_after = last_ts - 1
            for item in page:
                yield item, next_after
            previous_ids = page_ids
            after = next_after
        raise RedditPaginationError(f"페이지 수가 안전 한도 {self.max_pages}를 초과했습니다.")


def trim_item(item: Mapping[str, Any], kind: str) -> dict[str, Any]:
    return {key: item[key] for key in KEEP_FIELDS[kind] if key in item}


def _read_ids(path: Path) -> set[str]:
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
                    raise RedditStorageError(f"손상된 JSONL 레코드: {path}:{line_number}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                    raise RedditStorageError(f"JSONL 레코드에 유효한 id가 없습니다: {path}:{line_number}")
                ids.add(record["id"])
    except OSError as exc:
        raise RedditStorageError(f"JSONL 파일을 읽을 수 없습니다: {path}") from exc
    return ids


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RedditStorageError(f"상태 파일을 읽을 수 없습니다: {path}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("next_after", 0), int):
        raise RedditStorageError(f"상태 파일 구조가 올바르지 않습니다: {path}")
    return state


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(dict(state), ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise RedditStorageError(f"상태 파일을 저장할 수 없습니다: {path}") from exc


def collect_unit(
    client: ArcticShiftClient,
    target: Target,
    month: str,
    unit_start: int,
    unit_end: int,
    out_dir: Path,
    *,
    progress: Callable[[str], None] = print,
    should_stop: Callable[[], bool] = lambda: False,
) -> int:
    """한 서브레딧·종류의 한 달 구간 [unit_start, unit_end)를 끝까지 수집한다.

    페이지마다 JSONL에 덧붙인 뒤 상태 파일의 커서를 갱신하므로, 중간에 끊겨도 다음 실행이
    마지막 커서부터 이어받는다(이미 저장된 id는 건너뜀). ``should_stop``이 참이 되면 그 페이지까지
    저장하고 미완료 상태로 멈춘다.
    """
    unit_dir = Path(out_dir) / target.subreddit / target.kind
    try:
        unit_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RedditStorageError(f"저장 디렉터리를 만들 수 없습니다: {unit_dir}") from exc
    data_path = unit_dir / f"{month}.jsonl"
    state_path = unit_dir / f"{month}.state.json"
    state = _read_state(state_path)

    if state.get("unit_start", unit_start) > unit_start:
        raise RedditConfigError(
            f"{target.subreddit}/{target.kind}/{month}: 기존 수집 시작({_format_ts(state['unit_start'])})보다 "
            "이른 구간은 이어받을 수 없습니다. 다른 --out-dir을 쓰세요."
        )
    after = max(state.get("next_after", unit_start - 1), unit_start - 1)
    if after >= unit_end - 1:
        client.stats.add(units_skipped=1)
        return 0

    seen = _read_ids(data_path)
    saved = 0
    buffer: list[str] = []
    cursor = after

    def flush(next_after: int, complete: bool) -> None:
        nonlocal buffer
        if buffer:
            try:
                with open(data_path, "a", encoding="utf-8", newline="") as handle:
                    handle.writelines(buffer)
            except OSError as exc:
                raise RedditStorageError(f"항목을 저장할 수 없습니다: {data_path}") from exc
            buffer = []
        _write_state(state_path, {
            "unit_start": state.get("unit_start", unit_start),
            "next_after": next_after,
            "complete_until": _format_ts(unit_end) if complete else None,
            "items": len(seen),
        })

    for item, next_after in client.iter_items(target.subreddit, target.kind, after, unit_end):
        if next_after != cursor:
            if buffer:
                flush(cursor, complete=False)  # 페이지 경계: 앞 페이지까지 저장된 상태로 커서 갱신
            if should_stop():
                client.stats.add(items=saved)
                progress(f"[{target.subreddit}/{target.kind} {month}] 중단 요청 — 커서 {_format_ts(cursor + 1)}까지 저장")
                return saved
        cursor = next_after
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        buffer.append(json.dumps(trim_item(item, target.kind), ensure_ascii=False) + "\n")
        saved += 1
    flush(unit_end - 1, complete=True)

    client.stats.add(items=saved, units_done=1)
    progress(
        f"[{target.subreddit}/{target.kind} {month}] {_format_ts(after + 1)}~{_format_ts(unit_end)} (끝 제외): "
        f"신규 {saved}건, 누적 {len(seen)}건 (전체 요청 {client.stats.requests})"
    )
    return saved


def log_run(
    out_dir: Path, targets: list[Target], start: date | datetime, end_exclusive: date | datetime, stats: FetchStats
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "targets": [f"{t.subreddit}/{t.kind}" for t in targets],
            "start": _format_ts(_to_ts(start)),
            "end_exclusive": _format_ts(_to_ts(end_exclusive)),
            **stats.as_dict(),
        }
        record.pop("started_at", None)
        with open(out_dir / RUNS_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise RedditStorageError(f"실행 로그를 저장할 수 없습니다: {out_dir / RUNS_LOG}") from exc
