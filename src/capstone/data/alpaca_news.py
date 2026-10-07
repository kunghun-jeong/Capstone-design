"""Alpaca Market Data 뉴스 API(Benzinga 기반) 수집기.

- 종목 x 월 단위로 구간을 쪼개 `next_page_token`을 끝까지 따라가며 수집한다.
- 429/5xx/연결 오류는 백오프 후 재시도하고, 요청 사이에는 최소 간격을 둔다.
- 완료된 (종목, 월) 구간은 상태 파일에 기록해 재실행 시 건너뛴다 (이어받기).
- 원본은 `<out_dir>/<SYMBOL>/<YYYY>.jsonl`에 기사 1건=1줄로 저장한다. 가공은 하지 않는다.

API 키는 환경변수(APCA_API_KEY_ID, APCA_API_SECRET_KEY)에서만 읽으며 로그/예외에 남기지 않는다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import requests

logger = logging.getLogger(__name__)

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"

PAGE_LIMIT = 50  # 문서상 최대값
MAX_PAGES_PER_CHUNK = 2000  # 무한 루프 방지용 안전장치 (50건 x 2000 = 10만 건/월)
STATE_FILE = "_state.json"
RUNS_LOG = "_runs.jsonl"


class AlpacaAuthError(RuntimeError):
    """401/403 — 키가 없거나 잘못됐거나 플랜 권한 부족. 재시도하지 않는다."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class AlpacaRequestError(RuntimeError):
    """재시도 한도를 넘긴 오류, 또는 재시도 불가능한 4xx."""


def load_credentials(env_path: Path | None = None) -> tuple[str, str]:
    """환경변수에서 키를 읽는다. `.env`가 있으면 python-dotenv로 로드한다."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path) if env_path else load_dotenv()
    except ImportError:  # dotenv 없이 환경변수만 쓰는 경우
        pass
    key, secret = os.environ.get(KEY_ENV), os.environ.get(SECRET_ENV)
    if not key or not secret:
        raise RuntimeError(f"환경변수 {KEY_ENV}, {SECRET_ENV}가 설정되어 있지 않다 (.env.example 참고).")
    return key, secret


def load_universe(config_path: Path) -> list[str]:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(t).upper() for t in cfg["tickers"]]


@dataclass
class FetchStats:
    requests: int = 0
    ok: int = 0
    rate_limited_429: int = 0
    server_errors: int = 0
    connection_errors: int = 0
    retries: int = 0
    articles: int = 0
    chunks_done: int = 0
    chunks_skipped: int = 0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["elapsed_sec"] = round(time.time() - self.started_at, 1)
        return d


class AlpacaNewsClient:
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        min_interval: float = 0.4,
        max_retries: int = 8,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._session = session or requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self._sleep = sleep
        self._last_request = 0.0
        self.stats = FetchStats()

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        delay = min(self.backoff_base * (2**attempt), self.backoff_max)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        return delay

    def get_page(self, params: dict) -> dict:
        """한 페이지를 가져온다. 429/5xx/연결 오류는 백오프 후 재시도."""
        last_err = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats.requests += 1
            try:
                resp = self._session.get(NEWS_URL, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                self.stats.connection_errors += 1
                last_err = f"{type(e).__name__}"
                retry_after = None
            else:
                if resp.status_code == 200:
                    self.stats.ok += 1
                    return resp.json()
                if resp.status_code in (401, 403):
                    raise AlpacaAuthError(resp.status_code, resp.text[:500])
                if resp.status_code == 429:
                    self.stats.rate_limited_429 += 1
                elif resp.status_code >= 500:
                    self.stats.server_errors += 1
                else:
                    raise AlpacaRequestError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                last_err = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")

            if attempt == self.max_retries:
                break
            self.stats.retries += 1
            delay = self._backoff(attempt, retry_after)
            logger.warning("%s — %.1fs 후 재시도 (%d/%d)", last_err, delay, attempt + 1, self.max_retries)
            self._sleep(delay)
        raise AlpacaRequestError(f"재시도 한도 초과 ({last_err})")

    def iter_articles(self, symbols: str, start: str, end: str, *, sort: str = "asc") -> Iterator[dict]:
        """[start, end) 구간의 기사를 `next_page_token`이 없어질 때까지 순회한다."""
        params: dict = {
            "symbols": symbols,
            "start": start,
            "end": end,
            "limit": PAGE_LIMIT,
            "sort": sort,
            "include_content": "true",
        }
        seen_tokens: set[str] = set()
        for _ in range(MAX_PAGES_PER_CHUNK):
            data = self.get_page(params)
            yield from data.get("news") or []
            token = data.get("next_page_token")
            if not token:
                return
            if token in seen_tokens:
                raise AlpacaRequestError("next_page_token이 반복됨 — 페이지네이션 루프 의심")
            seen_tokens.add(token)
            params["page_token"] = token
        raise AlpacaRequestError(f"페이지 수가 {MAX_PAGES_PER_CHUNK}를 넘음 — 구간을 더 잘게 나눠야 함")


# ---------------------------------------------------------------- 구간/저장 ----


def month_chunks(start: date, end: date) -> list[tuple[str, str, str]]:
    """[start, end] 날짜를 월 단위 구간으로 쪼갠다. (month_key, start_rfc3339, end_rfc3339) 목록.

    구간은 [시작, 다음 달 1일 00:00Z) 형태이며, 첫/마지막 구간은 start/end로 잘린다.
    end 당일 전체를 포함하도록 end+1일 00:00Z까지로 잡는다.
    """
    chunks = []
    cur = date(start.year, start.month, 1)
    last_excl = date.fromordinal(end.toordinal() + 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        lo = max(cur, start)
        hi = min(nxt, last_excl)
        chunks.append((f"{cur.year}-{cur.month:02d}", f"{lo.isoformat()}T00:00:00Z", f"{hi.isoformat()}T00:00:00Z"))
        cur = nxt
    return chunks


def _read_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": {}}


def _write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)  # 원자적 교체 — 중간에 끊겨도 상태 파일이 깨지지 않는다


def _existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(str(json.loads(line)["id"]))
    return ids


def collect_symbol(
    client: AlpacaNewsClient,
    symbol: str,
    start: date,
    end: date,
    out_dir: Path,
    *,
    refetch_current: bool = True,
    progress: Callable[[str], None] = print,
) -> int:
    """한 종목의 [start, end] 기사를 월 단위로 수집해 저장한다. 새로 저장한 기사 수를 반환."""
    sym_dir = out_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    state_path = sym_dir / STATE_FILE
    state = _read_state(state_path)
    today = datetime.now(timezone.utc).date()
    total_new = 0

    for month, lo, hi in month_chunks(start, end):
        done = state["completed"].get(month)
        # 구간이 끝나기 전에 받은 달(fetched_through < 구간 끝)은 미완료로 보고 다시 받는다.
        partial = done is not None and done.get("fetched_through", "9999") < hi[:10]
        if done is not None and not (refetch_current and partial):
            client.stats.chunks_skipped += 1
            continue

        articles = list(client.iter_articles(symbol, lo, hi))
        year_file = sym_dir / f"{month[:4]}.jsonl"
        seen = _existing_ids(year_file)
        fresh = [a for a in articles if str(a["id"]) not in seen]
        # 한 번에 append → 중간 단절 시 상태가 갱신되지 않아 다음 실행에서 다시 받는다 (id로 중복 제거).
        with open(year_file, "a", encoding="utf-8") as f:
            for a in fresh:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        state["completed"][month] = {
            "start": lo,
            "end": hi,
            "api_articles": len(articles),
            "new_written": len(fresh),
            "fetched_through": min(hi[:10], today.isoformat()),
        }
        _write_state(state_path, state)
        client.stats.articles += len(fresh)
        client.stats.chunks_done += 1
        total_new += len(fresh)
        progress(f"[{symbol}] {month} api={len(articles)} new={len(fresh)} (누적 요청 {client.stats.requests}, 429={client.stats.rate_limited_429})")
    return total_new


def log_run(out_dir: Path, symbols: list[str], start: date, end: date, stats: FetchStats) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbols": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        **stats.as_dict(),
    }
    rec.pop("started_at", None)
    with open(out_dir / RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
