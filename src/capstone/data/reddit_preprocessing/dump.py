"""서브레딧 덤프(``<subreddit>_<comments|submissions>.zst``) → 1단계 산출물(언급 행 + 귀속일별 활동량).

덤프는 Watchful1의 top-40k 서브레딧 판(Pushshift/Arctic Shift, 2005-06 ~ 2023-12)으로, 서브레딧·종류별 파일 하나에
수년치 NDJSON이 zstd로 압축돼 있다. 풀어서 저장하면 수십~수백 GB라, 스트림으로 읽으며 [start, end) 구간만
묶음(chunk) 단위로 ``process_records``에 넘긴다. 결과는 API 수집분의 1단계 결과와 같은 스키마다.

API 수집분과 다른 점(리포트에 적을 것):
 - 옛 레코드는 ``created_utc``가 문자열이다 → int로 바꾼다.
 - ``name``(t1_/t3_ 접두) 이 없는 레코드가 있다 → 종류 접두사 + id로 채운다.
 - ``_meta``(첫 수집 시각·2차 수집)가 없다 → ``has_2nd_retrieval``은 False, ``is_edited``는 ``edited``만 본다.
 - ``removed_by_category``는 덤프 수집 시점 값이다. API(게시 후 약 30초)와 수집 시점이 달라 제거 비율이 다를 수 있다.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import zstandard

from .config import Config
from .normalize import ROW_COLUMNS, TickerMatcher, process_records

_TS = re.compile(rb'"created_utc"\s*:\s*"?(\d+)')
_PREFIX = {"comments": "t1_", "posts": "t3_"}
CHUNK = 200_000


def dump_kind(path: Path) -> tuple[str, str]:
    """``wallstreetbets_comments.zst`` → ("wallstreetbets", "comments"). submissions는 API 쪽 이름인 posts로."""
    stem = Path(path).name.removesuffix(".zst")
    sub, _, kind = stem.rpartition("_")
    if kind not in ("comments", "submissions") or not sub:
        raise ValueError(f"덤프 파일 이름은 <subreddit>_<comments|submissions>.zst 여야 합니다: {path}")
    return sub, "posts" if kind == "submissions" else "comments"


def iter_records(path: Path, start: datetime, end: datetime, kind: str):
    """[start, end) 구간 레코드만 dict로. 구간 밖 줄은 JSON을 풀지 않고 시각만 정규식으로 보고 건너뛴다."""
    lo, hi = int(start.timestamp()), int(end.timestamp())
    with Path(path).open("rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        for line in io.BufferedReader(reader, buffer_size=1 << 20):
            m = _TS.search(line)
            if not m or not lo <= int(m.group(1)) < hi:
                continue
            rec = json.loads(line)
            rec["created_utc"] = int(rec["created_utc"])
            if not rec.get("name") and rec.get("id"):
                rec["name"] = _PREFIX[kind] + str(rec["id"])
            yield rec


def process_dump(path: Path, cfg: Config, start: datetime, end: datetime,
                 chunk: int = CHUNK) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """덤프 1개(의 [start, end) 구간) → (언급 행, 활동량, 읽은 레코드 수)."""
    sub, kind = dump_kind(path)
    matcher = TickerMatcher(cfg.tickers, cfg.ticker_patterns, cfg.aliases, cfg.other_tickers)
    rows, acts, batch, n = [], [], [], 0

    def flush():
        r, a = process_records(batch, sub, kind, cfg, matcher)
        if not r.empty:
            rows.append(r)
        if not a.empty:
            acts.append(a)
        batch.clear()

    for rec in iter_records(path, start, end, kind):
        batch.append(rec)
        n += 1
        if len(batch) >= chunk:
            flush()
    if batch:
        flush()
    out_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ROW_COLUMNS)
    out_act = (pd.concat(acts, ignore_index=True).groupby(["subreddit", "kind", "day"], as_index=False).sum()
               if acts else pd.DataFrame())
    return out_rows, out_act, n


def year_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """[start, end)를 UTC 연도 경계로 자른다 — 큰 덤프(WSB 댓글)를 연도별 작업으로 나눠 병렬 처리하려고."""
    out, cur = [], start
    while cur < end:
        nxt = min(datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc), end)
        out.append((cur, nxt))
        cur = nxt
    return out
