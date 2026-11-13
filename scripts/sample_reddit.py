"""Reddit(Arctic Shift) 시간당 표본 수집 CLI — 전체 수집이 너무 큰 구간(WSB 댓글 2020-10~2026-01 등)용.

매 UTC 시간마다 [h, h+1시간) 안의 무작위 시각(시드 고정)부터 asc로 한 페이지(최대 100건)를 받는다.
그 페이지가 걸친 시간(초)으로 그 시간대 전체 건수를 추정해, 각 글에 ``_sample_weight``(= 추정 건수 / 받은 건수)를 붙인다.
전처리는 이 가중치로 건수·비율·감성 평균을 계산한다(가중치가 없으면 1 = 전체 수집).

  python scripts/sample_reddit.py --subreddit wallstreetbets --kind comments --start 2020-10-01 --end 2026-02-01

출력: ``<out-dir>/<subreddit>/<kind>/<YYYY-MM>.jsonl`` (+ ``<YYYY-MM>.sample.csv``: 시간별 요청 시각·건수·추정치).
이미 표본이 있는 시간은 건너뛰므로 같은 명령으로 이어받는다. --workers는 서로 다른 월을 동시에 받는 스레드 수(최대 2).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.reddit_arctic_shift import PAGE_LIMIT, ArcticShiftClient  # noqa: E402

SEED = 20261114
HOUR = 3600


def hour_estimate(n: int, first_ts: int, last_ts: int, offset_ts: int, hour_end: int) -> float:
    """그 시간대 전체 건수 추정. 페이지가 시간 끝까지 못 갔으면(꽉 찬 페이지) 밀도 × 3600,
    시간 안에서 끝났으면(덜 찬 페이지) 받은 건수 + 앞부분(시간 시작~offset) 밀도 보정."""
    if n == 0:
        return 0.0
    if n >= PAGE_LIMIT:
        span = max(last_ts - offset_ts, 1)          # offset부터 마지막 글까지
        return n / span * HOUR
    # 덜 찬 페이지: offset~시간 끝 구간은 다 받았다. 시간 시작~offset 구간은 같은 밀도로 가정
    covered = max(hour_end - offset_ts, 1)
    return n * HOUR / covered


def month_hours(year: int, month: int, lo: datetime, hi: datetime) -> list[int]:
    start = max(datetime(year, month, 1, tzinfo=timezone.utc), lo)
    end = min(datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc), hi)
    h, out = start, []
    while h < end:
        out.append(int(h.timestamp()))
        h += timedelta(hours=1)
    return out


def sample_month(sub: str, kind: str, year: int, month: int, lo: datetime, hi: datetime, out_dir: Path, lock) -> dict:
    d = out_dir / sub / kind
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{year}-{month:02d}"
    index_path, data_path = d / f"{tag}.sample.csv", d / f"{tag}.jsonl"
    done = set()
    if index_path.exists():
        with index_path.open() as fh:
            done = {int(r["hour"]) for r in csv.DictReader(fh)}
    client = ArcticShiftClient()
    new_index = not index_path.exists()
    n_req = 0
    with index_path.open("a", newline="") as ifh, data_path.open("a", encoding="utf-8") as dfh:
        w = csv.writer(ifh)
        if new_index:
            w.writerow(["hour", "offset", "n", "first_ts", "last_ts", "est_hour", "weight"])
        for h in month_hours(year, month, lo, hi):
            if h in done:
                continue
            rng = random.Random(f"{SEED}:{sub}:{kind}:{h}")
            offset = h + rng.randrange(0, HOUR - 60)        # 마지막 1분 전까지
            page = client.get_page(kind, {"subreddit": sub, "after": offset - 1, "before": h + HOUR,
                                          "limit": PAGE_LIMIT, "sort": "asc"})
            n_req += 1
            n = len(page)
            first_ts = page[0]["created_utc"] if page else 0
            last_ts = page[-1]["created_utc"] if page else 0
            est = hour_estimate(n, first_ts, last_ts, offset, h + HOUR)
            weight = est / n if n else 0.0
            for rec in page:          # 데이터 먼저 쓰고 색인을 나중에 써야 끊겨도 색인에 없는 시간만 다시 받는다
                rec["_sample_weight"] = weight
                dfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dfh.flush()
            w.writerow([h, offset, n, first_ts, last_ts, round(est, 1), round(weight, 4)])
            ifh.flush()
    return {"month": tag, "requests": n_req}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subreddit", required=True)
    parser.add_argument("--kind", choices=["comments", "posts"], required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "reddit_sample")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers는 1~2")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    lo = datetime.combine(date.fromisoformat(args.start), datetime.min.time(), timezone.utc)
    hi = datetime.combine(date.fromisoformat(args.end), datetime.min.time(), timezone.utc)
    months, cur = [], lo.replace(day=1)
    while cur < hi:
        months.append((cur.year, cur.month))
        cur = cur.replace(year=cur.year + (cur.month == 12), month=cur.month % 12 + 1)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="sample") as pool:
        for r in pool.map(lambda ym: sample_month(args.subreddit, args.kind, *ym, lo, hi, args.out_dir, lock), months):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
