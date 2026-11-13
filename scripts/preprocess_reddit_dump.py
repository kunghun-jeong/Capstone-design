"""서브레딧 덤프(.zst) → 1단계 조각(parquet). 전처리 1단계(preprocess_reddit.py --stage 1)가 이 조각을 합친다.

  python scripts/preprocess_reddit_dump.py --dump-dir data/raw/reddit_dump/subreddits23 \\
      --start 2020-10-01 --end 2024-01-01 --out data/interim/reddit_parts --workers 6

덤프 파일은 ``<subreddit>_<comments|submissions>.zst`` 이름이어야 한다(Watchful1 top-40k 판 그대로).
파일마다 [start, end)를 UTC 연도로 나눠 작업 하나씩 만들고, 결과를
``<out>/dump_<subreddit>_<kind>_<YYYY>.rows.parquet`` / ``.activity.parquet``로 쓴다. 이미 있는 조각은 건너뛴다.
대상 서브레딧·종목·노이즈 규칙은 configs/reddit_preprocessing.yaml을 따른다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.reddit_preprocessing.config import load_config  # noqa: E402
from capstone.data.reddit_preprocessing.dump import dump_kind, process_dump, year_ranges  # noqa: E402


def _utc(value: str) -> datetime:
    d = date.fromisoformat(value)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _job(path: str, config: str, lo: str, hi: str, out: str) -> dict:
    t = time.time()
    cfg = load_config(Path(config))
    rows, act, n = process_dump(Path(path), cfg, datetime.fromisoformat(lo), datetime.fromisoformat(hi))
    sub, kind = dump_kind(Path(path))
    name = f"dump_{sub}_{kind}_{datetime.fromisoformat(lo).year}"
    tmp = Path(out) / f".{name}"
    rows.to_parquet(f"{tmp}.rows.parquet", index=False)
    act.to_parquet(f"{tmp}.activity.parquet", index=False)
    for suffix in (".rows.parquet", ".activity.parquet"):   # 둘 다 쓴 뒤에만 보이게(중간에 끊겨도 반쪽 조각이 안 남는다)
        Path(f"{tmp}{suffix}").rename(Path(out) / f"{name}{suffix}")
    return {"part": name, "records": n, "mention_rows": len(rows), "sec": round(time.time() - t, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, 포함)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, 미포함)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "interim" / "reddit_parts")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "reddit_preprocessing.yaml")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    start, end = _utc(args.start), _utc(args.end)
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for path in sorted(args.dump_dir.rglob("*.zst")):
        sub, kind = dump_kind(path)
        for lo, hi in year_ranges(start, end):
            if (args.out / f"dump_{sub}_{kind}_{lo.year}.rows.parquet").exists():
                continue
            jobs.append((str(path), str(args.config), lo.isoformat(), hi.isoformat(), str(args.out)))
    # 큰 파일(WSB 댓글)부터 시작해 전체 시간을 줄인다
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)
    print(f"작업 {len(jobs)}개, workers={args.workers}", flush=True)
    log = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_job, *j) for j in jobs]):
            r = fut.result()
            log.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    manifest = args.out / "_dump_manifest.json"
    prev = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    prev.setdefault("runs", []).append({
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dump_dir": str(args.dump_dir), "start": args.start, "end": args.end,
        "files": {p.name: p.stat().st_size for p in sorted(args.dump_dir.rglob("*.zst"))},
        "parts": log,
    })
    manifest.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
