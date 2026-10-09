#!/usr/bin/env python3
"""회사 IR 보도자료 RSS 수집 CLI — configs/ir_press_release_feeds.yaml의 6종목을
data/raw/ir_press_releases/에 저장한다.

예)
  python scripts/fetch_ir_press_releases.py --user-agent "Your-Name your-contact"
  python scripts/fetch_ir_press_releases.py --summarize-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.ir_press_releases import (  # noqa: E402
    collect,
    load_feeds,
    print_summary,
    summarize,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "ir_press_release_feeds.yaml")
    p.add_argument("--user-agent", default="", help="요청에 쓸 User-Agent (이름 + 연락 방법 권장)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ir_press_releases")
    p.add_argument("--max-pages", type=int, default=None, help="종목당 이 페이지까지만 받고 멈춤 (맛보기용)")
    p.add_argument("--sleep-sec", type=float, default=0.5)
    p.add_argument("--summarize-only", action="store_true", help="수집 없이 저장된 결과 요약만 출력")
    args = p.parse_args()

    out_path = args.out_dir / "press_releases.csv"

    if not args.summarize_only:
        feeds = load_feeds(args.config)
        ua = args.user_agent or "Capstone-design (SKKU student project) contact-via-github-issues"
        print(f"대상: {[f.ticker for f in feeds]}\n")
        stats = collect(feeds, out_path, user_agent=ua, max_pages=args.max_pages, sleep_sec=args.sleep_sec)
        print(f"\n저장: {out_path}")
        print("통계:", stats.as_dict())
        if stats.errors:
            print("\n[오류 발생한 종목]")
            for e in stats.errors:
                print(" -", e)

    if not out_path.exists():
        print(f"결과 파일이 없음: {out_path}", file=sys.stderr)
        return 1
    print_summary(summarize(pd.read_csv(out_path, dtype=str).fillna("")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
