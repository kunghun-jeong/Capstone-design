#!/usr/bin/env python3
"""FNSPID 뉴스 수집 CLI — 대상 종목 기사만 걸러 data/raw/fnspid_news/에 저장한다.

예)
  python scripts/fetch_fnspid_news.py --max-chunks 5          # 맛보기 (앞 100만 행)
  python scripts/fetch_fnspid_news.py --file nasdaq           # 본문 포함 23GB 파일 전체 (15~20분)
  python scripts/fetch_fnspid_news.py --file all --symbols NVDA,AMD
  python scripts/fetch_fnspid_news.py --summarize-only       # 수집 없이 저장된 결과 요약만

종목은 --symbols > configs/universe.yaml > 코드 기본값 순으로 정한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.fnspid_news import (  # noqa: E402
    DEFAULT_CHUNK_ROWS,
    FILES,
    collect,
    load_tickers,
    print_summary,
    source_url,
    summarize,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", choices=sorted(FILES), default="nasdaq", help="nasdaq(본문 포함, 기본) 또는 all(제목 전용)")
    p.add_argument("--source", help="URL 대신 읽을 로컬 CSV 경로 (미리 내려받은 경우)")
    p.add_argument("--symbols", help="쉼표 구분 티커. 생략하면 --config, 그것도 없으면 코드 기본값")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "universe.yaml")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "fnspid_news")
    p.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    p.add_argument("--max-chunks", type=int, default=None, help="이만큼만 읽고 멈춤 (맛보기용)")
    p.add_argument("--summarize-only", action="store_true", help="수집 없이 저장된 결과의 요약만 출력")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else load_tickers(args.config)
    out_path = args.out_dir / f"{args.file}_filtered.csv"

    if not args.summarize_only:
        source = args.source or source_url(args.file)
        print(f"읽는 중: {source}\n대상 종목: {symbols}\n")
        stats = collect(source, symbols, out_path, chunk_rows=args.chunk_rows, max_chunks=args.max_chunks)
        if stats.stopped_early:
            print(f"\n--max-chunks {args.max_chunks}에서 멈춤 (전체를 받으려면 옵션 없이 실행)")
        print(f"\n저장: {out_path}")
        print("통계:", stats.as_dict())

    if not out_path.exists():
        print(f"결과 파일이 없음: {out_path}", file=sys.stderr)
        return 1
    print_summary(summarize(pd.read_csv(out_path, dtype=str)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
