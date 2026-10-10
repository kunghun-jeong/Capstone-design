#!/usr/bin/env python3
"""Massive 뉴스 API 수집 CLI.

예)
  python scripts/fetch_massive_news.py --symbols NVDA,AMD --start 2023-01-01
  python scripts/fetch_massive_news.py --config configs/universe.yaml --start 2024-01-01 --end 2024-12-31

MASSIVE_API_KEY 환경변수만 사용합니다. API 요청은 최소 12초 간격으로 실행됩니다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.massive_news import (  # noqa: E402
    MassiveAuthError,
    MassiveCredentialsError,
    MassiveNewsClient,
    MassivePaginationError,
    MassiveRequestError,
    MassiveResponseError,
    MassiveStorageError,
    collect_symbol,
    load_api_key,
    load_universe,
    log_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", help="쉼표로 구분한 티커. 생략하면 --config의 tickers를 사용")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "universe.yaml")
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="종료일 포함 (기본값: 오늘 UTC)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "massive_news",
        help="티커별 연도 JSONL 저장 디렉터리",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end = args.end or datetime.now(timezone.utc).date()
    if args.start > end:
        parser.error("--start는 --end보다 늦을 수 없습니다.")
    symbols = (
        [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        if args.symbols
        else load_universe(args.config)
    )
    if not symbols:
        parser.error("수집할 티커가 없습니다.")

    try:
        client = MassiveNewsClient(load_api_key())
    except MassiveCredentialsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        for symbol in symbols:
            collect_symbol(client, symbol, args.start, end, args.out_dir)
    except MassiveAuthError as exc:
        print(f"인증/권한 오류: {exc}", file=sys.stderr)
        return 2
    except (MassiveRequestError, MassivePaginationError, MassiveResponseError, MassiveStorageError) as exc:
        print(f"API 요청 오류: {exc}", file=sys.stderr)
        return 1
    finally:
        log_run(args.out_dir, symbols, args.start, end, client.stats)
        print("통계:", client.stats.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
