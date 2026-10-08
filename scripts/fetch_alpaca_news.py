#!/usr/bin/env python3
"""Alpaca 뉴스 수집 CLI.

예)
  python scripts/fetch_alpaca_news.py --symbols NVDA --start 2016-01-01 --end 2016-01-31 \
      --out-dir data/raw/alpaca_news_smoke
  python scripts/fetch_alpaca_news.py --start 2015-01-01          # configs/universe.yaml 전 종목, 오늘까지
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.alpaca_news import (  # noqa: E402
    AlpacaAuthError,
    AlpacaNewsClient,
    collect_symbol,
    load_credentials,
    load_universe,
    log_run,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="쉼표 구분 티커. 생략하면 --config의 tickers 전체")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "universe.yaml")
    p.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1), help="YYYY-MM-DD (기본 2015-01-01)")
    p.add_argument("--end", type=date.fromisoformat, default=None, help="YYYY-MM-DD (기본 오늘, UTC)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "alpaca_news")
    p.add_argument("--min-interval", type=float, default=0.4, help="요청 간 최소 간격(초)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end = args.end or datetime.now(timezone.utc).date()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else load_universe(args.config)

    key, secret = load_credentials(REPO_ROOT / ".env")
    client = AlpacaNewsClient(key, secret, min_interval=args.min_interval)

    try:
        for sym in symbols:
            collect_symbol(client, sym, args.start, end, args.out_dir)
    except AlpacaAuthError as e:
        print(f"\n[인증/권한 오류] {e}", file=sys.stderr)
        return 2
    finally:
        log_run(args.out_dir, symbols, args.start, end, client.stats)
        print("통계:", {k: v for k, v in client.stats.as_dict().items() if k != "started_at"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
