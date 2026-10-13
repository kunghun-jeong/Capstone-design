"""Massive 뉴스 API 수집 CLI

티커별 마지막 성공 조회 종료 시각부터 지정한 종료 시각까지 조회하고, ``next_url`` 페이지를 끝까지 따라갑니다.
받은 기사는 ``published_utc``의 UTC 연도를 기준으로 연도별 JSONL 파일에 저장합니다.

예)
  python scripts/fetch_massive_news.py --symbols NVDA,AMD --start 2023-01-01 --end 2024-01-01
  python scripts/fetch_massive_news.py --start 2016-06-22 --end 2026-10-01
  python scripts/fetch_massive_news.py --start 2026-10-12T00:00:00Z --end 2026-10-13T00:00:00Z

날짜만 입력하면 UTC 자정으로 해석합니다. 종료 시각은 미포함이며, 날짜 입력의 --end는 그 날짜
UTC 자정부터 제외됩니다. 시각은 Massive API 형식에 맞춰 YYYY-MM-DDTHH:MM:SSZ로 입력합니다.

API 요청은 최소 12초 간격으로 실행됩니다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "massive_news_ticker"
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.massive_news import (  # noqa: E402
    MassiveCredentialsError,
    MassiveUnauthorizedError,
    MassivePermissionError,
    MassiveNewsClient,
    MassivePaginationError,
    MassiveRequestError,
    MassiveResponseError,
    MassiveStorageError,
    MassiveConfigError,
    collect_symbol,
    load_api_key,
    load_universe,
    log_run,
)


def parse_datetime(value: str) -> datetime:
    """날짜는 UTC 자정으로, Z 형식 시각은 UTC 시각으로 변환한다."""
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError:
        parsed_date = None
    if parsed_date is not None:
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)

    try:
        parsed_datetime = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "YYYY-MM-DD 또는 UTC 시각 형식 YYYY-MM-DDTHH:MM:SSZ를 입력하세요."
        ) from exc
    return parsed_datetime.replace(tzinfo=timezone.utc)


def main() -> int:

    exit_code = 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", help="쉼표로 구분한 티커. 생략하면 --config의 tickers를 사용")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "universe.yaml")
    parser.add_argument(
        "--start",
        type=parse_datetime,
        default=datetime(2016, 6, 22, tzinfo=timezone.utc),
        help="초기 시작 경계: YYYY-MM-DD 또는 UTC 시각 YYYY-MM-DDTHH:MM:SSZ (기본값: 2016-06-22)",
    )
    parser.add_argument(
        "--end",
        type=parse_datetime,
        default=None,
        help="종료 미포함 경계: YYYY-MM-DD 또는 UTC 시각 YYYY-MM-DDTHH:MM:SSZ (기본값: 현재 UTC 시각의 1시간 전)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="티커별 연도 JSONL 저장 디렉터리 (기본값: data/raw/massive_news_ticker)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end_exclusive = args.end or (datetime.now(timezone.utc) - timedelta(hours=1))
    if args.start >= end_exclusive:
        parser.error("--start는 --end보다 빨라야 합니다.")

    try:
        symbols = (
            [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
            if args.symbols
            else load_universe(args.config)
        )
    except MassiveConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2

    if not symbols:
        parser.error("수집할 티커가 없습니다.")

    try:
        client = MassiveNewsClient(load_api_key())
    except MassiveCredentialsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        for symbol in symbols:
            collect_symbol(client, symbol, args.start, end_exclusive, args.out_dir)
    except MassiveUnauthorizedError as exc:
        print(f"인증 오류: {exc}", file=sys.stderr)
        return 2
    except MassivePermissionError as exc:
        print(f"권한 오류: {exc}", file=sys.stderr)
        return 2
    except MassiveConfigError as exc:
        print(f"설정 및 검증 에러: {exc}", file=sys.stderr)
        return 2
    except MassiveRequestError as exc:
        print(f"API 요청 오류: {exc}", file=sys.stderr)
        return 1
    except (MassiveResponseError, MassivePaginationError) as exc:
        print(f"데이터 응답 및 파싱 / 로직 오류: {exc}", file=sys.stderr)
        return 1
    except MassiveStorageError as exc:
        print(f"파일 시스템 및 저장소 오류: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            log_run(args.out_dir, symbols, args.start, end_exclusive, client.stats)
        except MassiveStorageError as exc:
            print(f"실행 로그 저장 오류: {exc}", file=sys.stderr)
            exit_code = 1
        print("통계:", client.stats.as_dict())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
