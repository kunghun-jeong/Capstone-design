"""Reddit(Arctic Shift) 게시글·댓글 수집 CLI

서브레딧·종류(posts/comments)·UTC 월 단위로 [start, end) 구간을 받아
``<out-dir>/<subreddit>/<kind>/<YYYY-MM>.jsonl``에 저장합니다. 중간에 끊겨도 다시 실행하면
월별 상태 파일(``<YYYY-MM>.state.json``)의 커서부터 이어받습니다. API 키는 필요 없습니다.

예)
  python scripts/fetch_reddit.py --subreddits stocks --kinds posts --start 2026-09-01 --end 2026-09-02
  python scripts/fetch_reddit.py --start 2026-04-24 --end 2026-10-24      # configs/reddit_subreddits.yaml 전체
  python scripts/fetch_reddit.py --subreddits wallstreetbets --kinds comments --workers 2

날짜만 입력하면 UTC 자정으로 해석합니다. 종료 시각은 미포함입니다.
--end를 생략하면 현재 UTC 시각의 36시간 전(정시로 내림)입니다 — 게시 후 약 36시간 동안은 score가
자리표시 값(1 또는 0)이기 때문입니다. --start를 생략하면 --end의 183일 전 UTC 자정입니다.
--workers는 서로 다른 월을 동시에 받는 스레드 수이며, 무료 서비스 부담을 고려해 최대 2입니다.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "reddit"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reddit_subreddits.yaml"
MAX_WORKERS = 2
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.reddit_arctic_shift import (  # noqa: E402
    KINDS,
    MIN_REQUEST_INTERVAL,
    ArcticShiftClient,
    FetchStats,
    RedditConfigError,
    RedditPaginationError,
    RedditRequestError,
    RedditResponseError,
    RedditStorageError,
    Target,
    _to_ts,
    collect_unit,
    load_targets,
    log_run,
    month_units,
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
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "YYYY-MM-DD 또는 UTC 시각 형식 YYYY-MM-DDTHH:MM:SSZ를 입력하세요."
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def default_end(now: datetime) -> datetime:
    return (now - timedelta(hours=36)).replace(minute=0, second=0, microsecond=0)


def default_start(end: datetime) -> datetime:
    start = end - timedelta(days=183)
    return datetime(start.year, start.month, start.day, tzinfo=timezone.utc)


def select_targets(targets: list[Target], subreddits: str | None, kinds: str | None) -> list[Target]:
    """설정의 Target 목록을 --subreddits/--kinds로 거른다. 설정에 없는 서브레딧도 지정할 수 있다."""
    kind_filter = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
    if kind_filter and any(k not in KINDS for k in kind_filter):
        raise RedditConfigError(f"--kinds는 {KINDS} 중에서 골라야 합니다: {kinds}")
    if subreddits:
        configured: dict[str, list[str]] = {}
        for t in targets:
            configured.setdefault(t.subreddit.lower(), []).append(t.kind)
        selected = []
        for name in (s.strip() for s in subreddits.split(",") if s.strip()):
            # 설정에 있는 서브레딧은 설정의 kinds를, 없는 서브레딧은 posts·comments 전부를 기본으로 한다.
            kinds_for_name = configured.get(name.lower(), list(KINDS))
            selected.extend(Target(name, k) for k in kinds_for_name if not kind_filter or k in kind_filter)
        return selected
    return [t for t in targets if not kind_filter or t.kind in kind_filter]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subreddits", help="쉼표로 구분한 서브레딧 이름. 생략하면 --config 전체")
    parser.add_argument("--kinds", help="posts, comments 중 쉼표로 구분 (생략하면 설정값)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", type=parse_datetime, default=None, help="시작 경계 (기본: --end의 183일 전)")
    parser.add_argument("--end", type=parse_datetime, default=None, help="종료 미포함 경계 (기본: 현재 36시간 전)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-interval", type=float, default=MIN_REQUEST_INTERVAL, help="스레드당 요청 간 최소 간격(초)")
    parser.add_argument("--workers", type=int, default=1, help=f"동시에 받을 월 단위 수 (1~{MAX_WORKERS})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers는 1~{MAX_WORKERS} 사이여야 합니다.")
    if args.min_interval < MIN_REQUEST_INTERVAL:
        parser.error(f"--min-interval은 {MIN_REQUEST_INTERVAL}초 이상이어야 합니다.")
    end_exclusive = args.end or default_end(datetime.now(timezone.utc))
    start = args.start or default_start(end_exclusive)
    if start >= end_exclusive:
        parser.error("--start는 --end보다 빨라야 합니다.")

    try:
        targets = select_targets(load_targets(args.config), args.subreddits, args.kinds)
    except RedditConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    if not targets:
        parser.error("수집할 서브레딧이 없습니다.")

    units = [(t, m, a, b) for t in targets for (m, a, b) in month_units(_to_ts(start), _to_ts(end_exclusive))]
    stats = FetchStats()
    stop = threading.Event()
    local = threading.local()

    def run(unit: tuple[Target, str, int, int]) -> int:
        if stop.is_set():
            return 0
        if not hasattr(local, "client"):
            local.client = ArcticShiftClient(min_interval=args.min_interval, stats=stats)
        target, month, unit_start, unit_end = unit
        return collect_unit(local.client, target, month, unit_start, unit_end, args.out_dir, should_stop=stop.is_set)

    print(f"대상 {len(targets)}개 × 월 {len(units) // len(targets)}개 = 단위 {len(units)}개, workers={args.workers}", flush=True)
    exit_code = 0
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="reddit")
    try:
        futures = [executor.submit(run, unit) for unit in units]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        failed = [f for f in done if f.exception() is not None]
        if failed:
            stop.set()
            wait(futures)
            raise failed[0].exception()
    except KeyboardInterrupt:
        stop.set()
        print("중단 요청 — 진행 중인 페이지까지 저장하고 멈춥니다.", file=sys.stderr)
        exit_code = 130
    except RedditConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        exit_code = 2
    except RedditRequestError as exc:
        print(f"API 요청 오류: {exc}", file=sys.stderr)
        exit_code = 1
    except (RedditResponseError, RedditPaginationError) as exc:
        print(f"응답/페이지네이션 오류: {exc}", file=sys.stderr)
        exit_code = 1
    except RedditStorageError as exc:
        print(f"저장소 오류: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        try:
            log_run(args.out_dir, targets, start, end_exclusive, stats)
        except RedditStorageError as exc:
            print(f"실행 로그 저장 오류: {exc}", file=sys.stderr)
            exit_code = exit_code or 1
        print("통계:", stats.as_dict(), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
