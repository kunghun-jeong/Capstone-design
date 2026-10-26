"""수집한 Reddit(Arctic Shift) 데이터 집계 — reports/reddit_arctic_shift_report.md의 숫자를 재현한다.

예)
  python scripts/analyze_reddit.py --start 2026-04-24 --end 2026-08-01
  python scripts/analyze_reddit.py --start 2026-04-24 --end 2026-10-24 --json out.json

출력: 서브레딧·종류별 수집량/품질, 게시 시각의 미국 장 시간대 분포, 대상 종목 언급량(일별 커버리지·집중도).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
BOTS = {"AutoModerator", "VisualMod"}
PLACEHOLDER_TEXT = {"[removed]", "[deleted]"}
# 2026년 NYSE 휴장일 중 이 스크립트가 다룰 수 있는 범위 (장 시간대 분류용)
NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
}

# (대문자 티커·$티커, 회사명[대소문자 무시]) — 회사명은 일반 단어와 겹치면 오탐이 생긴다(Amazon, Google 등).
TICKERS: dict[str, tuple[str, str | None]] = {
    "NVDA": (r"NVDA", r"nvidia"),
    "AMD": (r"AMD", r"advanced micro devices"),
    "MU": (r"MU", r"micron"),
    "TSM": (r"TSMC?", r"taiwan semi"),
    "AVGO": (r"AVGO", r"broadcom"),
    "SNDK": (r"SNDK", r"sandisk"),
    "MRVL": (r"MRVL", r"marvell"),
    "MSFT": (r"MSFT", r"microsoft"),
    "GOOG": (r"GOOGL?", r"alphabet|google"),
    "AMZN": (r"AMZN", r"amazon"),
    "META": (r"META", r"meta platforms|facebook"),
}
STRICT = {t: re.compile(rf"(?<![A-Za-z0-9])\$?(?:{p})(?![A-Za-z0-9])") for t, (p, _) in TICKERS.items()}
NAMES = {t: re.compile(rf"\b(?:{n})\b", re.I) for t, (_, n) in TICKERS.items() if n}


def session_of(ts: int) -> str:
    t = datetime.fromtimestamp(ts, ET)
    if t.weekday() >= 5 or t.date() in NYSE_HOLIDAYS_2026:
        return "휴장일"
    minutes = t.hour * 60 + t.minute
    if minutes < 4 * 60 or minutes >= 20 * 60:
        return "야간(20:00~04:00)"
    if minutes < 9 * 60 + 30:
        return "프리마켓(04:00~09:30)"
    if minutes < 16 * 60:
        return "정규장(09:30~16:00)"
    return "애프터마켓(16:00~20:00)"


def text_of(row: dict, kind: str) -> str:
    if kind == "posts":
        return f"{row.get('title') or ''}\n{row.get('selftext') or ''}"
    return row.get("body") or ""


def analyze(root: Path, start: datetime, end: datetime) -> dict:
    lo, hi = int(start.timestamp()), int(end.timestamp())
    n_days = (end - start).days
    sources: dict[str, dict] = {}
    sessions: Counter = Counter()
    strict_daily: dict[str, Counter] = defaultdict(Counter)
    loose_daily: dict[str, Counter] = defaultdict(Counter)
    by_source: dict[str, Counter] = defaultdict(Counter)

    for kind_dir in sorted(root.glob("*/*")):
        if not kind_dir.is_dir() or kind_dir.parent.name.startswith("_"):
            continue
        sub, kind = kind_dir.parent.name, kind_dir.name
        key = f"{sub}/{kind}"
        c = Counter()
        lags, scores, lengths = [], [], []
        for path in sorted(kind_dir.glob("*.jsonl")):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    created = int(row["created_utc"])
                    if not lo <= created < hi:
                        continue
                    c["rows"] += 1
                    meta = row.get("_meta") or {}
                    text = text_of(row, kind)
                    body = (row.get("selftext") if kind == "posts" else row.get("body")) or ""
                    deleted_author = row.get("author") == "[deleted]"
                    removed_text = body.strip() in PLACEHOLDER_TEXT or (row.get("title") or "").startswith("[ Removed")
                    removed_flag = bool(row.get("removed_by_category") or meta.get("removal_type"))
                    bot = row.get("author") in BOTS
                    c["deleted_author"] += deleted_author
                    c["removed_text"] += removed_text
                    c["removed_flag"] += removed_flag
                    c["bot"] += bot
                    c["refreshed"] += "retrieved_2nd_on" in meta
                    if isinstance(row.get("retrieved_on"), (int, float)):
                        lags.append(row["retrieved_on"] - created)
                    scores.append(row.get("score") or 0)
                    if kind == "posts" and removed_text:
                        # 게시글 본문만 지워지고 제목은 남는 경우가 많아 제목이 있으면 분석에 쓴다.
                        text = "" if (row.get("title") or "").startswith("[ Removed") else row.get("title") or ""
                    usable = bool(text.strip()) and not bot and not (kind == "comments" and removed_text)
                    if not usable:
                        continue
                    c["usable"] += 1
                    lengths.append(len(text))
                    sessions[session_of(created)] += 1
                    day = datetime.fromtimestamp(created, timezone.utc).date().isoformat()
                    for ticker, pattern in STRICT.items():
                        hit_strict = bool(pattern.search(text))
                        hit_name = ticker in NAMES and bool(NAMES[ticker].search(text))
                        if hit_strict:
                            strict_daily[ticker][day] += 1
                        if hit_strict or hit_name:
                            loose_daily[ticker][day] += 1
                            by_source[ticker][key] += 1
        if not c["rows"]:
            continue
        lags.sort()
        sources[key] = {
            "rows": c["rows"],
            "per_day": round(c["rows"] / n_days, 1),
            "usable": c["usable"],
            "deleted_author_pct": round(100 * c["deleted_author"] / c["rows"], 1),
            "removed_text_pct": round(100 * c["removed_text"] / c["rows"], 1),
            "removed_flag_pct": round(100 * c["removed_flag"] / c["rows"], 1),
            "bot_pct": round(100 * c["bot"] / c["rows"], 1),
            "refreshed_pct": round(100 * c["refreshed"] / c["rows"], 1),
            "score_le1_pct": round(100 * sum(s <= 1 for s in scores) / len(scores), 1),
            "median_score": statistics.median(scores),
            "capture_lag_median_s": lags[len(lags) // 2] if lags else None,
            "capture_lag_p99_s": lags[int(len(lags) * 0.99)] if lags else None,
            "median_len": statistics.median(lengths) if lengths else None,
        }

    tickers = {}
    for ticker in TICKERS:
        loose, strict = loose_daily[ticker], strict_daily[ticker]
        total = sum(loose.values())
        daily = [loose.get((start.date() + timedelta(d)).isoformat(), 0) for d in range(n_days)]
        monthly: Counter = Counter()
        for day, count in strict.items():
            monthly[day[:7]] += count
        tickers[ticker] = {
            "monthly_strict": dict(sorted(monthly.items())),
            "strict_total": sum(strict.values()),
            "loose_total": total,
            "days_with_mention": sum(v > 0 for v in daily),
            "n_days": n_days,
            "median_per_day": statistics.median(daily),
            "top3_day_share_pct": round(100 * sum(sorted(daily)[-3:]) / total, 1) if total else None,
            "wsb_comment_share_pct": round(100 * by_source[ticker]["wallstreetbets/comments"] / total, 1) if total else None,
            "top_sources": by_source[ticker].most_common(3),
        }
    total_sessions = sum(sessions.values())
    return {
        "window": [start.date().isoformat(), end.date().isoformat()],
        "sources": sources,
        "sessions_pct": {k: round(100 * v / total_sessions, 1) for k, v in sessions.most_common()},
        "tickers": tickers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "raw" / "reddit")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="미포함")
    parser.add_argument("--json", type=Path, help="결과를 JSON으로도 저장")
    args = parser.parse_args()
    start = datetime(args.start.year, args.start.month, args.start.day, tzinfo=timezone.utc)
    end = datetime(args.end.year, args.end.month, args.end.day, tzinfo=timezone.utc)
    result = analyze(args.root, start, end)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
