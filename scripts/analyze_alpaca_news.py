#!/usr/bin/env python3
"""수집된 Alpaca 뉴스 JSONL을 집계해 리포트용 통계를 출력한다 (데이터는 읽기만 한다)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TAG = re.compile(r"<[a-zA-Z/][^>]*>")


def load(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*/*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                a = json.loads(line)
                rows.append(
                    {
                        "query_symbol": f.parent.name,
                        "id": a["id"],
                        "created_at": a.get("created_at"),
                        "updated_at": a.get("updated_at"),
                        "n_symbols": len(a.get("symbols") or []),
                        "has_content": bool((a.get("content") or "").strip()),
                        "content_len": len(a.get("content") or ""),
                        "has_html": bool(TAG.search(a.get("content") or "")),
                        "has_summary": bool((a.get("summary") or "").strip()),
                        "source": a.get("source") or "",
                    }
                )
    df = pd.DataFrame(rows)
    for c in ("created_at", "updated_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "raw" / "alpaca_news")
    root = ap.parse_args().root
    df = load(root)
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 200)

    print("총 행(종목별 중복 포함):", len(df), "| 고유 id:", df["id"].nunique())
    print("\n== 종목 x 연도 (created_at 연도) ==")
    df["cy"] = df["created_at"].dt.year
    print(pd.crosstab(df["query_symbol"], df["cy"], margins=True))
    print("\n== 종목 x 연도 (updated_at 연도, API 필터 기준) ==")
    print(pd.crosstab(df["query_symbol"], df["updated_at"].dt.year, margins=True))
    print("\n== 종목별 첫/마지막 created_at ==")
    print(df.groupby("query_symbol")["created_at"].agg(["min", "max"]))

    u = df.drop_duplicates("id")
    print("\n== 고유 기사 기준 ==")
    print("content 포함률: %.1f%%  HTML 포함률(content 중): %.1f%%  summary 포함률: %.1f%%" % (
        100 * u.has_content.mean(), 100 * u[u.has_content].has_html.mean(), 100 * u.has_summary.mean()))
    print("여러 종목 기사(n_symbols>=2): %.1f%%  | n_symbols 중앙값 %s, 최대 %s" % (
        100 * (u.n_symbols >= 2).mean(), u.n_symbols.median(), u.n_symbols.max()))
    print("\n연도별 content/summary 포함률 (고유 기사):")
    print(u.groupby("cy")[["has_content", "has_summary"]].mean().round(3))
    print("\nsource 분포:", u["source"].value_counts().to_dict())

    lag = (u["updated_at"] - u["created_at"]).dt.total_seconds()
    print("\n== updated_at - created_at ==")
    print("updated != created 비율: %.1f%%" % (100 * (lag != 0).mean()))
    print("음수(updated<created):", int((lag < 0).sum()))
    print(lag[lag > 0].describe(percentiles=[0.5, 0.9, 0.99]).round(0))
    print("1일 이상 차이 비율: %.1f%%, 30일 이상: %.1f%%" % (100 * (lag > 86400).mean(), 100 * (lag > 30 * 86400).mean()))
    print("\n종목별 id 중복(같은 종목 쿼리 내 중복):", int(df.duplicated(["query_symbol", "id"]).sum()))


if __name__ == "__main__":
    main()
