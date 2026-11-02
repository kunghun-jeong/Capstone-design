"""Alpaca / Massive 원본 JSONL → 공통 스키마 DataFrame.

공통 컬럼: article_id, source, published_utc, title, summary, body, url, publisher, tickers(list), n_tags.
결측은 빈 문자열이 아니라 None으로 통일한다. 원본 필드는 지우지 않고 raw_extra(JSON)에 보존한다.
소스 안에서는 원본 id로 중복을 제거한다 (같은 기사가 종목별 폴더에 반복 저장되므로) — 태그는 합집합.
"""
from __future__ import annotations

import html
import json
import re
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# 일부 본문이 XML처럼 생겨 나는 경고. 텍스트만 뽑는 용도라 무시해도 된다.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

COLUMNS = [
    "article_id", "source", "published_utc", "title", "summary", "body",
    "url", "publisher", "tickers", "n_tags", "raw_extra",
]
_WS = re.compile(r"\s+")


def clean_text(value) -> str | None:
    """HTML 태그/엔티티 제거 + 공백 정규화. 빈 결과는 None."""
    if value is None:
        return None
    text = str(value)
    if "<" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = _WS.sub(" ", html.unescape(text)).strip()
    return text or None


def read_jsonl_tree(root: Path) -> list[dict]:
    """``<root>/<TICKER>/<YYYY>.jsonl`` 전부를 읽는다 (``_runs.jsonl`` 등 밑줄 파일 제외)."""
    records: list[dict] = []
    for path in sorted(Path(root).glob("*/[0-9][0-9][0-9][0-9].jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _alpaca_row(a: dict) -> dict:
    tags = [s for s in (a.get("symbols") or []) if isinstance(s, str)]
    return {
        "article_id": f"alpaca:{a['id']}",
        "source": "alpaca",
        # 시점 정합에는 created_at (updated_at은 수정 시각이라 늦을 수 있다 — alpaca 리포트 7절)
        "published_utc": a["created_at"],
        "title": clean_text(a.get("headline")),
        "summary": clean_text(a.get("summary")),
        "body": clean_text(a.get("content")),
        "url": a.get("url") or None,
        "publisher": a.get("source") or None,
        "tickers": tags,
        "raw_extra": {"updated_at": a.get("updated_at"), "author": a.get("author")},
    }


def _massive_row(a: dict) -> dict:
    tags = [s for s in (a.get("tickers") or []) if isinstance(s, str)]
    publisher = a.get("publisher") or {}
    return {
        "article_id": f"massive:{a['id']}",
        "source": "massive",
        "published_utc": a["published_utc"],
        "title": clean_text(a.get("title")),
        "summary": clean_text(a.get("description")),   # Massive에는 본문이 없다
        "body": None,
        "url": a.get("article_url") or None,
        "publisher": publisher.get("name") if isinstance(publisher, dict) else None,
        "tickers": tags,
        "raw_extra": {"keywords": a.get("keywords"), "insights": a.get("insights"), "author": a.get("author")},
    }


_BUILDERS = {"alpaca": _alpaca_row, "massive": _massive_row}


def normalize(records: list[dict], source: str) -> pd.DataFrame:
    if source not in _BUILDERS:
        raise ValueError(f"알 수 없는 source: {source}")
    merged: dict[str, dict] = {}
    for rec in records:
        row = _BUILDERS[source](rec)
        prev = merged.get(row["article_id"])
        if prev is None:
            merged[row["article_id"]] = row
        else:  # 종목별 폴더에 반복 저장된 같은 기사: 태그 합집합
            prev["tickers"] = sorted(set(prev["tickers"]) | set(row["tickers"]))
    df = pd.DataFrame(list(merged.values()), columns=[c for c in COLUMNS if c != "n_tags"])
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    df["tickers"] = df["tickers"].map(lambda t: sorted(set(t)))
    df["n_tags"] = df["tickers"].map(len)
    df["raw_extra"] = df["raw_extra"].map(lambda d: json.dumps(d, ensure_ascii=False, sort_keys=True))
    return df[COLUMNS].reset_index(drop=True)
