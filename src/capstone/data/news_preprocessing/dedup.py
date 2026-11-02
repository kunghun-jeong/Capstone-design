"""소스 간 중복 제거 + (기사, 종목) explode.

중복 제거는 기사를 지우되 "얼마나 퍼졌나"는 남긴다: sources/n_sources(합쳐진 소스), n_rows(합쳐진 원본 행 수).
소스 간 매칭은 (1) 정규화 URL, (2) 정규화 제목 + 귀속일. 근사 중복(신디케이션)은 하지 않는다.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

import pandas as pd

_SOURCE_PRIORITY = {"alpaca": 0, "massive": 1}   # 대표 레코드 선호: 본문이 있는 Alpaca
_TRACKING = re.compile(r"^(utm_|fbclid|gclid)")


def normalize_url(url) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    parts = urlsplit(url.strip().lower())
    host = parts.netloc.removeprefix("www.")
    query = "&".join(p for p in parts.query.split("&") if p and not _TRACKING.match(p))
    key = f"{host}{parts.path.rstrip('/')}" + (f"?{query}" if query else "")
    return key or None


def normalize_title(title) -> str | None:
    if not isinstance(title, str):
        return None
    key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return key or None


def _text_len(row) -> int:
    return sum(len(x) for x in (row["title"], row["summary"], row["body"]) if isinstance(x, str))


def dedup_cross_source(df: pd.DataFrame) -> pd.DataFrame:
    """normalize()+귀속일(attributed_day) 컬럼이 붙은 결합 테이블 → 기사 단위 테이블."""
    df = df.copy().reset_index(drop=True)
    df["_url_key"] = df["url"].map(normalize_url)
    df["_title_key"] = df["title"].map(normalize_title)

    # union-find: URL 일치 또는 (제목, 귀속일) 일치 행을 한 그룹으로
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_by(keys: pd.Series) -> None:
        first: dict = {}
        for i, key in enumerate(keys):
            if key is None or (isinstance(key, float) and pd.isna(key)) or key is pd.NaT:
                continue
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i

    union_by(df["_url_key"])
    union_by(pd.Series(list(zip(df["_title_key"], df["attributed_day"])), index=df.index).where(df["_title_key"].notna()))
    df["_group"] = [find(i) for i in range(len(df))]

    out = []
    for _, g in df.groupby("_group", sort=False):
        order = sorted(
            g.index,
            key=lambda i: (_SOURCE_PRIORITY.get(df.at[i, "source"], 9), -_text_len(df.loc[i]), df.at[i, "published_utc"]),
        )
        rep = df.loc[order[0]].copy()
        rep["tickers"] = sorted({t for ts in g["tickers"] for t in ts})
        rep["n_tags"] = len(rep["tickers"])
        rep["sources"] = sorted(g["source"].unique())
        rep["n_sources"] = len(rep["sources"])
        rep["n_rows"] = len(g)
        rep["publishers"] = sorted({p for p in g["publisher"] if isinstance(p, str)})
        rep["source_ids"] = sorted(g["article_id"])
        out.append(rep)
    res = pd.DataFrame(out).drop(columns=["_url_key", "_title_key", "_group"]).reset_index(drop=True)
    return res.sort_values(["published_utc", "article_id"]).reset_index(drop=True)


def _mentions(title, ticker: str, aliases: list[str]) -> bool:
    if not isinstance(title, str):
        return False
    # 티커는 대문자 단어 그대로(MU가 일반 단어와 섞이지 않게), 별칭은 대소문자 무시
    if re.search(rf"\b{re.escape(ticker)}\b", title):
        return True
    return any(re.search(rf"\b{re.escape(a)}\b", title, flags=re.IGNORECASE) for a in aliases)


def explode_by_ticker(articles: pd.DataFrame, tickers: list[str], aliases: dict[str, list[str]], broad_threshold: int) -> pd.DataFrame:
    """기사 × (유니버스 ∩ 태그) 행. 행을 버리지 않고 is_primary / is_broad 플래그만 붙인다."""
    universe = set(tickers)
    rows = []
    for rec in articles.itertuples(index=False):
        for t in rec.tickers:
            if t not in universe:
                continue
            rows.append({
                "article_id": rec.article_id, "ticker": t, "attributed_day": rec.attributed_day,
                "published_utc": rec.published_utc, "n_tags": rec.n_tags, "n_rows": rec.n_rows,
                "n_sources": rec.n_sources, "publisher": rec.publishers[0] if rec.publishers else None,
                "is_primary": _mentions(rec.title, t, aliases.get(t, [])),
                "is_broad": rec.n_tags > broad_threshold,
                "title": rec.title, "summary": rec.summary,
            })
    cols = ["article_id", "ticker", "attributed_day", "published_utc", "n_tags", "n_rows", "n_sources",
            "publisher", "is_primary", "is_broad", "title", "summary"]
    return pd.DataFrame(rows, columns=cols)
