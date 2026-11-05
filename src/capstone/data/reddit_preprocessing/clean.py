"""2단계: 노이즈 제거 → 복붙 합치기 → (글, 종목) explode.

제거 규칙(사유별 건수를 남긴다): 본문 없음(삭제·제거), 봇, 운영진, 고정 글, 링크만, (설정 시) 제거 표시 게시글.
짧은 글은 지우지 않는다 — "NVDA 🚀"도 실제 언급이라 언급량에서 빼면 신호가 바뀐다. is_short로 감성 입력에서만 뺀다.
같은 날·같은 서브레딧·같은 본문(복붙·밈)은 1건으로 합치고 n_rows(합쳐진 원본 수)를 남긴다 — 뉴스 dedup과 같은 원칙.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import Config

_WS = re.compile(r"\s+")

AT_COLUMNS = [
    # 뉴스 article_ticker와 같은 이름·의미
    "article_id", "ticker", "attributed_day", "published_utc", "n_tags", "n_rows", "n_sources",
    "publisher", "is_primary", "is_broad", "title", "summary",
    # Reddit 고유
    "subreddit", "kind", "is_name_only", "is_short", "is_removed", "is_edited", "n_other_tickers",
    "author_hash", "text_for_sentiment", "link_title", "source_ids",
]


def drop_reasons(cfg: Config) -> dict[str, str]:
    reasons = {
        "deleted_text": "is_deleted_text", "bot": "is_bot", "mod": "is_mod",
        "stickied": "is_stickied", "link_only": "is_link_only",
    }
    if cfg.drop_removed_posts:
        reasons["removed_post"] = "is_removed"
    return reasons


def remove_noise(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """규칙 순서대로 처음 걸린 사유 하나로 센다(중복 집계 없음)."""
    keep = pd.Series(True, index=df.index)
    removed: dict[str, int] = {}
    for reason, col in drop_reasons(cfg).items():
        hit = keep & df[col].astype(bool)
        removed[reason] = int(hit.sum())
        keep &= ~hit
    return df[keep].reset_index(drop=True), removed


def normalize_body(text) -> str | None:
    if not isinstance(text, str):
        return None
    key = _WS.sub(" ", text.lower()).strip()
    return key or None


def merge_copies(df: pd.DataFrame) -> pd.DataFrame:
    """(subreddit, 귀속일, 정규화 본문)이 같은 글을 묶는다. 대표 = 가장 먼저 올라온 글."""
    df = df.sort_values(["published_utc", "article_id"]).reset_index(drop=True)
    key = pd.Series(list(zip(df["subreddit"], df["attributed_day"], df["text_for_sentiment"].map(normalize_body))))
    group = key.factorize()[0]
    rep = df.groupby(group, sort=False).head(1).copy()
    g = df.groupby(group, sort=False)
    rep["n_rows"] = g.size().to_numpy()
    rep["source_ids"] = g["article_id"].agg(list).to_numpy()
    rep["is_edited"] = g["is_edited"].any().to_numpy()
    return rep.reset_index(drop=True)


def _primary(kind: str, title, ticker_in_title: bool, n_tags: int) -> bool:
    # 그 글이 이 종목 하나만 말하고, 게시글이면 제목에 그 티커가 있을 때 (뉴스 "제목에 등장"의 Reddit판)
    return n_tags == 1 and (kind == "comment" or ticker_in_title)


def explode_by_ticker(articles: pd.DataFrame, cfg: Config, title_tickers) -> pd.DataFrame:
    """글 × 종목 행. title_tickers(title) → 제목에 나온 유니버스 티커 목록."""
    rows = []
    for rec in articles.itertuples(index=False):
        in_title = set(title_tickers(rec.title)) if rec.kind == "post" else set()
        targets = [(t, False) for t in rec.tickers]
        if cfg.include_name_only:
            targets += [(t, True) for t in rec.name_only_tickers]
        for t, name_only in targets:
            rows.append({
                "article_id": rec.article_id, "ticker": t, "attributed_day": rec.attributed_day,
                "published_utc": rec.published_utc, "n_tags": rec.n_tags, "n_rows": rec.n_rows, "n_sources": 1,
                "publisher": rec.publisher,
                "is_primary": (not name_only) and _primary(rec.kind, rec.title, t in in_title, rec.n_tags),
                "is_broad": rec.n_tags + rec.n_other_tickers >= cfg.broad_tag_threshold,
                "title": rec.title, "summary": rec.summary,
                "subreddit": rec.subreddit, "kind": rec.kind, "is_name_only": name_only,
                "is_short": rec.is_short, "is_removed": rec.is_removed, "is_edited": rec.is_edited,
                "n_other_tickers": rec.n_other_tickers, "author_hash": rec.author_hash,
                "text_for_sentiment": rec.text_for_sentiment, "link_title": rec.link_title,
                "source_ids": rec.source_ids,
            })
    return pd.DataFrame(rows, columns=AT_COLUMNS)
