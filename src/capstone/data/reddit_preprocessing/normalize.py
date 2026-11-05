"""1단계: Arctic Shift JSONL → 공통 스키마 + 종목 언급 판정 + 노이즈 플래그 (행은 버리지 않는다).

원본 529만 건을 메모리에 올리지 않도록 파일 단위로 처리한다. 파일마다 두 가지를 낸다.
 - 행(rows): 유니버스 티커나 회사명이 하나라도 나온 글만. 뉴스 기사 단위 컬럼 + Reddit 고유 컬럼.
 - 활동량(activity): 모든 글을 (subreddit, kind, 귀속일)별 건수로만 센다 — 언급 비율의 분모.
노이즈는 여기서 플래그만 붙이고 실제 제거는 2단계(clean.py)에서 한다.
"나중에 알게 된 정보"(score, num_comments, _meta.was_deleted_later 등)는 필터에 쓰지 않는다 — 계획 ④.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd

from ..news_preprocessing.trading_calendar import attribute_trading_day
from .config import Config

PLACEHOLDER_TEXT = {"[removed]", "[deleted]"}
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_FILE_RE = re.compile(r"^\d{4}-\d{2}\.jsonl$")

ROW_COLUMNS = [
    # 뉴스 기사 단위와 같은 이름·의미
    "article_id", "source", "published_utc", "title", "summary", "body", "url", "publisher", "tickers", "n_tags",
    # Reddit 고유
    "subreddit", "kind", "attributed_day", "name_only_tickers", "other_tickers", "n_other_tickers",
    "text_for_sentiment", "n_chars", "link_title", "author_hash",
    "is_deleted_text", "is_bot", "is_mod", "is_stickied", "is_link_only", "is_removed", "is_short", "is_edited",
    # 분석용(피처·필터 금지): 첫 수집 이후에 바뀐 값
    "score_t36h", "num_comments_t36h", "has_2nd_retrieval", "was_deleted_later",
]
ACTIVITY_FLAGS = ["is_deleted_text", "is_bot", "is_mod", "is_stickied", "is_link_only", "is_removed", "is_short", "is_edited"]


def clean_text(value) -> str | None:
    """HTML 엔티티 해제(&amp; 등) + 공백 정규화. 빈 결과·삭제 표시는 None.

    뉴스의 clean_text는 BeautifulSoup로 태그를 지우지만, Reddit 본문은 HTML이 아니라 마크다운이라
    "<3", "<$100" 같은 텍스트가 지워지지 않도록 태그 제거는 하지 않는다.
    """
    if not isinstance(value, str):
        return None
    text = _WS.sub(" ", html.unescape(value)).strip()
    if not text or text in PLACEHOLDER_TEXT:
        return None
    return text


class TickerMatcher:
    """글 → (유니버스 티커, 회사명만 나온 종목, 유니버스 밖 티커).

    티커는 대문자 단어 경계 + 선택적 '$'(대소문자 구분, MU·META가 일반 단어와 섞이지 않게).
    '$'가 붙으면 소문자도 인정한다($nvda). 회사명은 대소문자 무시.
    """

    def __init__(self, tickers: list[str], patterns: dict[str, str], aliases: dict[str, list[str]], other: list[str]):
        self.universe = list(tickers)
        pats = {t: patterns.get(t, re.escape(t)) for t in tickers}
        pats.update({t: re.escape(t) for t in other if t not in pats})
        alt = "|".join(f"(?:{p})" for p in sorted(pats.values(), key=len, reverse=True))
        self._upper = re.compile(rf"(?<![A-Za-z0-9])\$?({alt})(?![A-Za-z0-9])")
        self._dollar = re.compile(rf"(?<![A-Za-z0-9])\$({alt})(?![A-Za-z0-9])", re.IGNORECASE)
        self._canon = [(t, re.compile(p)) for t, p in pats.items()]
        self._canon_cache: dict[str, str | None] = {}
        self._name_map = {a.lower(): t for t, names in aliases.items() if t in tickers for a in names}
        self._names = (
            re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(self._name_map, key=len, reverse=True)) + r")\b", re.IGNORECASE)
            if self._name_map else None
        )

    def canonical(self, token: str) -> str | None:
        key = token.upper()
        if key not in self._canon_cache:
            self._canon_cache[key] = next((t for t, rx in self._canon if rx.fullmatch(key)), None)
        return self._canon_cache[key]

    def match(self, text: str | None) -> tuple[list[str], list[str], list[str]]:
        if not text:
            return [], [], []
        found = {self.canonical(m.group(1)) for m in self._upper.finditer(text)}
        found |= {self.canonical(m.group(1)) for m in self._dollar.finditer(text)}
        found.discard(None)
        universe = sorted(found & set(self.universe))
        other = sorted(found - set(self.universe))
        names: list[str] = []
        if self._names is not None:
            names = sorted({self._name_of(m.group(1)) for m in self._names.finditer(text)} - set(universe))
        return universe, names, other

    def _name_of(self, token: str) -> str:
        # "MİCROSOFT"(터키어 İ)처럼 IGNORECASE로는 맞지만 .lower() 결과가 사전 키와 다른 경우가 있다
        hit = self._name_map.get(token.lower())
        if hit is None:
            hit = next(t for a, t in self._name_map.items() if re.fullmatch(re.escape(a), token, re.IGNORECASE))
        return hit


def author_hash(author, salt: str) -> str | None:
    if not isinstance(author, str) or not author or author in PLACEHOLDER_TEXT:
        return None
    return hashlib.sha256((salt + author).encode("utf-8")).hexdigest()[:16]


def _item(rec: dict, kind: str, cfg: Config, matcher: TickerMatcher, bots: set[str]) -> dict:
    """원본 1건 → 플래그와(언급이 있으면) 행 필드. 귀속일은 파일 단위로 한꺼번에 붙인다."""
    meta = rec.get("_meta") or {}
    if kind == "posts":
        title = clean_text(rec.get("title"))
        body = clean_text(rec.get("selftext"))
        text = "\n".join(x for x in (title, body) if x) or None
    else:
        title, body = None, clean_text(rec.get("body"))
        text = body
    stripped = _URL.sub("", text).strip() if text else ""
    author = rec.get("author")
    flags = {
        "is_deleted_text": text is None,
        "is_bot": isinstance(author, str) and author.lower() in bots,
        "is_mod": rec.get("distinguished") == "moderator",
        "is_stickied": bool(rec.get("stickied")),
        "is_link_only": text is not None and not stripped,
        # removed_by_category는 첫 수집(게시 후 약 30초) 시점 값. _meta.removal_type은 나중 값이라 쓰지 않는다.
        "is_removed": kind == "posts" and rec.get("removed_by_category") is not None,
        "is_short": text is not None and len(text) < cfg.min_chars,
        "is_edited": bool(rec.get("edited")) or bool(meta.get("is_edited")),
    }
    tickers, names, other = matcher.match(text)
    item = {"created_utc": rec["created_utc"], **flags, "_mention": bool(tickers or names)}
    if item["_mention"]:
        permalink = rec.get("permalink")
        item.update({
            "article_id": f"reddit:{rec.get('name') or rec['id']}",
            "source": "reddit",
            "title": title,
            "summary": None,
            "body": body,
            "url": f"https://www.reddit.com{permalink}" if permalink else None,
            "publisher": rec.get("subreddit"),
            "tickers": tickers,
            "n_tags": len(tickers),
            "subreddit": rec.get("subreddit"),
            "kind": "post" if kind == "posts" else "comment",
            "name_only_tickers": names,
            "other_tickers": other,
            "n_other_tickers": len(other),
            "text_for_sentiment": text,
            "n_chars": len(text) if text else 0,
            "link_title": clean_text(rec.get("link_title")),
            "author_hash": author_hash(author, cfg.author_salt),
            "score_t36h": rec.get("score"),
            "num_comments_t36h": rec.get("num_comments"),
            "has_2nd_retrieval": "retrieved_2nd_on" in meta,
            "was_deleted_later": bool(meta.get("was_deleted_later")),
        })
    return item


def list_raw_files(raw_dirs: list[Path]) -> list[tuple[Path, str, str]]:
    """``<raw>/<subreddit>/<posts|comments>/<YYYY-MM>.jsonl`` → (경로, subreddit, kind). ``_logs`` 등 밑줄 폴더 제외."""
    out = []
    for raw in raw_dirs:
        for kind in ("posts", "comments"):
            for path in sorted(Path(raw).glob(f"*/{kind}/*.jsonl")):
                sub = path.parent.parent.name
                if not sub.startswith("_") and _FILE_RE.match(path.name):
                    out.append((path, sub, kind))
    return out


def process_file(path: Path, subreddit: str, kind: str, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 파일 1개 → (언급 행 DataFrame, 귀속일별 활동량 DataFrame)."""
    matcher = TickerMatcher(cfg.tickers, cfg.ticker_patterns, cfg.aliases, cfg.other_tickers)
    bots = {b.lower() for b in cfg.bots}
    items = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(_item(json.loads(line), kind, cfg, matcher, bots))
    if not items:
        return pd.DataFrame(columns=ROW_COLUMNS), pd.DataFrame()

    # 뉴스 산출물과 같은 타입(datetime64[ns, UTC])
    published = pd.to_datetime(pd.Series([it["created_utc"] for it in items]), unit="s", utc=True).astype("datetime64[ns, UTC]")
    days = attribute_trading_day(published, cfg.cutoff)

    flags = pd.DataFrame({f: [it[f] for it in items] for f in ACTIVITY_FLAGS})
    noise = flags[["is_deleted_text", "is_bot", "is_mod", "is_stickied", "is_link_only"]].any(axis=1)
    flags["is_clean"] = ~noise
    flags["is_clean_excl_removed"] = ~noise & ~flags["is_removed"]
    flags["day"] = days.values
    activity = flags.groupby("day").agg(
        n_items=("is_clean", "size"),
        **{f"n_{f.removeprefix('is_')}": (f, "sum") for f in ACTIVITY_FLAGS},
        n_clean=("is_clean", "sum"),
        n_clean_excl_removed=("is_clean_excl_removed", "sum"),
    ).reset_index()
    activity.insert(0, "kind", "post" if kind == "posts" else "comment")
    activity.insert(0, "subreddit", subreddit)

    keep = [i for i, it in enumerate(items) if it["_mention"]]
    rows = pd.DataFrame([items[i] for i in keep])
    if rows.empty:
        return pd.DataFrame(columns=ROW_COLUMNS), activity
    rows["published_utc"] = published.iloc[keep].reset_index(drop=True)
    rows["attributed_day"] = days.iloc[keep].reset_index(drop=True)
    return rows[ROW_COLUMNS], activity
