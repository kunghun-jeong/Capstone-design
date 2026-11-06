"""Reddit 감성 파이프라인.

score: interim/posts_comments → (전처리 2단계와 같은 정제) → 고유 글 텍스트 → FinBERT → ``paths.scores``
aggregate: 점수 + 글×종목 + processed/ticker_day → ``paths.output`` + ``_run_sentiment.json``
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from capstone.data.news_preprocessing.pipeline import _git_sha, _sha256
from capstone.data.reddit_preprocessing.config import Config, load_config
from capstone.data.reddit_preprocessing.pipeline import build_article_ticker

from .aggregate import join_with_mentions, sentiment_ticker_day
from .config import SentimentConfig
from .finbert import FinbertScorer, score_articles


def load_article_ticker(root: Path, cfg: Config) -> pd.DataFrame:
    rows = pd.read_parquet(root / cfg.paths["interim"] / "posts_comments.parquet")
    return build_article_ticker(rows, cfg)[0]


def articles_to_score(article_ticker: pd.DataFrame, exclude_short: bool) -> pd.DataFrame:
    at = article_ticker[~article_ticker["is_short"]] if exclude_short else article_ticker
    return at.drop_duplicates("article_id").rename(columns={"text_for_sentiment": "text"})[["article_id", "text"]]


def run_score(scfg: SentimentConfig, root: Path) -> dict:
    cfg = load_config(root / scfg.preprocessing_config)
    articles = articles_to_score(load_article_ticker(root, cfg), scfg.exclude_short)
    scorer = FinbertScorer(scfg.model_name, scfg.revision, scfg.max_length, scfg.batch_size, scfg.device, scfg.half_precision)
    print(f"model={scorer.name}@{scorer.revision[:10]} device={scorer.device} dtype={scorer.dtype} articles={len(articles):,}", flush=True)
    scores = score_articles(articles, scorer, root / scfg.paths["scores"], log=lambda m: print(m, flush=True))
    return {"articles": len(articles), "scored": len(scores), "device": scorer.device, "dtype": scorer.dtype}


def run_aggregate(scfg: SentimentConfig, root: Path, config_path: Path) -> dict:
    cfg = load_config(root / scfg.preprocessing_config)
    article_ticker = load_article_ticker(root, cfg)
    scores = pd.read_parquet(root / scfg.paths["scores"])
    sentiment = sentiment_ticker_day(article_ticker, scores, cfg, scale_floor=scfg.sentiment_scale_floor,
                                     exclude_short=scfg.exclude_short)
    ticker_day = pd.read_parquet(root / cfg.paths["processed"] / "ticker_day.parquet")
    out = join_with_mentions(ticker_day, sentiment)
    out_path = root / scfg.paths["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(root),
        "config_sha256": _sha256(config_path),
        "preprocessing_config_sha256": _sha256(root / scfg.preprocessing_config),
        "model": scfg.model_name, "revision": scfg.revision, "max_length": scfg.max_length,
        "scores_used": int(scores["article_id"].isin(article_ticker["article_id"]).sum()),
        "rows": len(out),
        "sent_mean_missing_days": int(out["sent_mean"].isna().sum()),
        "sent_mean_z_missing_days": int(out["sent_mean_z"].isna().sum()),
    }
    (out_path.parent / "_run_sentiment.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return log
