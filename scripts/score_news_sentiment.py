"""뉴스 2단계 전처리(노이즈 표시) + FinBERT 감성 점수 + 종목·일 감성 피처 (모델 B 입력).

  python scripts/preprocess_news.py          # 1단계 (먼저)
  python scripts/score_news_sentiment.py     # 2단계 + 감성

입력: data/interim/news/articles.parquet, data/processed/news/{article_ticker,ticker_day}.parquet
출력: data/processed/news/{article_ticker_scored,news_features}.parquet, 점수 캐시 data/interim/news/finbert_scores.parquet
상세: docs/plans/news_preprocessing_plan.md, src/capstone/data/news_preprocessing/{noise,sentiment}.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.news_preprocessing.config import load_config  # noqa: E402
from capstone.data.news_preprocessing.noise import flag_noise, sentiment_text  # noqa: E402
from capstone.data.news_preprocessing.sentiment import (  # noqa: E402
    MODEL_NAME, FinBertScorer, aggregate_sentiment, score_articles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "news_preprocessing.yaml")
    parser.add_argument("--max-length", type=int, default=128, help="FinBERT 입력 최대 토큰")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    interim, processed = REPO_ROOT / cfg.paths["interim"], REPO_ROOT / cfg.paths["processed"]
    articles = pd.read_parquet(interim / "articles.parquet")
    article_ticker = pd.read_parquet(processed / "article_ticker.parquet")
    ticker_day = pd.read_parquet(processed / "ticker_day.parquet")

    flagged = flag_noise(articles, cfg.broad_tag_threshold)
    print("제외 사유별 기사 수 (전체 기간):")
    print(flagged["drop_reason"].fillna("(유지)").value_counts().to_string())
    in_window = flagged["attributed_day"].between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))
    print(f"\n학습 구간({cfg.start}~{cfg.end}) 기사 {int(in_window.sum())}건 중 제외 {int(flagged.loc[in_window, 'drop_reason'].notna().sum())}건")

    keep = flagged[flagged["drop_reason"].isna()].copy()
    keep["text"] = sentiment_text(keep)
    scorer = FinBertScorer(MODEL_NAME, max_length=args.max_length, batch_size=args.batch_size, threads=args.threads)
    cache = interim / f"finbert_scores_len{args.max_length}.parquet"
    scores = score_articles(keep["article_id"].tolist(), keep["text"].tolist(), scorer, cache)

    meta = flagged[["article_id", "drop_reason", "is_list_promo", "is_analyst_action"]]
    scored = article_ticker.merge(meta, on="article_id", how="left").merge(scores, on="article_id", how="left")
    scored.to_parquet(processed / "article_ticker_scored.parquet", index=False)

    sent = aggregate_sentiment(scored, cfg.tickers, cfg.warmup_start, cfg.end, window=cfg.window, min_periods=cfg.min_periods)
    features = ticker_day.merge(sent, on=["ticker", "day"], how="left", validate="one_to_one")
    features = features[features["day"].between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].reset_index(drop=True)
    features.to_parquet(processed / "news_features.parquet", index=False)
    print(f"\nnews_features: {len(features)}행 ({features['ticker'].nunique()}종목 x {features['day'].nunique()}거래일), "
          f"감성 있는 날 {100 * features['has_sent'].mean():.1f}%")


if __name__ == "__main__":
    main()
