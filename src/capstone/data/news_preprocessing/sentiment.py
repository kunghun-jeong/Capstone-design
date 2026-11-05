"""FinBERT 감성 점수 + 종목·일 감성 집계.

기사 단위로 한 번만 점수를 매기고(다종목 기사도 1회), 기사×종목 표에 붙여 집계한다. 점수는 캐시 parquet에
이어쓰기로 저장해서 중단 후 재실행해도 이어받는다.
 - 모델: ProsusAI/finbert (positive / negative / neutral 확률). 점수 = p_pos - p_neg (-1 ~ +1)
 - 입력: 제목 + 요약, 최대 max_length 토큰
집계: "평소 대비 편차" = 그날 평균 감성 - 과거 window 거래일 평균(당일 제외). 글이 없던 날은 NaN으로 두고
has_sent로 표시한다 (0으로 채우지 않는다 - 뉴스 없음과 중립은 다르다).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .trading_calendar import trading_sessions

MODEL_NAME = "ProsusAI/finbert"
SCORE_COLS = ["p_pos", "p_neg", "p_neu", "sent_score"]


class FinBertScorer:
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = 128, batch_size: int = 32, threads: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self._torch = torch
        self.max_length, self.batch_size = max_length, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {int(i): str(n).lower() for i, n in self.model.config.id2label.items()}
        self._idx = {name: i for i, name in labels.items()}   # 라벨 순서는 모델 설정에서 읽는다
        missing = {"positive", "negative", "neutral"} - set(self._idx)
        if missing:
            raise ValueError(f"모델 라벨에 {missing}가 없습니다: {labels}")

    def score(self, texts: list[str]) -> np.ndarray:
        """texts → (n, 3) 확률 배열, 열 순서 [p_pos, p_neg, p_neu]. 길이순으로 묶어 패딩 낭비를 줄인다."""
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 3), dtype="float32")
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1).numpy()
                out[idx] = probs[:, [self._idx["positive"], self._idx["negative"], self._idx["neutral"]]]
        return out


def score_articles(ids: list[str], texts: list[str], scorer, cache_path: Path, chunk: int = 2000, log=print) -> pd.DataFrame:
    """캐시에 없는 기사만 점수를 매겨 캐시에 이어쓰고, 요청한 id 전체의 점수 표를 돌려준다."""
    cache_path = Path(cache_path)
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["article_id", *SCORE_COLS])
    done = set(cache["article_id"])
    todo = [(i, t) for i, t in zip(ids, texts) if i not in done]
    log(f"감성 점수: 캐시 {len(done)}건, 새로 계산 {len(todo)}건")
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        probs = scorer.score([t for _, t in part])
        new = pd.DataFrame({"article_id": [i for i, _ in part], "p_pos": probs[:, 0], "p_neg": probs[:, 1], "p_neu": probs[:, 2]})
        new["sent_score"] = new["p_pos"] - new["p_neg"]
        cache = pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
        log(f"  {min(start + chunk, len(todo))}/{len(todo)}")
    return cache[cache["article_id"].isin(set(ids))].reset_index(drop=True)


def aggregate_sentiment(
    scored: pd.DataFrame, tickers: list[str], warmup_start, end, *, window: int, min_periods: int, std_floor: float = 0.05,
) -> pd.DataFrame:
    """기사×종목 표(attributed_day, ticker, article_id, is_primary, drop_reason, sent_score …) → (ticker, day) 감성 피처.

    drop_reason이 있는 기사는 뺀다. sent_dev = 그날 평균 - 과거 window 거래일 평균(당일 제외),
    sent_z = sent_dev / max(과거 표준편차, std_floor).
    """
    sessions = trading_sessions(warmup_start, end)
    grid = pd.MultiIndex.from_product([tickers, sessions], names=["ticker", "day"]).to_frame(index=False)
    clean = scored[scored["drop_reason"].isna() & scored["sent_score"].notna()].rename(columns={"attributed_day": "day"})
    g = clean.groupby(["ticker", "day"])
    agg = pd.DataFrame({
        "n_scored": g["article_id"].nunique(),
        "sent_mean": g["sent_score"].mean(),
        "p_pos_mean": g["p_pos"].mean(),
        "p_neg_mean": g["p_neg"].mean(),
    })
    prim = clean[clean["is_primary"]].groupby(["ticker", "day"])
    agg["n_scored_primary"] = prim["article_id"].nunique()
    agg["sent_mean_primary"] = prim["sent_score"].mean()
    out = grid.merge(agg.reset_index(), on=["ticker", "day"], how="left").sort_values(["ticker", "day"]).reset_index(drop=True)
    out["n_scored"] = out["n_scored"].fillna(0).astype(int)
    out["n_scored_primary"] = out["n_scored_primary"].fillna(0).astype(int)
    out["has_sent"] = out["n_scored"] > 0

    def _dev(col: str) -> tuple[pd.Series, pd.Series]:
        base_mean = out.groupby("ticker")[col].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        base_std = out.groupby("ticker")[col].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
        dev = out[col] - base_mean
        return dev, dev / base_std.clip(lower=std_floor)

    out["sent_dev"], out["sent_z"] = _dev("sent_mean")
    out["sent_dev_primary"], out["sent_z_primary"] = _dev("sent_mean_primary")
    return out
