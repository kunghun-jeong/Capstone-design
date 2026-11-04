"""FinBERT 글 단위 감성 점수.

점수 = P(positive) − P(negative) ∈ [−1, 1]. 라벨은 확률이 가장 큰 쪽.
같은 글은 한 번만 계산하고(article_id 단위), 결과를 parquet에 덧붙여 저장해서 중간에 끊겨도 이어서 계산한다.
torch/transformers는 무거워서 실제 모델을 쓸 때만 import한다(테스트·CI는 가짜 scorer 사용).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

LABELS = ["positive", "negative", "neutral"]
SCORE_COLUMNS = ["article_id", "text_sha1", "p_pos", "p_neg", "p_neu", "sent_score", "sent_label", "model", "revision"]


class Scorer(Protocol):
    name: str
    revision: str

    def score(self, texts: list[str]) -> np.ndarray:
        """(len(texts), 3) 확률, 열 순서는 LABELS."""


def label_order(id2label: dict) -> list[int]:
    """모델 출력 열 → LABELS 순서로 고르는 인덱스. 모델마다 라벨 순서가 달라서 이름으로 맞춘다."""
    by_name = {str(v).lower(): int(k) for k, v in id2label.items()}
    missing = set(LABELS) - set(by_name)
    if missing:
        raise ValueError(f"모델 라벨에 {sorted(missing)}가 없습니다: {id2label}")
    return [by_name[label] for label in LABELS]


class FinbertScorer:
    def __init__(self, name: str, revision: str, max_length: int = 512, batch_size: int = 128,
                 device: str = "auto", half_precision: str | bool = "auto"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        half = device != "cpu" if half_precision == "auto" else bool(half_precision)
        self.name, self.revision, self.device = name, revision, device
        self.dtype = "float16" if half else "float32"
        self.max_length, self.batch_size = max_length, batch_size
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision, dtype=getattr(torch, self.dtype))
        self._model = model.eval().to(device)
        self._order = label_order(model.config.id2label)

    def score(self, texts: list[str]) -> np.ndarray:
        # 길이순으로 묶어 padding 낭비를 줄이고, 끝나면 원래 순서로 되돌린다
        idx = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), len(LABELS)), dtype=np.float32)
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                sel = idx[i:i + self.batch_size]
                batch = self._tok([texts[j] for j in sel], truncation=True, max_length=self.max_length,
                                  padding=True, return_tensors="pt").to(self.device)
                probs = self._model(**batch).logits.float().softmax(-1).cpu().numpy()
                out[sel] = probs[:, self._order]
        return out


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _to_frame(ids, hashes, probs: np.ndarray, scorer: Scorer) -> pd.DataFrame:
    df = pd.DataFrame({"article_id": ids, "text_sha1": hashes, "p_pos": probs[:, 0], "p_neg": probs[:, 1], "p_neu": probs[:, 2]})
    df["sent_score"] = df["p_pos"] - df["p_neg"]
    df["sent_label"] = np.array(LABELS)[probs.argmax(axis=1)]
    df["model"], df["revision"] = scorer.name, scorer.revision
    return df[SCORE_COLUMNS]


def score_articles(articles: pd.DataFrame, scorer: Scorer, out_path: Path, chunk_size: int = 20_000,
                   log: Callable[[str], None] = print) -> pd.DataFrame:
    """articles(article_id, text) → 글 단위 점수. out_path에 이미 있는(같은 모델·리비전·같은 텍스트) 글은 건너뛴다."""
    out_path = Path(out_path)
    todo = articles[["article_id", "text"]].drop_duplicates("article_id").reset_index(drop=True)
    todo["text_sha1"] = todo["text"].map(text_sha1)
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=SCORE_COLUMNS)
    other = done[(done["model"] != scorer.name) | (done["revision"] != scorer.revision)]
    if len(other):
        raise ValueError(f"{out_path}에 다른 모델/리비전 점수 {len(other)}건이 있습니다. 파일을 옮기고 다시 실행하세요.")
    # 텍스트가 바뀐 글(전처리 규칙 변경 등)은 다시 계산한다
    done = done.merge(todo[["article_id", "text_sha1"]], on=["article_id", "text_sha1"], how="inner")
    todo = todo[~todo["article_id"].isin(done["article_id"])].reset_index(drop=True)
    log(f"점수 있음 {len(done):,} / 새로 계산 {len(todo):,}")
    for start in range(0, len(todo), chunk_size):
        part = todo.iloc[start:start + chunk_size]
        probs = scorer.score(part["text"].tolist())
        done = pd.concat([done, _to_frame(part["article_id"].values, part["text_sha1"].values, probs, scorer)], ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(out_path, index=False)
        log(f"  {min(start + chunk_size, len(todo)):,}/{len(todo):,}")
    return done[done["article_id"].isin(articles["article_id"])].reset_index(drop=True)
