"""Colab(GPU)용 FinBERT 점수화 — 레포 의존성 없이 단독 실행.

로컬 `finbert_scores_len128.parquet` 캐시와 같은 형식(article_id, p_pos, p_neg, p_neu, sent_score)으로 저장한다.
입력은 `colab_finbert_input.parquet`(article_id, text). 설정은 로컬 FinBertScorer와 동일하다
(ProsusAI/finbert, max_length 128, truncation, 점수 = p_pos - p_neg).

Colab 사용:
  !pip -q install transformers
  # colab_finbert_input.parquet 을 업로드한 뒤
  !python colab_finbert_score.py colab_finbert_input.parquet finbert_scores_colab.parquet
로컬에서는 결과를 `data/interim/news/finbert_scores_len128.parquet` 에 이어붙인 뒤
`python scripts/score_news_sentiment.py` 를 실행한다(캐시에 있으면 재계산하지 않는다).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME, MAX_LENGTH, BATCH = "ProsusAI/finbert", 128, 128


def main(src: str, dst: str) -> None:
    df = pd.read_parquet(src)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(df)}건, device={device}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval().to(device)
    if device == "cuda":
        model.half()
    idx = {str(n).lower(): int(i) for i, n in model.config.id2label.items()}
    cols = [idx["positive"], idx["negative"], idx["neutral"]]

    texts = df["text"].tolist()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 3), dtype="float32")
    with torch.inference_mode():
        for s in range(0, len(texts), BATCH):
            ids = order[s:s + BATCH]
            enc = tok([texts[i] for i in ids], padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt").to(device)
            out[ids] = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()[:, cols]
            if (s // BATCH) % 100 == 0:
                print(f"{s}/{len(texts)}", flush=True)

    res = pd.DataFrame({"article_id": df["article_id"], "p_pos": out[:, 0], "p_neg": out[:, 1], "p_neu": out[:, 2]})
    res["sent_score"] = res["p_pos"] - res["p_neg"]
    res.to_parquet(dst, index=False)
    print("저장:", dst, res.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
