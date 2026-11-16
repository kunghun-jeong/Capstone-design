"""Reddit 감성 점수(FinBERT) → 종목·일 감성 피처 CLI.

  pip install -r requirements-sentiment.txt          # torch, transformers (score 단계만 필요)
  python scripts/score_reddit_sentiment.py            # score + aggregate
  python scripts/score_reddit_sentiment.py --stage aggregate   # 점수는 그대로 두고 집계만 다시

먼저 scripts/preprocess_reddit.py로 전처리 산출물을 만들어 둬야 한다.
score는 글 단위 결과를 이어서 저장하므로 중간에 끊겨도 같은 명령으로 이어서 계산한다.
모델·리비전·길이·기기 설정은 configs/reddit_sentiment.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.features.sentiment.config import load_sentiment_config  # noqa: E402
from capstone.features.sentiment.pipeline import run_aggregate, run_score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "reddit_sentiment.yaml")
    parser.add_argument("--stage", choices=["score", "aggregate", "all"], default="all")
    args = parser.parse_args()
    scfg = load_sentiment_config(args.config)
    if args.stage in ("score", "all"):
        print("score", json.dumps(run_score(scfg, REPO_ROOT), ensure_ascii=False), flush=True)
    if args.stage in ("aggregate", "all"):
        log = run_aggregate(scfg, REPO_ROOT, args.config)
        print("aggregate", json.dumps({k: log[k] for k in ("rows", "sent_mean_missing_days", "sent_mean_z_missing_days")}), flush=True)


if __name__ == "__main__":
    main()
