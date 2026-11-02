"""모델 B용 뉴스 전처리 CLI (Alpaca + Massive raw JSONL → parquet).

  python scripts/preprocess_news.py
  python scripts/preprocess_news.py --config configs/news_preprocessing.yaml

입력/출력 경로와 기간·컷오프·베이스라인은 전부 설정 파일에서 읽는다. 상세는 docs/plans/news_preprocessing_plan.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.news_preprocessing.config import load_config  # noqa: E402
from capstone.data.news_preprocessing.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "news_preprocessing.yaml")
    args = parser.parse_args()
    log = run(load_config(args.config), REPO_ROOT, args.config)
    print(json.dumps(log["rows"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
