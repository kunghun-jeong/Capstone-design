"""모델 B용 Reddit 전처리 CLI (Arctic Shift raw JSONL → parquet).

  python scripts/preprocess_reddit.py                 # 1단계 + 2단계
  python scripts/preprocess_reddit.py --stage 2       # interim은 그대로 두고 2단계만 다시 (설정 변경 후)
  python scripts/preprocess_reddit.py --config configs/reddit_preprocessing.yaml --workers 4

1단계는 원본 약 3GB를 파일 단위로 병렬 처리한다(기본 workers = CPU 수 − 2).
입력/출력 경로와 기간·컷오프·베이스라인·노이즈 규칙은 전부 설정 파일에서 읽는다. 상세는 docs/plans/reddit_preprocessing_plan.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.data.reddit_preprocessing.config import load_config  # noqa: E402
from capstone.data.reddit_preprocessing.pipeline import run_stage1, run_stage2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "reddit_preprocessing.yaml")
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    parser.add_argument("--workers", type=int, default=None, help="1단계 병렬 프로세스 수")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage in ("1", "all"):
        log = run_stage1(cfg, REPO_ROOT, args.config, args.workers)
        print("stage1", json.dumps(log["rows"], ensure_ascii=False))
    if args.stage in ("2", "all"):
        log = run_stage2(cfg, REPO_ROOT, args.config)
        print("stage2", json.dumps({k: log[k] for k in ("removed", "merged_copies", "rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
