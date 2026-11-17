"""모델 B 실험: 선형(Ridge)·XGBoost·MLP × (뉴스 / Reddit / 둘 다) 를 walk-forward 로 비교한다.

  python scripts/train_model_b.py --model-a exp1_chronos2_H3.csv --reddit ticker_day_sentiment.parquet

결과(표·예측)는 --out-dir(기본 data/processed/model_b)에 저장한다. data/ 는 gitignore 이다.
Reddit 파일은 글 원문이 없는 종목·일 표지만 쓰지만, 수업 내부 자료이니 레포에 올리지 말 것.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capstone.models.model_b.experiment import build_features, load_dataset, run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-a", type=Path, required=True)
    p.add_argument("--reddit", type=Path, required=True)
    p.add_argument("--news", type=Path, default=REPO_ROOT / "data/processed/news/news_features.parquet")
    p.add_argument("--start", default="2021-01-04")
    p.add_argument("--end", default="2026-09-30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/processed/model_b")
    args = p.parse_args()

    df = load_dataset(args.model_a, args.news, args.reddit, args.start, args.end)
    feat, sets = build_features(df)
    print(f"{len(feat)}행 ({feat['ticker'].nunique()}종목 x {feat['day'].nunique()}거래일), {feat['day'].min().date()} ~ {feat['day'].max().date()}")
    table, preds = run(feat, sets, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "results.csv", index=False)
    preds.to_parquet(args.out_dir / "predictions.parquet", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
