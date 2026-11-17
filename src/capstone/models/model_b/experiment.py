"""모델 B 학습·평가: 텍스트 피처로 모델 A의 잔차 ε(= real_ret − pred_ret)를 예측한다.

입력 세 개를 (ticker, day)로 합친다.
  - 모델 A 예측 CSV: date, ticker, real_ret, pred_ret, residual (GOOG는 GOOGL로 맞춘다)
  - 뉴스 피처:  data/processed/news/news_features.parquet
  - Reddit 피처: ticker_day_sentiment.parquet (원문이 없는 종목·일 표지만 쓴다. 레포에 올리지 말 것)

평가는 walk-forward(확장 창): 뒤쪽 `n_folds` 블록을 차례로 테스트하고, 학습은 테스트 시작 `purge`거래일 전까지만 쓴다.
지표는 날짜별 횡단면 Spearman IC(평균, 겹치는 3일 수익률을 고려한 t값)와, 최종 예측 r̂ = A + B̂ 를 real_ret 에 대해
본 IC 및 A 단독 대비 변화다. ε 기준 IC는 corr(ε, A) 때문에 착시가 생기므로 최종 판단은 r̂ 기준으로 한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NEWS_FEATS = ["n_articles_z", "n_primary_z", "sent_z", "sent_z_primary"]
REDDIT_FEATS = ["mention_rate_z", "sent_mean_z", "mention_x_sent_z"]
CLIP = 5.0


def load_dataset(model_a_csv, news_parquet, reddit_parquet, start="2021-01-04", end="2026-09-30") -> pd.DataFrame:
    a = pd.read_csv(model_a_csv, parse_dates=["date"])
    a["ticker"] = a["ticker"].replace({"GOOG": "GOOGL"})
    a = a.rename(columns={"date": "day"})[["day", "ticker", "real_ret", "pred_ret", "residual"]]
    news = pd.read_parquet(news_parquet)[["ticker", "day", "has_sent", *NEWS_FEATS]]
    red = pd.read_parquet(reddit_parquet)[["ticker", "day", "n_scored", *REDDIT_FEATS]]
    red["day"], news["day"] = pd.to_datetime(red["day"]), pd.to_datetime(news["day"])
    df = a.merge(news, on=["ticker", "day"], how="left", validate="one_to_one")
    df = df.merge(red, on=["ticker", "day"], how="left", validate="one_to_one")
    df = df[df["day"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
    if df["n_scored"].isna().any():
        raise ValueError("Reddit 피처가 없는 행이 있다 — 구간을 확인할 것")
    return df.sort_values(["day", "ticker"]).reset_index(drop=True)


def build_features(df: pd.DataFrame, sets: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """z 점수를 ±CLIP으로 자르고 결측은 0으로 채운다(+결측 지표). 반환: 피처 표, 피처 집합별 열 이름."""
    out = df[["day", "ticker", "real_ret", "pred_ret", "residual"]].copy()
    for c in NEWS_FEATS + REDDIT_FEATS:
        out[f"{c}_na"] = df[c].isna().astype("float32")
        out[c] = df[c].clip(-CLIP, CLIP).fillna(0.0)
    out["log_n_scored"] = np.log1p(df["n_scored"])
    news = NEWS_FEATS + ["sent_z_na", "sent_z_primary_na"]
    reddit = REDDIT_FEATS + ["log_n_scored", "sent_mean_z_na"]
    return out, {"news": news, "reddit": reddit, "both": news + reddit}


def make_models(seed: int = 0) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9))),
        "xgboost": lambda: XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.7, colsample_bytree=0.8,
                                        min_child_weight=50, reg_lambda=10.0, random_state=seed, n_jobs=4),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-1, learning_rate_init=1e-3, max_iter=200,
                                                                    early_stopping=True, validation_fraction=0.15,
                                                                    n_iter_no_change=10, random_state=seed)),
    }


def daily_ic(frame: pd.DataFrame, pred: str, target: str) -> pd.Series:
    """날짜별 횡단면 Spearman IC."""
    def one(g):
        return g[pred].rank().corr(g[target].rank()) if len(g) > 2 else np.nan
    return frame.groupby("day").apply(one, include_groups=False).dropna()


def ic_summary(ic: pd.Series, horizon: int = 3) -> dict:
    """평균 IC와 t값. 3일 수익률이 겹치므로 독립 표본 수를 n/horizon 으로 줄여 계산한다."""
    n_eff = max(len(ic) / horizon, 1.0)
    se = ic.std(ddof=1) / np.sqrt(n_eff)
    return {"ic": float(ic.mean()), "t": float(ic.mean() / se) if se > 0 else np.nan, "n_days": int(len(ic))}


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_folds(days: pd.Series, n_folds: int = 5, first_test_frac: float = 0.4, purge: int = 3) -> list[Fold]:
    ud = np.sort(days.unique())
    start = int(len(ud) * first_test_frac)
    edges = np.linspace(start, len(ud), n_folds + 1).astype(int)
    folds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        train_days, test_days = ud[: max(lo - purge, 0)], ud[lo:hi]
        folds.append(Fold(np.flatnonzero(days.isin(train_days)), np.flatnonzero(days.isin(test_days))))
    return folds


def run(feat: pd.DataFrame, sets: dict[str, list[str]], seed: int = 0, n_folds: int = 5, purge: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    models, folds = make_models(seed), walk_forward_folds(feat["day"], n_folds, purge=purge)
    preds = feat[["day", "ticker", "real_ret", "pred_ret", "residual"]].copy()
    test_mask = np.zeros(len(feat), dtype=bool)
    for f in folds:
        test_mask[f.test_idx] = True
    rows = []
    preds["const"] = np.nan   # 학습 구간 ε 평균만 쓰는 기준선(상수 편향만 맞춘 효과를 분리하기 위함)
    for f in folds:
        preds.loc[feat.index[f.test_idx], "const"] = feat.iloc[f.train_idx]["residual"].mean()
    c_sse = ((preds.loc[test_mask, "residual"] - preds.loc[test_mask, "const"]) ** 2).sum()
    for set_name, cols in sets.items():
        for m_name, factory in models.items():
            col = f"{m_name}__{set_name}"
            preds[col] = np.nan
            for f in folds:
                model = factory()
                model.fit(feat.iloc[f.train_idx][cols].to_numpy("float32"), feat.iloc[f.train_idx]["residual"].to_numpy())
                preds.loc[feat.index[f.test_idx], col] = model.predict(feat.iloc[f.test_idx][cols].to_numpy("float32"))
            t = preds[test_mask].copy()
            t["final"] = t["pred_ret"] + t[col]
            e, f_ = ic_summary(daily_ic(t, col, "residual")), ic_summary(daily_ic(t, "final", "real_ret"))
            sse, base = ((t["residual"] - t[col]) ** 2).sum(), (t["residual"] ** 2).sum()
            r2_vs_const = 1 - sse / c_sse
            rows.append({"features": set_name, "model": m_name, "ic_eps": e["ic"], "t_eps": e["t"],
                         "ic_final": f_["ic"], "t_final": f_["t"], "r2_oos_eps": 1 - sse / base, "r2_vs_const": r2_vs_const, "n_test_days": e["n_days"]})
    base_t = preds[test_mask]
    a_only = ic_summary(daily_ic(base_t, "pred_ret", "real_ret"))
    rows.append({"features": "(A only)", "model": "chronos-2", "ic_eps": np.nan, "t_eps": np.nan, "ic_final": a_only["ic"],
                 "t_final": a_only["t"], "r2_oos_eps": 0.0, "r2_vs_const": np.nan, "n_test_days": a_only["n_days"]})
    rows.append({"features": "(const ε̄)", "model": "train mean", "r2_oos_eps": 1 - c_sse / (preds.loc[test_mask, "residual"] ** 2).sum(),
                 "r2_vs_const": 0.0, "n_test_days": a_only["n_days"]})
    return pd.DataFrame(rows), preds
