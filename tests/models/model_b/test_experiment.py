import numpy as np
import pandas as pd

from capstone.models.model_b.experiment import daily_ic, ic_summary, walk_forward_folds


def _days(n_days=100, n_tickers=3):
    d = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.Series(np.repeat(d, n_tickers))


def test_walk_forward_purges_and_never_looks_ahead():
    days = _days()
    folds = walk_forward_folds(days, n_folds=5, first_test_frac=0.4, purge=3)
    assert len(folds) == 5
    ud = np.sort(days.unique())
    for f in folds:
        train_max, test_min = days.iloc[f.train_idx].max(), days.iloc[f.test_idx].min()
        assert train_max < test_min
        assert (np.flatnonzero(ud == test_min)[0] - np.flatnonzero(ud == train_max)[0]) > 3   # purge: 3거래일 이상 간격
        assert not set(f.train_idx) & set(f.test_idx)
    tests = np.concatenate([f.test_idx for f in folds])
    assert len(tests) == len(set(tests))   # 테스트 블록이 겹치지 않는다


def test_daily_ic_perfect_and_inverse():
    df = pd.DataFrame({"day": np.repeat(pd.bdate_range("2021-01-04", periods=5), 4),
                       "p": np.tile([1.0, 2, 3, 4], 5), "y": np.tile([10.0, 20, 30, 40], 5)})
    assert np.allclose(daily_ic(df, "p", "y"), 1.0)
    df["y"] = -df["y"]
    assert np.allclose(daily_ic(df, "p", "y"), -1.0)


def test_ic_summary_discounts_overlap():
    ic = pd.Series(np.random.default_rng(0).normal(0.05, 0.3, 300))
    s3, s1 = ic_summary(ic, horizon=3), ic_summary(ic, horizon=1)
    assert s3["t"] < s1["t"] and s3["n_days"] == 300
