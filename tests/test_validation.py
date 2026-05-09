import numpy as np
import pandas as pd

from research.validation import (
    PurgedKFold,
    EmbargoedKFold,
    CombinatorialPurgedCV,
    walk_forward_splits,
)


def _t1(n=200):
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    # each label evaluates 5 bars forward
    t1 = idx + pd.Timedelta(hours=5)
    return pd.Series(t1, index=idx)


def test_purged_kfold_no_overlap():
    t1 = _t1()
    splitter = PurgedKFold(n_splits=4)
    for train_idx, test_idx in splitter.split(t1):
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        # Purging: no training label may end inside the test window
        test_t0 = t1.index[test_idx[0]]
        test_t1 = t1.iloc[test_idx[-1]]
        train_ends = t1.iloc[train_idx]
        assert ((train_ends < test_t0) | (t1.index[train_idx] > test_t1)).all()


def test_embargoed_purged_drops_post_test_neighbors():
    t1 = _t1()
    splitter = EmbargoedKFold(n_splits=4, embargo_pct=0.05)
    for train_idx, test_idx in splitter.split(t1):
        test_end = t1.iloc[test_idx[-1]]
        # train samples just after the test window must be dropped
        train_starts = t1.index[train_idx]
        delta = (test_end - test_end) + (t1.index[-1] - t1.index[0]) * 0.05
        in_embargo = (train_starts > test_end) & (train_starts <= test_end + delta)
        assert not in_embargo.any()


def test_cpcv_path_count():
    t1 = _t1()
    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
    splits = list(cpcv.split(t1))
    # C(6, 2) = 15 distinct train/test combinations
    assert len(splits) == 15
    # Each backtest path is built from C(N-1, k-1)=5 partitions
    assert cpcv.n_paths() == 5


def test_walk_forward_no_overlap():
    idx = pd.date_range("2022-01-01", periods=500, freq="h", tz="UTC")
    splits = list(walk_forward_splits(
        idx, train_size=pd.Timedelta(hours=200), test_size=pd.Timedelta(hours=50),
        embargo=pd.Timedelta(hours=5),
    ))
    assert len(splits) > 1
    for tr, te in splits:
        assert idx[tr].max() < idx[te].min()
