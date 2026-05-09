"""Walk-forward, purged, and combinatorial-purged cross-validation.

These splitters operate on labels with overlapping evaluation horizons (e.g.
triple-barrier), so they require both the *event* index and the *t1* (event
end) for each label.

Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018), Ch. 7.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd


def _purge(train_idx: np.ndarray, t1: pd.Series, test_t0: pd.Timestamp,
           test_t1: pd.Timestamp) -> np.ndarray:
    """Drop training samples whose evaluation window overlaps the test window."""
    train_t0 = t1.index[train_idx]
    train_t1 = t1.iloc[train_idx]
    overlap = ~((train_t1 < test_t0) | (train_t0 > test_t1))
    return train_idx[~overlap]


def _embargo(train_idx: np.ndarray, idx: pd.DatetimeIndex,
             test_t1: pd.Timestamp, embargo: pd.Timedelta) -> np.ndarray:
    """Remove training samples that start inside the post-test embargo window."""
    if embargo.total_seconds() <= 0:
        return train_idx
    starts = idx[train_idx]
    keep = ~((starts > test_t1) & (starts <= test_t1 + embargo))
    return train_idx[keep]


@dataclass
class PurgedKFold:
    """K-fold split with purging only (no embargo)."""

    n_splits: int = 5

    def split(self, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(t1)
        idx_all = np.arange(n)
        folds = np.array_split(idx_all, self.n_splits)
        for test_idx in folds:
            test_t0 = t1.index[test_idx[0]]
            test_t1 = t1.iloc[test_idx[-1]]
            train_idx = np.setdiff1d(idx_all, test_idx)
            train_idx = _purge(train_idx, t1, test_t0, test_t1)
            yield train_idx, test_idx


@dataclass
class EmbargoedKFold:
    """Purged K-fold with a post-test embargo (fraction of total length)."""

    n_splits: int = 5
    embargo_pct: float = 0.01

    def split(self, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(t1)
        idx_all = np.arange(n)
        folds = np.array_split(idx_all, self.n_splits)
        if isinstance(t1.index, pd.DatetimeIndex):
            embargo_td = (t1.index[-1] - t1.index[0]) * self.embargo_pct
        else:
            embargo_td = pd.Timedelta(0)
        for test_idx in folds:
            test_t0 = t1.index[test_idx[0]]
            test_t1 = t1.iloc[test_idx[-1]]
            train_idx = np.setdiff1d(idx_all, test_idx)
            train_idx = _purge(train_idx, t1, test_t0, test_t1)
            train_idx = _embargo(train_idx, t1.index, test_t1, embargo_td)
            yield train_idx, test_idx


@dataclass
class CombinatorialPurgedCV:
    """CPCV (Lopez de Prado): splits the timeline into N groups and tests on every
    k-subset, producing C(N, k) backtest paths from the same data, all purged
    and embargoed against leakage."""

    n_groups: int = 6
    n_test_groups: int = 2
    embargo_pct: float = 0.01

    def split(self, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx_all = np.arange(len(t1))
        groups = np.array_split(idx_all, self.n_groups)
        if isinstance(t1.index, pd.DatetimeIndex):
            embargo_td = (t1.index[-1] - t1.index[0]) * self.embargo_pct
        else:
            embargo_td = pd.Timedelta(0)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[i] for i in combo])
            test_idx.sort()
            test_t0 = t1.index[test_idx[0]]
            test_t1 = t1.iloc[test_idx[-1]]
            train_idx = np.setdiff1d(idx_all, test_idx)
            train_idx = _purge(train_idx, t1, test_t0, test_t1)
            train_idx = _embargo(train_idx, t1.index, test_t1, embargo_td)
            yield train_idx, test_idx

    def n_paths(self) -> int:
        from math import comb
        return comb(self.n_groups - 1, self.n_test_groups - 1)


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_size: pd.Timedelta,
    test_size: pd.Timedelta,
    step: pd.Timedelta | None = None,
    embargo: pd.Timedelta = pd.Timedelta(0),
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Anchored / rolling walk-forward splitter."""
    step = step or test_size
    start = index[0]
    end = index[-1]

    train_end = start + train_size
    while train_end + test_size <= end:
        test_start = train_end + embargo
        test_end = test_start + test_size
        train_mask = (index >= start) & (index < train_end)
        test_mask = (index >= test_start) & (index < test_end)
        yield np.where(train_mask)[0], np.where(test_mask)[0]
        train_end += step
