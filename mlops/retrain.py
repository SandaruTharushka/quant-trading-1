"""Walk-forward retraining loop driver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from research.validation import walk_forward_splits


@dataclass
class WalkForwardRetrainer:
    train_size: pd.Timedelta
    test_size: pd.Timedelta
    embargo: pd.Timedelta = pd.Timedelta(0)

    def run(
        self,
        index: pd.DatetimeIndex,
        train_fn: Callable[[pd.DatetimeIndex, pd.DatetimeIndex], dict],
    ) -> list[dict]:
        results: list[dict] = []
        for train_idx, test_idx in walk_forward_splits(
            index, self.train_size, self.test_size, embargo=self.embargo
        ):
            train_ts = index[train_idx]
            test_ts = index[test_idx]
            results.append(train_fn(train_ts, test_ts))
        return results
