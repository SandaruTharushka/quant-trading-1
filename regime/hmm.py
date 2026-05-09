"""Gaussian HMM-based regime classifier.

Inputs are typically [returns, abs(returns), realized_vol]. Hidden states are
labeled by mean-return and mean-vol so they map to {trending up, trending down,
mean-reverting low-vol, high-vol crisis, ...} interpretably.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HMMRegime:
    n_states: int = 4
    cov_type: str = "full"
    n_iter: int = 200
    seed: int = 1337
    _model: object | None = None
    _state_labels: dict[int, str] | None = None

    def fit(self, X: pd.DataFrame) -> "HMMRegime":
        from hmmlearn.hmm import GaussianHMM
        m = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.cov_type,
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        m.fit(X.values)
        self._model = m
        self._state_labels = self._label_states(m, X)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("HMMRegime not fitted")
        states = self._model.predict(X.values)
        labels = [self._state_labels[s] for s in states]  # type: ignore[index]
        return pd.Series(labels, index=X.index, name="regime")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("HMMRegime not fitted")
        proba = self._model.predict_proba(X.values)
        cols = [self._state_labels[i] for i in range(self.n_states)]  # type: ignore[index]
        return pd.DataFrame(proba, index=X.index, columns=cols)

    @staticmethod
    def _label_states(model, X: pd.DataFrame) -> dict[int, str]:
        means = np.asarray(model.means_)
        # rank states by vol (assume column 0 = return, last column = vol-like proxy)
        vol_proxy = means[:, -1]
        ret = means[:, 0]
        order_vol = np.argsort(vol_proxy)
        labels: dict[int, str] = {}
        for rank, s in enumerate(order_vol):
            if rank == 0:
                base = "low_vol"
            elif rank == len(order_vol) - 1:
                base = "crisis"
            elif rank == 1:
                base = "calm"
            else:
                base = "high_vol"
            direction = "trend_up" if ret[s] > 0.0 else "trend_dn" if ret[s] < 0.0 else "neutral"
            labels[int(s)] = f"{base}_{direction}"
        return labels
