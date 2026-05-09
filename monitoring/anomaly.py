"""Anomaly detection on returns / signals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def isolation_forest_score(X: pd.DataFrame, contamination: float = 0.01) -> pd.Series:
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=contamination, random_state=1337)
    iso.fit(X.values)
    return pd.Series(-iso.score_samples(X.values), index=X.index, name="anomaly_score")
