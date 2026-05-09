"""Distribution drift monitoring (PSI + KS)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index."""
    breakpoints = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        return float("nan")
    ref_hist, _ = np.histogram(reference, bins=breakpoints)
    cur_hist, _ = np.histogram(current, bins=breakpoints)
    ref_p = ref_hist / max(ref_hist.sum(), 1)
    cur_p = cur_hist / max(cur_hist.sum(), 1)
    eps = 1e-6
    return float(np.sum((cur_p - ref_p) * np.log((cur_p + eps) / (ref_p + eps))))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    stat, p = ks_2samp(reference, current)
    return float(stat), float(p)


def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in reference.columns:
        if col not in current.columns:
            continue
        r = reference[col].dropna().values
        c = current[col].dropna().values
        if len(r) < 30 or len(c) < 30:
            continue
        rows.append({
            "feature": col,
            "psi": psi(r, c),
            "ks": ks_drift(r, c)[0],
            "ks_pvalue": ks_drift(r, c)[1],
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
