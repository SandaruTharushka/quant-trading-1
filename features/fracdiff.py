"""Fractional differentiation with fixed-window weights (Lopez de Prado).

The standard integer differencing destroys memory; fractional diff with the
smallest d that achieves stationarity preserves predictive content while
yielding stationary inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def _ffd_weights(d: float, thres: float = 1e-5) -> np.ndarray:
    """Fixed-window fractional-diff weights — drop those below `thres`."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
        if k > 10_000:
            break
    return np.array(w[::-1])  # newest-first when convolved


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """Apply fixed-width fractional differencing to a 1D price series."""
    w = _ffd_weights(d, thres)
    width = len(w)
    out = np.full(len(series), np.nan)
    vals = series.values.astype(float)
    for i in range(width - 1, len(vals)):
        window = vals[i - width + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(w, window))
    return pd.Series(out, index=series.index, name=f"ffd_{d:.2f}")


def find_min_d(
    series: pd.Series,
    d_grid: np.ndarray | None = None,
    thres: float = 1e-5,
    pvalue: float = 0.05,
) -> tuple[float, float]:
    """Smallest d in `d_grid` giving an ADF p-value below `pvalue`. Returns (d, pvalue)."""
    if d_grid is None:
        d_grid = np.linspace(0.0, 1.0, 11)
    log_p = np.log(series.dropna().astype(float))
    for d in d_grid:
        diffed = frac_diff_ffd(log_p, d, thres).dropna()
        if len(diffed) < 50:
            continue
        try:
            p = adfuller(diffed, maxlag=1, regression="c", autolag=None)[1]
        except Exception:
            continue
        if p < pvalue:
            return float(d), float(p)
    return float(d_grid[-1]), float("nan")
