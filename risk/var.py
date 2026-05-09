"""Value-at-Risk and Conditional VaR estimators."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def historical_var(returns: np.ndarray, confidence: float = 0.99) -> float:
    return float(-np.quantile(returns, 1 - confidence))


def parametric_var(returns: np.ndarray, confidence: float = 0.99) -> float:
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1) or 1e-12)
    return float(-(mu + sd * norm.ppf(1 - confidence)))


def cvar(returns: np.ndarray, confidence: float = 0.975) -> float:
    var = -np.quantile(returns, 1 - confidence)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) else float(var)
