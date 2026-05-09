"""Portfolio sizing & optimization.

All routines operate on either a single asset (vol target, Kelly fraction) or
on a covariance matrix (risk parity, CVaR). Outputs are *normalized* portfolio
weights subject to the platform's risk limits — they are not yet leverage-applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------- Vol targeting ----------

def vol_target_weights(
    raw_signal: pd.Series,
    realized_vol: pd.Series,
    vol_target_annual: float = 0.10,
    bars_per_year: int = 252,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Scale signal so realized vol matches target. Position is sign(signal) * gain."""
    target_per_bar = vol_target_annual / np.sqrt(bars_per_year)
    gain = (target_per_bar / realized_vol.replace(0, np.nan)).clip(upper=max_leverage)
    return (raw_signal.clip(-1, 1) * gain).fillna(0.0)


# ---------- Kelly ----------

def kelly_weights(mu: pd.Series, sigma: pd.Series, fraction: float = 0.25,
                  max_leverage: float = 3.0) -> pd.Series:
    """Fractional Kelly sizing (per-asset, independent)."""
    var = sigma.pow(2).replace(0, np.nan)
    raw = mu / var
    return (fraction * raw).fillna(0.0).clip(-max_leverage, max_leverage)


# ---------- Risk parity ----------

def risk_parity_weights(cov: np.ndarray, max_iter: int = 1000, tol: float = 1e-9) -> np.ndarray:
    """Equal risk contribution (long-only). Newton-style fixed-point iteration."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_vol = np.sqrt(w @ cov @ w)
        marg = cov @ w / max(port_vol, 1e-12)
        rc = w * marg
        target = port_vol / n
        diff = rc - target
        if np.max(np.abs(diff)) < tol:
            break
        w = w * (target / np.maximum(rc, 1e-12)) ** 0.5
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
    return w


# ---------- CVaR ----------

def cvar_weights(
    returns: np.ndarray,
    alpha: float = 0.95,
    max_leverage: float = 1.0,
    long_only: bool = False,
) -> np.ndarray:
    """Min-CVaR portfolio (Rockafellar-Uryasev) using SciPy SLSQP.

    `returns` is a [T, N] matrix of historical asset returns.
    """
    T, N = returns.shape

    def cvar(w: np.ndarray) -> float:
        port = returns @ w
        var_q = np.quantile(port, 1 - alpha)
        tail = port[port <= var_q]
        if len(tail) == 0:
            return -var_q
        return -tail.mean()

    bounds = [(0.0, max_leverage) if long_only else (-max_leverage, max_leverage)] * N
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.ones(N) / N
    res = minimize(cvar, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    return res.x if res.success else w0


# ---------- Correlation-aware sizing ----------

def correlation_aware_size(
    raw_weights: pd.Series,
    cov: pd.DataFrame,
    target_vol: float = 0.10,
    bars_per_year: int = 252,
) -> pd.Series:
    """Re-normalize raw signals so the *portfolio* hits a target vol — accounting for cov."""
    w = raw_weights.values
    pv = float(np.sqrt(w @ cov.values @ w))
    if pv <= 0:
        return raw_weights * 0
    target_per_bar = target_vol / np.sqrt(bars_per_year)
    gain = target_per_bar / pv
    return raw_weights * gain
