"""Performance metrics, including the Deflated Sharpe Ratio and PBO."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------- Standard performance ----------

def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=1))


def sortino(returns: pd.Series, periods_per_year: int = 252, mar: float = 0.0) -> float:
    excess = returns - mar / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd_std = np.sqrt((downside ** 2).mean())
    if dd_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / dd_std)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    eq = (1 + returns).cumprod()
    mdd = abs(max_drawdown(eq))
    if mdd == 0:
        return 0.0
    cagr = eq.iloc[-1] ** (periods_per_year / len(returns)) - 1
    return float(cagr / mdd)


def turnover(weights: pd.DataFrame) -> float:
    return float(weights.diff().abs().sum(axis=1).mean())


# ---------- Deflated Sharpe Ratio (Bailey & Lopez de Prado) ----------

def deflated_sharpe(
    sr_observed: float,
    n_returns: int,
    skew: float,
    kurt: float,
    n_trials: int,
    sr_variance_across_trials: float,
) -> float:
    """Probability that the observed SR exceeds the expected maximum SR
    achievable by chance under the null hypothesis."""
    sr0 = np.sqrt(sr_variance_across_trials) * (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + np.euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    num = (sr_observed - sr0) * np.sqrt(n_returns - 1)
    den = np.sqrt(1 - skew * sr_observed + (kurt - 1) / 4 * sr_observed ** 2)
    if den == 0:
        return float("nan")
    z = num / den
    return float(stats.norm.cdf(z))


def probability_of_backtest_overfitting(M: np.ndarray) -> float:
    """PBO via combinatorial-symmetric-cross-validation on a (T x N) matrix of
    per-period returns for N strategies. Returns probability the in-sample best
    strategy is below median out-of-sample."""
    T, N = M.shape
    if T < 4 or N < 2:
        return float("nan")
    s = T - (T % 2)
    M = M[:s]
    halves = np.array_split(np.arange(s), 2)
    losses = 0
    total = 0
    from itertools import combinations
    for in_sample in combinations(range(2), 1):
        is_idx = np.concatenate([halves[i] for i in in_sample])
        oos_idx = np.setdiff1d(np.arange(s), is_idx)
        is_perf = M[is_idx].mean(axis=0)
        oos_perf = M[oos_idx].mean(axis=0)
        n_star = int(np.argmax(is_perf))
        # rank of n_star OOS
        ranks_oos = (-oos_perf).argsort().argsort()
        rel_rank = (ranks_oos[n_star] + 1) / (N + 1)
        if rel_rank > 0.5:
            losses += 1
        total += 1
    return losses / total if total else float("nan")
