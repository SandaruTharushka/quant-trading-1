"""Changepoint detection: CUSUM and a lightweight BOCPD variant."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_changepoints(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on standardized returns."""
    r = series.diff().fillna(0.0).values
    sd = np.std(r) or 1.0
    r = r / sd
    pos = neg = 0.0
    out: list[pd.Timestamp] = []
    for i, x in enumerate(r):
        pos = max(0.0, pos + x)
        neg = min(0.0, neg + x)
        if pos > threshold:
            out.append(series.index[i])
            pos = 0.0
        elif neg < -threshold:
            out.append(series.index[i])
            neg = 0.0
    return pd.DatetimeIndex(out)


def bocpd_changepoints(series: pd.Series, hazard: float = 1.0 / 250.0) -> pd.Series:
    """Bayesian online change-point detection (Adams & MacKay 2007), Gaussian observation.

    Returns the run-length posterior mode at each step — a high value means
    'no recent regime shift'; a drop to ~0 marks a changepoint.
    """
    x = series.values.astype(float)
    T = len(x)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Prior NIG hyperparameters
    mu0 = float(np.mean(x))
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = 1.0

    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    run_modes = np.zeros(T, dtype=int)

    for t, xt in enumerate(x):
        # Predictive prob (Student-t)
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        from scipy.stats import t as student_t
        pred = student_t.pdf(xt, df=df, loc=mu, scale=scale)

        # Growth probabilities
        growth = R[: t + 1, t] * pred * (1 - hazard)
        # Changepoint probability
        cp = np.sum(R[: t + 1, t] * pred * hazard)

        R[1 : t + 2, t + 1] = growth
        R[0, t + 1] = cp
        Z = R[: t + 2, t + 1].sum() or 1.0
        R[: t + 2, t + 1] /= Z

        # Update sufficient stats and append fresh prior at index 0
        new_mu = (kappa * mu + xt) / (kappa + 1)
        new_kappa = kappa + 1
        new_alpha = alpha + 0.5
        new_beta = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], new_mu])
        kappa = np.concatenate([[kappa0], new_kappa])
        alpha = np.concatenate([[alpha0], new_alpha])
        beta = np.concatenate([[beta0], new_beta])

        run_modes[t] = int(np.argmax(R[: t + 2, t + 1]))

    return pd.Series(run_modes, index=series.index, name="run_length")
