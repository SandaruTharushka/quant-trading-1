"""Small research signal functions shared by scripts and the dashboard API."""
from __future__ import annotations

import numpy as np
import pandas as pd


def analyze(
    close: pd.Series,
    lookback: int = 24,
    long_threshold: float = 0.5,
    short_threshold: float = -0.5,
) -> tuple[pd.Timestamp, float, str]:
    returns = close.pct_change()
    momentum = np.log(close).diff().rolling(lookback).sum()
    volatility = returns.rolling(lookback).std() * np.sqrt(lookback)
    score = (momentum / volatility).replace([np.inf, -np.inf], np.nan).dropna()
    if score.empty:
        raise ValueError("Not enough valid candles to calculate a signal.")

    latest_score = float(score.iloc[-1])
    if latest_score > long_threshold:
        label = "LONG candidate"
    elif latest_score < short_threshold:
        label = "SHORT candidate"
    else:
        label = "FLAT"
    return score.index[-1], latest_score, label