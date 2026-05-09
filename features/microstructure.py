"""Market-microstructure features.

All functions are causal — no lookahead. Each accepts pandas inputs and returns
pandas Series so they compose cleanly with the feature pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Order flow & book ----------

def order_flow_imbalance(
    bid_size: pd.Series,
    ask_size: pd.Series,
    bid_price: pd.Series,
    ask_price: pd.Series,
) -> pd.Series:
    """Cont, Kukanov & Stoikov (2014) Order Flow Imbalance.

    OFI_t = e_t^b - e_t^a
    where e^b incorporates price moves at the best bid (similarly for asks).
    """
    bp_diff = bid_price.diff()
    ap_diff = ask_price.diff()

    e_b = pd.Series(0.0, index=bid_size.index)
    e_b[bp_diff > 0] = bid_size[bp_diff > 0]
    e_b[bp_diff == 0] = bid_size[bp_diff == 0].diff().fillna(0.0)
    e_b[bp_diff < 0] = -bid_size.shift(1)[bp_diff < 0]

    e_a = pd.Series(0.0, index=ask_size.index)
    e_a[ap_diff < 0] = ask_size[ap_diff < 0]
    e_a[ap_diff == 0] = ask_size[ap_diff == 0].diff().fillna(0.0)
    e_a[ap_diff > 0] = -ask_size.shift(1)[ap_diff > 0]

    return (e_b - e_a).rename("ofi")


def book_pressure(bid_sizes: pd.DataFrame, ask_sizes: pd.DataFrame) -> pd.Series:
    """Multi-level bid/ask pressure: log(sum_bid_size / sum_ask_size)."""
    b = bid_sizes.sum(axis=1).clip(lower=1e-9)
    a = ask_sizes.sum(axis=1).clip(lower=1e-9)
    return np.log(b / a).rename("book_pressure")


def queue_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """Top-of-book queue imbalance in [-1, 1]."""
    return ((bid_size - ask_size) / (bid_size + ask_size).clip(lower=1e-9)).rename("qi")


def vwap_deviation(price: pd.Series, volume: pd.Series, window: int = 100) -> pd.Series:
    """Rolling VWAP and deviation of price from it (z-scored by rolling stdev)."""
    pv = (price * volume).rolling(window).sum()
    v = volume.rolling(window).sum().clip(lower=1e-9)
    vwap = pv / v
    dev = price - vwap
    z = dev / dev.rolling(window).std().replace(0, np.nan)
    return z.rename("vwap_z")


def kyles_lambda(price_change: pd.Series, signed_volume: pd.Series, window: int = 200) -> pd.Series:
    """Rolling Kyle's lambda — price impact per unit signed volume.

    Estimated as Cov(dp, q) / Var(q) on a rolling window.
    """
    cov = price_change.rolling(window).cov(signed_volume)
    var = signed_volume.rolling(window).var()
    return (cov / var.replace(0, np.nan)).rename("kyle_lambda")


# ---------- Realized vol estimators ----------

def realized_volatility(returns: pd.Series, window: int = 100) -> pd.Series:
    return returns.rolling(window).std().rename("rv")


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    rs = (np.log(high / low) ** 2) / (4 * np.log(2))
    return np.sqrt(rs.rolling(window).mean()).rename("parkinson")


def garman_klass_volatility(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int = 20
) -> pd.Series:
    rs = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(rs.rolling(window).mean()).rename("gk")


def yang_zhang_volatility(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int = 20
) -> pd.Series:
    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)

    sigma_o = log_oc.rolling(window).var()
    sigma_c = log_co.rolling(window).var()
    rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt(sigma_o + k * sigma_c + (1 - k) * rs).rename("yz")


# ---------- Spread / liquidity ----------

def spread_dynamics(bid: pd.Series, ask: pd.Series, window: int = 100) -> pd.DataFrame:
    spread = (ask - bid).rename("spread")
    rel = (spread / ((bid + ask) / 2)).rename("rel_spread")
    return pd.DataFrame({
        "spread": spread,
        "rel_spread": rel,
        "spread_z": (spread - spread.rolling(window).mean()) /
                    spread.rolling(window).std().replace(0, np.nan),
    })


def liquidity_sweep(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """Detect sweeps of recent highs/lows that reverse — proxy for stop-runs."""
    prior_high = high.shift(1).rolling(lookback).max()
    prior_low = low.shift(1).rolling(lookback).min()
    sweep_up = ((high > prior_high) & (close < prior_high)).astype(int)
    sweep_dn = ((low < prior_low) & (close > prior_low)).astype(int)
    return pd.DataFrame({"sweep_up": sweep_up, "sweep_dn": sweep_dn})
