import numpy as np
import pandas as pd

from features.microstructure import (
    queue_imbalance, order_flow_imbalance, kyles_lambda,
    realized_volatility, parkinson_volatility,
)


def test_queue_imbalance_in_bounds(rng):
    bid = pd.Series(rng.integers(1, 100, size=200).astype(float))
    ask = pd.Series(rng.integers(1, 100, size=200).astype(float))
    qi = queue_imbalance(bid, ask)
    assert qi.between(-1, 1).all()


def test_ofi_zero_when_static():
    # No price or size moves -> OFI must be 0
    bid_size = pd.Series([10.0] * 50)
    ask_size = pd.Series([10.0] * 50)
    bid_price = pd.Series([99.0] * 50)
    ask_price = pd.Series([100.0] * 50)
    out = order_flow_imbalance(bid_size, ask_size, bid_price, ask_price)
    assert out.fillna(0.0).abs().max() == 0.0


def test_kyles_lambda_positive_with_impact(rng):
    n = 500
    q = pd.Series(rng.standard_normal(n))
    # price move = 0.5*q + noise -> Kyle's lambda should be ~0.5
    dp = 0.5 * q + 0.1 * pd.Series(rng.standard_normal(n))
    lam = kyles_lambda(dp, q, window=200).dropna()
    assert lam.mean() > 0.2


def test_realized_vol_matches_std():
    r = pd.Series(np.array([0.01, -0.02, 0.015, -0.005, 0.0] * 20))
    rv = realized_volatility(r, window=10).dropna()
    assert (rv > 0).all()


def test_parkinson_high_low_only():
    n = 200
    h = pd.Series(np.linspace(101, 110, n))
    l = pd.Series(np.linspace(99, 108, n))
    pv = parkinson_volatility(h, l, window=20).dropna()
    assert (pv > 0).all()
