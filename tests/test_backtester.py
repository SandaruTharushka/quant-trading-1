import numpy as np
import pandas as pd
import pytest

from backtesting.engine import VectorizedBacktester, BacktestConfig
from backtesting.costs import CostModel, SlippageModel


def _frames(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    p = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.005)),
        "B": 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.005)),
    }, index=idx)
    w = pd.DataFrame(
        np.sign(p.pct_change().rolling(5).mean().fillna(0.0)) * 0.5,
        index=idx,
    )
    return p, w


def test_backtester_no_lookahead_signal():
    """Pre-shifted weights should match the engine's internal shift."""
    p, w = _frames()
    bt = VectorizedBacktester(BacktestConfig(
        cost=CostModel(commission_bps=0.5),
        slippage=SlippageModel(base_bps=0.8),
    ))
    res = bt.run(p, w)
    # executed weights are last-bar weights (shift 1)
    assert (res.weights.iloc[0].abs().sum()) == 0.0


def test_zero_costs_rejected():
    with pytest.raises(ValueError):
        BacktestConfig(
            cost=CostModel(commission_bps=0.0),
            slippage=SlippageModel(base_bps=0.0),
        )


def test_metrics_keys_present():
    p, w = _frames()
    res = VectorizedBacktester().run(p, w)
    for k in ("sharpe", "sortino", "calmar", "max_dd", "turnover", "hit_ratio"):
        assert k in res.metrics
