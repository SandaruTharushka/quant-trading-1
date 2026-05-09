import numpy as np
import pandas as pd

from portfolio.optimizer import vol_target_weights, risk_parity_weights, kelly_weights
from risk.var import historical_var, parametric_var, cvar
from risk.engine import RiskEngine, RiskLimits, BreachAction


def test_vol_target_caps_leverage():
    n = 100
    sig = pd.Series(np.ones(n))
    rv = pd.Series(np.full(n, 1e-6))   # tiny vol -> would blow up
    w = vol_target_weights(sig, rv, vol_target_annual=0.10, bars_per_year=252,
                           max_leverage=2.0)
    assert w.abs().max() <= 2.0 + 1e-9


def test_risk_parity_equal_risk_contributions():
    cov = np.array([[1.0, 0.2, 0.1],
                    [0.2, 0.5, 0.0],
                    [0.1, 0.0, 0.3]])
    w = risk_parity_weights(cov)
    port_vol = float(np.sqrt(w @ cov @ w))
    rc = w * (cov @ w) / port_vol
    rc /= rc.sum()
    assert np.allclose(rc, 1 / 3, atol=1e-2)


def test_kelly_signed():
    mu = pd.Series([0.001, -0.002])
    sd = pd.Series([0.01, 0.02])
    w = kelly_weights(mu, sd, fraction=0.25)
    assert w.iloc[0] > 0 and w.iloc[1] < 0


def test_historical_var_sane(rng):
    r = rng.standard_normal(10_000) * 0.01
    v = historical_var(r, 0.99)
    assert 0.015 < v < 0.04


def test_cvar_at_least_var(rng):
    r = rng.standard_normal(5_000) * 0.01
    v = historical_var(r, 0.975)
    c = cvar(r, 0.975)
    assert c >= v - 1e-9


def test_risk_engine_drawdown_kill():
    eng = RiskEngine(limits=RiskLimits(max_drawdown=0.10))
    eng.reset_day(1.0)
    eng.update(1.0, pd.Series({"X": 0.5}), np.zeros(40))
    breaches = eng.update(0.85, pd.Series({"X": 0.5}), np.zeros(40))
    assert any(b.action == BreachAction.KILL for b in breaches)
    assert eng.killed
