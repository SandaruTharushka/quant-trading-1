import numpy as np
import pandas as pd
import pytest

from core.config import RiskSettings
from research.live_research import ForwardPaperLedger, build_research_report


def test_research_report_uses_holdout_and_capped_portfolio():
    rng = np.random.default_rng(19)
    close = 1800 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, 900)))
    index = pd.date_range("2025-01-01", periods=len(close), freq="h", tz="UTC")
    report = build_research_report(pd.DataFrame({"c": close}, index=index))

    assert report["train_bars"] + report["holdout_bars"] == report["bars"] - 73
    assert report["backtest"]["model"]["cum_return"] is not None
    assert report["model"]["name"] == "Logistic Regression"
    assert abs(report["portfolio"]["position_weight"]) <= 0.25


def test_forward_ledger_tracks_bar_and_risk_flatten():
    ledger = ForwardPaperLedger()
    start = pd.Timestamp("2026-09-20T10:00:00Z")
    initial = ledger.update("XAUUSDT", start, 100.0, 0.25)
    updated = ledger.update("XAUUSDT", start + pd.Timedelta(hours=1), 70.0, 0.25)

    assert initial["bars"] == 0
    assert updated["bars"] == 1
    assert updated["equity"] < 1.0
    assert updated["max_drawdown"] < 0
    assert "daily_loss" in updated["risk_breaches"]
    assert updated["position_weight"] == 0.0


def test_risk_settings_reject_unsafe_limits():
    with pytest.raises(ValueError):
        RiskSettings(max_drawdown=1.0)

    with pytest.raises(ValueError):
        RiskSettings(var_confidence=0.0)
