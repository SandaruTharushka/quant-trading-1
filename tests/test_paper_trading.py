import pytest

from paper_trading import PaperTradingAccount


def test_paper_account_tracks_long_profit_and_fee():
    account = PaperTradingAccount(initial_cash=10_000, max_position_pct=0.5)

    account.execute_market("XAUUSDT", "buy", 1.0, 100.0)
    snapshot = account.mark("XAUUSDT", 110.0)

    assert snapshot["positions"][0]["quantity"] == 1.0
    assert snapshot["positions"][0]["unrealized_pnl"] == pytest.approx(9.992)
    assert snapshot["equity"] < 10_010.0

    account.execute_market("XAUUSDT", "sell", 1.0, 110.0)
    snapshot = account.snapshot()
    assert not snapshot["positions"]
    assert snapshot["realized_pnl"] > 0
    assert snapshot["fees_paid"] > 0


def test_paper_account_supports_short_positions():
    account = PaperTradingAccount(initial_cash=10_000, max_position_pct=0.5)

    account.execute_market("XAUUSDT", "sell", 2.0, 100.0)
    snapshot = account.mark("XAUUSDT", 90.0)

    assert snapshot["positions"][0]["quantity"] == -2.0
    assert snapshot["positions"][0]["unrealized_pnl"] == pytest.approx(19.984)


def test_paper_account_rejects_position_limit_and_reset_clears_state():
    account = PaperTradingAccount(initial_cash=1_000, max_position_pct=0.25)

    with pytest.raises(ValueError, match="maximum position"):
        account.execute_market("XAUUSDT", "buy", 3.0, 100.0)

    account.execute_market("XAUUSDT", "buy", 1.0, 100.0)
    snapshot = account.reset()
    assert snapshot["equity"] == 1_000
    assert snapshot["positions"] == []
    assert snapshot["trades"] == []
