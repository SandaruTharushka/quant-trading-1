"""Chronological single-asset research and in-memory forward paper tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtesting.costs import CostModel, SlippageModel
from backtesting.engine import BacktestConfig, VectorizedBacktester
from portfolio.optimizer import vol_target_weights
from risk.engine import BreachAction, RiskEngine, RiskLimits

FEATURE_LAGS = (1, 3, 6, 12, 24)
BARS_PER_YEAR = 24 * 365


def _features(close: pd.Series) -> pd.DataFrame:
    log_returns = np.log(close).diff()
    frame = pd.DataFrame(index=close.index)
    for lag in FEATURE_LAGS:
        frame[f"return_{lag}"] = log_returns.rolling(lag).sum()
    frame["volatility_24"] = log_returns.rolling(24).std()
    frame["volatility_72"] = log_returns.rolling(72).std()
    return frame.replace([np.inf, -np.inf], np.nan)


def _weights(
    signal: pd.Series,
    returns: pd.Series,
    target_vol: float,
    max_position: float,
) -> pd.Series:
    realized_vol = returns.rolling(24).std()
    sized = vol_target_weights(
        signal,
        realized_vol,
        vol_target_annual=target_vol,
        bars_per_year=BARS_PER_YEAR,
        max_leverage=max_position,
    )
    return sized.clip(-max_position, max_position)


def _backtest(close: pd.Series, weights: pd.Series, commission_bps: float, slippage_bps: float) -> dict:
    result = VectorizedBacktester(
        BacktestConfig(
            cost=CostModel(commission_bps=commission_bps),
            slippage=SlippageModel(base_bps=slippage_bps),
            bars_per_year=BARS_PER_YEAR,
        )
    ).run(close.to_frame("asset"), weights.to_frame("asset"))
    return {key: float(value) for key, value in result.metrics.items()}


def build_research_report(
    bars: pd.DataFrame,
    *,
    target_vol: float = 0.10,
    max_position: float = 0.25,
    commission_bps: float = 0.5,
    slippage_bps: float = 0.8,
) -> dict:
    """Fit on the first 80%, evaluate only later bars, and forecast the latest close."""
    if "c" not in bars or len(bars) < 300:
        raise ValueError("At least 300 hourly close bars are required for research.")
    close = bars["c"].astype(float).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    if (close <= 0).any() or not np.isfinite(close.to_numpy()).all():
        raise ValueError("Close prices must be finite positive values.")

    features = _features(close)
    forward_returns = close.pct_change().shift(-1).rename("forward_return")
    samples = features.join(forward_returns).dropna()
    split = int(len(samples) * 0.8)
    if split < 100 or len(samples) - split < 30:
        raise ValueError("Not enough usable bars after feature warmup for a holdout test.")

    # The last training label ends at the first holdout timestamp; purge it.
    train = samples.iloc[: split - 1]
    test = samples.iloc[split - 1:]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=1337))
    model.fit(train[features.columns], (train["forward_return"] > 0).astype(int))

    probabilities = pd.Series(
        model.predict_proba(test[features.columns])[:, 1], index=test.index, name="probability_up"
    )
    model_signal = pd.Series(
        np.select([probabilities >= 0.55, probabilities <= 0.45], [1.0, -1.0], default=0.0),
        index=test.index,
    )
    momentum_signal = pd.Series(
        np.sign(features.loc[test.index, "return_24"]), index=test.index
    )
    returns = close.pct_change().fillna(0.0)
    model_weights = _weights(model_signal, returns, target_vol, max_position).reindex(test.index).fillna(0.0)
    momentum_weights = _weights(momentum_signal, returns, target_vol, max_position).reindex(test.index).fillna(0.0)
    holdout_prices = close.loc[test.index[0]:]
    model_metrics = _backtest(holdout_prices, model_weights, commission_bps, slippage_bps)
    momentum_metrics = _backtest(holdout_prices, momentum_weights, commission_bps, slippage_bps)

    actual_up = test["forward_return"] > 0
    accuracy = float(((probabilities >= 0.5) == actual_up).mean())
    latest_features = features.iloc[[-1]]
    latest_probability = float(model.predict_proba(latest_features)[0, 1])
    latest_signal = 1.0 if latest_probability >= 0.55 else -1.0 if latest_probability <= 0.45 else 0.0
    recent_volatility = float(returns.tail(24).std() * np.sqrt(BARS_PER_YEAR))
    latest_weight = float(
        _weights(
            pd.Series([latest_signal], index=[close.index[-1]]),
            returns,
            target_vol,
            max_position,
        ).iloc[-1]
    )

    return {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "bars": int(len(close)),
        "train_bars": int(len(train)),
        "holdout_bars": int(len(test)),
        "holdout_start": test.index[0].isoformat(),
        "holdout_end": test.index[-1].isoformat(),
        "model": {
            "name": "Logistic Regression",
            "accuracy": accuracy,
            "probability_up": latest_probability,
            "signal": "LONG candidate" if latest_signal > 0 else "SHORT candidate" if latest_signal < 0 else "FLAT",
            "position_weight": latest_weight,
        },
        "portfolio": {
            "target_annual_volatility": target_vol,
            "realized_annual_volatility": recent_volatility,
            "position_cap": max_position,
            "position_weight": latest_weight,
        },
        "backtest": {
            "method": "Chronological 80/20 holdout; boundary label purged",
            "model": model_metrics,
            "momentum": momentum_metrics,
            "commission_bps_per_side": commission_bps,
            "slippage_bps": slippage_bps,
        },
        "_latest_probability": latest_probability,
        "_latest_weight": latest_weight,
        "_latest_close": float(close.iloc[-1]),
        "_last_bar_time": close.index[-1].isoformat(),
    }


@dataclass
class ForwardPaperLedger:
    """Process-only paper account. A position is applied to the next observed bar."""
    equity: float = 1.0
    peak_equity: float = 1.0
    max_drawdown: float = 0.0
    bars: int = 0
    wins: int = 0
    position_weight: float = 0.0
    last_close: float | None = None
    last_time: pd.Timestamp | None = None
    day: date | None = None
    day_start_equity: float = 1.0
    returns: list[float] = field(default_factory=list)
    risk_engine: RiskEngine = field(default_factory=lambda: RiskEngine(limits=RiskLimits()))
    breach_labels: list[str] = field(default_factory=list)

    def update(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        close: float,
        next_weight: float,
        *,
        max_position: float = 0.25,
        commission_bps: float = 0.5,
        slippage_bps: float = 0.8,
    ) -> dict:
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        timestamp = timestamp.tz_convert("UTC")
        new_day = timestamp.date()
        if self.day != new_day:
            self.day = new_day
            self.day_start_equity = self.equity
            self.risk_engine.reset_day(self.equity)

        bar_return = 0.0
        if self.last_close is not None and self.last_close > 0:
            bar_return = close / self.last_close - 1.0
            turnover = abs(float(next_weight) - self.position_weight)
            costs = turnover * (commission_bps + slippage_bps) * 1e-4
            net_return = self.position_weight * bar_return - costs
            self.equity *= max(0.0, 1.0 + net_return)
            self.bars += 1
            self.wins += int(net_return > 0)
            self.returns.append(net_return)
            self.returns = self.returns[-500:]
            self.peak_equity = max(self.peak_equity, self.equity)
            drawdown = self.equity / self.peak_equity - 1.0
            self.max_drawdown = min(self.max_drawdown, drawdown)

        bounded_weight = float(np.clip(next_weight, -max_position, max_position))
        breaches = self.risk_engine.update(
            self.equity,
            pd.Series({symbol: bounded_weight}),
            np.asarray(self.returns, dtype=float),
        )
        self.breach_labels = [breach.rule for breach in breaches]
        if any(breach.action in (BreachAction.FLATTEN, BreachAction.KILL) for breach in breaches):
            bounded_weight = 0.0
        elif any(breach.action == BreachAction.REDUCE for breach in breaches):
            bounded_weight *= 0.5
        self.position_weight = bounded_weight
        self.last_close = float(close)
        self.last_time = timestamp
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "status": "PAPER",
            "bars": self.bars,
            "equity": self.equity,
            "cumulative_return": self.equity - 1.0,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.wins / self.bars if self.bars else 0.0,
            "position_weight": self.position_weight,
            "last_bar_time": self.last_time.isoformat() if self.last_time is not None else None,
            "risk_breaches": self.breach_labels,
            "risk_killed": self.risk_engine.killed,
        }
