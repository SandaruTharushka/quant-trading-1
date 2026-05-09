"""Live risk engine with circuit breakers and a kill-switch.

Stateful — call `update(equity, weights, returns_window)` each bar. Returns a
list of `RiskBreach` events; the strategy must respect breach.action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .var import historical_var, cvar


class BreachAction(str, Enum):
    WARN = "warn"
    REDUCE = "reduce"
    FLATTEN = "flatten"
    KILL = "kill"


@dataclass
class RiskBreach:
    rule: str
    action: BreachAction
    detail: str


@dataclass
class RiskLimits:
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.03
    var_confidence: float = 0.99
    max_var: float = 0.05
    max_cvar: float = 0.07
    max_gross_leverage: float = 3.0
    max_position_pct: float = 0.25
    vol_circuit_breaker: float = 0.04


@dataclass
class RiskEngine:
    limits: RiskLimits = field(default_factory=RiskLimits)
    _peak_equity: float = field(default=1.0, init=False)
    _start_of_day_eq: float = field(default=1.0, init=False)
    _killed: bool = field(default=False, init=False)

    def reset_day(self, equity: float) -> None:
        self._start_of_day_eq = equity

    def update(
        self,
        equity: float,
        weights: pd.Series,
        returns_window: np.ndarray,
    ) -> list[RiskBreach]:
        breaches: list[RiskBreach] = []
        if self._killed:
            return [RiskBreach("kill_switch", BreachAction.KILL, "previously tripped")]

        self._peak_equity = max(self._peak_equity, equity)
        dd = (equity - self._peak_equity) / self._peak_equity
        if dd <= -self.limits.max_drawdown:
            self._killed = True
            return [RiskBreach("max_drawdown", BreachAction.KILL,
                               f"dd={dd:.3f} <= -{self.limits.max_drawdown}")]

        daily_loss = (equity - self._start_of_day_eq) / max(self._start_of_day_eq, 1e-12)
        if daily_loss <= -self.limits.daily_loss_limit:
            breaches.append(RiskBreach("daily_loss", BreachAction.FLATTEN,
                                       f"daily={daily_loss:.3f}"))

        gross = float(weights.abs().sum())
        if gross > self.limits.max_gross_leverage:
            breaches.append(RiskBreach("gross_leverage", BreachAction.REDUCE,
                                       f"gross={gross:.2f}"))

        if (weights.abs() > self.limits.max_position_pct).any():
            sym = weights.abs().idxmax()
            breaches.append(RiskBreach("position_cap", BreachAction.REDUCE,
                                       f"{sym}={weights[sym]:.3f}"))

        if len(returns_window) > 30:
            v = historical_var(returns_window, self.limits.var_confidence)
            c = cvar(returns_window, max(0.0, self.limits.var_confidence - 0.015))
            if v > self.limits.max_var:
                breaches.append(RiskBreach("var", BreachAction.REDUCE, f"VaR={v:.3f}"))
            if c > self.limits.max_cvar:
                breaches.append(RiskBreach("cvar", BreachAction.REDUCE, f"CVaR={c:.3f}"))
            rv = float(np.std(returns_window[-20:], ddof=1) or 0.0)
            if rv > self.limits.vol_circuit_breaker:
                breaches.append(RiskBreach("vol_breaker", BreachAction.FLATTEN,
                                           f"rv={rv:.3f}"))
        return breaches

    @property
    def killed(self) -> bool:
        return self._killed
