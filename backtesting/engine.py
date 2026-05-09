"""Event-driven vectorized backtester.

Walks a panel of bar-level prices and a panel of target weights produced by
the strategy. Applies costs and slippage; rejects zero-cost configurations
(institutional discipline — every backtest must price its frictions).

For tick / book replay use `data.replay.HistoricalReplay` with the
same engine adapted to per-event step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.metrics import sharpe, sortino, calmar, max_drawdown, turnover
from .costs import CostModel, SlippageModel


@dataclass
class BacktestConfig:
    cost: CostModel = None
    slippage: SlippageModel = None
    bars_per_year: int = 252
    require_costs: bool = True

    def __post_init__(self):
        self.cost = self.cost or CostModel()
        self.slippage = self.slippage or SlippageModel()
        if self.require_costs and (
            self.cost.commission_bps == 0 and self.slippage.base_bps == 0
        ):
            raise ValueError("Costs/slippage cannot both be zero — institutional rule.")


@dataclass
class BacktestResult:
    equity: pd.Series
    pnl: pd.Series
    weights: pd.DataFrame
    metrics: dict[str, float]


class VectorizedBacktester:
    """Daily/intraday-bar level vectorized backtester with proper cost accounting."""

    def __init__(self, cfg: BacktestConfig | None = None) -> None:
        self.cfg = cfg or BacktestConfig()

    def run(
        self,
        prices: pd.DataFrame,        # [T, N]  asset close prices
        weights: pd.DataFrame,       # [T, N]  *target* portfolio weights at bar t (executed at t+1 open)
        adv: pd.DataFrame | None = None,  # average daily $ volume per asset
    ) -> BacktestResult:
        prices = prices.sort_index()
        weights = weights.reindex(prices.index).fillna(0.0)
        # execute next-bar — eliminate same-bar leakage
        exec_w = weights.shift(1).fillna(0.0)

        rets = prices.pct_change().fillna(0.0)
        gross_pnl = (exec_w * rets).sum(axis=1)

        # turnover-driven costs
        traded_w = exec_w.diff().abs().fillna(exec_w.abs())
        # notional traded ~ |Δw| * equity (keep equity ~ 1 nominal)
        equity = pd.Series(1.0, index=prices.index)
        adv_used = adv.reindex_like(prices).ffill() if adv is not None \
                   else pd.DataFrame(np.inf, index=prices.index, columns=prices.columns)

        net_pnl = pd.Series(0.0, index=prices.index)
        eq = 1.0
        for t in range(len(prices)):
            ts = prices.index[t]
            r = float(gross_pnl.iloc[t])
            traded_notional = traded_w.iloc[t] * eq
            adv_t = adv_used.iloc[t]
            costs = 0.0
            for sym in prices.columns:
                n = float(traded_notional[sym])
                a = float(adv_t.get(sym, np.inf))
                if n == 0:
                    continue
                costs += self.cfg.cost.per_trade(n)
                costs += self.cfg.slippage.slippage(n, a)
            pnl_t = r * eq - costs
            eq = eq + pnl_t
            equity.iloc[t] = eq
            net_pnl.iloc[t] = pnl_t

        rets_net = net_pnl / equity.shift(1).fillna(1.0)
        metrics = {
            "sharpe":    sharpe(rets_net, self.cfg.bars_per_year),
            "sortino":   sortino(rets_net, self.cfg.bars_per_year),
            "calmar":    calmar(rets_net, self.cfg.bars_per_year),
            "max_dd":    max_drawdown(equity),
            "turnover":  turnover(exec_w),
            "hit_ratio": float((rets_net > 0).mean()),
            "exposure":  float(exec_w.abs().sum(axis=1).mean()),
            "cum_return": float(equity.iloc[-1] - 1.0),
        }
        return BacktestResult(equity=equity, pnl=net_pnl, weights=exec_w, metrics=metrics)
