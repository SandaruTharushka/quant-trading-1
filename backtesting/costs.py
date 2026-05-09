"""Cost and slippage models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    commission_bps: float = 0.5      # round-trip not assumed; per-side bps
    fee_per_contract: float = 0.0    # absolute, for futures

    def per_trade(self, notional: float, contracts: float = 0.0) -> float:
        return abs(notional) * self.commission_bps * 1e-4 + abs(contracts) * self.fee_per_contract


@dataclass
class SlippageModel:
    base_bps: float = 0.8
    impact_coef: float = 0.1   # bps per unit of (notional / ADV)

    def slippage(self, notional: float, adv: float) -> float:
        rel = abs(notional) / max(adv, 1e-9)
        return abs(notional) * (self.base_bps + self.impact_coef * rel * 1e4) * 1e-4
