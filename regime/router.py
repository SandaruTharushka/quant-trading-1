"""Regime router — maps regime label to model + leverage + execution policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RegimePolicy:
    model: str
    max_leverage: float
    exec_style: str = "passive"   # passive | aggressive | twap | vwap | pov
    notes: str = ""


@dataclass
class RegimeRouter:
    """Default router — substring matching against regime labels.

    Example regime labels (from HMMRegime): "low_vol_trend_up", "crisis_neutral".
    """

    policies: dict[str, RegimePolicy] = field(default_factory=lambda: {
        "low_vol":  RegimePolicy(model="patchtst", max_leverage=2.0, exec_style="passive"),
        "calm":     RegimePolicy(model="itransformer", max_leverage=1.5, exec_style="passive"),
        "high_vol": RegimePolicy(model="tft", max_leverage=0.7, exec_style="vwap"),
        "crisis":   RegimePolicy(model="lstm_baseline", max_leverage=0.0, exec_style="aggressive",
                                  notes="hedge / de-risk only"),
    })
    default: RegimePolicy = field(default_factory=lambda: RegimePolicy(
        model="lstm_baseline", max_leverage=0.5, exec_style="passive"
    ))

    def route(self, label: str) -> RegimePolicy:
        for key, pol in self.policies.items():
            if key in label:
                return pol
        return self.default

    def with_overrides(self, **kw: Callable[[RegimePolicy], RegimePolicy]):
        return RegimeRouter(
            policies={k: f(v) for (k, v), f in zip(self.policies.items(), kw.values())},
            default=self.default,
        )
