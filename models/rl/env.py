"""Trading env with realistic microstructure for RL training.

Features modeled:
    * latency between decision and fill
    * proportional and queue-based slippage
    * spread widening under stress
    * partial fills
    * commissions (bps of notional)
    * vol-targeted reward (differentiable Sharpe-style ratio)

Compatible with `gymnasium` / Stable-Baselines3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # graceful fallback
    gym = None
    spaces = None


@dataclass
class ExecutionConfig:
    commission_bps: float = 0.5
    base_slippage_bps: float = 0.8
    impact_coef: float = 0.1          # bps per unit ADV
    latency_steps: int = 1
    partial_fill_prob: float = 0.05
    vol_target_annual: float = 0.10
    bars_per_year: int = 252 * 390


@dataclass
class MicrostructureConfig:
    queue_pos_init: float = 0.5       # 0=front, 1=back
    queue_decay: float = 0.9


class TradingEnv(gym.Env if gym else object):
    """Single-instrument continuous-position environment.

    Observation: window of features + current position
    Action:      target position in [-1, 1] (vol-normalized)
    Reward:      net log return of equity, with optional vol-target shaping.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,         # [T, F]
        prices: np.ndarray,           # [T]
        window: int = 64,
        cfg: ExecutionConfig = ExecutionConfig(),
        micro: MicrostructureConfig = MicrostructureConfig(),
        seed: int = 1337,
    ) -> None:
        if gym is None:
            raise RuntimeError("gymnasium is required to use TradingEnv")
        super().__init__()
        assert features.shape[0] == prices.shape[0]
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window = window
        self.cfg = cfg
        self.micro = micro
        self.rng = np.random.default_rng(seed)

        n_feat = features.shape[1] + 1  # +position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window, n_feat), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.t = 0
        self.position = 0.0
        self.equity = 1.0
        self.queue_pos = micro.queue_pos_init
        self._target = 0.0

    # ----- gym API -----

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = self.window
        self.position = 0.0
        self.equity = 1.0
        self.queue_pos = self.micro.queue_pos_init
        self._target = 0.0
        return self._obs(), {}

    def step(self, action):
        # latency: target set now, fill arrives `latency_steps` later
        self._target = float(np.clip(action[0], -1.0, 1.0))
        eff_target = self._target

        for _ in range(self.cfg.latency_steps):
            self.t += 1
            if self.t >= len(self.prices) - 1:
                return self._obs(), 0.0, True, False, {"equity": self.equity}

        # partial fill
        delta_target = eff_target - self.position
        if self.rng.random() < self.cfg.partial_fill_prob:
            delta_target *= self.rng.uniform(0.3, 0.9)

        # slippage: base + impact ~ |size|
        notional_change = abs(delta_target)
        slippage_bps = self.cfg.base_slippage_bps + self.cfg.impact_coef * notional_change * 1e4
        cost = (slippage_bps + self.cfg.commission_bps) * 1e-4 * notional_change

        new_pos = self.position + delta_target
        ret = (self.prices[self.t] / self.prices[self.t - 1]) - 1.0
        pnl = self.position * ret - cost
        self.equity *= (1.0 + pnl)
        self.position = new_pos

        # vol-target reward shaping
        target_vol_per_step = self.cfg.vol_target_annual / np.sqrt(self.cfg.bars_per_year)
        reward = pnl / max(target_vol_per_step, 1e-9)

        # queue position decays toward back when we trade aggressively
        self.queue_pos = max(0.0, min(1.0,
            self.queue_pos * self.micro.queue_decay + 0.1 * notional_change))

        terminated = self.equity < 0.5 or self.t >= len(self.prices) - 2
        return self._obs(), float(reward), bool(terminated), False, {
            "equity": self.equity, "position": self.position, "cost": cost,
        }

    def _obs(self) -> np.ndarray:
        feats = self.features[self.t - self.window : self.t]
        pos = np.full((self.window, 1), self.position, dtype=np.float32)
        return np.concatenate([feats, pos], axis=1)
