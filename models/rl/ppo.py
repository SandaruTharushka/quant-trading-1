"""PPO trainer wrapping Stable-Baselines3 against `TradingEnv`."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def train_ppo(
    env_fn: Callable,
    total_timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    n_envs: int = 4,
    save_path: str | Path | None = None,
    seed: int = 1337,
):
    """Train a PPO policy. Returns the trained model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([env_fn for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
    return model
