"""Pydantic-based settings for the platform.

All paths and credentials are environment-driven; nothing hardcoded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    root: Path = Field(default=Path("./_storage"))
    parquet_dir: Path = Field(default=Path("./_storage/parquet"))
    cache_dir: Path = Field(default=Path("./_storage/cache"))
    mlflow_uri: str = "file:./_storage/mlflow"


class RiskSettings(BaseSettings):
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.03
    vol_target_annual: float = 0.10
    var_confidence: float = 0.99
    cvar_confidence: float = 0.975
    max_gross_leverage: float = 3.0
    max_position_pct: float = 0.25


class ExecutionSettings(BaseSettings):
    base_commission_bps: float = 0.5
    base_slippage_bps: float = 0.8
    impact_coef: float = 0.1
    latency_ms: int = 50


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QT_", env_nested_delimiter="__", extra="ignore")

    env: str = "research"
    seed: int = 1337
    storage: StorageSettings = StorageSettings()
    risk: RiskSettings = RiskSettings()
    execution: ExecutionSettings = ExecutionSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
