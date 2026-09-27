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
    max_drawdown: float = Field(default=0.15, gt=0.0, lt=1.0)
    daily_loss_limit: float = Field(default=0.03, gt=0.0, lt=1.0)
    vol_target_annual: float = Field(default=0.10, gt=0.0)
    var_confidence: float = Field(default=0.99, gt=0.0, lt=1.0)
    cvar_confidence: float = Field(default=0.975, gt=0.0, lt=1.0)
    max_var: float = Field(default=0.05, gt=0.0)
    max_cvar: float = Field(default=0.07, gt=0.0)
    max_gross_leverage: float = Field(default=3.0, gt=0.0)
    max_position_pct: float = Field(default=0.25, gt=0.0, le=1.0)
    vol_circuit_breaker: float = Field(default=0.04, gt=0.0)


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
