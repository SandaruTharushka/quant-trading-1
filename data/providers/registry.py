"""Provider registry — single point of lookup for live and historical adapters."""
from __future__ import annotations

from typing import Callable, Dict

from .base import MarketDataProvider


class _Registry:
    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., MarketDataProvider]] = {}

    def register(self, name: str, factory: Callable[..., MarketDataProvider]) -> None:
        self._factories[name] = factory

    def get(self, name: str, **kwargs) -> MarketDataProvider:
        if name not in self._factories:
            raise KeyError(f"Unknown provider: {name}. Registered: {list(self._factories)}")
        return self._factories[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)


provider_registry = _Registry()
