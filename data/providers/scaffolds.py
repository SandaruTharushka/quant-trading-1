"""Scaffolded adapters — interface contract only.

These document the integration points for institutional venues. Each subclass
needs vendor SDK credentials to function; see the docstrings for the expected
auth/setup. They are registered so dependency injection works in tests.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import MarketDataProvider
from .registry import provider_registry


class _NotImplementedProvider(MarketDataProvider):
    """Base for providers that need vendor SDKs to actually run."""

    name = "abstract"
    requires: tuple[str, ...] = ()

    async def get_bars(self, symbol, start, end, timeframe="1m") -> pd.DataFrame:
        raise NotImplementedError(
            f"{self.name}: implement using vendor SDK ({', '.join(self.requires)})"
        )


class BybitProvider(_NotImplementedProvider):
    name = "bybit"
    requires = ("BYBIT_API_KEY", "BYBIT_API_SECRET")


class CMEProvider(_NotImplementedProvider):
    """CME via CME Smart Stream / iLink. Production access requires a vendor like Databento."""
    name = "cme"
    requires = ("CME_VENDOR", "CME_LOGIN")


class DatabentoProvider(_NotImplementedProvider):
    """Databento offers historical & live order book / MBO data for futures."""
    name = "databento"
    requires = ("DATABENTO_API_KEY",)


class DukascopyProvider(_NotImplementedProvider):
    """Dukascopy tick FX feed (JForex / HTTP archive)."""
    name = "dukascopy"
    requires = ("DUKASCOPY_USER", "DUKASCOPY_PASSWORD")


class PolygonProvider(_NotImplementedProvider):
    name = "polygon"
    requires = ("POLYGON_API_KEY",)


class IBKRProvider(_NotImplementedProvider):
    """Interactive Brokers via TWS / IB Gateway (ib_insync). Requires running gateway."""
    name = "ibkr"
    requires = ("IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID")


for _cls in (BybitProvider, CMEProvider, DatabentoProvider,
             DukascopyProvider, PolygonProvider, IBKRProvider):
    provider_registry.register(_cls.name, _cls)
