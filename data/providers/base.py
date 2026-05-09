"""Abstract market data provider interface.

Concrete adapters (Binance, Bybit, CME, Databento, Dukascopy, Polygon, IBKR)
implement this contract. The platform never imports them directly — it goes
through `provider_registry`.
"""
from __future__ import annotations

import abc
from datetime import datetime
from typing import AsyncIterator

import pandas as pd

from core.types import Bar, OrderBookSnapshot, Tick


class MarketDataProvider(abc.ABC):
    """Async-first provider contract."""

    name: str = "abstract"

    # -------- historical --------
    @abc.abstractmethod
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Return OHLCV bars indexed by UTC timestamp."""

    async def get_ticks(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not support tick history")

    async def get_book_snapshots(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not support book history")

    # -------- streaming --------
    async def stream_bars(self, symbol: str, timeframe: str = "1m") -> AsyncIterator[Bar]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream_ticks(self, symbol: str) -> AsyncIterator[Tick]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream_book(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]:
        raise NotImplementedError
        yield  # pragma: no cover

    # -------- contract handling (futures) --------
    def roll_calendar(self, root_symbol: str) -> pd.DataFrame:
        """Return a frame [contract, first_trade, last_trade, roll_date]."""
        raise NotImplementedError(f"{self.name} does not implement contract rolls")
