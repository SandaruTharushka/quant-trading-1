"""Core domain types used throughout the platform."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class AssetClass(str, Enum):
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    EQUITY = "equity"


@dataclass(slots=True, frozen=True)
class Bar:
    ts: datetime
    symbol: str
    o: float
    h: float
    l: float
    c: float
    v: float
    asset_class: AssetClass = AssetClass.FUTURE


@dataclass(slots=True, frozen=True)
class Tick:
    ts: datetime
    symbol: str
    price: float
    size: float
    side: Side


@dataclass(slots=True, frozen=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(slots=True, frozen=True)
class OrderBookSnapshot:
    ts: datetime
    symbol: str
    bids: Sequence[OrderBookLevel]
    asks: Sequence[OrderBookLevel]

    @property
    def mid(self) -> float:
        return 0.5 * (self.bids[0].price + self.asks[0].price)

    @property
    def spread(self) -> float:
        return self.asks[0].price - self.bids[0].price


@dataclass(slots=True, frozen=True)
class Signal:
    ts: datetime
    symbol: str
    score: float          # raw model output, e.g. expected return / direction prob
    confidence: float     # [0,1]
    horizon_s: int        # seconds
    regime: str | None = None
    meta: dict | None = None
