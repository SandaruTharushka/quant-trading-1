"""Smart order router — selects venue and execution algo.

The actual broker connections (Binance, Bybit, IBKR, MetaTrader bridge) implement
`Broker` and are registered here. This is async-first to let tens of children
fly concurrently with retries and emergency cancel.
"""
from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

from core.types import Side


class ExecutionAlgo(str, Enum):
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"


@dataclass
class ChildOrder:
    symbol: str
    side: Side
    qty: float
    price: float | None
    venue: str
    tag: str = ""


class Broker(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def place(self, order: ChildOrder) -> dict: ...

    @abc.abstractmethod
    async def cancel(self, order_id: str) -> bool: ...

    @abc.abstractmethod
    async def positions(self) -> dict: ...


class OrderRouter:
    def __init__(self) -> None:
        self.brokers: dict[str, Broker] = {}

    def register(self, broker: Broker) -> None:
        self.brokers[broker.name] = broker

    async def execute_parent(
        self,
        symbol: str,
        side: Side,
        qty: float,
        algo: ExecutionAlgo,
        venue: str,
        scheduler: Callable[[float, int], list[float]],
        n_children: int = 10,
        delay_s: float = 1.0,
        max_retries: int = 3,
    ) -> list[dict]:
        """Slice parent into children using `scheduler(qty, n_children)` -> qtys."""
        broker = self.brokers[venue]
        slices = scheduler(qty, n_children)
        results: list[dict] = []
        for q in slices:
            order = ChildOrder(symbol=symbol, side=side, qty=q, price=None, venue=venue,
                               tag=algo.value)
            for attempt in range(max_retries):
                try:
                    res = await broker.place(order)
                    results.append(res)
                    break
                except Exception as e:                           # noqa: BLE001
                    if attempt == max_retries - 1:
                        results.append({"error": str(e), "qty": q})
                        break
                    await asyncio.sleep(2 ** attempt)
            await asyncio.sleep(delay_s)
        return results

    async def emergency_flatten(self) -> None:
        for broker in self.brokers.values():
            positions = await broker.positions()
            for sym, qty in positions.items():
                if qty == 0:
                    continue
                side = Side.SELL if qty > 0 else Side.BUY
                await broker.place(ChildOrder(symbol=sym, side=side, qty=abs(qty),
                                              price=None, venue=broker.name,
                                              tag="emergency"))
