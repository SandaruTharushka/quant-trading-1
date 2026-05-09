"""Historical replay engine — feeds events to strategies as if live.

Supports bar, tick, and book streams with optional latency injection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator

import pandas as pd


@dataclass
class ReplayEvent:
    ts: pd.Timestamp
    symbol: str
    kind: str  # "bar" | "tick" | "book" | "feature"
    payload: dict


class HistoricalReplay:
    """Iterate multi-symbol history in strict timestamp order with optional latency offset."""

    def __init__(self, latency: timedelta = timedelta(milliseconds=0)) -> None:
        self.streams: list[tuple[str, str, pd.DataFrame]] = []
        self.latency = latency

    def add(self, symbol: str, kind: str, df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Replay sources need DatetimeIndex")
        self.streams.append((symbol, kind, df.sort_index()))

    def __iter__(self) -> Iterator[ReplayEvent]:
        merged = []
        for sym, kind, df in self.streams:
            for ts, row in df.iterrows():
                merged.append((ts, sym, kind, row.to_dict()))
        merged.sort(key=lambda e: e[0])
        for ts, sym, kind, payload in merged:
            yield ReplayEvent(ts=ts + self.latency, symbol=sym, kind=kind, payload=payload)
