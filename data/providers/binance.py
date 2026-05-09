"""Binance Futures adapter (interface scaffold).

Live calls require BINANCE_API_KEY / BINANCE_API_SECRET. The HTTP shape and rate
limits are real; auth and error handling are intentionally minimal.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import AsyncIterator

import pandas as pd

from core.types import Bar, AssetClass
from .base import MarketDataProvider
from .registry import provider_registry


class BinanceFuturesProvider(MarketDataProvider):
    name = "binance_futures"

    BASE = "https://fapi.binance.com"
    INTERVAL_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}

    def __init__(self, api_key: str | None = None, api_secret: str | None = None) -> None:
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        import httpx

        interval = self.INTERVAL_MAP[timeframe]
        url = f"{self.BASE}/fapi/v1/klines"
        all_rows: list[list] = []
        cur = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        async with httpx.AsyncClient(timeout=20.0) as client:
            while cur < end_ms:
                resp = await client.get(
                    url,
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": cur,
                        "endTime": end_ms,
                        "limit": 1500,
                    },
                )
                resp.raise_for_status()
                rows = resp.json()
                if not rows:
                    break
                all_rows.extend(rows)
                cur = rows[-1][0] + 1
        if not all_rows:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
        df = pd.DataFrame(
            all_rows,
            columns=[
                "open_time", "o", "h", "l", "c", "v",
                "close_time", "qv", "trades", "taker_b", "taker_q", "_",
            ],
        )
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["o", "h", "l", "c", "v"]:
            df[c] = df[c].astype(float)
        return df.set_index("ts")[["o", "h", "l", "c", "v"]]

    async def stream_bars(self, symbol: str, timeframe: str = "1m") -> AsyncIterator[Bar]:
        import json
        import websockets

        stream = f"{symbol.lower()}@kline_{self.INTERVAL_MAP[timeframe]}"
        url = f"wss://fstream.binance.com/ws/{stream}"
        async with websockets.connect(url, ping_interval=20) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                k = msg.get("k", {})
                if not k.get("x"):  # only emit closed bars
                    continue
                yield Bar(
                    ts=datetime.utcfromtimestamp(k["t"] / 1000),
                    symbol=symbol,
                    o=float(k["o"]),
                    h=float(k["h"]),
                    l=float(k["l"]),
                    c=float(k["c"]),
                    v=float(k["v"]),
                    asset_class=AssetClass.CRYPTO,
                )


provider_registry.register("binance_futures", BinanceFuturesProvider)
