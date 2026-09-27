"""Print a candidate signal from Binance Futures XAUUSDT 1-hour candles."""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.providers.binance import BinanceFuturesProvider
from research.signals import analyze


def print_signal(close: pd.Series) -> None:
    timestamp, score, label = analyze(close)
    print(
        f"{timestamp.isoformat()} | XAUUSDT 1h | close={close.loc[timestamp]:.4f} "
        f"| score={score:.3f} | {label}",
        flush=True,
    )


def as_utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


async def watch(provider: BinanceFuturesProvider, close: pd.Series) -> None:
    last_seen = close.index[-1]
    while True:
        try:
            async for bar in provider.stream_bars("XAUUSDT", timeframe="1h"):
                timestamp = as_utc_timestamp(bar.ts)
                if timestamp <= last_seen:
                    continue
                close.loc[timestamp] = float(bar.c)
                close = close.sort_index().tail(24 * 30 + 24)
                last_seen = timestamp
                print_signal(close)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Market-data stream error: {error}; reconnecting in 5s", flush=True)
            await asyncio.sleep(5)


async def main(once: bool) -> None:
    provider = BinanceFuturesProvider()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    bars = await provider.get_bars("XAUUSDT", start, end, timeframe="1h")
    bars = bars.loc[bars.index < pd.Timestamp(end)]
    if bars.empty:
        raise RuntimeError("No closed XAUUSDT Futures candles returned; verify the symbol is listed.")

    close = bars["c"].astype(float)
    print_signal(close)
    if not once:
        await watch(provider, close)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print the latest closed-candle signal and exit")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.once))