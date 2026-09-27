import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.providers.binance import BinanceFuturesProvider
from backtesting.engine import BacktestConfig, VectorizedBacktester
from backtesting.costs import CostModel, SlippageModel


async def main():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)

    provider = BinanceFuturesProvider()
    bars = await provider.get_bars(
        "XAUUSDT", start, end, timeframe="1h"
    )
    if bars.empty:
        raise ValueError("No bars returned; check if XAUUSDT is listed on Binance Futures.")

    close = bars["c"].astype(float)
    returns = close.pct_change()

    # 24-hour momentum, scaled by recent volatility
    momentum = np.log(close).diff().rolling(24).sum()
    volatility = returns.rolling(24).std() * np.sqrt(24)
    score = (momentum / volatility).replace([np.inf, -np.inf], np.nan)

    # Example rule only: long above 0.5, short below -0.5, otherwise flat
    position = pd.Series(
        np.select([score > 0.5, score < -0.5], [1.0, -1.0], default=0.0),
        index=close.index,
        name="XAUUSDT",
    )

    config = BacktestConfig(
        cost=CostModel(commission_bps=0.5),
        slippage=SlippageModel(base_bps=0.8),
        bars_per_year=24 * 365,
    )
    result = VectorizedBacktester(config).run(
        close.to_frame("XAUUSDT"),
        position.to_frame("XAUUSDT"),
    )

    latest = score.dropna().iloc[-1]
    label = "LONG candidate" if latest > 0.5 else "SHORT candidate" if latest < -0.5 else "FLAT"
    print(f"Latest: {label} (score={latest:.3f})")
    print("Backtest:", result.metrics)


if __name__ == "__main__":
    asyncio.run(main())