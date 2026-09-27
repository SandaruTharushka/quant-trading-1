# Research Dashboard

Run the FastAPI app from the repository root:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8001
```

Open `http://localhost:8001`. The dashboard can select currently trading Binance USDⓈ-M USDT perpetual assets. It loads up to 180 days of closed hourly candles on the first selection, then polls only recent candles and updates after a new bar closes.

## Research outputs

- Logistic Regression with standardized trailing returns and volatility features. The chronological 80/20 holdout purges the training label that crosses the test boundary.
- A 24-bar momentum benchmark evaluated on the same holdout.
- Both strategies use next-bar execution, configured commission and slippage, and the shared vectorized backtester.
- Portfolio sizing uses the shared volatility-targeting function and caps position weight using configured risk limits.
- The forward paper ledger starts when the app process starts for that asset. It tracks only subsequently completed bars, applies paper costs, and uses the shared risk engine. Its state is in memory and resets when the app restarts.

Model, backtest, and paper results are research outputs, not trade recommendations. No orders are submitted. The other neural, RL, regime, and microstructure components in the repository are not automatically trained or included in this live baseline.

The read-only project browser is limited to allowlisted workspace source and configuration files. API routes are documented at `/docs`.
