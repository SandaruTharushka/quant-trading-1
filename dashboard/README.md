# Dashboard

Pages (planned):

| Page | Source |
|---|---|
| Portfolio | `/api/portfolio` + `/ws/portfolio` |
| Risk | `/api/risk/limits`, real-time breach feed |
| Executions | parent → child fills, slippage overlays |
| Model confidence | per-asset score / probability heatmap |
| Regime states | HMM posterior on running window |
| Exposure map | gross/net by asset class & sector |
| Correlation matrix | rolling-window asset correlations |
| Order flow heatmap | LOB / OFI density |
| Feature importance | model-attribution panel |
| Drift monitor | PSI + KS over rolling windows |

Stack: React + Plotly + TradingView lightweight-charts. The server side is in
`api/app.py`; the React app is intentionally not vendored — generate it with
`npm create vite@latest dashboard -- --template react-ts` and wire it to
the FastAPI + WS endpoints.
