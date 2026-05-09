# Quant-Trading-1 — Institutional AI Quant Infrastructure

A modular research and production framework for systematic trading of futures and FX,
designed around the architectures used at quant funds (AHL, Two Sigma, Citadel-style).

This repo is intentionally structured as **research-grade implementations of the
distinctive institutional pieces** (validation, labeling, microstructure, regime,
risk-aware backtest) plus **clean interface scaffolds** for the components that
require external data feeds, brokers, or large model training to run end-to-end.

## What is real, runnable code

These modules are full implementations and are unit-testable on synthetic data:

| Module | Implementation |
|---|---|
| `features.fracdiff` | Fixed-width fractional differentiation (Lopez de Prado) |
| `features.labeling` | Triple-barrier labels + meta-labeling, CUSUM event sampling |
| `features.microstructure` | Order-flow imbalance, VWAP deviation, queue imbalance, OFI, Kyle's lambda, realized vol estimators |
| `features.fracdiff_ffd` | Fixed-width FFD weights with stationarity preserved |
| `regime.hmm` | Gaussian HMM regime classifier (returns + vol features) |
| `regime.cusum` | CUSUM/BOCPD-style changepoint detection |
| `research.validation` | PurgedKFold, embargo, CPCV, walk-forward, Deflated Sharpe, PBO |
| `research.metrics` | Sharpe, Sortino, Calmar, deflated Sharpe, MAR, turnover |
| `models.forecasters.patchtst` | PatchTST forecaster (PyTorch) |
| `models.forecasters.itransformer` | iTransformer forecaster |
| `models.forecasters.tft` | Temporal Fusion Transformer (compact) |
| `models.forecasters.nhits` | N-HiTS forecaster |
| `models.forecasters.baselines` | LSTM, GRU, CNN-LSTM baselines |
| `models.lob.deeplob` | DeepLOB (CNN + Inception + LSTM) |
| `models.lob.tlob` | Transformer-LOB |
| `models.rl.env` | Vectorized trading env w/ slippage, latency, partial fills, queue |
| `models.rl.ppo` | PPO trainer wrapping SB3-compatible env |
| `backtesting.engine` | Event-driven vectorized backtester (costs, slippage, vol-target) |
| `risk.engine` | VaR/CVaR, drawdown kill-switch, vol circuit breaker |
| `portfolio.optimizer` | Vol-targeting, Kelly, risk parity, CVaR, correlation-aware sizing |
| `core.config` / `core.logging` | Pydantic settings, structured logging |

## What is interface scaffold

These define the boundary contracts but require credentials / feeds / infra to run live:

- `data.providers.*` — Binance, Bybit, CME, Databento, Dukascopy, Polygon, IBKR adapters
- `execution.*` — live order routers (Binance Futures, Bybit, IBKR, MT bridge)
- `dashboard/` — FastAPI + WS server skeleton; React app not included
- `deployment/` — Dockerfiles, compose, MLflow setup
- `mlops/` — model registry / drift monitor / retraining loops

## Layout

```
core/           # config, logging, types, time utilities
data/           # provider adapters, parquet store, replay engine
features/       # feature pipelines (fracdiff, labels, microstructure, regime feats)
models/
  forecasters/  # PatchTST, iTransformer, TFT, N-HiTS, Mamba, LSTM/GRU/CNN-LSTM
  lob/          # DeepLOB, TLOB
  rl/           # env, PPO/SAC/DDQN trainers
research/       # validation (purged/CPCV/embargo), metrics, deflated Sharpe, PBO
backtesting/    # event-driven engine, slippage, fills, costs
execution/      # routers, smart order routing, TWAP/VWAP/POV
risk/           # VaR/CVaR, kill switch, circuit breakers, exposure caps
portfolio/      # vol-target, Kelly, risk parity, CVaR optimizer
regime/         # HMM, CUSUM, vol-clustering, regime router
monitoring/     # drift, anomaly, live dashboards
inference/      # ONNX/TensorRT export, async inference server
api/            # FastAPI app
dashboard/      # WS server + assets
mlops/          # MLflow registry, retraining, experiments
deployment/     # Docker, compose, k8s
tests/          # pytest tests for the algorithmic core
configs/        # YAML configs (data, models, risk, portfolio)
```

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -q
```

Run a sample research pipeline (synthetic data) to validate the institutional
validation + backtest path end-to-end:

```bash
python -m scripts.run_research_pipeline --config configs/research.yaml
```

## Design principles

- **No future leakage anywhere**: every label, feature, and split is causal.
- **Walk-forward by default**: retraining cadence and embargo are config-driven.
- **Vol-targeted PnL**: position sizing flows through `portfolio.optimizer`, never raw signal.
- **Costs are mandatory**: backtests reject zero-cost configs.
- **Regime-aware**: model selection and leverage are conditioned on the regime engine.
- **Reproducible**: seeded RNGs, hashed configs, MLflow run IDs.

## What this is NOT

- Not a retail indicator system. No RSI/MACD crossover logic anywhere.
- Not a single-shot backtest tool. The validation harness rejects in-sample tuning.
- Not a substitute for real venue infra. Live adapters are interface-level.
