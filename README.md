# STAlgo

### Open-Source Quantitative Trading Research & Algorithmic Trading Framework

**STAlgo** is an open-source quantitative trading research framework designed for developers, quantitative researchers, machine learning engineers, and algorithmic traders who want to experiment with modern systematic trading techniques.

The project aims to provide a modular foundation for building advanced quantitative trading systems using:

* Machine Learning
* Deep Learning
* Market Microstructure
* Event-Based Sampling
* Regime Detection
* Portfolio Optimization
* Risk Management
* Backtesting
* Reinforcement Learning
* Live/Paper Trading Infrastructure
* MLOps and Model Monitoring

STAlgo is currently a **research and development framework**, not a finished production trading system.

The repository is being made public so developers and researchers can improve the existing components, implement missing modules, fix known issues, add market-data providers, develop new models, and help evolve STAlgo into a robust open quantitative trading platform.

---

# Project Status

> **Current Stage: Research / Experimental**

STAlgo already contains a significant part of the research architecture required for a modern quantitative trading system.

However, several modules are still experimental, incomplete, or implemented only as interfaces/placeholders.

The project should **not currently be considered production-ready for live trading with real capital**.

Current approximate status:

| Component                   | Status                   |
| --------------------------- | ------------------------ |
| Research Architecture       | ✅ Strong Foundation      |
| Feature Engineering         | ✅ Implemented            |
| Fractional Differentiation  | ✅ Implemented            |
| CUSUM Event Sampling        | ✅ Implemented            |
| Triple Barrier Labeling     | ✅ Implemented            |
| Meta Labeling               | ✅ Implemented            |
| Microstructure Features     | ✅ Implemented            |
| Purged Cross Validation     | ✅ Implemented            |
| Walk-Forward Validation     | ✅ Implemented            |
| Regime Detection            | ✅ Implemented            |
| PatchTST                    | ✅ Implemented            |
| iTransformer                | ✅ Implemented            |
| Temporal Fusion Transformer | ✅ Implemented            |
| N-HiTS                      | ✅ Implemented            |
| LSTM / GRU                  | ✅ Implemented            |
| CNN-LSTM                    | ✅ Implemented            |
| TLOB                        | ✅ Implemented            |
| DeepLOB                     | ⚠️ Known Issue           |
| Portfolio Sizing            | ✅ Foundation             |
| Risk Engine                 | ✅ Foundation             |
| Backtesting                 | ⚠️ Research-Level        |
| Binance Data Provider       | ✅ Partial Implementation |
| Databento                   | 🚧 Interface / Planned   |
| CME Data                    | 🚧 Planned               |
| Dukascopy                   | 🚧 Interface / Planned   |
| Interactive Brokers         | 🚧 Interface / Planned   |
| Bybit                       | 🚧 Planned               |
| Polygon                     | 🚧 Planned               |
| Reinforcement Learning      | ⚠️ PPO Foundation        |
| SAC                         | 🚧 Planned               |
| DDQN                        | 🚧 Planned               |
| Execution Engine            | 🚧 Foundation / Planned  |
| Paper Trading               | 🚧 Planned               |
| Live Trading                | ❌ Not Production Ready   |
| MLflow                      | ⚠️ Basic Integration     |
| ONNX                        | ⚠️ Basic Integration     |
| TensorRT                    | 🚧 Planned               |
| Monitoring Dashboard        | 🚧 Planned               |
| Kubernetes Deployment       | 🚧 Planned               |

---

# Vision

The long-term goal of STAlgo is to create an extensible research and trading architecture capable of supporting the complete quantitative trading lifecycle:

```text
Market Data
    ↓
Data Validation
    ↓
Data Lake / Time-Series Storage
    ↓
Feature Engineering
    ↓
Event Sampling
    ↓
Label Generation
    ↓
Dataset Construction
    ↓
Regime Detection
    ↓
Machine Learning / Deep Learning
    ↓
Model Ensemble
    ↓
Probability Calibration
    ↓
Meta Labeling
    ↓
Signal Generation
    ↓
Portfolio Construction
    ↓
Risk Management
    ↓
Backtesting
    ↓
Paper Trading
    ↓
Execution
    ↓
Live Trading
    ↓
Monitoring
    ↓
Model Drift Detection
    ↓
Retraining
```

The objective is not to create a single fixed trading strategy.

Instead, STAlgo is intended to become a **research platform where multiple strategies, models, datasets, instruments, and execution techniques can be tested using a shared institutional-style architecture.**

---

# Core Architecture

STAlgo separates the quantitative trading workflow into independent modules.

```text
Data Providers
     │
     ▼
Data Processing
     │
     ▼
Feature Engineering
     │
     ▼
Labeling / Event Detection
     │
     ▼
Regime Detection
     │
     ▼
ML / Deep Learning Models
     │
     ▼
Signal Engine
     │
     ▼
Portfolio Construction
     │
     ▼
Risk Engine
     │
     ▼
Backtester
     │
     ▼
Execution Router
     │
     ▼
Broker / Exchange
```

This modular design should allow contributors to improve individual components without rewriting the entire system.

---

# Data Layer

STAlgo is designed to support multiple data sources.

The architecture can eventually support:

* Historical OHLCV
* Tick data
* Trades
* Quotes
* Level 1 market data
* Level 2 order books
* Futures data
* Forex data
* Cryptocurrency data
* Alternative datasets

## Current Providers

### Binance

The Binance provider currently contains the most complete data-provider implementation.

It supports work toward:

* Historical Futures Klines
* Market Data Retrieval
* WebSocket Data
* Closed-Bar Processing

### Planned / Incomplete Providers

The repository also contains architecture or placeholders intended for providers such as:

* Databento
* CME
* Dukascopy
* Interactive Brokers
* Polygon
* Bybit

Contributors are welcome to implement production-quality provider adapters.

A provider should eventually handle:

```text
Authentication
Historical Download
Live Streaming
Reconnect Logic
Rate Limits
Schema Normalization
Timestamp Normalization
Missing Data
Retries
Caching
Metadata
Contract Information
```

---

# Futures Market Support

One of STAlgo's important future targets is institutional-quality futures research.

Potential instruments include:

```text
GC   Gold Futures
ES   E-mini S&P 500
NQ   Nasdaq Futures
CL   Crude Oil
SI   Silver
6E   Euro FX
ZN   Treasury Notes
```

For proper futures support, additional work is required around:

* Contract specifications
* Tick size
* Tick value
* Contract multiplier
* Exchange calendars
* Trading sessions
* Contract expiry
* Futures rolls
* Continuous contracts
* Margin calculations
* Exchange fees
* Historical contract mapping

---

# Feature Engineering

STAlgo already contains several advanced quantitative feature-engineering components.

## Fractional Differentiation

Fractional differentiation is included to help create more stationary financial time series while attempting to preserve useful memory from the original series.

Implemented concepts include:

```text
Fractional Differentiation
Fixed-Width Fractional Differentiation
Minimum-d Search
Stationarity-Oriented Transformation
```

This is useful when traditional differencing removes too much information from financial price series.

---

# Event-Based Sampling

Traditional machine-learning trading systems frequently train on every available candle.

STAlgo supports a more selective event-driven approach.

Implemented concepts include:

### CUSUM Filter

CUSUM-based event detection can identify statistically meaningful price movements and reduce unnecessary observations.

```text
Raw Market Data
        ↓
CUSUM Filter
        ↓
Important Events
        ↓
Labeling
```

---

# Triple Barrier Labeling

STAlgo includes a Triple Barrier framework for constructing trading labels.

Conceptually:

```text
Entry
 │
 ├── Upper Barrier → Profit Event
 │
 ├── Lower Barrier → Loss Event
 │
 └── Vertical Barrier → Time Expiration
```

This approach provides more realistic labels than simple future-return classification.

---

# Meta Labeling

Meta labeling is also included.

Instead of asking only:

```text
Should the strategy buy or sell?
```

a meta model can attempt to answer:

```text
Should this signal actually be traded?
```

This architecture may eventually allow:

```text
Primary Model
     ↓
Direction
     ↓
Meta Model
     ↓
Trade / Ignore
     ↓
Position Size
```

---

# Market Microstructure

STAlgo contains several market-microstructure-related features.

Current or experimental features include:

* Order Flow Imbalance
* Queue Imbalance
* Book Pressure
* VWAP Deviation
* Kyle's Lambda
* Realized Volatility
* Parkinson Volatility
* Garman-Klass Volatility
* Yang-Zhang Volatility
* Spread Dynamics
* Liquidity Sweep Detection

These features are especially useful when Level 2 or high-frequency market data becomes available.

---

# Machine Learning and Deep Learning

STAlgo is designed to support multiple model families rather than depending on a single neural network.

Current implementations include:

## PatchTST

Patch-based transformer architecture for time-series forecasting.

Potential applications:

* Return prediction
* Volatility prediction
* Multi-horizon forecasting
* Market-state representation

---

## iTransformer

Transformer architecture where variables can be treated differently from traditional time-series transformer tokenization.

---

## Temporal Fusion Transformer

Temporal Fusion Transformer components are included for multi-horizon forecasting research.

Potential uses include:

* Multi-step return prediction
* Volatility forecasting
* Regime-aware forecasting
* Feature importance research

---

## N-HiTS

N-HiTS implementation is included for hierarchical time-series forecasting experiments.

---

# Recurrent Models

The project also includes traditional deep-learning baselines:

* LSTM
* GRU
* CNN-LSTM

These models are important because complex architectures should always be compared against simpler baselines.

---

# Limit Order Book Models

STAlgo includes models intended for Level 2 order-book research.

## TLOB

Transformer-based Limit Order Book research architecture.

Current implementation passes the basic forward-model tests.

## DeepLOB

DeepLOB architecture is present, but there is currently a known tensor-shape problem in the forward path.

The architecture currently expects the convolutional stack to produce a width compatible with:

```text
[B, Channels, Time, 1]
```

before passing information to the recurrent layer.

The current convolution output may retain an additional dimension.

This is an open contribution area.

---

# Regime Detection

Financial markets do not behave identically at all times.

STAlgo includes regime detection so trading models and risk parameters can potentially change according to market conditions.

## Hidden Markov Model

Gaussian HMM-based regime detection is included.

Possible regimes include concepts such as:

```text
Low Volatility Trend Up
Low Volatility Trend Down

Calm Trend Up
Calm Trend Down

High Volatility Trend Up
High Volatility Trend Down

Crisis Trend Up
Crisis Trend Down
```

---

# Regime Routing

STAlgo also contains architecture for selecting different behavior according to the detected regime.

Example concept:

```text
LOW VOLATILITY
    ↓
PatchTST
    ↓
Higher allowed exposure
    ↓
Passive execution
```

```text
HIGH VOLATILITY
    ↓
TFT
    ↓
Reduced exposure
    ↓
VWAP-style execution
```

```text
CRISIS
    ↓
Defensive Model
    ↓
Very low / zero leverage
    ↓
Aggressive risk reduction
```

This is currently research architecture and should not be interpreted as a finished trading policy.

---

# Research Validation

A major goal of STAlgo is avoiding misleading machine-learning results caused by information leakage.

The repository contains implementations related to:

* Purged K-Fold Cross Validation
* Embargoed K-Fold
* Combinatorial Purged Cross Validation
* Walk-Forward Validation
* Deflated Sharpe Ratio
* Probability of Backtest Overfitting

---

# Purged Cross Validation

Standard random K-Fold validation is often inappropriate for financial time-series datasets.

STAlgo therefore includes time-series-aware validation approaches that attempt to prevent overlapping training and testing information.

Additional work is still required to fully harden CPCV behavior for complex non-contiguous test combinations.

Contributions in this area are particularly welcome.

---

# Portfolio Construction

STAlgo contains a portfolio-management foundation including concepts such as:

* Volatility Targeting
* Fractional Kelly Sizing
* Risk Parity
* CVaR Optimization
* Correlation-Aware Position Sizing

A strategy should ideally not translate a model prediction directly into a full-size position.

The intended pipeline is:

```text
Prediction
    ↓
Confidence
    ↓
Volatility
    ↓
Correlation
    ↓
Portfolio Risk
    ↓
Position Size
```

---

# Risk Management

The repository contains a central risk-engine foundation.

Existing concepts include:

* Maximum Drawdown
* Daily Loss Limits
* Gross Leverage Limits
* Position Limits
* Value at Risk
* Conditional Value at Risk
* Volatility Circuit Breakers

Possible responses include:

```text
WARN
REDUCE
FLATTEN
KILL
```

Future development should connect these risk decisions directly to the execution system.

For example:

```text
Risk Violation
      ↓
Block New Orders
      ↓
Cancel Existing Orders
      ↓
Reduce / Flatten Positions
      ↓
Disable Strategy
      ↓
Alert Operator
```

---

# Backtesting

STAlgo contains a research-oriented backtesting engine.

The current implementation includes important protections such as delayed signal execution and transaction costs.

For example, strategy weights can be shifted so that a signal generated from one bar is not incorrectly executed using information from the same bar.

The framework also supports concepts such as:

```text
Commission
Slippage
Returns
Portfolio Weights
Equity Curve
Sharpe Ratio
Drawdown
```

However, the current backtester should still be considered **research-level**.

---

# Future Institutional Backtester

Developers interested in market simulation can contribute support for:

* Bid/Ask spread
* Tick-level execution
* Limit orders
* Market orders
* Stop orders
* Partial fills
* Queue position
* Latency
* Market impact
* Order rejection
* Exchange fees
* Contract multiplier
* Tick size
* Futures margin
* Session breaks
* Exchange calendars
* Futures rolls
* Funding
* Liquidity constraints

The goal is to eventually separate:

```text
Signal Performance
```

from:

```text
Execution Performance
```

so strategies can be evaluated realistically.

---

# Reinforcement Learning

STAlgo contains an experimental reinforcement-learning environment.

The environment is designed to eventually model trading realities such as:

* Transaction costs
* Slippage
* Latency
* Position changes
* Partial fills
* Queue behavior
* Risk-adjusted rewards

## PPO

PPO training support is currently the main implemented RL approach.

## Future RL Algorithms

Contributions are welcome for:

* SAC
* DDQN
* Distributional RL
* Offline RL
* Conservative Q-Learning
* Risk-sensitive RL

RL results should always be compared against simple deterministic baselines.

---

# Experiment Tracking and MLOps

Basic architecture exists for model and experiment management.

Areas include:

* MLflow integration
* Model artifacts
* ONNX export/inference

Future work should expand this into a full model lifecycle.

```text
Dataset Version
      ↓
Feature Version
      ↓
Training Run
      ↓
Hyperparameters
      ↓
Metrics
      ↓
Model Artifact
      ↓
Backtest
      ↓
Approval
      ↓
Paper Trading
      ↓
Production
      ↓
Monitoring
      ↓
Retraining
```

---

# Model Monitoring

Future production versions should monitor:

* Prediction drift
* Feature drift
* Data drift
* PnL decay
* Sharpe decay
* Calibration decay
* Regime instability
* Latency
* Execution quality
* Slippage
* Market impact
* Missing data
* Model health

---

# Configuration System

The repository contains YAML-based research configuration files.

These configurations are intended to define:

```text
Datasets
Instruments
Features
Models
Validation
Backtesting
Portfolio Rules
Risk Limits
```

However, some current research scripts still contain hard-coded configuration values.

A major contribution opportunity is to make the entire research pipeline configuration-driven.

The desired behavior is:

```bash
python run_research_pipeline.py --config configs/research.yaml
```

and have all parameters loaded from the selected YAML file.

---

# Example Research Flow

The current research pipeline demonstrates concepts approximately like:

```text
Synthetic / Historical Price Data
            ↓
Returns
            ↓
Fractional Differentiation
            ↓
Realized Volatility
            ↓
CUSUM Events
            ↓
Triple Barrier Labels
            ↓
Regime Detection
            ↓
Cross Validation
            ↓
Trading Signal
            ↓
Volatility Targeting
            ↓
Backtesting
            ↓
Performance Evaluation
```

Some example pipelines currently use simple or mock signals.

The deep-learning models are not yet fully connected into one end-to-end production training and inference pipeline.

---

# Desired End-to-End ML Pipeline

One of the project's most important future milestones is:

```text
Historical Market Data
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Event Sampling
        ↓
Label Generation
        ↓
Training Dataset
        ↓
Train / Validation / Test
        ↓
Walk-Forward Training
        ↓
Hyperparameter Optimization
        ↓
Deep Learning Models
        ↓
Model Ensemble
        ↓
Probability Calibration
        ↓
Meta Model
        ↓
Signal Generation
        ↓
Position Sizing
        ↓
Risk Engine
        ↓
Backtesting
        ↓
Paper Trading
        ↓
Live Trading
```

---

# Model Ensemble Vision

Future versions should allow multiple models to contribute to a final prediction.

Example:

```text
PatchTST ────────┐
                 │
iTransformer ────┤
                 │
TFT ─────────────┤
                 ├── Ensemble
N-HiTS ──────────┤
                 │
TLOB ────────────┤
                 │
Regime Model ────┘
        ↓
Calibrated Probability
        ↓
Meta Label
        ↓
Trading Signal
```

Possible ensemble techniques include:

* Weighted average
* Dynamic weighting
* Bayesian model averaging
* Stacking
* Regime-conditioned ensembles
* Mixture of Experts

---

# Known Issues

The following issues are currently known.

## DeepLOB Forward Pass

DeepLOB currently has a tensor dimension mismatch before the LSTM/recurrent stage.

This should be investigated and fixed before using DeepLOB experiments.

---

## HMM Dependency

The regime module uses `hmmlearn`.

Developers should make sure all project dependencies are installed before running regime experiments.

---

## Configuration Integration

Some research values are currently hardcoded instead of being read directly from the YAML configuration.

---

## CPCV Hardening

The Combinatorial Purged Cross Validation implementation requires additional review for non-contiguous test-group handling and independent embargo/purge behavior.

---

## Real Data Providers

Several market-data provider classes currently act as interfaces or scaffolding rather than complete production integrations.

---

## Research Pipeline Integration

The deep-learning model library exists, but the complete:

```text
Dataset
→ Training
→ Validation
→ Model Selection
→ Prediction
→ Meta Labeling
→ Portfolio
→ Execution
```

pipeline still needs to be connected.

---

# Current Testing Status

At the time of the latest internal project review:

```text
31 tests collected
30 tests passed
1 test failed
```

The known failure was related to the DeepLOB tensor-shape issue.

Contributors should run the complete test suite before submitting changes.

---

# Development Priorities

Developers looking for areas to contribute can focus on the following roadmap.

## Phase 1 — Stabilize Current Codebase

Priority:

* Fix DeepLOB
* Remove hidden hardcoded configuration
* Improve unit tests
* Improve type checking
* Improve logging
* Improve exception handling
* Improve configuration validation
* Review CPCV implementation
* Add deterministic random seeds
* Improve documentation

---

## Phase 2 — Market Data Infrastructure

Implement:

* Databento integration
* CME futures data
* Dukascopy
* Interactive Brokers
* Bybit
* Polygon
* Unified market-data schema
* Data quality checks
* Historical downloader
* Live market stream
* Local caching
* Data replay

---

## Phase 3 — Dataset Engine

Create a proper dataset pipeline containing:

* Feature Store
* Label Store
* Dataset Versioning
* Time-Series Splits
* Walk-Forward Datasets
* Normalization
* Missing-data handling
* Leakage checks

---

## Phase 4 — Training Engine

Build a reusable training engine for:

* PatchTST
* iTransformer
* TFT
* N-HiTS
* LSTM
* GRU
* CNN-LSTM
* DeepLOB
* TLOB

Include:

* Early stopping
* Checkpoints
* Learning-rate schedules
* Mixed precision
* GPU training
* Hyperparameter optimization
* Experiment tracking

---

## Phase 5 — Ensemble and Signal Engine

Implement:

* Model ensemble
* Probability calibration
* Meta labeling
* Regime-conditioned models
* Signal filtering
* Confidence thresholds
* Position-sizing integration

---

## Phase 6 — Institutional Backtesting

Develop:

* Event-driven simulator
* Tick simulation
* Order lifecycle
* Partial fills
* Market impact
* Limit-order simulation
* Queue modeling
* Latency simulation
* Futures contracts
* Margin
* Exchange fees

---

## Phase 7 — Paper Trading

Connect the complete system to simulated brokerage/exchange environments.

Required components:

```text
Strategy
   ↓
Signal
   ↓
Risk Check
   ↓
Order Management System
   ↓
Execution Management System
   ↓
Broker
```

---

## Phase 8 — Live Trading Infrastructure

Future production architecture should include:

* Order Management System
* Execution Management System
* Broker adapters
* Reconciliation
* Position tracking
* Kill switch
* Health monitoring
* Retry logic
* Connection recovery
* Duplicate-order protection
* Persistent order state
* Audit logs

---

## Phase 9 — MLOps

Implement:

* Dataset versioning
* Model registry
* Model approvals
* Artifact tracking
* Drift monitoring
* Scheduled retraining
* Champion/challenger models
* ONNX optimization
* GPU inference
* TensorRT where appropriate

---

## Phase 10 — Monitoring Dashboard

Potential dashboard functionality:

* Live PnL
* Positions
* Orders
* Exposure
* Drawdown
* VaR / CVaR
* Strategy health
* Model predictions
* Regime
* Latency
* Slippage
* Data health
* Broker connectivity
* Risk alerts

---

# How Developers Can Help

STAlgo is being released publicly because the project still has many areas where developers and researchers can contribute.

We welcome contributions in:

### Quantitative Research

* Labeling techniques
* Cross validation
* Portfolio optimization
* Risk models
* Alpha research
* Statistical arbitrage
* Market microstructure

### Machine Learning

* Transformers
* Time-series models
* Representation learning
* Meta learning
* Ensemble models
* Calibration
* Explainability

### Reinforcement Learning

* PPO improvements
* SAC
* DDQN
* Offline RL
* Risk-aware RL

### Market Data

* Databento
* CME
* Interactive Brokers
* Dukascopy
* Bybit
* Polygon
* Binance

### Trading Infrastructure

* OMS
* EMS
* Broker adapters
* Paper trading
* Execution algorithms
* Risk controls
* Monitoring

### MLOps

* MLflow
* ONNX
* TensorRT
* Docker
* CI/CD
* Kubernetes
* Experiment tracking
* Drift detection

---

# Contribution Guidelines

Contributions are welcome.

Before submitting a pull request:

1. Fork the repository.
2. Create a feature branch.
3. Keep changes focused.
4. Add or update tests.
5. Do not introduce look-ahead bias.
6. Do not introduce data leakage.
7. Document major architectural changes.
8. Run the test suite.
9. Explain assumptions in the pull request.
10. Include research evidence when adding trading logic.

Example:

```bash
git checkout -b feature/databento-provider
```

Make your changes.

Then:

```bash
git add .
git commit -m "Add Databento market data provider"
git push origin feature/databento-provider
```

Open a Pull Request explaining:

```text
What was implemented
Why it was implemented
How it was tested
Known limitations
Future improvements
```

---

# Research Standards

Financial machine learning can easily produce misleading results.

All contributors should attempt to avoid:

* Look-ahead bias
* Survivorship bias
* Data snooping
* Target leakage
* Incorrect cross validation
* Unrealistic transaction costs
* Same-bar signal execution
* Unrealistic fills
* Overfitting
* Unreported failed experiments

A strategy showing excellent historical performance is **not automatically a valid trading strategy**.

---

# Security

Do not commit:

* API keys
* Broker credentials
* Exchange secrets
* Private datasets
* Passwords
* Database credentials

Use environment variables or secret-management systems.

Example:

```text
DATABENTO_API_KEY=
BINANCE_API_KEY=
BINANCE_API_SECRET=
IBKR_ACCOUNT=
```

Never store live credentials in source control.

---

# Disclaimer

STAlgo is provided for:

* Research
* Education
* Software development
* Quantitative experimentation

It is **not financial advice**, an investment recommendation, or a promise of profitability.

Trading futures, forex, cryptocurrencies, stocks, derivatives, and leveraged financial instruments involves significant risk.

Backtested results do not guarantee future performance.

Machine-learning models can fail when market conditions change.

Use this software entirely at your own risk.

Do not deploy experimental code with real capital without extensive independent validation, paper trading, risk controls, security review, and operational testing.

---

# Project Philosophy

STAlgo is based on a simple principle:

> **A trading model is only one small component of a complete quantitative trading system.**

A serious trading architecture also requires:

```text
Reliable Data
+
Correct Labels
+
Leakage-Safe Validation
+
Robust Models
+
Portfolio Construction
+
Risk Management
+
Realistic Execution
+
Monitoring
+
Operational Controls
```

The project therefore focuses not only on predicting markets, but on building the infrastructure required to research, evaluate, deploy, and monitor systematic trading strategies properly.

---

# STAlgo Future Architecture

```text
                    ┌─────────────────────┐
                    │    Market Sources   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Market Data Layer  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Data Quality Engine │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Feature Store     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Event / Label Layer │
                    └──────────┬──────────┘
                               │
             ┌─────────────────▼─────────────────┐
             │         ML / DL Model Layer       │
             │                                   │
             │ PatchTST | TFT | iTransformer     │
             │ N-HiTS | LSTM | TLOB | DeepLOB   │
             └─────────────────┬─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Model Ensemble    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Regime Router     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Signal Engine     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Portfolio Manager   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Risk Engine      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Execution Engine    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │ Broker / Exchange / Paper  │
                 └─────────────┬──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Monitoring / MLOps  │
                    └─────────────────────┘
```

---

# Help Build STAlgo

STAlgo is not being published as a finished product.

It is being published as an **open foundation for developers, quantitative researchers, traders, data engineers, and machine-learning engineers who want to build the next generation of systematic trading infrastructure together.**

If you are interested in:

* Quantitative Finance
* Algorithmic Trading
* Machine Learning
* Deep Learning
* Market Microstructure
* Futures
* Reinforcement Learning
* Backtesting
* Execution Systems
* MLOps

you are welcome to explore the repository, open issues, submit improvements, and contribute new modules.

---

## STAlgo

**Open Quant Research. Build. Test. Improve.**

> Research first. Validate everything. Risk before returns.
