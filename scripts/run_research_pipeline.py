"""End-to-end research pipeline on synthetic data.

Demonstrates the institutional pipeline from raw prices to PnL:

    raw price ─► fracdiff ─► EWMA vol ─► CUSUM events
                                    │
                                    ▼
                          triple-barrier labels
                                    │
                                    ▼
                         CPCV over labeled events
                                    │
                                    ▼
                     vol-target sizing + backtest with costs
                                    │
                                    ▼
                Sharpe / Deflated Sharpe / PBO / drift report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.fracdiff import frac_diff_ffd
from features.labeling import (
    cusum_events, triple_barrier_labels, TripleBarrier,
    apply_volatility_targets,
)
from features.microstructure import realized_volatility
from research.validation import CombinatorialPurgedCV
from research.metrics import sharpe, deflated_sharpe
from regime.hmm import HMMRegime
from portfolio.optimizer import vol_target_weights
from backtesting.engine import VectorizedBacktester, BacktestConfig
from backtesting.costs import CostModel, SlippageModel


def synthesize(n: int = 5000, seed: int = 1337) -> pd.Series:
    rng = np.random.default_rng(seed)
    drift = 0.0001
    vol = 0.01 + 0.005 * np.sin(np.linspace(0, 8 * np.pi, n))   # vol-clustering
    shocks = rng.standard_normal(n) * vol + drift
    # add a regime break midway
    shocks[n // 2:] = shocks[n // 2:] * 1.8 - 0.0003
    p = 100 * np.exp(np.cumsum(shocks))
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(p, index=idx, name="price")


def main(config_path: str | None = None) -> None:
    price = synthesize()
    log_p = np.log(price)
    rets = log_p.diff().dropna()

    # --- features ---
    ffd = frac_diff_ffd(log_p, d=0.4).dropna()
    rv = realized_volatility(rets, window=60).dropna()
    vol_for_labels = apply_volatility_targets(rets, span=60).reindex(price.index).ffill()

    # --- events + labels ---
    events = cusum_events(price, threshold=float(rets.std()) * 3.0)
    print(f"CUSUM events: {len(events)}")

    labels = triple_barrier_labels(
        price, events, vol_for_labels,
        cfg=TripleBarrier(pt_mult=2.0, sl_mult=2.0, vert_bars=24),
    )
    print(f"Triple-barrier labels: {len(labels)}")

    # --- regime ---
    feats = pd.concat([rets.rename("r"), rets.abs().rename("ar"), rv.rename("rv")], axis=1).dropna()
    regime = HMMRegime(n_states=3).fit(feats)
    rstates = regime.predict(feats)
    print("Regime distribution:")
    print(rstates.value_counts())

    # --- CPCV over labels (illustrative; we just print path counts) ---
    if len(labels) > 30:
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
        n_paths = cpcv.n_paths()
        n_splits = sum(1 for _ in cpcv.split(labels["t1"]))
        print(f"CPCV n_splits={n_splits}, distinct backtest paths={n_paths}")

    # --- mock-signal -> vol-target -> backtest ---
    signal = ffd.rolling(20).mean().pipe(np.tanh).reindex(price.index).fillna(0.0)
    weights = vol_target_weights(signal, rv.reindex(price.index).ffill(),
                                 vol_target_annual=0.10, bars_per_year=24 * 252)
    weights_df = weights.to_frame(name=price.name)
    prices_df = price.to_frame()

    bt = VectorizedBacktester(
        BacktestConfig(cost=CostModel(commission_bps=0.5),
                       slippage=SlippageModel(base_bps=0.8))
    )
    res = bt.run(prices_df, weights_df.rename(columns={price.name: price.name}))
    print("Backtest metrics:")
    for k, v in res.metrics.items():
        print(f"  {k:>12s}: {v:.4f}")

    rets_net = res.pnl / res.equity.shift(1).fillna(1.0)
    sr = sharpe(rets_net, periods_per_year=24 * 252)
    print(f"Sharpe={sr:.3f}")

    if len(rets_net) > 100 and rets_net.std() > 0:
        dsr = deflated_sharpe(
            sr_observed=sr,
            n_returns=len(rets_net),
            skew=float(rets_net.skew()),
            kurt=float(rets_net.kurt()),
            n_trials=10,
            sr_variance_across_trials=0.5,
        )
        print(f"Deflated Sharpe (P[true SR > 0]): {dsr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.yaml")
    args = parser.parse_args()
    main(args.config)
