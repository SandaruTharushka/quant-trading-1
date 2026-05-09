import numpy as np
import pandas as pd

from features.labeling import (
    cusum_events,
    triple_barrier_labels,
    apply_volatility_targets,
    TripleBarrier,
)


def test_cusum_emits_events(synthetic_price):
    rets = np.log(synthetic_price).diff().dropna()
    events = cusum_events(synthetic_price, threshold=float(rets.std()) * 2.5)
    assert len(events) > 0
    assert events.is_monotonic_increasing


def test_triple_barrier_no_lookahead(synthetic_price):
    rets = np.log(synthetic_price).diff().fillna(0.0)
    vol = apply_volatility_targets(rets, span=50)
    events = cusum_events(synthetic_price, threshold=float(rets.std()) * 2.5)
    labels = triple_barrier_labels(
        synthetic_price, events, vol,
        cfg=TripleBarrier(pt_mult=1.5, sl_mult=1.5, vert_bars=24),
    )
    assert len(labels) > 0
    # t1 must always be > event time (no instant labels)
    assert (pd.to_datetime(labels["t1"]) > labels.index).all()


def test_labels_three_classes(synthetic_price):
    rets = np.log(synthetic_price).diff().fillna(0.0)
    vol = apply_volatility_targets(rets, span=50)
    events = cusum_events(synthetic_price, threshold=float(rets.std()) * 2.0)
    labels = triple_barrier_labels(synthetic_price, events, vol)
    assert set(labels["bin"].unique()).issubset({-1, 0, 1})
