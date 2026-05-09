"""Triple-barrier + meta-labeling + CUSUM event sampling (Lopez de Prado)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------- CUSUM event sampler ----------

def cusum_events(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Return timestamps where cumulative log-return crosses ±threshold.

    Symmetric CUSUM filter — drops dependent observations and produces
    information-driven sampling rather than time-driven.
    """
    log_r = np.log(series.astype(float)).diff().fillna(0.0).values
    s_pos = 0.0
    s_neg = 0.0
    events: list[pd.Timestamp] = []
    idx = series.index
    for i, r in enumerate(log_r):
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos > threshold:
            s_pos = 0.0
            events.append(idx[i])
        elif s_neg < -threshold:
            s_neg = 0.0
            events.append(idx[i])
    return pd.DatetimeIndex(events)


# ---------- Triple barrier ----------

@dataclass
class TripleBarrier:
    pt_mult: float = 2.0   # profit-take multiple of vol
    sl_mult: float = 2.0   # stop-loss multiple of vol
    vert_bars: int = 50    # vertical barrier in bars


def _vert_index(t0: pd.Timestamp, bars: int, idx: pd.DatetimeIndex) -> pd.Timestamp | None:
    pos = idx.searchsorted(t0)
    end = pos + bars
    if end >= len(idx):
        return None
    return idx[end]


def triple_barrier_labels(
    prices: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    cfg: TripleBarrier = TripleBarrier(),
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute first-touch triple-barrier labels for each event.

    Returns DataFrame indexed by event time with columns:
        t1: first-touch time
        ret: return at first touch
        bin: {-1, 0, 1} (or {0, 1} when side is given for meta-labeling)
        target_vol: target volatility used to size barriers
    """
    out = pd.DataFrame(index=events, columns=["t1", "ret", "bin", "target_vol"])
    log_p = np.log(prices.astype(float))

    for t0 in events:
        if t0 not in volatility.index or pd.isna(volatility.loc[t0]):
            continue
        vol = float(volatility.loc[t0])
        if vol <= 0:
            continue
        t_vert = _vert_index(t0, cfg.vert_bars, prices.index)
        if t_vert is None:
            continue
        path = log_p.loc[t0:t_vert]
        if len(path) < 2:
            continue
        rets = (path - path.iloc[0])
        if side is not None and t0 in side.index:
            s = int(np.sign(side.loc[t0]))
            if s == 0:
                continue
            rets = rets * s
            up = cfg.pt_mult * vol
            dn = -cfg.sl_mult * vol
        else:
            up = cfg.pt_mult * vol
            dn = -cfg.sl_mult * vol

        touch_up = rets[rets >= up]
        touch_dn = rets[rets <= dn]
        candidates: list[tuple[pd.Timestamp, float]] = []
        if not touch_up.empty:
            candidates.append((touch_up.index[0], 1.0))
        if not touch_dn.empty:
            candidates.append((touch_dn.index[0], -1.0))
        candidates.append((t_vert, 0.0))
        candidates.sort(key=lambda x: x[0])
        t_first, _label = candidates[0]
        ret_first = float(np.expm1(rets.loc[t_first]))

        if side is not None:
            # meta-labeling: label is 1 if the side made money, else 0
            bin_ = 1 if ret_first > 0 else 0
        else:
            if t_first == t_vert and abs(ret_first) < up * 0.5:
                bin_ = 0
            else:
                bin_ = int(np.sign(ret_first)) if ret_first != 0 else 0

        out.loc[t0] = [t_first, ret_first, bin_, vol]
    return out.dropna()


def meta_label(
    primary_signals: pd.Series,
    prices: pd.Series,
    volatility: pd.Series,
    cfg: TripleBarrier = TripleBarrier(),
) -> pd.DataFrame:
    """Train labels for a meta-model that decides whether to take the primary side."""
    events = primary_signals.index
    return triple_barrier_labels(prices, events, volatility, cfg, side=primary_signals)


def apply_volatility_targets(
    returns: pd.Series, span: int = 100, floor: float = 1e-4
) -> pd.Series:
    """EWMA realized-vol used to size triple-barrier widths."""
    return returns.ewm(span=span).std().clip(lower=floor)
