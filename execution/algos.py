"""Execution schedulers — return list of child quantities summing to parent."""
from __future__ import annotations

import numpy as np


def twap(qty: float, n_children: int) -> list[float]:
    return [qty / n_children] * n_children


def vwap(qty: float, n_children: int, volume_profile: np.ndarray | None = None) -> list[float]:
    if volume_profile is None:
        volume_profile = np.ones(n_children)
    p = np.asarray(volume_profile, dtype=float)
    if p.sum() <= 0:
        return twap(qty, n_children)
    p = p / p.sum()
    return (qty * p).tolist()


def pov(qty: float, n_children: int, market_volume: np.ndarray, participation: float = 0.10) -> list[float]:
    """Participation-of-volume — child qty capped at `participation * market_volume`."""
    caps = participation * np.asarray(market_volume, dtype=float)
    out: list[float] = []
    remaining = qty
    for c in caps:
        x = min(c, remaining / n_children) if n_children else 0.0
        out.append(x)
        remaining -= x
        if remaining <= 0:
            break
    return out
