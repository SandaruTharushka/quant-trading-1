import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(1337)


@pytest.fixture
def synthetic_price(rng):
    n = 2000
    drift = 0.0001
    vol = 0.01
    r = rng.standard_normal(n) * vol + drift
    p = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(p, index=idx, name="price")
