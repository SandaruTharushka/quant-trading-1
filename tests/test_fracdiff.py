import numpy as np
import pandas as pd

from features.fracdiff import frac_diff_ffd, _ffd_weights


def test_ffd_weights_decay():
    w = _ffd_weights(0.5, thres=1e-3)
    # weights are stored newest-last; magnitudes should taper
    assert len(w) > 1
    assert abs(w[-1]) > abs(w[0])  # newest has largest weight (1.0 for d∈(0,1) at lag 0)


def test_fracdiff_preserves_correlation_with_price(synthetic_price):
    series = np.log(synthetic_price)
    diffed = frac_diff_ffd(series, d=0.4).dropna()
    assert len(diffed) > 100
    # Fractional diff should still correlate positively with the original log price
    corr = pd.concat([series, diffed], axis=1).dropna().corr().iloc[0, 1]
    assert corr > 0.0


def test_fracdiff_d_one_equals_first_diff(synthetic_price):
    series = np.log(synthetic_price)
    diffed = frac_diff_ffd(series, d=1.0, thres=1e-12).dropna()
    plain = series.diff().dropna()
    aligned = diffed.align(plain, join="inner")
    np.testing.assert_allclose(aligned[0].values, aligned[1].values, atol=1e-6)
