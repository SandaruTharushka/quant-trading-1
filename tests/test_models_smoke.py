"""Smoke tests for model forward shapes — no training, just shape correctness."""
import pytest

torch = pytest.importorskip("torch")

from models.forecasters.patchtst import PatchTST
from models.forecasters.itransformer import ITransformer
from models.forecasters.tft import TFT
from models.forecasters.nhits import NHiTS
from models.forecasters.baselines import LSTMBaseline, GRUBaseline, CNNLSTM
from models.lob.deeplob import DeepLOB
from models.lob.tlob import TLOB


def test_patchtst_forward():
    m = PatchTST(n_channels=3, seq_len=64, pred_len=8, patch_len=16, stride=8)
    x = torch.randn(4, 3, 64)
    y = m(x)
    assert y.shape == (4, 3, 8)


def test_itransformer_forward():
    m = ITransformer(n_channels=3, seq_len=64, pred_len=8)
    x = torch.randn(4, 3, 64)
    y = m(x)
    assert y.shape == (4, 3, 8)


def test_tft_forward():
    m = TFT(n_channels=3, seq_len=32, pred_len=4)
    x = torch.randn(2, 3, 32)
    y = m(x)
    assert y.shape == (2, 3, 4)


def test_nhits_forward():
    m = NHiTS(n_channels=2, seq_len=64, pred_len=8, pool_kernels=(8, 4, 1))
    x = torch.randn(3, 2, 64)
    y = m(x)
    assert y.shape == (3, 2, 8)


def test_baselines_forward():
    for cls in (LSTMBaseline, GRUBaseline, CNNLSTM):
        m = cls(n_channels=2, seq_len=32, pred_len=4)
        x = torch.randn(2, 2, 32)
        y = m(x)
        assert y.shape == (2, 2, 4)


def test_deeplob_forward():
    m = DeepLOB(n_classes=3, lob_levels=10)
    x = torch.randn(2, 1, 100, 40)
    y = m(x)
    assert y.shape == (2, 3)


def test_tlob_forward():
    m = TLOB(n_features=40, seq_len=100, n_classes=3)
    x = torch.randn(2, 100, 40)
    y = m(x)
    assert y.shape == (2, 3)
