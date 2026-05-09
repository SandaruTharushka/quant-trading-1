"""iTransformer (Liu et al. 2024) — invert the dimensions: tokens are *variates*."""
from __future__ import annotations

import torch
import torch.nn as nn


class ITransformer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]  -> per-variate token via Linear over time
        z = self.embed(x)                    # [B, C, D]
        z = self.encoder(z)                  # [B, C, D]
        return self.proj(z)                  # [B, C, H]
