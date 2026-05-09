"""Transformer-LOB — multi-head attention over time on LOB features."""
from __future__ import annotations

import torch
import torch.nn as nn


class TLOB(nn.Module):
    def __init__(
        self,
        n_features: int = 40,
        seq_len: int = 100,
        n_classes: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        z = self.embed(x) + self.pos[:, : x.size(1)]
        z = self.encoder(z)
        return self.head(z[:, -1])
