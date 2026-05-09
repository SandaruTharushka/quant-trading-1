"""Compact Temporal Fusion Transformer.

This is a deliberately stripped TFT (Lim et al. 2021) — gated residual networks,
variable selection, and a multi-head attention block over the encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.lin(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.glu = GatedLinearUnit(d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        return self.norm(x + self.glu(h))


class VariableSelection(nn.Module):
    def __init__(self, n_vars: int, d: int) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.flatten = GRN(n_vars * d)
        self.score = nn.Linear(n_vars * d, n_vars)
        self.var_grns = nn.ModuleList([GRN(d) for _ in range(n_vars)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, V]
        B, L, V = x.shape
        emb = self.embed(x.unsqueeze(-1))           # [B, L, V, D]
        flat = emb.reshape(B, L, V * emb.size(-1))
        weights = torch.softmax(self.score(self.flatten(flat)), dim=-1)  # [B, L, V]
        per_var = torch.stack(
            [grn(emb[..., i, :]) for i, grn in enumerate(self.var_grns)], dim=2
        )                                             # [B, L, V, D]
        return (weights.unsqueeze(-1) * per_var).sum(dim=2)              # [B, L, D]


class TFT(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.var_sel = VariableSelection(n_channels, d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.grn = GRN(d_model, dropout)
        self.head = nn.Linear(d_model * seq_len, n_channels * pred_len)
        self.n_channels = n_channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]  ->  [B, L, C]
        z = x.transpose(1, 2)
        z = self.var_sel(z)                  # [B, L, D]
        z, _ = self.lstm(z)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        z = self.grn(z + attn_out)
        z = z.flatten(1)
        out = self.head(z).view(x.size(0), self.n_channels, self.pred_len)
        return out
