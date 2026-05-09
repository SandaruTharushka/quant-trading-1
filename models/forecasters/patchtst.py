"""PatchTST (Nie et al. 2023) — patch + channel-independent transformer for time series."""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = x.unfold(-1, self.patch_len, self.stride)  # [B, C, N, P]
        return self.proj(x)


class PatchTST(nn.Module):
    """Channel-independent PatchTST. Each channel attends across patches."""

    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        head: str = "regression",  # regression | distribution
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.pred_len = pred_len
        self.patch = PatchEmbedding(patch_len, stride, d_model)
        n_patches = (seq_len - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, n_patches, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        self.head_kind = head
        out_dim = pred_len * (2 if head == "distribution" else 1)
        self.head = nn.Linear(d_model * n_patches, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        z = self.patch(x) + self.pos_embed                    # [B, C, N, D]
        B, C, N, D = z.shape
        z = z.reshape(B * C, N, D)
        z = self.encoder(z)                                    # [B*C, N, D]
        z = z.reshape(B, C, N * D)
        out = self.head(z)                                     # [B, C, pred_len * k]
        if self.head_kind == "distribution":
            mu, log_sigma = out.chunk(2, dim=-1)
            return torch.stack([mu, log_sigma], dim=-1)        # [B, C, H, 2]
        return out                                             # [B, C, H]
