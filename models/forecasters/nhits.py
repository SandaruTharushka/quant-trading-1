"""N-HiTS (Challu et al. 2022) — multi-rate hierarchical interpolation forecaster."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NHiTSBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_pool: int,
        hidden: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_pool = n_pool
        in_dim = seq_len // n_pool
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(last, hidden), nn.ReLU()]
            last = hidden
        self.mlp = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden, in_dim)
        self.theta_f = nn.Linear(hidden, max(1, pred_len // n_pool))
        self.pred_len = pred_len
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L]
        x_p = F.avg_pool1d(x.unsqueeze(1), kernel_size=self.n_pool).squeeze(1)
        h = self.mlp(x_p)
        b_low = self.theta_b(h)
        f_low = self.theta_f(h)
        backcast = F.interpolate(b_low.unsqueeze(1), size=self.seq_len,
                                 mode="linear", align_corners=False).squeeze(1)
        forecast = F.interpolate(f_low.unsqueeze(1), size=self.pred_len,
                                 mode="linear", align_corners=False).squeeze(1)
        return backcast, forecast


class NHiTS(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        pred_len: int,
        pool_kernels: tuple[int, ...] = (8, 4, 1),
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.blocks = nn.ModuleList([
            NHiTSBlock(seq_len, pred_len, n_pool=k, hidden=hidden) for k in pool_kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]  -- channel-wise (independent)
        B, C, L = x.shape
        xc = x.reshape(B * C, L)
        forecast = torch.zeros(B * C, self.blocks[0].pred_len, device=x.device)
        residual = xc
        for blk in self.blocks:
            b, f = blk(residual)
            residual = residual - b
            forecast = forecast + f
        return forecast.reshape(B, C, -1)
