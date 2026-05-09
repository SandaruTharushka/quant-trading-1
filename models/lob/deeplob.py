"""DeepLOB (Zhang et al. 2019) — CNN + Inception + LSTM for limit order books.

Input shape: [B, 1, T, 40] for 10-level LOB (alternating price/size, bid/ask).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Inception(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1, 1)), nn.LeakyReLU(0.01),
            nn.Conv2d(out_c, out_c, (3, 1), padding=(1, 0)), nn.LeakyReLU(0.01),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1, 1)), nn.LeakyReLU(0.01),
            nn.Conv2d(out_c, out_c, (5, 1), padding=(2, 0)), nn.LeakyReLU(0.01),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_c, out_c, (1, 1)), nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class DeepLOB(nn.Module):
    def __init__(self, n_classes: int = 3, lob_levels: int = 10, lstm_hidden: int = 64) -> None:
        super().__init__()
        feat = lob_levels * 4  # price/size for bid/ask × levels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding=(1, 0)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding=(1, 0)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding=(1, 0)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding=(1, 0)), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (1, lob_levels // 2), stride=(1, lob_levels // 2)),
            nn.LeakyReLU(0.01),
        )
        self.inc = _Inception(32, 32)
        self.lstm = nn.LSTM(96, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)
        self.feat = feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T, F]
        z = self.conv(x)
        z = self.inc(z)               # [B, 96, T', 1]
        z = z.squeeze(-1).transpose(1, 2)  # [B, T', 96]
        z, _ = self.lstm(z)
        return self.head(z[:, -1])
