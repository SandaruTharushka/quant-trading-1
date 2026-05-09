"""Recurrent baselines and CNN-LSTM hybrid."""
from __future__ import annotations

import torch
import torch.nn as nn


class _Recurrent(nn.Module):
    def __init__(self, cell, n_channels, hidden, pred_len, n_layers, dropout):
        super().__init__()
        self.rnn = cell(n_channels, hidden, num_layers=n_layers,
                        dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, n_channels * pred_len)
        self.n_channels = n_channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        out, _ = self.rnn(z)
        last = out[:, -1]
        return self.head(last).view(x.size(0), self.n_channels, self.pred_len)


class LSTMBaseline(_Recurrent):
    def __init__(self, n_channels, seq_len, pred_len, hidden=128, n_layers=2, dropout=0.1):
        super().__init__(nn.LSTM, n_channels, hidden, pred_len, n_layers, dropout)


class GRUBaseline(_Recurrent):
    def __init__(self, n_channels, seq_len, pred_len, hidden=128, n_layers=2, dropout=0.1):
        super().__init__(nn.GRU, n_channels, hidden, pred_len, n_layers, dropout)


class CNNLSTM(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len, conv=64, hidden=128, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, conv, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv, conv, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv, hidden, batch_first=True, dropout=dropout, num_layers=2)
        self.head = nn.Linear(hidden, n_channels * pred_len)
        self.n_channels = n_channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.head(out[:, -1]).view(x.size(0), self.n_channels, self.pred_len)
