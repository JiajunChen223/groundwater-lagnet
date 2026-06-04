from __future__ import annotations

import torch
from torch import nn


class CNNLSTMForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        padding = kernel_size - 1
        self.conv = nn.Conv1d(input_dim, hidden_size, kernel_size=kernel_size, padding=padding)
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, N, F = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * N, F, L)
        z = torch.relu(self.conv(z))[..., :L].transpose(1, 2)
        out, _ = self.rnn(z)
        y = self.head(self.dropout(out[:, -1])).reshape(B, N, self.horizon)
        return y.permute(0, 2, 1).unsqueeze(-1)
