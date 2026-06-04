from __future__ import annotations

import torch
from torch import nn


class LSTMForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.rnn = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, N, F = x.shape
        z = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        out, _ = self.rnn(z)
        y = self.head(out[:, -1]).reshape(B, N, self.horizon)
        return y.permute(0, 2, 1).unsqueeze(-1)

