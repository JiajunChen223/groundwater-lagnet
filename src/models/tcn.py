from __future__ import annotations

import torch
from torch import nn

from .modules import ForecastHead, MultiSourceEncoder, TemporalConvBlock


class TCNForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        d_model: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, d_model, dropout)
        self.temporal = TemporalConvBlock(d_model, num_layers, kernel_size, dropout)
        self.head = ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x)
        h = self.temporal(h)
        return self.head(h)
