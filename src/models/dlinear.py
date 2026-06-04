from __future__ import annotations

import torch
from torch import nn


class DLinearForecast(nn.Module):
    def __init__(self, input_dim: int, num_nodes: int, horizon: int, input_window: int = 90, individual: bool = False) -> None:
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.individual = individual
        if individual:
            self.linear = nn.ModuleList([nn.Linear(input_window, horizon) for _ in range(input_dim)])
        else:
            self.linear = nn.Linear(input_window, horizon)
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, N, F] -> [B, N, F, L]
        z = x.permute(0, 2, 3, 1)
        if self.individual:
            outs = [layer(z[:, :, i, :]) for i, layer in enumerate(self.linear)]
            y_feat = torch.stack(outs, dim=-1)
        else:
            y_feat = self.linear(z).permute(0, 1, 3, 2)
        y = self.proj(y_feat).squeeze(-1)
        return y.permute(0, 2, 1).unsqueeze(-1)

