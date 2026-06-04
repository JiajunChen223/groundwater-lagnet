from __future__ import annotations

import torch
from torch import nn


class ITransformerForecast(nn.Module):
    """Lightweight inverted Transformer baseline for multivariate node series."""

    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        input_window: int = 90,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.horizon = horizon
        self.token_count = num_nodes * input_dim
        self.value_proj = nn.Linear(input_window, d_model)
        self.token_emb = nn.Parameter(torch.randn(self.token_count, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)
        self.feature_proj = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, N, F = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B, N * F, L)
        h = self.value_proj(z) + self.token_emb[: N * F].unsqueeze(0)
        h = self.encoder(h)
        y_tokens = self.head(h).reshape(B, N, F, self.horizon).permute(0, 3, 1, 2)
        y = self.feature_proj(y_tokens).squeeze(-1)
        return y.unsqueeze(-1)

