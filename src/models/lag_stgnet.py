from __future__ import annotations

import torch
from torch import nn

from .modules import (
    AdaptiveGraphModule,
    ForecastHead,
    GraphConv,
    LagMeteorologicalModule,
    MultiSourceEncoder,
    StationAdaptiveLagModule,
    TemporalConvBlock,
)


class LAGSTGNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        d_model: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        lag_kernels: list[int] | None = None,
        meteo_indices: list[int] | None = None,
        graph_alpha: float = 0.5,
        use_lag: bool = True,
        use_graph: bool = True,
        use_learned_adj: bool = True,
        residual_connection: bool = True,
        target_index: int = 0,
        lag_type: str = "global",
    ) -> None:
        super().__init__()
        self.use_lag = use_lag
        self.use_graph = use_graph
        self.residual_connection = residual_connection
        self.target_index = target_index
        self.lag_type = lag_type
        self.meteo_indices = meteo_indices or [i for i in range(1, input_dim)]
        self.encoder = MultiSourceEncoder(input_dim, d_model, dropout)
        if use_lag:
            kernels = lag_kernels or [3, 7, 14, 30]
            if lag_type == "station":
                self.lag = StationAdaptiveLagModule(len(self.meteo_indices), d_model, num_nodes, kernels, dropout)
            elif lag_type == "global":
                self.lag = LagMeteorologicalModule(len(self.meteo_indices), d_model, kernels, dropout)
            else:
                raise ValueError(f"Unsupported lag_type: {lag_type}")
        else:
            self.lag = None
        if use_graph:
            self.graph_builder = AdaptiveGraphModule(num_nodes, d_model, graph_alpha, use_learned_adj)
            self.graph_conv = GraphConv(d_model, dropout)
        else:
            self.graph_builder = None
            self.graph_conv = None
        self.temporal = TemporalConvBlock(d_model, num_layers, kernel_size=3, dropout=dropout)
        self.head = ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x)
        if self.use_lag and self.lag is not None:
            meteo = x[..., self.meteo_indices]
            h = h + self.lag(meteo)
        if self.use_graph and self.graph_builder is not None and self.graph_conv is not None:
            A = self.graph_builder(adj)
            h = h + self.graph_conv(h, A)
        h = self.temporal(h)
        y = self.head(h)
        if self.residual_connection:
            last_value = x[:, -1:, :, self.target_index : self.target_index + 1]
            y = y + last_value.repeat(1, y.shape[1], 1, 1)
        return y
