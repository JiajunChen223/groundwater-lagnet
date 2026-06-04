from __future__ import annotations

import torch
from torch import nn

from .modules import AdaptiveGraphModule, ForecastHead, GraphConv, MultiSourceEncoder, TemporalConvBlock


class SimpleGraphTemporalForecast(nn.Module):
    """API-compatible graph temporal baseline used as a lightweight starter."""

    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_learned_adj: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, d_model, dropout)
        self.graph_builder = AdaptiveGraphModule(num_nodes, d_model, alpha=0.7, use_learned_adj=use_learned_adj)
        self.graph = GraphConv(d_model, dropout)
        self.temporal = TemporalConvBlock(d_model, num_layers, kernel_size=3, dropout=dropout)
        self.head = ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x)
        h = h + self.graph(h, self.graph_builder(adj))
        h = self.temporal(h)
        return self.head(h)


class STGCNForecast(SimpleGraphTemporalForecast):
    pass


class GraphWaveNetForecast(SimpleGraphTemporalForecast):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_learned_adj"] = True
        super().__init__(*args, **kwargs)


class AGCRNForecast(SimpleGraphTemporalForecast):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_learned_adj"] = True
        super().__init__(*args, **kwargs)


class DiffusionGraphConv(nn.Module):
    def __init__(self, d_model: int, diffusion_steps: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.proj = nn.Linear(d_model * (self.diffusion_steps + 1), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        supports = [h]
        z = h
        for _ in range(self.diffusion_steps):
            if z.dim() == 3:
                z = torch.einsum("ij,bjd->bid", adj, z)
            else:
                z = torch.einsum("ij,btjd->btid", adj, z)
            supports.append(z)
        return self.dropout(torch.relu(self.proj(torch.cat(supports, dim=-1))))


class GraphGRUCell(nn.Module):
    def __init__(self, d_model: int, diffusion_steps: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.diffusion = DiffusionGraphConv(d_model, diffusion_steps, dropout)
        self.gru = nn.GRUCell(d_model, d_model)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z = self.diffusion(x_t + h_t, adj)
        B, N, D = z.shape
        return self.gru(z.reshape(B * N, D), h_t.reshape(B * N, D)).reshape(B, N, D)


class DCRNNForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        d_model: int = 64,
        num_layers: int = 1,
        diffusion_steps: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, d_model, dropout)
        self.cells = nn.ModuleList([GraphGRUCell(d_model, diffusion_steps, dropout) for _ in range(num_layers)])
        self.head = ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        if adj is None:
            raise ValueError("DCRNNForecast requires an adjacency matrix")
        z = self.encoder(x)
        B, L, N, D = z.shape
        states = [torch.zeros(B, N, D, device=x.device, dtype=z.dtype) for _ in self.cells]
        outputs = []
        for t in range(L):
            inp = z[:, t]
            for i, cell in enumerate(self.cells):
                states[i] = cell(inp, states[i], adj)
                inp = states[i]
            outputs.append(inp)
        h = torch.stack(outputs, dim=1)
        return self.head(h)


class MixHopGraphConv(nn.Module):
    def __init__(self, d_model: int, hops: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.hops = int(hops)
        self.proj = nn.Linear(d_model * (self.hops + 1), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        outs = [h]
        z = h
        for _ in range(self.hops):
            z = torch.einsum("ij,btjd->btid", adj, z)
            outs.append(z)
        return self.dropout(torch.relu(self.proj(torch.cat(outs, dim=-1))))


class MTGNNForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        d_model: int = 64,
        num_layers: int = 3,
        hops: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, d_model, dropout)
        self.graph_builder = AdaptiveGraphModule(num_nodes, d_model, alpha=0.5, use_learned_adj=True)
        self.mixhop = MixHopGraphConv(d_model, hops=hops, dropout=dropout)
        self.temporal = TemporalConvBlock(d_model, num_layers, kernel_size=3, dropout=dropout)
        self.head = ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x)
        A = self.graph_builder(adj)
        h = h + self.mixhop(h, A)
        h = self.temporal(h)
        return self.head(h)


class ASTGCNForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon: int,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, d_model, dropout)
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.graph = GraphConv(d_model, dropout)
        self.temporal = TemporalConvBlock(d_model, num_layers, kernel_size=3, dropout=dropout)
        self.head = ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        if adj is None:
            raise ValueError("ASTGCNForecast requires an adjacency matrix")
        h = self.encoder(x)
        B, L, N, D = h.shape
        z = h.permute(0, 2, 1, 3).reshape(B * N, L, D)
        z, _ = self.temporal_attn(z, z, z, need_weights=False)
        h = z.reshape(B, N, L, D).permute(0, 2, 1, 3)
        h = h + self.graph(h, adj)
        h = self.temporal(h)
        return self.head(h)


class PatchTSTWrapper(nn.Module):
    """Small TCN fallback with the same project interface as future PatchTST."""

    def __init__(self, input_dim: int, num_nodes: int, horizon: int, d_model: int = 64, dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, d_model, dropout)
        self.temporal = TemporalConvBlock(d_model, num_layers=3, kernel_size=5, dropout=dropout)
        self.head = ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.temporal(self.encoder(x)))
