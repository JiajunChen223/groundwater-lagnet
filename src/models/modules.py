from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiSourceEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LagMeteorologicalModule(nn.Module):
    def __init__(self, meteo_dim: int, d_model: int, kernels: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(meteo_dim, d_model, kernel_size=k, padding=k - 1) for k in kernels]
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(d_model * len(kernels), d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(kernels)),
        )
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, meteo: torch.Tensor) -> torch.Tensor:
        B, L, N, M = meteo.shape
        flat = meteo.permute(0, 2, 3, 1).reshape(B * N, M, L)
        outs = []
        for conv in self.branches:
            z = conv(flat)[..., :L]
            outs.append(z.transpose(1, 2).reshape(B, N, L, -1).permute(0, 2, 1, 3))
        stacked = torch.stack(outs, dim=-2)
        pooled = torch.cat([o.mean(dim=(1, 2)) for o in outs], dim=-1)
        alpha = torch.softmax(self.weight_mlp(pooled), dim=-1).view(B, 1, 1, len(outs), 1)
        fused = (stacked * alpha).sum(dim=-2)
        return fused * self.gate(fused)


class StationAdaptiveLagModule(nn.Module):
    def __init__(
        self,
        meteo_dim: int,
        d_model: int,
        num_nodes: int,
        kernels: list[int],
        dropout: float = 0.0,
        emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(meteo_dim, d_model, kernel_size=k, padding=k - 1) for k in kernels]
        )
        self.node_emb = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.weight_mlp = nn.Sequential(
            nn.Linear(d_model * len(kernels) + emb_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(kernels)),
        )
        self.gate = nn.Sequential(nn.Linear(d_model + emb_dim, d_model), nn.Sigmoid())

    def forward(self, meteo: torch.Tensor) -> torch.Tensor:
        B, L, N, M = meteo.shape
        flat = meteo.permute(0, 2, 3, 1).reshape(B * N, M, L)
        outs = []
        for conv in self.branches:
            z = conv(flat)[..., :L]
            outs.append(z.transpose(1, 2).reshape(B, N, L, -1).permute(0, 2, 1, 3))
        stacked = torch.stack(outs, dim=-2)
        pooled = torch.cat([o.mean(dim=1) for o in outs], dim=-1)
        emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)
        alpha = torch.softmax(self.weight_mlp(torch.cat([pooled, emb], dim=-1)), dim=-1)
        alpha = alpha.view(B, 1, N, len(outs), 1)
        fused = (stacked * alpha).sum(dim=-2)
        gate_emb = emb[:, None, :, :].expand(-1, L, -1, -1)
        return fused * self.gate(torch.cat([fused, gate_emb], dim=-1))


class GraphConv(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h_graph = torch.einsum("ij,btjd->btid", adj, h)
        return self.dropout(F.relu(self.proj(h_graph)))


class AdaptiveGraphModule(nn.Module):
    def __init__(self, num_nodes: int, d_model: int, alpha: float = 0.5, use_learned_adj: bool = True) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.use_learned_adj = use_learned_adj
        self.emb1 = nn.Parameter(torch.randn(num_nodes, d_model) * 0.1)
        self.emb2 = nn.Parameter(torch.randn(num_nodes, d_model) * 0.1)

    def learned_adj(self) -> torch.Tensor:
        scores = F.relu(self.emb1 @ self.emb2.T)
        scores = scores + torch.eye(scores.shape[0], device=scores.device)
        return torch.softmax(scores, dim=-1)

    def forward(self, corr_adj: torch.Tensor | None) -> torch.Tensor:
        if not self.use_learned_adj:
            if corr_adj is None:
                raise ValueError("corr_adj is required when learned adjacency is disabled")
            return corr_adj
        A_learn = self.learned_adj()
        if corr_adj is None:
            return A_learn
        return self.alpha * corr_adj + (1.0 - self.alpha) * A_learn


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model: int, num_layers: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(layers)
        self.kernel_size = kernel_size

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, N, D = h.shape
        z = h.permute(0, 2, 3, 1).reshape(B * N, D, L)
        for layer in self.layers:
            z = layer(z)
            if isinstance(layer, nn.Conv1d):
                z = z[..., :L]
        return z.reshape(B, N, D, L).permute(0, 3, 1, 2)


class ForecastHead(nn.Module):
    def __init__(self, d_model: int, horizon: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, horizon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        last = h[:, -1]
        y = self.proj(last)
        return y.permute(0, 2, 1).unsqueeze(-1)
