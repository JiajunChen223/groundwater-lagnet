from __future__ import annotations

import torch
from torch import nn


def build_loss(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "mae":
        return nn.L1Loss()
    if lowered == "mse":
        return nn.MSELoss()
    if lowered == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unsupported loss: {name}")

