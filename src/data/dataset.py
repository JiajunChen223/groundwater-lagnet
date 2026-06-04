from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

SplitName = Literal["train", "val", "test"]


class GroundwaterWindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        processed_path: str | Path,
        input_window: int,
        horizon: int,
        split: SplitName,
    ) -> None:
        data = np.load(processed_path, allow_pickle=True)
        self.X = data["X"].astype(np.float32)
        self.mask = data["mask"].astype(np.float32)
        self.target_index = int(data["target_index"])
        self.input_window = int(input_window)
        self.horizon = int(horizon)
        self.split = split
        self.indices = self._build_indices(int(data["train_end"]), int(data["val_end"]))

    def _build_indices(self, train_end: int, val_end: int) -> list[int]:
        indices: list[int] = []
        T = self.X.shape[0]
        for target_start in range(self.input_window, T - self.horizon + 1):
            target_end = target_start + self.horizon
            if self.split == "train" and target_end <= train_end:
                indices.append(target_start)
            elif self.split == "val" and target_start >= train_end and target_end <= val_end:
                indices.append(target_start)
            elif self.split == "test" and target_start >= val_end:
                indices.append(target_start)
        if not indices:
            raise ValueError(f"No samples for split={self.split}, window={self.input_window}, horizon={self.horizon}")
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        target_start = self.indices[idx]
        x = self.X[target_start - self.input_window : target_start]
        m = self.mask[target_start - self.input_window : target_start]
        last = self.X[target_start - 1 : target_start, :, self.target_index : self.target_index + 1]
        y = self.X[target_start : target_start + self.horizon, :, self.target_index : self.target_index + 1]
        return torch.from_numpy(x), torch.from_numpy(y - last), torch.from_numpy(m)


def load_processed_metadata(processed_path: str | Path) -> dict[str, np.ndarray | int | float]:
    data = np.load(processed_path, allow_pickle=True)
    required = {"target_unit_to_cm", "paper_error_scale_cm"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(
            "Processed data is missing paper-scale metadata "
            f"{missing}. Re-run scripts/prepare_data.py with the final config."
        )
    meta: dict[str, np.ndarray | int] = {
        "dates": data["dates"],
        "station_ids": data["station_ids"],
        "feature_names": data["feature_names"],
        "target_index": int(data["target_index"]),
        "train_end": int(data["train_end"]),
        "val_end": int(data["val_end"]),
        "mean": data["mean"],
        "std": data["std"],
        "target_unit_to_cm": float(data["target_unit_to_cm"]),
        "paper_error_scale_cm": float(data["paper_error_scale_cm"]),
    }
    return meta
