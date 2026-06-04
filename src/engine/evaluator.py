from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import regression_metrics
from ..utils.io import save_json


def evaluate_predictions(prediction_path: str | Path, output_path: str | Path | None = None) -> dict[str, float]:
    data = np.load(prediction_path)
    metrics = regression_metrics(data["y_true"], data["y_pred"])
    if output_path is not None:
        save_json(metrics, output_path)
    return metrics

