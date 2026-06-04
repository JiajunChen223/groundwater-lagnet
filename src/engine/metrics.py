from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mae(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.nanmean(np.abs(yp - yt)))


def rmse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.sqrt(np.nanmean((yp - yt) ** 2)))


def r2_score(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    ss_res = np.nansum((yt - yp) ** 2)
    ss_tot = np.nansum((yt - np.nanmean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mape(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor, eps: float = 1e-6) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.nanmean(np.abs((yt - yp) / denom)) * 100.0)


def regression_metrics(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def paper_regression_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    paper_error_scale_cm: float,
) -> dict[str, float | str]:
    scale = float(paper_error_scale_cm)
    if scale <= 0:
        raise ValueError(f"paper_error_scale_cm must be positive, got {scale}")
    yt = _to_numpy(y_true) * scale
    yp = _to_numpy(y_pred) * scale
    return {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "R2": r2_score(yt, yp),
        "MAPE": mape(yt, yp),
        "Unit": "cm",
        "paper_error_scale_cm": scale,
    }


def direction_accuracy(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor, threshold: float = 0.0) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    true_dir = np.where(yt < -threshold, 0, np.where(yt > threshold, 2, 1))
    pred_dir = np.where(yp < -threshold, 0, np.where(yp > threshold, 2, 1))
    return float(np.mean(true_dir == pred_dir))


def macro_f1(y_true_cls: np.ndarray | torch.Tensor, y_pred_cls: np.ndarray | torch.Tensor, num_classes: int = 3) -> float:
    yt, yp = _to_numpy(y_true_cls).reshape(-1), _to_numpy(y_pred_cls).reshape(-1)
    scores: list[float] = []
    for cls in range(num_classes):
        tp = np.sum((yt == cls) & (yp == cls))
        fp = np.sum((yt != cls) & (yp == cls))
        fn = np.sum((yt == cls) & (yp != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(float(2 * precision * recall / (precision + recall)))
    return float(np.mean(scores))
