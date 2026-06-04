from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessedData:
    X: np.ndarray
    mask: np.ndarray
    dates: np.ndarray
    station_ids: np.ndarray
    feature_names: np.ndarray
    target_index: int
    train_end: int
    val_end: int
    mean: np.ndarray
    std: np.ndarray
    target_unit_to_cm: float
    paper_error_scale_cm: float


def _add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dayofyear = out[date_col].dt.dayofyear.to_numpy()
    out["month_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    out["month_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    return out


def _complete_daily_grid(df: pd.DataFrame, date_col: str, station_col: str) -> pd.DataFrame:
    dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    stations = pd.Index(sorted(df[station_col].astype(str).unique()), name=station_col)
    grid = pd.MultiIndex.from_product([dates, stations], names=[date_col, station_col]).to_frame(index=False)
    src = df.copy()
    src[station_col] = src[station_col].astype(str)
    return grid.merge(src, on=[date_col, station_col], how="left")


def _pivot_feature(df: pd.DataFrame, date_col: str, station_col: str, feature: str, stations: np.ndarray) -> np.ndarray:
    table = df.pivot(index=date_col, columns=station_col, values=feature)
    table = table.reindex(columns=stations)
    return table.to_numpy(dtype=np.float32)


def preprocess_groundwater(config: dict[str, Any]) -> ProcessedData:
    data_cfg = config["data"] if "data" in config else config
    raw_path = Path(data_cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    date_col = data_cfg.get("date_col", "date")
    station_col = data_cfg.get("station_col", "station_id")
    feature_cols = list(data_cfg["feature_cols"])
    target_col = data_cfg.get("target_col", "gwl")

    df = pd.read_csv(raw_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = _complete_daily_grid(df, date_col, station_col)
    df = _add_time_features(df, date_col)
    stations = np.array(sorted(df[station_col].astype(str).unique()), dtype=object)
    dates = np.array(sorted(df[date_col].unique()))
    split = data_cfg.get("split_ratio", [0.7, 0.1, 0.2])
    train_end = int(len(dates) * float(split[0]))
    val_end = train_end + int(len(dates) * float(split[1]))
    fill_method = data_cfg.get("fill_method", "interpolate")

    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for feature in feature_cols:
        values = _pivot_feature(df, date_col, station_col, feature, stations)
        valid = np.isfinite(values).astype(np.float32)
        filled = _fill_feature_values(values, train_end=train_end, fill_method=fill_method)
        arrays.append(filled)
        masks.append(valid)

    X = np.stack(arrays, axis=-1).astype(np.float32)
    mask = np.stack(masks, axis=-1).astype(np.float32)
    target_index = feature_cols.index(target_col)

    station_filter = data_cfg.get("station_filter", {})
    if station_filter.get("type") == "meteo_response":
        train_end_preview = int(len(dates) * float(data_cfg.get("split_ratio", [0.7, 0.1, 0.2])[0]))
        top_k = min(int(station_filter.get("top_k", X.shape[1])), X.shape[1])
        lags = [int(x) for x in station_filter.get("lags", [7, 14, 30, 60])]
        horizons = [int(x) for x in station_filter.get("horizons", [30, 60])]
        meteo_names = set(station_filter.get("meteo_cols", ["tp", "e", "rain", "et"]))
        meteo_indices = [i for i, name in enumerate(feature_cols) if name in meteo_names]
        scores = _meteo_response_scores(X[:train_end_preview], target_index, meteo_indices, lags, horizons)
        keep = np.argsort(scores)[-top_k:]
        keep = np.sort(keep)
        X = X[:, keep, :]
        mask = mask[:, keep, :]
        stations = stations[keep]
    elif station_filter.get("type") == "top_variance":
        top_k = min(int(station_filter.get("top_k", X.shape[1])), X.shape[1])
        variances = np.nanvar(X[:, :, target_index], axis=0)
        keep = np.argsort(variances)[-top_k:]
        keep = np.sort(keep)
        X = X[:, keep, :]
        mask = mask[:, keep, :]
        stations = stations[keep]

    train_slice = X[:train_end]
    mean = train_slice.reshape(-1, X.shape[-1]).mean(axis=0)
    std = train_slice.reshape(-1, X.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    target_unit_to_cm = float(data_cfg.get("target_unit_to_cm", 1.0))
    paper_error_scale_cm = float(std[target_index] * target_unit_to_cm)
    X = ((X - mean) / std).astype(np.float32)

    processed = ProcessedData(
        X=X,
        mask=mask,
        dates=dates.astype("datetime64[D]"),
        station_ids=stations,
        feature_names=np.array(feature_cols, dtype=object),
        target_index=target_index,
        train_end=train_end,
        val_end=val_end,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        target_unit_to_cm=target_unit_to_cm,
        paper_error_scale_cm=paper_error_scale_cm,
    )
    return processed


def _fill_feature_values(values: np.ndarray, train_end: int, fill_method: str) -> np.ndarray:
    if fill_method == "interpolate":
        filled = pd.DataFrame(values).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
        return _fill_remaining_with_column_means(filled, train_end)
    if fill_method == "causal_ffill":
        filled = pd.DataFrame(values).ffill().to_numpy(dtype=np.float32)
        return _fill_remaining_with_train_means(filled, train_end)
    raise ValueError(f"Unsupported fill_method: {fill_method}")


def _fill_remaining_with_column_means(values: np.ndarray, train_end: int) -> np.ndarray:
    if np.isfinite(values).all():
        return values.astype(np.float32)
    col_means = np.nanmean(values, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    inds = np.where(~np.isfinite(values))
    values[inds] = np.take(col_means, inds[1])
    return values.astype(np.float32)


def _fill_remaining_with_train_means(values: np.ndarray, train_end: int) -> np.ndarray:
    if np.isfinite(values).all():
        return values.astype(np.float32)
    train = values[:train_end]
    col_means = np.nanmean(train, axis=0)
    global_mean = np.nanmean(train)
    if not np.isfinite(global_mean):
        global_mean = 0.0
    col_means = np.where(np.isfinite(col_means), col_means, global_mean)
    inds = np.where(~np.isfinite(values))
    values[inds] = np.take(col_means, inds[1])
    return values.astype(np.float32)


def _meteo_response_scores(
    X_train: np.ndarray,
    target_index: int,
    meteo_indices: list[int],
    lags: list[int],
    horizons: list[int],
) -> np.ndarray:
    N = X_train.shape[1]
    scores = np.zeros(N, dtype=np.float32)
    counts = np.zeros(N, dtype=np.float32)
    if not meteo_indices:
        return np.nanvar(X_train[:, :, target_index], axis=0).astype(np.float32)

    target = X_train[:, :, target_index]
    T = target.shape[0]
    for horizon in horizons:
        for lag in lags:
            start = lag
            end = T - horizon
            if end <= start + 10:
                continue
            delta = target[start + horizon : T] - target[start:end]
            for meteo_idx in meteo_indices:
                meteo = X_train[start - lag : end - lag, :, meteo_idx]
                corr = _column_abs_corr(meteo, delta)
                scores += corr
                counts += np.isfinite(corr).astype(np.float32)
    scores = scores / np.maximum(counts, 1.0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _column_abs_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a_center = a - a.mean(axis=0, keepdims=True)
    b_center = b - b.mean(axis=0, keepdims=True)
    numerator = (a_center * b_center).sum(axis=0)
    denom = np.sqrt((a_center**2).sum(axis=0) * (b_center**2).sum(axis=0))
    return np.abs(numerator / np.maximum(denom, eps)).astype(np.float32)


def save_processed(data: ProcessedData, processed_dir: str | Path) -> Path:
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.npz"
    arrays: dict[str, np.ndarray] = {
        "X": data.X,
        "mask": data.mask,
        "dates": data.dates.astype("datetime64[D]").astype(str),
        "station_ids": data.station_ids.astype(str),
        "feature_names": data.feature_names.astype(str),
        "target_index": np.array(data.target_index, dtype=np.int64),
        "train_end": np.array(data.train_end, dtype=np.int64),
        "val_end": np.array(data.val_end, dtype=np.int64),
        "mean": data.mean,
        "std": data.std,
        "target_unit_to_cm": np.array(data.target_unit_to_cm, dtype=np.float32),
        "paper_error_scale_cm": np.array(data.paper_error_scale_cm, dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)
    return path


def prepare_from_config(config: dict[str, Any]) -> Path:
    data = preprocess_groundwater(config)
    data_cfg = config["data"] if "data" in config else config
    return save_processed(data, data_cfg["processed_dir"])
