from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import GroundwaterWindowDataset, load_processed_metadata
from src.data.preprocess import prepare_from_config


def write_synthetic_csv(path: Path, days: int = 180, stations: int = 4) -> None:
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for station in range(stations):
        phase = station / max(stations, 1)
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "station_id": f"W{station:03d}",
                    "gwl": 10.0 + phase + 0.2 * np.sin(i / 12.0) + 0.01 * i,
                    "rain": max(0.0, np.sin(i / 5.0 + phase)),
                    "et": 1.5 + 0.1 * np.cos(i / 8.0 + phase),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_dataset_shapes_and_time_splits(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    write_synthetic_csv(raw_path)
    cfg = {
        "data": {
            "raw_path": str(raw_path),
            "processed_dir": str(tmp_path / "processed"),
            "date_col": "date",
            "station_col": "station_id",
            "target_col": "gwl",
            "feature_cols": ["gwl", "rain", "et", "month_sin", "month_cos"],
            "input_window": 30,
            "split_ratio": [0.7, 0.1, 0.2],
            "target_unit_to_cm": 10.0,
        }
    }
    processed = prepare_from_config(cfg)
    meta = load_processed_metadata(processed)
    assert meta["target_unit_to_cm"] == 10.0
    assert float(meta["paper_error_scale_cm"]) > 0.0
    ds = GroundwaterWindowDataset(processed, input_window=30, horizon=30, split="train")
    x, y, mask = ds[0]
    assert x.shape == (30, 4, 5)
    assert y.shape == (30, 4, 1)
    assert mask.shape == (30, 4, 5)
    assert max(ds.indices) + 30 <= 126

    expected = ds.X[30:60, :, :1] - ds.X[29:30, :, :1]
    assert np.allclose(y.numpy(), expected)


def test_causal_ffill_uses_train_statistics_for_leading_gaps(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    write_synthetic_csv(raw_path, days=120, stations=3)
    df = pd.read_csv(raw_path)
    df.loc[(df["station_id"] == "W000") & (df.index < 6), "gwl"] = np.nan
    df.to_csv(raw_path, index=False)
    cfg = {
        "data": {
            "raw_path": str(raw_path),
            "processed_dir": str(tmp_path / "processed_causal"),
            "date_col": "date",
            "station_col": "station_id",
            "target_col": "gwl",
            "feature_cols": ["gwl", "rain", "et", "month_sin", "month_cos"],
            "input_window": 30,
            "split_ratio": [0.7, 0.1, 0.2],
            "fill_method": "causal_ffill",
            "target_unit_to_cm": 1.0,
        }
    }
    processed = prepare_from_config(cfg)
    data = np.load(processed, allow_pickle=True)
    assert np.isfinite(data["X"]).all()
    assert float(data["paper_error_scale_cm"]) > 0.0
