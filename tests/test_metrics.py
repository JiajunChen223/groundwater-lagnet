from __future__ import annotations

import numpy as np

from src.engine.metrics import mae, mape, paper_regression_metrics, r2_score, rmse


def test_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert r2_score(y, y) == 1.0
    assert mape(y, y) == 0.0


def test_metrics_nonzero_error() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    assert mae(y_true, y_pred) > 0.0
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


def test_paper_regression_metrics_convert_standardized_errors_to_cm() -> None:
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([1.0, 1.0, 4.0])
    metrics = paper_regression_metrics(y_true, y_pred, paper_error_scale_cm=25.0)
    assert metrics["Unit"] == "cm"
    assert metrics["MAE"] == mae(y_true, y_pred) * 25.0
    assert np.isclose(metrics["RMSE"], rmse(y_true, y_pred) * 25.0)
