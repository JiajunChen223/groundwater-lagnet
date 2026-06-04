from __future__ import annotations

import torch

from src.models.factory import build_model, meteo_indices_from_features


def test_default_meteo_indices_exclude_calendar_features() -> None:
    feature_names = ["GWL", "HYRAS_pr", "DWD_evapo_r", "month_sin", "month_cos"]
    assert meteo_indices_from_features(feature_names) == [1, 2]


def test_explicit_lag_feature_cols_include_calendar_features() -> None:
    feature_names = ["GWL", "HYRAS_pr", "DWD_evapo_r", "month_sin", "month_cos"]
    selected = ["HYRAS_pr", "DWD_evapo_r", "month_sin", "month_cos"]
    assert meteo_indices_from_features(feature_names, selected) == [1, 2, 3, 4]


def test_model_shapes() -> None:
    config = {
        "model": {
            "d_model": 16,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "lag_kernels": [3, 7],
            "use_lag": True,
            "use_graph": True,
            "use_learned_adj": True,
            "lag_type": "station",
        },
        "graph": {"alpha": 0.5},
    }
    x = torch.randn(2, 24, 5, 5)
    adj = torch.eye(5)
    config["model"]["input_window"] = 24
    for name in [
        "lstm",
        "cnn_lstm",
        "tcn",
        "dlinear",
        "itransformer",
        "stgcn",
        "dcrnn",
        "graph_wavenet",
        "agcrn",
        "mtgnn",
        "astgcn",
        "patchtst",
        "lag_stgnet",
    ]:
        model = build_model(name, input_dim=5, num_nodes=5, horizon=30, config=config, feature_names=["gwl", "rain", "et", "month_sin", "month_cos"])
        y = model(x, adj=adj)
        assert y.shape == (2, 30, 5, 1)
