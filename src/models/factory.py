from __future__ import annotations

from typing import Any

from torch import nn

from .agcrn import AGCRNForecast
from .astgcn import ASTGCNForecast
from .dcrnn import DCRNNForecast
from .dlinear import DLinearForecast
from .graph_wavenet import GraphWaveNetForecast
from .itransformer import ITransformerForecast
from .hybrid import CNNLSTMForecast
from .lag_stgnet import LAGSTGNet
from .lstm import LSTMForecast
from .mtgnn import MTGNNForecast
from .patchtst_wrapper import PatchTSTWrapper
from .stgcn import STGCNForecast
from .tcn import TCNForecast


def meteo_indices_from_features(feature_names: list[str], selected_names: list[str] | None = None) -> list[int]:
    if selected_names is not None:
        selected = {name.lower() for name in selected_names}
        idx = [i for i, name in enumerate(feature_names) if name.lower() in selected]
        missing = selected - {feature_names[i].lower() for i in idx}
        if missing:
            raise ValueError(f"Unknown lag_feature_cols: {sorted(missing)}")
        return idx
    wanted = {
        "rain",
        "rainfall",
        "precip",
        "precipitation",
        "tp",
        "et",
        "eto",
        "e",
        "temp",
        "temperature",
        "hyras_pr",
        "dwd_evapo_r",
        "dwd_evapo_fao",
    }
    idx = [i for i, name in enumerate(feature_names) if name.lower() in wanted]
    if idx:
        return idx
    calendar_tokens = ("month_", "week_", "day_", "doy_", "season_")
    return [
        i
        for i, name in enumerate(feature_names)
        if i > 0 and not name.lower().startswith(calendar_tokens)
    ]


def build_model(
    name: str,
    input_dim: int,
    num_nodes: int,
    horizon: int,
    config: dict[str, Any],
    feature_names: list[str] | None = None,
) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.pop("name", None)
    model_cfg.pop("graph_topk", None)
    lag_feature_cols = model_cfg.pop("lag_feature_cols", None)
    if "graph" in config and "alpha" in config["graph"]:
        model_cfg.setdefault("graph_alpha", config["graph"]["alpha"])

    lower = name.lower()
    common = {"input_dim": input_dim, "num_nodes": num_nodes, "horizon": horizon}
    if lower == "lstm":
        return LSTMForecast(**common, **_pick(model_cfg, ["hidden_size", "num_layers", "dropout"]))
    if lower == "cnn_lstm":
        return CNNLSTMForecast(**common, **_pick(model_cfg, ["hidden_size", "num_layers", "kernel_size", "dropout"]))
    if lower == "tcn":
        return TCNForecast(**common, **_pick(model_cfg, ["d_model", "num_layers", "dropout"]))
    if lower == "dlinear":
        return DLinearForecast(**common, **_pick(model_cfg, ["input_window", "individual"]))
    if lower == "itransformer":
        return ITransformerForecast(**common, **_pick(model_cfg, ["input_window", "d_model", "num_layers", "nhead", "dropout"]))
    if lower == "stgcn":
        return STGCNForecast(**common, **_pick(model_cfg, ["d_model", "num_layers", "dropout"]))
    if lower == "dcrnn":
        return DCRNNForecast(**common, **_pick(model_cfg, ["d_model", "num_layers", "diffusion_steps", "dropout"]))
    if lower == "graph_wavenet":
        return GraphWaveNetForecast(**common, **_pick(model_cfg, ["d_model", "num_layers", "dropout"]))
    if lower == "agcrn":
        return AGCRNForecast(**common, **_pick(model_cfg, ["d_model", "num_layers", "dropout"]))
    if lower == "mtgnn":
        return MTGNNForecast(**common, **_pick(model_cfg, ["d_model", "num_layers", "hops", "dropout"]))
    if lower == "astgcn":
        return ASTGCNForecast(**common, **_pick(model_cfg, ["d_model", "num_layers", "nhead", "dropout"]))
    if lower == "patchtst":
        return PatchTSTWrapper(**common, **_pick(model_cfg, ["d_model", "dropout"]))
    if lower == "lag_stgnet":
        if feature_names is not None:
            model_cfg["meteo_indices"] = meteo_indices_from_features(feature_names, lag_feature_cols)
        return LAGSTGNet(**common, **_pick(model_cfg, [
            "d_model",
            "num_layers",
            "dropout",
            "lag_kernels",
            "meteo_indices",
            "graph_alpha",
            "use_lag",
            "use_graph",
            "use_learned_adj",
            "residual_connection",
            "target_index",
            "lag_type",
        ]))
    raise ValueError(f"Unsupported model: {name}")


def _pick(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}
