from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import GroundwaterWindowDataset, load_processed_metadata
from src.data.graph import build_corr_graph, load_adj, save_adj
from src.data.preprocess import prepare_from_config
from src.engine.trainer import Trainer
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def ensure_processed(config: dict) -> Path:
    processed_dir = Path(config["data"]["processed_dir"])
    data_path = processed_dir / "data.npz"
    adj_path = processed_dir / "adj_corr.npy"
    if not data_path.exists():
        prepare_from_config(config)
    if not adj_path.exists():
        data = np.load(data_path, allow_pickle=True)
        adj = build_corr_graph(data["X"][: int(data["train_end"])], int(data["target_index"]), int(config.get("graph", {}).get("topk", 15)))
        save_adj(adj, adj_path)
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final/frenchpiezo_main.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()
    if args.horizon != 30:
        raise ValueError("This open package supports the final paper horizon only: 30 days.")

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.save_dir is not None:
        config.setdefault("output", {})["save_dir"] = args.save_dir
    set_seed(int(config.get("seed", 2022)))
    model_name = args.model or config.get("model", {}).get("name", "lag_stgnet")
    processed_path = ensure_processed(config)
    meta = load_processed_metadata(processed_path)
    feature_names = [str(x) for x in meta["feature_names"]]

    train_ds = GroundwaterWindowDataset(
        processed_path,
        config["data"]["input_window"],
        args.horizon,
        "train",
    )
    val_ds = GroundwaterWindowDataset(
        processed_path,
        config["data"]["input_window"],
        args.horizon,
        "val",
    )
    test_ds = GroundwaterWindowDataset(
        processed_path,
        config["data"]["input_window"],
        args.horizon,
        "test",
    )
    train_cfg = config.get("train", {})
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 32)), shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg.get("batch_size", 32)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))
    test_loader = DataLoader(test_ds, batch_size=int(train_cfg.get("batch_size", 32)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    sample_x = train_ds[0][0]
    model = build_model(model_name, sample_x.shape[-1], sample_x.shape[-2], args.horizon, config, feature_names)
    adj = load_adj(Path(config["data"]["processed_dir"]) / "adj_corr.npy")
    run_dir = (
        Path(config.get("output", {}).get("save_dir", "outputs"))
        / "runs"
        / f"{model_name}_h{args.horizon}_s{int(config.get('seed', 2022))}_{int(time.time())}"
    )
    paper_error_scale_cm = float(meta["paper_error_scale_cm"])
    trainer = Trainer(
        model,
        config,
        run_dir,
        device=config.get("device", "cuda"),
        adj=adj,
        paper_error_scale_cm=paper_error_scale_cm,
    )
    trainer.fit(train_loader, val_loader)
    metrics = trainer.test(test_loader)
    print("metrics.json uses the standardized target-delta scale.")
    print(f"metrics_paper.json converts MAE/RMSE to cm with paper_error_scale_cm={paper_error_scale_cm:.8g}.")
    print(metrics)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
