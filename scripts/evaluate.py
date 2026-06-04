from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GroundwaterWindowDataset, load_processed_metadata
from src.data.graph import load_adj
from src.engine.metrics import direction_accuracy, paper_regression_metrics, regression_metrics
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final/frenchpiezo_main.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.horizon != 30:
        raise ValueError("This open package supports the final paper horizon only: 30 days.")

    config = load_config(args.config)
    model_name = args.model or config.get("model", {}).get("name", "lag_stgnet")
    processed_path = Path(config["data"]["processed_dir"]) / "data.npz"
    meta = load_processed_metadata(processed_path)
    feature_names = [str(x) for x in meta["feature_names"]]
    test_ds = GroundwaterWindowDataset(
        processed_path,
        config["data"]["input_window"],
        args.horizon,
        "test",
    )
    test_loader = DataLoader(test_ds, batch_size=int(config.get("train", {}).get("batch_size", 32)), shuffle=False)
    sample_x = test_ds[0][0]
    model = build_model(model_name, sample_x.shape[-1], sample_x.shape[-2], args.horizon, config, feature_names)
    device = torch.device("cuda" if config.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else sorted((Path(config.get("output", {}).get("save_dir", "outputs")) / "runs").glob(f"{model_name}_h{args.horizon}_*/checkpoint_best.pt"))[-1]
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    adj = torch.tensor(load_adj(Path(config["data"]["processed_dir"]) / "adj_corr.npy"), dtype=torch.float32, device=device)
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y, mask = batch[:3]
            out = model(x.to(device), adj=adj, mask=mask.to(device))
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    metrics = regression_metrics(y_true, y_pred)
    metrics["DirectionAcc"] = direction_accuracy(y_true[:, -1], y_pred[:, -1], threshold=0.0)
    paper_error_scale_cm = float(meta["paper_error_scale_cm"])
    paper_metrics = paper_regression_metrics(y_true, y_pred, paper_error_scale_cm)
    paper_metrics["DirectionAcc"] = metrics["DirectionAcc"]
    out_dir = ckpt_path.parent
    np.savez_compressed(out_dir / "predictions_eval.npz", y_true=y_true, y_pred=y_pred)
    save_json(metrics, out_dir / "metrics_eval.json")
    save_json(paper_metrics, out_dir / "metrics_eval_paper.json")
    print("metrics_eval.json uses the standardized target-delta scale.")
    print(f"metrics_eval_paper.json converts MAE/RMSE to cm with paper_error_scale_cm={paper_error_scale_cm:.8g}.")
    print(metrics)


if __name__ == "__main__":
    main()
