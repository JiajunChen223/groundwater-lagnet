from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import build_loss
from .metrics import direction_accuracy, paper_regression_metrics, regression_metrics
from ..utils.config import save_config
from ..utils.io import save_json


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        run_dir: str | Path,
        device: str = "cuda",
        adj: np.ndarray | None = None,
        paper_error_scale_cm: float | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.adj = torch.tensor(adj, dtype=torch.float32, device=self.device) if adj is not None else None
        self.paper_error_scale_cm = paper_error_scale_cm
        train_cfg = config.get("train", {})
        self.loss_fn = build_loss(train_cfg.get("loss", "huber"))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        self.best_path = self.run_dir / "checkpoint_best.pt"
        save_config(config, self.run_dir / "config.yaml")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        train_cfg = self.config.get("train", {})
        epochs = int(train_cfg.get("epochs", 100))
        patience = int(train_cfg.get("patience", 15))
        best_val = float("inf")
        wait = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch, "val_loss": val_loss}, self.best_path)
            else:
                wait += 1
                if wait >= patience:
                    break

        save_json({"history": history, "best_val_loss": best_val}, self.run_dir / "train_history.json")
        return {"best_val_loss": best_val, "epochs_ran": len(history)}

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        losses: list[float] = []
        for batch in loader:
            x, y, mask = batch
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)
            with torch.set_grad_enabled(train):
                out = self.model(x, adj=self.adj, mask=mask)
                loss = self.loss_fn(out, y)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def load_best(self) -> None:
        ckpt = torch.load(self.best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []
        for batch in loader:
            x, y, mask = batch
            x = x.to(self.device)
            mask = mask.to(self.device)
            pred = self.model(x, adj=self.adj, mask=mask)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.numpy())
        return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)

    def test(self, loader: DataLoader, save_predictions: bool = True) -> dict[str, float]:
        self.load_best()
        y_true, y_pred = self.predict(loader)
        metrics = regression_metrics(y_true, y_pred)
        metrics["DirectionAcc"] = direction_accuracy(y_true[:, -1], y_pred[:, -1], threshold=0.0)
        save_json(metrics, self.run_dir / "metrics.json")
        if self.paper_error_scale_cm is not None:
            paper_metrics = paper_regression_metrics(y_true, y_pred, self.paper_error_scale_cm)
            paper_metrics["DirectionAcc"] = metrics["DirectionAcc"]
            save_json(paper_metrics, self.run_dir / "metrics_paper.json")
        if save_predictions:
            np.savez_compressed(self.run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)
        return metrics
