from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_adj(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    row_sum = A.sum(axis=1, keepdims=True)
    return A / np.maximum(row_sum, eps)


def build_corr_graph(X_train: np.ndarray, target_index: int = 0, topk: int = 10) -> np.ndarray:
    gwl = X_train[:, :, target_index]
    C = np.corrcoef(gwl.T)
    C = np.nan_to_num(np.abs(C), nan=0.0, posinf=0.0, neginf=0.0)
    N = C.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    k = max(0, min(int(topk), N - 1))
    for i in range(N):
        scores = C[i].copy()
        scores[i] = -np.inf
        if k > 0:
            idx = np.argpartition(scores, -k)[-k:]
            A[i, idx] = C[i, idx]
    A = np.maximum(A, A.T)
    A += np.eye(N, dtype=np.float32)
    return normalize_adj(A)


def save_adj(A: np.ndarray, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, A.astype(np.float32))


def load_adj(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.float32)
