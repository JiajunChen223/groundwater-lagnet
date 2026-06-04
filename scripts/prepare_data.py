from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.graph import build_corr_graph, save_adj
from src.data.preprocess import prepare_from_config
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final/frenchpiezo_main.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_path = prepare_from_config(config)
    data = np.load(processed_path, allow_pickle=True)
    train_end = int(data["train_end"])
    target_index = int(data["target_index"])
    topk = int(config.get("graph", {}).get("topk", 15))
    adj = build_corr_graph(data["X"][:train_end], target_index=target_index, topk=topk)
    adj_path = Path(config["data"]["processed_dir"]) / "adj_corr.npy"
    save_adj(adj, adj_path)
    print(f"saved processed data: {processed_path}")
    print(f"saved corr graph: {adj_path}")


if __name__ == "__main__":
    main()
