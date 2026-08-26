"""End-to-end runner for the SAGE qualifying-position regression.

Loads the full enriched database (1950–2026), builds the causal round-state
graph, trains the single-readout SAGE regressor, and reports MAE/RMSE against
two leak-free baselines (global mean and per-driver trailing mean) on the
fixed year-based split (train 1950–2021, val 2022–2023, test 2024–2026).
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from relbench.datasets import get_dataset
from relbench.metrics import mae as relbench_mae, rmse as relbench_rmse

sys.path.append(os.path.abspath("src"))

import config as cfg
import data.tasks as data_tasks
from data.temporal_graph import build_temporal_graph
from models.sage_regressor import SageQualifyingRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def year_mask(years: torch.Tensor, allowed) -> torch.Tensor:
    allowed = set(allowed)
    return torch.tensor([int(y) in allowed for y in years.tolist()], dtype=torch.bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE qualifying-position regression")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # --- load database + graph -------------------------------------------------
    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    dataset_name = cfg.active_dataset_name()
    print(f"-> Loading database {dataset_name} ...")
    raw_dataset = get_dataset(dataset_name, download=False)
    db = raw_dataset.get_db(upto_test_timestamp=False)

    print("-> Building causal round-state graph ...")
    graph_data, node_to_col_names_dict, node_to_col_stats = build_temporal_graph(db)
    print(
        f"   nodes: " + ", ".join(f"{nt}={graph_data[nt].num_nodes}" for nt in graph_data.node_types)
    )

    # --- split by year ---------------------------------------------------------
    qual_year = graph_data["qualifying"].year
    y = graph_data["qualifying"].y
    driver_id = graph_data["qualifying"].driver_id

    train_mask = year_mask(qual_year, cfg.TRAIN_YEARS)
    val_mask = year_mask(qual_year, cfg.VAL_YEARS)
    test_mask = year_mask(qual_year, cfg.TEST_YEARS)
    print(
        f"   split: train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"test={int(test_mask.sum())}"
    )

    # --- model -----------------------------------------------------------------
    model = SageQualifyingRegressor(
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    tf_dict = {nt: graph_data[nt].tf.to(device) for nt in graph_data.node_types}
    edge_index_dict = {et: ei.to(device) for et, ei in graph_data.edge_index_dict.items()}
    y_dev = y.to(device)
    train_idx = train_mask.nonzero(as_tuple=True)[0].to(device)
    val_idx = val_mask.nonzero(as_tuple=True)[0].to(device)
    test_idx = test_mask.nonzero(as_tuple=True)[0].to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- train loop (full-batch) ----------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(tf_dict, edge_index_dict)
        loss = criterion(preds[train_idx], y_dev[train_idx])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                all_preds = model(tf_dict, edge_index_dict)
            val_mae = relbench_mae(
                y_dev[val_idx].cpu().numpy(), all_preds[val_idx].cpu().numpy()
            )
            print(f"epoch {epoch+1:3d} | train loss {loss.item():.4f} | val MAE {val_mae:.4f}")

    # --- test metrics ----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        preds = model(tf_dict, edge_index_dict).cpu().numpy()
    y_np = y.cpu().numpy()

    test_preds = preds[test_mask.numpy()]
    test_y = y_np[test_mask.numpy()]
    test_mae = relbench_mae(test_y, test_preds)
    test_rmse = relbench_rmse(test_y, test_preds)
    print(f"\nSAGE test MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")

    # --- baselines (leak-free: fitted on train only) ---------------------------
    y_train = y_np[train_mask.numpy()]
    driver_train = driver_id.numpy()[train_mask.numpy()]
    global_mean = float(y_train.mean())

    driver_mean = (
        pd.DataFrame({"driver_id": driver_train, "y": y_train})
        .groupby("driver_id")["y"]
        .mean()
    )
    test_driver = driver_id.numpy()[test_mask.numpy()]
    per_driver_pred = pd.Series(test_driver).map(driver_mean).fillna(global_mean).to_numpy()

    base_global_mae = relbench_mae(test_y, np.full_like(test_y, global_mean))
    base_global_rmse = relbench_rmse(test_y, np.full_like(test_y, global_mean))
    base_driver_mae = relbench_mae(test_y, per_driver_pred)
    base_driver_rmse = relbench_rmse(test_y, per_driver_pred)

    print(f"baseline global-mean  MAE {base_global_mae:.4f} | RMSE {base_global_rmse:.4f}")
    print(f"baseline per-driver   MAE {base_driver_mae:.4f} | RMSE {base_driver_rmse:.4f}")
    print(f"\nSAGE beats global-mean: {test_mae < base_global_mae} | "
          f"beats per-driver: {test_mae < base_driver_mae}")


if __name__ == "__main__":
    main()
