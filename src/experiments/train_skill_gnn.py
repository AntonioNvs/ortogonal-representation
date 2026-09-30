"""Train SkillGNN on the causal round-state graph with Plackett-Luce ranking loss."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath("src"))

import config as cfg
import data.tasks as data_tasks
from baselines.skill_gnn_skill import (
    export_race_skills,
    get_skill_gnn_db,
    save_skill_gnn_encoder,
    season_skill_from_races,
)
from data.temporal_graph import build_temporal_graph
from models.ranking_likelihood import batch_pl_nll
from models.skill_gnn import SkillGNN
from relbench.datasets import get_dataset
from utils.device import get_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def year_mask(years: torch.Tensor, allowed) -> torch.Tensor:
    allowed = set(allowed)
    return torch.tensor([int(y) in allowed for y in years.tolist()], dtype=torch.bool)


def race_pl_loss_for_mask(
    model: SkillGNN,
    x_dict: dict,
    res,
    row_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    idx = row_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    race_ids = res.race_id[idx].to(device)
    positions = res.position[idx].to(device)
    driver_state_idx = res.driver_state_idx[idx].to(device)
    constructor_state_idx = res.constructor_state_idx[idx].to(device)
    grid = res.grid[idx].to(device)

    utilities_list = []
    ranks_list = []
    for rid in torch.unique(race_ids):
        rmask = race_ids == rid
        u, _ = model.race_utilities(
            x_dict,
            driver_state_idx[rmask],
            constructor_state_idx[rmask],
            grid[rmask],
        )
        utilities_list.append(u)
        ranks_list.append(positions[rmask])

    return batch_pl_nll(utilities_list, ranks_list)


def pairwise_accuracy(
    model: SkillGNN,
    x_dict: dict,
    res,
    row_mask: torch.Tensor,
    device: torch.device,
) -> float:
    idx = row_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return float("nan")

    correct = 0
    total = 0
    race_ids = res.race_id[idx].to(device)
    positions = res.position[idx].to(device)
    driver_state_idx = res.driver_state_idx[idx].to(device)
    constructor_state_idx = res.constructor_state_idx[idx].to(device)
    grid = res.grid[idx].to(device)

    for rid in torch.unique(race_ids):
        rmask = race_ids == rid
        u, _ = model.race_utilities(
            x_dict,
            driver_state_idx[rmask],
            constructor_state_idx[rmask],
            grid[rmask],
        )
        pos = positions[rmask]
        n = u.numel()
        for i in range(n):
            for j in range(i + 1, n):
                if pos[i] == pos[j]:
                    continue
                total += 1
                if (pos[i] < pos[j] and u[i] > u[j]) or (pos[j] < pos[i] and u[j] > u[i]):
                    correct += 1
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SkillGNN race-ranking model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--grid-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=cfg.DEFAULT_GPU_ID)
    parser.add_argument("--output-dir", type=str, default="output/skill_model")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.gpu_id)

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

    res = graph_data["results"]
    base_mask = res.in_ranking & (res.driver_state_idx >= 0) & (res.constructor_state_idx >= 0)
    train_mask = base_mask & year_mask(res.year, cfg.TRAIN_YEARS)
    val_mask = base_mask & year_mask(res.year, cfg.VAL_YEARS)
    test_mask = base_mask & year_mask(res.year, cfg.TEST_YEARS)
    print(
        f"   ranked results: train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"test={int(test_mask.sum())}"
    )

    model = SkillGNN(
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        grid_weight=args.grid_weight,
    ).to(device)

    tf_dict = {nt: graph_data[nt].tf.to(device) for nt in graph_data.node_types}
    edge_index_dict = {et: ei.to(device) for et, ei in graph_data.edge_index_dict.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_pl = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        x_dict = model.encode(tf_dict, edge_index_dict)
        train_loss = race_pl_loss_for_mask(model, x_dict, res, train_mask, device)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x_dict = model.encode(tf_dict, edge_index_dict)
            val_pl = race_pl_loss_for_mask(model, x_dict, res, val_mask, device)
            val_acc = pairwise_accuracy(model, x_dict, res, val_mask, device)

        if val_pl.item() < best_val_pl - 1e-4:
            best_val_pl = val_pl.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(
                f"epoch {epoch+1:3d} | train PL {train_loss.item():.4f} | "
                f"val PL {val_pl.item():.4f} | val pairwise acc {val_acc:.4f}",
                flush=True,
            )

        if epochs_no_improve >= args.patience:
            print(f"early stop at epoch {epoch+1} (best val PL {best_val_pl:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        x_dict = model.encode(tf_dict, edge_index_dict)
        test_pl = race_pl_loss_for_mask(model, x_dict, res, test_mask, device)
        test_acc = pairwise_accuracy(model, x_dict, res, test_mask, device)
    print(f"\nSkillGNN test PL {test_pl.item():.4f} | pairwise acc {test_acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "skill_gnn.pth")
    meta_path = os.path.join(args.output_dir, "skill_gnn_meta.json")
    torch.save(model.state_dict(), ckpt_path)
    enc_path = save_skill_gnn_encoder(ckpt_path, node_to_col_names_dict, node_to_col_stats)

    meta = {
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "grid_weight": args.grid_weight,
            "seed": args.seed,
        },
        "metrics": {
            "best_val_pl": best_val_pl,
            "test_pl": float(test_pl.item()),
            "test_pairwise_acc": float(test_acc),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    race_df = export_race_skills(model, graph_data, tf_dict, edge_index_dict, device)
    season_df = season_skill_from_races(race_df)
    season_df.to_csv(os.path.join(args.output_dir, "season_skill.csv"), index=False)
    print(f"wrote {ckpt_path}, {enc_path}, and {meta_path}")


if __name__ == "__main__":
    main()
