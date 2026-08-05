"""
Walk-forward temporal curve experiment.

For each target season, train on historical years + races 1..k, then evaluate
MAE (mean absolute error on the active regression target -- position,
positionOrder or points, per cfg.TASK_NAME) on race k+1. Produces a curve of
MAE vs number of races used.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import seaborn as sns
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import config as cfg
import train
from train import (
    DEFAULT_MODEL_CONFIGS,
    add_edge_round_masks,
    get_device,
    get_race_metadata,
    load_db_and_graph,
    prepare_curve_step,
    set_global_seed,
    train_and_evaluate,
)

CURVE_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#9467bd", "#e15759"]


def parse_target_years(value):
    if not value:
        return list(cfg.TEMPORAL_CURVE_TARGET_YEARS)
    return [int(y.strip()) for y in str(value).split(",") if y.strip()]


def verify_no_leakage(db, graph_data, target_year, k):
    """Assert round k+1 edges are excluded from the step-k edge mask."""
    masks = add_edge_round_masks(db, graph_data, target_year, k)
    _, rounds_by_year = get_race_metadata(db)
    future_round = k + 1
    if future_round not in rounds_by_year.get(target_year, []):
        return True

    races_df = db.table_dict["races"].df
    future_race_ids = set(
        races_df[
            (races_df["year"] == target_year) & (races_df["round"] == future_round)
        ]["raceId"].tolist()
    )
    if not future_race_ids:
        return True

    for edge_type in graph_data.edge_types:
        src_table = edge_type[0]
        if src_table not in db.table_dict:
            continue
        src_df = db.table_dict[src_table].df
        if "raceId" not in src_df.columns:
            continue

        mask = masks[edge_type]
        edge_index = graph_data[edge_type].edge_index
        src_node_ids = edge_index[0].cpu().numpy()
        visible_race_ids = src_df.iloc[src_node_ids[mask]]["raceId"].values
        leaked = future_race_ids.intersection(set(visible_race_ids.tolist()))
        if leaked:
            raise AssertionError(
                f"Leakage at year={target_year}, k={k}: "
                f"round {future_round} edges visible in {edge_type}"
            )
    return True


def run_temporal_curve(
    target_years,
    epochs=10,
    seed=42,
    output_dir="output/temporal_curve/",
    deterministic=False,
    max_k=None,
    gpu_id=None,
):
    model_cfg = DEFAULT_MODEL_CONFIGS[cfg.TEMPORAL_CURVE_MODEL]
    set_global_seed(seed, deterministic=deterministic)

    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    db, graph_data, node_to_col_names_dict, node_to_col_stats, instances_df, task = (
        load_db_and_graph()
    )
    device = get_device(gpu_id)
    graph_data = graph_data.to(device)

    _, rounds_by_year = get_race_metadata(db)

    for target_year in target_years:
        season_rounds = rounds_by_year.get(target_year, [])
        if len(season_rounds) < 2:
            print(f"Skipping {target_year}: fewer than 2 rounds")
            continue

        max_step = len(season_rounds) - 1
        if max_k is not None:
            max_step = min(max_step, max_k)

        print(f"\n=== Target season {target_year} | steps k=1..{max_step} ===")

        for k in range(1, max_step + 1):
            eval_round = k + 1
            print(
                f"\n--- {target_year} | k={k} (train rounds 1..{k}, "
                f"eval round {eval_round}) ---"
            )

            verify_no_leakage(db, graph_data, target_year, k)

            train_loader, val_loader, eval_loader, edge_index_dict, eval_df = (
                prepare_curve_step(db, graph_data, instances_df, target_year, k)
            )

            if len(eval_df) == 0:
                print(f"  Skipping k={k}: no instances for round {eval_round}")
                continue

            edge_index_dict = {et: ei.to(device) for et, ei in edge_index_dict.items()}

            result = train_and_evaluate(
                name=f"model_orthogonal_curve_{target_year}_k{k}",
                lambda_orthogonal=model_cfg["lambda_orthogonal"],
                aux_weight=model_cfg.get("aux_weight", 0.5),
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=eval_loader,
                graph_data=graph_data,
                node_to_col_names_dict=node_to_col_names_dict,
                node_to_col_stats=node_to_col_stats,
                device=device,
                epochs=epochs,
                train_edge_index_dict=edge_index_dict,
                val_edge_index_dict=edge_index_dict,
                test_edge_index_dict=edge_index_dict,
                save_model=False,
                task=task,
            )

            row = {
                "target_year": target_year,
                "k": k,
                "eval_round": eval_round,
                "n_train": len(train_loader.dataset),
                "n_eval": len(eval_df),
                "mae": result["test_metrics"]["mae"],
                "rmse": result["test_metrics"]["rmse"],
                "r2": result["test_metrics"]["r2"],
                "spearman": result["test_metrics"]["spearman"],
                "auroc_top3": result["test_metrics"]["auroc_top3"],
                "loss": result["test_metrics"]["loss"],
                "main": result["test_metrics"]["main"],
                "orth": result["test_metrics"]["orth"],
                "cos_global": result["test_metrics"]["cos_global"],
            }
            all_rows.append(row)
            print(f"  MAE (round {eval_round}): {row['mae']:.4f}")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_years": target_years,
        "epochs": epochs,
        "seed": seed,
        "gpu_id": gpu_id if gpu_id is not None else cfg.DEFAULT_GPU_ID,
        "model_config": model_cfg,
        "max_k": max_k,
    }

    csv_path = os.path.join(output_dir, "temporal_curve_results.csv")
    json_path = os.path.join(output_dir, "temporal_curve_results.json")
    plot_path = os.path.join(output_dir, "temporal_curve.png")

    write_csv(all_rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": all_rows}, f, indent=2)

    if all_rows:
        save_plot(all_rows, plot_path, target_years)

    print(f"\nResults saved to {output_dir}")
    return all_rows, metadata


def write_csv(rows, path):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows, path, target_years):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    years_in_data = sorted({r["target_year"] for r in rows})
    for idx, year in enumerate(years_in_data):
        year_rows = sorted(
            [r for r in rows if r["target_year"] == year],
            key=lambda r: r["k"],
        )
        ks = [r["k"] for r in year_rows]
        maes = [r["mae"] for r in year_rows]
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        ax.plot(ks, maes, marker="o", label=str(year), color=color, linewidth=2)

    ax.set_xlabel("Number of races used for training (k)")
    ax.set_ylabel("MAE on next race (k+1)")
    ax.set_title("Temporal Advantage Curve (Walk-Forward MAE)")
    ax.set_ylim(bottom=0.0)
    ax.legend(title="Season")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward temporal curve: AUROC vs races used (lambda=1)"
    )
    parser.add_argument(
        "--target_years",
        type=str,
        default=",".join(str(y) for y in cfg.TEMPORAL_CURVE_TARGET_YEARS),
        help="Comma-separated target seasons (default: TEST_YEARS)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/temporal_curve/")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--max_k",
        type=int,
        default=None,
        help="Limit to first max_k steps per season (for smoke tests)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=cfg.DEFAULT_GPU_ID,
        help=f"CUDA device index (default: {cfg.DEFAULT_GPU_ID})",
    )
    args = parser.parse_args()

    target_years = parse_target_years(args.target_years)
    run_temporal_curve(
        target_years=target_years,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        deterministic=args.deterministic,
        max_k=args.max_k,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
