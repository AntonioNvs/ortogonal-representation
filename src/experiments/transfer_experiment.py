"""
Held-Out Driver Transfer Experiment.

Tests whether the orthogonal decomposition generalises to genuine
counterfactuals: when a driver switches teams between season T and T+1,
can the model predict their T+1 performance using the old driver embedding
combined with the new constructor's embedding?

Uses existing trained models (no re-training).  The (driver, new_constructor)
pair never appeared together in the 2000-2021 training window, so every
prediction is a genuine counterfactual.

Usage
-----
    # Smoke test (1 year, 2 models)
    python -m src.experiments.transfer_experiment \\
        --min-transfer-year 2023 --max-transfer-year 2023 \\
        --model-configs zero,high --device cuda:7

    # Full run
    python -m src.experiments.transfer_experiment \\
        --min-transfer-year 2022 --max-transfer-year 2026 \\
        --model-configs zero,low,high --device cuda:7
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from train import (
    DEFAULT_MODEL_CONFIGS,
    F1AlignedDataset,
    _build_instances_from_task,
    add_edge_year_masks,
    get_active_task,
    get_device,
    load_db_and_graph,
    parse_model_grid,
    set_global_seed,
)
from models.pipeline_fusion import F1OrthogonalPipeline

# ---------------------------------------------------------------------------
# Rebranding patterns — constructorRef changes that are the same organisation
# ---------------------------------------------------------------------------
REBRANDING_PAIRS = {
    # (old_ref, new_ref): both are the same team
    ("alphatauri", "rb"),
    ("alfa", "sauber"),
    ("sauber", "audi"),
}


def _is_rebranding(old_ref: str, new_ref: str) -> bool:
    """Check whether a constructorRef change is a rebranding, not a transfer."""
    old_l = old_ref.lower().strip()
    new_l = new_ref.lower().strip()
    return (old_l, new_l) in REBRANDING_PAIRS


# ---------------------------------------------------------------------------
# 1. Transfer detection
# ---------------------------------------------------------------------------

def detect_driver_transfers(
    results_df: pd.DataFrame,
    races_df: pd.DataFrame,
    min_year: int = 2000,
    max_year: int = 2026,
) -> pd.DataFrame:
    """Identify drivers who switched constructors between consecutive seasons.

    Returns a DataFrame with columns: driverId, old_constructorId,
    new_constructorId, old_season, new_season, old_constructor_ref,
    new_constructor_ref, driver_ref.
    """
    merged = results_df[["resultId", "raceId", "driverId", "constructorId"]].merge(
        races_df[["raceId", "year"]], on="raceId", how="inner"
    )

    # Mode constructor per (driver, year)
    season_constructor = (
        merged.groupby(["driverId", "year"])["constructorId"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )

    transfers = []
    for driver_id, group in season_constructor.groupby("driverId"):
        group = group.sort_values("year")
        for i in range(len(group) - 1):
            row_t = group.iloc[i]
            row_t1 = group.iloc[i + 1]
            if (
                row_t["year"] + 1 == row_t1["year"]
                and row_t["constructorId"] != row_t1["constructorId"]
                and min_year <= row_t1["year"] <= max_year
            ):
                transfers.append({
                    "driverId": int(driver_id),
                    "old_constructorId": int(row_t["constructorId"]),
                    "new_constructorId": int(row_t1["constructorId"]),
                    "old_season": int(row_t["year"]),
                    "new_season": int(row_t1["year"]),
                })

    transfers_df = pd.DataFrame(transfers)
    if transfers_df.empty:
        return transfers_df

    # Annotate with names
    transfers_df = transfers_df.sort_values(["new_season", "driverId"]).reset_index(drop=True)
    return transfers_df


# ---------------------------------------------------------------------------
# 2. Naive baselines
# ---------------------------------------------------------------------------

def compute_naive_baselines(
    instances_df: pd.DataFrame,
    outcome_lookup: pd.DataFrame,
    transfers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute driver and constructor average positionOrder in the prior season.

    Uses ``instances_df`` (which has ``driverId``, ``constructorId``, ``year``,
    ``resultId``) merged with ``outcome_lookup`` (which has ``positionOrder``
    per ``resultId``, captured before the task removed outcome columns).

    Returns transfers_df augmented with baseline_driver_mean and
    baseline_constructor_mean columns.
    """
    merged = instances_df[["driverId", "constructorId", "year", "resultId"]].merge(
        outcome_lookup[["resultId", "positionOrder"]], on="resultId", how="inner"
    )

    driver_means = (
        merged.groupby(["driverId", "year"])["positionOrder"]
        .mean()
        .reset_index()
        .rename(columns={"positionOrder": "baseline_driver_mean"})
    )

    constructor_means = (
        merged.groupby(["constructorId", "year"])["positionOrder"]
        .mean()
        .reset_index()
        .rename(columns={"positionOrder": "baseline_constructor_mean"})
    )

    result = transfers_df.merge(
        driver_means,
        left_on=["driverId", "old_season"],
        right_on=["driverId", "year"],
        how="left",
    ).drop(columns=["year"])

    result = result.merge(
        constructor_means,
        left_on=["new_constructorId", "old_season"],
        right_on=["constructorId", "year"],
        how="left",
    ).drop(columns=["constructorId", "year"])

    return result


# ---------------------------------------------------------------------------
# 3. Edge mask restricted to training years only
# ---------------------------------------------------------------------------

def build_train_edge_mask(db, graph_data):
    """Return an edge_index_dict with edges restricted to cfg.TRAIN_YEARS."""
    masks = add_edge_year_masks(db, graph_data)
    train_edge_index_dict = {
        et: graph_data[et].edge_index[:, masks[et]["train"]].contiguous()
        for et in graph_data.edge_types
    }
    return train_edge_index_dict


# ---------------------------------------------------------------------------
# 4. Evaluation dataset builder
# ---------------------------------------------------------------------------

def build_eval_dataset_for_transfer(
    instances_df: pd.DataFrame,
    transfer_year: int,
    driver_ids: list,
) -> tuple:
    """Build an F1AlignedDataset for a specific transfer year and driver set.

    Returns (eval_dataset, eval_df).
    """
    eval_df = instances_df[
        (instances_df["year"] == transfer_year)
        & (instances_df["driverId"].isin(driver_ids))
    ].copy()

    if eval_df.empty:
        return None, eval_df

    eval_dataset = F1AlignedDataset(eval_df)
    return eval_dataset, eval_df


# ---------------------------------------------------------------------------
# 5. Counterfactual inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def inference_transfer(
    model: F1OrthogonalPipeline,
    graph_data,
    graph_tf_dict,
    edge_index_dict: dict,
    eval_loader: DataLoader,
    driver_to_old_constructor: dict,
    device: torch.device,
) -> dict:
    """Run inference with both actual (new) and swapped (old) constructor IDs.

    Returns a dict of per-sample arrays: driver_ids, constructor_ids, targets,
    pred_new, pred_old, qualifying_pos, grid_pos.
    """
    model.eval()
    results = {
        "driver_ids": [],
        "constructor_ids": [],
        "targets": [],
        "pred_new": [],
        "pred_old": [],
        "qualifying_pos": [],
        "grid_pos": [],
    }

    for batch in eval_loader:
        driver_ids, constructor_ids, qualifying_pos, grid_pos, targets, _ = [
            b.to(device) for b in batch
        ]

        # Prediction (a): actual new constructor
        logits_new, _, _, _, _, _ = model(
            graph_x_dict=None,
            graph_edge_index_dict=edge_index_dict,
            target_constructor_ids=constructor_ids,
            target_driver_ids=driver_ids,
            qualifying_position=qualifying_pos,
            grid=grid_pos,
            graph_tf_dict=graph_tf_dict,
        )

        # Prediction (b): swap to old constructor
        old_c_ids = torch.tensor(
            [driver_to_old_constructor[d_id.item()] for d_id in driver_ids],
            dtype=torch.long,
            device=device,
        )
        logits_old, _, _, _, _, _ = model(
            graph_x_dict=None,
            graph_edge_index_dict=edge_index_dict,
            target_constructor_ids=old_c_ids,
            target_driver_ids=driver_ids,
            qualifying_position=qualifying_pos,
            grid=grid_pos,
            graph_tf_dict=graph_tf_dict,
        )

        results["driver_ids"].extend(driver_ids.cpu().tolist())
        results["constructor_ids"].extend(constructor_ids.cpu().tolist())
        results["targets"].extend(targets.cpu().tolist())
        results["pred_new"].extend(logits_new.squeeze(-1).cpu().tolist())
        results["pred_old"].extend(logits_old.squeeze(-1).cpu().tolist())
        results["qualifying_pos"].extend(qualifying_pos.cpu().tolist())
        results["grid_pos"].extend(grid_pos.cpu().tolist())

    return results


# ---------------------------------------------------------------------------
# 6. Model loading
# ---------------------------------------------------------------------------

def _patch_state_dict(model: F1OrthogonalPipeline, checkpoint_path: str, device: torch.device):
    """Load a checkpoint with size-mismatch patching (follows analyze_impact_2023.py)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = model.state_dict()
    patched = {}

    for k, v in checkpoint.items():
        if k in model_state:
            try:
                target_shape = model_state[k].shape
            except RuntimeError:
                continue
            if v.shape != target_shape:
                new_v = model_state[k].clone()
                slices = tuple(
                    slice(0, min(dim_v, dim_t))
                    for dim_v, dim_t in zip(v.shape, target_shape)
                )
                new_v[slices] = v[slices]
                patched[k] = new_v
            else:
                patched[k] = v

    model.load_state_dict(patched, strict=False)


def load_model(
    model_config: dict,
    graph_data,
    node_to_col_names_dict: dict,
    node_to_col_stats: dict,
    device: torch.device,
) -> F1OrthogonalPipeline:
    """Instantiate and load a trained model from disk."""
    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}
    model = F1OrthogonalPipeline(
        num_nodes_dict=num_nodes_dict,
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
        latent_dim=32,
    )

    model_path = f"output/models/{model_config['name']}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Initialise lazy parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.encoder(graph_data.tf_dict)
        except Exception:
            pass

    _patch_state_dict(model, model_path, device)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 7. Aggregation
# ---------------------------------------------------------------------------

def aggregate_transfer_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-method MAE from per-race predictions."""
    methods = []
    for col in results_df.columns:
        if col.startswith("pred_") or col.startswith("baseline_"):
            ae = (results_df[col] - results_df["actual"]).abs()
            methods.append({
                "method": col,
                "mae": ae.mean(),
                "std": ae.std(),
                "n": ae.count(),
            })

    return pd.DataFrame(methods).sort_values("mae")


# ---------------------------------------------------------------------------
# 8. Main orchestration
# ---------------------------------------------------------------------------

def run_transfer_experiment(
    model_configs: list,
    min_transfer_year: int = 2022,
    max_transfer_year: int = 2026,
    device: torch.device = None,
    seed: int = 42,
):
    """Run the full transfer experiment."""
    if device is None:
        device = get_device()

    set_global_seed(seed)

    # --- Load data once ---
    print("=" * 60)
    print("Loading database and building graph...")
    db, graph_data, node_to_col_names_dict, node_to_col_stats, instances_df, task = (
        load_db_and_graph()
    )

    # outcome_lookup (captured before task removes outcome columns) gives us
    # positionOrder for baseline computation
    _, outcome_lookup = get_active_task()

    results_df = db.table_dict["results"].df
    races_df = db.table_dict["races"].df
    drivers_df = db.table_dict["drivers"].df
    constructors_df = db.table_dict["constructors"].df

    # Name lookups
    driver_names = (
        drivers_df["driverRef"].to_dict()
        if "driverRef" in drivers_df.columns
        else {i: f"Driver_{i}" for i in drivers_df.index}
    )
    constructor_names = (
        constructors_df["constructorRef"].to_dict()
        if "constructorRef" in constructors_df.columns
        else {i: f"Constructor_{i}" for i in constructors_df.index}
    )

    # --- Detect transfers ---
    print("Detecting driver transfers...")
    transfers_df = detect_driver_transfers(
        results_df, races_df, min_year=min_transfer_year, max_year=max_transfer_year
    )

    if transfers_df.empty:
        print("No transfers found in the specified year range.")
        return

    # Annotate with names
    transfers_df["driver_ref"] = transfers_df["driverId"].map(driver_names)
    transfers_df["old_constructor_ref"] = transfers_df["old_constructorId"].map(constructor_names)
    transfers_df["new_constructor_ref"] = transfers_df["new_constructorId"].map(constructor_names)

    # Filter out rebrandings
    transfers_df["_is_rebrand"] = transfers_df.apply(
        lambda r: _is_rebranding(r["old_constructor_ref"], r["new_constructor_ref"]),
        axis=1,
    )
    n_rebrand = transfers_df["_is_rebrand"].sum()
    transfers_df = transfers_df[~transfers_df["_is_rebrand"]].copy()
    transfers_df = transfers_df.drop(columns=["_is_rebrand"])

    print(f"Found {len(transfers_df)} genuine transfers ({n_rebrand} rebrandings excluded).")

    # --- Compute naive baselines ---
    transfers_df = compute_naive_baselines(instances_df, outcome_lookup, transfers_df)

    # --- Build edge mask (2000-2021 only) ---
    print("Building train-only edge mask...")
    train_edge_index_dict = build_train_edge_mask(db, graph_data)

    # Move graph to device
    graph_data = graph_data.to(device)
    train_edge_index_dict = {et: ei.to(device) for et, ei in train_edge_index_dict.items()}
    graph_tf_dict = graph_data.tf_dict

    # --- Collect all per-race rows ---
    all_rows = []

    for model_cfg in model_configs:
        model_level = model_cfg["model_level"]
        lambda_val = model_cfg["lambda_orthogonal"]
        print(f"\n{'=' * 60}")
        print(f"Model: {model_cfg['name']} (lambda={lambda_val})")

        model = load_model(
            model_cfg, graph_data, node_to_col_names_dict, node_to_col_stats, device
        )

        for new_season, season_transfers in transfers_df.groupby("new_season"):
            season_transfers = season_transfers.reset_index(drop=True)
            driver_ids = season_transfers["driverId"].unique().tolist()
            old_season = int(season_transfers["old_season"].iloc[0])

            print(f"  Season {old_season} -> {new_season}: "
                  f"{len(season_transfers)} transfers, {len(driver_ids)} drivers")

            # Build eval dataset for this season's transferred drivers
            eval_dataset, eval_df = build_eval_dataset_for_transfer(
                instances_df, int(new_season), driver_ids
            )

            if eval_dataset is None or len(eval_dataset) == 0:
                print(f"    No race data found for {new_season} — skipping.")
                continue

            eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

            # Build dict: driver -> old constructor
            driver_to_old_constructor = dict(
                zip(season_transfers["driverId"], season_transfers["old_constructorId"])
            )

            # Leakage check
            for _, row in season_transfers.iterrows():
                train_pairs = instances_df[
                    (instances_df["driverId"] == row["driverId"])
                    & (instances_df["constructorId"] == row["new_constructorId"])
                    & (instances_df["year"].isin(cfg.TRAIN_YEARS))
                ]
                if len(train_pairs) > 0:
                    print(f"    WARNING: (driver={row['driver_ref']}, "
                          f"constructor={row['new_constructor_ref']}) appears in "
                          f"training data — this is NOT a genuine counterfactual!")

            # Run inference
            inf_results = inference_transfer(
                model, graph_data, graph_tf_dict, train_edge_index_dict,
                eval_loader, driver_to_old_constructor, device,
            )

            # Merge with eval_df metadata and baselines
            eval_df = eval_df.reset_index(drop=True)
            eval_df["pred_new"] = inf_results["pred_new"]
            eval_df["pred_old"] = inf_results["pred_old"]

            # Merge baselines
            eval_df = eval_df.merge(
                season_transfers[
                    ["driverId", "old_constructorId", "new_constructorId",
                     "old_constructor_ref", "new_constructor_ref",
                     "driver_ref", "old_season", "new_season",
                     "baseline_driver_mean", "baseline_constructor_mean"]
                ],
                on="driverId", how="left",
            )

            # Build output rows
            for _, erow in eval_df.iterrows():
                all_rows.append({
                    "model": model_cfg["name"],
                    "lambda_orthogonal": lambda_val,
                    "driver_name": str(erow.get("driver_ref", erow["driverId"])),
                    "old_constructor": str(erow.get("old_constructor_ref", "")),
                    "new_constructor": str(erow.get("new_constructor_ref", "")),
                    "old_season": int(erow.get("old_season", -1)),
                    "new_season": int(new_season),
                    "round": int(erow.get("round", -1)),
                    "raceId": int(erow.get("raceId", -1)),
                    "actual": float(erow["y"]),
                    "pred_new": float(erow["pred_new"]),
                    "pred_old": float(erow["pred_old"]),
                    "baseline_driver": float(erow.get("baseline_driver_mean", np.nan)),
                    "baseline_constructor": float(erow.get("baseline_constructor_mean", np.nan)),
                })

    # --- Build results DataFrame ---
    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        print("No results produced.")
        return

    os.makedirs("output/transfer_experiment", exist_ok=True)

    # --- Per-model aggregation ---
    summary_rows = []
    for model_name, group in results_df.groupby("model"):
        for pred_col in ["pred_new", "pred_old", "baseline_driver", "baseline_constructor"]:
            if pred_col in group.columns:
                ae = (group[pred_col] - group["actual"]).abs()
                summary_rows.append({
                    "model": model_name,
                    "method": pred_col,
                    "mae": ae.mean(),
                    "std": ae.std(),
                    "n": ae.count(),
                })

    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "mae"])

    # --- Save ---
    results_path = "output/transfer_experiment/transfer_results.csv"
    summary_path = "output/transfer_experiment/transfer_summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print("TRANSFER EXPERIMENT RESULTS")
    print(f"{'=' * 60}")
    print(f"\nPer-race results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"\n{summary_df.to_string(index=False)}")

    # --- Interpretation ---
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")

    for model_name in summary_df["model"].unique():
        model_summary = summary_df[summary_df["model"] == model_name]
        mae_new = model_summary[model_summary["method"] == "pred_new"]["mae"].values
        mae_driver = model_summary[model_summary["method"] == "baseline_driver"]["mae"].values
        mae_constructor = model_summary[model_summary["method"] == "baseline_constructor"]["mae"].values
        mae_old = model_summary[model_summary["method"] == "pred_old"]["mae"].values

        print(f"\n[{model_name}]:")
        if len(mae_new) and len(mae_driver):
            beats_driver = mae_new[0] < mae_driver[0]
            print(f"  pred_new beats baseline_driver: {beats_driver} "
                  f"({mae_new[0]:.3f} vs {mae_driver[0]:.3f})")
        if len(mae_new) and len(mae_constructor):
            beats_constructor = mae_new[0] < mae_constructor[0]
            print(f"  pred_new beats baseline_constructor: {beats_constructor} "
                  f"({mae_new[0]:.3f} vs {mae_constructor[0]:.3f})")
        if len(mae_new) and len(mae_old):
            beats_old = mae_new[0] < mae_old[0]
            print(f"  pred_new beats pred_old: {beats_old} "
                  f"({mae_new[0]:.3f} vs {mae_old[0]:.3f})")

    return results_df, summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Held-Out Driver Transfer Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--min-transfer-year", type=int, default=2022,
        help="Earliest transfer season to include (default: 2022).",
    )
    parser.add_argument(
        "--max-transfer-year", type=int, default=2026,
        help="Latest transfer season to include (default: 2026).",
    )
    parser.add_argument(
        "--model-configs", type=str, default="zero,high",
        help="Comma-separated model configs: zero,low,high (default: zero,high).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g., 'cuda:7', 'cpu'). Defaults to cfg.DEFAULT_GPU_ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    if args.device is not None:
        if args.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        device = get_device()

    model_configs = parse_model_grid(args.model_configs)

    print(f"Transfer experiment: {args.min_transfer_year}-{args.max_transfer_year}")
    print(f"Model configs: {[m['name'] for m in model_configs]}")
    print(f"Device: {device}")

    run_transfer_experiment(
        model_configs=model_configs,
        min_transfer_year=args.min_transfer_year,
        max_transfer_year=args.max_transfer_year,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()