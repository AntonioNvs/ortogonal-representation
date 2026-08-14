"""Transfer Tier-Alignment Experiment.

Answers the question: *"does the counterfactual simulation produce plausible
results?"*

The counterfactual engine in ``experiments.transfer_experiment`` already
computes, for each genuine driver->constructor transfer between season T and
T+1, the isolated *car effect*

    Delta = pred_new - pred_old

by swapping only the constructor embedding (driver embedding and the driver's
actual T+1 qualifying/grid are held fixed).  ``Delta`` is in finishing-position
units, so a negative value means the move was predicted to help (better finish).

The plausibility bar used here is *directional vs. the market* -- the paper's
fair-market hypothesis.  We attach to every transfer a leak-free tier direction

    tier_dir = score(new_team @ T) - score(old_team @ T)

where ``score`` maps tiers S/A/B -> 3/2/1 and both teams are evaluated at the
*old* season T using the trailing moving average in
``validation.team_tiers`` (which only sees data <= T), made lineage-aware so a
rebrand (e.g. Alfa Romeo -> Audi) carries its rank across the boundary.

Plausible means the two point the same way:
    promotion  (tier_dir > 0)  ->  Delta < 0   (better finish)
    demotion   (tier_dir < 0)  ->  Delta > 0
    lateral    (tier_dir = 0)  ->  Delta ~ 0

Reported per model: Spearman rho(tier_dir, -Delta) (positive = plausible) and a
sign-agreement table.  The headline claim is that the orthogonal (high-lambda)
model aligns best.

Usage
-----
    # Smoke test
    python -m src.experiments.transfer_tier_alignment \\
        --min-transfer-year 2023 --max-transfer-year 2023 \\
        --model-configs zero,high --device cpu

    # Full run
    python -m src.experiments.transfer_tier_alignment \\
        --min-transfer-year 2022 --max-transfer-year 2026 \\
        --model-configs zero,low,high --device cuda:7
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _p in (ROOT_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg
from experiments.transfer_experiment import (
    _is_rebranding,
    build_eval_dataset_for_transfer,
    build_train_edge_mask,
    detect_driver_transfers,
    inference_transfer,
    load_model,
)
from train import get_device, load_db_and_graph, parse_model_grid, set_global_seed
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import (
    TIER_TO_SCORE,
    compute_constructor_season_points,
    compute_team_tiers,
)


# ---------------------------------------------------------------------------
# Tier direction
# ---------------------------------------------------------------------------

def build_lineage_season_tier(team_tier: pd.DataFrame, lid_map: dict) -> dict:
    """Map ``lineage_id -> {season: (tier, score)}``.

    ``team_tier`` is the output of ``compute_team_tiers`` (per-constructor rows).
    We re-key by lineage so a rebranded team can be looked up at a season where
    only its previous name competed.
    """
    df = team_tier.copy()
    df["lineage_id"] = df["constructorId"].map(lid_map)
    lookup: dict = {}
    for _, row in df.iterrows():
        lid = row["lineage_id"]
        if lid is None or (isinstance(lid, float) and np.isnan(lid)):
            continue
        lookup.setdefault(lid, {})[int(row["season"])] = (row["tier"], row["score"])
    return lookup


def tier_at_season(lookup: dict, lineage_id, season: int):
    """Return ``(tier, score)`` for a lineage at ``season``, falling back to the
    nearest observed season <= ``season`` (so a rebrand / gap still resolves).
    Returns None if the lineage has no observed season on or before ``season``.
    """
    sub = lookup.get(lineage_id)
    if not sub:
        return None
    if season in sub:
        return sub[season]
    past = [s for s in sub if s <= season]
    if not past:
        return None
    return sub[max(past)]


def _tier_score(row) -> float:
    """Scalar tier score (S=3, A=2, B=1), NaN-safe."""
    return float(TIER_TO_SCORE.get(row, np.nan))


# ---------------------------------------------------------------------------
# Aggregation / scoring
# ---------------------------------------------------------------------------

def summarize_transfer_tiers(transfers_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model Spearman rho + sign-agreement table from per-transfer rows.

    ``transfers_df`` has one row per (driver, transfer, model) with columns
    ``tier_dir`` and ``delta_mean`` (see ``run_transfer_tier_alignment``).
    """
    rows = []
    for model_name, g in transfers_df.groupby("model", sort=True):
        g = g.dropna(subset=["tier_dir", "delta_mean"])
        n = int(len(g))
        if n < 3:
            continue

        # Spearman rho(tier_dir, -Delta): positive = plausible.
        rho, p = spearmanr(g["tier_dir"], -g["delta_mean"])

        promo = g[g["tier_dir"] > 0]
        lateral = g[g["tier_dir"] == 0]
        demo = g[g["tier_dir"] < 0]

        def frac(sub, cond):
            return float((cond(sub)).mean()) if len(sub) else float("nan")

        rows.append({
            "model": model_name,
            "model_level": g["model_level"].iloc[0] if "model_level" in g else "",
            "lambda_orthogonal": g["lambda_orthogonal"].iloc[0] if "lambda_orthogonal" in g else np.nan,
            "n_transfers": n,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "n_promotions": int(len(promo)),
            "promotion_mean_delta": float(promo["delta_mean"].mean()) if len(promo) else float("nan"),
            "promotion_agree_frac": frac(promo, lambda s: s["delta_mean"] < 0),
            "n_lateral": int(len(lateral)),
            "lateral_mean_delta": float(lateral["delta_mean"].mean()) if len(lateral) else float("nan"),
            "n_demotions": int(len(demo)),
            "demotion_mean_delta": float(demo["delta_mean"].mean()) if len(demo) else float("nan"),
            "demotion_agree_frac": frac(demo, lambda s: s["delta_mean"] > 0),
            "overall_agree_frac": float(
                ((promo["delta_mean"] < 0).sum()
                 + (demo["delta_mean"] > 0).sum())
                / max(len(promo) + len(demo), 1)
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["lambda_orthogonal"], ascending=True
    ).reset_index(drop=True)


def _write(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_transfer_tier_alignment(
    model_configs: list,
    min_transfer_year: int = 2022,
    max_transfer_year: int = 2026,
    tier_window: int | None = None,
    device: torch.device | None = None,
    seed: int = 42,
    output_dir: str | None = None,
):
    tier_window = tier_window or cfg.TIER_WINDOW
    output_dir = output_dir or cfg.TRANSFER_TIER_ALIGNMENT_OUTPUT_DIR
    if device is None:
        device = get_device()

    set_global_seed(seed)

    # --- Load data + graph once (same id space throughout) ---
    print("=" * 60)
    print("Loading database and building graph...")
    db, graph_data, node_to_col_names_dict, node_to_col_stats, instances_df, _task = (
        load_db_and_graph()
    )

    drivers_df = db.table_dict["drivers"].df
    constructors_df = db.table_dict["constructors"].df
    results_df = db.table_dict["results"].df
    races_df = db.table_dict["races"].df

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

    # --- Deterministic, lineage-aware team tiers ---
    print("Computing lineage-aware team tiers...")
    points_df = compute_constructor_season_points(db)
    lid_map = lineage_id_by_constructor(constructors_df)
    team_tier = compute_team_tiers(
        points_df,
        window=tier_window,
        p_S=cfg.TIER_S_FRAC,
        p_A=cfg.TIER_A_FRAC,
        lineage=lid_map,
    )
    lineage_season_tier = build_lineage_season_tier(team_tier, lid_map)

    # --- Detect transfers ---
    print("Detecting driver transfers...")
    transfers_df = detect_driver_transfers(
        results_df, races_df, min_year=min_transfer_year, max_year=max_transfer_year
    )
    if transfers_df.empty:
        print("No transfers found in the specified year range.")
        return

    transfers_df["driver_ref"] = transfers_df["driverId"].map(driver_names)
    transfers_df["old_constructor_ref"] = transfers_df["old_constructorId"].map(constructor_names)
    transfers_df["new_constructor_ref"] = transfers_df["new_constructorId"].map(constructor_names)

    transfers_df["_is_rebrand"] = transfers_df.apply(
        lambda r: _is_rebranding(r["old_constructor_ref"], r["new_constructor_ref"]),
        axis=1,
    )
    n_rebrand = transfers_df["_is_rebrand"].sum()
    transfers_df = transfers_df[~transfers_df["_is_rebrand"]].copy()
    transfers_df = transfers_df.drop(columns=["_is_rebrand"])
    print(f"Found {len(transfers_df)} genuine transfers ({n_rebrand} rebrandings excluded).")

    # --- Attach tier direction (evaluated at old season T) ---
    def _tier_dir(row):
        old = tier_at_season(lineage_season_tier, lid_map.get(row["old_constructorId"]), int(row["old_season"]))
        new = tier_at_season(lineage_season_tier, lid_map.get(row["new_constructorId"]), int(row["old_season"]))
        if old is None or new is None:
            return pd.Series({"old_tier": None, "new_tier": None, "tier_dir": np.nan})
        return pd.Series({
            "old_tier": old[0],
            "new_tier": new[0],
            "tier_dir": _tier_score(new[0]) - _tier_score(old[0]),
        })

    tier_cols = transfers_df.apply(_tier_dir, axis=1)
    transfers_df = pd.concat([transfers_df, tier_cols], axis=1)

    n_na_tier = transfers_df["tier_dir"].isna().sum()
    if n_na_tier:
        print(f"WARNING: {n_na_tier} transfers have no tier at the old season "
              f"(brand-new lineage with no prior observation) — excluded from scoring.")

    # --- Train-only edge mask ---
    print("Building train-only edge mask...")
    train_edge_index_dict = build_train_edge_mask(db, graph_data)
    graph_data = graph_data.to(device)
    train_edge_index_dict = {et: ei.to(device) for et, ei in train_edge_index_dict.items()}
    graph_tf_dict = graph_data.tf_dict

    # --- Run inference, collect per-race rows + per-transfer aggregates ---
    all_race_rows = []
    all_transfer_rows = []

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

            eval_dataset, eval_df = build_eval_dataset_for_transfer(
                instances_df, int(new_season), driver_ids
            )
            if eval_dataset is None or len(eval_dataset) == 0:
                print(f"    No race data found for {new_season} — skipping.")
                continue

            eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
            driver_to_old_constructor = dict(
                zip(season_transfers["driverId"], season_transfers["old_constructorId"])
            )

            inf_results = inference_transfer(
                model, graph_data, graph_tf_dict, train_edge_index_dict,
                eval_loader, driver_to_old_constructor, device,
            )

            eval_df = eval_df.reset_index(drop=True)
            eval_df["pred_new"] = inf_results["pred_new"]
            eval_df["pred_old"] = inf_results["pred_old"]
            eval_df["delta"] = eval_df["pred_new"] - eval_df["pred_old"]

            meta = season_transfers[
                ["driverId", "driver_ref", "old_constructorId", "new_constructorId",
                 "old_constructor_ref", "new_constructor_ref",
                 "old_season", "new_season", "old_tier", "new_tier", "tier_dir"]
            ]
            eval_df = eval_df.merge(meta, on="driverId", how="left")

            # Per-race rows
            for _, erow in eval_df.iterrows():
                all_race_rows.append({
                    "model": model_cfg["name"],
                    "model_level": model_level,
                    "lambda_orthogonal": lambda_val,
                    "driver_ref": str(erow.get("driver_ref", "")),
                    "driverId": int(erow["driverId"]),
                    "old_constructor_ref": str(erow.get("old_constructor_ref", "")),
                    "new_constructor_ref": str(erow.get("new_constructor_ref", "")),
                    "old_season": int(erow.get("old_season", -1)),
                    "new_season": int(new_season),
                    "round": int(erow.get("round", -1)),
                    "raceId": int(erow.get("raceId", -1)),
                    "actual": float(erow["y"]),
                    "pred_new": float(erow["pred_new"]),
                    "pred_old": float(erow["pred_old"]),
                    "delta": float(erow["delta"]),
                    "old_tier": erow.get("old_tier"),
                    "new_tier": erow.get("new_tier"),
                    "tier_dir": erow.get("tier_dir"),
                })

            # Per-transfer aggregates (collapse correlated per-race deltas)
            for (driver_id, old_ref, new_ref), g in eval_df.groupby(
                ["driverId", "old_constructor_ref", "new_constructor_ref"], sort=False
            ):
                all_transfer_rows.append({
                    "model": model_cfg["name"],
                    "model_level": model_level,
                    "lambda_orthogonal": lambda_val,
                    "driver_ref": str(g["driver_ref"].iloc[0]),
                    "driverId": int(driver_id),
                    "old_constructor_ref": str(old_ref),
                    "new_constructor_ref": str(new_ref),
                    "old_season": int(g["old_season"].iloc[0]),
                    "new_season": int(g["new_season"].iloc[0]),
                    "old_tier": g["old_tier"].iloc[0],
                    "new_tier": g["new_tier"].iloc[0],
                    "tier_dir": g["tier_dir"].iloc[0],
                    "delta_mean": float(g["delta"].mean()),
                    "pred_new_mean": float(g["pred_new"].mean()),
                    "pred_old_mean": float(g["pred_old"].mean()),
                    "actual_mean": float(g["actual"].mean()),
                    "n_races": int(len(g)),
                })

    # --- Persist + score ---
    races_df_out = pd.DataFrame(all_race_rows)
    transfers_df_out = pd.DataFrame(all_transfer_rows)

    if transfers_df_out.empty:
        print("No results produced.")
        return

    summary_df = summarize_transfer_tiers(transfers_df_out)

    _write(races_df_out, os.path.join(output_dir, "tier_alignment_races.csv"))
    _write(transfers_df_out, os.path.join(output_dir, "tier_alignment_transfers.csv"))
    _write(summary_df, os.path.join(output_dir, "tier_alignment_summary.csv"))

    print(f"\n{'=' * 60}")
    print("TRANSFER TIER-ALIGNMENT RESULTS")
    print(f"{'=' * 60}")
    print(f"Per-race rows:      {os.path.join(output_dir, 'tier_alignment_races.csv')}")
    print(f"Per-transfer rows:  {os.path.join(output_dir, 'tier_alignment_transfers.csv')}")
    print(f"Summary:            {os.path.join(output_dir, 'tier_alignment_summary.csv')}")
    print(f"\n{summary_df.to_string(index=False)}")

    print(f"\n{'=' * 60}")
    print("INTERPRETATION (positive rho = plausible)")
    print(f"{'=' * 60}")
    for _, r in summary_df.iterrows():
        print(f"\n[{r['model']}] lambda={r['lambda_orthogonal']}: "
              f"rho(tier_dir, -Delta) = {r['spearman_rho']:+.3f} (p={r['spearman_p']:.3g}, n={r['n_transfers']})")
        print(f"  promotions: mean Delta {r['promotion_mean_delta']:+.3f} "
              f"({r['promotion_agree_frac']:.0%} have Delta<0)")
        print(f"  laterals:   mean Delta {r['lateral_mean_delta']:+.3f} (n={r['n_lateral']})")
        print(f"  demotions:  mean Delta {r['demotion_mean_delta']:+.3f} "
              f"({r['demotion_agree_frac']:.0%} have Delta>0)")

    return transfers_df_out, races_df_out, summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transfer tier-alignment: is the counterfactual car-effect plausible?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--min-transfer-year", type=int, default=2022,
                        help="Earliest transfer season to include (default: 2022).")
    parser.add_argument("--max-transfer-year", type=int, default=2026,
                        help="Latest transfer season to include (default: 2026).")
    parser.add_argument("--model-configs", type=str, default="zero,low,high",
                        help="Comma-separated model configs: zero,low,high.")
    parser.add_argument("--tier-window", type=int, default=None,
                        help="Tier moving-average window (default: cfg.TIER_WINDOW).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda:7', 'cpu').")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: cfg.TRANSFER_TIER_ALIGNMENT_OUTPUT_DIR).")

    args = parser.parse_args()

    if args.device is not None:
        device = torch.device("cpu") if args.device == "cpu" else torch.device(args.device)
    else:
        device = get_device()

    model_configs = parse_model_grid(args.model_configs)

    print(f"Transfer tier-alignment: {args.min_transfer_year}-{args.max_transfer_year}")
    print(f"Model configs: {[m['name'] for m in model_configs]}")
    print(f"Device: {device}")

    run_transfer_tier_alignment(
        model_configs=model_configs,
        min_transfer_year=args.min_transfer_year,
        max_transfer_year=args.max_transfer_year,
        tier_window=args.tier_window,
        device=device,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
