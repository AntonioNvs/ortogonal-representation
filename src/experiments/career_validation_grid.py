"""Sweep ``horizon x window`` for the career-validation framework.

Rerun the pipeline over a grid of ``(TIER_WINDOW, TIER_HORIZON)`` values so
the paper can report the region in which rho is stable — instead of a single
number pinned to the arbitrary defaults ``(window=3, horizon=3)``.

For each cell we cache the tiers computation (it only depends on
``window``) and rebuild the forward labels for each ``horizon``. The skill
scorer is loaded once and reused — it is the expensive step.

Usage:
    python -m src.experiments.career_validation_grid \\
        --skill-source kalman \\
        --horizons 1 2 3 4 5 \\
        --windows 2 3 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _p in (ROOT_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg
from experiments.career_validation import load_raw_db, load_skill
from validation.career_labels import driver_season_constructor, forward_tier_outcome
from validation.inference import cluster_bootstrap_spearman, permutation_within_season
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers


def run_grid(
    skill_source: str = "kalman",
    horizons: list[int] | None = None,
    windows: list[int] | None = None,
    min_year: int | None = None,
    lineage: bool = False,
    require_full_horizon: bool = True,
    output_dir: str | None = None,
    n_bootstrap: int = 2000,
    n_perm: int = 5000,
    device=None,
):
    horizons = horizons or [1, 2, 3, 4, 5]
    windows = windows or [2, 3, 5]
    min_year = min_year or cfg.CAREER_VALIDATION_MIN_YEAR
    output_dir = output_dir or os.path.join(cfg.CAREER_VALIDATION_OUTPUT_DIR, "grid")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading raw DB and precomputing tiers per window...")
    db = load_raw_db()
    lid_map = (
        lineage_id_by_constructor(db.table_dict["constructors"].df) if lineage else None
    )
    driver_season = driver_season_constructor(db)
    points_df = compute_constructor_season_points(db)
    points_df = points_df[points_df["season"] >= min_year]

    tiers_cache: dict[int, pd.DataFrame] = {}
    for w in windows:
        tiers_cache[w] = compute_team_tiers(
            points_df, window=w,
            p_S=cfg.TIER_S_FRAC, p_A=cfg.TIER_A_FRAC,
            lineage=lid_map,
        )

    print(f"Loading skill scores (source={skill_source})...")
    ref_tiers = tiers_cache[windows[0]]  # constructor_tier baseline needs one.
    skill = load_skill(skill_source, device, db=db, team_tier=ref_tiers)

    rows = []
    for w in windows:
        team_tier = tiers_cache[w]

        for h in horizons:
            labels = forward_tier_outcome(
                driver_season, team_tier,
                horizon=h,
                require_full_horizon=require_full_horizon,
            )
            merged = skill.merge(
                labels,
                left_on=["driverId", "season"],
                right_on=["driverId", "season_T"],
                how="inner",
            ).dropna(subset=["skill_score", "outcome_score"])

            if merged.empty or merged["skill_score"].nunique() < 2:
                rows.append({
                    "window": w, "horizon": h,
                    "n_rows": int(len(merged)),
                    "n_drivers": int(merged["driverId"].nunique()) if len(merged) else 0,
                    "rho": float("nan"),
                    "cluster_ci_low": float("nan"),
                    "cluster_ci_high": float("nan"),
                    "perm_p_value": float("nan"),
                })
                continue

            rho, _ = spearmanr(merged["skill_score"], merged["outcome_score"])
            cb = cluster_bootstrap_spearman(
                merged, cluster_col="driverId", n_bootstrap=n_bootstrap,
            )
            perm = permutation_within_season(
                merged, season_col="season_T", n_perm=n_perm,
            )
            row = {
                "window": w, "horizon": h,
                "n_rows": int(len(merged)),
                "n_drivers": int(merged["driverId"].nunique()),
                "rho": float(rho),
                "cluster_ci_low": cb["ci_low"],
                "cluster_ci_high": cb["ci_high"],
                "perm_p_value": perm["p_value"],
            }
            rows.append(row)
            print(
                f"  window={w} horizon={h}  n={row['n_rows']:4d}  drivers={row['n_drivers']:3d}"
                f"  rho={row['rho']:+.3f}"
                f"  CI[{row['cluster_ci_low']:+.3f},{row['cluster_ci_high']:+.3f}]"
                f"  p={row['perm_p_value']:.3g}"
            )

    grid_df = pd.DataFrame(rows).sort_values(["window", "horizon"]).reset_index(drop=True)
    grid_df.to_csv(os.path.join(output_dir, "grid.csv"), index=False)

    # Try to render a heatmap; skip on failure (matplotlib is a soft dep here).
    try:
        _render_heatmap(grid_df, output_dir)
    except Exception as e:
        print(f"  (skipping heatmap: {e})")

    with open(os.path.join(output_dir, "grid_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "skill_source": skill_source,
            "horizons": horizons,
            "windows": windows,
            "min_year": min_year,
            "lineage": lineage,
            "require_full_horizon": require_full_horizon,
            "cells": rows,
        }, f, indent=2)

    print(f"\nGrid written to: {output_dir}")
    return grid_df


def _render_heatmap(grid_df: pd.DataFrame, output_dir: str):
    import matplotlib.pyplot as plt

    piv_rho = grid_df.pivot(index="window", columns="horizon", values="rho")
    piv_p = grid_df.pivot(index="window", columns="horizon", values="perm_p_value")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(piv_rho.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Spearman rho")
    axes[0].set_xticks(range(len(piv_rho.columns)), piv_rho.columns)
    axes[0].set_yticks(range(len(piv_rho.index)), piv_rho.index)
    axes[0].set_xlabel("horizon"); axes[0].set_ylabel("window")
    for i in range(piv_rho.shape[0]):
        for j in range(piv_rho.shape[1]):
            v = piv_rho.values[i, j]
            if not np.isnan(v):
                axes[0].text(j, i, f"{v:+.2f}", ha="center", va="center",
                             color="white" if abs(v) > 0.5 else "black", fontsize=9)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(piv_p.values, aspect="auto", cmap="viridis_r", vmin=0, vmax=0.1)
    axes[1].set_title("Within-season permutation p")
    axes[1].set_xticks(range(len(piv_p.columns)), piv_p.columns)
    axes[1].set_yticks(range(len(piv_p.index)), piv_p.index)
    axes[1].set_xlabel("horizon"); axes[1].set_ylabel("window")
    for i in range(piv_p.shape[0]):
        for j in range(piv_p.shape[1]):
            v = piv_p.values[i, j]
            if not np.isnan(v):
                axes[1].text(j, i, f"{v:.3f}", ha="center", va="center",
                             color="white" if v < 0.05 else "black", fontsize=9)
    fig.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "grid_heatmap.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Grid sweep of (window, horizon) for career validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skill-source", type=str, default="kalman")
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=None)
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--lineage", action="store_true")
    parser.add_argument("--no-full-horizon", action="store_true",
                        help="Disable require_full_horizon (default: enabled for the grid).")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_grid(
        skill_source=args.skill_source,
        horizons=args.horizons,
        windows=args.windows,
        min_year=args.min_year,
        lineage=args.lineage,
        require_full_horizon=not args.no_full_horizon,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
        device=args.device,
    )


if __name__ == "__main__":
    main()
