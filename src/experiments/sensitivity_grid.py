#!/usr/bin/env python3
"""Sensitivity grid: sweep cohort threshold and tier cut.

Recomputes headline career metrics over

    skill_pct_threshold in {0.70, 0.75, 0.80}
    p_S                in {0.25, 0.30, 0.35}

for each skill source and reports resolution rate, underrated AUROC, and
within-stratum partial rho per cell, plus a summary of whether the primary
source's ordering vs the baseline is stable across the grid.

Run:
    python src/experiments/sensitivity_grid.py \
        --sources bradley_terry orthogonal_shapley --baseline bradley_terry
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.skill_loader import load_skill_export, season_scores_for_career
from baselines.skill_gnn_skill import get_skill_gnn_db
from experiments.evaluate_skill_model import join_career
from validation.career_metrics import parse_horizon_arg
from validation.inconsistency import mark_underrated, underrated_promotion_auroc, underrated_resolution_rate
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers

DEFAULT_SOURCES = ["bradley_terry", "orthogonal_shapley"]
SKILL_PCTS = [0.70, 0.75, 0.80]
P_S_GRID = [0.25, 0.30, 0.35]


def _stratum_partial(marked: "pd.DataFrame") -> float:
    from validation.inference import stratum_partial_spearman

    return stratum_partial_spearman(marked, seed=0).get("partial_rho", float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity grid for career validation")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--baseline", type=str, default="bradley_terry")
    parser.add_argument("--output-dir", type=str, default="output/sensitivity_grid")
    parser.add_argument("--horizon", type=str, default="inf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument(
        "--fixed-cohort",
        action="store_true",
        help="Define underrated cohort on model-free teammate_residual so the flag is shared across sources",
    )
    args = parser.parse_args()
    horizon = parse_horizon_arg(args.horizon)

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()
    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)

    # Precompute skill exports once (independent of p_S / threshold).
    skills = {}
    for source in args.sources:
        export = load_skill_export(source, db, max_year=args.max_year)
        skills[source] = season_scores_for_career(export)

    # Model-free cohort (Section 1 fix): flag on teammate_residual so the
    # underrated set is identical across sources.
    cohort_skill = None
    cohort_skill_col = None
    if args.fixed_cohort:
        cohort_skill = season_scores_for_career(
            load_skill_export("teammate_residual", db, max_year=args.max_year)
        )
        cohort_skill_col = "cohort_skill_score"

    cells = []
    for p_S in P_S_GRID:
        team_tier = compute_team_tiers(
            compute_constructor_season_points(db), lineage=lineage, p_S=p_S, p_A=0.35
        )
        joined_by_source = {
            source: join_career(skills[source], db, team_tier, horizon=horizon, cohort_skill=cohort_skill)
            for source in args.sources
        }
        for threshold in SKILL_PCTS:
            row = {"p_S": p_S, "skill_pct_threshold": threshold}
            for source in args.sources:
                marked = mark_underrated(
                    joined_by_source[source], skill_pct_threshold=threshold,
                    cohort_skill_col=cohort_skill_col,
                )
                res = underrated_resolution_rate(marked, seed=args.seed)
                auroc = underrated_promotion_auroc(marked, seed=args.seed)
                row[f"{source}_n"] = int(marked["underrated_flag"].sum())
                row[f"{source}_resolution"] = res["resolution_rate"]
                row[f"{source}_auroc"] = auroc["auroc"]
                row[f"{source}_partial_rho"] = _stratum_partial(marked)
            cells.append(row)

    # Stability summary: how often primary (non-baseline) sources beat baseline.
    import pandas as pd

    grid = pd.DataFrame(cells)
    # On a fixed cohort the flag/promoted labels are shared, so resolution rate
    # is identical across sources and is NOT a discriminator (Section 1). Only
    # AUROC and within-stratum Spearman differ; report those as the summary.
    discriminator_metrics = ("auroc", "partial_rho") if args.fixed_cohort else (
        "resolution", "auroc", "partial_rho",
    )
    summary = {}
    for source in args.sources:
        if source == args.baseline:
            continue
        base = args.baseline
        for metric in discriminator_metrics:
            wins = (
                (grid[f"{source}_{metric}"] >= grid[f"{base}_{metric}"])
                & ~grid[f"{source}_{metric}"].isna()
            ).sum()
            n_cells = grid[f"{source}_{metric}"].notna().sum()
            summary.setdefault(source, {})[f"{metric}_beats_baseline_frac"] = (
                float(wins / n_cells) if n_cells else float("nan")
            )
            summary[source][f"{metric}_n_cells"] = int(n_cells)

    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "baseline": args.baseline,
        "fixed_cohort": bool(args.fixed_cohort),
        "cohort_skill_col": cohort_skill_col,
        "grid": grid.to_dict(orient="records"),
        "stability_summary": summary,
    }
    with open(os.path.join(args.output_dir, "sensitivity_grid.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    grid.to_csv(os.path.join(args.output_dir, "sensitivity_grid.csv"), index=False)

    print(json.dumps(summary, indent=2, default=float))
    print(f"wrote {args.output_dir}/sensitivity_grid.json")


if __name__ == "__main__":
    main()
