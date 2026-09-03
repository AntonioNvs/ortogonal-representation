#!/usr/bin/env python3
"""Run career validation across multiple skill sources and compare gates."""

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
from validation.career_metrics import compute_career_metrics, parse_horizon_arg
from validation.inference import compare_resolution_rates
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers

DEFAULT_SOURCES = ["bradley_terry", "plackett_luce", "bayesian_ssm", "orthogonal_shapley"]
BASELINE = "bradley_terry"


def evaluate_career_gates(report: dict, *, baseline_report: dict | None = None) -> dict:
    """Evaluate career gates for one source against optional BT baseline."""
    ur = report.get("underrated_resolution", {})
    uauroc = report.get("underrated_promotion_auroc", {})
    ups = report.get("underrated_partial_spearman", {})
    partial = report.get("partial_spearman", {})

    bt_ur = (baseline_report or {}).get("underrated_resolution", {})
    bt_uauroc = (baseline_report or {}).get("underrated_promotion_auroc", {})
    bt_ups = (baseline_report or {}).get("underrated_partial_spearman", {})

    resolution_rate = ur.get("resolution_rate", float("nan"))
    resolution_ci_low = ur.get("ci_low", float("nan"))
    bt_resolution = bt_ur.get("resolution_rate", float("nan"))

    auroc_val = uauroc.get("auroc", float("nan"))
    auroc_ci_low = uauroc.get("ci_low", float("nan"))
    bt_auroc = bt_uauroc.get("auroc", float("nan"))

    underrated_rho = ups.get("partial_rho", float("nan"))
    underrated_ci_low = ups.get("ci_low", float("nan"))
    bt_underrated_rho = bt_ups.get("partial_rho", float("nan"))

    is_baseline = baseline_report is None
    return {
        "source": report.get("skill_source", ""),
        "n_underrated": report.get("n_underrated", 0),
        "resolution_rate": resolution_rate,
        "resolution_ci_low": resolution_ci_low,
        "underrated_auroc": auroc_val,
        "underrated_auroc_ci_low": auroc_ci_low,
        "underrated_partial_rho": underrated_rho,
        "underrated_partial_ci_low": underrated_ci_low,
        "partial_rho_all": partial.get("partial_rho", float("nan")),
        "partial_ci_low_all": partial.get("ci_low", float("nan")),
        "gates": {
            "resolution_vs_baseline": (
                is_baseline
                or (resolution_rate >= bt_resolution and resolution_ci_low > 0.5)
            ),
            "underrated_auroc_vs_baseline": (
                is_baseline or (auroc_val >= bt_auroc and auroc_ci_low > 0.45)
            ),
            "underrated_partial_vs_baseline": (
                is_baseline or (underrated_rho >= bt_underrated_rho and underrated_ci_low > -0.1)
            ),
            "partial_all_diagnostic": partial.get("partial_rho", 0) >= 0.15
            and partial.get("ci_low", -1) > 0,
        },
        "baseline_reference": {
            "resolution_rate": bt_resolution,
            "underrated_auroc": bt_auroc,
            "underrated_partial_rho": bt_underrated_rho,
        }
        if not is_baseline
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare career validation across skill sources")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", type=str, default="output/career_validation")
    parser.add_argument("--horizon", type=str, default="inf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Load correlation.json from output dirs instead of recomputing",
    )
    args = parser.parse_args()
    horizon = parse_horizon_arg(args.horizon)

    reports: dict[str, dict] = {}

    if not args.reuse_existing:
        data_tasks.register_all(
            enriched_db_dir=cfg.ENRICHED_DB_DIR,
            min_year=cfg.MIN_YEAR,
            max_year=cfg.MAX_YEAR,
            val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
            test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
        )
        db = get_skill_gnn_db()
        lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
        team_tier = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)

    for source in args.sources:
        out_dir = os.path.join(args.output_dir, source)
        corr_path = os.path.join(out_dir, "correlation.json")

        if args.reuse_existing and os.path.isfile(corr_path):
            with open(corr_path) as f:
                reports[source] = json.load(f)
            print(f"loaded {source} from {corr_path}")
            continue

        print(f"running career validation for {source}...")
        export = load_skill_export(source, db)
        skill = season_scores_for_career(export)
        joined = join_career(skill, db, team_tier, horizon=horizon)
        report, marked = compute_career_metrics(
            joined, skill_source=source, horizon=horizon, seed=args.seed
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(corr_path, "w") as f:
            json.dump(report, f, indent=2, default=float)
        with open(os.path.join(out_dir, "resolution_report.json"), "w") as f:
            json.dump(
                {
                    "underrated_resolution": report["underrated_resolution"],
                    "underrated_promotion_auroc": report["underrated_promotion_auroc"],
                    "time_to_promotion": report["time_to_promotion"],
                    "resolution_by_decade": report["resolution_by_decade"],
                    "underrated_partial_spearman": report["underrated_partial_spearman"],
                },
                f,
                indent=2,
                default=float,
            )
        joined.to_csv(os.path.join(out_dir, "joined.csv"), index=False)
        marked[marked["underrated_flag"]].to_csv(
            os.path.join(out_dir, "inconsistencies.csv"), index=False
        )
        reports[source] = report

    baseline = reports.get(BASELINE)
    gate_results = []
    for source, report in reports.items():
        gate_results.append(
            evaluate_career_gates(
                report,
                baseline_report=None if source == BASELINE else baseline,
            )
        )

    comparison = {
        "horizon": "inf" if horizon is None else horizon,
        "baseline": BASELINE,
        "sources": reports,
        "gates": gate_results,
        "resolution_comparison": compare_resolution_rates(
            {s: r for s, r in reports.items()}
        ),
        "calibration_notes": {
            "resolution_gate": "rate >= BT baseline AND cluster CI low > 0.5",
            "auroc_gate": "AUROC >= BT baseline AND cluster CI low > 0.45 (calibrated: BT=0.57, n=49)",
            "underrated_partial_gate": "rho >= BT baseline AND cluster CI low > -0.1 (small-n stratum)",
            "partial_all_gate": "diagnostic only: partial rho >= 0.15, CI low > 0",
        },
    }

    comp_path = os.path.join(args.output_dir, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=float)

    lines = [
        "# Career validation comparison",
        "",
        f"Horizon: {comparison['horizon']}",
        f"Baseline: {BASELINE}",
        "",
        "| Source | n_underrated | resolution | AUROC | underrated ρ | beats BT resolution |",
        "|--------|--------------|------------|-------|--------------|---------------------|",
    ]
    for g in gate_results:
        lines.append(
            f"| {g['source']} | {g['n_underrated']} | "
            f"{g['resolution_rate']:.3f} | {g['underrated_auroc']:.3f} | "
            f"{g['underrated_partial_rho']:.3f} | "
            f"{g['gates']['resolution_vs_baseline']} |"
        )
    with open(os.path.join(args.output_dir, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"gates": gate_results, "comparison_path": comp_path}, indent=2, default=float))


if __name__ == "__main__":
    main()
