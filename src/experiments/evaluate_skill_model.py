"""Evaluate skill models against baselines and validation gates."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath("src"))

import config as cfg
import data.tasks as data_tasks
from baselines.bradley_terry_skill import load_bradley_terry_skill
from baselines.bayesian_comparator import load_bayesian_comparator_skill
from baselines.skill_gnn_skill import load_skill_gnn_skill
from baselines.teammate_residual import load_teammate_residual_skill
from baselines.skill_gnn_skill import get_skill_gnn_db
from data.mobility import build_race_pairs_for_bt, compute_mobility_report
from data.skill_dataset import SkillDatasetConfig, build_skill_dataset
from validation.baselines import load_constructor_tier, load_points_share
from validation.career_labels import driver_season_constructor, forward_tier_outcome
from explain.skill_gnn_probes import ProbeSampleConfig, infer_claim_level, run_xai_probes
from validation.inference import moved_up_auroc
from validation.skill_validation import compare_skill_sources, evaluate_gates
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import TIER_TO_SCORE, compute_constructor_season_points, compute_team_tiers


def load_skill_by_source(
    source: str,
    db,
    team_tier=None,
    *,
    checkpoint_path: str = "output/skill_model/skill_gnn.pth",
    meta_path: str = "output/skill_model/skill_gnn_meta.json",
):
    if source == "skill_gnn":
        return load_skill_gnn_skill(
            db,
            checkpoint_path=checkpoint_path,
            meta_path=meta_path,
        )
    if source == "bradley_terry":
        return load_bradley_terry_skill()
    if source == "teammate_residual":
        return load_teammate_residual_skill(db)
    if source == "bayesian_comparator":
        return load_bayesian_comparator_skill(db)
    if source == "points_share":
        return load_points_share(db)
    if source == "constructor_tier":
        return load_constructor_tier(db, team_tier)
    raise ValueError(f"unknown skill source: {source}")


def join_career(skill: pd.DataFrame, db, team_tier: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    ds = driver_season_constructor(db)
    forward = forward_tier_outcome(ds, team_tier, horizon=horizon, require_full_horizon=True)
    tier_at_t = team_tier.rename(columns={"season": "season_T"})[
        ["constructorId", "season_T", "tier"]
    ]
    ds_t = ds.rename(columns={"season": "season_T"})
    ds_t = ds_t.merge(tier_at_t, on=["constructorId", "season_T"], how="left")
    ds_t["constructor_tier_score_at_T"] = ds_t["tier"].map(TIER_TO_SCORE)

    joined = skill.rename(columns={"season": "season_T"}).merge(
        forward, on=["driverId", "season_T"], how="inner"
    )
    joined = joined.merge(
        ds_t[["driverId", "season_T", "constructor_tier_score_at_T"]],
        on=["driverId", "season_T"],
        how="left",
    )
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate skill ranking models")
    parser.add_argument(
        "--skill-sources",
        nargs="+",
        default=[
            "skill_gnn",
            "bradley_terry",
            "teammate_residual",
            "bayesian_comparator",
            "constructor_tier",
        ],
    )
    parser.add_argument("--output-dir", type=str, default="output/skill_evaluation")
    parser.add_argument("--checkpoint", type=str, default="output/skill_model/skill_gnn.pth")
    parser.add_argument("--meta", type=str, default="output/skill_model/skill_gnn_meta.json")
    parser.add_argument("--xai-seed", type=int, default=42)
    args = parser.parse_args()

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()

    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    points = compute_constructor_season_points(db)
    team_tier = compute_team_tiers(points, window=3, lineage=lineage)

    mobility = compute_mobility_report(db, SkillDatasetConfig(max_year=2025))
    skill_df = build_skill_dataset(db, SkillDatasetConfig(max_year=2025))
    pairs = build_race_pairs_for_bt(skill_df)

    results: dict = {"mobility": mobility.to_dict(), "sources": {}}
    bt_pairwise_acc = float("nan")
    skill_gnn_pairwise_acc = float("nan")
    skill_gnn_metrics: dict = {}

    for source in args.skill_sources:
        print(f"evaluating {source}...")
        skill = load_skill_by_source(
            source,
            db,
            team_tier,
            checkpoint_path=args.checkpoint,
            meta_path=args.meta,
        )
        joined = join_career(skill, db, team_tier)
        metrics = compare_skill_sources(joined)
        auroc = moved_up_auroc(joined)

        skill_by_driver = skill.groupby("driverId")["skill_score"].mean().to_dict()
        proxy_nll = 1.0 - sum(
            1
            for _, r in pairs.iterrows()
            if skill_by_driver.get(int(r["driverA"]), 0) > skill_by_driver.get(int(r["driverB"]), 0)
        ) / max(len(pairs), 1)

        pairwise_acc = 1.0 - proxy_nll
        results["sources"][source] = {
            **metrics,
            "moved_up_auroc": auroc.get("auroc", float("nan")),
            "pairwise_acc_proxy": pairwise_acc,
            "n_joined": len(joined),
        }
        if source == "bradley_terry":
            bt_pairwise_acc = pairwise_acc
        if source == "skill_gnn":
            skill_gnn_pairwise_acc = pairwise_acc
            skill_gnn_metrics = metrics

    if "skill_gnn" in args.skill_sources and os.path.isfile(args.checkpoint):
        print("running XAI probes on skill_gnn...")
        xai_report = run_xai_probes(
            db,
            checkpoint_path=args.checkpoint,
            meta_path=args.meta,
            config=ProbeSampleConfig(seed=args.xai_seed),
        )
        results["xai"] = {
            "constructor_leakage_rho": xai_report["constructor_leakage_rho"],
            "swap_invariance": xai_report["swap_invariance"],
            "channel_decomposition": xai_report["channel_decomposition"],
            "n_samples": xai_report["n_samples"],
            "seed": xai_report["seed"],
        }

        gate_results = evaluate_gates(
            primary_metrics=skill_gnn_metrics,
            leakage_rho=xai_report["constructor_leakage_rho"],
            swap_skill_diff=xai_report["swap_invariance"]["skill_diff"],
            bt_pairwise_acc=bt_pairwise_acc,
            model_pairwise_acc=skill_gnn_pairwise_acc,
        )
        results["gates"] = [
            {
                "name": g.name,
                "passed": g.passed,
                "metrics": g.metrics,
                "notes": g.notes,
            }
            for g in gate_results
        ]
        results["claim_level"] = infer_claim_level(
            partial_rho=skill_gnn_metrics.get("partial_rho", float("nan")),
            partial_ci_low=skill_gnn_metrics.get("partial_ci_low", float("nan")),
            leakage_rho=xai_report["constructor_leakage_rho"],
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
