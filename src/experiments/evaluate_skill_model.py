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
from baselines.skill_loader import load_skill_export, season_scores_for_career
from baselines.skill_gnn_skill import get_skill_gnn_db
from data.race_panel import RacePanelConfig, build_race_panel
from skill.contract import InferenceMode
from validation.benchmark import benchmark_source, evaluate_benchmark_gates, write_benchmark_report
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers


def load_skill_by_source(source, db, team_tier=None, **kwargs):
    """Backward-compatible season-level loader."""
    export = load_skill_export(
        source if source != "bayesian_comparator" else "bayesian_ssm",
        db,
        max_year=kwargs.get("max_year", 2025),
        checkpoint_path=kwargs.get("checkpoint_path", "output/skill_model/skill_gnn.pth"),
        meta_path=kwargs.get("meta_path", "output/skill_model/skill_gnn_meta.json"),
    )
    return season_scores_for_career(export)


def join_career(
    skill: pd.DataFrame,
    db,
    team_tier: pd.DataFrame,
    horizon: int | None = None,
    *,
    enrich_trajectory: bool = True,
    cohort_skill: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join season skill scores to forward tier outcomes.

    Parameters
    ----------
    horizon : int or None
        If None (default), use rest-of-career mean tier score (infinite horizon).
        If int, use fixed forward window T+1..T+horizon with require_full_horizon.
    cohort_skill : DataFrame or None
        Optional model-free season scores (``driverId``, ``season``,
        ``skill_score``) merged in as ``cohort_skill_score`` for a fixed
        underrated-cohort definition.
    """
    from validation.career_labels import career_outcome_labels, driver_season_constructor
    from validation.skill_trajectory import enrich_career_join
    from validation.team_tiers import TIER_TO_SCORE

    ds = driver_season_constructor(db)
    if horizon is None:
        forward = career_outcome_labels(ds, team_tier, horizon=None)
    else:
        forward = career_outcome_labels(
            ds, team_tier, horizon=horizon, require_full_horizon=True
        )
    tier_at_t = team_tier.rename(columns={"season": "season_T"})[
        ["constructorId", "season_T", "tier", "score"]
    ]
    ds_t = ds.rename(columns={"season": "season_T"})
    ds_t = ds_t.merge(tier_at_t, on=["constructorId", "season_T"], how="left")
    ds_t["constructor_tier_score_at_T"] = ds_t["tier"].map(TIER_TO_SCORE)
    ds_t["constructor_score_at_T"] = ds_t["score"]

    joined = skill.rename(columns={"season": "season_T"}).merge(
        forward, on=["driverId", "season_T"], how="inner"
    )
    joined = joined.merge(
        ds_t[["driverId", "season_T", "constructor_tier_score_at_T", "constructor_score_at_T"]],
        on=["driverId", "season_T"],
        how="left",
    )
    if enrich_trajectory:
        joined = enrich_career_join(joined, skill)
    if cohort_skill is not None:
        cs = cohort_skill[["driverId", "season", "skill_score"]].rename(
            columns={"season": "season_T", "skill_score": "cohort_skill_score"}
        )
        joined = joined.merge(cs, on=["driverId", "season_T"], how="left")
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate skill ranking models")
    parser.add_argument(
        "--skill-sources",
        nargs="+",
        default=["bradley_terry", "teammate_residual", "bayesian_ssm", "constructor_tier"],
    )
    parser.add_argument("--output-dir", type=str, default="output/skill_evaluation")
    parser.add_argument("--checkpoint", type=str, default="output/skill_model/skill_gnn.pth")
    parser.add_argument("--meta", type=str, default="output/skill_model/skill_gnn_meta.json")
    parser.add_argument("--xai-seed", type=int, default=42)
    parser.add_argument("--max-year", type=int, default=2025)
    args = parser.parse_args()

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()
    panel = build_race_panel(db, RacePanelConfig(max_year=args.max_year))
    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    team_tier = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)

    sources = ["skill_gnn" if s == "skill_gnn" else ("bayesian_ssm" if s == "bayesian_comparator" else s) for s in args.skill_sources]
    reports = {}
    for source in sources:
        print(f"evaluating {source}...")
        try:
            export = load_skill_export(
                source,
                db,
                max_year=args.max_year,
                checkpoint_path=args.checkpoint,
                meta_path=args.meta,
            )
        except FileNotFoundError as exc:
            if source == "skill_gnn":
                print(f"skipping skill_gnn: {exc}")
                continue
            raise
        reports[source] = benchmark_source(
            export,
            db,
            team_tier,
            panel,
            checkpoint_path=args.checkpoint if source == "skill_gnn" else None,
            meta_path=args.meta,
            xai_seed=args.xai_seed,
        )

    gates = evaluate_benchmark_gates(reports)
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {"sources": reports, "gates": gates}
    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    write_benchmark_report(reports, args.output_dir)
    print(f"wrote {args.output_dir}/evaluation.json")


if __name__ == "__main__":
    main()
