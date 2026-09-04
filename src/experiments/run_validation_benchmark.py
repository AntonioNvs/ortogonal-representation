#!/usr/bin/env python3
"""Run unified validation benchmark for one or more skill sources."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.skill_loader import load_skill_export
from data.race_panel import RacePanelConfig, build_race_panel
from skill.contract import InferenceMode
from validation.benchmark import benchmark_source, write_benchmark_report
from validation.career_metrics import parse_horizon_arg
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-agnostic validation benchmark")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["bradley_terry"],
        help="skill sources: bradley_terry, bayesian_ssm, skill_gnn, ...",
    )
    parser.add_argument("--output-dir", type=str, default="output/validation_benchmark")
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="output/skill_model/skill_gnn.pth")
    parser.add_argument("--meta", type=str, default="output/skill_model/skill_gnn_meta.json")
    parser.add_argument("--xai-seed", type=int, default=42)
    parser.add_argument(
        "--horizon",
        type=str,
        default="inf",
        help="Career forward horizon: 'inf' for rest-of-career (default) or integer seasons",
    )
    parser.add_argument(
        "--fixed-cohort",
        action="store_true",
        help="Define underrated cohort on model-free teammate_residual and add a paired bootstrap comparison",
    )
    parser.add_argument("--min-year", type=int, default=None, help="Restrict career metrics to season_T >= min_year")
    parser.add_argument(
        "--era-windows",
        action="store_true",
        help="Also emit era_windows: modern (>=2010), hybrid (>=1990), common (>=2014), full",
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
    from baselines.skill_gnn_skill import get_skill_gnn_db

    db = get_skill_gnn_db()
    panel = build_race_panel(db, RacePanelConfig(max_year=args.max_year))
    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    team_tier = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)

    cohort_skill = None
    if args.fixed_cohort:
        from baselines.skill_loader import season_scores_for_career

        cohort_skill = season_scores_for_career(
            load_skill_export("teammate_residual", db, max_year=args.max_year)
        )

    reports = {}
    joined_by_source = {}
    for source in args.sources:
        print(f"benchmarking {source}...")
        export = load_skill_export(
            source,
            db,
            max_year=args.max_year,
            inference_mode=InferenceMode.FILTERED,
            force_recompute=args.force_recompute,
            checkpoint_path=args.checkpoint,
            meta_path=args.meta,
        )
        reports[source] = benchmark_source(
            export,
            db,
            team_tier,
            panel,
            checkpoint_path=args.checkpoint if source == "skill_gnn" else None,
            meta_path=args.meta,
            xai_seed=args.xai_seed,
            horizon=horizon,
            cohort_skill=cohort_skill,
            min_year=args.min_year,
            era_windows=args.era_windows,
        )
        if args.fixed_cohort:
            from validation.benchmark import join_career_from_export

            joined = join_career_from_export(
                export, db, team_tier, horizon=horizon, cohort_skill=cohort_skill
            )
            # Apply the same era window as the career/survival blocks so the
            # fixed cohort is the >=min_year underrated set (the within-season
            # percentile is recomputed inside mark_underrated on the window).
            if args.min_year is not None and "season_T" in joined.columns:
                joined = joined[joined["season_T"] >= args.min_year]
            joined_by_source[source] = joined

    extra = {}
    if args.fixed_cohort and len(joined_by_source) >= 2:
        from validation.inference import fixed_cohort_paired_comparison

        extra["fixed_cohort"] = fixed_cohort_paired_comparison(
            joined_by_source, cohort_skill_col="cohort_skill_score", seed=args.xai_seed
        )

    write_benchmark_report(reports, args.output_dir, extra=extra)
    print(f"wrote {args.output_dir}/benchmark.json and benchmark.md")


if __name__ == "__main__":
    main()
