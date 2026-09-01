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
    args = parser.parse_args()

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

    reports = {}
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
        )

    write_benchmark_report(reports, args.output_dir)
    print(f"wrote {args.output_dir}/benchmark.json and benchmark.md")


if __name__ == "__main__":
    main()
