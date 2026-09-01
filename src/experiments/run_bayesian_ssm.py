#!/usr/bin/env python3
"""Fit Lindner et al. (2026) Bayesian SSM via Stan/NUTS."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.bayesian_ssm import StanFitConfig, export_bayesian_ssm
from baselines.skill_gnn_skill import get_skill_gnn_db
from skill.contract import InferenceMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian state-space skill model (Stan/NUTS)")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--output-dir", type=str, default="output/skill_exports/bayesian_ssm")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Minimal MCMC for testing")
    parser.add_argument(
        "--mode",
        choices=["filtered", "smoothed"],
        default="smoothed",
        help="smoothed for paper replication; filtered for causal gates",
    )
    args = parser.parse_args()
    if args.smoke_test and args.end_year - args.start_year > 1:
        args.start_year = args.end_year - 1
        print(f"smoke-test: using {args.start_year}-{args.end_year} for faster MCMC")

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()
    stan_config = StanFitConfig(
        chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.draws,
        seed=args.seed,
    )
    export = export_bayesian_ssm(
        db,
        start_year=args.start_year,
        end_year=args.end_year,
        inference_mode=InferenceMode(args.mode),
        output_dir=args.output_dir,
        stan_config=stan_config,
        smoke_test=args.smoke_test,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    export.race.to_parquet(os.path.join(args.output_dir, "race_skill.parquet"), index=False)
    export.season.to_csv(os.path.join(args.output_dir, "season_skill.csv"), index=False)
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(export.metadata.to_dict(), f, indent=2)
    print(f"wrote export to {args.output_dir} ({len(export.race)} race rows)")


if __name__ == "__main__":
    main()
