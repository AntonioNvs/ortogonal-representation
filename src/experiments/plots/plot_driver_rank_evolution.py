#!/usr/bin/env python3
"""Plot four-panel driver skill evolution across seasons."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.skill_loader import load_skill_export
from baselines.skill_gnn_skill import get_skill_gnn_db
from skill.contract import InferenceMode
from visualization.driver_rank_evolution import plot_driver_rank_evolution
from visualization.style import resolve_plot_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-season driver rank evolution")
    parser.add_argument("--source", type=str, default="bradley_terry")
    parser.add_argument("--driver", action="append", required=True, dest="drivers")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--show-sd", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()
    export = load_skill_export(
        args.source, db, inference_mode=InferenceMode.FILTERED, force_recompute=args.force_recompute
    )
    out = args.output or f"output/plots/{args.source}_evolution_{args.start_year}_{args.end_year}"
    plot_driver_rank_evolution(
        export.season,
        export.race,
        drivers=args.drivers,
        start_year=args.start_year,
        end_year=args.end_year,
        show_sd=args.show_sd,
        output_path=out,
    )
    out_dir, _ = resolve_plot_output_dir(out)
    print(f"saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
