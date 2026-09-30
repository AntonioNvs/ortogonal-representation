#!/usr/bin/env python3
"""Plot team tier heatmap (lineage-aware)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.skill_gnn_skill import get_skill_gnn_db
from visualization.style import resolve_plot_output_dir
from visualization.team_tier_heatmap import build_tier_heatmap_from_db, plot_team_tier_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Team tier heatmap")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()
    tiers, lm = build_tier_heatmap_from_db(db, args.start_year, args.end_year)
    out = args.output or f"output/plots/tier_heatmap_{args.start_year}_{args.end_year}"
    plot_team_tier_heatmap(tiers, lm, start_year=args.start_year, end_year=args.end_year, output_path=out)
    out_dir, _ = resolve_plot_output_dir(out)
    print(f"saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
