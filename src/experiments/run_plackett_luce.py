#!/usr/bin/env python3
"""Export walk-forward Plackett-Luce skill artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.plackett_luce_skill import export_plackett_luce
from baselines.skill_gnn_skill import get_skill_gnn_db
from skill.contract import InferenceMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plackett-Luce benchmark export")
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=str, default="output/skill_exports/plackett_luce")
    parser.add_argument("--epochs-per-step", type=int, default=cfg.PL_EPOCHS_PER_STEP)
    parser.add_argument("--lr", type=float, default=cfg.PL_LR)
    parser.add_argument("--weight-decay", type=float, default=cfg.PL_WEIGHT_DECAY)
    args = parser.parse_args()

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()
    export = export_plackett_luce(
        db,
        max_year=args.max_year,
        inference_mode=InferenceMode.FILTERED,
        lr=args.lr,
        epochs=args.epochs_per_step,
        weight_decay=args.weight_decay,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    export.race.to_parquet(os.path.join(args.output_dir, "race_skill.parquet"), index=False)
    export.season.to_csv(os.path.join(args.output_dir, "season_skill.csv"), index=False)
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(export.metadata.to_dict(), f, indent=2)
    print(f"wrote export to {args.output_dir} ({len(export.race)} race rows)")


if __name__ == "__main__":
    main()
