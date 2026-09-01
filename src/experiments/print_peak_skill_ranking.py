#!/usr/bin/env python3
"""Print global top-N peak (driver, season) pairs from a skill export."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

import config as cfg
import data.tasks as data_tasks
from baselines.skill_loader import load_skill_export
from baselines.skill_gnn_skill import get_skill_gnn_db
from skill.contract import InferenceMode
from skill.scoring import peak_season_skill
from utils.naming import build_driver_name_map


def _driver_label(row: pd.Series, name_by_id: dict[int, str]) -> str:
    name = row.get("driver_name")
    if pd.notna(name) and str(name).strip():
        return str(name).strip()
    return name_by_id.get(int(row["driverId"]), f"driverId={int(row['driverId'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print top peak (driver, season) pairs from a skill export"
    )
    parser.add_argument("--source", type=str, default="bradley_terry")
    parser.add_argument("--top", type=int, default=10, help="Number of rows to print")
    parser.add_argument("--min-races", type=int, default=1, help="Minimum races in the season")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="output/skill_model/skill_gnn.pth")
    parser.add_argument("--meta", type=str, default="output/skill_model/skill_gnn_meta.json")
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
        args.source,
        db,
        inference_mode=InferenceMode.FILTERED,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
    )

    peaks = peak_season_skill(export.race)
    if args.min_races > 1:
        peaks = peaks[peaks["n_races"] >= args.min_races]

    name_by_id = build_driver_name_map(db.table_dict["drivers"].df)

    top = peaks.head(args.top)
    print(f"Top {len(top)} peak (driver, season) — source={args.source} (season mean)")
    print(f"{'rank':>4}  {'driver':<24}  {'season':>6}  {'peak':>6}  {'races':>5}")
    print("-" * 52)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        label = _driver_label(row, name_by_id)
        print(
            f"{rank:>4}  {label:<24}  {int(row['season']):>6}  "
            f"{float(row['peak_skill']):>6.2f}  {int(row['n_races']):>5}"
        )


if __name__ == "__main__":
    main()
