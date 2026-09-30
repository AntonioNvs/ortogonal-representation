"""Run XAI falsification tests on a trained SkillGNN checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.skill_gnn_skill import get_skill_gnn_db
from explain.skill_gnn_probes import ProbeSampleConfig, run_xai_probes


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI falsification tests for SkillGNN")
    parser.add_argument("--checkpoint", type=str, default="output/skill_model/skill_gnn.pth")
    parser.add_argument("--meta", type=str, default="output/skill_model/skill_gnn_meta.json")
    parser.add_argument("--output", type=str, default="output/skill_evaluation/xai_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--min-year", type=int, default=2024)
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

    config = ProbeSampleConfig(
        min_year=args.min_year,
        max_year=args.max_year,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    report = run_xai_probes(
        db,
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
        config=config,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")

    gates = report.get("gates", {})
    if gates.get("constructor_leakage") and gates.get("swap_invariance"):
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
