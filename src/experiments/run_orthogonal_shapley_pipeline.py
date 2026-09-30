#!/usr/bin/env python3
"""End-to-end pipeline for OrthogonalShapleyGNN: train, export, validate, plots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data.tasks as data_tasks
from baselines.orthogonal_shapley_skill import get_orthogonal_shapley_db
from baselines.skill_loader import load_skill_export
from data.race_panel import RacePanelConfig, build_race_panel
from skill.contract import InferenceMode
from validation.benchmark import benchmark_source, write_benchmark_report
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers

DEFAULT_MODEL_DIR = "output/orthogonal_shapley_model"
DEFAULT_EXPORT_DIR = "output/skill_exports/orthogonal_shapley"
DEFAULT_BENCHMARK_DIR = "output/validation_benchmark/orthogonal_shapley"
DEFAULT_PLOTS_DIR = "output/plots/orthogonal_shapley"


def _run_train(args: argparse.Namespace) -> None:
  cmd = [
    sys.executable,
    "src/experiments/train_orthogonal_shapley_gnn.py",
    "--epochs", str(args.epochs),
    "--seed", str(args.seed),
    "--gpu-id", str(args.gpu_id),
    "--output-dir", args.model_dir,
    "--lambda-orth", str(args.lambda_orth),
    "--aux-driver-weight", str(args.aux_driver_weight),
    "--aux-constructor-weight", str(args.aux_constructor_weight),
    "--lambda-ctx-aux", str(args.lambda_ctx_aux),
    "--lambda-pair", str(args.lambda_pair),
    "--lambda-attr", str(args.lambda_attr),
    "--hidden-dim", str(args.hidden_dim),
    "--num-layers", str(args.num_layers),
    "--mlp-hidden", str(args.mlp_hidden),
  ]
  if args.smoke_test:
    cmd.append("--smoke-test")
  if args.lambda_grid:
    cmd.extend(["--lambda-grid", args.lambda_grid])
  print("Running:", " ".join(cmd))
  subprocess.run(cmd, check=True)


def _run_export(args: argparse.Namespace) -> None:
  data_tasks.register_all(
    enriched_db_dir=cfg.ENRICHED_DB_DIR,
    min_year=cfg.MIN_YEAR,
    max_year=cfg.MAX_YEAR,
    val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
    test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
  )
  db = get_orthogonal_shapley_db()
  ckpt = os.path.join(args.model_dir, "orthogonal_shapley.pth")
  meta = os.path.join(args.model_dir, "orthogonal_shapley_meta.json")
  baselines = os.path.join(args.model_dir, "coalition_baselines.json")
  export = load_skill_export(
    "orthogonal_shapley",
    db,
    max_year=args.max_year,
    inference_mode=InferenceMode.FILTERED,
    output_dir=args.export_dir,
    checkpoint_path=ckpt,
    meta_path=meta,
    baselines_path=baselines,
    force_recompute=True,
  )
  export.validate()
  print(f"exported {len(export.race)} race rows to {args.export_dir}")


def _run_validate(args: argparse.Namespace) -> None:
  data_tasks.register_all(
    enriched_db_dir=cfg.ENRICHED_DB_DIR,
    min_year=cfg.MIN_YEAR,
    max_year=cfg.MAX_YEAR,
    val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
    test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
  )
  db = get_orthogonal_shapley_db()
  panel = build_race_panel(db, RacePanelConfig(max_year=args.max_year))
  lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
  team_tier = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)

  ckpt = os.path.join(args.model_dir, "orthogonal_shapley.pth")
  meta = os.path.join(args.model_dir, "orthogonal_shapley_meta.json")
  baselines = os.path.join(args.model_dir, "coalition_baselines.json")

  reports = {}
  for source in args.compare_sources:
    print(f"benchmarking {source}...")
    export = load_skill_export(
      source,
      db,
      max_year=args.max_year,
      inference_mode=InferenceMode.FILTERED,
      force_recompute=(source == "orthogonal_shapley"),
      checkpoint_path=ckpt if source == "orthogonal_shapley" else "output/skill_model/skill_gnn.pth",
      meta_path=meta if source == "orthogonal_shapley" else "output/skill_model/skill_gnn_meta.json",
      baselines_path=baselines if source == "orthogonal_shapley" else None,
      output_dir=args.export_dir if source == "orthogonal_shapley" else None,
    )
    reports[source] = benchmark_source(
      export,
      db,
      team_tier,
      panel,
      checkpoint_path=ckpt if source == "orthogonal_shapley" else None,
      meta_path=meta if source == "orthogonal_shapley" else None,
      baselines_path=baselines if source == "orthogonal_shapley" else None,
      xai_seed=args.seed,
    )

  os.makedirs(args.benchmark_dir, exist_ok=True)
  write_benchmark_report(reports, args.benchmark_dir)
  print(f"wrote benchmark to {args.benchmark_dir}")


def _run_plots(args: argparse.Namespace) -> None:
  plot_cmds = [
    [
      sys.executable,
      "src/experiments/plots/plot_driver_season_skill.py",
      "--source", "orthogonal_shapley",
      "--season", "2024",
      "--driver", "verstappen",
      "--output", os.path.join(args.plots_dir, "season_skill_2024"),
    ],
    [
      sys.executable,
      "src/experiments/plots/plot_driver_rank_evolution.py",
      "--source", "orthogonal_shapley",
      "--driver", "verstappen",
      "--driver", "hamilton",
      "--start-year", "2018",
      "--end-year", "2024",
      "--output", os.path.join(args.plots_dir, "rank_evolution"),
    ],
    [
      sys.executable,
      "src/experiments/plots/plot_entity_attribution.py",
      "--source", "orthogonal_shapley",
      "--season", "2024",
      "--output", os.path.join(args.plots_dir, "attribution_2024"),
    ],
  ]
  for cmd in plot_cmds:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


STAGES = ("train", "export", "validate", "plots", "all")


def main() -> None:
  parser = argparse.ArgumentParser(description="OrthogonalShapleyGNN end-to-end pipeline")
  parser.add_argument(
    "--stages",
    nargs="+",
    default=["all"],
    choices=STAGES,
    help="pipeline stages to run",
  )
  parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--export-dir", type=str, default=DEFAULT_EXPORT_DIR)
  parser.add_argument("--benchmark-dir", type=str, default=DEFAULT_BENCHMARK_DIR)
  parser.add_argument("--plots-dir", type=str, default=DEFAULT_PLOTS_DIR)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--lambda-orth", type=float, default=2.0)
  parser.add_argument("--lambda-grid", type=str, default=None)
  parser.add_argument("--aux-driver-weight", type=float, default=0.5)
  parser.add_argument("--aux-constructor-weight", type=float, default=0.75)
  parser.add_argument("--lambda-ctx-aux", type=float, default=0.25)
  parser.add_argument("--lambda-pair", type=float, default=0.25)
  parser.add_argument("--lambda-attr", type=float, default=0.1)
  parser.add_argument("--hidden-dim", type=int, default=64)
  parser.add_argument("--num-layers", type=int, default=3)
  parser.add_argument("--mlp-hidden", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--gpu-id", type=int, default=cfg.DEFAULT_GPU_ID)
  parser.add_argument("--max-year", type=int, default=2025)
  parser.add_argument("--smoke-test", action="store_true")
  parser.add_argument(
    "--compare-sources",
    nargs="+",
    default=["orthogonal_shapley", "bradley_terry"],
    help="sources for validation benchmark",
  )
  args = parser.parse_args()

  stages = set(args.stages)
  if "all" in stages:
    stages = {"train", "export", "validate", "plots"}

  if "train" in stages:
    _run_train(args)
  if "export" in stages:
    _run_export(args)
  if "validate" in stages:
    _run_validate(args)
  if "plots" in stages:
    _run_plots(args)

  print("Pipeline complete.")


if __name__ == "__main__":
  main()
