#!/usr/bin/env python3
"""No-attribution-balance ablation for OrthogonalShapleyGNN.

Trains a twin of frozen Model A with ``lambda_attr=0`` (all other losses and
hyperparameters unchanged), then exports skills, runs the fixed ≥2014 validation
benchmark against baselines, regenerates the mobility screen, and writes a
compact comparison report vs the regularized checkpoint.

    python src/experiments/run_attr_ablation.py --stages all --gpu-id 0 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg

# Match frozen Model A (output/orthogonal_shapley_model/orthogonal_shapley_meta.json).
FROZEN_MODEL_A = {
    "hidden_dim": 128,
    "num_layers": 4,
    "mlp_hidden": 128,
    "lambda_orth": 2.0,
    "aux_driver_weight": 0.5,
    "aux_constructor_weight": 0.75,
    "lambda_ctx_aux": 0.25,
    "lambda_pair": 0.25,
    "lambda_attr": 0.0,  # ablation: drop attribution_balance_loss only
    "target_driver_share": 0.38,
    "target_constructor_share": 0.30,
    "lambda_rw": 0.5,
    "lambda_shrink": 0.05,
    "lambda_quali": 0.0,
    "orth_warmup_epochs": 10,
    "max_grad_norm": 1.0,
    "epochs": 100,
    "patience": 10,
}

DEFAULT_MODEL_DIR = "output/orthogonal_shapley_no_attr"
DEFAULT_EXPORT_DIR = "output/skill_exports/orthogonal_shapley_no_attr"
DEFAULT_BENCHMARK_DIR = "output/validation_benchmark/orthogonal_shapley_no_attr"
DEFAULT_UNDERVALUED_DIR = "output/applications/undervalued_no_attr"
DEFAULT_REPORT_PATH = "output/applications/attr_ablation_report.json"
REGULARIZED_MODEL_DIR = "output/orthogonal_shapley_model"
REGULARIZED_BENCHMARK = "output/validation_benchmark/benchmark.json"

STAGES = ("train", "export", "validate", "undervalued", "report", "all")


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _train(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "src/experiments/train_orthogonal_shapley_gnn.py",
        "--epochs", str(args.epochs),
        "--patience", str(FROZEN_MODEL_A["patience"]),
        "--seed", str(args.seed),
        "--gpu-id", str(args.gpu_id),
        "--output-dir", args.model_dir,
        "--hidden-dim", str(FROZEN_MODEL_A["hidden_dim"]),
        "--num-layers", str(FROZEN_MODEL_A["num_layers"]),
        "--mlp-hidden", str(FROZEN_MODEL_A["mlp_hidden"]),
        "--lambda-orth", str(FROZEN_MODEL_A["lambda_orth"]),
        "--aux-driver-weight", str(FROZEN_MODEL_A["aux_driver_weight"]),
        "--aux-constructor-weight", str(FROZEN_MODEL_A["aux_constructor_weight"]),
        "--lambda-ctx-aux", str(FROZEN_MODEL_A["lambda_ctx_aux"]),
        "--lambda-pair", str(FROZEN_MODEL_A["lambda_pair"]),
        "--lambda-attr", "0.0",
        "--target-driver-share", str(FROZEN_MODEL_A["target_driver_share"]),
        "--target-constructor-share", str(FROZEN_MODEL_A["target_constructor_share"]),
        "--lambda-rw", str(FROZEN_MODEL_A["lambda_rw"]),
        "--lambda-shrink", str(FROZEN_MODEL_A["lambda_shrink"]),
        "--lambda-quali", str(FROZEN_MODEL_A["lambda_quali"]),
        "--orth-warmup-epochs", str(FROZEN_MODEL_A["orth_warmup_epochs"]),
        "--max-grad-norm", str(FROZEN_MODEL_A["max_grad_norm"]),
        "--use-additive-readout",
    ]
    if args.smoke_test:
        cmd.append("--smoke-test")
    _run(cmd)

    meta_path = Path(args.model_dir) / "orthogonal_shapley_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    assert float(meta["config"]["lambda_attr"]) == 0.0
    assert meta.get("ablation") == "no_attribution_balance"


def _export(args: argparse.Namespace) -> None:
    from baselines.orthogonal_shapley_skill import get_orthogonal_shapley_db
    from baselines.skill_loader import load_skill_export
    from skill.contract import InferenceMode
    import data.tasks as data_tasks

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
        gpu_id=args.gpu_id,
    )
    export.validate()
    print(f"exported {len(export.race)} race rows to {args.export_dir}")


def _validate(args: argparse.Namespace) -> None:
    ckpt = os.path.join(args.model_dir, "orthogonal_shapley.pth")
    meta = os.path.join(args.model_dir, "orthogonal_shapley_meta.json")
    baselines = os.path.join(args.model_dir, "coalition_baselines.json")
    cmd = [
        sys.executable,
        "src/experiments/run_validation_benchmark.py",
        "--sources", "bradley_terry", "plackett_luce", "bayesian_ssm", "orthogonal_shapley",
        "--output-dir", args.benchmark_dir,
        "--max-year", str(args.max_year),
        "--horizon", "inf",
        "--min-year", "2014",
        "--fixed-cohort",
        "--era-windows",
        "--checkpoint", ckpt,
        "--meta", meta,
        "--baselines", baselines,
        "--orthogonal-export-dir", args.export_dir,
        "--xai-seed", str(args.seed),
        "--gpu-id", str(args.gpu_id),
    ]
    _run(cmd)


def _undervalued(args: argparse.Namespace) -> None:
    ckpt = os.path.join(args.model_dir, "orthogonal_shapley.pth")
    meta = os.path.join(args.model_dir, "orthogonal_shapley_meta.json")
    baselines = os.path.join(args.model_dir, "coalition_baselines.json")
    cmd = [
        sys.executable,
        "src/experiments/run_undervalued_backtest.py",
        "--checkpoint", ckpt,
        "--meta", meta,
        "--baselines", baselines,
        "--max-year", "2026",
        "--min-backtest-year", "2000",
        "--horizon", "3",
        "--output-dir", args.undervalued_dir,
        "--gpu-id", str(args.gpu_id),
    ]
    _run(cmd)


def _extract_headline(source_block: dict[str, Any]) -> dict[str, Any]:
    career = source_block.get("career", {})
    locked = source_block.get("locked_test", {})
    survival = (
        source_block.get("survival", {})
        .get("eligible", {})
        .get("cox", {})
    )
    shares = source_block.get("shapley_season_mean", {})
    xai = source_block.get("xai", {})
    return {
        "pl_nll": locked.get("pl_nll"),
        "pairwise_acc": locked.get("pairwise_acc"),
        "partial_rho_continuous": career.get("partial_rho_continuous"),
        "partial_rho_continuous_ci_low": career.get("partial_rho_continuous_ci_low"),
        "partial_rho_continuous_ci_high": career.get("partial_rho_continuous_ci_high"),
        "hazard_ratio": survival.get("hazard_ratio"),
        "hazard_ratio_ci_low": survival.get("ci_low"),
        "hazard_ratio_ci_high": survival.get("ci_high"),
        "shapley_shares": shares,
        "constructor_leakage_rho": xai.get("constructor_leakage_rho"),
        "swap_invariance": xai.get("swap_invariance"),
        "inference_note": locked.get("note") or locked.get("inference_mode"),
    }


def _report(args: argparse.Namespace) -> None:
    ablation_bench_path = Path(args.benchmark_dir) / "benchmark.json"
    with open(ablation_bench_path) as f:
        ablation_bench = json.load(f)

    regularized = None
    if os.path.isfile(args.regularized_benchmark):
        with open(args.regularized_benchmark) as f:
            reg_bench = json.load(f)
        if "orthogonal_shapley" in reg_bench.get("sources", {}):
            regularized = _extract_headline(reg_bench["sources"]["orthogonal_shapley"])

    no_attr_meta_path = Path(args.model_dir) / "orthogonal_shapley_meta.json"
    with open(no_attr_meta_path) as f:
        no_attr_meta = json.load(f)

    sources = ablation_bench.get("sources", {})
    report = {
        "provenance": {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "ablation": "no_attribution_balance",
            "claim_level": "car_adjusted_performance",
            "repository": "https://github.com/AntonioNvs/ortogonal-representation",
            "note": (
                "Ablation drops only attribution_balance_loss (lambda_attr=0). "
                "Bayesian locked-test race metrics are smoothed/in-sample and not "
                "a fair held-out comparison."
            ),
        },
        "config": {
            "model_dir": args.model_dir,
            "seed": args.seed,
            "frozen_model_a_hparams": FROZEN_MODEL_A,
            "lambda_attr_ablation": 0.0,
            "lambda_attr_regularized": 0.1,
        },
        "no_attr_meta": {
            "ablation": no_attr_meta.get("ablation"),
            "lambda_attr": no_attr_meta.get("config", {}).get("lambda_attr"),
            "metrics": no_attr_meta.get("metrics"),
        },
        "regularized_orthogonal_shapley": regularized,
        "no_attr_orthogonal_shapley": _extract_headline(
            sources.get("orthogonal_shapley", {})
        ),
        "baselines": {
            name: _extract_headline(block)
            for name, block in sources.items()
            if name != "orthogonal_shapley"
        },
    }

    undervalued_path = Path(args.undervalued_dir) / "undervalued.json"
    if undervalued_path.is_file():
        with open(undervalued_path) as f:
            uv = json.load(f)
        report["no_attr_undervalued_backtest"] = uv.get("backtest", {}).get("summary")
        forecast = uv.get("watchlists", {}).get("forecast", {})
        report["no_attr_watchlist_lead"] = (
            forecast.get("watchlist", [{}])[0] if forecast.get("watchlist") else None
        )

    out = Path(args.report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="No-attribution-balance ablation pipeline")
    p.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=STAGES,
        help="pipeline stages to run",
    )
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    p.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    p.add_argument("--undervalued-dir", default=DEFAULT_UNDERVALUED_DIR)
    p.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    p.add_argument(
        "--regularized-benchmark",
        default=REGULARIZED_BENCHMARK,
        help="existing regularized Model A benchmark.json for side-by-side report",
    )
    p.add_argument("--epochs", type=int, default=FROZEN_MODEL_A["epochs"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=cfg.DEFAULT_GPU_ID)
    p.add_argument("--max-year", type=int, default=2025)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    stages = set(args.stages)
    if "all" in stages:
        stages = {"train", "export", "validate", "undervalued", "report"}

    if "train" in stages:
        _train(args)
    if "export" in stages:
        _export(args)
    if "validate" in stages:
        _validate(args)
    if "undervalued" in stages:
        _undervalued(args)
    if "report" in stages:
        _report(args)

    print("Attr ablation pipeline complete.")


if __name__ == "__main__":
    main()
