import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import config as cfg
import train
from experiments.significance_stats import build_significance_summary


@contextmanager
def patched_temporal_split(train_years, val_years, test_years):
    old_train = list(cfg.TRAIN_YEARS)
    old_val = list(cfg.VAL_YEARS)
    old_test = list(cfg.TEST_YEARS)
    cfg.TRAIN_YEARS = list(train_years)
    cfg.VAL_YEARS = list(val_years)
    cfg.TEST_YEARS = list(test_years)
    try:
        yield
    finally:
        cfg.TRAIN_YEARS = old_train
        cfg.VAL_YEARS = old_val
        cfg.TEST_YEARS = old_test


def get_split_scenarios(design: str) -> List[Dict]:
    if design == "fixed_split_repeated_seeds":
        return [{
            "split_id": "default",
            "train_years": list(cfg.TRAIN_YEARS),
            "val_years": list(cfg.VAL_YEARS),
            "test_years": list(cfg.TEST_YEARS),
        }]
    if design == "rolling_temporal_splits_repeated_seeds":
        return [
            {
                "split_id": "window_2018_2020",
                "train_years": list(range(2000, 2016)),
                "val_years": [2016, 2017],
                "test_years": [2018, 2019, 2020],
            },
            {
                "split_id": "window_2020_2022",
                "train_years": list(range(2000, 2018)),
                "val_years": [2018, 2019],
                "test_years": [2020, 2021, 2022],
            },
            {
                "split_id": "window_2021_2023",
                "train_years": list(range(2000, 2019)),
                "val_years": [2019, 2020],
                "test_years": [2021, 2022, 2023],
            },
        ]
    raise ValueError(f"Unsupported design: {design}")


def flatten_run_row(row: Dict) -> Dict:
    meta = row.get("run_metadata", {})
    metrics = row.get("test_metrics", {})
    config_data = row.get("configuration", {})
    return {
        "model_name": row.get("model_name"),
        "model_level": row.get("model_level"),
        "run_id": meta.get("run_id"),
        "seed": meta.get("seed"),
        "split_id": meta.get("split_id"),
        "design": meta.get("design"),
        "lambda_orthogonal": config_data.get("lambda_orthogonal"),
        "aux_weight": config_data.get("aux_weight"),
        "epochs": config_data.get("epochs"),
        "auroc": metrics.get("auroc"),
        "loss": metrics.get("loss"),
        "bce": metrics.get("bce"),
        "orth": metrics.get("orth"),
        "cos_global": metrics.get("cos_global"),
        "model_path": row.get("model_path"),
    }


def write_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_report(summary: Dict, output_path: str):
    lines = []
    lines.append("# Ortho Model Statistical Comparison")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Metric: `{summary['metric']}`")
    lines.append("")
    lines.append("## AUROC Per Model")
    lines.append("")
    lines.append("| Model | Mean | Std | 95% CI | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for model, stats in sorted(summary["model_summary"].items()):
        lines.append(
            f"| {model} | {stats['mean']:.6f} | {stats['std']:.6f} | "
            f"[{stats['ci_low']:.6f}, {stats['ci_high']:.6f}] | {stats['n']} |"
        )
    lines.append("")
    lines.append("## Pairwise Significance")
    lines.append("")
    lines.append("| Pair | Delta Mean (A-B) | 95% CI | p-value | p-adjusted (Holm) | Reject H0 | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pair, content in sorted(summary["pairwise"].items()):
        if "error" in content:
            lines.append(f"| {pair} | - | - | - | - | - | 0 |")
            continue
        delta = content["delta"]
        test = content["test"]
        holm = content.get("holm", {})
        lines.append(
            f"| {pair} | {delta['delta_mean']:.6f} | [{delta['ci_low']:.6f}, {delta['ci_high']:.6f}] | "
            f"{test['p_value']:.6f} | {holm.get('p_adjusted', float('nan')):.6f} | "
            f"{holm.get('reject_null', False)} | {delta['n']} |"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_experiment(args):
    split_scenarios = get_split_scenarios(args.design)
    all_results = []
    model_levels = [x.strip() for x in args.model_grid.split(",") if x.strip()]

    for split in split_scenarios:
        split_id = split["split_id"]
        with patched_temporal_split(split["train_years"], split["val_years"], split["test_years"]):
            for run_idx in range(args.n_runs):
                seed = args.seed_start + run_idx
                print(
                    f"\n=== Design={args.design} | Split={split_id} | "
                    f"Run={run_idx + 1}/{args.n_runs} | Seed={seed} ==="
                )
                train.set_global_seed(seed, deterministic=args.deterministic)
                run_rows = train.train_models(
                    epochs=args.epochs,
                    run_ablation=False,
                    model_grid=args.model_grid,
                    output_file=args.output_file,
                    run_metadata={
                        "run_id": run_idx,
                        "seed": seed,
                        "design": args.design,
                        "split_id": split_id,
                    },
                    write_output=False,
                )
                all_results.extend(run_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nRaw run-level results written to {args.output_file}")

    flat_rows = [flatten_run_row(row) for row in all_results]
    csv_path = os.path.join(args.output_dir, "run_level_metrics.csv")
    write_csv(flat_rows, csv_path)
    print(f"Run-level CSV written to {csv_path}")

    summary = build_significance_summary(
        run_rows=all_results,
        model_levels=model_levels,
        metric="auroc",
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        ci=0.95,
        seed=args.seed_start,
    )

    summary_json_path = os.path.join(args.output_dir, "significance_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary JSON written to {summary_json_path}")

    summary_md_path = os.path.join(args.output_dir, "significance_summary.md")
    render_markdown_report(summary, summary_md_path)
    print(f"Summary Markdown written to {summary_md_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated experiments and significance analysis for ortho levels."
    )
    parser.add_argument(
        "--design",
        type=str,
        default="fixed_split_repeated_seeds",
        choices=["fixed_split_repeated_seeds", "rolling_temporal_splits_repeated_seeds"],
        help="Repeated-run design to execute.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per model run.")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of seeds per split scenario.")
    parser.add_argument("--seed_start", type=int, default=42, help="Initial seed for repeated runs.")
    parser.add_argument(
        "--model_grid",
        type=str,
        default="zero,low,high",
        help="Comma-separated model levels. Valid: zero, low, high",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/significance",
        help="Directory to write report artifacts.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/significance/run_results.json",
        help="Path to write raw run-level JSON results.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=5000,
        help="Bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=20000,
        help="Random sign-flip permutations for paired p-values.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic torch backend flags when setting seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
