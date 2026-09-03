#!/usr/bin/env python3
"""Render the validation figure set for the MIT Sloan abstract.

Pure post-hoc renderer: it reads the benchmark and sensitivity-grid JSON
artifacts (already produced on the A100 box) and emits the four figures that
carry the paper's validation story. It does **not** load the DB or re-run any
model, so the figures are bit-for-bit reproducible from the JSON.

Figures emitted to ``output/plots/validation/``:

    fair_market_forest   — anchor: Orth vs BT (vs Bayesian) on the underrated
                           trio + continuous car control, with bootstrap CIs.
    survival_km_<source> — KM time-to-promotion by skill tertile (anchor pair).
    sensitivity_diff_<m> — Orth − BT divergence heatmap across the threshold grid.
    recoverability_probe — constructor recoverability null vs held-out AUC.

Run:
    python src/experiments/plots/plot_validation_figures.py \
        --benchmark-json output/validation_benchmark/benchmark.json \
        --sensitivity-json output/sensitivity_grid/sensitivity_grid.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from visualization.benchmark_forest import plot_benchmark_forest
from visualization.recoverability_probe import plot_recoverability
from visualization.sensitivity_heatmap import plot_sensitivity_diff
from visualization.survival_curve import plot_km_tertiles

_PRIMARY = "orthogonal_shapley"
_BASELINE = "bradley_terry"
_SURVIVAL_SOURCES = ("orthogonal_shapley", "bradley_terry")
_SENSITIVITY_METRICS = ("resolution", "auroc", "partial_rho")


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)  # json.load parses NaN/Infinity emitted by json.dump(default=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render validation figures from benchmark JSON")
    parser.add_argument("--benchmark-json", type=str,
                        default="output/validation_benchmark/benchmark.json")
    parser.add_argument("--sensitivity-json", type=str,
                        default="output/sensitivity_grid/sensitivity_grid.json")
    parser.add_argument("--output-dir", type=str, default="output/plots/validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    benchmark = _load_json(args.benchmark_json)
    sources = benchmark.get("sources", {})

    # 1. Anchor forest: the fair-market metrics with honest CIs.
    plot_benchmark_forest(
        sources,
        metric=["underrated_resolution", "underrated_auroc",
                "underrated_partial_rho", "partial_rho_continuous"],
        primary_key=_PRIMARY,
        output_path=os.path.join(args.output_dir, "fair_market_forest"),
    )

    # 2. Anchor companion: KM curves by skill tertile, per survival source.
    for source in _SURVIVAL_SOURCES:
        rep = sources.get(source, {})
        survival = rep.get("survival", {})
        eligible = survival.get("eligible")
        if not isinstance(eligible, dict):
            continue
        plot_km_tertiles(
            eligible,
            title=f"{source.replace('_', ' ').title()} — time-to-promotion by skill tertile",
            output_path=os.path.join(args.output_dir, f"survival_km_{source}"),
        )

    # 3. Robustness: sensitivity-grid divergence heatmaps.
    if os.path.exists(args.sensitivity_json):
        grid = _load_json(args.sensitivity_json)
        import pandas as pd

        grid_df = pd.DataFrame(grid.get("grid", []))
        metrics = _SENSITIVITY_METRICS
        if grid.get("fixed_cohort"):
            # Resolution rate is shared across sources on a fixed cohort — its
            # diff is identically zero, so it is not a discriminator to plot.
            metrics = tuple(m for m in metrics if m != "resolution")
        for metric in metrics:
            try:
                plot_sensitivity_diff(
                    grid_df, metric=metric, source=_PRIMARY, baseline=_BASELINE,
                    output_path=os.path.join(args.output_dir, f"sensitivity_diff_{metric}"),
                )
            except KeyError as exc:
                print(f"  skip sensitivity {metric}: {exc}")

    # 4. Honesty: constructor recoverability null vs held-out AUC. Prefer the
    #    career (team-switcher) probe when present — it is the clean test for the
    #    hard-identification variant.
    xai = sources.get(_PRIMARY, {}).get("xai", {})
    rec = xai.get("constructor_recoverability_career") or xai.get("constructor_recoverability")
    if isinstance(rec, dict):
        plot_recoverability(
            rec,
            title="Does the driver career channel leak the constructor?",
            output_path=os.path.join(args.output_dir, "recoverability_probe"),
        )

    print(f"wrote validation figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
