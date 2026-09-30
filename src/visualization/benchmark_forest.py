"""Forest plots of career-validation metrics across skill sources.

Post-hoc renderer over ``benchmark.json``: for a chosen metric (or list of
metrics) draw one horizontal dot + 95% bootstrap-CI per source, with the
metric's null value as a per-row reference line. ``orthogonal_shapley`` is
highlighted as the primary source.

This is the anchor figure for the fair-market test: it shows, with honest
uncertainty, whether the primary model beats Bradley--Terry (and Bayesian SSM)
on the same estimand.
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from visualization.style import apply_plot_style, despine_axes, finalize_axes, save_figure

# Metric spec: label, null value, and (value, ci_low, ci_high) paths within a
# benchmark report's ``sources.<src>.career`` dict.
_METRICS: dict[str, dict] = {
    "underrated_resolution": {
        "label": "Underrated resolution rate",
        "null": 0.5,
        "value": ("career", "underrated_resolution", "resolution_rate"),
        "ci_low": ("career", "underrated_resolution", "ci_low"),
        "ci_high": ("career", "underrated_resolution", "ci_high"),
    },
    "underrated_auroc": {
        "label": "Underrated promotion AUROC",
        "null": 0.5,
        "value": ("career", "underrated_promotion_auroc", "auroc"),
        "ci_low": ("career", "underrated_promotion_auroc", "ci_low"),
        "ci_high": ("career", "underrated_promotion_auroc", "ci_high"),
    },
    "underrated_partial_rho": {
        "label": "Underrated partial ρ",
        "null": 0.0,
        "value": ("career", "underrated_partial_rho"),
        "ci_low": ("career", "underrated_partial_ci_low"),
        "ci_high": ("career", "underrated_partial_ci_high"),
    },
    "partial_rho": {
        "label": "Partial ρ (skill | tier at T)",
        "null": 0.0,
        "value": ("career", "partial_rho"),
        "ci_low": ("career", "partial_ci_low"),
        "ci_high": ("career", "partial_ci_high"),
    },
    "partial_rho_continuous": {
        "label": "Partial ρ (skill | car score at T)",
        "null": 0.0,
        "value": ("career", "partial_rho_continuous"),
        "ci_low": ("career", "partial_rho_continuous_ci_low"),
        "ci_high": ("career", "partial_rho_continuous_ci_high"),
    },
    "survival_hr": {
        "label": "Cox HR (skill → time-to-promotion)",
        "null": 1.0,
        "value": ("survival", "eligible", "cox", "hazard_ratio"),
        "ci_low": ("survival", "eligible", "cox", "hr_lo"),
        "ci_high": ("survival", "eligible", "cox", "hr_hi"),
    },
}

# Source -> colour. Orthogonal Shapley is the primary; Bradley-Terry the
# reference baseline; Bayesian SSM the in-window comparator.
_SOURCE_COLORS: dict[str, str] = {
    "orthogonal_shapley": "#1f77b4",
    "bradley_terry": "#7f7f7f",
    "bayesian_ssm": "#ff7f0e",
    "skill_gnn": "#2ca02c",
}

_PRIMARY_KEY = "orthogonal_shapley"


def _path_get(report: dict, path: tuple, default: float = float("nan")) -> float:
    """Walk a nested dict path and coerce to float (NaN on any miss)."""
    cur = report
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if cur is None:
        return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _source_color(source: str) -> str:
    return _SOURCE_COLORS.get(source, "#9e9e9e")


def _resolve_metrics(metric: Optional[Iterable[str]]) -> list[str]:
    if metric is None:
        return [
            "underrated_resolution",
            "underrated_auroc",
            "underrated_partial_rho",
            "partial_rho_continuous",
        ]
    if isinstance(metric, str):
        return [metric]
    return list(metric)


def plot_benchmark_forest(
    reports: dict,
    *,
    metric: Optional[Iterable[str] | str] = None,
    primary_key: str = _PRIMARY_KEY,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Forest plot of one or more metrics across sources.

    ``reports`` maps a source name to its benchmark report dict (the values of
    ``benchmark.json["sources"]``). When ``metric`` is a list, one row per
    metric (small multiples) with each row's own null reference line.
    """
    apply_plot_style()
    metrics = _resolve_metrics(metric)
    sources = [s for s in reports if isinstance(reports.get(s), dict)]

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, mkey in zip(axes, metrics):
        spec = _METRICS[mkey]
        label, null = spec["label"], spec["null"]

        rows = []
        for source in sources:
            value = _path_get(reports[source], spec["value"])
            lo = _path_get(reports[source], spec["ci_low"])
            hi = _path_get(reports[source], spec["ci_high"])
            if np.isnan(value):
                continue
            rows.append((source, value, lo, hi))

        # Primary model on top for visual salience, then value-descending.
        rows.sort(key=lambda r: (r[0] != primary_key, -(r[1] if not np.isnan(r[1]) else -np.inf)))

        y = np.arange(len(rows))
        for yi, (source, value, lo, hi) in zip(y, rows):
            color = _source_color(source)
            marker = "D" if source == primary_key else "o"
            size = 9 if source == primary_key else 7
            lw = 2.4 if source == primary_key else 1.6
            if not (np.isnan(lo) and np.isnan(hi)):
                ax.errorbar(
                    value, yi,
                    xerr=[[value - lo], [hi - value]],
                    fmt="none", ecolor=color, elinewidth=lw, capsize=3, alpha=0.9,
                )
            ax.scatter(value, yi, color=color, marker=marker, s=size ** 2, zorder=3,
                       edgecolors="white", linewidths=0.6)

        ax.axvline(null, color="0.75", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0].replace("_", " ") for r in rows], fontsize=10)
        ax.set_title(label, loc="left", pad=6, color="dimgrey", fontsize=11)
        ax.set_xlim(_nice_xlim(rows, null, spec))
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        finalize_axes(ax)

    axes[-1].set_xlabel("Estimate ± 95% cluster-bootstrap CI", color="dimgrey", labelpad=8)
    fig.suptitle(
        "Fair-market validation: does skill predict promotion — and its timing — above the car?",
        fontsize=13, y=0.995, color="dimgrey",
    )
    fig.text(
        0.99, 0.005,
        "Dashed line = metric null · diamond = orthogonal Shapley · bars = bootstrap CI",
        ha="right", va="bottom", fontsize=8.5, color="dimgrey", style="italic",
    )

    despine_axes(top=True, right=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    if output_path:
        save_figure(fig, output_path, metadata={"metrics": metrics})
    return fig


def _nice_xlim(rows: list, null: float, spec: dict) -> tuple[float, float]:
    """Axis limits padded to the data, but always covering the null value."""
    finite = [r[1] for r in rows if not np.isnan(r[1])]
    los = [r[2] for r in rows if not np.isnan(r[2])]
    his = [r[3] for r in rows if not np.isnan(r[3])]
    if not finite:
        return (0.0, 1.0)
    vals = finite + los + his + [null]
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo) if hi > lo else 0.05
    return (lo - pad, hi + pad)
