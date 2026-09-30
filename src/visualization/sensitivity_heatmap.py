"""Sensitivity-grid divergence heatmap.

Post-hoc renderer over ``sensitivity_grid.json``: for one metric, show
``primary − baseline`` across the ``skill_pct_threshold × p_S`` grid as a
diverging heatmap (red = primary leads, blue = baseline leads). The claim this
figure makes is "the Orth > BT ordering is stable across the threshold grid,
and here is where it is strongest / weakest."
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from visualization.style import apply_plot_style, despine_axes, finalize_axes, save_figure

_METRIC_LABELS = {
    "resolution": "Resolution rate",
    "auroc": "Underrated AUROC",
    "partial_rho": "Underrated partial ρ",
}


def _pivot_diff(
    grid: pd.DataFrame,
    *,
    metric: str,
    source: str,
    baseline: str,
) -> pd.DataFrame:
    """Return a threshold × p_S pivot of ``source − baseline`` for ``metric``."""
    src_col = f"{source}_{metric}"
    base_col = f"{baseline}_{metric}"
    df = grid.copy()
    for col in (src_col, base_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in grid; have {list(df.columns)}")
    df["diff"] = pd.to_numeric(df[src_col], errors="coerce") - pd.to_numeric(
        df[base_col], errors="coerce"
    )
    return df.pivot(index="skill_pct_threshold", columns="p_S", values="diff")


def plot_sensitivity_diff(
    grid: pd.DataFrame,
    *,
    metric: str = "partial_rho",
    source: str = "orthogonal_shapley",
    baseline: str = "bradley_terry",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Diverging heatmap of ``source − baseline`` over the sensitivity grid."""
    apply_plot_style()
    pivot = _pivot_diff(grid, metric=metric, source=source, baseline=baseline)

    # Center the colormap at 0; symmetric extent keeps the zero line neutral.
    vmax = max(1e-6, float(np.nanmax(np.abs(pivot.to_numpy()))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(False)

    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(val) > 0.6 * vmax else "0.2")

    ax.set_xticks(range(cols))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(rows))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index], fontsize=9)
    ax.set_xlabel("p_S (S-tier cut)", color="dimgrey", labelpad=8)
    ax.set_ylabel("skill percentile threshold", color="dimgrey", labelpad=8)
    label = _METRIC_LABELS.get(metric, metric)
    ax.set_title(
        f"{source.replace('_', ' ').title()} − {baseline.replace('_', ' ').title()}: {label}",
        loc="left", pad=7, color="dimgrey", fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(length=0, labelcolor="dimgrey")
    cbar.outline.set_visible(False)
    cbar.set_label("Δ vs baseline", color="dimgrey", labelpad=8)

    finalize_axes(ax)
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path, metadata={"metric": metric, "source": source, "baseline": baseline})
    return fig
