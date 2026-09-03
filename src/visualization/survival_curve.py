"""Kaplan-Meier time-to-promotion curves stratified by skill tertile.

Post-hoc renderer over ``benchmark.json``: the ``sources.<src>.survival`` block
carries a pooled ``km`` curve plus per-tertile ``km_tertiles`` curves and a Cox
hazard ratio with cluster-bootstrap CI. This figure shows the fair-market test
in time: high-skill drivers reach a better team sooner.

One figure per source; the runner lays ``orthogonal_shapley`` and
``bradley_terry`` side by side.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from visualization.style import apply_plot_style, despine_axes, finalize_axes, save_figure

_TERTILE_COLORS = {"top": "#1f77b4", "mid": "#9e9e9e", "bottom": "#d62728"}
_TERTILE_LABELS = {"top": "Top skill tertile", "mid": "Middle tertile", "bottom": "Bottom tertile"}


def plot_km_tertiles(
    survival_block: dict,
    *,
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Step KM curves by skill tertile from one ``survival`` block.

    ``survival_block`` is ``benchmark.json["sources"][src]["survival"]["eligible"]``
    (or ``["underrated"]``): expects ``km_tertiles``, ``cox``, and
    ``logrank_top_vs_bottom_tertile``.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    tertiles = survival_block.get("km_tertiles", {})
    if not tertiles or "note" in tertiles:
        ax.text(0.5, 0.5, "insufficient skill spread for tertiles", ha="center",
                va="center", color="dimgrey", transform=ax.transAxes)
        finalize_axes(ax)
        despine_axes()
        if output_path:
            save_figure(fig, output_path, metadata={"title": title})
        return fig

    # Draw tertiles in a stable bottom → top order so the top curve is on top.
    for label in ("bottom", "mid", "top"):
        if label not in tertiles:
            continue
        strat = tertiles[label]
        km = strat.get("km", {})
        times = np.asarray(km.get("times", []), dtype=float)
        surv = np.asarray(km.get("survival", []), dtype=float)
        if times.size == 0:
            continue
        # Step function: survival starts at 1 at t=0 and drops at each event time.
        ax.step(
            np.concatenate([[0.0], times]),
            np.concatenate([[1.0], surv]),
            where="post",
            color=_TERTILE_COLORS[label],
            linewidth=2.2 if label == "top" else 1.6,
            label=f"{_TERTILE_LABELS[label]} (n={strat.get('n', '?')})",
        )

    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Seasons to first promotion", color="dimgrey", labelpad=8)
    ax.set_ylabel("P(not yet promoted)", color="dimgrey", labelpad=8)
    ax.set_title(title or "Time-to-promotion by skill tertile", loc="left", pad=7, color="dimgrey")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.8,
              edgecolor="lightgrey", labelcolor="dimgrey", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    # Annotate the Cox HR and log-rank p so the figure carries its own inference.
    cox = survival_block.get("cox", {})
    hr = cox.get("hazard_ratio", float("nan"))
    hr_lo = cox.get("hr_lo", float("nan"))
    hr_hi = cox.get("hr_hi", float("nan"))
    lr = survival_block.get("logrank_top_vs_bottom_tertile", {})
    lr_p = lr.get("p_value", float("nan"))

    notes = []
    if not np.isnan(hr):
        hr_txt = f"HR = {hr:.2f}"
        if not (np.isnan(hr_lo) and np.isnan(hr_hi)):
            hr_txt += f" [{hr_lo:.2f}, {hr_hi:.2f}]"
        notes.append(hr_txt)
    if not np.isnan(lr_p):
        notes.append(f"log-rank p = {lr_p:.3g}")
    if notes:
        ax.text(0.99, 0.03, "  ".join(notes), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9, color="dimgrey", style="italic")

    finalize_axes(ax)
    despine_axes(top=True, right=True)
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path, metadata={"title": title, "hr": hr, "logrank_p": lr_p})
    return fig
