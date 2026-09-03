"""Constructor-recoverability null distribution.

Post-hoc renderer over ``benchmark.json``: the supervised leakage probe
(``constructor_recoverability``) fits a linear probe that predicts constructor
identity from the driver-state embedding. This figure plots the permuted-label
null AUC distribution, the held-out macro-AUC as a vertical line, and shades
the 95th percentile. "No leakage" is the honest falsification claim: the held-out
AUC sits inside (not above) the null envelope.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from visualization.style import apply_plot_style, despine_axes, finalize_axes, save_figure


def plot_recoverability(
    recoverability: dict,
    *,
    title: str = "Does the driver channel leak the constructor?",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Null-AUC histogram with the held-out macro-AUC and p95 threshold."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    null = [v for v in recoverability.get("null_aucs", []) if not np.isnan(v)]
    macro_auc = recoverability.get("macro_auc", float("nan"))
    p95 = recoverability.get("null_auc_p95", float("nan"))

    if not null:
        ax.text(0.5, 0.5, "no null distribution recorded", ha="center", va="center",
                color="dimgrey", transform=ax.transAxes)
        finalize_axes(ax)
        despine_axes()
        if output_path:
            save_figure(fig, output_path, metadata={"title": title})
        return fig

    ax.hist(null, bins=min(20, max(8, int(np.sqrt(len(null))))), color="#9e9e9e",
            alpha=0.55, edgecolor="white", density=False, label="Null (permuted labels)")

    if not np.isnan(p95):
        ax.axvline(p95, color="#d62728", linewidth=1.4, linestyle="--",
                   label=f"Null 95th pct = {p95:.3f}")
        ax.axvspan(p95, ax.get_xlim()[1], color="#d62728", alpha=0.06)

    if not np.isnan(macro_auc):
        ax.axvline(macro_auc, color="#1f77b4", linewidth=2.0,
                   label=f"Held-out macro-AUC = {macro_auc:.3f}")

    ax.set_xlabel("Macro-AUC (constructor recoverability)", color="dimgrey", labelpad=8)
    ax.set_ylabel("Permutations", color="dimgrey", labelpad=8)
    ax.set_title(title, loc="left", pad=7, color="dimgrey")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.8,
              edgecolor="lightgrey", labelcolor="dimgrey", fontsize=9)

    leakage = recoverability.get("leakage", None)
    verdict = "leakage detected" if leakage else ("no leakage" if leakage is not None else "")
    if verdict:
        color = "#d62728" if leakage else "#2ca02c"
        ax.text(0.99, 0.97, verdict, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color=color)

    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    finalize_axes(ax)
    despine_axes(top=True, right=True)
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path, metadata={
            "macro_auc": macro_auc, "null_auc_p95": p95, "leakage": leakage,
        })
    return fig
