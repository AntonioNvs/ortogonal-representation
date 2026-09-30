"""Season Shapley attribution stacked bars."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skill.decomposition import bootstrap_shapley_ci
from visualization.style import apply_plot_style, despine_axes, finalize_axes, save_figure

# Sequential cubehelix for the ordered decomposition: driver (primary) → constructor → context (residual).
_CUBEHELIX = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
DRIVER_COLOR = _CUBEHELIX[5]
CONSTRUCTOR_COLOR = _CUBEHELIX[3]
CONTEXT_COLOR = _CUBEHELIX[0]


def plot_entity_attribution(
    race_df: pd.DataFrame,
    *,
    season: int,
    output_path: Optional[str] = None,
    sort_by: str = "driver",
) -> plt.Figure:
    apply_plot_style()
    sub = race_df[race_df["season"] == season].copy()
    shapley = bootstrap_shapley_ci(sub, seed=42)
    if shapley.empty:
        raise ValueError(f"No Shapley data for season {season}")

    names = sub.groupby("driverId")["driver_name"].first()
    shapley["driver_name"] = shapley["driverId"].map(names).fillna(shapley["driverId"].astype(str))
    if sort_by == "driver":
        shapley = shapley.sort_values("share_driver", ascending=False)
    else:
        shapley = shapley.sort_values("driver_name")

    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(shapley))))
    y = np.arange(len(shapley))
    driver = shapley["share_driver"].to_numpy()
    constructor = shapley["share_constructor"].to_numpy()
    context = shapley["share_context"].to_numpy()

    ax.barh(y, driver, color=DRIVER_COLOR, label="Driver")
    ax.barh(y, constructor, left=driver, color=CONSTRUCTOR_COLOR, label="Constructor")
    ax.barh(y, context, left=driver + constructor, color=CONTEXT_COLOR, label="Context")
    ax.set_yticks(y)
    ax.set_yticklabels(shapley["driver_name"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Explained variance share", color="dimgrey", labelpad=8)
    ax.set_title(
        f"Performance decomposition by driver — {season}", loc="left", pad=7, color="dimgrey"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)

    if "share_driver_lo" in shapley.columns:
        for i, row in enumerate(shapley.itertuples()):
            ax.errorbar(
                row.share_driver,
                i,
                xerr=[[row.share_driver - row.share_driver_lo], [row.share_driver_hi - row.share_driver]],
                fmt="none",
                ecolor="0.3",
                capsize=2,
                linewidth=0.8,
            )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="lightgrey",
        labelcolor="dimgrey",
    )

    top = shapley.iloc[0]
    fig.text(
        0.01,
        0.01,
        f'{top["driver_name"]} leads: {top["share_driver"] * 100:.0f}% of variance attributed to the driver',
        ha="left",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )

    finalize_axes(ax)
    despine_axes()
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path, metadata={"season": season, "sort_by": sort_by})
    return fig
