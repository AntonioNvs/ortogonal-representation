"""Season Shapley attribution stacked bars."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skill.decomposition import bootstrap_shapley_ci
from visualization.style import apply_plot_style, save_figure


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

    ax.barh(y, driver, color="#2ecc71", label="Driver")
    ax.barh(y, constructor, left=driver, color="#3498db", label="Constructor")
    ax.barh(y, context, left=driver + constructor, color="#95a5a6", label="Context")
    ax.set_yticks(y)
    ax.set_yticklabels(shapley["driver_name"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Explained variance share")
    ax.set_title(f"Performance decomposition by driver — {season}")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    if "share_driver_lo" in shapley.columns:
        for i, row in enumerate(shapley.itertuples()):
            ax.errorbar(
                row.share_driver,
                i,
                xerr=[[row.share_driver - row.share_driver_lo], [row.share_driver_hi - row.share_driver]],
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=1,
            )

    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path, metadata={"season": season, "sort_by": sort_by})
    return fig
