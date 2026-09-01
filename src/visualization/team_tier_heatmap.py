"""Team tier heatmap (lineage-aware)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from validation.team_lineage import build_lineage_map, lineage_row_label
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers
from visualization.style import apply_plot_style, save_figure

TIER_MAP = {"S": 3, "A": 2, "B": 1}
TIER_CMAP = ListedColormap(["#d73027", "#fee08b", "#1a9850"])
TIER_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], TIER_CMAP.N)
MISSING_COLOR = "#ececec"


def _prepare_tier_matrix(
    team_tier: pd.DataFrame,
    lineage_map: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    sub = team_tier[(team_tier["season"] >= start_year) & (team_tier["season"] <= end_year)].copy()
    lm = lineage_map.set_index("constructorId")
    sub["lineage_id"] = sub["constructorId"].map(lm["lineage_id"])
    sub["display_name"] = sub["constructorId"].map(lm["name"])

    pivot = sub.pivot_table(index="lineage_id", columns="season", values="tier", aggfunc="first")
    numeric = pivot.map(lambda x: TIER_MAP.get(x, float("nan")) if pd.notna(x) else float("nan"))
    row_means = numeric.mean(axis=1, skipna=True)
    order = row_means.sort_values(ascending=False).index
    pivot = pivot.loc[order]
    numeric = numeric.loc[order]

    labels: dict[str, str] = {}
    for lineage_id in order:
        chunk = sub[sub["lineage_id"] == lineage_id][["season", "display_name"]]
        labels[str(lineage_id)] = lineage_row_label(chunk)

    return pivot, numeric, labels


def plot_team_tier_heatmap(
    team_tier: pd.DataFrame,
    lineage_map: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    output_path: Optional[str] = None,
) -> plt.Figure:
    apply_plot_style()
    pivot, numeric, labels = _prepare_tier_matrix(
        team_tier, lineage_map, start_year=start_year, end_year=end_year
    )

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(pivot.columns)), max(6, 0.38 * len(pivot))),
        facecolor="white",
    )
    ax.set_facecolor("white")
    ax.grid(False)

    data = np.ma.masked_invalid(numeric.to_numpy(dtype=float))
    cmap = TIER_CMAP.copy()
    cmap.set_bad(color=MISSING_COLOR)
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=TIER_NORM,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([labels[str(lid)] for lid in pivot.index])
    ax.set_xlabel("Season")
    ax.set_ylabel("Constructor lineage")
    ax.set_title(f"Constructor tier heatmap ({start_year}–{end_year})")
    ax.tick_params(axis="both", length=0)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = "white" if val == "S" else "black"
                ax.text(j, i, val, ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 3], fraction=0.03, pad=0.02)
    cbar.ax.set_yticklabels(["B", "A", "S"])
    cbar.outline.set_visible(False)
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path, metadata={"start_year": start_year, "end_year": end_year})
    return fig


def build_tier_heatmap_from_db(db, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from validation.team_lineage import lineage_id_by_constructor

    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    points = compute_constructor_season_points(db)
    tiers = compute_team_tiers(points, window=3, lineage=lineage)
    lm = build_lineage_map(db.table_dict["constructors"].df)
    return tiers, lm
