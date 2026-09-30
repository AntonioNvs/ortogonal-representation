"""Multi-season driver rank evolution (four-panel row)."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.naming import resolve_driver_ref
from visualization.style import apply_plot_style, despine_axes, finalize_axes, save_figure


def plot_driver_rank_evolution(
    season_df: pd.DataFrame,
    race_df: pd.DataFrame,
    *,
    drivers: List[str],
    start_year: int,
    end_year: int,
    show_sd: bool = False,
    output_path: Optional[str] = None,
) -> plt.Figure:
    apply_plot_style()
    n = len(drivers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    line_color = sns.cubehelix_palette(6, rot=-0.25, light=0.7)[5]

    refs = {}
    if "driverRef" in race_df.columns:
        for r in race_df.drop_duplicates("driverId").itertuples():
            if getattr(r, "driverRef", None):
                refs[str(r.driverRef).lower()] = int(r.driverId)

    changes = []  # (title, first_skill, last_skill) for the figure-level insight
    for ax, query in zip(axes, drivers):
        did, _ = resolve_driver_ref(query, refs)
        if did is None and "driver_name" in race_df.columns:
            m = race_df[race_df["driver_name"].str.lower().str.contains(query.lower(), na=False)]
            did = int(m["driverId"].iloc[0]) if not m.empty else None
        if did is None:
            ax.set_title(query, loc="left", pad=7, color="dimgrey")
            finalize_axes(ax)
            continue

        g = season_df[
            (season_df["driverId"] == did)
            & (season_df["season"] >= start_year)
            & (season_df["season"] <= end_year)
        ].sort_values("season")
        if g.empty:
            ax.set_title(query, loc="left", pad=7, color="dimgrey")
            finalize_axes(ax)
            continue

        name = race_df.loc[race_df["driverId"] == did, "driver_name"].dropna()
        title = name.iloc[0] if len(name) else query
        y = g["skill_0_10" if "skill_0_10" in g.columns else "skill_score"].astype(float)
        x = g["season"].astype(int)

        ax.plot(x, y, color=line_color, linewidth=1.8, zorder=3)
        ax.scatter(x, y, color=line_color, s=32, zorder=4, edgecolors="white", linewidths=0.8)
        if show_sd and "skill_lo" in g.columns:
            ax.fill_between(
                x,
                g["skill_lo"].astype(float),
                g["skill_hi"].astype(float),
                alpha=0.25,
                color=line_color,
            )
        elif "skill_lo" in g.columns:
            ax.errorbar(
                x,
                y,
                yerr=[y - g["skill_lo"], g["skill_hi"] - y],
                fmt="none",
                ecolor="0.4",
                capsize=3,
            )

        ax.set_title(title, loc="left", pad=7, color="dimgrey", fontsize=11)
        ax.set_xlabel("Season", color="dimgrey", labelpad=6)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)
        finalize_axes(ax)

        if len(g):
            changes.append((title, float(y.iloc[0]), float(y.iloc[-1])))

    axes[0].set_ylabel("Season-mean skill (0–10)", color="dimgrey", labelpad=8)
    fig.suptitle(f"Driver skill evolution ({start_year}–{end_year})", fontsize=14, y=0.98, color="dimgrey")

    if changes:
        top_title, top_delta = max(
            ((t, last - first) for t, first, last in changes), key=lambda c: abs(c[1])
        )
        fig.text(
            0.99,
            0.01,
            f"{top_title} shows the largest change ({top_delta:+.2f})",
            ha="right",
            va="bottom",
            fontsize=9,
            color="dimgrey",
            style="italic",
        )

    despine_axes()
    fig.tight_layout()
    if output_path:
        save_figure(
            fig,
            output_path,
            metadata={"drivers": drivers, "start_year": start_year, "end_year": end_year},
        )
    return fig
