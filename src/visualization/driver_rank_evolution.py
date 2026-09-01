"""Multi-season driver rank evolution (Yamauchi-style four-panel row)."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.naming import resolve_driver_ref
from visualization.style import apply_plot_style, save_figure


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

    refs = {}
    if "driverRef" in race_df.columns:
        for r in race_df.drop_duplicates("driverId").itertuples():
            if getattr(r, "driverRef", None):
                refs[str(r.driverRef).lower()] = int(r.driverId)

    for ax, query in zip(axes, drivers):
        did, _ = resolve_driver_ref(query, refs)
        if did is None and "driver_name" in race_df.columns:
            m = race_df[race_df["driver_name"].str.lower().str.contains(query.lower(), na=False)]
            did = int(m["driverId"].iloc[0]) if not m.empty else None
        if did is None:
            ax.set_title(query)
            continue

        g = season_df[
            (season_df["driverId"] == did)
            & (season_df["season"] >= start_year)
            & (season_df["season"] <= end_year)
        ].sort_values("season")
        if g.empty:
            ax.set_title(query)
            continue

        name = race_df.loc[race_df["driverId"] == did, "driver_name"].dropna()
        title = name.iloc[0] if len(name) else query
        y = g["skill_0_10" if "skill_0_10" in g.columns else "skill_score"].astype(float)
        x = g["season"].astype(int)
        ax.plot(x, y, color="0.15", linewidth=1.5)
        ax.scatter(x, y, color="0.15", s=30, zorder=3)
        if show_sd and "skill_lo" in g.columns:
            ax.fill_between(
                x,
                g["skill_lo"].astype(float),
                g["skill_hi"].astype(float),
                alpha=0.25,
                color="0.5",
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
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Season")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Season-mean skill (0–10)")
    fig.suptitle(f"Driver skill evolution ({start_year}–{end_year})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path:
        save_figure(
            fig,
            output_path,
            metadata={"drivers": drivers, "start_year": start_year, "end_year": end_year},
        )
    return fig
