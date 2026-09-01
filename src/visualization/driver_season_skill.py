"""Within-season driver skill trajectory on [0, 10]."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.naming import resolve_driver_ref
from visualization.style import apply_plot_style, save_figure


def plot_driver_season_skill(
    race_df: pd.DataFrame,
    *,
    season: int,
    drivers: List[str],
    output_path: Optional[str] = None,
) -> plt.Figure:
    apply_plot_style()
    sub = race_df[race_df["season"] == season].copy()
    refs = {
        str(r.driverRef if hasattr(r, "driverRef") else ""): int(r.driverId)
        for r in sub.drop_duplicates("driverId").itertuples()
        if hasattr(r, "driverRef")
    }
    if not refs:
        refs = {str(d): int(d) for d in sub["driverId"].unique()}

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(drivers), 1)))

    for i, query in enumerate(drivers):
        did, _ = resolve_driver_ref(query, refs)
        if did is None:
            # try driverId directly from export names
            name_match = sub[sub["driver_name"].str.lower().str.contains(query.lower(), na=False)]
            if name_match.empty:
                continue
            did = int(name_match["driverId"].iloc[0])
        g = sub[sub["driverId"] == did].sort_values("round")
        if g.empty:
            continue
        label = g["driver_name"].iloc[0] if "driver_name" in g.columns else str(did)
        team = g["constructor_name"].iloc[-1] if "constructor_name" in g.columns else ""
        ax.plot(
            g["round"],
            g["skill_0_10"],
            marker="o",
            linewidth=2.2,
            color=colors[i],
            label=f"{label} ({team})" if team else label,
        )
        if g["skill_lo"].notna().any() and g["skill_hi"].notna().any():
            ax.fill_between(
                g["round"],
                g["skill_lo"],
                g["skill_hi"],
                alpha=0.2,
                color=colors[i],
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Driver skill (0–10)")
    ax.set_ylim(0, 10)
    ax.set_title(f"Driver skill trajectory — {season} season")
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path, metadata={"season": season, "drivers": drivers})
    return fig
