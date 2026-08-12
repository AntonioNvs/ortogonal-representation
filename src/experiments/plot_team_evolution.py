"""Plot constructor championship evolution over seasons (one line per team).

Reads the raw enriched DB, computes season-end constructor standings via
``validation.team_tiers.compute_constructor_season_points``, and draws one line
per constructor showing how their standing evolves.

Y-axis (``position``) is inverted so 1st sits at the top. Teams absent from a
season leave a gap in their line (they simply did not compete).

Usage:
    python -m src.experiments.plot_team_evolution --min-year 2010 --max-year 2020
    python -m src.experiments.plot_team_evolution --min-year 2000 --max-year 2026 --metric score
    python -m src.experiments.plot_team_evolution --min-year 2010 --max-year 2020 --top-k 8
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _p in (ROOT_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.enriched_dataset import EnrichedF1Dataset
from validation.team_tiers import compute_constructor_season_points

# Distinct, colorblind-friendly palette (ordered roughly by "classic" prominence
# so the legend reads naturally; teams beyond this get a shared gray).
TEAM_PALETTE = {
    "Ferrari": "#e8002d",
    "Mercedes": "#00d2be",
    "Red Bull": "#1e1e6e",
    "McLaren": "#ff8700",
    "Williams": "#005aff",
    "Renault": "#fff500",
    "Alpine": "#0090ff",
    "Aston Martin": "#006f62",
    "Racing Point": "#f596c8",
    "Force India": "#f596c8",
    "Sauber": "#9b0000",
    "Alfa Romeo": "#900000",
    "AlphaTauri": "#2b4562",
    "Toro Rosso": "#469bff",
    "Haas": "#b6babd",
    "Lotus F1": "#ffb800",
    "Lotus": "#ffb800",
    "Caterham": "#006f62",
    "Marussia": "#004456",
    "Manor": "#004456",
    "Virgin": "#004456",
    "HRT": "#b2995e",
    "Brawn": "#d4f0a0",
    "Toyota": "#e60000",
    "BMW Sauber": "#a1caf1",
    "Honda": "#f80000",
    "BAR": "#e60000",
    "Jordan": "#f9e79f",
    "Super Aguri": "#e60000",
    "Spyker": "#ff9900",
    "Midland": "#ff9900",
}
GRAY = "#c8c8c8"


def _team_color(name: str) -> str:
    return TEAM_PALETTE.get(name, GRAY)


def build_evolution_df(db, min_year: int, max_year: int) -> pd.DataFrame:
    """Season-end position/share per constructor, with friendly names."""
    points = compute_constructor_season_points(db)
    names = db.table_dict["constructors"].df[["constructorId", "name"]]
    df = points.merge(names, on="constructorId", how="left")
    df = df[(df["season"] >= min_year) & (df["season"] <= max_year)]
    return df


def plot_team_evolution(
    min_year: int = 2010,
    max_year: int = 2020,
    metric: str = "position",
    top_k: int | None = None,
    output: str = "output/team_evolution.png",
    figsize: tuple = (12, 6),
):
    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)
    df = build_evolution_df(db, min_year, max_year)

    seasons = sorted(df["season"].unique())

    # Which teams to draw? Default: every team present in the window; if
    # --top-k is given, keep the k teams with the best average `metric`.
    teams = sorted(df["name"].astype(str).unique())
    if top_k is not None:
        rank = df.groupby("name")[metric].agg(["mean", "count"]).sort_values("mean")
        teams = list(rank.head(top_k).index)
    else:
        # Order by best average `metric` so the legend reads from strongest down.
        rank = df.groupby("name")[metric].agg(["mean", "count"]).sort_values("mean")
        teams = list(rank.index)

    fig, ax = plt.subplots(figsize=figsize)

    for team in teams:
        tdf = df[df["name"] == team].sort_values("season")
        series = tdf.set_index("season")[metric].reindex(seasons)
        ax.plot(
            seasons,
            series.values,
            marker="o",
            markersize=4,
            linewidth=2,
            color=_team_color(team),
            label=team,
        )

    ax.set_xlabel("Season")
    ax.set_xticks(seasons)
    ax.set_xticklabels([str(s) for s in seasons], rotation=45)

    if metric == "position":
        # Lower (better) rank on top.
        ax.invert_yaxis()
        ax.set_ylabel("Constructor championship position (1 = best)")
        ax.set_yticks(np.arange(1, max(2, int(df["position"].max()) + 1)))
    elif metric == "score":
        ax.set_ylabel("Smoothed points share (score)")
    elif metric == "share":
        ax.set_ylabel("Season points share")
    else:
        ax.set_ylabel(metric)

    ax.set_title(f"Constructor championship evolution ({min_year}–{max_year})")
    ax.grid(True, alpha=0.3)

    # Legend below the axes, multiple columns, ordered strongest-first.
    ncol = max(1, len(teams) // 6 + 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol, fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output} ({len(teams)} teams)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot constructor championship evolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-year", type=int, default=2020)
    parser.add_argument(
        "--metric",
        type=str,
        default="position",
        choices=["position", "share", "score"],
        help="Y metric: championship position (inverted), raw share, or smoothed score.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Only plot the k best teams (by avg metric).")
    parser.add_argument("--output", type=str, default="output/team_evolution.png")
    args = parser.parse_args()

    plot_team_evolution(
        min_year=args.min_year,
        max_year=args.max_year,
        metric=args.metric,
        top_k=args.top_k,
        output=args.output,
    )


if __name__ == "__main__":
    main()
