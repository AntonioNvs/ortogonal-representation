"""Plot constructor championship evolution over seasons (one line per team).

Reads the raw enriched DB, computes season-end constructor standings via
``validation.team_tiers.compute_constructor_season_points``, and draws one line
per constructor *lineage* (acquisitions/renames are merged into a single
continuous line labelled ``X/Y`` — see ``validation.team_lineage``).

Y-axis (``position``) is inverted so 1st sits at the top. Teams absent from a
season leave a gap in their line (they simply did not compete). The ``score``
and ``share`` metrics use a lineage-aware trailing average, so a rebranded team
does not reset its rank at the boundary.

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

import config as cfg
from data.enriched_dataset import EnrichedF1Dataset
from validation.team_lineage import build_lineage_map, lineage_id_by_constructor, lineage_label
from validation.team_tiers import _add_score, compute_constructor_season_points

# Distinct, colorblind-friendly palette keyed by constructor *name* (or the
# current name of a lineage). Teams beyond this get a shared gray.
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
    "Audi": "#a44a3f",
    "AlphaTauri": "#2b4562",
    "Toro Rosso": "#469bff",
    "RB": "#37424f",
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
    "Jaguar": "#005f2f",
}
GRAY = "#c8c8c8"


def _lineage_color(lineage_id: str, label: str) -> str:
    """Pick a stable color for a lineage: try its current name, then older names."""
    for cand in (label.split("/")[-1], lineage_id):
        if cand in TEAM_PALETTE:
            return TEAM_PALETTE[cand]
    for seg in reversed(label.split("/")):
        if seg in TEAM_PALETTE:
            return TEAM_PALETTE[seg]
    return GRAY


def build_evolution_df(db, min_year: int, max_year: int, window: int) -> pd.DataFrame:
    """Season-end position/share/score per constructor, lineage-annotated."""
    points = compute_constructor_season_points(db)
    constructors = db.table_dict["constructors"].df
    lineage = build_lineage_map(constructors)
    points = points.merge(
        lineage[["constructorId", "name", "lineage_id"]], on="constructorId", how="left"
    )
    points = points[(points["season"] >= min_year) & (points["season"] <= max_year)]

    # Lineage-aware smoothed score (rebrand carries the rank across).
    lid_map = lineage_id_by_constructor(constructors)
    points["score"] = _add_score(points, window, lineage=lid_map)["score"]
    return points


def _order_lineages(df: pd.DataFrame, metric: str) -> list[str]:
    """Lineage ids ordered strongest-first for the chosen metric."""
    ascending = metric == "position"  # lower position = stronger
    rank = df.groupby("lineage_id")[metric].agg(["mean", "count"]).sort_values("mean", ascending=ascending)
    return list(rank.index)


def plot_team_evolution(
    min_year: int = 2010,
    max_year: int = 2020,
    metric: str = "position",
    window: int | None = None,
    top_k: int | None = None,
    output: str = "output/team_evolution.png",
    figsize: tuple = (12, 6),
):
    window = window or cfg.TIER_WINDOW
    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)
    df = build_evolution_df(db, min_year, max_year, window)

    seasons = sorted(df["season"].unique())
    lineages = _order_lineages(df, metric)
    if top_k is not None:
        lineages = lineages[:top_k]

    fig, ax = plt.subplots(figsize=figsize)

    for lid in lineages:
        ldf = df[df["lineage_id"] == lid].sort_values("season")
        series = ldf.set_index("season")[metric].reindex(seasons)
        label = lineage_label(ldf)
        ax.plot(
            seasons,
            series.values,
            marker="o",
            markersize=4,
            linewidth=2,
            color=_lineage_color(lid, label),
            label=label,
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
        ax.set_ylabel("Smoothed points share (lineage-aware)")
    elif metric == "share":
        ax.set_ylabel("Season points share")
    else:
        ax.set_ylabel(metric)

    ax.set_title(f"Constructor championship evolution ({min_year}–{max_year})")
    ax.grid(True, alpha=0.3)

    # Legend below the axes, multiple columns, ordered strongest-first.
    ncol = max(1, len(lineages) // 6 + 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol, fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output} ({len(lineages)} lineages)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot constructor championship evolution (lineage-aware).",
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
    parser.add_argument("--window", type=int, default=None, help="Trailing-average window (default: cfg.TIER_WINDOW).")
    parser.add_argument("--top-k", type=int, default=None, help="Only plot the k best lineages (by avg metric).")
    parser.add_argument("--output", type=str, default="output/team_evolution.png")
    args = parser.parse_args()

    plot_team_evolution(
        min_year=args.min_year,
        max_year=args.max_year,
        metric=args.metric,
        window=args.window,
        top_k=args.top_k,
        output=args.output,
    )


if __name__ == "__main__":
    main()
