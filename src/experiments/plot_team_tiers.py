"""Plot the deterministic team tiers (S/A/B) as a season x team heatmap.

Complement to ``plot_team_evolution``: instead of a line per team, this shows
the *tier* each team landed in per season (same deterministic assignment used by
the career-validation framework). Teams are ordered strongest-first.

Optionally (``--skill-source kalman``) a second panel overlays the model: a
boxplot of driver ``skill_score`` grouped by the team tier they drove for. If the
model "makes sense", median skill should be ordered S > A > B.

Usage:
    python -m src.experiments.plot_team_tiers --min-year 2010 --max-year 2020
    python -m src.experiments.plot_team_tiers --min-year 2000 --max-year 2026 --top-k 10
    python -m src.experiments.plot_team_tiers --skill-source kalman --checkpoint output/kalman/kalman_gnn.pth
"""

from __future__ import annotations

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
from validation.career_labels import driver_season_constructor
from validation.team_tiers import (
    TIER_TO_SCORE,
    compute_constructor_season_points,
    compute_team_tiers,
)

# Medal-style, distinct-lightness colors. Text labels are overlaid on every cell
# so the plot stays unambiguous even if a color is hard to distinguish.
TIER_COLORS = {
    "S": "#f0c000",  # gold
    "A": "#6a9bd0",  # silver-blue
    "B": "#b07a3e",  # bronze
}
TIER_ORDER = ["S", "A", "B"]
MISSING_COLOR = "#ffffff"  # team did not compete that season


def _tier_color(tier) -> str:
    return TIER_COLORS.get(tier, MISSING_COLOR)


def build_tier_df(db, min_year: int, max_year: int, window: int) -> pd.DataFrame:
    """Deterministic S/A/B tier per (constructor, season), with names."""
    points = compute_constructor_season_points(db)
    points = points[(points["season"] >= min_year) & (points["season"] <= max_year)]
    tiers = compute_team_tiers(
        points, window=window, p_S=cfg.TIER_S_FRAC, p_A=cfg.TIER_A_FRAC
    )
    names = db.table_dict["constructors"].df[["constructorId", "name"]]
    tiers = tiers.merge(names, on="constructorId", how="left")
    return tiers


def _team_order(tiers: pd.DataFrame) -> list[str]:
    """Teams ordered strongest-first: mean tier-scalar desc, ties by total points."""
    tiers = tiers.copy()
    tiers["_v"] = tiers["tier"].map(TIER_TO_SCORE).astype(float)
    agg = (
        tiers.groupby("name")
        .agg(_v=("_v", "mean"), _n=("season", "count"))
        .sort_values(["_v", "_n"], ascending=[False, False])
    )
    return list(agg.index)


def plot_team_tiers(
    min_year: int = 2010,
    max_year: int = 2020,
    window: int | None = None,
    top_k: int | None = None,
    skill_source: str | None = None,
    checkpoint: str | None = None,
    device: str | None = None,
    output: str = "output/team_tiers.png",
    figsize: tuple | None = None,
):
    window = window or cfg.TIER_WINDOW
    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)
    tiers = build_tier_df(db, min_year, max_year, window)

    seasons = sorted(tiers["season"].unique())
    teams = _team_order(tiers)
    if top_k is not None:
        teams = teams[:top_k]

    # Build a matrix of tier values (NaN -> team absent that season).
    tier_value = tiers.pivot_table(
        index="name", columns="season", values="tier", aggfunc="first"
    ).reindex(index=teams, columns=seasons)

    has_skill = skill_source is not None
    nrows = 2 if has_skill else 1
    if figsize is None:
        figsize = (max(10, 0.6 * len(seasons) + 4), 0.35 * len(teams) + (4 if has_skill else 3))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1.6] if has_skill else [1]})

    ax = axes[0] if has_skill else axes

    # --- Heatmap ---
    n_rows, n_cols = tier_value.shape
    img = np.zeros((n_rows, n_cols, 3))
    for i, team in enumerate(teams):
        for j, season in enumerate(seasons):
            tier = tier_value.iloc[i, j]
            if pd.isna(tier):
                img[i, j] = [1.0, 1.0, 1.0]
            else:
                hexc = _tier_color(tier).lstrip("#")
                img[i, j] = [int(hexc[k : k + 2], 16) / 255.0 for k in (0, 2, 4)]

    ax.imshow(img, aspect="auto", interpolation="nearest")

    # Overlay tier letters.
    for i, team in enumerate(teams):
        for j, season in enumerate(seasons):
            tier = tier_value.iloc[i, j]
            if not pd.isna(tier):
                ax.text(j, i, tier, ha="center", va="center", fontsize=8, fontweight="bold", color="#1a1a1a")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(s) for s in seasons], rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(teams, fontsize=9)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_title(f"Team tiers by season ({min_year}–{max_year})  —  S/A/B, strongest on top")
    # Light grid so absent cells (white) still read as "not competing".
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.5, alpha=0.6)
    ax.tick_params(which="minor", length=0)

    # --- Skill-vs-tier panel ---
    if has_skill:
        ax_skill = axes[1]
        from validation.kalman_skill import load_kalman_skill

        skill = load_kalman_skill(
            checkpoint_path=checkpoint, device=device
        )
        driver_season = driver_season_constructor(db)
        # Map each (driver, season) to the tier of their team that season.
        ds_tier = driver_season.merge(
            tiers[["constructorId", "season", "tier"]],
            on=["constructorId", "season"],
            how="inner",
        )
        merged = skill.merge(
            ds_tier[["driverId", "season", "tier"]], on=["driverId", "season"], how="inner"
        ).dropna(subset=["skill_score"])

        groups = [merged.loc[merged["tier"] == t, "skill_score"].values for t in TIER_ORDER]
        bp = ax_skill.boxplot(groups, labels=TIER_ORDER, patch_artist=True, widths=0.6)
        for patch, t in zip(bp["boxes"], TIER_ORDER):
            patch.set_facecolor(_tier_color(t))
            patch.set_alpha(0.55)
        ax_skill.set_ylabel("Driver skill score")
        ax_skill.set_title(
            f"Driver skill vs. team tier (model source: {skill_source})"
        )
        ax_skill.grid(True, axis="y", alpha=0.3)

        # Medians printed for a quick S>A>B check.
        medians = {
            t: (float(np.median(g)) if len(g) else float("nan")) for t, g in zip(TIER_ORDER, groups)
        }
        counts = {t: int(len(g)) for t, g in zip(TIER_ORDER, groups)}
        print(f"  skill-by-tier medians: " + "  ".join(f"{t}={medians[t]:+.3f}(n={counts[t]})" for t in TIER_ORDER))

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output} ({len(teams)} teams, {len(seasons)} seasons)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot team tiers as a heatmap (optionally vs. driver skill).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-year", type=int, default=2020)
    parser.add_argument("--window", type=int, default=None, help="Tier moving-average window (default: cfg.TIER_WINDOW).")
    parser.add_argument("--top-k", type=int, default=None, help="Only show the k strongest teams.")
    parser.add_argument("--skill-source", type=str, default=None, help="e.g. 'kalman' to add the skill-vs-tier panel.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (for --skill-source kalman).")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. 'cuda:7', 'cpu').")
    parser.add_argument("--output", type=str, default="output/team_tiers.png")
    args = parser.parse_args()

    plot_team_tiers(
        min_year=args.min_year,
        max_year=args.max_year,
        window=args.window,
        top_k=args.top_k,
        skill_source=args.skill_source,
        checkpoint=args.checkpoint,
        device=args.device,
        output=args.output,
    )


if __name__ == "__main__":
    main()
