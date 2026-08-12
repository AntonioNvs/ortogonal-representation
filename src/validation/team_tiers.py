"""Deterministic team-tier assignment (pure data, model-agnostic).

A team's tier in a season is computed from its *points share* of that season,
smoothed by a trailing moving average (so the tier at T only uses data <= T),
then cut by fixed thresholds calibrated once over the whole history:

    score(team, season) = trailing mean of share over the last ``window`` seasons
    tier = S if score >= theta_S
           A if theta_A <= score < theta_S
           B otherwise

``theta_S`` is the median score of the constructors' champions; ``theta_A`` is
the median score of top-3 finishers. Because both thresholds are global
constants, tiers are comparable across eras, and historically dominant teams
(e.g. Ferrari) land in S almost every season without any manual rule.
"""

from __future__ import annotations

import pandas as pd

# Tier label -> scalar score used for ranking/correlation. Mirrors
# ``config.TIER_TO_SCORE`` (single small stable constant, kept in both places).
TIER_TO_SCORE = {"S": 3, "A": 2, "B": 1}


def compute_constructor_season_points(db) -> pd.DataFrame:
    """Season-end constructor points and their share of the season.

    Source: ``constructor_standings`` (cumulative per race), joined to ``races``
    (year/round) and ``constructors`` (ref). The season total is the value at the
    last round of each season; ``share = points / sum(points) over the season``.

    Returns columns: [constructorId, constructorRef, season, position, points, share].
    """
    standings = db.table_dict["constructor_standings"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]
    constructors = db.table_dict["constructors"].df[["constructorId", "constructorRef"]]

    df = standings.merge(races, on="raceId", how="inner")
    df = df.merge(constructors, on="constructorId", how="inner")
    df = df.sort_values(["constructorId", "year", "round"])

    # Season-end row = last round of each (constructor, season).
    season_end = df.groupby(["constructorId", "year"], as_index=False).last()
    season_end = season_end.rename(columns={"year": "season"})

    season_end["position"] = pd.to_numeric(season_end["position"], errors="coerce")
    season_end["points"] = pd.to_numeric(season_end["points"], errors="coerce").fillna(0.0)

    season_totals = season_end.groupby("season")["points"].transform("sum")
    season_end["share"] = season_end["points"] / season_totals.replace(0.0, float("nan"))
    season_end["share"] = season_end["share"].fillna(0.0)

    cols = ["constructorId", "constructorRef", "season", "position", "points", "share"]
    return season_end[cols].reset_index(drop=True)


def _add_score(points_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add a trailing moving-average ``score`` column (per constructor)."""
    df = points_df.sort_values(["constructorId", "season"]).copy()
    df["score"] = df.groupby("constructorId")["share"].transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    return df


def calibrate_thresholds(points_df: pd.DataFrame, window: int = 3) -> tuple[float, float]:
    """Return (theta_S, theta_A) as medians of champion / top-3 smoothed scores."""
    df = _add_score(points_df, window)
    champions = df[df["position"] == 1.0]["score"].dropna()
    top3 = df[df["position"].isin([1.0, 2.0, 3.0])]["score"].dropna()

    if champions.empty or top3.empty:
        raise ValueError("Cannot calibrate tier thresholds: no champion/top-3 rows found.")

    theta_S = float(champions.median())
    theta_A = float(top3.median())
    if theta_A > theta_S:
        # Guarantee S is strictly the top band.
        theta_S, theta_A = theta_A, theta_S
    return theta_S, theta_A


def compute_team_tiers(
    points_df: pd.DataFrame,
    window: int = 3,
    theta_S: float | None = None,
    theta_A: float | None = None,
) -> pd.DataFrame:
    """Assign S/A/B tiers per (constructor, season).

    Returns columns: [constructorId, constructorRef, season, score, tier].
    """
    if theta_S is None or theta_A is None:
        theta_S, theta_A = calibrate_thresholds(points_df, window)

    df = _add_score(points_df, window)
    df["tier"] = df["score"].apply(
        lambda s: "S" if s >= theta_S else ("A" if s >= theta_A else "B")
    )
    return df[["constructorId", "constructorRef", "season", "score", "tier"]].reset_index(
        drop=True
    )
