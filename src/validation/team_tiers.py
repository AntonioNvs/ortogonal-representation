"""Deterministic team-tier assignment (pure data, model-agnostic).

A team's tier in a season is assigned by ranking teams on their *smoothed
points share* (a trailing moving average, so the tier at T uses only data <= T)
and cutting each season into fixed proportions:

    S = top ~30% of teams (by smoothed share) in that season
    A = next ~35%
    B = the remainder (>= 35%, absorbing any rounding leftover)

Proportions are fixed, so "S" always means "roughly the best third of the
grid", which keeps a historically dominant team (e.g. Ferrari) in S in almost
every season while staying deterministic and leak-free.
"""

from __future__ import annotations

import pandas as pd

# Tier label -> scalar score used for ranking/correlation. Mirrors
# ``config.TIER_TO_SCORE`` (single small stable constant, kept in both places).
TIER_TO_SCORE = {"S": 3, "A": 2, "B": 1}

# Fixed per-season proportions (share of the grid in each tier). Remainder of
# the integer split goes to B.
P_S = 0.30
P_A = 0.35


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


def _add_score(
    points_df: pd.DataFrame,
    window: int,
    lineage: dict | None = None,
) -> pd.DataFrame:
    """Add a trailing moving-average ``score`` column.

    Grouped by constructor, unless ``lineage`` (a ``constructorId -> lineage_id``
    mapping) is given, in which case the average is computed per lineage so a
    rebranded/acquired team carries its score across the boundary.
    """
    df = points_df.copy()
    if lineage is not None:
        df["_group"] = df["constructorId"].map(lineage).fillna(df["constructorId"])
    else:
        df["_group"] = df["constructorId"]
    df = df.sort_values(["_group", "season"])
    df["score"] = df.groupby("_group")["share"].transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    return df.drop(columns=["_group"])


def compute_team_tiers(
    points_df: pd.DataFrame,
    window: int = 3,
    p_S: float = P_S,
    p_A: float = P_A,
    lineage: dict | None = None,
) -> pd.DataFrame:
    """Assign S/A/B tiers per (constructor, season) by fixed proportions.

    Within each season, teams are ranked by their smoothed share (descending),
    then the top ``floor(p_S * n)`` are S, the next ``floor(p_A * n)`` are A,
    and the rest are B (absorbing the integer rounding leftover).

    ``lineage`` optionally makes the smoothing lineage-aware (see
    ``validation.team_lineage``) so a rebranded team keeps its rank.

    Returns columns: [constructorId, constructorRef, season, score, tier].
    """
    df = _add_score(points_df, window, lineage=lineage).copy()

    # Deterministic rank within season: score desc, then points desc, then id.
    df = df.sort_values(
        ["season", "score", "points", "constructorId"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    tier_labels = []
    for _, grp in df.groupby("season", sort=True):
        n = len(grp)
        n_s = int(p_S * n)
        n_a = int(p_A * n)
        n_b = n - n_s - n_a
        tier_labels.extend(["S"] * n_s + ["A"] * n_a + ["B"] * n_b)

    df["tier"] = tier_labels
    return df[["constructorId", "constructorRef", "season", "score", "tier"]].reset_index(
        drop=True
    )
