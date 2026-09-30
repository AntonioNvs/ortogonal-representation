"""Deterministic team-tier assignment (pure data, model-agnostic)."""

from __future__ import annotations

import pandas as pd

TIER_TO_SCORE = {"S": 3, "A": 2, "B": 1}
P_S = 0.30
P_A = 0.35


def compute_constructor_season_points(db) -> pd.DataFrame:
    standings = db.table_dict["constructor_standings"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]
    constructors = db.table_dict["constructors"].df[["constructorId", "constructorRef"]]

    df = standings.merge(races, on="raceId", how="inner")
    df = df.merge(constructors, on="constructorId", how="left")
    df = df.sort_values(["constructorId", "year", "round"])
    season_end = df.groupby(["constructorId", "year"], as_index=False).last()
    season_end = season_end.rename(columns={"year": "season"})
    season_end["points"] = pd.to_numeric(season_end["points"], errors="coerce").fillna(0.0)
    season_totals = season_end.groupby("season")["points"].transform("sum")
    season_end["share"] = season_end["points"] / season_totals.replace(0.0, float("nan"))
    season_end["share"] = season_end["share"].fillna(0.0)
    return season_end[
        ["constructorId", "constructorRef", "season", "position", "points", "share"]
    ].reset_index(drop=True)


def _add_score(points_df: pd.DataFrame, window: int, lineage: dict | None = None) -> pd.DataFrame:
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
    df = _add_score(points_df, window, lineage=lineage).copy()
    df = df.sort_values(
        ["season", "score", "points", "constructorId"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    tier_labels = []
    for _, grp in df.groupby("season", sort=True):
        n = len(grp)
        n_s = int(p_S * n)
        n_a = int(p_A * n)
        tier_labels.extend(["S"] * n_s + ["A"] * n_a + ["B"] * (n - n_s - n_a))
    df["tier"] = tier_labels
    return df[["constructorId", "constructorRef", "season", "score", "tier"]].reset_index(drop=True)
