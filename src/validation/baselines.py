"""Skill-scorer baselines depending only on the raw DB."""

from __future__ import annotations

import pandas as pd

from validation.team_tiers import TIER_TO_SCORE


def load_points_share(db) -> pd.DataFrame:
    standings = db.table_dict["standings"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]
    df = standings.merge(races, on="raceId", how="inner")
    df = df.sort_values(["driverId", "year", "round"])
    season_end = df.groupby(["driverId", "year"], as_index=False).last()
    season_end = season_end.rename(columns={"year": "season"})
    season_end["points"] = pd.to_numeric(season_end["points"], errors="coerce").fillna(0.0)
    season_max = season_end.groupby("season")["points"].transform("max")
    season_end["skill_score"] = season_end["points"] / season_max.replace(0.0, float("nan"))
    season_end["skill_score"] = season_end["skill_score"].fillna(0.0)
    return season_end[["driverId", "season", "skill_score"]].sort_values(["driverId", "season"]).reset_index(drop=True)


def load_constructor_tier(db, team_tier: pd.DataFrame) -> pd.DataFrame:
    from validation.career_labels import driver_season_constructor

    ds = driver_season_constructor(db)
    tier_lookup = team_tier.set_index(["constructorId", "season"])["tier"].to_dict()
    ds["tier"] = [
        tier_lookup.get((int(cid), int(s))) for cid, s in zip(ds["constructorId"], ds["season"])
    ]
    ds = ds.dropna(subset=["tier"])
    ds["skill_score"] = ds["tier"].map(TIER_TO_SCORE).astype(float)
    return ds[["driverId", "season", "skill_score"]].sort_values(["driverId", "season"]).reset_index(drop=True)
