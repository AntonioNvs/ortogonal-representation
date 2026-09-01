"""Forward career-outcome labels (model-agnostic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .team_tiers import TIER_TO_SCORE


def driver_season_constructor(db) -> pd.DataFrame:
    results = db.table_dict["results"].df[["raceId", "driverId", "constructorId"]]
    races = db.table_dict["races"].df[["raceId", "year"]]
    drivers = db.table_dict["drivers"].df[["driverId", "driverRef"]]

    merged = results.merge(races, on="raceId", how="inner")
    season_constructor = (
        merged.groupby(["driverId", "year"])["constructorId"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
        .rename(columns={"year": "season"})
    )
    return season_constructor.merge(drivers, on="driverId", how="left")[
        ["driverId", "driverRef", "season", "constructorId"]
    ]


def forward_tier_outcome(
    driver_season: pd.DataFrame,
    team_tier: pd.DataFrame,
    horizon: int = 3,
    tier_to_score: dict | None = None,
    require_full_horizon: bool = False,
) -> pd.DataFrame:
    if tier_to_score is None:
        tier_to_score = TIER_TO_SCORE

    tier_lookup = team_tier.set_index(["constructorId", "season"])["tier"]
    rows = []
    for (driver_id, driver_ref), grp in driver_season.groupby(["driverId", "driverRef"], sort=True):
        grp = grp.sort_values("season")
        season_to_constructor = dict(zip(grp["season"].astype(int), grp["constructorId"]))
        for season_t in grp["season"].astype(int):
            scores = []
            for offset in range(1, horizon + 1):
                s = season_t + offset
                constructor = season_to_constructor.get(s)
                if constructor is None:
                    continue
                key = (int(constructor), int(s))
                if key not in tier_lookup.index:
                    continue
                tier = tier_lookup.loc[key]
                if tier in tier_to_score:
                    scores.append(tier_to_score[tier])
            if not scores:
                continue
            rows.append(
                {
                    "driverId": int(driver_id),
                    "driverRef": driver_ref,
                    "season_T": int(season_t),
                    "outcome_score": float(np.mean(scores)),
                    "n_observed": len(scores),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["driverId", "driverRef", "season_T", "outcome_score", "n_observed"]
        )
    out = pd.DataFrame(rows)
    if require_full_horizon:
        out = out[out["n_observed"] >= horizon].reset_index(drop=True)
    return out
