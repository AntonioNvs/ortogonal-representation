"""Forward career-outcome labels (pure data, model-agnostic).

A driver's career is traced through ``results`` (which team they drove for each
season). The forward outcome at season T is the mean tier of the teams they
drive for in seasons T+1 .. T+horizon, mapped to a scalar via ``TIER_TO_SCORE``.
This is the label the skill score is correlated against: a good driver is one
whose presence in the following years coincides with high-tier teams (the
"efficient market" hypothesis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .team_tiers import TIER_TO_SCORE


def driver_season_constructor(db) -> pd.DataFrame:
    """Map each (driver, season) to their (modal) constructor.

    Joins ``results`` -> ``races`` (raceId -> year) -> ``drivers`` (ref), then
    takes the modal ``constructorId`` per (driver, season), handling mid-season
    transfers by picking the team they raced for most.

    Returns columns: [driverId, driverRef, season, constructorId].
    """
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
    """Forward career-outcome score for each (driver, season T).

    ``outcome_score`` is the mean tier-scalar of the driver's teams in
    T+1 .. T+horizon (only observed seasons count). Rows with no observed
    forward season are dropped.

    ``require_full_horizon`` (default False for back-compat): if True, only
    keep rows where all ``horizon`` forward seasons are observed. This is the
    principled choice — otherwise a driver's last observable season (n=1)
    weighs the same as a full-horizon career point (n=horizon), and drivers
    who retire after T have their last positive evidence dropped. Recommend
    enabling for the paper's headline numbers.

    Returns columns: [driverId, driverRef, season_T, outcome_score, n_observed].
    """
    if tier_to_score is None:
        tier_to_score = TIER_TO_SCORE

    # (constructorId, season) -> tier
    tier_lookup = team_tier.set_index(["constructorId", "season"])["tier"]

    rows = []
    for (driver_id, driver_ref), grp in driver_season.groupby(
        ["driverId", "driverRef"], sort=True
    ):
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
