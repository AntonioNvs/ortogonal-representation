"""Forward career-outcome labels (model-agnostic)."""

from __future__ import annotations

from typing import Optional

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


def rest_of_career_outcome(
    driver_season: pd.DataFrame,
    team_tier: pd.DataFrame,
    tier_to_score: dict | None = None,
) -> pd.DataFrame:
    """Mean tier score over all future seasons until career end or data cutoff.

    Unlike ``forward_tier_outcome(horizon=3)``, uses every remaining active
    season (infinite horizon) and does not require a fixed number of future
    observations.
    """
    if tier_to_score is None:
        tier_to_score = TIER_TO_SCORE

    tier_lookup = team_tier.set_index(["constructorId", "season"])["tier"]
    rows = []
    for (driver_id, driver_ref), grp in driver_season.groupby(["driverId", "driverRef"], sort=True):
        grp = grp.sort_values("season")
        season_to_constructor = dict(zip(grp["season"].astype(int), grp["constructorId"]))
        seasons = sorted(season_to_constructor.keys())
        for season_t in seasons:
            tier_at_t_key = (int(season_to_constructor[season_t]), int(season_t))
            if tier_at_t_key not in tier_lookup.index:
                continue
            tier_at_t = tier_lookup.loc[tier_at_t_key]
            if tier_at_t not in tier_to_score:
                continue
            tier_score_at_t = tier_to_score[tier_at_t]

            future_scores: list[float] = []
            future_tier_scores: list[float] = []
            first_promotion_season: Optional[int] = None
            for s in seasons:
                if s <= season_t:
                    continue
                constructor = season_to_constructor.get(s)
                if constructor is None:
                    continue
                key = (int(constructor), int(s))
                if key not in tier_lookup.index:
                    continue
                tier = tier_lookup.loc[key]
                if tier not in tier_to_score:
                    continue
                score = tier_to_score[tier]
                future_scores.append(float(score))
                future_tier_scores.append(float(score))
                if first_promotion_season is None and score > tier_score_at_t:
                    first_promotion_season = int(s)

            if not future_scores:
                continue
            rows.append(
                {
                    "driverId": int(driver_id),
                    "driverRef": driver_ref,
                    "season_T": int(season_t),
                    "outcome_score": float(np.mean(future_scores)),
                    "n_future_seasons": len(future_scores),
                    "peak_tier_score": float(max(future_tier_scores)),
                    "first_promotion_season": first_promotion_season,
                    "seasons_to_promotion": (
                        int(first_promotion_season - season_t)
                        if first_promotion_season is not None
                        else None
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "driverId",
                "driverRef",
                "season_T",
                "outcome_score",
                "n_future_seasons",
                "peak_tier_score",
                "first_promotion_season",
                "seasons_to_promotion",
            ]
        )
    return pd.DataFrame(rows)


def career_outcome_labels(
    driver_season: pd.DataFrame,
    team_tier: pd.DataFrame,
    *,
    horizon: Optional[int] = None,
    require_full_horizon: bool = False,
) -> pd.DataFrame:
    """Dispatch to fixed-horizon or rest-of-career outcome labels."""
    if horizon is None:
        return rest_of_career_outcome(driver_season, team_tier)
    return forward_tier_outcome(
        driver_season,
        team_tier,
        horizon=horizon,
        require_full_horizon=require_full_horizon,
    )
