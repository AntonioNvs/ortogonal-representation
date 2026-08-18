"""Skill-scorer baselines that only depend on the raw DB.

Each baseline exposes the same contract as ``kalman_skill.load_kalman_skill``:
a function returning a DataFrame ``[driverId, season, skill_score]``. Because
they consume only observable seasonal aggregates, they can run without any
trained model — a reviewer can rerun them from the raw enriched database.

Two baselines are shipped here:

* ``load_points_share``: normalised season points share.
  ``skill(driver, T) = points(driver, T) / max_points(T)``. This is the most
  naive market-aware baseline: "the driver who scored the most points this
  year is the most skilled." It should be *hard* for the Kalman skill to
  exceed this in absolute rho — the interesting question is whether Kalman
  exceeds it *conditional* on constructor_tier(T) (see partial_spearman).

* ``load_constructor_tier``: ``skill(driver, T) = tier_score(constructor(T))``.
  This is the *adversary* for the paper's central thesis. If it beats the
  Kalman skill, the model has not decomposed driver from car — the
  "efficient market" hypothesis holds tautologically through the team, not
  through anything the driver did. If Kalman only ties or beats it
  *marginally*, the decomposition is fragile.

A third baseline — the trained ``no_orthogonal`` GNN's driver readout —
would live here too but requires the (non-Kalman) evaluation pipeline; left
as a stub for a follow-up.
"""

from __future__ import annotations

import pandas as pd

from .team_tiers import TIER_TO_SCORE


def load_points_share(db) -> pd.DataFrame:
    """Per-(driver, season) share of the season's total driver points."""
    standings = db.table_dict["standings"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]

    df = standings.merge(races, on="raceId", how="inner")
    df = df.sort_values(["driverId", "year", "round"])
    # Season-end row = last round of each (driver, season).
    season_end = df.groupby(["driverId", "year"], as_index=False).last()
    season_end = season_end.rename(columns={"year": "season"})
    season_end["points"] = pd.to_numeric(season_end["points"], errors="coerce").fillna(0.0)

    season_max = season_end.groupby("season")["points"].transform("max")
    season_end["skill_score"] = season_end["points"] / season_max.replace(0.0, float("nan"))
    season_end["skill_score"] = season_end["skill_score"].fillna(0.0)

    return (
        season_end[["driverId", "season", "skill_score"]]
        .sort_values(["driverId", "season"])
        .reset_index(drop=True)
    )


def load_constructor_tier(db, team_tier: pd.DataFrame) -> pd.DataFrame:
    """The driver's team's tier in season T, mapped to a scalar.

    This baseline collapses "driver skill" to "car quality this year". It is
    the adversarial control for the decomposition thesis.
    """
    # (driverId, season) -> constructorId, via modal team of the year.
    from .career_labels import driver_season_constructor
    ds = driver_season_constructor(db)

    tier_lookup = team_tier.set_index(["constructorId", "season"])["tier"].to_dict()
    ds["tier"] = [
        tier_lookup.get((int(cid), int(s))) for cid, s in zip(ds["constructorId"], ds["season"])
    ]
    ds = ds.dropna(subset=["tier"])
    ds["skill_score"] = ds["tier"].map(TIER_TO_SCORE).astype(float)
    return (
        ds[["driverId", "season", "skill_score"]]
        .sort_values(["driverId", "season"])
        .reset_index(drop=True)
    )
