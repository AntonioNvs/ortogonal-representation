"""Teammate-centered residual baseline: driver minus team mean per race."""

from __future__ import annotations

import pandas as pd

from data.skill_dataset import SkillDatasetConfig, build_skill_dataset
from relbench.base import Database


def load_teammate_residual_skill(db: Database, max_year: int = 2025) -> pd.DataFrame:
    """Season mean of (driver race_skill - teammate mean race_skill)."""
    df = build_skill_dataset(db, SkillDatasetConfig(max_year=max_year))
    ranked = df[df["in_race_ranking"]].copy()

    residuals = []
    for (race_id, constructor_id), grp in ranked.groupby(["raceId", "constructorId"]):
        team_mean = grp["race_skill"].mean()
        for _, row in grp.iterrows():
            residuals.append(
                {
                    "driverId": int(row["driverId"]),
                    "season": int(row["year"]),
                    "residual": float(row["race_skill"] - team_mean),
                }
            )
    res_df = pd.DataFrame(residuals)
    season_skill = (
        res_df.groupby(["driverId", "season"])["residual"]
        .mean()
        .reset_index()
        .rename(columns={"residual": "skill_score"})
    )
    return season_skill.sort_values(["driverId", "season"]).reset_index(drop=True)
