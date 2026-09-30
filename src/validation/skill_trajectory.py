"""Driver skill trajectory features for career validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _linear_slope(years: np.ndarray, values: np.ndarray) -> float:
    if len(years) < 2 or np.std(values) == 0:
        return 0.0
    coeffs = np.polyfit(years.astype(float), values.astype(float), 1)
    return float(coeffs[0])


def compute_skill_trajectory(season_skill: pd.DataFrame) -> pd.DataFrame:
    """Per (driverId, season_T) trajectory features from season-level skill.

    Parameters
    ----------
    season_skill : DataFrame
        Columns at minimum ``driverId``, ``season``, ``skill_score``.
    """
    required = {"driverId", "season", "skill_score"}
    missing = required - set(season_skill.columns)
    if missing:
        raise ValueError(f"season_skill missing columns: {missing}")

    df = season_skill[["driverId", "season", "skill_score"]].copy()
    df = df.rename(columns={"season": "season_T"})
    df = df.sort_values(["driverId", "season_T"])

    rows = []
    for driver_id, grp in df.groupby("driverId", sort=True):
        grp = grp.sort_values("season_T")
        seasons = grp["season_T"].astype(int).to_numpy()
        skills = grp["skill_score"].astype(float).to_numpy()
        n_seasons = np.arange(1, len(seasons) + 1)
        for i, season_t in enumerate(seasons):
            start = max(0, i - 2)
            window_seasons = seasons[start : i + 1]
            window_skills = skills[start : i + 1]
            skill_slope = _linear_slope(window_seasons, window_skills)
            skill_delta = float(skills[i] - skills[i - 1]) if i > 0 else 0.0
            career_n = int(i + 1)
            if career_n <= 2:
                career_phase = "debut"
            elif career_n <= 6:
                career_phase = "mid"
            else:
                career_phase = "veteran"
            rows.append(
                {
                    "driverId": int(driver_id),
                    "season_T": int(season_t),
                    "skill_slope_3yr": skill_slope,
                    "skill_delta": skill_delta,
                    "career_n_seasons": career_n,
                    "career_phase": career_phase,
                }
            )
    return pd.DataFrame(rows)


def enrich_career_join(joined: pd.DataFrame, season_skill: pd.DataFrame) -> pd.DataFrame:
    """Attach trajectory columns to a career-validation join."""
    traj = compute_skill_trajectory(season_skill)
    return joined.merge(traj, on=["driverId", "season_T"], how="left")
