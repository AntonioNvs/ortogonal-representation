"""Lightweight alternating-effects baseline (season-level random effects).

Deterministic iterative proportional fitting on teammate + cross-team pairs.
Not a full Bayesian model — deterministic alternating-effects IPF baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.mobility import build_race_pairs_for_bt
from data.skill_dataset import SkillDatasetConfig, build_skill_dataset
from relbench.base import Database


def _fit_additive_effects(
    pairs: pd.DataFrame,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Simple alternating updates for driver theta and constructor-season q."""
    drivers = sorted(set(pairs["driverA"]).union(set(pairs["driverB"])))
    cs_keys = sorted(
        set(zip(pairs["constructorA"], pairs["year"])).union(
            zip(pairs["constructorB"], pairs["year"])
        )
    )
    theta = {d: 0.0 for d in drivers}
    q = {k: 0.0 for k in cs_keys}

    for _ in range(max_iter):
        old_theta = theta.copy()
        # Update theta: mean residual of wins
        for d in drivers:
            wins = pairs[(pairs["driverA"] == d) | (pairs["driverB"] == d)]
            if wins.empty:
                continue
            vals = []
            for _, r in wins.iterrows():
                if r["driverA"] == d:
                    q_a = q[(int(r["constructorA"]), int(r["year"]))]
                    q_b = q[(int(r["constructorB"]), int(r["year"]))]
                    vals.append(1.0 - (theta[r["driverB"]] + q_b - q_a))
                else:
                    q_a = q[(int(r["constructorA"]), int(r["year"]))]
                    q_b = q[(int(r["constructorB"]), int(r["year"]))]
                    vals.append(-(theta[r["driverA"]] + q_a - q_b))
            theta[d] = float(np.mean(vals)) if vals else 0.0

        # Center theta
        mean_t = np.mean(list(theta.values()))
        theta = {k: v - mean_t for k, v in theta.items()}

        # Update q similarly (simplified)
        for key in cs_keys:
            cid, year = key
            sub = pairs[
                ((pairs["constructorA"] == cid) | (pairs["constructorB"] == cid))
                & (pairs["year"] == year)
            ]
            if sub.empty:
                continue
            vals = []
            for _, r in sub.iterrows():
                base = theta[r["driverA"]] - theta[r["driverB"]]
                vals.append(0.5 - base if r["constructorA"] == cid else -(0.5 - base))
            q[key] = float(np.mean(vals)) if vals else 0.0

        delta = max(abs(theta[d] - old_theta[d]) for d in drivers)
        if delta < tol:
            break
    return theta, q


def load_bayesian_comparator_skill(db: Database, max_year: int = 2025) -> pd.DataFrame:
    df = build_skill_dataset(db, SkillDatasetConfig(max_year=max_year))
    pairs = build_race_pairs_for_bt(df)
    theta, _ = _fit_additive_effects(pairs)

    rows = []
    for (driver_id, season), grp in df.groupby(["driverId", "year"]):
        rows.append(
            {
                "driverId": int(driver_id),
                "season": int(season),
                "skill_score": float(theta.get(int(driver_id), 0.0)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["driverId", "season"]).sort_values(
        ["driverId", "season"]
    ).reset_index(drop=True)
