"""Support score for the counterfactual swap.

A driver's counterfactual skill is only *identified* when the model has
observed enough variation to separate the driver from the car. The canonical
signal of that variation is a **transfer**: if driver X drove for two teams,
the model has seen X's embedding in two different car contexts and can
interpolate; if a team had two drivers, it has seen that car with two
different drivers. A one-team rookie (e.g. Antonelli@2025) has neither, so
their swap is an extrapolation.

This module turns that intuition into a per-(driver, season) score:

    support(X, T) = n_constructors_up_to_T(X) + 0.5 * n_seasons_up_to_T(X)

Both counts are *cumulative up to season T* (a driver's support grows as their
career does). Buckets are thresholded, not quantile-cut, so they are stable
and interpretable:

    high    — n_constructors >= 2           (has transferred: the natural experiment)
    low     — n_constructors == 1 and n_seasons <= 2   (rookie / one-team)
    medium  — otherwise                     (veteran on a single team)
"""

from __future__ import annotations

import pandas as pd

from data.temporal_graph import TemporalGraph


def compute_support(graph: TemporalGraph) -> pd.DataFrame:
    """Return ``[driverId, season, support_score, support_bucket]``."""
    ds = graph.driver_season.set_index("node_idx")
    cs_map = graph.constructor_season.set_index("node_idx")["constructorId"]

    fr = graph.raced_in[["driver_season", "constructor_season"]].copy()
    fr["driverId"] = fr["driver_season"].map(ds["driverId"])
    fr["season"] = fr["driver_season"].map(ds["season"])
    fr["constructorId"] = fr["constructor_season"].map(cs_map)

    # Distinct (driver, season, constructor) triples.
    pairs = (
        fr[["driverId", "season", "constructorId"]]
        .drop_duplicates()
        .dropna(subset=["driverId", "season", "constructorId"])
    )
    pairs["driverId"] = pairs["driverId"].astype(int)
    pairs["season"] = pairs["season"].astype(int)
    pairs["constructorId"] = pairs["constructorId"].astype(int)

    rows = []
    for driver_id, grp in pairs.groupby("driverId", sort=True):
        grp = grp.sort_values("season")
        seen_constructors: set[int] = set()
        n_seasons = 0
        for season, sg in grp.groupby("season"):
            seen_constructors.update(sg["constructorId"].unique().tolist())
            n_seasons += 1
            n_cons = len(seen_constructors)
            support = n_cons + 0.5 * n_seasons
            if n_cons >= 2:
                bucket = "high"
            elif n_cons == 1 and n_seasons <= 2:
                bucket = "low"
            else:
                bucket = "medium"
            rows.append(
                {
                    "driverId": int(driver_id),
                    "season": int(season),
                    "support_score": float(support),
                    "support_bucket": bucket,
                }
            )

    return pd.DataFrame(rows).sort_values(["driverId", "season"]).reset_index(drop=True)
