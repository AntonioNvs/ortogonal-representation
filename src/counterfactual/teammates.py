"""Teammate-pair construction for the beat-teammate objective.

Two drivers are *teammates* if they share the same ``constructor_season``
within the same race. For such a pair, the car, race, and circuit are all
identical — the ONLY thing that differs is the driver. Predicting which of the
two finished ahead is therefore impossible to solve from the car; it is the
cleanest objective for forcing gradient into the driver embedding.

Each returned row is an *ordered* pair where A finished ahead of B (lower
``positionOrder``), so the label is always ``1`` ("A beat B") by construction.
"""

from __future__ import annotations

import pandas as pd


def build_teammate_pairs(raced_in: pd.DataFrame) -> pd.DataFrame:
    """Form (A, B) ordered teammate pairs within the same (race, constructor).

    ``raced_in`` must have the columns produced by
    :meth:`data.temporal_graph.TemporalGraph.raced_in`: ``race``,
    ``constructor_season``, ``circuit``, ``driver_season``, ``positionOrder``,
    ``year``, and a ``driverId`` column (the latter is joined by callers).

    Returns one row per ordered pair where A finished ahead of B:
        [race, constructor_season, circuit, driver_A, driver_B, driverA_id,
         driverB_id, year]
    """
    rows = []
    for (race, cs), grp in raced_in.groupby(["race", "constructor_season"]):
        grp = grp.sort_values("positionOrder")
        drv = grp["driver_season"].tolist()
        drv_id = grp["driverId"].tolist()
        circuit = int(grp["circuit"].iloc[0])
        year = int(grp["year"].iloc[0])
        n = len(drv)
        for i in range(n):
            for j in range(i + 1, n):
                # i finished ahead of j (lower positionOrder).
                rows.append(
                    {
                        "race": int(race),
                        "constructor_season": int(cs),
                        "circuit": circuit,
                        "driver_A": int(drv[i]),
                        "driver_B": int(drv[j]),
                        "driverA_id": int(drv_id[i]),
                        "driverB_id": int(drv_id[j]),
                        "year": year,
                    }
                )
    return pd.DataFrame(rows)
