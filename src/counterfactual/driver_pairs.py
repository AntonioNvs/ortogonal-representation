"""Pair and graph builders for the Bradley-Terry driver-vs-car model.

Two outputs:

``build_race_pairs`` — every *ordered* pair (A finished ahead of B) within each
race, with the two drivers' stable ids and their constructor-season node
indices. These are the observations for the Bradley-Terry likelihood (both
teammate and cross-team pairs; cross-team pairs identify the car strength q).

``build_teammate_edges`` — the driver *collaboration graph*: two drivers are
linked if they shared a constructor in a season (i.e. were teammates). Used as
the graph for the Laplacian regularizer on theta, giving the relational graph
a leak-free role (a prior over latents, not a predictor of outcomes).
"""

from __future__ import annotations

import pandas as pd

from data.temporal_graph import TemporalGraph


def _with_driver_id(graph: TemporalGraph) -> pd.DataFrame:
    fr = graph.raced_in[
        ["driver_season", "constructor_season", "race", "positionOrder", "year"]
    ].copy()
    fr["driverId"] = fr["driver_season"].map(
        graph.driver_season.set_index("node_idx")["driverId"]
    )
    return fr


def build_race_pairs(graph: TemporalGraph) -> pd.DataFrame:
    """All ordered within-race pairs (A finished ahead of B).

    Returns columns ``[driverA, driverB, cs_A, cs_B, year]`` where ``driverA``
    and ``driverB`` are stable ``driverId`` values and ``cs_*`` are
    ``constructor_season`` node indices.
    """
    fr = _with_driver_id(graph)
    rows = []
    for race, grp in fr.groupby("race", sort=True):
        grp = grp.sort_values("positionOrder")
        drv = grp["driverId"].tolist()
        cs = grp["constructor_season"].tolist()
        year = int(grp["year"].iloc[0])
        n = len(drv)
        for i in range(n):
            for j in range(i + 1, n):
                # i finished ahead of j.
                rows.append(
                    {
                        "driverA": int(drv[i]),
                        "driverB": int(drv[j]),
                        "cs_A": int(cs[i]),
                        "cs_B": int(cs[j]),
                        "year": year,
                    }
                )
    return pd.DataFrame(rows)


def driver_id_to_index(pairs: pd.DataFrame) -> dict[int, int]:
    """Map a stable ``driverId`` to a dense 0..N-1 index over all drivers
    appearing in ``pairs``."""
    ids = sorted(set(pairs["driverA"]).union(set(pairs["driverB"])))
    return {int(d): i for i, d in enumerate(ids)}


def build_teammate_edges(graph: TemporalGraph) -> list[tuple[int, int]]:
    """Undirected edges between drivers who shared a constructor in a season.

    Returns a list of ``(driverId_i, driverId_j)`` pairs (i < j).
    """
    fr = _with_driver_id(graph)[["driverId", "constructor_season"]]
    edges: set[tuple[int, int]] = set()
    for _, grp in fr.groupby("constructor_season", sort=True):
        drv = sorted(set(grp["driverId"].astype(int)))
        for i in range(len(drv)):
            for j in range(i + 1, len(drv)):
                edges.add((drv[i], drv[j]))
    return sorted(edges)
