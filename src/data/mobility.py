"""Identification diagnostics: mobility, connectivity, and support scores.

Measures whether the assignment structure provides enough variation to separate
driver from constructor effects (teammate comparisons + transfers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .skill_dataset import build_skill_dataset, SkillDatasetConfig
from relbench.base import Database


@dataclass
class MobilityReport:
    n_drivers: int
    n_constructors: int
    n_races: int
    n_teammate_pairs: int
    n_transfer_drivers: int
    largest_constructor_component: int
    weak_link_pairs: int
    support: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "n_drivers": self.n_drivers,
            "n_constructors": self.n_constructors,
            "n_races": self.n_races,
            "n_teammate_pairs": self.n_teammate_pairs,
            "n_transfer_drivers": self.n_transferors if hasattr(self, "n_transferors") else self.n_transfer_drivers,
            "largest_constructor_component": self.largest_constructor_component,
            "weak_link_pairs": self.weak_link_pairs,
        }


def _union_find_components(edges: List[Tuple[int, int]], nodes: Set[int]) -> Dict[int, int]:
    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return {n: find(n) for n in nodes}


def count_teammate_pairs(df: pd.DataFrame) -> int:
    """Ordered pairs (A ahead of B) among teammates in the same race."""
    count = 0
    for (_, race_id), grp in df.groupby(["constructorId", "raceId"]):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("race_position_order")
        n = len(grp)
        count += n * (n - 1) // 2
    return count


def build_transfer_edges(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Connect constructors if a driver raced for both (mobility graph)."""
    edges: Set[Tuple[int, int]] = set()
    for driver_id, grp in df.groupby("driverId"):
        by_season = grp.groupby("year")["constructorId"].agg(lambda x: x.mode().iloc[0])
        cons = by_season.unique().tolist()
        for i in range(len(cons)):
            for j in range(i + 1, len(cons)):
                a, b = int(cons[i]), int(cons[j])
                edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def compute_support_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(driver, season) support based on cumulative transfers and tenure."""
    ds = (
        df.groupby(["driverId", "year", "constructorId"])
        .size()
        .reset_index(name="n")
        .rename(columns={"year": "season"})
    )
    rows = []
    for driver_id, grp in ds.groupby("driverId"):
        grp = grp.sort_values("season")
        seen: Set[int] = set()
        n_seasons = 0
        for season, sg in grp.groupby("season"):
            seen.update(sg["constructorId"].unique().tolist())
            n_seasons += 1
            n_cons = len(seen)
            score = n_cons + 0.5 * n_seasons
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
                    "support_score": float(score),
                    "support_bucket": bucket,
                    "n_constructors": n_cons,
                    "n_seasons": n_seasons,
                }
            )
    return pd.DataFrame(rows).sort_values(["driverId", "season"]).reset_index(drop=True)


def compute_mobility_report(
    db: Database,
    config: Optional[SkillDatasetConfig] = None,
) -> MobilityReport:
    """Full mobility / identification diagnostic report."""
    df = build_skill_dataset(db, config)
    constructors = set(df["constructorId"].astype(int).unique())
    transfer_edges = build_transfer_edges(df)
    components = _union_find_components(transfer_edges, constructors)
    comp_sizes = pd.Series(list(components.values())).value_counts()
    largest = int(comp_sizes.max()) if len(comp_sizes) else 0

    # Drivers with >=2 constructors in career (up to max year)
    n_transfer = 0
    for _, grp in df.groupby("driverId"):
        if grp["constructorId"].nunique() >= 2:
            n_transfer += 1

    support = compute_support_scores(df)

    return MobilityReport(
        n_drivers=int(df["driverId"].nunique()),
        n_constructors=int(df["constructorId"].nunique()),
        n_races=int(df["raceId"].nunique()),
        n_teammate_pairs=count_teammate_pairs(df),
        n_transfer_drivers=n_transfer,
        largest_constructor_component=largest,
        weak_link_pairs=0,  # populated by extended AKM-style analysis if needed
        support=support,
    )


def build_race_pairs_for_bt(df: pd.DataFrame) -> pd.DataFrame:
    """Bradley–Terry observations: ordered pairs within each race.

    Returns [driverA, driverB, constructorA, constructorB, year, raceId].
    """
    rows = []
    ranked = df[df["in_race_ranking"]].copy()
    for race_id, grp in ranked.groupby("raceId"):
        grp = grp.sort_values("race_position_order")
        drv = grp["driverId"].tolist()
        cons = grp["constructorId"].tolist()
        year = int(grp["year"].iloc[0])
        n = len(drv)
        for i in range(n):
            for j in range(i + 1, n):
                rows.append(
                    {
                        "driverA": int(drv[i]),
                        "driverB": int(drv[j]),
                        "constructorA": int(cons[i]),
                        "constructorB": int(cons[j]),
                        "year": year,
                        "raceId": int(race_id),
                    }
                )
    return pd.DataFrame(rows)
