"""
Utilities for extracting teammate pairs from race results.

For each race, we identify driver pairs that drove for the same constructor
and determine which driver finished ahead of the other.
"""

from __future__ import annotations

import pandas as pd


def extract_teammate_pairs(
    results_df: pd.DataFrame,
    race_id: int,
) -> list[tuple[int, int, int, float]]:
    """Extract beat-teammate labels for a single race.

    For each constructor with exactly 2 drivers in the race, create a pair
    (driver_A, driver_B, constructor_id, label) where label = 1 if driver_A
    finished ahead of driver_B (by ``positionOrder``), 0 otherwise.

    The ordering (A, B) is deterministic: driver with the lower ``driverId``
    is always A, and the label is flipped accordingly.

    Args:
        results_df: DataFrame with columns [raceId, driverId, constructorId,
                   positionOrder].
        race_id: The race to extract pairs for.

    Returns:
        List of (driver_A, driver_B, constructor_id, label) tuples.
    """
    race_results = results_df[results_df["raceId"] == race_id].copy()

    if race_results.empty:
        return []

    # Group by constructor
    pairs = []
    for constructor_id, group in race_results.groupby("constructorId"):
        drivers = group.sort_values("driverId")  # deterministic ordering

        if len(drivers) != 2:
            # Skip teams with 1 or 3+ drivers (very rare in F1 post-2000)
            continue

        driver_a = drivers.iloc[0]
        driver_b = drivers.iloc[1]

        driver_a_id = int(driver_a["driverId"])
        driver_b_id = int(driver_b["driverId"])
        constructor_id_int = int(constructor_id)

        # label = 1 if A beat B (A has lower positionOrder = better finish)
        label = 1.0 if driver_a["positionOrder"] < driver_b["positionOrder"] else 0.0

        pairs.append((driver_a_id, driver_b_id, constructor_id_int, label))

    return pairs


def extract_teammate_pairs_from_race_results(
    results_df: pd.DataFrame,
    race_id: int,
) -> list[dict]:
    """Extract beat-teammate pairs with richer metadata.

    Returns a list of dicts with keys: driver_a, driver_b, constructor_id,
    label, position_a, position_b, qualifying_a, qualifying_b, grid_a, grid_b.

    Args:
        results_df: DataFrame with columns [raceId, driverId, constructorId,
                   positionOrder, grid]. Must also contain qualifying_position
                   if available (merged separately).
        race_id: The race to extract pairs for.

    Returns:
        List of dicts.
    """
    race_results = results_df[results_df["raceId"] == race_id].copy()

    if race_results.empty:
        return []

    pairs = []
    for constructor_id, group in race_results.groupby("constructorId"):
        drivers = group.sort_values("driverId")

        if len(drivers) != 2:
            continue

        driver_a = drivers.iloc[0]
        driver_b = drivers.iloc[1]

        label = 1.0 if driver_a["positionOrder"] < driver_b["positionOrder"] else 0.0

        pair = {
            "driver_a": int(driver_a["driverId"]),
            "driver_b": int(driver_b["driverId"]),
            "constructor_id": int(constructor_id),
            "label": label,
            "position_a": int(driver_a["positionOrder"]),
            "position_b": int(driver_b["positionOrder"]),
        }

        # Optional: add qualifying and grid if available
        if "qualifying_position" in driver_a:
            pair["qualifying_a"] = float(driver_a["qualifying_position"]) if pd.notna(driver_a["qualifying_position"]) else 0.0
            pair["qualifying_b"] = float(driver_b["qualifying_position"]) if pd.notna(driver_b["qualifying_position"]) else 0.0
        if "grid" in driver_a:
            pair["grid_a"] = float(driver_a["grid"]) if pd.notna(driver_a["grid"]) else 0.0
            pair["grid_b"] = float(driver_b["grid"]) if pd.notna(driver_b["grid"]) else 0.0

        pairs.append(pair)

    return pairs


def count_teammate_pairs(results_df: pd.DataFrame, race_ids: list[int] | None = None) -> int:
    """Count total teammate pairs across a set of race IDs.

    Args:
        results_df: DataFrame with race results.
        race_ids: Races to count. If None, uses all races in the DataFrame.

    Returns:
        Total number of (driver_A, driver_B) pairs.
    """
    if race_ids is None:
        race_ids = results_df["raceId"].unique().tolist()

    total = 0
    for race_id in race_ids:
        pairs = extract_teammate_pairs(results_df, race_id)
        total += len(pairs)
    return total