"""
Kalman-GNN data layer: chronological race ordering, sliding window edge cache,
and per-race batch construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Chronological race list
# ---------------------------------------------------------------------------


class ChronologicalRaceList:
    """Ordered list of all races in the database, with metadata.

    Provides lookups: race index → raceId, raceId → race index, and the
    ordered sequence of race IDs for sliding window traversal.
    """

    def __init__(self, db):
        """Build from a RelBench database.

        Args:
            db: RelBench database with tables ``races`` and ``results``.
        """
        races_df = db.table_dict["races"].df.copy()

        # Sort by date, then by raceId for determinism
        races_df = races_df.sort_values(["year", "round", "raceId"]).reset_index(drop=True)

        self.race_ids = races_df["raceId"].astype(int).tolist()
        self.years = races_df["year"].astype(int).tolist()
        self.rounds = races_df["round"].astype(int).tolist()

        # race_id → position in the chronological list
        self.race_id_to_idx = {int(rid): i for i, rid in enumerate(self.race_ids)}

        # Pre-compute which race IDs are in each split year range
        self._race_ids_set = set(self.race_ids)

    def __len__(self) -> int:
        return len(self.race_ids)

    def get_race_at(self, idx: int) -> dict:
        """Return metadata for the race at chronological position ``idx``."""
        return {
            "race_id": self.race_ids[idx],
            "year": self.years[idx],
            "round": self.rounds[idx],
            "idx": idx,
        }

    def get_idx_for_race(self, race_id: int) -> int | None:
        """Return chronological index for a race_id, or None if not found."""
        return self.race_id_to_idx.get(race_id)

    def get_prev_race_ids(self, race_idx: int, window_size: int) -> list[int]:
        """Return the ``window_size`` race IDs immediately before ``race_idx``.

        Returns an empty list if ``race_idx == 0``. Returns fewer than
        ``window_size`` if ``race_idx < window_size``.
        """
        start = max(0, race_idx - window_size)
        return self.race_ids[start:race_idx]

    def get_train_val_test_indices(
        self, train_years: list[int], val_years: list[int], test_years: list[int]
    ) -> dict[str, np.ndarray]:
        """Return boolean masks for each split.

        Args:
            train_years: years for training.
            val_years: years for validation.
            test_years: years for testing.

        Returns:
            Dict with keys "train", "val", "test" → boolean NumPy arrays
            of length ``len(self)``.
        """
        years_arr = np.array(self.years, dtype=int)
        train_mask = np.isin(years_arr, train_years)
        val_mask = np.isin(years_arr, val_years)
        test_mask = np.isin(years_arr, test_years)
        return {"train": train_mask, "val": val_mask, "test": test_mask}


# ---------------------------------------------------------------------------
# Sliding window edge cache
# ---------------------------------------------------------------------------


class SlidingWindowEdgeCache:
    """Pre-computed per-race edge index dictionaries for fast sliding window
    assembly.

    For each race, stores the subset of each edge type's edge_index where the
    source node belongs to that race.  A window of K races is assembled by
    concatenating the per-race dictionaries for the K preceding races.
    """

    def __init__(self, graph_data, db, race_list: ChronologicalRaceList, window_size: int = 20):
        """Pre-compute edge indices per race.

        Args:
            graph_data: HeteroData from ``make_pkey_fkey_graph``.
            db: RelBench database.
            race_list: Chronological race ordering.
            window_size: K, the number of races in each sliding window.
        """
        self.window_size = window_size
        self.race_list = race_list
        self._cache: dict[int, dict[tuple, torch.Tensor]] = {}
        self._edge_types = list(graph_data.edge_types)

        self._precompute(graph_data, db)

    def _precompute(self, graph_data, db):
        """Slice each edge type's edge_index by race and store in cache."""
        races_df = db.table_dict["races"].df
        year_of_race = dict(zip(races_df["raceId"], races_df["year"]))
        round_of_race = dict(zip(races_df["raceId"], races_df["round"]))

        for edge_type in self._edge_types:
            src_table = edge_type[0]
            num_edges = graph_data[edge_type].edge_index.shape[1]

            if src_table not in db.table_dict:
                # No temporal signal → store full edge_index under key -1 per race
                continue

            src_df = db.table_dict[src_table].df
            if "raceId" not in src_df.columns:
                continue

            edge_index = graph_data[edge_type].edge_index
            src_node_ids = edge_index[0].cpu().numpy()
            race_ids = src_df.iloc[src_node_ids]["raceId"].values

            for race_id in np.unique(race_ids):
                mask = race_ids == race_id
                if int(race_id) not in self._cache:
                    self._cache[int(race_id)] = {}
                self._cache[int(race_id)][edge_type] = edge_index[:, mask].clone()

    def get_window_for_race(self, race_idx: int) -> dict[tuple, torch.Tensor]:
        """Assemble the edge index dictionary for the window preceding ``race_idx``.

        Args:
            race_idx: chronological index of the *target* race (not in the window).

        Returns:
            edge_index_dict with edges from the K races before ``race_idx``.
        """
        prev_race_ids = self.race_list.get_prev_race_ids(race_idx, self.window_size)

        # Collect all edge types present in any cached race
        all_edge_types = set()
        for rid in prev_race_ids:
            if rid in self._cache:
                all_edge_types.update(self._cache[rid].keys())

        window_edges = {}
        for et in all_edge_types:
            chunks = []
            for rid in prev_race_ids:
                if rid in self._cache and et in self._cache[rid]:
                    chunks.append(self._cache[rid][et])
            if chunks:
                window_edges[et] = torch.cat(chunks, dim=1)

        return window_edges

    def get_window_for_race_id(self, race_id: int) -> dict[tuple, torch.Tensor]:
        """Convenience: assemble window from race_id instead of race_idx."""
        race_idx = self.race_list.get_idx_for_race(race_id)
        if race_idx is None:
            return {}
        return self.get_window_for_race(race_idx)


# ---------------------------------------------------------------------------
# Per-race batch construction
# ---------------------------------------------------------------------------


def build_race_batch(
    race_idx: int,
    edge_cache: SlidingWindowEdgeCache,
    results_df: pd.DataFrame,
    race_list: ChronologicalRaceList,
) -> dict:
    """Build a single-race batch for the Kalman-GNN training loop.

    Args:
        race_idx: chronological index of the target race.
        edge_cache: pre-computed sliding window edge indices.
        results_df: full results DataFrame (with positionOrder, grid, etc.).
        race_list: chronological race ordering.

    Returns:
        Dict with keys:
          - race_id: int
          - race_idx: int
          - year: int
          - round: int
          - edge_index_dict: dict of tensors (GNN edges for window)
          - active_driver_ids: Tensor of driver indices
          - active_constructor_ids: Tensor of constructor indices
          - teammate_pairs: list of (driver_a, driver_b, constructor_id, label)
          - qualifying_positions: dict driver_id → position
          - grids: dict driver_id → grid
    """
    from .teammate_utils import extract_teammate_pairs

    race_meta = race_list.get_race_at(race_idx)
    race_id = race_meta["race_id"]

    # GNN window edges
    edge_index_dict = edge_cache.get_window_for_race(race_idx)

    # Active participants
    race_results = results_df[results_df["raceId"] == race_id]
    active_driver_ids = torch.tensor(
        race_results["driverId"].astype(int).unique().tolist(), dtype=torch.long
    )
    active_constructor_ids = torch.tensor(
        race_results["constructorId"].astype(int).unique().tolist(), dtype=torch.long
    )

    # Teammate pairs
    teammate_pairs = extract_teammate_pairs(results_df, race_id)

    # Qualifying and grid positions
    qualifying_positions = {}
    grids = {}
    for _, row in race_results.iterrows():
        did = int(row["driverId"])
        qualifying_positions[did] = float(row.get("qualifying_position", 0.0)) if pd.notna(row.get("qualifying_position")) else 0.0
        grids[did] = float(row.get("grid", 0.0)) if pd.notna(row.get("grid")) else 0.0

    return {
        "race_id": race_id,
        "race_idx": race_idx,
        "year": race_meta["year"],
        "round": race_meta["round"],
        "edge_index_dict": edge_index_dict,
        "active_driver_ids": active_driver_ids,
        "active_constructor_ids": active_constructor_ids,
        "teammate_pairs": teammate_pairs,
        "qualifying_positions": qualifying_positions,
        "grids": grids,
    }