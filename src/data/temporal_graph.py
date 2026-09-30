"""Causal round-state graph builder for the SAGE position regression.

Replaces the static ``make_pkey_fkey_graph`` graph with a graph that is
*temporally causal by construction*: every edge points from an event at an
earlier time to a node at a later time, so no node can ever aggregate its own
future. See ``docs/plans/2026-08-25-sage-position-regression-design.md`` for the
full rationale.

Node types
----------
- ``driver_state``    — one node per ``(driverId, raceId)`` in results ∪ qualifying.
  Input features are the driver's *static* attributes; the temporal evidence
  arrives via message passing.
- ``constructor_state`` — one node per ``(constructorId, raceId)``.
- ``qualifying``      — the target node (features ``number``, ``date``; label
  ``position``, removed from the input features).
- ``results``         — raw race-result evidence (leaf node).
- ``race``            — one per race (features ``year``, ``round``, ...).
- ``circuit``         — static.

Edge types (all directional, causal)
------------------------------------
- ``same_driver`` / ``same_driver_cross``        — driver_state chain within a
  season / across the season boundary.
- ``same_constructor`` / ``same_constructor_cross`` — the team analogue.
- ``result_of_driver`` / ``result_of_constructor`` — ``results@(T,k-1)`` feeds
  the state node of race ``(T,k)``.
- ``race_to_circuit`` — ``race`` aggregates its ``circuit``.
- ``qualifying_to_race`` — the target sees the race's circuit/era context.
- ``driver_state_to_qualifying`` / ``constructor_state_to_qualifying`` — the
  target aggregates its driver's and team's pre-race state.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index
from torch_frame import stype
from torch_frame.data import Dataset as TF_Dataset
from torch_frame.utils import infer_df_stype

from relbench.base import Database

# Static feature columns copied onto the state nodes. Excludes pkey/fkey.
DRIVER_STATE_COLS = ["driverRef", "code", "forename", "surname", "dob", "nationality"]
CONSTRUCTOR_STATE_COLS = ["constructorRef", "name", "nationality"]

# Columns kept as the target node's input features. ``position`` is the label
# and is excluded; ``qualifyId``/``raceId``/``driverId``/``constructorId`` are
# keys and are excluded.
QUALIFYING_INPUT_COLS = ["number", "date"]

# Feature columns per node type (already excludes pkey/fkey). ``races.time`` is
# deliberately omitted: it is a string-of-time-of-day column that ``infer_df_stype``
# can misclassify as timestamp and that can be empty/NaT for many rows — and the
# race node already carries its era signal via ``year`` and ``date``.
RACE_COLS = ["name", "year", "round", "date"]
CIRCUIT_COLS = ["circuitRef", "name", "location", "country", "lat", "lng", "alt"]
RESULTS_COLS = [
    "number", "grid", "position", "positionOrder", "points", "laps",
    "milliseconds", "fastestLap", "rank", "statusId", "date",
]
# ``constructor_results`` is one row per (constructor, race) with aggregated
# points — the correct *team* evidence node (``results`` is per-driver, so it
# cannot serve as a unique team result).
CONSTRUCTOR_RESULTS_COLS = ["points", "date"]


def deduplicate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse Ergast shared-drive duplicate ``(driverId, raceId)`` rows.

    The pristine Ergast source (1950s) can list the same driver twice for one
    race — one row with ``position=NaN`` (shared drive) and one with the actual
    finish.  We keep the row with the best (lowest, non-null) position so every
    ``(driverId, raceId)`` pair is unique across nodes, edges, and PL metadata.
    """
    dup = results.duplicated(subset=["driverId", "raceId"], keep=False)
    if dup.any():
        n_pairs = int(
            results.loc[dup, ["driverId", "raceId"]].drop_duplicates().shape[0]
        )
        print(
            f"[temporal_graph] deduplicated {int(dup.sum())} results rows across "
            f"{n_pairs} (driverId, raceId) pairs (1950s shared-drive); kept best position."
        )
    return (
        results.sort_values("position", ascending=True, na_position="last")
        .drop_duplicates(subset=["driverId", "raceId"], keep="first")
        .reset_index(drop=True)
    )


def deduplicate_constructor_results(constructor_results: pd.DataFrame) -> pd.DataFrame:
    """Defensive dedup for ``(constructorId, raceId)`` (expected unique)."""
    dup = constructor_results.duplicated(subset=["constructorId", "raceId"], keep=False)
    if dup.any():
        print(
            f"[temporal_graph] deduplicated {int(dup.sum())} constructor_results rows "
            f"on (constructorId, raceId); kept first."
        )
    return (
        constructor_results.drop_duplicates(subset=["constructorId", "raceId"], keep="first")
        .reset_index(drop=True)
    )


def _materialize_node(df: pd.DataFrame, col_to_stype: Dict[str, Any]) -> Tuple:
    """Materialize a single node-type DataFrame into a tensor frame + col stats,
    mirroring ``relbench.modeling.graph.make_pkey_fkey_graph``."""
    # Text columns get no embedder in this project; treat them as categoricals.
    for col in list(col_to_stype.keys()):
        if col_to_stype[col] in (stype.text_embedded, stype.text_tokenized):
            col_to_stype[col] = stype.categorical
    dataset = TF_Dataset(df=df, col_to_stype=col_to_stype).materialize()
    return dataset.tensor_frame, dataset.col_stats


def _edge(src: np.ndarray, dst: np.ndarray) -> torch.Tensor:
    """Build a sorted (2, E) edge index from (src, dst) node index arrays."""
    return sort_edge_index(torch.stack([torch.as_tensor(src), torch.as_tensor(dst)]))


def build_temporal_graph(db: Database) -> Tuple[HeteroData, Dict[str, Any], Dict[str, Any]]:
    """Build the causal round-state graph over the full (1950–2026) database.

    Returns ``(data, node_to_col_names_dict, node_to_col_stats)`` where ``data``
    is a :class:`HeteroData` carrying ``.tf`` per node type and the edge index
    dict, plus ``data["qualifying"].year`` / ``.driver_id`` / ``.constructor_id``
    (for the split and baselines) and ``data["qualifying"].y`` (the label).
    """
    drivers = db.table_dict["drivers"].df
    constructors = db.table_dict["constructors"].df
    circuits = db.table_dict["circuits"].df
    races = db.table_dict["races"].df
    results = deduplicate_results(db.table_dict["results"].df)
    constructor_results = deduplicate_constructor_results(
        db.table_dict["constructor_results"].df
    )
    qualifying = db.table_dict["qualifying"].df

    race_meta = races.set_index("raceId")[["year", "round"]]

    # ------------------------------------------------------------------
    # 1. State nodes: one per (entity, race), sorted chronologically.
    # ------------------------------------------------------------------
    driver_pairs = (
        pd.concat([results[["driverId", "raceId"]], qualifying[["driverId", "raceId"]]])
        .drop_duplicates()
        .merge(race_meta, left_on="raceId", right_index=True)
        .sort_values(["driverId", "year", "round"])
        .reset_index(drop=True)
    )
    constructor_pairs = (
        pd.concat(
            [results[["constructorId", "raceId"]], qualifying[["constructorId", "raceId"]]]
        )
        .drop_duplicates()
        .merge(race_meta, left_on="raceId", right_index=True)
        .sort_values(["constructorId", "year", "round"])
        .reset_index(drop=True)
    )

    driver_state_df = driver_pairs[["driverId", "raceId"]].merge(
        drivers[["driverId"] + DRIVER_STATE_COLS], on="driverId", how="left"
    )[DRIVER_STATE_COLS].reset_index(drop=True)

    constructor_state_df = constructor_pairs[["constructorId", "raceId"]].merge(
        constructors[["constructorId"] + CONSTRUCTOR_STATE_COLS], on="constructorId", how="left"
    )[CONSTRUCTOR_STATE_COLS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Static / leaf / target node frames (ordered by pkey == row position).
    # ------------------------------------------------------------------
    race_df = races[RACE_COLS].reset_index(drop=True)
    circuit_df = circuits[CIRCUIT_COLS].reset_index(drop=True)
    results_df = results[RESULTS_COLS].reset_index(drop=True)
    constructor_results_df = constructor_results[CONSTRUCTOR_RESULTS_COLS].reset_index(drop=True)
    qualifying_df = qualifying[QUALIFYING_INPUT_COLS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Materialize every node type.
    # ------------------------------------------------------------------
    data = HeteroData()
    node_to_col_names_dict: Dict[str, Any] = {}
    node_to_col_stats: Dict[str, Any] = {}

    frames = {
        "driver_state": driver_state_df,
        "constructor_state": constructor_state_df,
        "race": race_df,
        "circuit": circuit_df,
        "results": results_df,
        "constructor_results": constructor_results_df,
        "qualifying": qualifying_df,
    }
    for node_type, df in frames.items():
        col_to_stype = infer_df_stype(df)
        tf, col_stats = _materialize_node(df, col_to_stype)
        data[node_type].tf = tf
        data[node_type].num_nodes = len(df)
        node_to_col_names_dict[node_type] = tf.col_names_dict
        node_to_col_stats[node_type] = col_stats

    # ------------------------------------------------------------------
    # 4. Temporal edges within each entity's career (vectorized).
    # ------------------------------------------------------------------
    did = driver_pairs["driverId"].to_numpy()
    dyr = driver_pairs["year"].to_numpy()
    dsame = did[1:] == did[:-1]
    dsrc = np.flatnonzero(dsame)
    ddst = dsrc + 1
    dcross = dyr[ddst] != dyr[dsrc]
    data["driver_state", "same_driver", "driver_state"].edge_index = _edge(
        dsrc[~dcross], ddst[~dcross]
    )
    data["driver_state", "same_driver_cross", "driver_state"].edge_index = _edge(
        dsrc[dcross], ddst[dcross]
    )

    cid = constructor_pairs["constructorId"].to_numpy()
    cyr = constructor_pairs["year"].to_numpy()
    csame = cid[1:] == cid[:-1]
    csrc = np.flatnonzero(csame)
    cdst = csrc + 1
    ccross = cyr[cdst] != cyr[csrc]
    data["constructor_state", "same_constructor", "constructor_state"].edge_index = _edge(
        csrc[~ccross], cdst[~ccross]
    )
    data["constructor_state", "same_constructor_cross", "constructor_state"].edge_index = _edge(
        csrc[ccross], cdst[ccross]
    )

    # ------------------------------------------------------------------
    # 5. Evidence edges: results@(T,k-1) -> state@(T,k).
    # Node index = row position in the deduplicated ``results`` frame (not the
    # raw ``resultId`` PK, which has gaps after dedup).
    # ------------------------------------------------------------------
    result_id_map = pd.Series(
        np.arange(len(results), dtype=np.int64),
        index=pd.MultiIndex.from_frame(results[["driverId", "raceId"]]),
    )
    dprev_race = np.full(len(driver_pairs), -1, dtype=np.int64)
    dprev_race[ddst] = driver_pairs["raceId"].to_numpy()[dsrc]
    dvalid = dprev_race >= 0
    dlookup = pd.MultiIndex.from_arrays(
        [driver_pairs["driverId"].to_numpy()[dvalid], dprev_race[dvalid]],
        names=["driverId", "raceId"],
    )
    dprev_result = result_id_map.reindex(dlookup).fillna(-1).astype(np.int64).to_numpy()
    dhas_prev = dprev_result >= 0
    data["results", "result_of_driver", "driver_state"].edge_index = _edge(
        dprev_result[dhas_prev], np.flatnonzero(dvalid)[dhas_prev]
    )

    # Same for constructor_results: node index = row position after dedup.
    cresult_id_map = pd.Series(
        np.arange(len(constructor_results), dtype=np.int64),
        index=pd.MultiIndex.from_frame(constructor_results[["constructorId", "raceId"]]),
    )
    cprev_race = np.full(len(constructor_pairs), -1, dtype=np.int64)
    cprev_race[cdst] = constructor_pairs["raceId"].to_numpy()[csrc]
    cvalid = cprev_race >= 0
    clookup = pd.MultiIndex.from_arrays(
        [constructor_pairs["constructorId"].to_numpy()[cvalid], cprev_race[cvalid]],
        names=["constructorId", "raceId"],
    )
    cprev_result = cresult_id_map.reindex(clookup).fillna(-1).astype(np.int64).to_numpy()
    chas_prev = cprev_result >= 0
    data["constructor_results", "result_of_constructor", "constructor_state"].edge_index = _edge(
        cprev_result[chas_prev], np.flatnonzero(cvalid)[chas_prev]
    )

    # ------------------------------------------------------------------
    # 6. Context edges: circuit -> race -> qualifying. The destination
    #    aggregates the source, so ``race`` aggregates its ``circuit`` and the
    #    target ``qualifying`` node aggregates ``race`` (which carries the
    #    circuit + era context after one SAGE layer).
    # ------------------------------------------------------------------
    race_circuit = races["circuitId"].to_numpy()
    data["circuit", "circuit_to_race", "race"].edge_index = _edge(
        race_circuit, np.arange(len(races))
    )
    qual_race = qualifying["raceId"].to_numpy()
    data["race", "race_to_qualifying", "qualifying"].edge_index = _edge(
        qual_race, np.arange(len(qualifying))
    )

    # ------------------------------------------------------------------
    # 7. Context edges: state -> qualifying (target aggregates state).
    # ------------------------------------------------------------------
    driver_state_map = pd.Series(
        driver_pairs.index.to_numpy(),
        index=pd.MultiIndex.from_frame(driver_pairs[["driverId", "raceId"]]),
    )
    qual_driver_state = driver_state_map.reindex(
        pd.MultiIndex.from_frame(qualifying[["driverId", "raceId"]])
    ).fillna(-1).astype(np.int64).to_numpy()
    qual_d_ok = qual_driver_state >= 0
    data["driver_state", "driver_state_to_qualifying", "qualifying"].edge_index = _edge(
        qual_driver_state[qual_d_ok], np.flatnonzero(qual_d_ok)
    )

    constructor_state_map = pd.Series(
        constructor_pairs.index.to_numpy(),
        index=pd.MultiIndex.from_frame(constructor_pairs[["constructorId", "raceId"]]),
    )
    qual_constructor_state = constructor_state_map.reindex(
        pd.MultiIndex.from_frame(qualifying[["constructorId", "raceId"]])
    ).fillna(-1).astype(np.int64).to_numpy()
    qual_c_ok = qual_constructor_state >= 0
    data["constructor_state", "constructor_state_to_qualifying", "qualifying"].edge_index = _edge(
        qual_constructor_state[qual_c_ok], np.flatnonzero(qual_c_ok)
    )

    # ------------------------------------------------------------------
    # 8. Attach per-target-node year/round (split + trailing-mean key), label,
    #    and entity ids.
    # ------------------------------------------------------------------
    qual_year = qualifying["raceId"].map(race_meta["year"]).to_numpy()
    qual_round = qualifying["raceId"].map(race_meta["round"]).to_numpy()

    # Normalized target: pole (position 1) -> 0, back of the grid -> 1, using
    # the *session* size (number of drivers who set a time in that race) as the
    # grid. This makes the target invariant to era-dependent grid sizes (22+ cars
    # in the 1950s, 20 today): a "5th place" then means the same skill share in
    # any era. ``y`` is the [0,1] label; raw position is not attached.
    position = qualifying["position"].to_numpy(dtype=np.float64)
    n_cars = (
        qualifying.groupby("raceId")["qualifyId"]
        .transform("size")
        .to_numpy(dtype=np.float64)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        y_norm = (position - 1.0) / (n_cars - 1.0)
    y_norm = np.where(n_cars > 1, y_norm, 0.0)

    # Surface impossible positions: any row whose position exceeds its session
    # size lands outside [0,1] and is almost certainly an enrichment artifact.
    out_of_range = (y_norm < 0.0) | (y_norm > 1.0)
    if out_of_range.any():
        n_bad = int(out_of_range.sum())
        bad_idx = np.flatnonzero(out_of_range)
        bad = qualifying.iloc[bad_idx[:5]][["qualifyId", "raceId", "driverId", "position"]]
        print(
            f"[temporal_graph] WARNING: {n_bad} qualifying rows have position outside "
            f"[1, session_size] (pos range {position[out_of_range].min():.0f}.."
            f"{position[out_of_range].max():.0f}). sample: {bad.to_dict('records')}"
        )

    data["qualifying"].year = torch.from_numpy(qual_year.astype(np.int64))
    data["qualifying"].round = torch.from_numpy(qual_round.astype(np.int64))
    data["qualifying"].y = torch.from_numpy(y_norm.astype(np.float32))
    data["qualifying"].driver_id = torch.from_numpy(qualifying["driverId"].to_numpy(dtype=np.int64))
    data["qualifying"].constructor_id = torch.from_numpy(
        qualifying["constructorId"].to_numpy(dtype=np.int64)
    )

    # ------------------------------------------------------------------
    # 9. Results metadata for race-ranking skill GNN (Plackett-Luce target).
    # ------------------------------------------------------------------
    results_with_meta = results.merge(
        race_meta, left_on="raceId", right_index=True
    )
    result_driver_state = driver_state_map.reindex(
        pd.MultiIndex.from_frame(results_with_meta[["driverId", "raceId"]])
    ).fillna(-1).astype(np.int64).to_numpy()
    result_constructor_state = constructor_state_map.reindex(
        pd.MultiIndex.from_frame(results_with_meta[["constructorId", "raceId"]])
    ).fillna(-1).astype(np.int64).to_numpy()

    position_res = results_with_meta["position"].to_numpy(dtype=np.float64)
    in_ranking = ~np.isnan(position_res)

    race_id_to_idx = pd.Series(
        np.arange(len(races), dtype=np.int64),
        index=races["raceId"].to_numpy(),
    )
    result_race_idx = (
        race_id_to_idx.reindex(results_with_meta["raceId"].to_numpy())
        .fillna(-1)
        .astype(np.int64)
        .to_numpy()
    )

    data["results"].year = torch.from_numpy(results_with_meta["year"].to_numpy(dtype=np.int64))
    data["results"].round = torch.from_numpy(results_with_meta["round"].to_numpy(dtype=np.int64))
    data["results"].race_id = torch.from_numpy(results_with_meta["raceId"].to_numpy(dtype=np.int64))
    data["results"].driver_id = torch.from_numpy(
        results_with_meta["driverId"].to_numpy(dtype=np.int64)
    )
    # Contiguous per-driver index for the career-shared embedding (hard
    # identification): one slot per unique driverId, stable across a career.
    driver_ids_sorted = sorted(int(d) for d in drivers["driverId"].unique())
    driver_to_career = {d: i for i, d in enumerate(driver_ids_sorted)}
    data.num_drivers = len(driver_ids_sorted)
    data["results"].driver_career_idx = torch.from_numpy(
        results_with_meta["driverId"]
        .map(driver_to_career)
        .to_numpy(dtype=np.int64)
    )
    data["results"].constructor_id = torch.from_numpy(
        results_with_meta["constructorId"].to_numpy(dtype=np.int64)
    )
    data["results"].position = torch.from_numpy(position_res.astype(np.float32))
    data["results"].in_ranking = torch.from_numpy(in_ranking)
    data["results"].driver_state_idx = torch.from_numpy(result_driver_state)
    data["results"].constructor_state_idx = torch.from_numpy(result_constructor_state)
    data["results"].race_idx = torch.from_numpy(result_race_idx)
    grid = results_with_meta["grid"].to_numpy(dtype=np.float64)
    data["results"].grid = torch.from_numpy(np.where(np.isnan(grid), 20.0, grid).astype(np.float32))

    data.validate()
    return data, node_to_col_names_dict, node_to_col_stats
