"""Temporal F1 graph with per-season meta-nodes.

The counterfactual branches need a graph in which *time is a first-class
structural element* rather than a recurrent state. This module builds that
graph: instead of one ``driver`` node per person, there is one
``driver_season`` node per (driver, season) pair, linked forward in time by a
``same_driver`` edge. The same holds for constructors. This makes a
counterfactual ("driver X in team Y's car at season T") a literal substitution
of a node in the readout, without retraining and without a recurrent state that
leaks the future into the past.

Node types (all counts derive from the enriched rel-f1 DB):

    driver_season      — one per (driverId, year) that actually raced
    constructor_season — one per (constructorId, year) that actually raced
    circuit            — one per circuitId (static; no season copies)
    race               — one per raceId

Edge types (PyG ``(src, rel, dst)`` triples):

    driver_season -> drives_for -> constructor_season   (modal team of season)
    driver_season -> same_driver -> driver_season       (season T -> T+1, directional)
    constructor_season -> same_constructor -> constructor_season  (T -> T+1)
    driver_season -> raced_in -> race                   (participation; prediction target)
    race -> held_at -> circuit

The prediction target lives on the ``raced_in`` edge: for each (driver_season,
race) the model predicts ``positionOrder / n_racers``. Because a driver may
change team mid-season, the *actual* constructor at each race (not the modal
one) is carried in :attr:`TemporalGraph.raced_in` as an extra column so the
readout can consume the correct ``constructor_season`` embedding per race.

There is intentionally **no engine node**: the enriched Ergast/Jolpica schema
has no engine table (engine supplier is folded into the constructor entry).
See the marginal-attribution design doc for how that affects the Shapley
player set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Canonical order of node types (used to make index assignment deterministic).
NODE_TYPES: tuple[str, ...] = (
    "driver_season",
    "constructor_season",
    "circuit",
    "race",
)

# Static numeric columns exposed per node type (see TemporalGraph.static).
CIRCUIT_FEATURES: tuple[str, ...] = ("lat", "lng", "alt")
RACE_FEATURES: tuple[str, ...] = ("round", "year")


@dataclass
class TemporalGraph:
    """Structural graph + the mapping tables that turn it back into F1 rows.

    ``data`` is a PyG :class:`HeteroData` carrying only *structure*
    (``edge_index_dict`` and per-type ``num_nodes``). Node features are
    deliberately omitted: the model owns learned embeddings for the two
    identity-like node types and small MLP encoders for the static ones. This
    keeps data and parameters cleanly separated.
    """

    data: HeteroData

    # node idx -> original identifiers
    driver_season: pd.DataFrame = field(default_factory=pd.DataFrame)
    constructor_season: pd.DataFrame = field(default_factory=pd.DataFrame)
    circuit: pd.DataFrame = field(default_factory=pd.DataFrame)
    race: pd.DataFrame = field(default_factory=pd.DataFrame)

    # static numeric features per node type, aligned to node idx order.
    static: dict[str, np.ndarray] = field(default_factory=dict)

    # Prediction table: one row per raced_in edge. ``edge_idx`` matches the
    # order of ``data[("driver_season", "raced_in", "race")].edge_index``
    # columns. ``constructor_season`` is the *actual* team at that race.
    raced_in: pd.DataFrame = field(default_factory=pd.DataFrame)

    # id -> node idx fast lookups (built in build()).
    driver_season_idx: dict[tuple[int, int], int] = field(default_factory=dict)
    constructor_season_idx: dict[tuple[int, int], int] = field(default_factory=dict)
    circuit_idx: dict[int, int] = field(default_factory=dict)
    race_idx: dict[int, int] = field(default_factory=dict)

    @property
    def num_driver_seasons(self) -> int:
        return int(self.data["driver_season"].num_nodes)

    @property
    def num_constructor_seasons(self) -> int:
        return int(self.data["constructor_season"].num_nodes)

    @property
    def num_circuits(self) -> int:
        return int(self.data["circuit"].num_nodes)

    @property
    def num_races(self) -> int:
        return int(self.data["race"].num_nodes)


def build_temporal_graph(db, min_year: Optional[int] = None) -> TemporalGraph:
    """Build the temporal meta-node graph from the enriched rel-f1 database.

    Args:
        db: a RelBench :class:`Database` (from ``EnrichedF1Dataset().get_db``).
        min_year: optional lower bound on seasons included.

    Returns:
        A fully-populated :class:`TemporalGraph`.
    """
    results = db.table_dict["results"].df.copy()
    races = db.table_dict["races"].df.copy()
    qualifying = db.table_dict["qualifying"].df.copy()
    circuits = db.table_dict["circuits"].df.copy()

    # --- One table: results joined with season + circuit + qualifying --------
    results = results.merge(
        races[["raceId", "year", "round", "circuitId"]], on="raceId", how="inner"
    )
    results = results.merge(
        qualifying[["raceId", "driverId", "position"]].rename(
            columns={"position": "qualifying_position"}
        ),
        on=["raceId", "driverId"],
        how="left",
    )

    if min_year is not None:
        results = results[results["year"] >= min_year].copy()

    # Deterministic row order (independent of DB load order) so the raced_in
    # edge tensor and the raced_in DataFrame stay aligned index-for-index.
    results = results.sort_values(["year", "round", "driverId"]).reset_index(drop=True)

    # n_racers per race for normalising positionOrder.
    n_racers = results.groupby("raceId")["driverId"].transform("size")
    results["position_norm"] = results["positionOrder"] / n_racers

    # --- Node index assignment (deterministic, sorted) ----------------------
    ds_keys = list(zip(results["driverId"].astype(int), results["year"].astype(int)))
    cs_keys = list(zip(results["constructorId"].astype(int), results["year"].astype(int)))

    ds_uniq = sorted(set(ds_keys))
    cs_uniq = sorted(set(cs_keys))
    circ_uniq = sorted(races["circuitId"].astype(int).unique().tolist())
    race_uniq = sorted(results["raceId"].astype(int).unique().tolist())

    driver_season_idx = {k: i for i, k in enumerate(ds_uniq)}
    constructor_season_idx = {k: i for i, k in enumerate(cs_uniq)}
    circuit_idx = {k: i for i, k in enumerate(circ_uniq)}
    race_idx = {k: i for i, k in enumerate(race_uniq)}

    driver_season = pd.DataFrame(
        [{"node_idx": i, "driverId": k[0], "season": k[1]} for i, k in enumerate(ds_uniq)]
    )
    constructor_season = pd.DataFrame(
        [{"node_idx": i, "constructorId": k[0], "season": k[1]} for i, k in enumerate(cs_uniq)]
    )
    circuit = pd.DataFrame(
        [{"node_idx": i, "circuitId": k} for i, k in enumerate(circ_uniq)]
    )
    race = pd.DataFrame(
        [{"node_idx": i, "raceId": k} for i, k in enumerate(race_uniq)]
    )

    # Map original ids -> node idx per row.
    results["ds_idx"] = results[["driverId", "year"]].apply(
        lambda r: driver_season_idx[(int(r.driverId), int(r.year))], axis=1
    )
    results["cs_idx"] = results[["constructorId", "year"]].apply(
        lambda r: constructor_season_idx[(int(r.constructorId), int(r.year))], axis=1
    )
    results["race_idx"] = results["raceId"].map(race_idx)
    results["circuit_idx"] = results["circuitId"].map(circuit_idx)

    # --- Edges --------------------------------------------------------------
    edge_index_dict: dict[tuple, torch.Tensor] = {}

    # drives_for: modal constructor per (driver, season) -> one edge.
    modal = (
        results.groupby(["driverId", "year"])["cs_idx"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"cs_idx": "modal_cs_idx"})
    )
    modal["ds_idx"] = modal.apply(
        lambda r: driver_season_idx[(int(r.driverId), int(r.year))], axis=1
    )
    drives_for = torch.tensor(
        [modal["ds_idx"].tolist(), modal["modal_cs_idx"].tolist()], dtype=torch.long
    )
    edge_index_dict[("driver_season", "drives_for", "constructor_season")] = drives_for

    # raced_in: every result row (driver_season -> race).
    raced_in = torch.tensor(
        [results["ds_idx"].tolist(), results["race_idx"].tolist()], dtype=torch.long
    )
    edge_index_dict[("driver_season", "raced_in", "race")] = raced_in

    # held_at: race -> circuit (deduplicated).
    race_circuit = results[["race_idx", "circuit_idx"]].drop_duplicates().sort_values("race_idx")
    held_at = torch.tensor(
        [race_circuit["race_idx"].tolist(), race_circuit["circuit_idx"].tolist()],
        dtype=torch.long,
    )
    edge_index_dict[("race", "held_at", "circuit")] = held_at

    # same_driver / same_constructor: season T -> T+1 per entity.
    edge_index_dict[("driver_season", "same_driver", "driver_season")] = _temporal_chain_edges(
        driver_season_idx, driver_season, id_col="driverId"
    )
    edge_index_dict[
        ("constructor_season", "same_constructor", "constructor_season")
    ] = _temporal_chain_edges(
        constructor_season_idx, constructor_season, id_col="constructorId"
    )

    # --- Static features -----------------------------------------------------
    static: dict[str, np.ndarray] = {}
    circuit_feat = circuits.set_index("circuitId").reindex(circ_uniq)
    static["circuit"] = np.nan_to_num(
        circuit_feat[list(CIRCUIT_FEATURES)].to_numpy(dtype=float), nan=0.0
    )
    race_feat = (
        races.set_index("raceId")
        .reindex(race_uniq)[["round", "year"]]
        .to_numpy(dtype=float)
    )
    static["race"] = np.nan_to_num(race_feat, nan=0.0)

    # --- HeteroData assembly ------------------------------------------------
    data = HeteroData()
    data["driver_season"].num_nodes = len(ds_uniq)
    data["constructor_season"].num_nodes = len(cs_uniq)
    data["circuit"].num_nodes = len(circ_uniq)
    data["race"].num_nodes = len(race_uniq)
    for et, ei in edge_index_dict.items():
        data[et].edge_index = ei

    # --- Prediction table (raced_in rows) -----------------------------------
    raced_in_frame = results[
        ["ds_idx", "race_idx", "cs_idx", "position_norm", "positionOrder",
         "grid", "qualifying_position", "year", "circuit_idx"]
    ].copy()
    raced_in_frame = raced_in_frame.reset_index(drop=True)
    raced_in_frame = raced_in_frame.rename(
        columns={
            "ds_idx": "driver_season",
            "race_idx": "race",
            "cs_idx": "constructor_season",
            "circuit_idx": "circuit",
        }
    )

    return TemporalGraph(
        data=data,
        driver_season=driver_season,
        constructor_season=constructor_season,
        circuit=circuit,
        race=race,
        static=static,
        raced_in=raced_in_frame,
        driver_season_idx=driver_season_idx,
        constructor_season_idx=constructor_season_idx,
        circuit_idx=circuit_idx,
        race_idx=race_idx,
    )


def _temporal_chain_edges(idx_map: dict, node_frame: pd.DataFrame, id_col: str) -> torch.Tensor:
    """Build directional T -> T+1 edges for each entity in ``id_col``.

    ``node_frame`` must contain ``node_idx``, ``season`` and ``id_col``. For
    each entity, consecutive seasons are linked ``season_T -> season_{T+1}``.
    """
    src, dst = [], []
    grouped = node_frame.sort_values([id_col, "season"]).groupby(id_col)
    for _, grp in grouped:
        seasons = grp["season"].tolist()
        idxs = grp["node_idx"].tolist()
        for i in range(len(seasons) - 1):
            if seasons[i + 1] == seasons[i] + 1:
                src.append(idxs[i])
                dst.append(idxs[i + 1])
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)
