"""Kalman-GNN skill adapter for the career-validation framework.

Implements the "skill scorer" contract: a function returning a DataFrame
``[driverId, season, skill_score]`` where ``skill_score(driverId, T)`` is the
driver's scalar skill (``skill_head(v_drivers)``) at the *end* of season T.

The scalar is obtained by replaying the chronological race sequence from the
trained checkpoint, exactly as ``train_kalman`` does, and snapshotting
``compute_skill`` at each season boundary. Because the replay is chronological
and the snapshot uses only data <= T, it is leakage-free w.r.t. the forward
career outcome.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

SRC_DIR = str(Path(__file__).resolve().parents[1])
ROOT_DIR = str(Path(__file__).resolve().parents[2])
for _p in (ROOT_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg
from data.kalman_dataset import (
    ChronologicalRaceList,
    SlidingWindowEdgeCache,
    build_race_batch,
)
from models.kalman_gnn import KalmanGNNPipeline
from train import get_active_task, get_device, load_db_and_graph, set_global_seed

DEFAULT_CHECKPOINT = "output/kalman/kalman_gnn.pth"


def load_kalman_skill(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device=None,
    seed: int = 42,
) -> pd.DataFrame:
    """Replay the trained Kalman-GNN and snapshot per-driver skill per season.

    Args:
        checkpoint_path: path to ``kalman_gnn.pth``.
        device: torch device string (e.g. ``"cuda:7"``) or None to use the
            default from ``cfg``.
        seed: reproducibility (the replay itself is deterministic given the
            checkpoint).

    Returns:
        DataFrame with columns ``[driverId, season, skill_score]``, one row per
        (driver active in that season, season).
    """
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    set_global_seed(seed)

    # --- Data setup (mirrors train_kalman.main) ---
    _, outcome_lookup = get_active_task()
    db, graph_data, node_to_col_names_dict, node_to_col_stats, instances_df, task = (
        load_db_and_graph()
    )

    results_df = db.table_dict["results"].df

    # Merge qualifying position and positionOrder (outcome columns are stripped
    # by the task, so positionOrder is recovered from the pre-task snapshot).
    qual_df = db.table_dict["qualifying"].df[["driverId", "raceId", "position"]].rename(
        columns={"position": "qualifying_position"}
    )
    results_df = results_df.merge(qual_df, on=["driverId", "raceId"], how="left")
    if "positionOrder" in outcome_lookup.columns:
        results_df = results_df.merge(
            outcome_lookup[["resultId", "positionOrder"]], on="resultId", how="left"
        )

    race_list = ChronologicalRaceList(db)
    edge_cache = SlidingWindowEdgeCache(
        graph_data, db, race_list, window_size=cfg.KALMAN_WINDOW_SIZE
    )

    # --- Model (same dims as train_kalman) ---
    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}
    model = KalmanGNNPipeline(
        num_drivers=graph_data["drivers"].num_nodes,
        num_constructors=graph_data["constructors"].num_nodes,
        num_nodes_dict=num_nodes_dict,
        state_dim=cfg.KALMAN_STATE_DIM,
        msg_dim=cfg.KALMAN_MSG_DIM,
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Kalman checkpoint not found: {checkpoint_path}")

    # --- Static features + lazy-init (must precede load_state_dict) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph_data = graph_data.to(device)
        try:
            static_x_dict = model.encode_static_features_nograd(graph_data.tf_dict)
        except Exception:
            static_x_dict = {}
        try:
            full_edge_dict = {
                et: ei.to(device) for et, ei in graph_data.edge_index_dict.items()
            }
            dummy_x = {}
            for nt in graph_data.node_types:
                n = graph_data[nt].num_nodes
                dummy_x[nt] = (
                    static_x_dict[nt] if nt in static_x_dict else torch.randn(n, 1, device=device)
                )
            model.graph_encoder(dummy_x, full_edge_dict)
        except Exception:
            pass

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # The whole framework correlates ``skill_head(v_drivers)`` against forward
    # career outcomes. If the readout is missing from the checkpoint, the
    # ``skill_score`` returned here is the *randomly initialised* head — the
    # correlation would be pure noise and silently plausible. Fail loudly.
    critical = {"skill_head.weight"}
    critical_missing = critical.intersection(missing)
    if critical_missing:
        raise RuntimeError(
            f"[kalman_skill] refusing to score with an untrained readout: "
            f"missing critical parameters {sorted(critical_missing)} in "
            f"{checkpoint_path}. Retrain or point --checkpoint at a run whose "
            f"state_dict contains the skill head."
        )
    if missing or unexpected:
        print(
            f"[kalman_skill] load_state_dict mismatch (non-critical): "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval()

    # --- Replay chronologically, snapshot skill at each season boundary ---
    v_drivers, v_constructors = model.get_initial_state()
    v_drivers = v_drivers.to(device)
    v_constructors = v_constructors.to(device)

    years = race_list.years
    season_last_idx = {}
    for idx, y in enumerate(years):
        season_last_idx[y] = idx  # last assignment wins -> last race of the year

    # Drivers active in each season (same remapped id space as the graph).
    active_drivers_by_season = (
        results_df.merge(
            db.table_dict["races"].df[["raceId", "year"]], on="raceId", how="inner"
        )
        .groupby("year")["driverId"]
        .apply(lambda s: s.astype(int).unique().tolist())
        .to_dict()
    )

    rows = []
    n_races = len(race_list)
    with torch.no_grad():
        for race_idx in range(n_races):
            batch = build_race_batch(race_idx, edge_cache, results_df, race_list)
            edge_dict = {
                et: ei.to(device) for et, ei in batch["edge_index_dict"].items()
            }
            active_drv = batch["active_driver_ids"].to(device)
            active_cons = batch["active_constructor_ids"].to(device)

            v_drivers, v_constructors = model.forward_step(
                v_drivers, v_constructors,
                static_x_dict, edge_dict,
                active_drv, active_cons,
            )

            year = years[race_idx]
            if season_last_idx.get(year) == race_idx:
                skill_drivers, _ = model.compute_skill(v_drivers, v_constructors)
                skill = skill_drivers.squeeze(-1)  # (num_drivers,)
                for did in active_drivers_by_season.get(year, []):
                    rows.append(
                        {
                            "driverId": int(did),
                            "season": int(year),
                            "skill_score": float(skill[did].item()),
                        }
                    )

    return pd.DataFrame(rows).sort_values(["driverId", "season"]).reset_index(drop=True)
