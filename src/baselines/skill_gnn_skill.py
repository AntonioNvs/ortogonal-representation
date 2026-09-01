"""Load season skill scores from a trained SkillGNN checkpoint."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import pandas as pd
import torch

import config as cfg
import data.tasks as data_tasks
from data.temporal_graph import build_temporal_graph
from models.skill_gnn import SkillGNN
from relbench.base import Database
from relbench.datasets import get_dataset
from utils.device import get_device


def get_skill_gnn_db() -> Database:
    """Database for SkillGNN train/inference — must match ``train_skill_gnn.py``."""
    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    dataset_name = cfg.active_dataset_name()
    return get_dataset(dataset_name, download=False).get_db(upto_test_timestamp=False)


def encoder_sidecar_path(checkpoint_path: str) -> str:
    base, _ = os.path.splitext(checkpoint_path)
    return f"{base}_encoder.pt"


def save_skill_gnn_encoder(
    checkpoint_path: str,
    node_to_col_names_dict: Dict[str, Any],
    node_to_col_stats: Dict[str, Any],
) -> str:
    """Persist encoder schema next to the checkpoint for reload compatibility."""
    path = encoder_sidecar_path(checkpoint_path)
    torch.save(
        {
            "node_to_col_names_dict": node_to_col_names_dict,
            "node_to_col_stats": node_to_col_stats,
        },
        path,
    )
    return path


def load_skill_gnn_model_and_graph(
    db: Database,
    checkpoint_path: str = "output/skill_model/skill_gnn.pth",
    meta_path: str = "output/skill_model/skill_gnn_meta.json",
) -> Tuple[SkillGNN, Any, Dict, Dict, torch.device]:
    """Load SkillGNN weights plus causal graph tensors."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"SkillGNN checkpoint not found at {checkpoint_path}. "
            "Train with: python src/experiments/train_skill_gnn.py --seed 42"
        )

    device = get_device()
    with open(meta_path) as f:
        meta = json.load(f)

    graph_data, node_to_col_names_dict, node_to_col_stats = build_temporal_graph(db)
    sidecar = encoder_sidecar_path(checkpoint_path)
    if os.path.isfile(sidecar):
        enc = torch.load(sidecar, map_location="cpu", weights_only=False)
        node_to_col_names_dict = enc["node_to_col_names_dict"]
        node_to_col_stats = enc["node_to_col_stats"]

    model = SkillGNN(
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
        hidden_dim=meta.get("config", {}).get("hidden_dim", 128),
        num_layers=meta.get("config", {}).get("num_layers", 4),
        grid_weight=meta.get("config", {}).get("grid_weight", 0.05),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    tf_dict = {nt: graph_data[nt].tf.to(device) for nt in graph_data.node_types}
    edge_index_dict = {et: ei.to(device) for et, ei in graph_data.edge_index_dict.items()}
    return model, graph_data, tf_dict, edge_index_dict, device


@torch.no_grad()
def export_race_skills(
    model: SkillGNN,
    graph_data,
    tf_dict,
    edge_index_dict,
    device: torch.device,
    max_year: int = 2025,
) -> pd.DataFrame:
    model.eval()
    x_dict = model.encode(tf_dict, edge_index_dict)

    res = graph_data["results"]
    mask = res.in_ranking & (res.year <= max_year)
    mask = mask & (res.driver_state_idx >= 0) & (res.constructor_state_idx >= 0)
    idx = mask.nonzero(as_tuple=True)[0]

    _, skill = model.race_utilities(
        x_dict,
        res.driver_state_idx[idx].to(device),
        res.constructor_state_idx[idx].to(device),
        res.grid[idx].to(device),
    )

    rows = pd.DataFrame(
        {
            "driverId": res.driver_id[idx].cpu().numpy(),
            "year": res.year[idx].cpu().numpy(),
            "round": res.round[idx].cpu().numpy(),
            "raceId": res.race_id[idx].cpu().numpy(),
            "race_skill": skill.cpu().numpy(),
            "constructorId": res.constructor_id[idx].cpu().numpy(),
        }
    )
    return rows


def season_skill_from_races(race_df: pd.DataFrame) -> pd.DataFrame:
    """One skill score per (driver, season): mean race_skill in that season."""
    agg = (
        race_df.groupby(["driverId", "year"], as_index=False)
        .agg(skill_score=("race_skill", "mean"), n_races=("race_skill", "size"))
        .rename(columns={"year": "season"})
    )
    return agg.sort_values(["driverId", "season"]).reset_index(drop=True)


def load_skill_gnn_skill(
    db: Database | None = None,
    checkpoint_path: str = "output/skill_model/skill_gnn.pth",
    meta_path: str = "output/skill_model/skill_gnn_meta.json",
    max_year: int = 2025,
) -> pd.DataFrame:
    if db is None:
        db = get_skill_gnn_db()
    model, graph_data, tf_dict, edge_index_dict, device = load_skill_gnn_model_and_graph(
        db, checkpoint_path=checkpoint_path, meta_path=meta_path
    )
    race_df = export_race_skills(
        model, graph_data, tf_dict, edge_index_dict, device, max_year=max_year
    )
    return season_skill_from_races(race_df)
