"""Load and export OrthogonalShapleyGNN skills with coalition Shapley attribution."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import pandas as pd
import torch

import config as cfg
import data.tasks as data_tasks
from data.temporal_graph import build_temporal_graph
from explain.coalition_shapley import CoalitionBaselines, exact_shapley_utilities
from models.orthogonal_shapley_gnn import OrthogonalShapleyGNN
from relbench.base import Database
from relbench.datasets import get_dataset
from utils.device import get_device


def get_orthogonal_shapley_db() -> Database:
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


def save_orthogonal_shapley_encoder(
  checkpoint_path: str,
  node_to_col_names_dict: Dict[str, Any],
  node_to_col_stats: Dict[str, Any],
) -> str:
  path = encoder_sidecar_path(checkpoint_path)
  torch.save(
    {
      "node_to_col_names_dict": node_to_col_names_dict,
      "node_to_col_stats": node_to_col_stats,
    },
    path,
  )
  return path


def load_orthogonal_shapley_model_and_graph(
  db: Database,
  checkpoint_path: str = "output/orthogonal_shapley_model/orthogonal_shapley.pth",
  meta_path: str = "output/orthogonal_shapley_model/orthogonal_shapley_meta.json",
  baselines_path: str | None = None,
) -> Tuple[OrthogonalShapleyGNN, Any, Dict, Dict, torch.device, CoalitionBaselines]:
  if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
      f"OrthogonalShapleyGNN checkpoint not found at {checkpoint_path}. "
      "Train with: python src/experiments/train_orthogonal_shapley_gnn.py --seed 42"
    )

  device = get_device()
  with open(meta_path) as f:
    meta = json.load(f)

  if baselines_path is None:
    baselines_path = meta.get(
      "coalition_baselines_path",
      os.path.join(os.path.dirname(checkpoint_path), "coalition_baselines.json"),
    )
  with open(baselines_path) as f:
    baselines = CoalitionBaselines.from_dict(json.load(f), device)

  graph_data, node_to_col_names_dict, node_to_col_stats = build_temporal_graph(db)
  sidecar = encoder_sidecar_path(checkpoint_path)
  if os.path.isfile(sidecar):
    enc = torch.load(sidecar, map_location="cpu", weights_only=False)
    node_to_col_names_dict = enc["node_to_col_names_dict"]
    node_to_col_stats = enc["node_to_col_stats"]

  config = meta.get("config", {})
  model = OrthogonalShapleyGNN(
    node_to_col_names_dict=node_to_col_names_dict,
    node_to_col_stats=node_to_col_stats,
    hidden_dim=config.get("hidden_dim", 128),
    num_layers=config.get("num_layers", 4),
    mlp_hidden=config.get("mlp_hidden", 128),
    use_additive_readout=config.get("use_additive_readout", False),
    num_drivers=int(getattr(graph_data, "num_drivers", 0)),
  ).to(device)
  model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
  model.eval()

  tf_dict = {nt: graph_data[nt].tf.to(device) for nt in graph_data.node_types}
  edge_index_dict = {et: ei.to(device) for et, ei in graph_data.edge_index_dict.items()}
  return model, graph_data, tf_dict, edge_index_dict, device, baselines


@torch.no_grad()
def export_race_skills(
  model: OrthogonalShapleyGNN,
  graph_data,
  tf_dict,
  edge_index_dict,
  device: torch.device,
  *,
  baselines: CoalitionBaselines,
  max_year: int = 2025,
  name_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
  """Race-level Shapley driver contributions and entity decomposition."""
  model.eval()
  x_dict = model.encode(tf_dict, edge_index_dict)
  res = graph_data["results"]
  mask = res.in_ranking & (res.year <= max_year)
  mask = (
    mask
    & (res.driver_state_idx >= 0)
    & (res.constructor_state_idx >= 0)
    & (res.race_idx >= 0)
  )
  idx = mask.nonzero(as_tuple=True)[0]

  d_idx = res.driver_state_idx[idx].to(device)
  c_idx = res.constructor_state_idx[idx].to(device)
  race_idx = res.race_idx[idx].to(device)
  grid = res.grid[idx].to(device)
  round_num = res.round[idx].to(device)

  d_emb = x_dict["driver_state"][d_idx]
  c_emb = x_dict["constructor_state"][c_idx]
  ctx = model.context_vector(x_dict, race_idx, grid, round_num)

  career_emb = None
  if hasattr(res, "driver_career_idx") and model.driver_career is not None:
    career_emb = model.driver_career(res.driver_career_idx[idx].to(device))

  phi_d, phi_c, phi_x, residual = exact_shapley_utilities(
    model, d_emb, c_emb, ctx, baselines, career_emb
  )

  race_id_cpu = res.race_id[idx].cpu().numpy()

  # Hard within-race centering of the driver *skill score* (Section 2): the
  # skill becomes an intra-race deviation, removing the car/era/circuit level by
  # construction. PL is invariant to a per-race translation, so this is a
  # post-hoc export transform that does not conflict with training. The Shapley
  # channels (contrib_driver/constructor/context) are left un-centered so the
  # attribution still satisfies exact efficiency (sum == v_full − v_empty).
  phi_d_np = phi_d.cpu().numpy()
  phi_d_centered = (
    phi_d_np
    - pd.Series(phi_d_np).groupby(race_id_cpu).transform("mean").to_numpy()
  )

  rows = pd.DataFrame(
    {
      "driverId": res.driver_id[idx].cpu().numpy(),
      "season": res.year[idx].cpu().numpy(),
      "round": res.round[idx].cpu().numpy(),
      "raceId": race_id_cpu,
      "constructorId": res.constructor_id[idx].cpu().numpy(),
      "raw_skill": phi_d_centered,
      "contrib_driver": phi_d_np,
      "contrib_constructor": phi_c.cpu().numpy(),
      "contrib_context": phi_x.cpu().numpy(),
      "contrib_residual": residual.cpu().numpy(),
      "as_of_round": res.round[idx].cpu().numpy(),
    }
  )

  if name_panel is not None:
    meta = name_panel[
      ["driverId", "raceId", "driver_name", "constructor_name", "lineage_id"]
    ].drop_duplicates()
    rows = rows.merge(meta, on=["driverId", "raceId"], how="left")
  else:
    rows["driver_name"] = ""
    rows["constructor_name"] = ""
    rows["lineage_id"] = ""

  return rows


def export_orthogonal_shapley(
  db: Database,
  *,
  checkpoint_path: str = "output/orthogonal_shapley_model/orthogonal_shapley.pth",
  meta_path: str = "output/orthogonal_shapley_model/orthogonal_shapley_meta.json",
  baselines_path: str | None = None,
  max_year: int = 2025,
  inference_mode=None,
) -> "SkillExport":
  from data.race_panel import RacePanelConfig, build_race_panel
  from skill.contract import InferenceMode
  from skill.export import build_skill_export

  inference_mode = inference_mode or InferenceMode.FILTERED
  model, graph_data, tf_dict, edge_index_dict, device, baselines = (
    load_orthogonal_shapley_model_and_graph(
      db,
      checkpoint_path=checkpoint_path,
      meta_path=meta_path,
      baselines_path=baselines_path,
    )
  )
  panel = build_race_panel(db, RacePanelConfig(max_year=max_year))
  race_df = export_race_skills(
    model,
    graph_data,
    tf_dict,
    edge_index_dict,
    device,
    baselines=baselines,
    max_year=max_year,
    name_panel=panel,
  )
  race_df["inference_mode"] = inference_mode.value
  race_df["support_bucket"] = "medium"
  return build_skill_export(
    race_df,
    skill_source="orthogonal_shapley",
    inference_mode=inference_mode,
    max_year=max_year,
    extra_meta={
      "checkpoint_path": checkpoint_path,
      "attribution": "coalition_shapley_exact",
    },
  )


def season_skill_from_races(race_df: pd.DataFrame) -> pd.DataFrame:
  col = "raw_skill" if "raw_skill" in race_df.columns else "race_skill"
  agg = (
    race_df.groupby(["driverId", "season" if "season" in race_df.columns else "year"], as_index=False)
    .agg(skill_score=(col, "mean"), n_races=(col, "size"))
  )
  if "year" in agg.columns:
    agg = agg.rename(columns={"year": "season"})
  return agg.sort_values(["driverId", "season"]).reset_index(drop=True)
