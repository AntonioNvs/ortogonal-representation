"""XAI falsification probes for OrthogonalShapleyGNN."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from baselines.orthogonal_shapley_skill import load_orthogonal_shapley_model_and_graph
from explain.coalition_shapley import exact_shapley_utilities
from explain.skill_gnn_probes import ProbeSampleConfig, evaluate_xai_gates
from models.orthogonal_shapley_gnn import OrthogonalShapleyGNN
from relbench.base import Database


def _race_swap_lookup(graph_data, sample_idx: torch.Tensor) -> Dict[int, int]:
  res = graph_data["results"]
  rows = pd.DataFrame(
    {
      "idx": sample_idx.cpu().numpy(),
      "race_id": res.race_id[sample_idx].cpu().numpy(),
      "constructor_state_idx": res.constructor_state_idx[sample_idx].cpu().numpy(),
    }
  )
  full = pd.DataFrame(
    {
      "idx": np.arange(res.year.shape[0]),
      "race_id": res.race_id.cpu().numpy(),
      "constructor_state_idx": res.constructor_state_idx.cpu().numpy(),
    }
  )
  full = full[full["constructor_state_idx"] >= 0]

  lookup: Dict[int, int] = {}
  for _, row in rows.iterrows():
    same_race = full[
      (full["race_id"] == row["race_id"])
      & (full["constructor_state_idx"] != row["constructor_state_idx"])
    ]
    if same_race.empty:
      continue
    lookup[int(row["idx"])] = int(same_race.iloc[0]["constructor_state_idx"])
  return lookup


def _row_context(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_idx: torch.Tensor,
  device: torch.device,
) -> torch.Tensor:
  return model.context_vector(
    x_dict,
    res.race_idx[row_idx].to(device),
    res.grid[row_idx].to(device),
    res.round[row_idx].to(device),
  )


@torch.no_grad()
def constructor_leakage_probe(
  model: OrthogonalShapleyGNN,
  graph_data,
  tf_dict,
  edge_index_dict,
  device: torch.device,
  baselines,
  sample_idx: torch.Tensor,
) -> float:
  if sample_idx.numel() < 5:
    return float("nan")

  x_dict = model.encode(tf_dict, edge_index_dict)
  res = graph_data["results"]
  d_idx = res.driver_state_idx[sample_idx].to(device)
  c_idx = res.constructor_state_idx[sample_idx].to(device)

  d_emb = x_dict["driver_state"][d_idx]
  c_emb = x_dict["constructor_state"][c_idx]
  ctx = _row_context(model, x_dict, res, sample_idx, device)
  phi_d, _, _, _ = exact_shapley_utilities(model, d_emb, c_emb, ctx, baselines)
  constructor_norm = c_emb.norm(dim=-1).cpu().numpy()
  driver_skill = phi_d.cpu().numpy()

  if np.std(driver_skill) < 1e-9 or np.std(constructor_norm) < 1e-9:
    return float("nan")
  rho, _ = spearmanr(driver_skill, constructor_norm)
  return float(rho)


@torch.no_grad()
def swap_invariance_test(
  model: OrthogonalShapleyGNN,
  graph_data,
  tf_dict,
  edge_index_dict,
  device: torch.device,
  baselines,
  sample_idx: torch.Tensor,
  config: ProbeSampleConfig,
) -> Dict[str, float]:
  if sample_idx.numel() == 0:
    return {"skill_diff": float("nan"), "utility_swap_delta": float("nan"), "n_swaps": 0}

  x_dict = model.encode(tf_dict, edge_index_dict)
  res = graph_data["results"]
  swap_lookup = _race_swap_lookup(graph_data, sample_idx)

  rng = np.random.default_rng(config.seed)
  eligible = [int(i) for i in sample_idx.cpu().numpy() if int(i) in swap_lookup]
  if not eligible:
    return {"skill_diff": float("nan"), "utility_swap_delta": float("nan"), "n_swaps": 0}

  n = min(config.swap_samples, len(eligible))
  chosen = rng.choice(eligible, size=n, replace=False)

  skill_diffs = []
  utility_deltas = []
  for row_idx in chosen:
    row_t = torch.tensor([row_idx], dtype=torch.long)
    alt_c = torch.tensor([swap_lookup[row_idx]], device=device)

    d_idx = res.driver_state_idx[row_t].to(device)
    c_idx = res.constructor_state_idx[row_t].to(device)
    alt_c_t = alt_c
    race_idx = res.race_idx[row_t].to(device)
    grid = res.grid[row_t].to(device)
    round_num = res.round[row_t].to(device)

    d_emb = x_dict["driver_state"][d_idx]
    c_emb = x_dict["constructor_state"][c_idx]
    c_emb_alt = x_dict["constructor_state"][alt_c_t]
    ctx = _row_context(model, x_dict, res, row_t, device)

    phi_d_orig, _, _, _ = exact_shapley_utilities(model, d_emb, c_emb, ctx, baselines)
    phi_d_swap, _, _, _ = exact_shapley_utilities(model, d_emb, c_emb_alt, ctx, baselines)

    u_orig, _, _, _, _ = model.race_utilities(
      x_dict, d_idx, c_idx, race_idx, grid, round_num
    )
    u_swap, _, _, _, _ = model.race_utilities(
      x_dict, d_idx, alt_c_t, race_idx, grid, round_num
    )

    skill_diffs.append(abs(float(phi_d_orig.item() - phi_d_swap.item())))
    utility_deltas.append(abs(float(u_orig.item() - u_swap.item())))

  return {
    "skill_diff": float(np.mean(skill_diffs)) if skill_diffs else float("nan"),
    "utility_swap_delta": float(np.mean(utility_deltas)) if utility_deltas else float("nan"),
    "n_swaps": len(skill_diffs),
  }


@torch.no_grad()
def shapley_efficiency_probe(
  model: OrthogonalShapleyGNN,
  graph_data,
  tf_dict,
  edge_index_dict,
  device: torch.device,
  baselines,
  sample_idx: torch.Tensor,
) -> Dict[str, float]:
  if sample_idx.numel() == 0:
    return {"mean_efficiency_error": float("nan")}

  x_dict = model.encode(tf_dict, edge_index_dict)
  res = graph_data["results"]
  d_idx = res.driver_state_idx[sample_idx].to(device)
  c_idx = res.constructor_state_idx[sample_idx].to(device)

  d_emb = x_dict["driver_state"][d_idx]
  c_emb = x_dict["constructor_state"][c_idx]
  ctx = _row_context(model, x_dict, res, sample_idx, device)
  phi_d, phi_c, phi_x, residual = exact_shapley_utilities(
    model, d_emb, c_emb, ctx, baselines
  )
  return {
    "mean_efficiency_error": float(torch.mean(torch.abs(residual)).item()),
    "mean_driver_share": float(
      torch.mean(phi_d.abs() / (phi_d.abs() + phi_c.abs() + phi_x.abs() + 1e-9)).item()
    ),
  }


@torch.no_grad()
def run_orthogonal_shapley_probes(
  db: Database,
  checkpoint_path: str = "output/orthogonal_shapley_model/orthogonal_shapley.pth",
  meta_path: str = "output/orthogonal_shapley_model/orthogonal_shapley_meta.json",
  baselines_path: str | None = None,
  config: Optional[ProbeSampleConfig] = None,
) -> Dict[str, Any]:
  from explain.skill_gnn_probes import sample_race_rows

  config = config or ProbeSampleConfig()
  model, graph_data, tf_dict, edge_index_dict, device, baselines = (
    load_orthogonal_shapley_model_and_graph(
      db,
      checkpoint_path=checkpoint_path,
      meta_path=meta_path,
      baselines_path=baselines_path,
    )
  )

  sample_idx = sample_race_rows(graph_data, config)
  leakage_rho = constructor_leakage_probe(
    model, graph_data, tf_dict, edge_index_dict, device, baselines, sample_idx
  )
  swap = swap_invariance_test(
    model, graph_data, tf_dict, edge_index_dict, device, baselines, sample_idx, config
  )
  efficiency = shapley_efficiency_probe(
    model, graph_data, tf_dict, edge_index_dict, device, baselines, sample_idx
  )
  gates = evaluate_xai_gates(leakage_rho, swap["skill_diff"])

  return {
    "skill_source": "orthogonal_shapley",
    "constructor_leakage_rho": leakage_rho,
    "swap_invariance": swap,
    "shapley_efficiency": efficiency,
    "gates": gates,
    "n_samples": int(sample_idx.numel()),
    "seed": config.seed,
  }
