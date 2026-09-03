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
def constructor_recoverability_probe(
  model: OrthogonalShapleyGNN,
  graph_data,
  tf_dict,
  edge_index_dict,
  device: torch.device,
  sample_idx: torch.Tensor,
  *,
  seed: int = 42,
  null_perm: int = 50,
) -> Dict[str, Any]:
  """Supervised probe: can constructor identity be recovered from driver state?

  Fits a one-vs-rest linear probe predicting ``constructorId`` from the driver's
  season-long ``driver_state`` embedding (deduplicated by ``driver_state_idx``).
  Reports held-out macro-AUC plus a null distribution (permuted constructor
  labels) so the gate is set against an empirical chance level, not an arbitrary
  correlation threshold.

  This is the honest falsification: leakage means "the car's identity is
  reconstructible from the driver channel alone."
  """
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import roc_auc_score
  from sklearn.model_selection import StratifiedKFold
  from sklearn.preprocessing import StandardScaler

  if sample_idx.numel() < 20:
    return {"macro_auc": float("nan"), "null_auc_p95": float("nan"),
            "n_states": int(sample_idx.numel()), "note": "insufficient rows"}

  x_dict = model.encode(tf_dict, edge_index_dict)
  res = graph_data["results"]
  d_idx = res.driver_state_idx[sample_idx]
  c_id = res.constructor_id[sample_idx]

  emb = x_dict["driver_state"][d_idx].cpu().numpy()
  labels = c_id.cpu().numpy()

  # Deduplicate by driver_state_idx (season-long state; avoid double-counting rows).
  states = res.driver_state_idx[sample_idx].cpu().numpy()
  df = pd.DataFrame({"state": states, "label": labels, "emb": list(emb)})
  df = df.drop_duplicates(subset=["state"]).reset_index(drop=True)
  X = np.vstack(df["emb"].to_numpy())
  y = df["label"].to_numpy()
  # Drop constructor classes with <2 states (not evaluable one-vs-rest).
  counts = pd.Series(y).value_counts()
  keep = pd.Series(y).isin(counts[counts >= 2].index).to_numpy()
  X, y = X[keep], y[keep]

  if len(np.unique(y)) < 2 or len(y) < 20:
    return {"macro_auc": float("nan"), "null_auc_p95": float("nan"),
            "n_states": int(len(y)), "note": "too few separable classes"}

  scaler = StandardScaler().fit(X)
  Xs = scaler.transform(X)

  def _macro_auc(yy):
    # Stratified 5-fold, one-vs-rest macro-AUC.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    classes = np.unique(yy)
    for tr, te in skf.split(Xs, yy):
      clf = LogisticRegression(max_iter=1000, C=1.0)
      clf.fit(Xs[tr], yy[tr])
      probs = clf.predict_proba(Xs[te])
      for k, c in enumerate(clf.classes_):
        ybin = (yy[te] == c).astype(int)
        if ybin.sum() == 0 or ybin.sum() == len(te):
          continue
        aucs.append(roc_auc_score(ybin, probs[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")

  macro_auc = _macro_auc(y)

  # Null distribution: permute constructor labels, recompute macro-AUC.
  rng = np.random.default_rng(seed)
  null_aucs = []
  for _ in range(null_perm):
    yp = rng.permutation(y)
    a = _macro_auc(yp)
    if not np.isnan(a):
      null_aucs.append(a)
  null_p95 = float(np.quantile(null_aucs, 0.95)) if null_aucs else float("nan")

  return {
    "macro_auc": macro_auc,
    "null_auc_mean": float(np.mean(null_aucs)) if null_aucs else float("nan"),
    "null_auc_p95": null_p95,
    "null_aucs": [float(a) for a in null_aucs],
    "leakage": bool(not np.isnan(macro_auc) and not np.isnan(null_p95) and macro_auc > null_p95),
    "n_states": int(len(y)),
    "n_classes": int(np.unique(y).size),
  }


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
  recoverability = constructor_recoverability_probe(
    model, graph_data, tf_dict, edge_index_dict, device, sample_idx, seed=config.seed
  )
  swap = swap_invariance_test(
    model, graph_data, tf_dict, edge_index_dict, device, baselines, sample_idx, config
  )
  efficiency = shapley_efficiency_probe(
    model, graph_data, tf_dict, edge_index_dict, device, baselines, sample_idx
  )
  gates = evaluate_xai_gates(leakage_rho, swap["skill_diff"], recoverability)

  return {
    "skill_source": "orthogonal_shapley",
    "constructor_leakage_rho": leakage_rho,
    "constructor_recoverability": recoverability,
    "swap_invariance": swap,
    "shapley_efficiency": efficiency,
    "gates": gates,
    "n_samples": int(sample_idx.numel()),
    "seed": config.seed,
  }
