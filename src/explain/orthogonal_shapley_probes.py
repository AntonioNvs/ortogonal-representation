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
def constructor_recoverability_career_probe(
  model: OrthogonalShapleyGNN,
  graph_data,
  tf_dict,
  edge_index_dict,
  device: torch.device,
  sample_idx: torch.Tensor | None = None,
  *,
  seed: int = 42,
  null_perm: int = 50,
) -> Dict[str, Any]:
  """Supervised probe on the *career* (car-free) driver embedding.

  The hard-identification Section 1 makes the driver skill the sum of a
  career-shared embedding and a per-season offset. This probe asks whether the
  **career** embedding alone leaks constructor identity. Because the embedding is
  constant across a driver's career, the clean test is restricted to drivers who
  switched teams: for those drivers the embedding cannot vary between teams, so a
  held-out AUC near chance (~0.5) means the career channel is car-free, while an
  AUC well above the null p95 means it still encodes the car.

  The pool is built from the **full** results table (not the 2024-2025 probe
  sample), because the career embedding is constant per driver and we want every
  team-switcher. Rows are aggregated to one (driver, constructor) pair each with
  equal weight, so a driver's multi-season stint at a dominant team does not swamp
  a short stint at a second team. Held-out splits are grouped by **driver**
  (GroupKFold), so a driver's embedding is never seen at train time — the AUC
  measures cross-driver recoverability, not memorisation of driver identity.
  """
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import roc_auc_score
  from sklearn.model_selection import GroupKFold
  from sklearn.preprocessing import StandardScaler

  res = graph_data["results"]
  if model.driver_career is None or not hasattr(res, "driver_career_idx"):
    return {"macro_auc": float("nan"), "null_auc_p95": float("nan"),
            "note": "career embedding not present"}

  valid = (
    res.in_ranking
    & (res.driver_career_idx >= 0)
    & (res.constructor_state_idx >= 0)
    & (res.constructor_id >= 0)
  )
  pos = valid.nonzero(as_tuple=True)[0]
  if pos.numel() < 20:
    return {"macro_auc": float("nan"), "null_auc_p95": float("nan"),
            "n_drivers": 0, "note": "insufficient rows"}

  career_idx = res.driver_career_idx[pos].cpu().numpy()
  c_id = res.constructor_id[pos].cpu().numpy()
  driver_id = res.driver_id[pos].cpu().numpy()

  # Team-switchers only: a career-constant embedding is only a meaningful leak
  # signal for drivers who drove for >=2 constructors.
  df = pd.DataFrame({"driver": driver_id, "constructor": c_id, "career": career_idx})
  n_teams = df.groupby("driver")["constructor"].nunique()
  switchers = n_teams[n_teams >= 2].index
  df = df[df["driver"].isin(switchers)].reset_index(drop=True)

  if df.empty:
    return {"macro_auc": float("nan"), "null_auc_p95": float("nan"),
            "n_drivers": int(switchers.size), "note": "no team-switchers"}

  # Aggregate to one (driver, constructor) pair with equal weight; the career
  # embedding is constant per driver so a single representative index suffices.
  pairs = df.groupby(["driver", "constructor"], as_index=False)["career"].first()
  # Drop constructor classes with <2 drivers (one-vs-rest needs a non-trivial +).
  counts = pairs["constructor"].value_counts()
  pairs = pairs[pairs["constructor"].isin(counts[counts >= 2].index)].reset_index(drop=True)

  n_drivers = int(pairs["driver"].nunique())
  if n_drivers < 5 or pairs["constructor"].nunique() < 2 or len(pairs) < 10:
    return {"macro_auc": float("nan"), "null_auc_p95": float("nan"),
            "n_drivers": n_drivers, "n_pairs": int(len(pairs)),
            "note": "too few team-switchers or pairs"}

  emb = model.driver_career(
    torch.from_numpy(pairs["career"].to_numpy()).to(device)
  ).cpu().numpy()
  X = emb
  y = pairs["constructor"].to_numpy()
  groups = pairs["driver"].to_numpy()

  scaler = StandardScaler().fit(X)
  Xs = scaler.transform(X)

  n_splits = min(5, n_drivers)

  def _macro_auc(yy):
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(Xs, yy, groups=groups):
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
    "n_drivers": n_drivers,
    "n_pairs": int(len(pairs)),
    "n_classes": int(np.unique(y).size),
    "restricted_to_team_switchers": True,
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
  career_recoverability = constructor_recoverability_career_probe(
    model, graph_data, tf_dict, edge_index_dict, device, seed=config.seed
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
    "constructor_recoverability_career": career_recoverability,
    "swap_invariance": swap,
    "shapley_efficiency": efficiency,
    "gates": gates,
    "n_samples": int(sample_idx.numel()),
    "seed": config.seed,
  }
