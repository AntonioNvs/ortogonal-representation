"""Train OrthogonalShapleyGNN with Plackett-Luce ranking loss."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath("src"))

import config as cfg
import data.tasks as data_tasks
from baselines.orthogonal_shapley_skill import (
  export_race_skills,
  save_orthogonal_shapley_encoder,
  season_skill_from_races,
)
from data.temporal_graph import build_temporal_graph
from explain.coalition_shapley import (
  CoalitionBaselines,
  attribution_balance_loss,
  compute_train_baselines,
  exact_shapley_utilities,
)
from models.orthogonal_shapley_gnn import ARCH_VERSION, OrthogonalShapleyGNN
from models.ranking_likelihood import (
  batch_pairwise_ranking_loss,
  batch_pl_nll,
)
from relbench.datasets import get_dataset
from utils.device import get_device


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def year_mask(years: torch.Tensor, allowed) -> torch.Tensor:
  allowed = set(allowed)
  return torch.tensor([int(y) in allowed for y in years.tolist()], dtype=torch.bool)


def _collect_race_batches(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_mask: torch.Tensor,
  device: torch.device,
) -> Tuple[
  List[torch.Tensor],
  List[torch.Tensor],
  List[torch.Tensor],
  List[torch.Tensor],
  List[torch.Tensor],
]:
  """Per-race utility lists for fused, aux heads, and ranks."""
  idx = row_mask.nonzero(as_tuple=True)[0]
  if idx.numel() == 0:
    return [], [], [], [], []

  race_ids = res.race_id[idx].to(device)
  positions = res.position[idx].to(device)
  driver_state_idx = res.driver_state_idx[idx].to(device)
  constructor_state_idx = res.constructor_state_idx[idx].to(device)
  race_idx = res.race_idx[idx].to(device)
  grid = res.grid[idx].to(device)
  round_num = res.round[idx].to(device)
  driver_career_idx = (
    res.driver_career_idx[idx].to(device)
    if hasattr(res, "driver_career_idx")
    else None
  )

  fused_list: List[torch.Tensor] = []
  driver_list: List[torch.Tensor] = []
  cons_list: List[torch.Tensor] = []
  ctx_list: List[torch.Tensor] = []
  ranks_list: List[torch.Tensor] = []

  for rid in torch.unique(race_ids):
    rmask = race_ids == rid
    u_fused, u_d, u_c, u_x, _ctx = model.race_utilities(
      x_dict,
      driver_state_idx[rmask],
      constructor_state_idx[rmask],
      race_idx[rmask],
      grid[rmask],
      round_num[rmask],
      driver_career_idx[rmask] if driver_career_idx is not None else None,
    )
    fused_list.append(u_fused)
    driver_list.append(u_d)
    cons_list.append(u_c)
    ctx_list.append(u_x)
    ranks_list.append(positions[rmask])

  return fused_list, driver_list, cons_list, ctx_list, ranks_list


def race_loss_for_mask(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_mask: torch.Tensor,
  device: torch.device,
  *,
  aux_driver_weight: float = 0.5,
  aux_constructor_weight: float = 0.75,
  lambda_ctx_aux: float = 0.25,
  lambda_pair: float = 0.25,
  baselines: CoalitionBaselines | None = None,
  lambda_attr: float = 0.1,
  target_driver_share: float = 0.38,
  target_constructor_share: float = 0.30,
  attr_subsample_frac: float = 0.2,
  attr_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Returns (total_loss, pl_loss, orth_loss, attr_loss)."""
  idx = row_mask.nonzero(as_tuple=True)[0]
  zero = torch.tensor(0.0, device=device)
  if idx.numel() == 0:
    return zero, zero, zero, zero

  fused_list, driver_list, cons_list, ctx_list, ranks_list = (
    _collect_race_batches(model, x_dict, res, row_mask, device)
  )

  pl_fused = batch_pl_nll(fused_list, ranks_list)
  pl_driver = batch_pl_nll(driver_list, ranks_list)
  pl_cons = batch_pl_nll(cons_list, ranks_list)
  pl_ctx = batch_pl_nll(ctx_list, ranks_list)
  pl_total = (
    pl_fused
    + aux_driver_weight * pl_driver
    + aux_constructor_weight * pl_cons
    + lambda_ctx_aux * pl_ctx
  )

  pair_loss = batch_pairwise_ranking_loss(fused_list, ranks_list)

  driver_state_idx = res.driver_state_idx[idx].to(device)
  constructor_state_idx = res.constructor_state_idx[idx].to(device)
  race_idx = res.race_idx[idx].to(device)
  grid = res.grid[idx].to(device)
  round_num = res.round[idx].to(device)
  driver_career_idx = (
    res.driver_career_idx[idx].to(device)
    if hasattr(res, "driver_career_idx")
    else None
  )

  orth = model.paired_orthogonal_loss(
    x_dict,
    driver_state_idx,
    constructor_state_idx,
    race_idx,
    grid,
    round_num,
  )

  attr_loss = zero
  if baselines is not None and lambda_attr > 0 and fused_list:
    n_races = len(fused_list)
    n_sample = max(1, int(n_races * attr_subsample_frac))
    rng = np.random.default_rng(attr_seed)
    chosen = rng.choice(n_races, size=min(n_sample, n_races), replace=False)

    phi_d_parts: List[torch.Tensor] = []
    phi_c_parts: List[torch.Tensor] = []
    phi_x_parts: List[torch.Tensor] = []
    race_ids = res.race_id[idx].to(device)
    unique_rids = torch.unique(race_ids)
    for ci in chosen:
      rid = unique_rids[ci]
      rmask = race_ids == rid
      d_emb = x_dict["driver_state"][driver_state_idx[rmask]]
      c_emb = x_dict["constructor_state"][constructor_state_idx[rmask]]
      ctx = model.context_vector(
        x_dict,
        race_idx[rmask],
        grid[rmask],
        round_num[rmask],
      )
      career_emb = (
        model.driver_career(driver_career_idx[rmask])
        if driver_career_idx is not None
        else None
      )
      phi_d, phi_c, phi_x, _ = exact_shapley_utilities(
        model, d_emb, c_emb, ctx, baselines, career_emb
      )
      phi_d_parts.append(phi_d)
      phi_c_parts.append(phi_c)
      phi_x_parts.append(phi_x)

    if phi_d_parts:
      attr_loss = attribution_balance_loss(
        torch.cat(phi_d_parts),
        torch.cat(phi_c_parts),
        torch.cat(phi_x_parts),
        target_driver_share=target_driver_share,
        target_constructor_share=target_constructor_share,
      )

  total = pl_total + lambda_pair * pair_loss + lambda_attr * attr_loss
  return total, pl_total, orth, attr_loss


def pairwise_accuracy(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_mask: torch.Tensor,
  device: torch.device,
) -> float:
  idx = row_mask.nonzero(as_tuple=True)[0]
  if idx.numel() == 0:
    return float("nan")

  correct = 0
  total = 0
  race_ids = res.race_id[idx].to(device)
  positions = res.position[idx].to(device)
  driver_state_idx = res.driver_state_idx[idx].to(device)
  constructor_state_idx = res.constructor_state_idx[idx].to(device)
  race_idx = res.race_idx[idx].to(device)
  grid = res.grid[idx].to(device)
  round_num = res.round[idx].to(device)
  driver_career_idx = (
    res.driver_career_idx[idx].to(device)
    if hasattr(res, "driver_career_idx")
    else None
  )

  for rid in torch.unique(race_ids):
    rmask = race_ids == rid
    u_fused, _, _, _, _ = model.race_utilities(
      x_dict,
      driver_state_idx[rmask],
      constructor_state_idx[rmask],
      race_idx[rmask],
      grid[rmask],
      round_num[rmask],
      driver_career_idx[rmask] if driver_career_idx is not None else None,
    )
    pos = positions[rmask]
    n = u_fused.numel()
    for i in range(n):
      for j in range(i + 1, n):
        if pos[i] == pos[j]:
          continue
        total += 1
        if (pos[i] < pos[j] and u_fused[i] > u_fused[j]) or (
          pos[j] < pos[i] and u_fused[j] > u_fused[i]
        ):
          correct += 1
  return correct / max(total, 1)


def mean_attribution_shares(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_mask: torch.Tensor,
  device: torch.device,
  baselines: CoalitionBaselines,
  *,
  max_races: int = 50,
  seed: int = 0,
) -> Tuple[float, float, float]:
  """Mean absolute Shapley share on a subsample of races."""
  idx = row_mask.nonzero(as_tuple=True)[0]
  if idx.numel() == 0:
    return float("nan"), float("nan"), float("nan")

  race_ids_cpu = res.race_id[idx]
  unique_rids = torch.unique(race_ids_cpu)
  n = min(max_races, unique_rids.numel())
  rng = np.random.default_rng(seed)
  chosen = rng.choice(unique_rids.cpu().numpy(), size=n, replace=False)

  shares_d: List[float] = []
  shares_c: List[float] = []
  shares_x: List[float] = []
  for rid in chosen:
    row_idx = idx[race_ids_cpu.numpy() == rid]
    d_emb = x_dict["driver_state"][res.driver_state_idx[row_idx].to(device)]
    c_emb = x_dict["constructor_state"][res.constructor_state_idx[row_idx].to(device)]
    ctx = model.context_vector(
      x_dict,
      res.race_idx[row_idx].to(device),
      res.grid[row_idx].to(device),
      res.round[row_idx].to(device),
    )
    career_emb = None
    if hasattr(res, "driver_career_idx") and model.driver_career is not None:
      career_emb = model.driver_career(res.driver_career_idx[row_idx].to(device))
    phi_d, phi_c, phi_x, _ = exact_shapley_utilities(
      model, d_emb, c_emb, ctx, baselines, career_emb
    )
    denom = phi_d.abs() + phi_c.abs() + phi_x.abs() + 1e-9
    shares_d.append(float(torch.mean(phi_d.abs() / denom).item()))
    shares_c.append(float(torch.mean(phi_c.abs() / denom).item()))
    shares_x.append(float(torch.mean(phi_x.abs() / denom).item()))

  return float(np.mean(shares_d)), float(np.mean(shares_c)), float(np.mean(shares_x))


def top3_auroc(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_mask: torch.Tensor,
  device: torch.device,
) -> float:
  """Diagnostic AUROC for top-3 finish (not used for model selection)."""
  idx = row_mask.nonzero(as_tuple=True)[0]
  if idx.numel() < 10:
    return float("nan")

  driver_state_idx = res.driver_state_idx[idx].to(device)
  constructor_state_idx = res.constructor_state_idx[idx].to(device)
  race_idx = res.race_idx[idx].to(device)
  grid = res.grid[idx].to(device)
  round_num = res.round[idx].to(device)
  positions = res.position[idx].cpu().numpy()
  driver_career_idx = (
    res.driver_career_idx[idx].to(device)
    if hasattr(res, "driver_career_idx")
    else None
  )

  u_fused, _, _, _, _ = model.race_utilities(
    x_dict, driver_state_idx, constructor_state_idx, race_idx, grid, round_num,
    driver_career_idx,
  )
  scores = u_fused.detach().cpu().numpy()
  labels = (positions <= 3).astype(np.int32)
  if labels.sum() == 0 or labels.sum() == len(labels):
    return float("nan")
  finite = np.isfinite(scores)
  if finite.sum() < 10:
    return float("nan")
  labels = labels[finite]
  scores = scores[finite]
  if labels.sum() == 0 or labels.sum() == len(labels):
    return float("nan")
  return float(roc_auc_score(labels, scores))


def offset_frac_diagnostic(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_mask: torch.Tensor,
  device: torch.device,
) -> float:
  """std(u_season) / (std(u_career) + std(u_season)) over the given rows.

  High value means the per-race offset still dominates the driver identity;
  shrinkage should push it down toward the car-free career channel.
  """
  idx = row_mask.nonzero(as_tuple=True)[0]
  if idx.numel() == 0:
    return float("nan")
  d_emb = x_dict["driver_state"][res.driver_state_idx[idx].to(device)]
  u_season = model.aux_driver_season(d_emb).squeeze(-1)
  if model.driver_career is None or not hasattr(res, "driver_career_idx"):
    return float("nan")
  career_emb = model.driver_career(res.driver_career_idx[idx].to(device))
  u_career = model.aux_driver_career(career_emb).squeeze(-1)
  denom = u_career.std() + u_season.std() + 1e-9
  return float((u_season.std() / denom).item())


def orth_lambda_at_epoch(epoch: int, target: float, warmup_epochs: int) -> float:
  """Linear warmup for orthogonal penalty (avoids early gradient domination)."""
  if warmup_epochs <= 0:
    return target
  return target * min(1.0, (epoch + 1) / warmup_epochs)


def composite_val_score(val_pl: float, val_acc: float) -> float:
  """Lower is better: balances PL NLL and pairwise accuracy."""
  acc = val_acc if np.isfinite(val_acc) else 0.0
  return val_pl + 0.5 * (1.0 - acc)


def train_one_config(
  *,
  lambda_orth: float,
  aux_driver_weight: float,
  aux_constructor_weight: float,
  lambda_ctx_aux: float,
  lambda_pair: float,
  lambda_attr: float,
  target_driver_share: float,
  target_constructor_share: float,
  use_additive_readout: bool,
  hidden_dim: int,
  num_layers: int,
  mlp_hidden: int,
  epochs: int,
  lr: float,
  weight_decay: float,
  patience: int,
  seed: int,
  device: torch.device,
  output_dir: str,
  smoke_test: bool = False,
  max_grad_norm: float = 1.0,
  orth_warmup_epochs: int = 10,
  lambda_rw: float = 0.5,
  lambda_shrink: float = 0.05,
  lambda_quali: float = 0.0,
) -> Dict:
  set_seed(seed)

  data_tasks.register_all(
    enriched_db_dir=cfg.ENRICHED_DB_DIR,
    min_year=cfg.MIN_YEAR,
    max_year=cfg.MAX_YEAR,
    val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
    test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
  )
  dataset_name = cfg.active_dataset_name()
  print(f"-> Loading database {dataset_name} ...")
  db = get_dataset(dataset_name, download=False).get_db(upto_test_timestamp=False)

  print("-> Building causal round-state graph ...")
  graph_data, node_to_col_names_dict, node_to_col_stats = build_temporal_graph(db)

  res = graph_data["results"]
  base_mask = (
    res.in_ranking
    & (res.driver_state_idx >= 0)
    & (res.constructor_state_idx >= 0)
    & (res.race_idx >= 0)
  )
  train_mask = base_mask & year_mask(res.year, cfg.TRAIN_YEARS)
  val_mask = base_mask & year_mask(res.year, cfg.VAL_YEARS)
  test_mask = base_mask & year_mask(res.year, cfg.TEST_YEARS)

  if smoke_test:
    epochs = min(epochs, 3)
    patience = 2
    lambda_attr = 0.0

  model = OrthogonalShapleyGNN(
    node_to_col_names_dict=node_to_col_names_dict,
    node_to_col_stats=node_to_col_stats,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    mlp_hidden=mlp_hidden,
    use_additive_readout=use_additive_readout,
    num_drivers=int(getattr(graph_data, "num_drivers", 0)),
  ).to(device)

  tf_dict = {nt: graph_data[nt].tf.to(device) for nt in graph_data.node_types}
  edge_index_dict = {et: ei.to(device) for et, ei in graph_data.edge_index_dict.items()}

  # Temporal random-walk chain (consecutive races of the same driver) and the
  # driver_state mask restricted to the training window. Both are built once and
  # reused every epoch by temporal_smoothness_loss.
  chain = torch.cat(
    [
      edge_index_dict[("driver_state", "same_driver", "driver_state")],
      edge_index_dict[("driver_state", "same_driver_cross", "driver_state")],
    ],
    dim=1,
  )
  n_ds = graph_data["driver_state"].num_nodes
  train_ds_mask = torch.zeros(n_ds, dtype=torch.bool, device=device)
  train_ds_mask[res.driver_state_idx[train_mask].to(device)] = True

  # Qualifying pace head target + training mask (2nd pace signal). Captured once,
  # reused every epoch. The mask is the same era window as the ranking train mask
  # (year <= train_max_year), so the auxiliary head never sees the held-out seasons.
  quali_y = graph_data["qualifying"].y.to(device)
  train_max_year = max(cfg.TRAIN_YEARS)
  quali_train_mask = graph_data["qualifying"].year.to(device) <= train_max_year
  quali_val_mask = year_mask(
    graph_data["qualifying"].year, cfg.VAL_YEARS
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
  )

  best_val_score = float("inf")
  best_val_pl = float("inf")
  best_state = None
  epochs_no_improve = 0
  running_baselines: CoalitionBaselines | None = None

  print(
    "-> PL = Plackett-Luce NLL (lower is better): race ranking loss over finish order.",
    flush=True,
  )

  for epoch in range(epochs):
    lam = orth_lambda_at_epoch(epoch, lambda_orth, orth_warmup_epochs)
    model.train()
    optimizer.zero_grad()
    x_dict = model.encode(tf_dict, edge_index_dict)

    if running_baselines is None or epoch % 5 == 0:
      with torch.no_grad():
        running_baselines = compute_train_baselines(
          model,
          x_dict,
          train_mask,
          res.driver_state_idx.to(device),
          res.constructor_state_idx.to(device),
          res.race_idx.to(device),
          res.grid.to(device),
          res.round.to(device),
          getattr(res, "driver_career_idx", None).to(device)
          if hasattr(res, "driver_career_idx")
          else None,
        )

    train_total, pl_loss, orth_loss, attr_loss = race_loss_for_mask(
      model,
      x_dict,
      res,
      train_mask,
      device,
      aux_driver_weight=aux_driver_weight,
      aux_constructor_weight=aux_constructor_weight,
      lambda_ctx_aux=lambda_ctx_aux,
      lambda_pair=lambda_pair,
      baselines=running_baselines,
      lambda_attr=lambda_attr,
      target_driver_share=target_driver_share,
      target_constructor_share=target_constructor_share,
      attr_seed=seed + epoch,
    )
    rw_loss, shrink_loss = model.temporal_smoothness_loss(x_dict, chain, train_ds_mask)
    quali_loss = torch.zeros((), device=device)
    if lambda_quali > 0:
      qpred = model.quali_readout(x_dict["qualifying"]).squeeze(-1)
      quali_loss = torch.nn.functional.mse_loss(
        qpred[quali_train_mask], quali_y[quali_train_mask]
      )
    total_loss = (
      train_total
      + lam * orth_loss
      + lambda_rw * rw_loss
      + lambda_shrink * shrink_loss
      + lambda_quali * quali_loss
    )
    total_loss.backward()
    if max_grad_norm > 0:
      params = [p for p in model.parameters() if p.grad is not None]
      if params:
        total_norm = torch.linalg.vector_norm(
          torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in params])
        )
        if torch.isfinite(total_norm):
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    model.eval()
    with torch.no_grad():
      x_dict = model.encode(tf_dict, edge_index_dict)
      val_baselines = compute_train_baselines(
        model,
        x_dict,
        train_mask,
        res.driver_state_idx.to(device),
        res.constructor_state_idx.to(device),
        res.race_idx.to(device),
        res.grid.to(device),
        res.round.to(device),
        getattr(res, "driver_career_idx", None).to(device)
        if hasattr(res, "driver_career_idx")
        else None,
      )
      _, val_pl, _, val_attr = race_loss_for_mask(
        model,
        x_dict,
        res,
        val_mask,
        device,
        aux_driver_weight=aux_driver_weight,
        aux_constructor_weight=aux_constructor_weight,
        lambda_ctx_aux=lambda_ctx_aux,
        lambda_pair=lambda_pair,
        baselines=val_baselines,
        lambda_attr=lambda_attr,
        target_driver_share=target_driver_share,
        target_constructor_share=target_constructor_share,
        attr_seed=seed + epoch,
      )
      val_acc = pairwise_accuracy(model, x_dict, res, val_mask, device)
      val_auroc = top3_auroc(model, x_dict, res, val_mask, device)
      share_d, share_c, share_x = mean_attribution_shares(
        model, x_dict, res, val_mask, device, val_baselines, seed=seed + epoch
      )
      offset_frac = offset_frac_diagnostic(model, x_dict, res, train_mask, device)

    val_score = composite_val_score(val_pl.item(), val_acc)
    scheduler.step(val_pl.item())

    if val_score < best_val_score - 1e-4:
      best_val_score = val_score
      best_val_pl = val_pl.item()
      best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    print(
      f"epoch {epoch+1:3d} | train PL {pl_loss.item():.4f} | "
      f"orth {orth_loss.item():.4f} (λ={lam:.2f}) | attr {attr_loss.item():.4f} | "
      f"rw {rw_loss.item():.4f} | shrink {shrink_loss.item():.4f} | "
      f"quali {quali_loss.item():.4f} | "
      f"off_frac {offset_frac:.2f} | "
      f"val PL {val_pl.item():.4f} | val pairwise {val_acc:.4f} | "
      f"val score {val_score:.4f} | shares D/C/X {share_d:.2f}/{share_c:.2f}/{share_x:.2f} | "
      f"top3 AUROC {val_auroc:.4f} (diag)",
      flush=True,
    )

    if epochs_no_improve >= patience:
      print(f"early stop at epoch {epoch+1} (best val score {best_val_score:.4f})")
      break

  if best_state is not None:
    model.load_state_dict(best_state)

  model.eval()
  with torch.no_grad():
    x_dict = model.encode(tf_dict, edge_index_dict)
    _, test_pl, test_orth, test_attr = race_loss_for_mask(
      model,
      x_dict,
      res,
      test_mask,
      device,
      aux_driver_weight=aux_driver_weight,
      aux_constructor_weight=aux_constructor_weight,
      lambda_ctx_aux=lambda_ctx_aux,
      lambda_pair=lambda_pair,
      lambda_attr=0.0,
    )
    test_acc = pairwise_accuracy(model, x_dict, res, test_mask, device)
    test_auroc = top3_auroc(model, x_dict, res, test_mask, device)
    baselines = compute_train_baselines(
      model,
      x_dict,
      train_mask,
      res.driver_state_idx.to(device),
      res.constructor_state_idx.to(device),
      res.race_idx.to(device),
      res.grid.to(device),
      res.round.to(device),
      getattr(res, "driver_career_idx", None).to(device)
      if hasattr(res, "driver_career_idx")
      else None,
    )
    test_share_d, test_share_c, test_share_x = mean_attribution_shares(
      model, x_dict, res, test_mask, device, baselines, max_races=100, seed=seed
    )
    final_offset_frac = offset_frac_diagnostic(model, x_dict, res, train_mask, device)
    final_val_quali_loss = (
      torch.nn.functional.mse_loss(
        model.quali_readout(x_dict["qualifying"]).squeeze(-1)[quali_val_mask],
        quali_y[quali_val_mask],
      )
      if lambda_quali > 0
      else torch.zeros((), device=device)
    )

  os.makedirs(output_dir, exist_ok=True)
  ckpt_path = os.path.join(output_dir, "orthogonal_shapley.pth")
  meta_path = os.path.join(output_dir, "orthogonal_shapley_meta.json")
  baselines_path = os.path.join(output_dir, "coalition_baselines.json")

  torch.save(model.state_dict(), ckpt_path)
  enc_path = save_orthogonal_shapley_encoder(
    ckpt_path, node_to_col_names_dict, node_to_col_stats
  )
  with open(baselines_path, "w") as f:
    json.dump(baselines.to_dict(), f, indent=2)

  meta = {
    "arch_version": ARCH_VERSION,
    "config": {
      "hidden_dim": hidden_dim,
      "num_layers": num_layers,
      "mlp_hidden": mlp_hidden,
      "lambda_orth": lambda_orth,
      "aux_driver_weight": aux_driver_weight,
      "aux_constructor_weight": aux_constructor_weight,
      "lambda_ctx_aux": lambda_ctx_aux,
      "lambda_pair": lambda_pair,
      "lambda_attr": lambda_attr,
      "target_driver_share": target_driver_share,
      "target_constructor_share": target_constructor_share,
      "use_additive_readout": use_additive_readout,
      "orth_warmup_epochs": orth_warmup_epochs,
      "max_grad_norm": max_grad_norm,
      "lambda_rw": lambda_rw,
      "lambda_shrink": lambda_shrink,
      "lambda_quali": lambda_quali,
      "seed": seed,
    },
    "metrics": {
      "best_val_score": best_val_score,
      "best_val_pl": best_val_pl,
      "test_pl": float(test_pl.item()),
      "test_pairwise_acc": float(test_acc),
      "test_top3_auroc": float(test_auroc),
      "test_orth_loss": float(test_orth.item()),
      "test_attr_loss": float(test_attr.item()),
      "test_driver_share": test_share_d,
      "test_constructor_share": test_share_c,
      "test_context_share": test_share_x,
      "offset_frac_train": final_offset_frac,
      "quali_val_mse": float(final_val_quali_loss.item()),
    },
    "coalition_baselines_path": baselines_path,
  }
  with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

  race_df = export_race_skills(
    model, graph_data, tf_dict, edge_index_dict, device,
    baselines=baselines,
  )
  season_df = season_skill_from_races(race_df)
  season_df.to_csv(os.path.join(output_dir, "season_skill.csv"), index=False)

  print(
    f"\nOrthogonalShapleyGNN test PL NLL {test_pl.item():.4f} | "
    f"pairwise acc {test_acc:.4f} | top3 AUROC {test_auroc:.4f} (diagnostic) | "
    f"shares D/C/X {test_share_d:.2f}/{test_share_c:.2f}/{test_share_x:.2f}"
  )
  print(f"wrote {ckpt_path}, {enc_path}, {meta_path}, {baselines_path}")

  return {
    "checkpoint": ckpt_path,
    "meta": meta_path,
    "baselines": baselines_path,
    "meta_dict": meta,
    "best_val_pl": best_val_pl,
    "best_val_score": best_val_score,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Train OrthogonalShapleyGNN")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-5)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--hidden-dim", type=int, default=128)
  parser.add_argument("--num-layers", type=int, default=4)
  parser.add_argument("--mlp-hidden", type=int, default=128)
  parser.add_argument("--lambda-orth", type=float, default=2.0)
  parser.add_argument("--aux-driver-weight", type=float, default=0.5)
  parser.add_argument("--aux-constructor-weight", type=float, default=0.75)
  parser.add_argument("--lambda-ctx-aux", type=float, default=0.25)
  parser.add_argument("--lambda-pair", type=float, default=0.25)
  parser.add_argument("--lambda-attr", type=float, default=0.1)
  parser.add_argument("--target-driver-share", type=float, default=0.38)
  parser.add_argument("--target-constructor-share", type=float, default=0.30)
  parser.add_argument(
    "--use-additive-readout",
    action="store_true",
    help="read utility as u_d + u_c + u_x (exact Shapley) instead of fused MLP",
  )
  parser.add_argument(
    "--lambda-grid",
    type=str,
    default=None,
    help="comma-separated orth lambdas for ablation, e.g. 0,0.1,1",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--gpu-id", type=int, default=cfg.DEFAULT_GPU_ID)
  parser.add_argument(
    "--output-dir", type=str, default="output/orthogonal_shapley_model"
  )
  parser.add_argument("--smoke-test", action="store_true")
  parser.add_argument(
    "--max-grad-norm",
    type=float,
    default=1.0,
    help="gradient clipping when total norm is finite (0=off)",
  )
  parser.add_argument(
    "--orth-warmup-epochs",
    type=int,
    default=10,
    help="linear warmup epochs for lambda_orth",
  )
  parser.add_argument(
    "--lambda-rw",
    type=float,
    default=0.5,
    help="random-walk smoothness weight on the driver season offset",
  )
  parser.add_argument(
    "--lambda-shrink",
    type=float,
    default=0.05,
    help="shrinkage weight on the driver season offset level",
  )
  parser.add_argument(
    "--lambda-quali",
    type=float,
    default=0.0,
    help="MSE weight on the auxiliary qualifying pace head (0 = disabled)",
  )
  args = parser.parse_args()

  device = get_device(args.gpu_id)

  lambdas = [args.lambda_orth]
  if args.lambda_grid:
    lambdas = [float(x.strip()) for x in args.lambda_grid.split(",")]

  common = dict(
    aux_driver_weight=args.aux_driver_weight,
    aux_constructor_weight=args.aux_constructor_weight,
    lambda_ctx_aux=args.lambda_ctx_aux,
    lambda_pair=args.lambda_pair,
    lambda_attr=args.lambda_attr,
    target_driver_share=args.target_driver_share,
    target_constructor_share=args.target_constructor_share,
    use_additive_readout=args.use_additive_readout,
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
    mlp_hidden=args.mlp_hidden,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience,
    seed=args.seed,
    device=device,
    smoke_test=args.smoke_test,
    max_grad_norm=args.max_grad_norm,
    orth_warmup_epochs=args.orth_warmup_epochs,
    lambda_rw=args.lambda_rw,
    lambda_shrink=args.lambda_shrink,
    lambda_quali=args.lambda_quali,
  )

  if len(lambdas) == 1:
    train_one_config(lambda_orth=lambdas[0], output_dir=args.output_dir, **common)
    return

  best_result: Optional[Dict] = None
  for lam in lambdas:
    print(f"\n=== lambda_orth={lam} ===")
    out_dir = os.path.join(args.output_dir, f"lambda_{lam}")
    result = train_one_config(lambda_orth=lam, output_dir=out_dir, **common)
    if best_result is None or result["best_val_score"] < best_result["best_val_score"]:
      best_result = result

  if best_result:
    import shutil

    os.makedirs(args.output_dir, exist_ok=True)
    for fname in [
      "orthogonal_shapley.pth",
      "orthogonal_shapley_meta.json",
      "coalition_baselines.json",
      "orthogonal_shapley_encoder.pt",
      "season_skill.csv",
    ]:
      src = os.path.join(os.path.dirname(best_result["checkpoint"]), fname)
      if os.path.isfile(src):
        shutil.copy2(src, os.path.join(args.output_dir, fname))
    print(
      f"\nBest config copied to {args.output_dir} "
      f"(val score {best_result['best_val_score']:.4f})"
    )


if __name__ == "__main__":
  main()
