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
from explain.coalition_shapley import CoalitionBaselines, compute_train_baselines
from models.orthogonal_shapley_gnn import OrthogonalShapleyGNN
from models.ranking_likelihood import batch_pl_nll
from relbench.datasets import get_dataset
from utils.device import get_device


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def year_mask(years: torch.Tensor, allowed) -> torch.Tensor:
  allowed = set(allowed)
  return torch.tensor([int(y) in allowed for y in years.tolist()], dtype=torch.bool)


def race_pl_loss_for_mask(
  model: OrthogonalShapleyGNN,
  x_dict: dict,
  res,
  row_mask: torch.Tensor,
  device: torch.device,
  *,
  use_aux: bool = True,
  aux_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """PL loss on fused utility + optional aux heads. Returns (total, orth_loss)."""
  idx = row_mask.nonzero(as_tuple=True)[0]
  if idx.numel() == 0:
    zero = torch.tensor(0.0, device=device)
    return zero, zero

  race_ids = res.race_id[idx].to(device)
  positions = res.position[idx].to(device)
  driver_state_idx = res.driver_state_idx[idx].to(device)
  constructor_state_idx = res.constructor_state_idx[idx].to(device)
  grid = res.grid[idx].to(device)

  fused_list: List[torch.Tensor] = []
  driver_list: List[torch.Tensor] = []
  cons_list: List[torch.Tensor] = []
  ranks_list: List[torch.Tensor] = []

  for rid in torch.unique(race_ids):
    rmask = race_ids == rid
    u_fused, u_d, u_c, _ = model.race_utilities(
      x_dict,
      driver_state_idx[rmask],
      constructor_state_idx[rmask],
      grid[rmask],
    )
    fused_list.append(u_fused)
    driver_list.append(u_d)
    cons_list.append(u_c)
    ranks_list.append(positions[rmask])

  pl_fused = batch_pl_nll(fused_list, ranks_list)
  if use_aux:
    pl_driver = batch_pl_nll(driver_list, ranks_list)
    pl_cons = batch_pl_nll(cons_list, ranks_list)
    pl_total = pl_fused + aux_weight * (pl_driver + pl_cons)
  else:
    pl_total = pl_fused

  orth = model.paired_orthogonal_loss(
    x_dict, driver_state_idx, constructor_state_idx
  )
  return pl_total, orth


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
  grid = res.grid[idx].to(device)

  for rid in torch.unique(race_ids):
    rmask = race_ids == rid
    u_fused, _, _, _ = model.race_utilities(
      x_dict,
      driver_state_idx[rmask],
      constructor_state_idx[rmask],
      grid[rmask],
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
  grid = res.grid[idx].to(device)
  positions = res.position[idx].cpu().numpy()

  u_fused, _, _, _ = model.race_utilities(
    x_dict, driver_state_idx, constructor_state_idx, grid
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


def orth_lambda_at_epoch(epoch: int, target: float, warmup_epochs: int) -> float:
  """Linear warmup for orthogonal penalty (avoids early gradient domination)."""
  if warmup_epochs <= 0:
    return target
  return target * min(1.0, (epoch + 1) / warmup_epochs)


def train_one_config(
  *,
  lambda_orth: float,
  aux_weight: float,
  hidden_dim: int,
  num_layers: int,
  epochs: int,
  lr: float,
  weight_decay: float,
  patience: int,
  seed: int,
  device: torch.device,
  output_dir: str,
  smoke_test: bool = False,
  max_grad_norm: float = 1.0,
  orth_warmup_epochs: int = 5,
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
  base_mask = res.in_ranking & (res.driver_state_idx >= 0) & (res.constructor_state_idx >= 0)
  train_mask = base_mask & year_mask(res.year, cfg.TRAIN_YEARS)
  val_mask = base_mask & year_mask(res.year, cfg.VAL_YEARS)
  test_mask = base_mask & year_mask(res.year, cfg.TEST_YEARS)

  if smoke_test:
    epochs = min(epochs, 3)
    patience = 2

  model = OrthogonalShapleyGNN(
    node_to_col_names_dict=node_to_col_names_dict,
    node_to_col_stats=node_to_col_stats,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
  ).to(device)

  tf_dict = {nt: graph_data[nt].tf.to(device) for nt in graph_data.node_types}
  edge_index_dict = {et: ei.to(device) for et, ei in graph_data.edge_index_dict.items()}

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
  )

  best_val_pl = float("inf")
  best_state = None
  epochs_no_improve = 0

  print(
    "-> PL = Plackett-Luce NLL (lower is better): race ranking loss over finish order.",
    flush=True,
  )

  for epoch in range(epochs):
    lam = orth_lambda_at_epoch(epoch, lambda_orth, orth_warmup_epochs)
    model.train()
    optimizer.zero_grad()
    x_dict = model.encode(tf_dict, edge_index_dict)
    pl_loss, orth_loss = race_pl_loss_for_mask(
      model, x_dict, res, train_mask, device, aux_weight=aux_weight
    )
    total_loss = pl_loss + lam * orth_loss
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
      val_pl, _ = race_pl_loss_for_mask(
        model, x_dict, res, val_mask, device, aux_weight=aux_weight
      )
      val_acc = pairwise_accuracy(model, x_dict, res, val_mask, device)
      val_auroc = top3_auroc(model, x_dict, res, val_mask, device)

    scheduler.step(val_pl.item())

    if val_pl.item() < best_val_pl - 1e-4:
      best_val_pl = val_pl.item()
      best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    print(
      f"epoch {epoch+1:3d} | train PL NLL {pl_loss.item():.4f} | "
      f"orth {orth_loss.item():.4f} (λ={lam:.2f}) | "
      f"val PL NLL {val_pl.item():.4f} | val pairwise acc {val_acc:.4f} | "
      f"val top3 AUROC {val_auroc:.4f} (diag)",
      flush=True,
    )

    if epochs_no_improve >= patience:
      print(f"early stop at epoch {epoch+1} (best val PL {best_val_pl:.4f})")
      break

  if best_state is not None:
    model.load_state_dict(best_state)

  model.eval()
  with torch.no_grad():
    x_dict = model.encode(tf_dict, edge_index_dict)
    test_pl, test_orth = race_pl_loss_for_mask(
      model, x_dict, res, test_mask, device, aux_weight=aux_weight
    )
    test_acc = pairwise_accuracy(model, x_dict, res, test_mask, device)
    test_auroc = top3_auroc(model, x_dict, res, test_mask, device)
    baselines = compute_train_baselines(
      model, x_dict, train_mask,
      res.driver_state_idx.to(device),
      res.constructor_state_idx.to(device),
      res.grid.to(device),
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
    "config": {
      "hidden_dim": hidden_dim,
      "num_layers": num_layers,
      "lambda_orth": lambda_orth,
      "aux_weight": aux_weight,
      "orth_warmup_epochs": orth_warmup_epochs,
      "max_grad_norm": max_grad_norm,
      "seed": seed,
    },
    "metrics": {
      "best_val_pl": best_val_pl,
      "test_pl": float(test_pl.item()),
      "test_pairwise_acc": float(test_acc),
      "test_top3_auroc": float(test_auroc),
      "test_orth_loss": float(test_orth.item()),
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
    f"pairwise acc {test_acc:.4f} | top3 AUROC {test_auroc:.4f} (diagnostic)"
  )
  print(f"wrote {ckpt_path}, {enc_path}, {meta_path}, {baselines_path}")

  return {
    "checkpoint": ckpt_path,
    "meta": meta_path,
    "baselines": baselines_path,
    "meta_dict": meta,
    "best_val_pl": best_val_pl,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Train OrthogonalShapleyGNN")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-5)
  parser.add_argument("--patience", type=int, default=15)
  parser.add_argument("--hidden-dim", type=int, default=32)
  parser.add_argument("--num-layers", type=int, default=2)
  parser.add_argument("--lambda-orth", type=float, default=1.0)
  parser.add_argument("--aux-weight", type=float, default=0.5)
  parser.add_argument("--lambda-grid", type=str, default=None,
                      help="comma-separated orth lambdas for ablation, e.g. 0,0.1,1")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--gpu-id", type=int, default=cfg.DEFAULT_GPU_ID)
  parser.add_argument("--output-dir", type=str, default="output/orthogonal_shapley_model")
  parser.add_argument("--smoke-test", action="store_true")
  parser.add_argument("--max-grad-norm", type=float, default=0.0,
                      help="gradient clipping when total norm is finite (0=off, matches SkillGNN)")
  parser.add_argument("--orth-warmup-epochs", type=int, default=5,
                      help="linear warmup epochs for lambda_orth")
  args = parser.parse_args()

  device = get_device(args.gpu_id)

  lambdas = [args.lambda_orth]
  if args.lambda_grid:
    lambdas = [float(x.strip()) for x in args.lambda_grid.split(",")]

  if len(lambdas) == 1:
    train_one_config(
      lambda_orth=lambdas[0],
      aux_weight=args.aux_weight,
      hidden_dim=args.hidden_dim,
      num_layers=args.num_layers,
      epochs=args.epochs,
      lr=args.lr,
      weight_decay=args.weight_decay,
      patience=args.patience,
      seed=args.seed,
      device=device,
      output_dir=args.output_dir,
      smoke_test=args.smoke_test,
      max_grad_norm=args.max_grad_norm,
      orth_warmup_epochs=args.orth_warmup_epochs,
    )
    return

  best_result: Optional[Dict] = None
  for lam in lambdas:
    print(f"\n=== lambda_orth={lam} ===")
    out_dir = os.path.join(args.output_dir, f"lambda_{lam}")
    result = train_one_config(
      lambda_orth=lam,
      aux_weight=args.aux_weight,
      hidden_dim=args.hidden_dim,
      num_layers=args.num_layers,
      epochs=args.epochs,
      lr=args.lr,
      weight_decay=args.weight_decay,
      patience=args.patience,
      seed=args.seed,
      device=device,
      output_dir=out_dir,
      smoke_test=args.smoke_test,
      max_grad_norm=args.max_grad_norm,
      orth_warmup_epochs=args.orth_warmup_epochs,
    )
    if best_result is None or result["best_val_pl"] < best_result["best_val_pl"]:
      best_result = result

  if best_result:
    # Copy best to main output dir
    import shutil
    os.makedirs(args.output_dir, exist_ok=True)
    for fname in ["orthogonal_shapley.pth", "orthogonal_shapley_meta.json",
                  "coalition_baselines.json", "orthogonal_shapley_encoder.pt",
                  "season_skill.csv"]:
      src = os.path.join(os.path.dirname(best_result["checkpoint"]), fname)
      if os.path.isfile(src):
        shutil.copy2(src, os.path.join(args.output_dir, fname))
    print(f"\nBest config copied to {args.output_dir} (val PL {best_result['best_val_pl']:.4f})")


if __name__ == "__main__":
  main()
