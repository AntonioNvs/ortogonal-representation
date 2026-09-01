"""Exact 3-player coalition Shapley for the fused MLP utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from models.orthogonal_shapley_gnn import CONTEXT_DIM, OrthogonalShapleyGNN

# Coalition bitmasks: driver=1, constructor=2, context=4
PLAYER_DRIVER = 1
PLAYER_CONSTRUCTOR = 2
PLAYER_CONTEXT = 4
ALL_PLAYERS = PLAYER_DRIVER | PLAYER_CONSTRUCTOR | PLAYER_CONTEXT

# Exact Shapley weights for n=3 (all subsets S not containing player i)
_SHAPLEY_WEIGHTS: Dict[int, Dict[int, float]] = {
  PLAYER_DRIVER: {
    0: 1 / 3,
    PLAYER_CONSTRUCTOR: 1 / 6,
    PLAYER_CONTEXT: 1 / 6,
    PLAYER_CONSTRUCTOR | PLAYER_CONTEXT: 1 / 3,
  },
  PLAYER_CONSTRUCTOR: {
    0: 1 / 3,
    PLAYER_DRIVER: 1 / 6,
    PLAYER_CONTEXT: 1 / 6,
    PLAYER_DRIVER | PLAYER_CONTEXT: 1 / 3,
  },
  PLAYER_CONTEXT: {
    0: 1 / 3,
    PLAYER_DRIVER: 1 / 6,
    PLAYER_CONSTRUCTOR: 1 / 6,
    PLAYER_DRIVER | PLAYER_CONSTRUCTOR: 1 / 3,
  },
}


@dataclass
class CoalitionBaselines:
  """Train-only reference embeddings and context for absent players."""

  driver_emb: torch.Tensor  # (hidden_dim,)
  constructor_emb: torch.Tensor  # (hidden_dim,)
  context: torch.Tensor  # (CONTEXT_DIM,)

  def to_dict(self) -> dict:
    return {
      "driver_emb": self.driver_emb.cpu().tolist(),
      "constructor_emb": self.constructor_emb.cpu().tolist(),
      "context": self.context.cpu().tolist(),
    }

  @classmethod
  def from_dict(cls, d: dict, device: torch.device) -> "CoalitionBaselines":
    return cls(
      driver_emb=torch.tensor(d["driver_emb"], device=device),
      constructor_emb=torch.tensor(d["constructor_emb"], device=device),
      context=torch.tensor(d["context"], device=device),
    )


def compute_train_baselines(
  model: OrthogonalShapleyGNN,
  x_dict: Dict[str, torch.Tensor],
  train_mask: torch.Tensor,
  driver_state_idx: torch.Tensor,
  constructor_state_idx: torch.Tensor,
  grid: torch.Tensor,
) -> CoalitionBaselines:
  """Mean embeddings and context from training-year result rows only."""
  idx = train_mask.nonzero(as_tuple=True)[0]
  d_idx = driver_state_idx[idx]
  c_idx = constructor_state_idx[idx]
  d_emb = x_dict["driver_state"][d_idx].mean(dim=0)
  c_emb = x_dict["constructor_state"][c_idx].mean(dim=0)
  ctx = model.context_features(grid[idx]).mean(dim=0)
  return CoalitionBaselines(driver_emb=d_emb, constructor_emb=c_emb, context=ctx)


def _coalition_value(
  model: OrthogonalShapleyGNN,
  coalition_mask: int,
  d_emb: torch.Tensor,
  c_emb: torch.Tensor,
  ctx: torch.Tensor,
  baselines: CoalitionBaselines,
) -> torch.Tensor:
  """Scalar utility for one coalition (batch of rows)."""
  d = d_emb if coalition_mask & PLAYER_DRIVER else baselines.driver_emb
  c = c_emb if coalition_mask & PLAYER_CONSTRUCTOR else baselines.constructor_emb
  x = ctx if coalition_mask & PLAYER_CONTEXT else baselines.context

  if d_emb.dim() == 1:
    fused = torch.cat([d, c, x], dim=-1).unsqueeze(0)
  else:
    batch = d_emb.shape[0]
    if not (coalition_mask & PLAYER_DRIVER):
      d = baselines.driver_emb.unsqueeze(0).expand(batch, -1)
    if not (coalition_mask & PLAYER_CONSTRUCTOR):
      c = baselines.constructor_emb.unsqueeze(0).expand(batch, -1)
    if not (coalition_mask & PLAYER_CONTEXT):
      x = baselines.context.unsqueeze(0).expand(batch, -1)
    fused = torch.cat([d, c, x], dim=-1)

  return model.utility_from_fused(fused)


def exact_shapley_utilities(
  model: OrthogonalShapleyGNN,
  d_emb: torch.Tensor,
  c_emb: torch.Tensor,
  ctx: torch.Tensor,
  baselines: CoalitionBaselines,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Exact 3-player Shapley values centered at v(empty coalition).

  Returns (phi_driver, phi_constructor, phi_context, efficiency_residual).
  """
  v_cache: Dict[int, torch.Tensor] = {}
  for mask in range(8):
    v_cache[mask] = _coalition_value(model, mask, d_emb, c_emb, ctx, baselines)

  phi_d = torch.zeros_like(v_cache[0])
  phi_c = torch.zeros_like(v_cache[0])
  phi_x = torch.zeros_like(v_cache[0])

  for player, weights in _SHAPLEY_WEIGHTS.items():
    contrib = torch.zeros_like(v_cache[0])
    for coalition, w in weights.items():
      with_player = coalition | player
      contrib = contrib + w * (v_cache[with_player] - v_cache[coalition])
    if player == PLAYER_DRIVER:
      phi_d = contrib
    elif player == PLAYER_CONSTRUCTOR:
      phi_c = contrib
    else:
      phi_x = contrib

  v_full = v_cache[ALL_PLAYERS]
  v_empty = v_cache[0]
  total_phi = phi_d + phi_c + phi_x
  residual = (v_full - v_empty) - total_phi
  return phi_d, phi_c, phi_x, residual


def shapley_efficiency_error(
  phi_d: torch.Tensor,
  phi_c: torch.Tensor,
  phi_x: torch.Tensor,
  v_full: torch.Tensor,
  v_empty: torch.Tensor,
) -> float:
  """Mean absolute efficiency violation."""
  total = phi_d + phi_c + phi_x
  target = v_full - v_empty
  return float(torch.mean(torch.abs(total - target)).item())
