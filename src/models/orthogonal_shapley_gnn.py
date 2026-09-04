"""SAGE + MLP fusion model with orthogonal regularization for race ranking.

Port of the historical ``F1OrthogonalPipeline`` (master @ 9c9af86) onto the
causal round-state temporal graph.  Driver and constructor state embeddings are
fused via concatenation with graph-derived pre-race context and passed through
an MLP readout.  Training uses Plackett-Luce NLL on fused and auxiliary utilities.

Two readouts are supported:

- ``fused`` (legacy) — ``classifier([d || c || ctx])``, a nonlinear MLP over the
  concatenated players.  Coalition Shapley over this readout is *not* additive,
  so driver/constructor interaction terms can dominate the attribution.
- ``additive`` (default for new runs) — ``u_d(d) + u_c(c) + u_x(ctx)`` using the
  three per-player heads.  Shapley over three linear players is exact by
  construction (``phi_i = u_i(real) - u_i(baseline)``, efficiency residual 0),
  so the driver/constructor/context shares can be steered directly by the
  attribution-balance loss.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType

from models.sage_regressor import EDGE_TYPES
from relbench.modeling.nn import HeteroEncoder

STATE_TYPES: List[NodeType] = ["driver_state", "constructor_state"]
NORM_TYPES: List[NodeType] = ["driver_state", "constructor_state", "race"]
CONTEXT_DIM = 32  # projected context vector size (Shapley third player)
SCALAR_CONTEXT_DIM = 2  # grid_norm, round_norm
ARCH_VERSION = "v3"


class OrthogonalShapleyGNN(nn.Module):
  """Heterogeneous SAGE encoder + fused (or additive) utility head."""

  def __init__(
    self,
    node_to_col_names_dict: Dict[NodeType, Any],
    node_to_col_stats: Dict[NodeType, Any],
    hidden_dim: int = 128,
    num_layers: int = 4,
    mlp_hidden: int = 128,
    context_dim: int = CONTEXT_DIM,
    use_additive_readout: bool = False,
    num_drivers: int = 0,
  ):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.context_dim = context_dim
    self.arch_version = ARCH_VERSION
    self.use_additive_readout = use_additive_readout

    self.encoder = HeteroEncoder(
      channels=hidden_dim,
      node_to_col_names_dict=node_to_col_names_dict,
      node_to_col_stats=node_to_col_stats,
    )

    self.convs = nn.ModuleList()
    self.norms = nn.ModuleList()
    for i in range(num_layers):
      aggr = "mean" if i == 0 else "max"
      self.convs.append(
        HeteroConv(
          {et: SAGEConv((-1, -1), hidden_dim, aggr=aggr) for et in EDGE_TYPES},
          aggr="sum",
        )
      )
      self.norms.append(
        nn.ModuleDict(
          {nt: LayerNorm(hidden_dim, mode="node") for nt in NORM_TYPES}
        )
      )

    context_input_dim = SCALAR_CONTEXT_DIM + hidden_dim
    self.context_mlp = nn.Sequential(
      nn.Linear(context_input_dim, mlp_hidden),
      nn.ReLU(),
      nn.Linear(mlp_hidden, context_dim),
    )

    classifier_input_dim = hidden_dim * 2 + context_dim
    self.classifier = nn.Sequential(
      nn.Linear(classifier_input_dim, mlp_hidden),
      nn.ReLU(),
      nn.Linear(mlp_hidden, mlp_hidden),
      nn.ReLU(),
      nn.Linear(mlp_hidden, 1),
    )
    self.num_drivers = num_drivers
    self.driver_career = nn.Embedding(num_drivers, hidden_dim) if num_drivers > 0 else None
    self.aux_driver_career = nn.Linear(hidden_dim, 1)
    self.aux_driver_season = nn.Linear(hidden_dim, 1)
    self.aux_constructor = nn.Linear(hidden_dim, 1)
    self.aux_context = nn.Linear(context_dim, 1)
    self.driver_ctx_orth = nn.Linear(hidden_dim, context_dim)

  def encode(self, tf_dict, edge_index_dict) -> Dict[str, torch.Tensor]:
    x_dict = self.encoder(tf_dict)

    for i, conv in enumerate(self.convs):
      h = conv(x_dict, edge_index_dict)
      h = {k: v.relu() for k, v in h.items()}
      for nt, x in x_dict.items():
        if nt not in h:
          h[nt] = x
      norms = self.norms[i]
      for nt, norm in norms.items():
        if nt in h:
          h[nt] = norm(h[nt] + x_dict[nt])
      x_dict = h

    return x_dict

  @staticmethod
  def context_scalars(
    grid: torch.Tensor,
    round_num: torch.Tensor,
    *,
    max_round: float = 26.0,
  ) -> torch.Tensor:
    """Pre-race scalar features: normalized grid and round within season."""
    g = grid.float()
    grid_norm = (g - 1.0) / 19.0
    denom = max(max_round - 1.0, 1.0)
    round_norm = (round_num.float() - 1.0) / denom
    return torch.stack([grid_norm, round_norm], dim=-1)

  def context_vector(
    self,
    x_dict: Dict[str, torch.Tensor],
    race_idx: torch.Tensor,
    grid: torch.Tensor,
    round_num: torch.Tensor,
  ) -> torch.Tensor:
    """Projected context embedding (circuit/era + grid/round scalars)."""
    scalars = self.context_scalars(grid, round_num)
    race_emb = x_dict["race"][race_idx]
    return self.context_mlp(torch.cat([scalars, race_emb], dim=-1))

  def fused_input(
    self,
    d_emb: torch.Tensor,
    c_emb: torch.Tensor,
    ctx: torch.Tensor,
  ) -> torch.Tensor:
    return torch.cat([d_emb, c_emb, ctx], dim=-1)

  def utility_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
    return self.classifier(fused).squeeze(-1)

  def driver_skill(
    self,
    d_emb: torch.Tensor,
    career_emb: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Driver player value = career-shared ability + per-season offset.

    ``career_emb`` is the car-free per-driver embedding (hard-identification
    Section 1); ``d_emb`` is the per-season GNN node. When ``career_emb`` is
    None (or the career embedding is disabled) the driver channel reverts to the
    legacy per-season head.
    """
    u_season = self.aux_driver_season(d_emb).squeeze(-1)
    if career_emb is not None and self.driver_career is not None:
      u_career = self.aux_driver_career(career_emb).squeeze(-1)
      return u_career + u_season
    return u_season

  def utility_additive(
    self,
    d_emb: torch.Tensor,
    c_emb: torch.Tensor,
    ctx: torch.Tensor,
    career_emb: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Sum of the three per-player heads. Shapley over this is exact."""
    u_d = self.driver_skill(d_emb, career_emb)
    u_c = self.aux_constructor(c_emb).squeeze(-1)
    u_x = self.aux_context(ctx).squeeze(-1)
    return u_d + u_c + u_x

  def race_utilities(
    self,
    x_dict: Dict[str, torch.Tensor],
    driver_state_idx: torch.Tensor,
    constructor_state_idx: torch.Tensor,
    race_idx: torch.Tensor,
    grid: torch.Tensor,
    round_num: torch.Tensor,
    driver_career_idx: torch.Tensor | None = None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fused_u, driver_aux_u, constructor_aux_u, context_aux_u, ctx).

    During export the driver channel is replaced by coalition Shapley values.
    ``driver_career_idx`` enables the career-shared driver head (hard
    identification); when None the driver channel is the legacy per-season head.
    """
    d_emb = x_dict["driver_state"][driver_state_idx]
    c_emb = x_dict["constructor_state"][constructor_state_idx]
    ctx = self.context_vector(x_dict, race_idx, grid, round_num)
    career_emb = (
      self.driver_career(driver_career_idx)
      if (driver_career_idx is not None and self.driver_career is not None)
      else None
    )
    u_d = self.driver_skill(d_emb, career_emb)
    u_c = self.aux_constructor(c_emb).squeeze(-1)
    u_x = self.aux_context(ctx).squeeze(-1)
    if self.use_additive_readout:
      u_fused = u_d + u_c + u_x
    else:
      fused = self.fused_input(d_emb, c_emb, ctx)
      u_fused = self.utility_from_fused(fused)
    return u_fused, u_d, u_c, u_x, ctx

  def paired_orthogonal_loss(
    self,
    x_dict: Dict[str, torch.Tensor],
    driver_state_idx: torch.Tensor,
    constructor_state_idx: torch.Tensor,
    race_idx: torch.Tensor,
    grid: torch.Tensor,
    round_num: torch.Tensor,
    *,
    normalized: bool = True,
  ) -> torch.Tensor:
    """Orthogonal penalty for driver vs constructor and driver vs context.

    When ``normalized=True`` (default), uses squared cosine similarity so the
    penalty stays O(1) regardless of embedding scale during early training.
    """
    z_drv = x_dict["driver_state"][driver_state_idx]
    z_cons = x_dict["constructor_state"][constructor_state_idx]
    z_ctx = self.context_vector(x_dict, race_idx, grid, round_num)
    z_drv_ctx = self.driver_ctx_orth(z_drv)
    if normalized:
      z_drv = torch.nn.functional.normalize(z_drv, p=2, dim=-1, eps=1e-6)
      z_cons = torch.nn.functional.normalize(z_cons, p=2, dim=-1, eps=1e-6)
      z_drv_ctx = torch.nn.functional.normalize(z_drv_ctx, p=2, dim=-1, eps=1e-6)
      z_ctx = torch.nn.functional.normalize(z_ctx, p=2, dim=-1, eps=1e-6)
    dot_dc = torch.sum(z_drv * z_cons, dim=-1)
    dot_dx = torch.sum(z_drv_ctx * z_ctx, dim=-1)
    return torch.mean(dot_dc ** 2 + dot_dx ** 2)

  def temporal_smoothness_loss(
    self,
    x_dict: Dict[str, torch.Tensor],
    chain_edge_index: torch.Tensor,
    node_mask: torch.Tensor | None = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(L_rw, L_shrink) over the scalar per-race driver offset.

    GP random-walk analogue: penalizes jumps of the offset between consecutive
    races of the same driver (RW) and the offset level (shrinkage), pushing the
    driver identity into the car-free career embedding.
    """
    u = self.aux_driver_season(x_dict["driver_state"]).squeeze(-1)  # (N_ds,)
    src, dst = chain_edge_index[0], chain_edge_index[1]
    if node_mask is not None:
      keep = node_mask[src] & node_mask[dst]
      src, dst = src[keep], dst[keep]
    rw = torch.mean((u[dst] - u[src]) ** 2) if src.numel() > 0 else u.new_zeros(())
    u_shrink = u[node_mask] if node_mask is not None else u
    shrink = torch.mean(u_shrink ** 2) if u_shrink.numel() > 0 else u.new_zeros(())
    return rw, shrink

  def forward(self, tf_dict, edge_index_dict):
    return self.encode(tf_dict, edge_index_dict)
