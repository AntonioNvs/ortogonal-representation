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
    self.aux_driver = nn.Linear(hidden_dim, 1)
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

  def utility_additive(
    self,
    d_emb: torch.Tensor,
    c_emb: torch.Tensor,
    ctx: torch.Tensor,
  ) -> torch.Tensor:
    """Sum of the three per-player heads. Shapley over this is exact."""
    u_d = self.aux_driver(d_emb).squeeze(-1)
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
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fused_u, driver_aux_u, constructor_aux_u, context_aux_u, ctx).

    During export the driver channel is replaced by coalition Shapley values.
    """
    d_emb = x_dict["driver_state"][driver_state_idx]
    c_emb = x_dict["constructor_state"][constructor_state_idx]
    ctx = self.context_vector(x_dict, race_idx, grid, round_num)
    u_d = self.aux_driver(d_emb).squeeze(-1)
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

  def forward(self, tf_dict, edge_index_dict):
    return self.encode(tf_dict, edge_index_dict)
