"""SAGE + MLP fusion model with orthogonal regularization for race ranking.

Port of the historical ``F1OrthogonalPipeline`` (master @ 9c9af86) onto the
causal round-state temporal graph.  Driver and constructor state embeddings are
fused via concatenation with pre-race context (grid) and passed through an MLP
readout.  Training uses Plackett-Luce NLL on fused and auxiliary utilities.
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
CONTEXT_DIM = 2  # grid_norm, qualifying_norm (grid proxy)


class OrthogonalShapleyGNN(nn.Module):
  """Heterogeneous SAGE encoder + fused MLP utility head."""

  def __init__(
    self,
    node_to_col_names_dict: Dict[NodeType, Any],
    node_to_col_stats: Dict[NodeType, Any],
    hidden_dim: int = 32,
    num_layers: int = 2,
    mlp_hidden: int = 32,
  ):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.context_dim = CONTEXT_DIM

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
          {nt: LayerNorm(hidden_dim, mode="node") for nt in STATE_TYPES}
        )
      )

    classifier_input_dim = hidden_dim * 2 + CONTEXT_DIM
    self.classifier = nn.Sequential(
      nn.Linear(classifier_input_dim, mlp_hidden),
      nn.ReLU(),
      nn.Linear(mlp_hidden, 1),
    )
    self.aux_driver = nn.Linear(hidden_dim, 1)
    self.aux_constructor = nn.Linear(hidden_dim, 1)

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
  def context_features(grid: torch.Tensor) -> torch.Tensor:
    """Pre-race context: normalized grid position (qualifying proxy)."""
    g = grid.float()
    grid_norm = (g - 1.0) / 19.0
    qual_norm = grid_norm  # grid is the qualifying result in modern F1
    return torch.stack([grid_norm, qual_norm], dim=-1)

  def fused_input(
    self,
    d_emb: torch.Tensor,
    c_emb: torch.Tensor,
    grid: torch.Tensor,
  ) -> torch.Tensor:
    ctx = self.context_features(grid)
    return torch.cat([d_emb, c_emb, ctx], dim=-1)

  def utility_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
    return self.classifier(fused).squeeze(-1)

  def race_utilities(
    self,
    x_dict: Dict[str, torch.Tensor],
    driver_state_idx: torch.Tensor,
    constructor_state_idx: torch.Tensor,
    grid: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fused_u, driver_aux_u, constructor_aux_u, driver_shapley_placeholder).

    During export the driver channel is replaced by coalition Shapley values.
    """
    d_emb = x_dict["driver_state"][driver_state_idx]
    c_emb = x_dict["constructor_state"][constructor_state_idx]
    fused = self.fused_input(d_emb, c_emb, grid)
    u_fused = self.utility_from_fused(fused)
    u_d = self.aux_driver(d_emb).squeeze(-1)
    u_c = self.aux_constructor(c_emb).squeeze(-1)
    return u_fused, u_d, u_c, u_d

  def paired_orthogonal_loss(
    self,
    x_dict: Dict[str, torch.Tensor],
    driver_state_idx: torch.Tensor,
    constructor_state_idx: torch.Tensor,
    *,
    normalized: bool = True,
  ) -> torch.Tensor:
    """Orthogonal penalty for matched driver-constructor pairs in a batch.

    When ``normalized=True`` (default), uses squared cosine similarity so the
    penalty stays O(1) regardless of embedding scale during early training.
    """
    z_drv = x_dict["driver_state"][driver_state_idx]
    z_cons = x_dict["constructor_state"][constructor_state_idx]
    if normalized:
      z_drv = torch.nn.functional.normalize(z_drv, p=2, dim=-1, eps=1e-6)
      z_cons = torch.nn.functional.normalize(z_cons, p=2, dim=-1, eps=1e-6)
    dot = torch.sum(z_drv * z_cons, dim=-1)
    return torch.mean(dot ** 2)

  def forward(self, tf_dict, edge_index_dict):
    return self.encode(tf_dict, edge_index_dict)
