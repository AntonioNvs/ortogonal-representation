"""GNN skill ranker on the causal round-state temporal graph.

Trains a SAGE encoder (shared substrate with ``sage_regressor.py``) and reads
out per-race utilities from ``driver_state`` + ``constructor_state`` embeddings.
Race finish order is trained with Plackett-Luce NLL; the driver channel is the
exported skill score ``f(D,T,R)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType

from models.sage_regressor import EDGE_TYPES
from relbench.modeling.nn import HeteroEncoder

SKILL_RESIDUAL_TYPES: List[NodeType] = ["driver_state", "constructor_state"]


class SkillGNN(nn.Module):
    def __init__(
        self,
        node_to_col_names_dict: Dict[NodeType, Any],
        node_to_col_stats: Dict[NodeType, Any],
        hidden_dim: int = 128,
        num_layers: int = 4,
        grid_weight: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_weight = grid_weight
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
                    {
                        nt: LayerNorm(hidden_dim, mode="node")
                        for nt in SKILL_RESIDUAL_TYPES
                    }
                )
            )

        self.driver_readout = nn.Linear(hidden_dim, 1)
        self.constructor_readout = nn.Linear(hidden_dim, 1)

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

    def race_utilities(
        self,
        x_dict: Dict[str, torch.Tensor],
        driver_state_idx: torch.Tensor,
        constructor_state_idx: torch.Tensor,
        grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (total utility, driver skill channel) for indexed result rows."""
        d_emb = x_dict["driver_state"][driver_state_idx]
        c_emb = x_dict["constructor_state"][constructor_state_idx]
        u_d = self.driver_readout(d_emb).squeeze(-1)
        u_c = self.constructor_readout(c_emb).squeeze(-1)
        u = u_d + u_c
        if grid is not None:
            u = u + self.grid_weight * (-(grid.float() - 1.0))
        return u, u_d

    def forward(self, tf_dict, edge_index_dict):
        return self.encode(tf_dict, edge_index_dict)
