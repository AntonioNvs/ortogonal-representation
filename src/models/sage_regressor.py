"""SAGE regressor for qualifying grid position.

Architecture per ``docs/plans/2026-08-25-sage-position-regression-design.md``:

- ``HeteroEncoder`` (input side, "por baixo") encodes each node type's tabular
  features to a ``hidden_dim`` embedding.
- A causal SAGE stack (``HeteroConv`` of ``SAGEConv``): conv1 ``mean``, conv2
  ``max``, with residual + ``LayerNorm`` on the state/target node types.
- A **single** ``Linear(hidden_dim, 1)`` readout over the ``qualifying`` node
  embedding — no MLP, no aux heads, no feature fusion, so gradient/importance
  attribution stays clean.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType

from relbench.modeling.nn import HeteroEncoder

# All edge types participating in message passing (see temporal_graph.py).
EDGE_TYPES: List[EdgeType] = [
    ("driver_state", "same_driver", "driver_state"),
    ("driver_state", "same_driver_cross", "driver_state"),
    ("constructor_state", "same_constructor", "constructor_state"),
    ("constructor_state", "same_constructor_cross", "constructor_state"),
    ("results", "result_of_driver", "driver_state"),
    ("constructor_results", "result_of_constructor", "constructor_state"),
    ("circuit", "circuit_to_race", "race"),
    ("race", "race_to_qualifying", "qualifying"),
    ("driver_state", "driver_state_to_qualifying", "qualifying"),
    ("constructor_state", "constructor_state_to_qualifying", "qualifying"),
]

# Edge types that feed the *target* node. The final SAGE layer only needs to
# produce the ``qualifying`` embedding, so it uses these alone; using the full
# set would create "dead" weights for node types whose final embeddings are
# never read out.
QUALIFYING_IN_EDGE_TYPES: List[EdgeType] = [
    ("driver_state", "driver_state_to_qualifying", "qualifying"),
    ("constructor_state", "constructor_state_to_qualifying", "qualifying"),
    ("race", "race_to_qualifying", "qualifying"),
]

# Node types that get residual + LayerNorm (the recurrent states and the target).
RESIDUAL_TYPES: List[NodeType] = ["driver_state", "constructor_state", "qualifying"]


class SageQualifyingRegressor(nn.Module):
    def __init__(
        self,
        node_to_col_names_dict: Dict[NodeType, Any],
        node_to_col_stats: Dict[NodeType, Any],
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = HeteroEncoder(
            channels=hidden_dim,
            node_to_col_names_dict=node_to_col_names_dict,
            node_to_col_stats=node_to_col_stats,
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            aggr = "mean" if i == 0 else "max"
            # The final layer only produces the target node embedding.
            edge_types = EDGE_TYPES if i < num_layers - 1 else QUALIFYING_IN_EDGE_TYPES
            self.convs.append(
                HeteroConv(
                    {et: SAGEConv((-1, -1), hidden_dim, aggr=aggr) for et in edge_types},
                    aggr="sum",
                )
            )
            # Only layer the nodes this layer actually produces; on the final
            # layer that is just ``qualifying``.
            produced = {et[2] for et in edge_types}
            self.norms.append(
                nn.ModuleDict(
                    {nt: LayerNorm(hidden_dim, mode="node") for nt in RESIDUAL_TYPES if nt in produced}
                )
            )

        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, tf_dict, edge_index_dict):
        x_dict = self.encoder(tf_dict)

        for i, conv in enumerate(self.convs):
            h = conv(x_dict, edge_index_dict)
            h = {k: v.relu() for k, v in h.items()}

            # Carry forward node types the conv didn't update (e.g. ``results``,
            # which is a leaf and only ever sends messages; and, on the final
            # layer, every node type except ``qualifying``).
            for nt, x in x_dict.items():
                if nt not in h:
                    h[nt] = x

            # Residual + LayerNorm on the recurrent state/target node types.
            norms = self.norms[i]
            for nt, norm in norms.items():
                h[nt] = norm(h[nt] + x_dict[nt])

            x_dict = h

        return self.readout(x_dict["qualifying"]).squeeze(-1)
