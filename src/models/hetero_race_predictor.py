"""Heterogeneous GNN predictor over the temporal meta-node F1 graph.

Pairs with :mod:`data.temporal_graph`. The model owns:

    * learned embeddings for the two identity-like node types
      (``driver_season``, ``constructor_season``) — one parameter vector per
      (entity, season), refined by message passing;
    * small MLP projections of the *static* features for ``circuit`` and
      ``race``;
    * a stack of heterogeneous SAGE layers over the graph edges;
    * a scalar *edge readout* that predicts ``position_norm`` for each
      ``raced_in`` (driver_season -> race) edge from the four participating
      node embeddings.

The readout is the **counterfactual intervention point**: to answer "driver X
in team Y's car at season T" we simply substitute the ``constructor_season``
index passed to :meth:`readout_from` with Y's node — no retraining, no change
to the driver embedding.

Directional ``same_driver`` / ``same_constructor`` edges (T -> T+1) are the
only cross-season connections, so message passing is forward-in-time and
leak-free by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

# Edge types the model message-passes over (must match temporal_graph.py).
EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("driver_season", "drives_for", "constructor_season"),
    ("driver_season", "same_driver", "driver_season"),
    ("constructor_season", "same_constructor", "constructor_season"),
    ("driver_season", "raced_in", "race"),
    ("race", "held_at", "circuit"),
)

# Node types in a fixed order, used by :meth:`node_features`.
NODE_TYPES: tuple[str, ...] = (
    "driver_season",
    "constructor_season",
    "circuit",
    "race",
)


class HeteroRacePredictor(nn.Module):
    """GNN that predicts race position from separable node embeddings."""

    def __init__(
        self,
        *,
        num_driver_season: int,
        num_constructor_season: int,
        num_circuit: int,
        num_race: int,
        state_dim: int = 32,
        num_layers: int = 2,
        circuit_feat_dim: int = 3,
        race_feat_dim: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim

        # Identity-like node types: one learned vector per (entity, season).
        self.driver_emb = nn.Embedding(num_driver_season, state_dim)
        self.constructor_emb = nn.Embedding(num_constructor_season, state_dim)

        # Static node types: project the raw numeric features up to state_dim.
        self.circuit_proj = nn.Sequential(
            nn.Linear(circuit_feat_dim, state_dim), nn.ReLU()
        )
        self.race_proj = nn.Sequential(
            nn.Linear(race_feat_dim, state_dim), nn.ReLU()
        )

        # Message passing stack. All node types are projected to ``state_dim``
        # (see ``node_features``), so we use explicit in_channels rather than
        # the lazy ``(-1, -1)`` form — lazy SAGEConv would leave parameters
        # uninitialised until the first forward, and any parameter *counting*
        # before that (e.g. logging n_params) would raise.
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {et: SAGEConv(state_dim, state_dim) for et in EDGE_TYPES},
                    aggr="mean",
                )
                for _ in range(num_layers)
            ]
        )

        # Edge readout: [driver, constructor, race, circuit] -> scalar.
        self.readout = nn.Sequential(
            nn.Linear(4 * state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def node_features(
        self, static: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Assemble the initial ``x_dict`` fed to the GNN.

        ``static`` carries the ``circuit`` and ``race`` feature tensors (from
        :attr:`TemporalGraph.static`); identity nodes use their embeddings.
        """
        return {
            "driver_season": self.driver_emb.weight,
            "constructor_season": self.constructor_emb.weight,
            "circuit": self.circuit_proj(static["circuit"]),
            "race": self.race_proj(static["race"]),
        }

    def encode(self, x_dict: dict[str, torch.Tensor], edge_index_dict) -> dict[str, torch.Tensor]:
        """Run the message-passing stack, returning refined node embeddings."""
        out = x_dict
        for conv in self.convs:
            out = conv(out, edge_index_dict)
            out = {k: F.relu(v) for k, v in out.items()}
        return out

    def readout_from(
        self,
        x_dict: dict[str, torch.Tensor],
        driver_idx: torch.Tensor,
        constructor_idx: torch.Tensor,
        race_idx: torch.Tensor,
        circuit_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict position_norm for the given (driver, constructor, race,
        circuit) node tuples. All four index tensors share the same length.

        The constructor index is the *intervention point*: swapping it for a
        different team's node produces the counterfactual.
        """
        h = torch.cat(
            [
                x_dict["driver_season"][driver_idx],
                x_dict["constructor_season"][constructor_idx],
                x_dict["race"][race_idx],
                x_dict["circuit"][circuit_idx],
            ],
            dim=-1,
        )
        return self.readout(h).squeeze(-1)

    def forward(self, data, static: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convenience: build features, run GNN, return refined ``x_dict``."""
        x_dict = self.node_features(static)
        return self.encode(x_dict, data.edge_index_dict)
