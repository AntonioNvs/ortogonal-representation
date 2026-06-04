import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphEncoder(nn.Module):
    """
    Unified heterogeneous GNN encoder that produces embeddings for BOTH
    constructors (team) and driver node types from the same graph.

    Two HeteroConv layers with SAGEConv:
      - conv1: aggregated with 'mean'  (captures average neighbor behavior)
      - conv2: aggregated with 'max'   (captures extreme/peak performances)

    Edge types covered:
      - results → constructors
      - qualifying → constructors
      - constructor_standings → constructors
      - results → driver
      - qualifying → driver

    Uses `x_dict` node features directly (from `make_pkey_fkey_graph`),
    NOT all-ones placeholders.
    """

    def __init__(self, num_nodes_dict, hidden_dim=32, out_dim=8):
        super().__init__()

        # Edge types relevant to constructors (team)
        cons_edges = [
            ("results", "f2p_constructorId", "constructors"),
            ("qualifying", "f2p_constructorId", "constructors"),
            ("constructor_standings", "f2p_constructorId", "constructors"),
        ]
        # Edge types relevant to driver
        driver_edges = [
            ("results", "f2p_driverId", "driver"),
            ("qualifying", "f2p_driverId", "driver"),
        ]
        all_edges = cons_edges + driver_edges

        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in all_edges},
            aggr="mean",
        )
        self.conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), out_dim) for et in all_edges},
            aggr="max",
        )

        self.ln_cons = nn.LayerNorm(out_dim)
        self.ln_drv = nn.LayerNorm(out_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Node features per type (from graph_data.x_dict).
        edge_index_dict : dict[tuple, Tensor]
            Edge indices per (src, rel, dst) type.

        Returns
        -------
        dict[str, Tensor]
            Output embeddings per node type, same dict shape as x_dict.
        """
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {key: x.relu() for key, x in out_dict.items()}

        # Residual-like: nodes not updated by conv1 keep their input features.
        for key, h in x_dict.items():
            if key not in out_dict:
                out_dict[key] = h

        out_dict = self.conv2(out_dict, edge_index_dict)

        # Apply LayerNorm to our target node types
        if "constructors" in out_dict:
            out_dict["constructors"] = self.ln_cons(out_dict["constructors"])
        if "driver" in out_dict:
            out_dict["driver"] = self.ln_drv(out_dict["driver"])

        return out_dict
