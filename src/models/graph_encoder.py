import torch
import torch.nn as nn
import warnings
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
    """

    def __init__(self, num_nodes_dict, hidden_dim=32, out_dim=8):
        super().__init__()

        # Edge types relevant to constructors (team)
        cons_edges = [
            ("results", "f2p_constructorId", "constructors"),
            ("qualifying", "f2p_constructorId", "constructors"),
            ("constructor_standings", "f2p_constructorId", "constructors"),
        ]
        # Edge types relevant to drivers
        driver_edges = [
            ("results", "f2p_driverId", "drivers"),
            ("qualifying", "f2p_driverId", "drivers"),
        ]
        all_edges = cons_edges + driver_edges

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message=".*There exist node types.*whose representations do not get updated during message passing.*"
            )
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
        if "drivers" in out_dict:
            out_dict["drivers"] = self.ln_drv(out_dict["drivers"])

        return out_dict

    def compute_paired_orthogonal_loss(self, out_dict, edge_index_dict):
        """
        Computes the orthogonal loss ONLY for driver-constructor pairs that 
        participated in the exact same event (result node).

        Parameters
        ----------
        out_dict : dict[str, Tensor]
            The output dictionary from the forward pass.
        edge_index_dict : dict[tuple, Tensor]
            The graph edge indices.

        Returns
        -------
        Tensor
            A scalar tensor representing the paired orthogonal loss.
        """
        z_drv = out_dict.get("drivers")
        z_cons = out_dict.get("constructors")

        if z_drv is None or z_cons is None:
            return torch.tensor(0.0, device=z_drv.device if z_drv is not None else torch.device('cpu'))

        # 1. Extract the specific edge sets
        res_drv = edge_index_dict.get(("results", "f2p_driverId", "drivers"))
        res_cons = edge_index_dict.get(("results", "f2p_constructorId", "constructors"))

        if res_drv is None or res_cons is None:
            return torch.tensor(0.0, device=z_drv.device)

        # 2. Vectorized Inner Join on the "results" node ID (row 0)
        # Find the maximum result node ID to size our mapping array
        max_res_id = max(res_drv[0].max().item(), res_cons[0].max().item()) + 1

        # Initialize mapping arrays with -1 (meaning "no connection found")
        device = z_drv.device
        res_to_drv = torch.full((max_res_id,), -1, dtype=torch.long, device=device)
        res_to_cons = torch.full((max_res_id,), -1, dtype=torch.long, device=device)

        # Populate mappings: index is result_id, value is driver_id / constructor_id
        res_to_drv[res_drv[0]] = res_drv[1]
        res_to_cons[res_cons[0]] = res_cons[1]

        # Filter for results that have BOTH a valid driver and a valid constructor mapping
        valid_mask = (res_to_drv != -1) & (res_to_cons != -1)

        valid_drv_indices = res_to_drv[valid_mask]
        valid_cons_indices = res_to_cons[valid_mask]

        # 3. Gather the matched embeddings
        z_drv_paired = z_drv[valid_drv_indices]
        z_cons_paired = z_cons[valid_cons_indices]

        # 4. Compute Orthogonal Loss (dot product squared)
        # We want the dot product between the paired vectors to be close to 0
        dot_products = torch.sum(z_drv_paired * z_cons_paired, dim=1)
        
        # Mean of squared dot products across all valid pairs in the batch
        paired_ortho_loss = torch.mean(dot_products ** 2)

        return paired_ortho_loss