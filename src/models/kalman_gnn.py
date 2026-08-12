"""
Kalman-GNN: Temporal State-Space Model for F1 Driver Skill Evaluation.

Each driver and constructor has a latent embedding that evolves smoothly over
time, race-by-race, via a Kalman-style additive update driven by a GNN that
processes a sliding window of recent races.

Architecture:
  v_driver_r = v_driver_{r-1} + delta * tanh(W * [v_{r-1}, h_r])
  skill = w_skill · v
  P(A beats B) = sigma(skill_A - skill_B)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.graph_encoder import HeteroGraphEncoder


class KalmanCell(nn.Module):
    """Smooth additive state update: v_r = v_{r-1} + delta * tanh(W * [v_{r-1}, h_r]).

    ``delta`` is a learned scalar controlling the maximum update magnitude per
    step.  When ``h_r = 0`` the update is ``delta * tanh(W_proj * v_{r-1})``,
    so the embedding of an idle entity drifts by at most ``delta`` per step.
    """

    def __init__(self, state_dim: int, msg_dim: int):
        super().__init__()
        self.proj = nn.Linear(state_dim + msg_dim, state_dim)
        self.delta = nn.Parameter(torch.tensor(0.1))

    def forward(self, v_prev: torch.Tensor, h_r: torch.Tensor) -> torch.Tensor:
        """Apply one Kalman update step.

        Args:
            v_prev: (N, state_dim) -- previous embeddings for all entities.
            h_r: (N, msg_dim) -- GNN message for each entity (zero for inactive).

        Returns:
            (N, state_dim) -- updated embeddings.
        """
        combined = torch.cat([v_prev, h_r], dim=-1)
        update = self.delta * torch.tanh(self.proj(combined))
        return v_prev + update


class KalmanGNNPipeline(nn.Module):
    """Top-level model combining a GNN encoder, Kalman state evolution, and a
    scalar skill readout for beat-teammate prediction.

    Reuses the existing ``HeteroGraphEncoder`` (SAGEConv-based) and, when
    ``node_to_col_names_dict``/``node_to_col_stats`` are provided, a RelBench
    ``HeteroEncoder`` for static node features.
    """

    def __init__(
        self,
        num_drivers: int,
        num_constructors: int,
        num_nodes_dict: dict,
        state_dim: int = 32,
        msg_dim: int = 8,
        node_to_col_names_dict: dict | None = None,
        node_to_col_stats: dict | None = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.msg_dim = msg_dim

        # --- Static node feature encoder (optional, same as current pipeline) ---
        if node_to_col_names_dict is not None and node_to_col_stats is not None:
            from relbench.modeling.nn import HeteroEncoder

            self.encoder = HeteroEncoder(
                channels=msg_dim,
                node_to_col_names_dict=node_to_col_names_dict,
                node_to_col_stats=node_to_col_stats,
            )
        else:
            self.encoder = None

        # --- GNN encoder (reused from existing codebase) ---
        self.graph_encoder = HeteroGraphEncoder(
            num_nodes_dict=num_nodes_dict,
            out_dim=msg_dim,
        )

        # --- Kalman cells (separate for drivers and constructors) ---
        self.driver_kalman = KalmanCell(state_dim, msg_dim)
        self.constructor_kalman = KalmanCell(state_dim, msg_dim)

        # --- Projection from msg_dim to state_dim for v0 initialization ---
        self.driver_proj = nn.Linear(msg_dim, state_dim, bias=False)
        self.constructor_proj = nn.Linear(msg_dim, state_dim, bias=False)

        # --- Learnable initial embeddings v_0 ---
        self.v0_drivers = nn.Parameter(torch.randn(num_drivers, state_dim) * 0.1)
        self.v0_constructors = nn.Parameter(torch.randn(num_constructors, state_dim) * 0.1)

        # --- Skill readout (scalar, interpretable) ---
        self.skill_head = nn.Linear(state_dim, 1, bias=False)

        # --- Pre-race input encoder (qualifying_position + grid) ---
        # Encoded separately so skill is purely driver ability, not race context.
        self.context_encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def get_initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (v_drivers, v_constructors) initialised to ``v0``."""
        return self.v0_drivers.clone(), self.v0_constructors.clone()

    def encode_static_features(self, graph_tf_dict: dict) -> dict[str, torch.Tensor]:
        """Encode static node features once (no temporal variation)."""
        if self.encoder is not None:
            return self.encoder(graph_tf_dict)
        return {}

    @torch.no_grad()
    def encode_static_features_nograd(self, graph_tf_dict: dict) -> dict[str, torch.Tensor]:
        """Encode static node features with no gradient tracking."""
        if self.encoder is not None:
            was_training = self.encoder.training
            self.encoder.eval()
            out = self.encoder(graph_tf_dict)
            if was_training:
                self.encoder.train()
            return out
        return {}

    def forward_step(
        self,
        v_drivers_prev: torch.Tensor,
        v_constructors_prev: torch.Tensor,
        graph_x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        active_driver_ids: torch.Tensor,
        active_constructor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single chronological step: GNN → Kalman update → updated embeddings.

        Args:
            v_drivers_prev: (num_drivers, state_dim) -- previous driver embeddings.
            v_constructors_prev: (num_constructors, state_dim) -- previous constructor embeddings.
            graph_x_dict: static node features (or None to use encoder).
            edge_index_dict: GNN edges for the current window.
            active_driver_ids: indices of drivers participating in the current race.
            active_constructor_ids: indices of constructors participating in the current race.

        Returns:
            (v_drivers_new, v_constructors_new) -- updated embeddings.
        """
        # 1. GNN forward pass on current window
        gnn_out = self.graph_encoder(graph_x_dict, edge_index_dict)
        h_drivers = gnn_out.get("drivers")  # (num_drivers, msg_dim)
        h_constructors = gnn_out.get("constructors")  # (num_constructors, msg_dim)

        device = v_drivers_prev.device

        # 2. Build full message tensors (zero for inactive entities)
        h_drivers_full = torch.zeros(v_drivers_prev.shape[0], self.msg_dim, device=device)
        if h_drivers is not None and len(active_driver_ids) > 0:
            h_drivers_full[active_driver_ids] = h_drivers[active_driver_ids]

        h_constructors_full = torch.zeros(
            v_constructors_prev.shape[0], self.msg_dim, device=device
        )
        if h_constructors is not None and len(active_constructor_ids) > 0:
            h_constructors_full[active_constructor_ids] = h_constructors[active_constructor_ids]

        # 3. Kalman update for all entities
        v_drivers_new = self.driver_kalman(v_drivers_prev, h_drivers_full)
        v_constructors_new = self.constructor_kalman(v_constructors_prev, h_constructors_full)

        return v_drivers_new, v_constructors_new

    def compute_skill(
        self, v_drivers: torch.Tensor, v_constructors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract scalar skill scores from embeddings.

        Returns:
            (skill_drivers, skill_constructors) -- each (N, 1).
        """
        return self.skill_head(v_drivers), self.skill_head(v_constructors)

    def predict_teammate(
        self,
        v_drivers: torch.Tensor,
        driver_a_ids: torch.Tensor,
        driver_b_ids: torch.Tensor,
        qualifying_a: torch.Tensor | None = None,
        qualifying_b: torch.Tensor | None = None,
        grid_a: torch.Tensor | None = None,
        grid_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict P(driver_A beats driver_B) for teammate pairs.

        The prediction is sigma(skill_A - skill_B + context_delta), where
        context_delta captures the qualifying/grid advantage of A over B.

        Args:
            v_drivers: (num_drivers, state_dim) -- current driver embeddings.
            driver_a_ids: (N_pairs,) -- indices of first driver in each pair.
            driver_b_ids: (N_pairs,) -- indices of second driver in each pair.
            qualifying_a, qualifying_b: (N_pairs,) or None -- qualifying positions.
            grid_a, grid_b: (N_pairs,) or None -- starting grid positions.

        Returns:
            (N_pairs,) -- logits (skill_A - skill_B + context).
        """
        skill = self.skill_head(v_drivers).squeeze(-1)  # (num_drivers,)
        skill_a = skill[driver_a_ids]
        skill_b = skill[driver_b_ids]
        logit = skill_a - skill_b

        # Add pre-race context if available
        if qualifying_a is not None and qualifying_b is not None:
            context_a = torch.stack([qualifying_a, grid_a if grid_a is not None else qualifying_a], dim=-1)
            context_b = torch.stack([qualifying_b, grid_b if grid_b is not None else qualifying_b], dim=-1)
            context_delta = self.context_encoder(context_a) - self.context_encoder(context_b)
            logit = logit + context_delta.squeeze(-1)

        return logit