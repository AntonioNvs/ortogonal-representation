"""Driver skill readout for the beat-teammate objective.

Once the model is trained on beat-teammate BCE, ``driver_head(emb_driver)``
is *already* car-independent: the objective forced the driver embedding to
discriminate teammates while the car was held identical, so the readout is the
driver's intrinsic skill by construction. No further counterfactual
intervention is needed — the "swap" is implicit in the training signal.

This module exposes that readout as the season-level skill score:

    skill(X, T) = driver_head(emb(driver_season=X@T))

matching the career-validation framework's contract. The position-regression
``readout_from`` path is retained in the model (for the marginal-attribution
branch) but is not used here.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.temporal_graph import TemporalGraph
from models.hetero_race_predictor import HeteroRacePredictor


def compute_driver_skill(
    model: HeteroRacePredictor,
    graph: TemporalGraph,
    x_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> pd.DataFrame:
    """Return ``[driverId, season, skill_score]`` for every driver-season node.

    ``skill_score = driver_head(emb(driver_season))`` — car-independent.
    Higher = better (the head is trained to be higher for the teammate that
    finishes ahead).
    """
    ds = graph.driver_season  # [node_idx, driverId, season]
    node_idx = torch.tensor(ds["node_idx"].to_numpy(), dtype=torch.long, device=device)

    with torch.no_grad():
        skill = model.driver_skill(x_dict, node_idx).cpu().numpy()

    out = pd.DataFrame(
        {
            "driverId": ds["driverId"].to_numpy(),
            "season": ds["season"].to_numpy(),
            "skill_score": skill,
        }
    )
    return out.sort_values(["driverId", "season"]).reset_index(drop=True)


# Backwards-compatible alias: earlier callers used this name for the swap
# aggregation. It now resolves to the direct driver-skill readout.
compute_counterfactual_skill = compute_driver_skill
