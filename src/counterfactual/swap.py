"""Counterfactual driver-in-car swap at inference time.

Given a trained :class:`HeteroRacePredictor` and its refined node embeddings,
the scalar skill of a driver in a season is the *expected* race outcome in an
**average car**, marginalising over the constructors actually active that
season and over the season's races:

    skill(X, T) = 1 - mean_{Y in constructors(T), r in races(T)}
                        f(emb(X@T), emb(Y@T), emb(race_r), emb(circuit_r))

The only intervention is substituting the ``constructor_season`` node in the
readout — the driver, race, and circuit embeddings are untouched. This is what
separates "driver skill" from "the car X happened to drive". ``1 - position``
maps "finishes first" to a higher score, so higher = better (matching the
career-validation framework's convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data.temporal_graph import TemporalGraph
from models.hetero_race_predictor import HeteroRacePredictor


def compute_counterfactual_skill(
    model: HeteroRacePredictor,
    graph: TemporalGraph,
    x_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> pd.DataFrame:
    """Return ``[driverId, season, skill_score]`` for every driver-season node.

    ``x_dict`` is the *refined* node embedding dict (output of
    ``model(data, static)``), already on ``device``.
    """
    fr = graph.raced_in
    race_to_circuit = fr[["race", "circuit"]].drop_duplicates().set_index("race")["circuit"]

    ds = graph.driver_season.set_index("node_idx")

    rows = []
    with torch.no_grad():
        for season, grp in fr.groupby("year", sort=True):
            drivers_T = np.sort(grp["driver_season"].unique())
            constructors_T = np.sort(grp["constructor_season"].unique())
            races_T = np.sort(grp["race"].unique())

            D, C, R = len(drivers_T), len(constructors_T), len(races_T)
            if D == 0 or C == 0 or R == 0:
                continue

            # Cross product drivers x constructors x races, in row-major order
            # (driver, constructor, race) so reshape(D, C*R) groups per driver.
            d_idx = np.repeat(drivers_T, C * R)
            c_idx = np.tile(np.repeat(constructors_T, R), D)
            r_idx = np.tile(races_T, D * C)
            circ_idx = race_to_circuit.loc[r_idx].to_numpy()

            logits = model.readout_from(
                x_dict,
                torch.tensor(d_idx, dtype=torch.long, device=device),
                torch.tensor(c_idx, dtype=torch.long, device=device),
                torch.tensor(r_idx, dtype=torch.long, device=device),
                torch.tensor(circ_idx, dtype=torch.long, device=device),
            ).reshape(D, C * R)

            mean_pos = logits.mean(dim=1)  # per driver: average over (car, race)
            skill = 1.0 - mean_pos

            for i, d in enumerate(drivers_T):
                driver_id = int(ds.loc[d, "driverId"])
                rows.append(
                    {
                        "driverId": driver_id,
                        "season": int(season),
                        "skill_score": float(skill[i].item()),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(["driverId", "season"])
        .reset_index(drop=True)
    )
