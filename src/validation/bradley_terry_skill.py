"""Walk-forward Bradley-Terry skill scorer (leak-free).

The career-validation framework correlates ``skill(driver, T)`` against the
*forward* outcome (mean tier of the driver's teams in T+1..T+horizon). For that
correlation to be honest, ``skill(driver, T)`` must use **only data <= T**.

Because Bradley-Terry's ``theta`` is a single time-invariant scalar per driver,
the naive fit-on-everything version would leak: ``theta`` fit on all years
would have "seen" T+1..T+horizon. This adapter avoids that with an
**expanding-window walk-forward**: for each season T it fits the model on all
races with ``year <= T`` (warm-starting from the previous season's fit) and
snapshots ``theta``. Each ``skill(driver, T)`` is therefore fit on the driver's
history up to T only — exactly the causal quantity the framework needs.

Runs the walk-forward on the fly (no checkpoint needed); the model is tiny
(~900 scalar parameters) and each step converges in a few epochs.

Usage:
    python -m src.experiments.career_validation --skill-source bradley_terry \\
        --require-full-horizon
"""

from __future__ import annotations

import pandas as pd
import torch

import config as cfg
from counterfactual.driver_pairs import build_race_pairs, driver_id_to_index
from counterfactual.support import compute_support
from data.enriched_dataset import EnrichedF1Dataset
from data.temporal_graph import build_temporal_graph
from models.bradley_terry import BradleyTerry


def load_bradley_terry_skill(device=None) -> pd.DataFrame:
    """Walk-forward Bradley-Terry skill scores.

    Returns columns: ``[driverId, season, skill_score, support_score,
    support_bucket]``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    lr = getattr(cfg, "BT_LR", 0.1)
    epochs_per_step = getattr(cfg, "BT_EPOCHS_PER_STEP", 30)

    print("Loading enriched F1 database...")
    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)

    print("Building temporal meta-node graph...")
    graph = build_temporal_graph(db)

    pairs = build_race_pairs(graph)
    idx = driver_id_to_index(pairs)
    num_drivers = len(idx)
    num_cs = graph.num_constructor_seasons
    print(f"  drivers={num_drivers}, constructor_seasons={num_cs}, pairs={len(pairs)}")

    # Tensors for the likelihood (labels are all 1: A finished ahead).
    driver_A = torch.tensor([idx[d] for d in pairs["driverA"]], dtype=torch.long, device=device)
    driver_B = torch.tensor([idx[d] for d in pairs["driverB"]], dtype=torch.long, device=device)
    cs_A = torch.tensor(pairs["cs_A"].to_numpy(), dtype=torch.long, device=device)
    cs_B = torch.tensor(pairs["cs_B"].to_numpy(), dtype=torch.long, device=device)
    labels = torch.ones(len(pairs), device=device)
    years = pairs["year"].to_numpy()

    # Active drivers per season (to snapshot only drivers who actually raced).
    active_by_year: dict[int, set[int]] = {}
    fr = graph.raced_in[["driver_season", "year"]].copy()
    fr["driverId"] = fr["driver_season"].map(
        graph.driver_season.set_index("node_idx")["driverId"]
    )
    for year, grp in fr.groupby("year"):
        active_by_year[int(year)] = set(grp["driverId"].astype(int))

    model = BradleyTerry(num_drivers, num_cs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rows = []
    for T in sorted(active_by_year):
        mask = years <= T
        # Warm-start: a few epochs on all data <= T (accumulates from prior T).
        for _ in range(epochs_per_step):
            logits = model.logits(
                driver_A[mask], driver_B[mask], cs_A[mask], cs_B[mask]
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels[mask]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        theta = model.driver_skill().detach().cpu().numpy()
        for d in sorted(active_by_year[T]):
            rows.append(
                {
                    "driverId": d,
                    "season": T,
                    "skill_score": float(theta[idx[d]]),
                }
            )

    skill = pd.DataFrame(rows).sort_values(["driverId", "season"]).reset_index(drop=True)

    # Support score: transfers/seasons history (shared with the other scorers).
    support = compute_support(graph)
    merged = skill.merge(support, on=["driverId", "season"], how="left")
    return merged[["driverId", "season", "skill_score", "support_score", "support_bucket"]]
