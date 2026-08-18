"""Diagnostic: does the ``driver_season`` embedding carry any skill signal?

The decisive test is **teammate discrimination**. For every (race, team) with
two drivers who both finished, we predict the finishing order using *only* the
driver embeddings — holding the constructor, race, and circuit embeddings
identical (they are, by construction, the same for both teammates). The ONLY
thing that differs between the two predictions is the ``driver_season`` node.

If the model has learned anything about drivers (as opposed to cars), it must
be able to tell which teammate finished ahead from the driver embedding alone,
so AUROC of the predicted ordering should be > 0.5. AUROC ≈ 0.5 means the
driver embedding is dead — the same failure as the Kalman ``skill_head``.

Run on the remote box after training:

    python -m src.experiments.diagnose_driver_signal

Reports AUROC / accuracy per split (train / val / test) plus a driver-clustered
bootstrap CI on the held-out (val+test) pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import roc_auc_score, accuracy_score

import config as cfg
from validation.counterfactual_skill import (
    DEFAULT_CHECKPOINT,
    load_counterfactual_model,
)


def build_teammate_pairs(raced_in: pd.DataFrame) -> pd.DataFrame:
    """Form (A, B) pairs of teammates within the same (race, team).

    Returns a DataFrame with one row per ordered pair where A finished ahead
    of B:
        [race, constructor_season, circuit, driver_A, driver_B, driverA_id,
         driverB_id, year]
    """
    rows = []
    for (race, cs), grp in raced_in.groupby(["race", "constructor_season"]):
        grp = grp.sort_values("positionOrder")
        drv = grp["driver_season"].tolist()
        drv_id = grp["driverId"].tolist()
        circuit = int(grp["circuit"].iloc[0])
        year = int(grp["year"].iloc[0])
        n = len(drv)
        for i in range(n):
            for j in range(i + 1, n):
                # i finished ahead of j (lower positionOrder).
                rows.append(
                    {
                        "race": int(race),
                        "constructor_season": int(cs),
                        "circuit": circuit,
                        "driver_A": int(drv[i]),
                        "driver_B": int(drv[j]),
                        "driverA_id": int(drv_id[i]),
                        "driverB_id": int(drv_id[j]),
                        "year": year,
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def score_pairs(model, x_dict, pairs: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Predicted probability that A (finished ahead) beats B, per pair.

    score > 0.5 means the model predicts A ahead of B, matching the label.
    """
    out = []
    for _, p in pairs.iterrows():
        pred_A = model.readout_from(
            x_dict,
            torch.tensor([p["driver_A"]], dtype=torch.long, device=device),
            torch.tensor([p["constructor_season"]], dtype=torch.long, device=device),
            torch.tensor([p["race"]], dtype=torch.long, device=device),
            torch.tensor([p["circuit"]], dtype=torch.long, device=device),
        )
        pred_B = model.readout_from(
            x_dict,
            torch.tensor([p["driver_B"]], dtype=torch.long, device=device),
            torch.tensor([p["constructor_season"]], dtype=torch.long, device=device),
            torch.tensor([p["race"]], dtype=torch.long, device=device),
            torch.tensor([p["circuit"]], dtype=torch.long, device=device),
        )
        # Lower predicted position == predicted to finish ahead. B's predicted
        # position minus A's: positive => A ahead (matches label 1).
        out.append((pred_B - pred_A).item())
    return np.asarray(out, dtype=float)


def _cluster_bootstrap_auroc(
    scores: np.ndarray, labels: np.ndarray, driver_ids: np.ndarray,
    n_bootstrap: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """AUROC CI resampling drivers (not pairs), so one driver's many pairs
    count as one unit of information."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(driver_ids)
    idx_by_drv = {d: np.where(driver_ids == d)[0] for d in uniq}
    reps = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_drv[d] for d in picks])
        if len(np.unique(labels[idx])) < 2:
            reps[b] = np.nan
            continue
        reps[b] = roc_auc_score(labels[idx], scores[idx])
    reps = reps[~np.isnan(reps)]
    if reps.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    model, graph, x_dict, device = load_counterfactual_model(checkpoint, args.device)

    fr = graph.raced_in.copy()
    fr["driverId"] = fr["driver_season"].map(
        graph.driver_season.set_index("node_idx")["driverId"]
    )

    pairs = build_teammate_pairs(fr)
    if pairs.empty:
        print("No teammate pairs found — check the graph build.")
        return

    scores = score_pairs(model, x_dict, pairs, device)
    labels = np.ones(len(pairs), dtype=int)  # A always finished ahead by construction
    years = pairs["year"].to_numpy()

    # Split masks over the pairs by year.
    split_years = {
        "train": cfg.TRAIN_YEARS,
        "val": cfg.VAL_YEARS,
        "test": cfg.TEST_YEARS,
    }

    print("\n" + "=" * 60)
    print("DRIVER-SIGNAL DIAGNOSTIC — teammate discrimination")
    print("=" * 60)
    print(f"n_pairs={len(pairs)}  (each pair: same team, same race, same car)")

    for split_name, yrs in split_years.items():
        m = np.isin(years, yrs)
        if m.sum() == 0 or len(np.unique(labels[m])) < 2:
            print(f"\n[{split_name}] n_pairs={m.sum()} — insufficient, skipped")
            continue
        auroc = roc_auc_score(labels[m], scores[m])
        # Accuracy at the natural 0-threshold: predicted A ahead iff score > 0.
        acc = accuracy_score(labels[m], (scores[m] > 0).astype(int))
        print(
            f"\n[{split_name}] n_pairs={m.sum()}  AUROC={auroc:.4f}  "
            f"accuracy={acc:.4f}  (chance = 0.50)"
        )

    # Held-out (val+test) cluster-bootstrap CI by driver.
    held = np.isin(years, cfg.VAL_YEARS + cfg.TEST_YEARS)
    if held.sum() >= 10 and len(np.unique(labels[held])) >= 2:
        lo, hi = _cluster_bootstrap_auroc(
            scores[held], labels[held], pairs["driverA_id"].to_numpy()[held]
        )
        auroc_held = roc_auc_score(labels[held], scores[held])
        print(
            f"\n[val+test] AUROC={auroc_held:.4f}  driver-cluster 95% CI "
            f"[{lo:.4f}, {hi:.4f}]"
        )

    print("\nInterpretation:")
    print("  AUROC ~= 0.50  -> driver_season embedding is dead (same disease as")
    print("                     the Kalman skill_head): the model predicts the car,")
    print("                     not the driver. The swap is averaging noise.")
    print("  AUROC >  0.55  -> there IS driver signal; the problem is elsewhere")
    print("                     (swap aggregation / support filtering).")


if __name__ == "__main__":
    main()
