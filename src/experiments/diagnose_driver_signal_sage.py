"""Driver-signal diagnostic for the original SAGE-GNN pipeline.

Mirrors :mod:`src.experiments.diagnose_driver_signal` but adapts the test to
the older :class:`F1OrthogonalPipeline` architecture (single embedding per
driver, no season meta-nodes).

The teammate discrimination test
--------------------------------
For every pair of drivers who raced in the same (race, constructor), the
constructor embedding, race identity, and (crucially) qualifying position and
grid are all identical or near-identical. The **only** thing that reliably
differs is the driver — so if the driver embedding carries any skill signal,
``aux_piloto(emb_driver)`` alone should rank the teammate who finished ahead
higher than the one who finished behind.

Metric: pairwise accuracy = P(model ranks the actual winner ahead), with a
binomial p-value against chance = 0.50 and a driver-clustered bootstrap CI.
Chance (0.50) with the CI covering 0.50 means the ``drivers`` embedding is
dead — the same failure mode diagnosed in the Kalman-GNN skill_head.

Run:
    python -m src.experiments.diagnose_driver_signal_sage
    python -m src.experiments.diagnose_driver_signal_sage \\
        --checkpoint output/models/model_no_orthogonal.pth  # ablation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for _p in (ROOT_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as cfg
from models.pipeline_fusion import F1OrthogonalPipeline

DEFAULT_CHECKPOINT = "output/models/model_orthogonal.pth"


def _cluster_bootstrap_accuracy(
    correct: np.ndarray, driver_ids: np.ndarray,
    n_bootstrap: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """Driver-clustered 95% CI on pairwise accuracy."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(driver_ids)
    idx_by_drv = {d: np.where(driver_ids == d)[0] for d in uniq}
    reps = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_drv[d] for d in picks])
        reps[b] = correct[idx].mean()
    return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def _binomial_p_two_sided(n_correct: int, n: int) -> float:
    from scipy.stats import binomtest
    return float(binomtest(n_correct, n, p=0.5, alternative="two-sided").pvalue)


def _build_teammate_pairs(db) -> pd.DataFrame:
    """Ordered (A, B) teammate pairs within (race, constructor), where A
    finished ahead of B (smaller positionOrder).

    Note: uses the *raw* enriched DB (not the task DB), because the task
    strips ``positionOrder`` and every other outcome column via
    ``remove_columns``. The raw DB is the same one used for support scoring
    elsewhere in the framework, so ``driverId`` / ``constructorId`` spaces
    match the model's.

    Columns: [driverId_A, driverId_B, constructorId, raceId, year].
    """
    results_df = db.table_dict["results"].df
    order_col = None
    for candidate in ("positionOrder", "position"):
        if candidate in results_df.columns:
            order_col = candidate
            break
    if order_col is None:
        raise RuntimeError(
            "Neither 'positionOrder' nor 'position' found in results table — "
            "cannot form ordered teammate pairs."
        )
    if order_col != "positionOrder":
        print(f"[diag] positionOrder missing — using '{order_col}' for pair order")

    results = results_df[["driverId", "constructorId", "raceId", order_col]].copy()
    results = results.dropna(subset=[order_col])
    results = results.rename(columns={order_col: "_order"})
    races = db.table_dict["races"].df[["raceId", "year"]]
    results = results.merge(races, on="raceId", how="left")

    rows = []
    for (rid, cid), grp in results.groupby(["raceId", "constructorId"]):
        grp = grp.sort_values("_order")
        drv = grp["driverId"].astype(int).tolist()
        year = int(grp["year"].iloc[0])
        n = len(drv)
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "driverId_A": drv[i],
                    "driverId_B": drv[j],
                    "constructorId": int(cid),
                    "raceId": int(rid),
                    "year": year,
                })
    return pd.DataFrame(rows)


@torch.no_grad()
def _driver_scores(checkpoint_path: str, device: torch.device):
    """Return scalar skill score per driver via ``aux_piloto(emb_driver)``.
    Position/positionOrder targets are negated so higher = better skill,
    matching the "beat teammate" label direction.

    Uses the training-time schema snapshot (``graph_meta.pt``) whenever
    present so the checkpoint's numerical projections load without a shape
    mismatch (the current task may strip columns the training task kept).
    """
    from train import build_graph, get_active_task

    task, _ = get_active_task()
    db = task.dataset.get_db(upto_test_timestamp=False)
    graph_data, node_col_names, node_col_stats = build_graph(db)

    # Prefer the training-time schema, if we have it — the current task's
    # remove_columns may be tighter than the one used when the checkpoint
    # was fit (e.g. results kept 'grid' AND 'number' at train time, only
    # 'grid' now), producing a size mismatch on numerical layers.
    meta_path = "output/models/graph_meta.pt"
    if os.path.exists(meta_path):
        print(f"Loading training-time schema from {meta_path}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        node_col_names = meta.get("node_to_col_names_dict", node_col_names)
        node_col_stats = meta.get("node_to_col_stats", node_col_stats)

    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    aux_w = ckpt.get("aux_piloto.weight")
    if aux_w is None:
        raise RuntimeError(
            f"{checkpoint_path} has no aux_piloto.weight — not an "
            "F1OrthogonalPipeline checkpoint."
        )
    latent_dim = int(aux_w.shape[1])

    model = F1OrthogonalPipeline(
        num_nodes_dict=num_nodes_dict,
        latent_dim=latent_dim,
        node_to_col_names_dict=node_col_names,
        node_to_col_stats=node_col_stats,
    ).to(device)

    graph_data = graph_data.to(device)
    dummy = torch.zeros(1, dtype=torch.long, device=device)
    dummy_f = torch.zeros(1, dtype=torch.float32, device=device)
    _ = model(
        graph_x_dict=None,
        graph_edge_index_dict=graph_data.edge_index_dict,
        target_constructor_ids=dummy,
        target_driver_ids=dummy,
        qualifying_position=dummy_f,
        grid=dummy_f,
        graph_tf_dict=graph_data.tf_dict,
    )

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if unexpected:
        print(f"[diag] ignoring unexpected keys: {sorted(unexpected)}")
    model.eval()

    x_dict = model.encoder(graph_data.tf_dict) if model.encoder is not None else None
    out_dict = model.graph_encoder(x_dict, graph_data.edge_index_dict)
    driver_emb = out_dict["drivers"]
    scores = model.aux_piloto(driver_emb).squeeze(-1)

    target_col = getattr(task, "target_col", None)
    if target_col in ("position", "positionOrder"):
        # Lower is better in the target -> flip so higher-score = better.
        scores = -scores

    return scores.detach().cpu().numpy(), db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading SAGE-GNN checkpoint: {args.checkpoint}")
    scores, _ = _driver_scores(args.checkpoint, device)

    # Use the raw enriched DB (task strips positionOrder / other outcome cols).
    from data.enriched_dataset import EnrichedF1Dataset
    raw_db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)

    print("Building teammate pairs...")
    pairs = _build_teammate_pairs(raw_db)
    if pairs.empty:
        print("No teammate pairs found.")
        return

    # Score margin: A - B. Positive => model predicts A > B (matches label).
    n_drivers = scores.shape[0]
    keep = (pairs["driverId_A"] < n_drivers) & (pairs["driverId_B"] < n_drivers)
    dropped = (~keep).sum()
    if dropped:
        print(f"[diag] dropped {int(dropped)} pairs with out-of-range driverId")
    pairs = pairs[keep].reset_index(drop=True)

    margins = scores[pairs["driverId_A"].to_numpy()] - scores[pairs["driverId_B"].to_numpy()]
    correct = (margins > 0).astype(float)
    years = pairs["year"].to_numpy()

    print("\n" + "=" * 62)
    print("DRIVER-SIGNAL DIAGNOSTIC (SAGE-GNN) — teammate discrimination")
    print("=" * 62)
    print(f"n_pairs={len(pairs)}  metric: pairwise accuracy on aux_piloto(emb_driver)")
    print("chance = 0.50; > 0.55 (CI excludes 0.5) = real driver signal\n")

    split_years = {
        "train": cfg.TRAIN_YEARS,
        "val": cfg.VAL_YEARS,
        "test": cfg.TEST_YEARS,
    }
    for name, yrs in split_years.items():
        m = np.isin(years, yrs)
        n = int(m.sum())
        if n < 20:
            print(f"[{name}] n_pairs={n} — too few, skipped")
            continue
        acc = float(correct[m].mean())
        n_correct = int(correct[m].sum())
        p = _binomial_p_two_sided(n_correct, n)
        print(f"[{name}] n_pairs={n}  accuracy={acc:.4f}  binomial p={p:.4g}")

    held = np.isin(years, list(cfg.VAL_YEARS) + list(cfg.TEST_YEARS))
    n_held = int(held.sum())
    if n_held >= 20:
        lo, hi = _cluster_bootstrap_accuracy(
            correct[held], pairs["driverId_A"].to_numpy()[held]
        )
        acc_held = float(correct[held].mean())
        print(
            f"\n[val+test] n_pairs={n_held}  accuracy={acc_held:.4f}  "
            f"driver-cluster 95% CI [{lo:.4f}, {hi:.4f}]"
        )

    # Extra: how much does the driver-embedding score actually move? If the
    # readout is essentially random, |margin| will be tiny relative to score
    # variance. This is the analog of the Kalman-GNN "norm ≈ random init" check.
    print("\nSanity — score dispersion:")
    print(f"  scores  mean={scores.mean():+.4f}  std={scores.std():.4f}  "
          f"range=[{scores.min():+.4f}, {scores.max():+.4f}]")
    print(f"  |margin| mean={np.abs(margins).mean():.4f}  "
          f"median={np.median(np.abs(margins)):.4f}")

    print("\nInterpretation:")
    print("  accuracy ~= 0.50 (CI covers 0.5) -> driver embedding is dead")
    print("      (same disease as Kalman skill_head): swap/counterfactual can't")
    print("      work on this encoder without extra supervision.")
    print("  accuracy >  0.55 (CI excludes 0.5) -> driver signal is real; the")
    print("      SAGE-GNN + aux_piloto head has learned something about drivers.")


if __name__ == "__main__":
    main()
