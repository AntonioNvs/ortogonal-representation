"""Walk-forward race-level Bradley–Terry benchmark with SkillExport."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

import config as cfg
from baselines.bradley_terry import BradleyTerry
from data.mobility import build_race_pairs_for_bt, compute_support_scores
from data.race_panel import RacePanelConfig, build_race_panel
from relbench.base import Database
from skill.contract import InferenceMode
from skill.export import build_skill_export


def _fit_bt_on_pairs(
    pairs: pd.DataFrame,
    drv_idx: dict,
    cs_idx: dict,
    *,
    device: torch.device,
    lr: float = cfg.BT_LR,
    epochs: int = cfg.BT_EPOCHS_PER_STEP,
) -> BradleyTerry:
    num_drivers = len(drv_idx)
    num_cs = len(cs_idx)
    if pairs.empty or num_drivers == 0:
        model = BradleyTerry(max(num_drivers, 1), max(num_cs, 1)).to(device)
        return model

    p = pairs.copy()
    p["cs_A"] = p.apply(lambda r: cs_idx[(int(r["constructorA"]), int(r["year"]))], axis=1)
    p["cs_B"] = p.apply(lambda r: cs_idx[(int(r["constructorB"]), int(r["year"]))], axis=1)
    p["idx_A"] = p["driverA"].map(drv_idx)
    p["idx_B"] = p["driverB"].map(drv_idx)

    driver_A = torch.tensor(p["idx_A"].to_numpy(), dtype=torch.long, device=device)
    driver_B = torch.tensor(p["idx_B"].to_numpy(), dtype=torch.long, device=device)
    cs_A = torch.tensor(p["cs_A"].to_numpy(), dtype=torch.long, device=device)
    cs_B = torch.tensor(p["cs_B"].to_numpy(), dtype=torch.long, device=device)
    labels = torch.ones(len(p), device=device)

    model = BradleyTerry(num_drivers, num_cs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        logits = model.logits(driver_A, driver_B, cs_A, cs_B)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def export_bradley_terry(
    db: Database,
    *,
    max_year: int = 2025,
    inference_mode: InferenceMode = InferenceMode.FILTERED,
    device=None,
) -> "SkillExport":
    """Expanding-window BT: causal as-of-round race export."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    panel = build_race_panel(db, RacePanelConfig(max_year=max_year))
    ranked = panel[panel["in_race_ranking"]].copy()
    pairs_all = build_race_pairs_for_bt(ranked.rename(columns={"season": "year"}))

    drivers = sorted(ranked["driverId"].astype(int).unique())
    cs_keys = sorted(
        set(zip(ranked["constructorId"].astype(int), ranked["season"].astype(int)))
    )
    drv_idx = {d: i for i, d in enumerate(drivers)}
    cs_idx = {k: i for i, k in enumerate(cs_keys)}

    checkpoints = (
        ranked[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )

    rows = []
    for _, ck in checkpoints.iterrows():
        season, rnd = int(ck["season"]), int(ck["round"])
        hist = ranked[
            (ranked["season"] < season)
            | ((ranked["season"] == season) & (ranked["round"] <= rnd))
        ]
        pairs = build_race_pairs_for_bt(hist.rename(columns={"season": "year"}))
        model = _fit_bt_on_pairs(pairs, drv_idx, cs_idx, device=device)
        theta = model.driver_skill().detach().cpu().numpy()
        q = model.q.weight.squeeze(-1).detach().cpu().numpy()

        current = ranked[(ranked["season"] == season) & (ranked["round"] == rnd)]
        for _, r in current.iterrows():
            did = int(r["driverId"])
            cid = int(r["constructorId"])
            if did not in drv_idx:
                continue
            cs_key = (cid, season)
            q_val = float(q[cs_idx[cs_key]]) if cs_key in cs_idx else 0.0
            t_val = float(theta[drv_idx[did]])
            rows.append(
                {
                    "driverId": did,
                    "season": season,
                    "round": rnd,
                    "raceId": int(r["raceId"]),
                    "constructorId": cid,
                    "lineage_id": r.get("lineage_id", ""),
                    "driver_name": r.get("driver_name", ""),
                    "constructor_name": r.get("constructor_name", ""),
                    "raw_skill": t_val,
                    "contrib_driver": t_val,
                    "contrib_constructor": q_val,
                    "contrib_context": 0.0,
                    "contrib_residual": 0.0,
                    "as_of_round": rnd,
                    "inference_mode": inference_mode.value,
                }
            )

    race_df = pd.DataFrame(rows)
    support = compute_support_scores(ranked.rename(columns={"season": "year"}))
    support = support.rename(columns={"year": "season"})
    race_df = race_df.merge(
        support[["driverId", "season", "support_bucket"]],
        on=["driverId", "season"],
        how="left",
    )
    race_df["support_bucket"] = race_df["support_bucket"].fillna("medium")

    return build_skill_export(
        race_df,
        skill_source="bradley_terry",
        inference_mode=inference_mode,
        max_year=max_year,
        walk_forward=True,
    )


def load_bradley_terry_skill(db: Database, max_year: int = 2025) -> pd.DataFrame:
    """Backward-compatible season loader."""
    from baselines.skill_loader import load_skill_export

    export = load_skill_export("bradley_terry", db, max_year=max_year)
    out = export.season[["driverId", "season", "skill_score", "support_bucket"]].copy()
    return out.sort_values(["driverId", "season"]).reset_index(drop=True)
