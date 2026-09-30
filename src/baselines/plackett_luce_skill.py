"""Walk-forward race-level Plackett-Luce benchmark with SkillExport."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import torch

import config as cfg
from baselines.bradley_terry import BradleyTerry
from data.mobility import build_race_groups_for_pl, compute_support_scores
from data.race_panel import RacePanelConfig, build_race_panel
from models.ranking_likelihood import batch_pl_nll
from relbench.base import Database
from skill.contract import InferenceMode
from skill.export import build_skill_export


def _fit_pl_on_races(
    race_groups: list[dict],
    num_drivers: int,
    num_cs: int,
    *,
    device: torch.device,
    lr: float = cfg.PL_LR,
    epochs: int = cfg.PL_EPOCHS_PER_STEP,
    weight_decay: float = cfg.PL_WEIGHT_DECAY,
    init_model: Optional[BradleyTerry] = None,
) -> BradleyTerry:
    if init_model is not None:
        model = BradleyTerry(num_drivers, num_cs).to(device)
        with torch.no_grad():
            model.theta.weight.copy_(init_model.theta.weight)
            model.q.weight.copy_(init_model.q.weight)
    else:
        model = BradleyTerry(num_drivers, num_cs).to(device)

    if not race_groups:
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        utilities_list = []
        ranks_list = []
        for race in race_groups:
            d_idx = torch.tensor(race["driver_idx"], dtype=torch.long, device=device)
            c_idx = torch.tensor(race["cs_idx"], dtype=torch.long, device=device)
            ranks = torch.tensor(race["ranks"], dtype=torch.float32, device=device)
            utilities_list.append(model.utilities(d_idx, c_idx))
            ranks_list.append(ranks)

        loss = batch_pl_nll(utilities_list, ranks_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def export_plackett_luce(
    db: Database,
    *,
    max_year: int = 2025,
    inference_mode: InferenceMode = InferenceMode.FILTERED,
    device=None,
    lr: float = cfg.PL_LR,
    epochs: int = cfg.PL_EPOCHS_PER_STEP,
    weight_decay: float = cfg.PL_WEIGHT_DECAY,
) -> "SkillExport":
    """Expanding-window Plackett-Luce: causal as-of-round race export."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    panel = build_race_panel(db, RacePanelConfig(max_year=max_year))
    ranked = panel[panel["in_race_ranking"]].copy()
    hist_df = ranked.rename(columns={"season": "year"})

    drivers = sorted(ranked["driverId"].astype(int).unique())
    cs_keys = sorted(
        set(zip(ranked["constructorId"].astype(int), ranked["season"].astype(int)))
    )
    drv_idx = {d: i for i, d in enumerate(drivers)}
    cs_idx = {k: i for i, k in enumerate(cs_keys)}
    num_drivers = len(drivers)
    num_cs = len(cs_keys)

    checkpoints = (
        ranked[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )

    rows = []
    model: Optional[BradleyTerry] = None
    for _, ck in checkpoints.iterrows():
        season, rnd = int(ck["season"]), int(ck["round"])
        hist = hist_df[
            (hist_df["year"] < season)
            | ((hist_df["year"] == season) & (hist_df["round"] <= rnd))
        ]
        race_groups = build_race_groups_for_pl(hist, drv_idx, cs_idx)
        model = _fit_pl_on_races(
            race_groups,
            num_drivers,
            num_cs,
            device=device,
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay,
            init_model=model,
        )
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
    support = compute_support_scores(hist_df)
    race_df = race_df.merge(
        support[["driverId", "season", "support_bucket"]],
        on=["driverId", "season"],
        how="left",
    )
    race_df["support_bucket"] = race_df["support_bucket"].fillna("medium")

    return build_skill_export(
        race_df,
        skill_source="plackett_luce",
        inference_mode=inference_mode,
        max_year=max_year,
        walk_forward=True,
    )
