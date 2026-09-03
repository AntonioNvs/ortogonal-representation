"""Tests for Plackett-Luce walk-forward skill export."""

from __future__ import annotations

import pandas as pd
import torch

from baselines.bradley_terry import BradleyTerry
from baselines.plackett_luce_skill import _fit_pl_on_races
from data.mobility import build_race_groups_for_pl
from models.ranking_likelihood import batch_pl_nll, plackett_luce_nll
from skill.contract import InferenceMode
from skill.export import build_skill_export


def test_pl_fitter_improves_on_synthetic_race():
    device = torch.device("cpu")
    drv_idx = {1: 0, 2: 1, 3: 2}
    cs_idx = {(10, 2024): 0, (20, 2024): 1, (30, 2024): 2}
    hist = pd.DataFrame(
        {
            "raceId": [1, 1, 1],
            "driverId": [1, 2, 3],
            "constructorId": [10, 20, 30],
            "year": [2024, 2024, 2024],
            "race_position_order": [1.0, 2.0, 3.0],
            "in_race_ranking": [True, True, True],
        }
    )
    race_groups = build_race_groups_for_pl(hist, drv_idx, cs_idx)
    model = _fit_pl_on_races(
        race_groups,
        num_drivers=3,
        num_cs=3,
        device=device,
        lr=0.1,
        epochs=100,
    )
    d_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    c_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    ranks = torch.tensor([1.0, 2.0, 3.0])
    u = model.utilities(d_idx, c_idx)
    loss = plackett_luce_nll(u, ranks)
    random_model = BradleyTerry(3, 3)
    u_rand = random_model.utilities(d_idx, c_idx)
    random_loss = plackett_luce_nll(u_rand, ranks)
    assert loss.item() < random_loss.item()
    assert u[0] > u[1] > u[2]


def test_plackett_luce_skill_export_smoke():
    race_df = pd.DataFrame(
        {
            "driverId": [1, 2, 1, 2],
            "season": [2024, 2024, 2024, 2024],
            "round": [1, 1, 2, 2],
            "raceId": [10, 10, 11, 11],
            "constructorId": [5, 6, 5, 6],
            "lineage_id": ["ferrari", "mercedes", "ferrari", "mercedes"],
            "driver_name": ["Driver A", "Driver B", "Driver A", "Driver B"],
            "constructor_name": ["Ferrari", "Mercedes", "Ferrari", "Mercedes"],
            "raw_skill": [1.0, 0.5, 1.1, 0.6],
            "contrib_driver": [1.0, 0.5, 1.1, 0.6],
            "contrib_constructor": [0.3, 0.4, 0.3, 0.4],
            "contrib_context": [0.0, 0.0, 0.0, 0.0],
            "contrib_residual": [0.0, 0.0, 0.0, 0.0],
            "as_of_round": [1, 1, 2, 2],
            "support_bucket": ["high", "high", "high", "high"],
        }
    )
    export = build_skill_export(
        race_df,
        skill_source="plackett_luce",
        inference_mode=InferenceMode.FILTERED,
        train_years=[2024],
    )
    export.validate()
    assert export.source == "plackett_luce"
    assert (export.race["skill_0_10"] >= 0).all()
    assert (export.race["skill_0_10"] <= 10).all()


def test_build_race_groups_for_pl_requires_two_finishers():
    drv_idx = {1: 0, 2: 1}
    cs_idx = {(10, 2024): 0}
    hist = pd.DataFrame(
        {
            "raceId": [1],
            "driverId": [1],
            "constructorId": [10],
            "year": [2024],
            "race_position_order": [1.0],
            "in_race_ranking": [True],
        }
    )
    assert build_race_groups_for_pl(hist, drv_idx, cs_idx) == []


def test_batch_pl_nll_empty():
    loss = batch_pl_nll([], [])
    assert float(loss.item()) == 0.0
