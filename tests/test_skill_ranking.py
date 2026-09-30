"""Tests for skill ranking pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from data.skill_dataset import DnfPolicy, SkillDatasetConfig, _normalize_rank, assert_skill_dataset_invariants
from models.ranking_likelihood import plackett_luce_nll
from skill.scoring import cumulative_season_skill, peak_season_skill
from visualization.driver_rankings import _resolve_drivers, plot_driver_rankings


def test_normalize_rank():
    assert float(_normalize_rank(1, 20)) == pytest.approx(1.0)
    assert float(_normalize_rank(20, 20)) == pytest.approx(0.0)
    assert np.isnan(float(_normalize_rank(1, 1)))

    pos = np.array([1.0, 10.0, 20.0])
    size = np.array([20.0, 20.0, 20.0])
    out = _normalize_rank(pos, size)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0 - 9.0 / 19.0)
    assert out[2] == pytest.approx(0.0)


def test_plackett_luce_perfect_order():
    u = torch.tensor([3.0, 2.0, 1.0])
    ranks = torch.tensor([1.0, 2.0, 3.0])
    loss = plackett_luce_nll(u, ranks)
    assert loss.item() < 0.5


def test_cumulative_season_skill_causal():
    race_df = pd.DataFrame(
        {
            "driverId": [1, 1, 1, 2, 2],
            "season": [2024, 2024, 2024, 2024, 2024],
            "round": [1, 2, 3, 1, 2],
            "raceId": [10, 11, 12, 10, 11],
            "skill_0_10": [9.0, 8.0, 7.0, 5.0, 6.0],
            "constructorId": [1, 1, 1, 2, 2],
            "driverRef": ["a", "a", "a", "b", "b"],
            "constructorRef": ["t1", "t1", "t1", "t2", "t2"],
        }
    )
    cum = cumulative_season_skill(race_df)
    d1_r2 = cum[(cum["driverId"] == 1) & (cum["round"] == 2)]["season_skill"].iloc[0]
    assert d1_r2 == pytest.approx(8.5)


def test_resolve_drivers():
    df = pd.DataFrame({"driverRef": ["verstappen", "norris"], "driverId": [1, 2], "season": [2024, 2024]})
    assert _resolve_drivers(df, ["verstappen"], season=2024) == [1]

    df2 = pd.DataFrame(
        {"driverRef": ["max_verstappen", "norris"], "driverId": [830, 847], "season": [2024, 2024]}
    )
    assert _resolve_drivers(df2, ["verstappen"], season=2024) == [830]
    assert _resolve_drivers(df2, ["norris", "max_verstappen"], season=2024) == [847, 830]


def test_plot_two_drivers():
    rankings = pd.DataFrame(
        {
            "driverId": [1, 1, 2, 2],
            "season": [2024, 2024, 2024, 2024],
            "round": [1, 2, 1, 2],
            "rank": [1, 1, 2, 2],
            "rank_lo": [1, 1, 2, 2],
            "rank_hi": [1, 1, 2, 2],
            "race_skill": [0.9, 0.88, 0.7, 0.72],
            "season_skill": [0.9, 0.89, 0.7, 0.71],
            "driverRef": ["max_verstappen", "max_verstappen", "norris", "norris"],
            "constructorRef": ["red_bull", "red_bull", "mclaren", "mclaren"],
        }
    )
    fig = plot_driver_rankings(rankings, 2024, ["verstappen", "norris"])
    assert len(fig.axes[0].lines) == 2
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_peak_season_skill():
    race = pd.DataFrame(
        {
            "driverId": [1, 1, 1, 2, 2],
            "season": [2024, 2024, 2024, 2024, 2024],
            "round": [1, 2, 3, 1, 2],
            "raceId": [101, 102, 103, 101, 102],
            "skill_0_10": [7.0, 9.5, 8.0, 6.0, 8.5],
            "driver_name": ["Max Verstappen"] * 3 + ["Lando Norris"] * 2,
            "constructor_name": ["Red Bull"] * 3 + ["McLaren"] * 2,
        }
    )
    peaks = peak_season_skill(race)
    assert len(peaks) == 2
    verstappen = peaks.loc[peaks["driverId"] == 1].iloc[0]
    norris = peaks.loc[peaks["driverId"] == 2].iloc[0]
    assert verstappen["peak_skill"] == pytest.approx((7.0 + 9.5 + 8.0) / 3)
    assert verstappen["n_races"] == 3
    assert norris["peak_skill"] == pytest.approx((6.0 + 8.5) / 2)
    ranked = peaks.sort_values("peak_skill", ascending=False).reset_index(drop=True)
    assert ranked.iloc[0]["driver_name"] == "Max Verstappen"


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/enriched/rel-f1/db").exists(),
    reason="enriched db not present",
)
def test_skill_dataset_invariants():
    import config as cfg
    import data.tasks as data_tasks
    from data.enriched_dataset import EnrichedF1Dataset
    from data.skill_dataset import build_skill_dataset

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=2000,
        max_year=2023,
    )
    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)
    df = build_skill_dataset(db, SkillDatasetConfig(min_year=2000, max_year=2023))
    assert_skill_dataset_invariants(df)
    assert len(df) > 1000
