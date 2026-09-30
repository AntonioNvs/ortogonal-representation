"""Tests for career validation framework v2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.career_labels import career_outcome_labels, rest_of_career_outcome
from validation.inconsistency import (
    mark_underrated,
    underrated_promotion_auroc,
    underrated_resolution_rate,
)
from validation.inference import partial_spearman, stratum_partial_spearman
from validation.skill_trajectory import compute_skill_trajectory, enrich_career_join


def _make_driver_season(rows):
    return pd.DataFrame(rows)


def _make_team_tier(rows):
    return pd.DataFrame(rows)


def test_rest_of_career_outcome_single_future_season():
    driver_season = _make_driver_season(
        [
            {"driverId": 1, "driverRef": "a", "season": 2020, "constructorId": 10},
            {"driverId": 1, "driverRef": "a", "season": 2021, "constructorId": 20},
        ]
    )
    team_tier = _make_team_tier(
        [
            {"constructorId": 10, "season": 2020, "tier": "B"},
            {"constructorId": 20, "season": 2021, "tier": "A"},
        ]
    )
    out = rest_of_career_outcome(driver_season, team_tier)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["season_T"] == 2020
    assert row["outcome_score"] == 2.0
    assert row["n_future_seasons"] == 1
    assert row["outcome_score"] > 1.0


def test_rest_of_career_includes_partial_future_not_full_horizon():
    driver_season = _make_driver_season(
        [
            {"driverId": 1, "driverRef": "a", "season": 2020, "constructorId": 10},
            {"driverId": 1, "driverRef": "a", "season": 2021, "constructorId": 10},
        ]
    )
    team_tier = _make_team_tier(
        [
            {"constructorId": 10, "season": 2020, "tier": "B"},
            {"constructorId": 10, "season": 2021, "tier": "B"},
        ]
    )
    out = rest_of_career_outcome(driver_season, team_tier)
    assert len(out) == 1
    assert out.iloc[0]["n_future_seasons"] == 1


def test_career_outcome_labels_horizon_none_vs_fixed():
    driver_season = _make_driver_season(
        [
            {"driverId": 1, "driverRef": "a", "season": 2018, "constructorId": 10},
            {"driverId": 1, "driverRef": "a", "season": 2019, "constructorId": 20},
            {"driverId": 1, "driverRef": "a", "season": 2020, "constructorId": 30},
            {"driverId": 1, "driverRef": "a", "season": 2021, "constructorId": 30},
        ]
    )
    team_tier = _make_team_tier(
        [
            {"constructorId": 10, "season": 2018, "tier": "B"},
            {"constructorId": 20, "season": 2019, "tier": "B"},
            {"constructorId": 30, "season": 2020, "tier": "A"},
            {"constructorId": 30, "season": 2021, "tier": "S"},
        ]
    )
    inf_out = career_outcome_labels(driver_season, team_tier, horizon=None)
    fixed_out = career_outcome_labels(
        driver_season, team_tier, horizon=2, require_full_horizon=True
    )
    inf_2018 = inf_out[inf_out["season_T"] == 2018].iloc[0]
    fixed_2018 = fixed_out[fixed_out["season_T"] == 2018].iloc[0]
    assert inf_2018["n_future_seasons"] == 3
    assert fixed_2018["n_observed"] == 2
    assert inf_2018["outcome_score"] > fixed_2018["outcome_score"]


def test_mark_underrated_top_skill_b_tier():
    df = pd.DataFrame(
        {
            "driverId": [1, 2, 3, 4, 5],
            "season_T": [2020] * 5,
            "skill_score": [10.0, 8.0, 6.0, 4.0, 2.0],
            "constructor_tier_score_at_T": [1, 1, 1, 1, 1],
            "outcome_score": [2.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    marked = mark_underrated(df)
    underrated = marked[marked["underrated_flag"]]
    assert len(underrated) == 2
    assert set(underrated["driverId"]) == {1, 2}
    promoted = underrated[underrated["promoted"] == 1]
    assert len(promoted) == 1
    assert promoted.iloc[0]["driverId"] == 1


def test_mark_underrated_excludes_s_tier():
    df = pd.DataFrame(
        {
            "driverId": [1, 2],
            "season_T": [2020, 2020],
            "skill_score": [10.0, 2.0],
            "constructor_tier_score_at_T": [3, 1],
            "outcome_score": [3.0, 1.0],
        }
    )
    marked = mark_underrated(df)
    assert marked["underrated_flag"].sum() == 0


def test_mark_underrated_low_skill_b_tier_excluded():
    df = pd.DataFrame(
        {
            "driverId": [1, 2, 3, 4],
            "season_T": [2020, 2020, 2020, 2020],
            "skill_score": [10.0, 8.0, 5.0, 2.0],
            "constructor_tier_score_at_T": [1, 1, 1, 1],
            "outcome_score": [2.0, 1.5, 1.0, 1.0],
        }
    )
    marked = mark_underrated(df)
    low_skill = marked[marked["driverId"] == 4]
    assert not low_skill.iloc[0]["underrated_flag"]


def test_underrated_resolution_rate():
    df = pd.DataFrame(
        {
            "driverId": [1, 2, 3, 4],
            "season_T": [2020, 2020, 2021, 2021],
            "skill_score": [10.0, 9.0, 10.0, 9.0],
            "constructor_tier_score_at_T": [1, 1, 1, 1],
            "outcome_score": [2.0, 1.0, 2.0, 1.0],
        }
    )
    marked = mark_underrated(df)
    marked["underrated_flag"] = True  # force all into stratum for rate test
    marked["promoted"] = (marked["outcome_score"] > marked["constructor_tier_score_at_T"]).astype(int)
    result = underrated_resolution_rate(marked, seed=42)
    assert result["resolution_rate"] == 0.5
    assert result["n_underrated"] == 4
    assert result["n_promoted"] == 2


def test_underrated_resolution_rate_cluster_bootstrap_reproducible():
    df = pd.DataFrame(
        {
            "driverId": [1, 1, 2, 2, 3, 3],
            "season_T": [2018, 2019, 2018, 2019, 2018, 2019],
            "skill_score": [10, 10, 9, 9, 5, 5],
            "constructor_tier_score_at_T": [1, 1, 1, 1, 1, 1],
            "outcome_score": [2, 2, 1, 1, 1, 1],
        }
    )
    marked = mark_underrated(df)
    marked["underrated_flag"] = True
    marked["promoted"] = (marked["outcome_score"] > marked["constructor_tier_score_at_T"]).astype(int)
    r1 = underrated_resolution_rate(marked, seed=123)
    r2 = underrated_resolution_rate(marked, seed=123)
    assert r1["ci_low"] == r2["ci_low"]
    assert r1["ci_high"] == r2["ci_high"]


def test_skill_trajectory_slope_and_delta():
    season_skill = pd.DataFrame(
        {
            "driverId": [1, 1, 1],
            "season": [2018, 2019, 2020],
            "skill_score": [5.0, 7.0, 9.0],
        }
    )
    traj = compute_skill_trajectory(season_skill)
    last = traj[traj["season_T"] == 2020].iloc[0]
    assert last["skill_delta"] == pytest.approx(2.0)
    assert last["skill_slope_3yr"] > 0
    assert last["career_phase"] == "mid"


def test_enrich_career_join_adds_trajectory_columns():
    joined = pd.DataFrame(
        {
            "driverId": [1, 1],
            "season_T": [2019, 2020],
            "skill_score": [6.0, 8.0],
            "outcome_score": [1.5, 2.0],
            "constructor_tier_score_at_T": [1, 1],
        }
    )
    season_skill = pd.DataFrame(
        {
            "driverId": [1, 1],
            "season": [2019, 2020],
            "skill_score": [6.0, 8.0],
        }
    )
    enriched = enrich_career_join(joined, season_skill)
    assert "skill_slope_3yr" in enriched.columns
    assert "skill_delta" in enriched.columns
    assert "career_phase" in enriched.columns


def test_stratum_partial_spearman_falls_back_when_tier_constant():
    df = pd.DataFrame(
        {
            "driverId": list(range(10)),
            "season_T": [2020] * 10,
            "skill_score": np.linspace(1, 10, 10),
            "outcome_score": np.linspace(1, 10, 10) + np.random.default_rng(0).normal(0, 0.1, 10),
            "constructor_tier_score_at_T": [1.0] * 10,
            "underrated_flag": [True] * 10,
        }
    )
    result = stratum_partial_spearman(df, seed=0)
    assert not np.isnan(result["partial_rho"])
    assert "constant in stratum" in result.get("note", "")
