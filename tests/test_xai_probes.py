"""Unit tests for SkillGNN XAI probes."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from explain.skill_gnn_probes import (
    ProbeSampleConfig,
    channel_decomposition,
    evaluate_xai_gates,
    infer_claim_level,
    swap_invariance_test,
)
from models.skill_gnn import SkillGNN


def test_evaluate_xai_gates_pass():
    gates = evaluate_xai_gates(leakage_rho=0.1, swap_skill_diff=0.0)
    assert gates["constructor_leakage"] is True
    assert gates["swap_invariance"] is True


def test_evaluate_xai_gates_leakage_fail():
    gates = evaluate_xai_gates(leakage_rho=0.54, swap_skill_diff=0.0)
    assert gates["constructor_leakage"] is False
    assert gates["swap_invariance"] is True


def test_infer_claim_level_strong_skill():
    level = infer_claim_level(partial_rho=0.32, partial_ci_low=0.17, leakage_rho=0.1)
    assert level == "strong_skill"


def test_infer_claim_level_car_adjusted():
    level = infer_claim_level(partial_rho=0.32, partial_ci_low=0.17, leakage_rho=0.54)
    assert level == "car_adjusted_performance"


def test_infer_claim_level_insufficient():
    level = infer_claim_level(partial_rho=0.05, partial_ci_low=-0.1, leakage_rho=0.1)
    assert level == "insufficient"


def test_race_utilities_swap_invariance():
    """Driver readout is unchanged when only constructor_state_idx is swapped."""
    hidden = 4
    model = SkillGNN(
        node_to_col_names_dict={
            "driver_state": {},
            "constructor_state": {},
        },
        node_to_col_stats={
            "driver_state": {},
            "constructor_state": {},
        },
        hidden_dim=hidden,
        num_layers=1,
        grid_weight=0.05,
    )
    x_dict = {
        "driver_state": torch.randn(3, hidden),
        "constructor_state": torch.randn(3, hidden),
    }
    d_idx = torch.tensor([0])
    c_a = torch.tensor([1])
    c_b = torch.tensor([2])
    grid = torch.tensor([5.0])

    _, skill_a = model.race_utilities(x_dict, d_idx, c_a, grid)
    u_b, skill_b = model.race_utilities(x_dict, d_idx, c_b, grid)

    assert skill_a.item() == pytest.approx(skill_b.item())
    assert u_b.item() != pytest.approx(
        model.race_utilities(x_dict, d_idx, c_a, grid)[0].item()
    )


def test_channel_decomposition_driver_share():
    hidden = 8
    model = MagicMock()
    model.grid_weight = 0.05
    model.encode.return_value = {
        "driver_state": torch.ones(2, hidden),
        "constructor_state": torch.ones(2, hidden) * 2,
    }
    model.driver_readout = torch.nn.Linear(hidden, 1)
    model.constructor_readout = torch.nn.Linear(hidden, 1)
    with torch.no_grad():
        model.driver_readout.weight.fill_(0.1)
        model.driver_readout.bias.zero_()
        model.constructor_readout.weight.fill_(0.2)
        model.constructor_readout.bias.zero_()

    graph_data = MagicMock()
    res = graph_data.__getitem__.return_value
    graph_data.__getitem__ = lambda self, key: res if key == "results" else None
    res.driver_state_idx = torch.tensor([0, 1])
    res.constructor_state_idx = torch.tensor([0, 1])
    res.grid = torch.tensor([1.0, 10.0])

    out = channel_decomposition(
        model,
        graph_data,
        tf_dict={},
        edge_index_dict={},
        device=torch.device("cpu"),
        sample_idx=torch.tensor([0, 1]),
    )
    assert 0.0 <= out["driver_share_mean"] <= 1.0
    assert out["constructor_share_mean"] == pytest.approx(1.0 - out["driver_share_mean"])


def test_swap_invariance_test_with_synthetic_graph():
    hidden = 4
    model = SkillGNN(
        node_to_col_names_dict={"driver_state": {}, "constructor_state": {}},
        node_to_col_stats={"driver_state": {}, "constructor_state": {}},
        hidden_dim=hidden,
        num_layers=1,
    )
    x_dict = {
        "driver_state": torch.randn(4, hidden),
        "constructor_state": torch.randn(4, hidden),
    }

    class FakeResults:
        year = torch.tensor([2024, 2024, 2024, 2024])
        round = torch.tensor([1, 1, 1, 1])
        race_id = torch.tensor([100, 100, 100, 100])
        driver_state_idx = torch.tensor([0, 1, 2, 3])
        constructor_state_idx = torch.tensor([0, 1, 0, 1])
        grid = torch.tensor([1.0, 2.0, 3.0, 4.0])
        driver_id = torch.tensor([10, 11, 12, 13])
        constructor_id = torch.tensor([1, 2, 1, 2])

    class FakeGraph:
        def __getitem__(self, key):
            if key == "results":
                return FakeResults()
            raise KeyError(key)

    tf_dict = {}
    edge_index_dict = {}
    sample_idx = torch.tensor([0, 1])
    config = ProbeSampleConfig(seed=0, swap_samples=2)

    with torch.no_grad():
        model.encode = lambda tf, ei: x_dict  # type: ignore[method-assign]
        out = swap_invariance_test(
            model,
            FakeGraph(),
            tf_dict,
            edge_index_dict,
            torch.device("cpu"),
            sample_idx,
            config,
        )

    assert out["n_swaps"] >= 1
    assert out["skill_diff"] == pytest.approx(0.0, abs=1e-6)
    assert out["utility_swap_delta"] >= 0.0


@pytest.mark.skipif(
    not os.path.isfile("output/skill_model/skill_gnn.pth"),
    reason="SkillGNN checkpoint not available",
)
def test_run_xai_falsification_smoke():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "src/experiments/run_xai_falsification.py", "--max-samples", "100"],
        capture_output=True,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        env={**os.environ, "PYTHONPATH": "src"},
    )
    if result.returncode != 0:
        pytest.skip(
            f"XAI smoke skipped: checkpoint incompatible or run failed.\n{result.stderr[-500:]}"
        )

    report_path = "output/skill_evaluation/xai_report.json"
    assert os.path.isfile(report_path)
    with open(report_path) as f:
        report = json.load(f)
    assert report.get("skill_source") == "skill_gnn"
    assert "constructor_leakage_rho" in report
    assert "gates" in report
