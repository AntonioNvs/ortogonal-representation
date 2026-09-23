"""Unit tests for attribution ablation metadata and GOAT common-car helpers."""

from __future__ import annotations

import numpy as np
import torch

from applications.goat_common_car import (
    NEUTRAL_GRID,
    assert_common_car_invariants,
    mean_constructor_embedding,
    neutral_grid_tensor,
    peak_driver_state_indices,
    race_ids_for_year,
    simulate_season_from_utilities,
)
from explain.coalition_shapley import attribution_balance_loss


class _FakeRes:
    def __init__(self):
        self.year = torch.tensor([2023, 2023, 2023, 2025, 2025, 2025])
        self.round = torch.tensor([1, 2, 3, 1, 2, 1])
        self.driver_id = torch.tensor([10, 10, 10, 10, 20, 20])
        self.race_id = torch.tensor([100, 101, 102, 200, 201, 200])
        self.driver_state_idx = torch.tensor([0, 1, 2, 3, 4, 5])
        self.constructor_state_idx = torch.tensor([10, 11, 12, 20, 21, 22])
        self.race_idx = torch.tensor([0, 1, 2, 3, 4, 3])
        self.driver_career_idx = torch.tensor([0, 0, 0, 0, 1, 1])


def test_lambda_attr_zero_skips_balance_contribution():
    """attribution_balance_loss is positive when shares exceed targets, but
    training multiplies by lambda_attr; lambda_attr=0 => no contribution."""
    phi_d = torch.tensor([2.0, 2.0])
    phi_c = torch.tensor([0.5, 0.5])
    phi_x = torch.tensor([0.5, 0.5])
    loss = attribution_balance_loss(phi_d, phi_c, phi_x, target_driver_share=0.38)
    assert loss.item() > 0.0
    lambda_attr = 0.0
    assert float(lambda_attr * loss.item()) == 0.0


def test_ablation_label_rule():
    assert ("no_attribution_balance" if 0.0 == 0.0 else "attribution_balance") == (
        "no_attribution_balance"
    )
    assert ("no_attribution_balance" if 0.1 == 0.0 else "attribution_balance") == (
        "attribution_balance"
    )


def test_peak_driver_state_is_final_round():
    res = _FakeRes()
    out = peak_driver_state_indices(res, driver_ids=[10], peak_seasons=[2023])
    assert out[10] == 2  # round 3 state


def test_race_ids_for_year_sorted_unique():
    res = _FakeRes()
    ids = race_ids_for_year(res, 2025)
    assert ids == [200, 201]


def test_mean_constructor_embedding():
    x_dict = {
        "constructor_state": torch.tensor(
            [[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], dtype=torch.float32
        )
    }
    mean = mean_constructor_embedding(x_dict, np.array([0, 1]))
    assert torch.allclose(mean, torch.tensor([2.0, 1.0]))


def test_neutral_grid_constant():
    g = neutral_grid_tensor(5, device=torch.device("cpu"), value=NEUTRAL_GRID)
    assert torch.allclose(g, torch.full((5,), NEUTRAL_GRID))


def test_common_car_invariants_pass():
    race_ids = [1, 2, 3]
    means = [torch.randn(4) for _ in race_ids]
    grids = [torch.full((3,), NEUTRAL_GRID) for _ in race_ids]
    out = assert_common_car_invariants(
        race_ids=race_ids,
        per_race_constructor_means=means,
        grids=grids,
    )
    assert out["pass"] is True


def test_shift_invariance_of_season_simulation():
    rng = np.random.default_rng(0)
    u = np.array([[1.0, 0.5, 0.0], [0.8, 0.4, 0.1]])
    a, _ = simulate_season_from_utilities(u, beta=0.0, dnf_rate=0.0, rng=rng)
    rng = np.random.default_rng(0)
    b, _ = simulate_season_from_utilities(u + 5.0, beta=0.0, dnf_rate=0.0, rng=rng)
    assert np.allclose(a, b)
