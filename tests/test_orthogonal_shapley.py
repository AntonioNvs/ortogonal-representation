"""Tests for OrthogonalShapleyGNN model, coalition Shapley, and export contract."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from explain.coalition_shapley import (
  CoalitionBaselines,
  exact_shapley_utilities,
  shapley_efficiency_error,
)
from models.orthogonal_shapley_gnn import CONTEXT_DIM, OrthogonalShapleyGNN
from skill.contract import InferenceMode
from skill.export import build_skill_export


class _TinyOrthogonalModel(torch.nn.Module):
  """Minimal stand-in for Shapley tests without full graph encoder."""

  def __init__(self, hidden_dim: int = 4):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(hidden_dim * 2 + CONTEXT_DIM, 8),
      torch.nn.ReLU(),
      torch.nn.Linear(8, 1),
    )

  def utility_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
    return self.classifier(fused).squeeze(-1)


def _make_baselines(hidden_dim: int = 4, device: torch.device | None = None) -> CoalitionBaselines:
  device = device or torch.device("cpu")
  return CoalitionBaselines(
    driver_emb=torch.zeros(hidden_dim, device=device),
    constructor_emb=torch.zeros(hidden_dim, device=device),
    context=torch.zeros(CONTEXT_DIM, device=device),
  )


def test_exact_shapley_efficiency():
  """Shapley values must sum to v(full) - v(empty)."""
  hidden = 4
  tiny = _TinyOrthogonalModel(hidden)
  device = torch.device("cpu")
  baselines = _make_baselines(hidden, device)

  n = 20
  d_emb = torch.randn(n, hidden)
  c_emb = torch.randn(n, hidden)
  ctx = torch.randn(n, CONTEXT_DIM)

  class _Wrapper:
    utility_from_fused = tiny.utility_from_fused

  wrapper = _Wrapper()

  phi_d, phi_c, phi_x, residual = exact_shapley_utilities(
    wrapper, d_emb, c_emb, ctx, baselines
  )

  from explain.coalition_shapley import _coalition_value, ALL_PLAYERS

  v_full = _coalition_value(wrapper, ALL_PLAYERS, d_emb, c_emb, ctx, baselines)
  v_empty = _coalition_value(wrapper, 0, d_emb, c_emb, ctx, baselines)
  err = shapley_efficiency_error(phi_d, phi_c, phi_x, v_full, v_empty)
  assert err < 1e-5
  assert float(torch.mean(torch.abs(residual)).item()) < 1e-5


def test_coalition_baselines_roundtrip():
  baselines = _make_baselines(8)
  d = baselines.to_dict()
  restored = CoalitionBaselines.from_dict(d, torch.device("cpu"))
  assert torch.allclose(baselines.driver_emb, restored.driver_emb)
  assert torch.allclose(baselines.context, restored.context)


def test_context_features_shape():
  grid = torch.tensor([1.0, 10.0, 20.0])
  ctx = OrthogonalShapleyGNN.context_features(grid)
  assert ctx.shape == (3, CONTEXT_DIM)


def test_skill_export_from_shapley_contribs():
  """build_skill_export accepts coalition Shapley contributions."""
  n = 50
  race_df = pd.DataFrame(
    {
      "driverId": np.random.randint(1, 10, n),
      "season": [2024] * n,
      "round": np.arange(1, n + 1) % 20 + 1,
      "raceId": np.arange(100, 100 + n),
      "constructorId": np.random.randint(1, 5, n),
      "lineage_id": ["ferrari"] * n,
      "driver_name": ["Test Driver"] * n,
      "constructor_name": ["Ferrari"] * n,
      "raw_skill": np.random.randn(n),
      "contrib_driver": np.random.randn(n),
      "contrib_constructor": np.random.randn(n) * 0.5,
      "contrib_context": np.random.randn(n) * 0.2,
      "contrib_residual": np.zeros(n),
      "as_of_round": np.arange(1, n + 1) % 20 + 1,
      "support_bucket": ["medium"] * n,
    }
  )
  export = build_skill_export(
    race_df,
    skill_source="orthogonal_shapley",
    inference_mode=InferenceMode.FILTERED,
    train_years=[2024],
  )
  export.validate()
  assert export.metadata.skill_source == "orthogonal_shapley"
  assert (export.race["skill_0_10"] >= 0).all()
  assert (export.race["skill_0_10"] <= 10).all()


def test_coalition_baselines_json_serializable():
  baselines = _make_baselines(4)
  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "baselines.json")
    with open(path, "w") as f:
      json.dump(baselines.to_dict(), f)
    with open(path) as f:
      loaded = json.load(f)
    restored = CoalitionBaselines.from_dict(loaded, torch.device("cpu"))
    assert restored.driver_emb.shape == (4,)
