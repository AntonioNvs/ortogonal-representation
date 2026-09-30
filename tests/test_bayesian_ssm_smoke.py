"""Smoke tests for Stan model compilation (optional)."""

from __future__ import annotations

import os
import pytest

STAN_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "models",
    "bayesian_ssm.stan",
)


@pytest.mark.skipif(not os.path.isfile(STAN_FILE), reason="stan file missing")
def test_stan_model_exists():
    assert os.path.getsize(STAN_FILE) > 100


@pytest.mark.skipif(
    os.environ.get("RUN_STAN_SMOKE") != "1",
    reason="Set RUN_STAN_SMOKE=1 to compile Stan model",
)
def test_stan_compiles():
    pytest.importorskip("cmdstanpy")
    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=os.path.abspath(STAN_FILE))
    assert model is not None
