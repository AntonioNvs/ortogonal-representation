"""Bradley-Terry model for the driver-vs-car decomposition.

The beat-teammate objective is exactly a Bradley-Terry ranking problem, and
the right latent for "driver skill" is a single time-invariant scalar per
driver — not an embedding per (driver, season). Two drivers racing the same
race give:

    P(A beats B) = sigma( theta_A - theta_B + q_{cA,s} - q_{cB,s} )

where

    * ``theta_d``  — one scalar per driver (intrinsic, time-invariant skill).
      This is the paper's product: car-independent driver ability.
    * ``q_{c,s}``  — one scalar per (constructor, season): car strength that
      year. It absorbs the car so ``theta`` carries only the driver.

The design matrix separates the two through the dataset's natural experiments:

    * a **transfer** (driver X in two constructors) ties X's ``theta`` to two
      different ``q`` values, pinning the driver apart from the car;
    * a **multi-driver team** (two drivers in one constructor) ties one ``q``
      to two ``theta`` values, pinning the car apart from the drivers.

For teammates ``cA == cB`` so ``q`` cancels and ``theta`` is identified; the
cross-team pairs identify ``q``. This is the standard, well-posed Bradley-Terry
with additive player + team effects — leak-free by construction (no temporal
edges, no embeddings that can memorise the future).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BradleyTerry(nn.Module):
    """Additive Bradley-Terry with driver (theta) and car (q) latents."""

    def __init__(self, num_drivers: int, num_constructor_seasons: int):
        super().__init__()
        # theta: intrinsic driver skill. Init near zero (the pair-difference
        # objective is shift-invariant, so a small symmetric init is natural).
        self.theta = nn.Embedding(num_drivers, 1)
        # q: car strength per (constructor, season). Init at zero.
        self.q = nn.Embedding(num_constructor_seasons, 1)
        nn.init.normal_(self.theta.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.q.weight)

    def logits(
        self,
        driver_A: torch.Tensor,
        driver_B: torch.Tensor,
        cs_A: torch.Tensor,
        cs_B: torch.Tensor,
    ) -> torch.Tensor:
        """Un-normalised log-odds that A beats B (A finished ahead)."""
        theta = self.theta(driver_A) - self.theta(driver_B)  # (N, 1)
        q = self.q(cs_A) - self.q(cs_B)  # (N, 1)
        return (theta + q).squeeze(-1)

    def driver_skill(self) -> torch.Tensor:
        """Return the theta vector (one scalar per driver)."""
        return self.theta.weight.squeeze(-1)
