"""Additive Bradley-Terry with driver (theta) and car (q) latents."""

from __future__ import annotations

import torch
import torch.nn as nn


class BradleyTerry(nn.Module):
    def __init__(self, num_drivers: int, num_constructor_seasons: int):
        super().__init__()
        self.theta = nn.Embedding(num_drivers, 1)
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
        theta = self.theta(driver_A) - self.theta(driver_B)
        q = self.q(cs_A) - self.q(cs_B)
        return (theta + q).squeeze(-1)

    def utilities(self, driver_idx: torch.Tensor, cs_idx: torch.Tensor) -> torch.Tensor:
        """Per-entry race utilities for Plackett-Luce: theta + q."""
        return (self.theta(driver_idx) + self.q(cs_idx)).squeeze(-1)

    def driver_skill(self) -> torch.Tensor:
        return self.theta.weight.squeeze(-1)
