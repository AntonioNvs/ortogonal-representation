"""Ranking likelihoods: pairwise Bradley-Terry and Plackett-Luce NLL."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_bt_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for ordered pairs (label=1 means A beat B)."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def plackett_luce_nll(utilities: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for a single race partial ranking.

    ``utilities``: (n,) higher = better predicted performance.
    ``ranks``: (n,) position order 1=best; sorted ascending for PL factorization.
    """
    n = utilities.numel()
    if n <= 1:
        return utilities.sum() * 0.0

    order = torch.argsort(ranks)
    u = utilities[order]
    loss = torch.tensor(0.0, device=utilities.device, dtype=utilities.dtype)
    for i in range(n):
        denom = torch.logsumexp(u[i:], dim=0)
        loss = loss - (u[i] - denom)
    return loss / n


def batch_pl_nll(
    utilities_list: list[torch.Tensor],
    ranks_list: list[torch.Tensor],
) -> torch.Tensor:
    if not utilities_list:
        return torch.tensor(0.0)
    losses = [plackett_luce_nll(u, r) for u, r in zip(utilities_list, ranks_list)]
    return torch.stack(losses).mean()


def pairwise_ranking_loss(utilities: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """Soft pairwise ranking loss for a single race (lower = better ranks)."""
    n = utilities.numel()
    if n <= 1:
        return utilities.sum() * 0.0

    loss = torch.tensor(0.0, device=utilities.device, dtype=utilities.dtype)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] == ranks[j]:
                continue
            sign = 1.0 if ranks[i] < ranks[j] else -1.0
            loss = loss + torch.nn.functional.softplus(-sign * (utilities[i] - utilities[j]))
            count += 1
    if count == 0:
        return utilities.sum() * 0.0
    return loss / count


def batch_pairwise_ranking_loss(
    utilities_list: list[torch.Tensor],
    ranks_list: list[torch.Tensor],
) -> torch.Tensor:
    if not utilities_list:
        return torch.tensor(0.0)
    losses = [
        pairwise_ranking_loss(u, r) for u, r in zip(utilities_list, ranks_list)
    ]
    return torch.stack(losses).mean()


def grid_adjusted_race_utilities(
    driver_u: torch.Tensor,
    constructor_u: torch.Tensor,
    context_u: torch.Tensor,
    grid_u: torch.Tensor,
    grid_positions: torch.Tensor,
) -> torch.Tensor:
    """Semantic sum with grid effect (lower grid = better, so negate normalized grid)."""
    # grid_positions: 1 = pole; convert to bonus
    grid_bonus = -0.1 * (grid_positions.float() - 1.0)
    return driver_u + constructor_u + context_u + grid_u + grid_bonus
