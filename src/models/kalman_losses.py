"""
Loss functions for the Kalman-GNN temporal skill model.

Composes four loss terms:
  L_pred     – beat-teammate binary cross-entropy
  L_smooth   – temporal smoothness of embeddings
  L_contrast – InfoNCE contrastive loss (temporal identity)
  L_skill    – skill forward consistency
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------


def beat_teammate_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for beat-teammate prediction.

    Args:
        logits: (N_pairs,) -- skill_A - skill_B (+ context delta).
        labels: (N_pairs,) -- 1.0 if A beat B, 0.0 otherwise.

    Returns:
        Scalar loss.
    """
    if len(logits) == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def smoothness_loss(
    v_curr: torch.Tensor,
    v_prev: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L2 penalty on embedding change: ||v_curr - v_prev||².

    Args:
        v_curr: (N, dim) -- current embeddings.
        v_prev: (N, dim) -- previous embeddings.
        active_mask: (N,) bool -- only compute for active entities if provided.

    Returns:
        Scalar loss (mean squared L2 norm of change).
    """
    diff = (v_curr - v_prev).pow(2).sum(dim=-1)  # (N,)
    if active_mask is not None and active_mask.any():
        diff = diff[active_mask]
    return diff.mean()


def temporal_infonce_loss(
    driver_embs: list[torch.Tensor],
    driver_active: list[torch.Tensor],
    gap_min: int = 1,
    gap_max: int = 5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss for temporal identity.

    For each driver at time t, create a positive pair with the same driver
    at time t + gap (gap ~ Uniform[gap_min, gap_max]).  Negative pairs are
    all other drivers active at time t + gap.

    This maximises I(v_driver_t ; v_driver_{t+gap}) -- the embedding retains
    what persists across time (skill) and discards transient factors.

    Args:
        driver_embs: list of (num_drivers, dim) tensors at each race step.
        driver_active: list of (num_drivers,) boolean tensors at each step.
        gap_min: minimum race gap for positive pairs.
        gap_max: maximum race gap for positive pairs.
        temperature: InfoNCE temperature.

    Returns:
        Scalar contrastive loss.
    """
    if len(driver_embs) < gap_min + 1:
        return torch.tensor(0.0, device=driver_embs[0].device)

    device = driver_embs[0].device
    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    # Sample a random gap per anchor position
    for t in range(len(driver_embs) - gap_max):
        gap = torch.randint(gap_min, gap_max + 1, (1,)).item()
        t_future = t + gap
        if t_future >= len(driver_embs):
            continue

        v_t = driver_embs[t]  # (N, dim)
        v_future = driver_embs[t_future]  # (N, dim)
        active_t = driver_active[t]
        active_future = driver_active[t_future]

        # Only consider drivers active at both time points
        valid = active_t & active_future
        if valid.sum() < 2:
            continue

        v_t_valid = v_t[valid]  # (M, dim)
        v_future_valid = v_future[valid]  # (M, dim)

        # Normalize
        v_t_norm = F.normalize(v_t_valid, p=2, dim=-1)
        v_future_norm = F.normalize(v_future_valid, p=2, dim=-1)

        # Similarity matrix: (M, M) where entry (i, j) is sim(v_t[i], v_future[j])
        sim = torch.mm(v_t_norm, v_future_norm.T) / temperature  # (M, M)

        # Positive: diagonal (same driver at both times)
        # InfoNCE: -log(exp(sim[i,i]) / sum_j exp(sim[i,j]))
        loss = F.cross_entropy(sim, torch.arange(sim.shape[0], device=device))
        total_loss = total_loss + loss
        n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / n_pairs


def skill_consistency_loss(
    skill_curr: torch.Tensor,
    skill_next: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SmoothL1 loss: skill_r should predict skill_{r+1}.

    Args:
        skill_curr: (N, 1) -- skill scores at time r.
        skill_next: (N, 1) -- skill scores at time r+1.
        active_mask: (N,) bool -- only compute for active entities.

    Returns:
        Scalar loss.
    """
    if active_mask is not None and active_mask.any():
        skill_curr = skill_curr[active_mask]
        skill_next = skill_next[active_mask]
    return F.smooth_l1_loss(skill_curr, skill_next)


# ---------------------------------------------------------------------------
# Loss manager
# ---------------------------------------------------------------------------


class KalmanLossManager:
    """Composes all Kalman-GNN losses with configurable weights.

    Args:
        lambda_pred: weight for beat-teammate BCE.
        lambda_smooth: weight for temporal smoothness.
        lambda_contrast: weight for InfoNCE contrastive loss.
        lambda_skill: weight for skill consistency.
        contrast_gap_min: minimum race gap for contrastive pairs.
        contrast_gap_max: maximum race gap for contrastive pairs.
        contrast_temperature: InfoNCE temperature.
    """

    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_contrast: float = 0.05,
        lambda_skill: float = 0.05,
        contrast_gap_min: int = 1,
        contrast_gap_max: int = 5,
        contrast_temperature: float = 0.1,
    ):
        self.lambda_pred = lambda_pred
        self.lambda_smooth = lambda_smooth
        self.lambda_contrast = lambda_contrast
        self.lambda_skill = lambda_skill
        self.contrast_gap_min = contrast_gap_min
        self.contrast_gap_max = contrast_gap_max
        self.contrast_temperature = contrast_temperature

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        v_drivers_curr: torch.Tensor,
        v_drivers_prev: torch.Tensor,
        v_constructors_curr: torch.Tensor,
        v_constructors_prev: torch.Tensor,
        active_driver_ids: torch.Tensor,
        active_constructor_ids: torch.Tensor,
        driver_emb_history: list[torch.Tensor] | None = None,
        driver_active_history: list[torch.Tensor] | None = None,
        skill_curr: torch.Tensor | None = None,
        skill_prev: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses and return a dict with individual terms + total.

        Returns:
            Dict with keys: total, pred, smooth, contrast, skill.
        """
        device = v_drivers_curr.device

        # L_pred: beat-teammate
        loss_pred = beat_teammate_loss(logits, labels)

        # L_smooth: temporal smoothness for drivers and constructors
        active_drv_mask = torch.zeros(v_drivers_curr.shape[0], dtype=torch.bool, device=device)
        if len(active_driver_ids) > 0:
            active_drv_mask[active_driver_ids] = True
        loss_smooth_drv = smoothness_loss(v_drivers_curr, v_drivers_prev, active_drv_mask)

        active_cons_mask = torch.zeros(v_constructors_curr.shape[0], dtype=torch.bool, device=device)
        if len(active_constructor_ids) > 0:
            active_cons_mask[active_constructor_ids] = True
        loss_smooth_cons = smoothness_loss(v_constructors_curr, v_constructors_prev, active_cons_mask)

        loss_smooth = loss_smooth_drv + loss_smooth_cons

        # L_contrast: InfoNCE (computed periodically)
        loss_contrast = torch.tensor(0.0, device=device)
        if (
            self.lambda_contrast > 0
            and driver_emb_history is not None
            and driver_active_history is not None
            and len(driver_emb_history) >= self.contrast_gap_min + 1
        ):
            loss_contrast = temporal_infonce_loss(
                driver_emb_history,
                driver_active_history,
                gap_min=self.contrast_gap_min,
                gap_max=self.contrast_gap_max,
                temperature=self.contrast_temperature,
            )

        # L_skill: skill consistency
        loss_skill = torch.tensor(0.0, device=device)
        if skill_curr is not None and skill_prev is not None:
            loss_skill = skill_consistency_loss(skill_curr, skill_prev, active_drv_mask)

        # Total
        total = (
            self.lambda_pred * loss_pred
            + self.lambda_smooth * loss_smooth
            + self.lambda_contrast * loss_contrast
            + self.lambda_skill * loss_skill
        )

        return {
            "total": total,
            "pred": loss_pred,
            "smooth": loss_smooth,
            "contrast": loss_contrast,
            "skill": loss_skill,
        }