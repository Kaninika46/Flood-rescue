"""Dice + focal loss for binary flood segmentation that ignores label -1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -1


def masked_dice_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    gamma: float = 2.0,
    alpha: float = 0.25,
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft Dice plus sigmoid focal loss over pixels whose label is not -1.

    Args:
        logits: ``(N, 1, H, W)`` or ``(N, H, W)`` raw scores.
        target: ``(N, H, W)`` integer labels: -1 ignore, 0 not water, 1 water.
        dice_weight, focal_weight: Weights of the two terms.
        gamma, alpha: Focal loss parameters (alpha weighs the water class).
        eps: Smoothing of the Dice ratio.

    Returns:
        A scalar. When every pixel is ignored it is 0 (and still has a gradient path).
    """
    if logits.dim() == 4:
        logits = logits[:, 0]
    logits = logits.float()
    valid = target != IGNORE_INDEX
    t = (target == 1).float()
    if not bool(valid.any()):
        return logits.sum() * 0.0

    v = valid.float()
    prob = torch.sigmoid(logits)
    inter = (prob * t * v).sum()
    dice = 1.0 - (2.0 * inter + eps) / ((prob * v).sum() + (t * v).sum() + eps)

    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p_t = prob * t + (1 - prob) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    focal = (alpha_t * (1 - p_t) ** gamma * bce * v).sum() / v.sum()
    return dice_weight * dice + focal_weight * focal
