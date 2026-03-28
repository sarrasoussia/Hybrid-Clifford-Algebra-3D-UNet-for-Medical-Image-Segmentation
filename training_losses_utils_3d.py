"""
Utilities to combine segmentation losses for 3D multi-head models.

Example usage:
  total_loss = compute_total_segmentation_loss(
      seg_logits=seg_logits,
      boundary_logits=boundary_logits,
      labels=labels,
      dice_loss_fn=...,
      boundary_loss_fn=...,
      distance_transform_loss_fn=...,
      tv_loss_fn=...,
      ga_reg_weight=...,
      mv=multivectors,
      weights=...
  )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ga_regularization_3d import (
    multivector_magnitude_inconsistency_loss,
    vector_component_smoothness_loss,
    grade_wise_regularization_loss,
)


@dataclass
class LossWeights3D:
    dice: float = 1.0
    boundary: float = 1.0
    tv: float = 0.0
    distance_transform: float = 0.0
    ga_magnitude_smooth: float = 0.0
    ga_vector_smooth: float = 0.0
    ga_grade_wise: float = 0.0


def compute_total_segmentation_loss(
    *,
    seg_logits: torch.Tensor,  # (B,C,D,H,W)
    boundary_logits: torch.Tensor,  # (B,1,D,H,W)
    labels: torch.Tensor,  # (B,D,H,W) integer
    dice_loss_fn,
    boundary_loss_fn,
    distance_transform_loss_fn=None,
    tv_loss_fn=None,
    # GA regularization input (optional):
    mv: Optional[torch.Tensor] = None,  # (B,C*8,D,H,W)
    weights: LossWeights3D = LossWeights3D(),
    foreground_class: int = 1,
) -> torch.Tensor:
    """
    Combines:
      total_loss = dice + boundary*w + ... (optional regularizers)
    """
    total = seg_logits.new_tensor(0.0)

    # Dice input expects probabilities in many implementations.
    seg_prob = F.softmax(seg_logits, dim=1)
    dice = dice_loss_fn(seg_prob, labels) if weights.dice != 0.0 else seg_logits.new_tensor(0.0)
    total = total + weights.dice * dice

    boundary = boundary_loss_fn(boundary_logits, labels) if weights.boundary != 0.0 else seg_logits.new_tensor(0.0)
    total = total + weights.boundary * boundary

    if weights.tv != 0.0 and tv_loss_fn is not None:
        # TV on foreground probability channel
        pred_fg = seg_prob[:, foreground_class : foreground_class + 1, ...]
        tv_loss = tv_loss_fn(pred_fg)
        total = total + weights.tv * tv_loss

    if weights.distance_transform != 0.0 and distance_transform_loss_fn is not None:
        pred_fg = seg_prob[:, foreground_class : foreground_class + 1, ...]
        dt_loss = distance_transform_loss_fn(pred_fg, labels)
        total = total + weights.distance_transform * dt_loss

    if mv is not None:
        if weights.ga_magnitude_smooth != 0.0:
            total = total + weights.ga_magnitude_smooth * multivector_magnitude_inconsistency_loss(mv)
        if weights.ga_vector_smooth != 0.0:
            total = total + weights.ga_vector_smooth * vector_component_smoothness_loss(mv)
        if weights.ga_grade_wise != 0.0:
            total = total + weights.ga_grade_wise * grade_wise_regularization_loss(mv)

    return total


__all__ = ["LossWeights3D", "compute_total_segmentation_loss"]

