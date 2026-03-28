"""
Regularization losses that enforce smoothness / consistency of multivector fields
based on Geometric Algebra structure (Cl(3,0)).

All operations are tensorized. No voxel-wise Python loops.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from ga_interpretability_3d import grade_magnitudes_Cl3_0, decompose_multivectors_Cl3_0
from segmentation_losses_3d import TotalVariationLoss3D


def multivector_magnitude_inconsistency_loss(mv: torch.Tensor, *, tv_reduction: str = "mean") -> torch.Tensor:
    """
    Penalize inconsistent multivector magnitudes across space by applying TV loss
    on grade magnitudes and summing.

    Args:
        mv: (B, C*8, D, H, W)
    """
    parts = grade_magnitudes_Cl3_0(mv)  # each (B,C,D,H,W)
    tv = TotalVariationLoss3D(reduction=tv_reduction)
    # Stack magnitudes into a pseudo-channel tensor (B,4,C,D,H,W) isn't TV-friendly,
    # so we apply TV per grade and sum.
    loss = (
        tv(parts["scalar_mag"])
        + tv(parts["vector_mag"])
        + tv(parts["bivector_mag"])
        + tv(parts["trivector_mag"])
    )
    return loss


def vector_component_smoothness_loss(mv: torch.Tensor, *, tv_reduction: str = "mean") -> torch.Tensor:
    """
    Encourage smooth variation of vector components by applying TV loss on vector magnitude.
    """
    parts = grade_magnitudes_Cl3_0(mv)
    tv = TotalVariationLoss3D(reduction=tv_reduction)
    return tv(parts["vector_mag"])


def grade_wise_regularization_loss(
    mv: torch.Tensor,
    *,
    grades: Sequence[str] = ("scalar", "vector", "bivector", "trivector"),
    tv_reduction: str = "mean",
) -> torch.Tensor:
    """
    Grade-wise regularization: apply TV on selected grade magnitudes.
    """
    parts = grade_magnitudes_Cl3_0(mv)
    tv = TotalVariationLoss3D(reduction=tv_reduction)

    mapping = {
        "scalar": parts["scalar_mag"],
        "vector": parts["vector_mag"],
        "bivector": parts["bivector_mag"],
        "trivector": parts["trivector_mag"],
    }
    loss = None
    for g in grades:
        if g not in mapping:
            raise ValueError(f"Unknown grade '{g}'")
        comp = mapping[g]
        loss = comp.new_tensor(0.0) if loss is None else loss
        loss = loss + tv(comp)
    if loss is None:
        return mv.new_tensor(0.0)
    return loss


__all__ = [
    "multivector_magnitude_inconsistency_loss",
    "vector_component_smoothness_loss",
    "grade_wise_regularization_loss",
]

