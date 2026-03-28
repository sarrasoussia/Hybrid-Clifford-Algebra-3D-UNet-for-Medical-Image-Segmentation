"""
Segmentation models with an auxiliary boundary head.

This module provides:
  - `SegmentationWithBoundaryHead`: wraps any segmentation model returning
       seg_logits: (B, C, D, H, W)
    and adds:
       boundary_logits: (B, 1, D, H, W)

  - `GARefinementWithBoundaryHead`: performs a lightweight GA refinement
    on the segmentation logits using:
       CliffordProjection3D -> CliffordConv3D -> projection back to logits
    and also returns multivector features for interpretability/regularization.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from clifford_layers import CliffordConv3D, CliffordProjection3D


class SegmentationWithBoundaryHead(nn.Module):
    """
    Add a boundary prediction head to an existing segmentation model.

    Output format:
      (seg_logits, boundary_logits)
    """

    def __init__(
        self,
        seg_model: nn.Module,
        seg_out_channels: int,
        *,
        boundary_out_channels: int = 1,
    ):
        super().__init__()
        self.seg_model = seg_model
        self.boundary_head = nn.Conv3d(
            seg_out_channels, boundary_out_channels, kernel_size=1, bias=True
        )

    def forward(self, x: torch.Tensor):
        seg_logits = self.seg_model(x)
        boundary_logits = self.boundary_head(seg_logits)
        return seg_logits, boundary_logits


class GARefinementWithBoundaryHead(nn.Module):
    """
    GA refinement on segmentation logits + boundary prediction head.

    Pipeline:
      seg_logits (B,C,D,H,W)
        -> CliffordProjection3D (B,clifford_dim*8,D,H,W)
        -> CliffordConv3D (B,clifford_dim*8,D,H,W)
        -> project back to seg logits (B,C,D,H,W)
        -> boundary head on refined logits (B,1,D,H,W)

    Forward output:
      (refined_seg_logits, boundary_logits, multivectors)
    """

    def __init__(
        self,
        backbone: nn.Module,
        seg_out_channels: int,
        clifford_dim: int = 16,
        *,
        restrict_grades: bool = False,
        use_grade_wise_scaling: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.seg_out_channels = seg_out_channels
        self.clifford_dim = clifford_dim

        self.proj_in = CliffordProjection3D(
            in_channels=seg_out_channels,
            out_channels=clifford_dim,
            use_grade_wise_scaling=use_grade_wise_scaling,
        )

        self.clifford_conv = CliffordConv3D(
            in_channels=clifford_dim,
            out_channels=clifford_dim,
            kernel_size=1,
            restrict_grades=restrict_grades,
            use_grade_wise_scaling=False,
        )

        self.proj_out = nn.Conv3d(
            clifford_dim * 8, seg_out_channels, kernel_size=1, bias=True
        )

        self.boundary_head = nn.Conv3d(seg_out_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        seg_logits = self.backbone(x)  # (B,C,D,H,W)

        mv = self.proj_in(seg_logits)  # (B,clifford_dim*8,D,H,W)
        mv = self.clifford_conv(mv)  # (B,clifford_dim*8,D,H,W)

        refined_logits = self.proj_out(mv)  # (B,C,D,H,W)
        boundary_logits = self.boundary_head(refined_logits)  # (B,1,D,H,W)

        return refined_logits, boundary_logits, mv


__all__ = [
    "SegmentationWithBoundaryHead",
    "GARefinementWithBoundaryHead",
]

