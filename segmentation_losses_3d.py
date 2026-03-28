"""
3D Spatial Consistency Losses for Segmentation

Implements:
  - Total Variation loss (TV) on predicted probabilities
  - Boundary loss using 3D Sobel edge detection on predictions and GT
  - Distance Transform loss (DTL) using scipy distance transform of GT

All losses are compatible with tensors shaped:
  - (B, C, D, H, W) for predictions/probabilities
  - (B, D, H, W) for integer ground-truth labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVariationLoss3D(nn.Module):
    """Encourages spatial smoothness on probability maps using anisotropic TV."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prob: (B, C, D, H, W), expected to be probabilities in [0,1] (or logits).
        """
        if prob.ndim != 5:
            raise ValueError(f"Expected prob.ndim=5, got {prob.ndim}")

        # Differences along spatial axes
        dD = torch.abs(prob[:, :, 1:, :, :] - prob[:, :, :-1, :, :])
        dH = torch.abs(prob[:, :, :, 1:, :] - prob[:, :, :, :-1, :])
        dW = torch.abs(prob[:, :, :, :, 1:] - prob[:, :, :, :, :-1])

        if self.reduction == "mean":
            return dD.mean() + dH.mean() + dW.mean()
        return dD.sum() + dH.sum() + dW.sum()


class _Sobel3D(nn.Module):
    """Fixed 3D Sobel filters producing gradient components gx, gy, gz."""

    def __init__(self, in_channels: int):
        super().__init__()

        # Standard separable Sobel in 3D built from:
        #   deriv = [-1, 0, 1]
        #   smooth = [1, 2, 1]
        #
        # With tensor layout (D, H, W):
        #   gx -> derivative along W
        #   gy -> derivative along H
        #   gz -> derivative along D
        deriv = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        smooth = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)

        # gx (D,H,W) = smooth(D) * smooth(H) * deriv(W)
        gx = smooth[:, None, None] * smooth[None, :, None] * deriv[None, None, :]
        # gy (D,H,W) = smooth(D) * deriv(H)   * smooth(W)
        gy = smooth[:, None, None] * deriv[None, :, None] * smooth[None, None, :]
        # gz (D,H,W) = deriv(D)   * smooth(H) * smooth(W)
        gz = deriv[:, None, None] * smooth[None, :, None] * smooth[None, None, :]

        # Convert to conv3d weight shape: (out_channels, in_channels/groups, kD,kH,kW)
        # We'll use groups=in_channels so each input channel has its own copy.
        gx_w = gx[None, None, :, :, :].repeat(in_channels, 1, 1, 1, 1)
        gy_w = gy[None, None, :, :, :].repeat(in_channels, 1, 1, 1, 1)
        gz_w = gz[None, None, :, :, :].repeat(in_channels, 1, 1, 1, 1)

        self.register_buffer("gx_w", gx_w)
        self.register_buffer("gy_w", gy_w)
        self.register_buffer("gz_w", gz_w)

        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (gx, gy, gz): each (B, C, D, H, W)
        """
        if x.ndim != 5:
            raise ValueError("Expected x to have shape (B,C,D,H,W)")
        B, C, D, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        gx = F.conv3d(x, self.gx_w, padding=1, groups=C)
        gy = F.conv3d(x, self.gy_w, padding=1, groups=C)
        gz = F.conv3d(x, self.gz_w, padding=1, groups=C)
        return gx, gy, gz


def sobel_edge_magnitude_3d(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute 3D Sobel gradient magnitude for a single-channel tensor.

    Args:
        x: (B, 1, D, H, W)
    Returns:
        edge magnitude: (B, 1, D, H, W)
    """
    # Cache kernels by dtype/device to avoid rebuilding filters every call.
    # (Since Sobel weights are fixed, this improves performance in training loops.)
    global _SOBEL3D_CACHE  # type: ignore[name-defined]
    try:
        cache = _SOBEL3D_CACHE
    except Exception:
        cache = {}
        _SOBEL3D_CACHE = cache

    key = (x.device.type, str(x.device), x.dtype)
    if key not in cache:
        cache[key] = _Sobel3D(in_channels=1).to(dtype=x.dtype, device=x.device)
    gx, gy, gz = cache[key](x)
    return torch.sqrt(gx * gx + gy * gy + gz * gz + eps)


class BoundaryLoss3D(nn.Module):
    """
    Boundary loss using 3D Sobel edge detection.

    Assumes the model provides boundary logits of shape:
        boundary_logits: (B, 1, D, H, W)
    And the ground truth segmentation labels are:
        target: (B, D, H, W) integer class ids
    """

    def __init__(
        self,
        foreground_class: int = 1,
        boundary_target_normalize: bool = True,
    ):
        super().__init__()
        self.foreground_class = foreground_class
        self.boundary_target_normalize = boundary_target_normalize

    def forward(self, boundary_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundary_logits: (B, 1, D, H, W), raw logits from boundary head
            target: (B, D, H, W), integer segmentation labels
        """
        if boundary_logits.ndim != 5 or boundary_logits.size(1) != 1:
            raise ValueError("boundary_logits must be (B, 1, D, H, W)")
        if target.ndim != 4:
            raise ValueError("target must be (B, D, H, W)")

        # GT foreground mask -> float for filtering
        fg = (target == self.foreground_class).to(dtype=boundary_logits.dtype).unsqueeze(1)  # (B,1,D,H,W)

        # Sobel magnitude on GT and normalize to [0,1] for stable BCEWithLogits.
        with torch.no_grad():
            edge_gt = sobel_edge_magnitude_3d(fg)  # (B,1,D,H,W)
            if self.boundary_target_normalize:
                maxv = edge_gt.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
                edge_gt = edge_gt / maxv
            edge_gt = edge_gt.clamp(0.0, 1.0)

        return F.binary_cross_entropy_with_logits(boundary_logits, edge_gt)


class DistanceTransformLoss3D(nn.Module):
    """
    Distance transform loss (DTL) computed from the ground-truth mask.

    Uses SciPy's `distance_transform_edt` to generate a distance-to-structure
    weighting map. Then penalizes squared error between predicted foreground
    probability and GT foreground mask with that weighting.

    Note: this loss is not fully differentiable w.r.t. the distance map (it is GT-only).
    """

    def __init__(
        self,
        foreground_class: int = 1,
        weight_normalize: bool = True,
        use_squared_error: bool = True,
    ):
        super().__init__()
        self.foreground_class = foreground_class
        self.weight_normalize = weight_normalize
        self.use_squared_error = use_squared_error

    def forward(self, pred_foreground_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_foreground_prob: (B, 1, D, H, W) in [0,1]
            target: (B, D, H, W) integer labels
        """
        if pred_foreground_prob.ndim != 5 or pred_foreground_prob.size(1) != 1:
            raise ValueError("pred_foreground_prob must be (B,1,D,H,W)")
        if target.ndim != 4:
            raise ValueError("target must be (B,D,H,W)")

        try:
            import numpy as np
            from scipy.ndimage import distance_transform_edt
        except Exception as e:
            raise ImportError(
                "DistanceTransformLoss3D requires scipy. Please install scipy to use this loss."
            ) from e

        # GT foreground mask (B,1,D,H,W) float
        y = (target == self.foreground_class).to(dtype=pred_foreground_prob.dtype).unsqueeze(1)

        # Distance map computed per-batch item (GT-only).
        # dist_to_fg: distance to closest foreground voxel
        # dist_to_bg: distance to closest background voxel
        # weight = dist_to_fg + dist_to_bg (large away from boundary)
        with torch.no_grad():
            y_np = y.detach().cpu().numpy().astype(np.bool_)
            w_list = []
            for b in range(y_np.shape[0]):
                mask = y_np[b, 0]  # (D,H,W) bool
                dist_to_fg = distance_transform_edt(~mask)
                dist_to_bg = distance_transform_edt(mask)
                w = dist_to_fg + dist_to_bg
                w_list.append(w)
            w = torch.from_numpy(np.stack(w_list, axis=0)).to(device=pred_foreground_prob.device, dtype=pred_foreground_prob.dtype)

            if self.weight_normalize:
                w = w / w.mean().clamp_min(1e-8)

        if self.use_squared_error:
            err = (pred_foreground_prob - y) ** 2
        else:
            err = torch.abs(pred_foreground_prob - y)

        return (w.unsqueeze(1) * err).mean()


__all__ = [
    "TotalVariationLoss3D",
    "BoundaryLoss3D",
    "DistanceTransformLoss3D",
]

