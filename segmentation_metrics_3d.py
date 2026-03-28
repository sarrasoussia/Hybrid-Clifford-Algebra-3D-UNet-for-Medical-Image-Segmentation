"""
Spatial Evaluation Metrics for 3D Segmentation.

Implements:
  - Dice coefficient (per class + mean)
  - Hausdorff distance (95th percentile) using surface voxels
  - Mean surface distance (MSD)
  - Connected component counting (number of objects)

Designed for predictions shaped:
  - logits/prob: (B, C, D, H, W)
  - labels:     (B, D, H, W)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Args:
        pred: (B, C, D, H, W) logits or probabilities
        target: (B, D, H, W) int labels
    Returns:
        dice: (C,) dice score (averaged over batch)
    """
    if pred.ndim != 5 or target.ndim != 4:
        raise ValueError("Expected pred (B,C,D,H,W) and target (B,D,H,W)")

    if pred.shape[1] == 1:
        # Binary case: pred is foreground logits/prob
        prob = torch.sigmoid(pred)
        y = (target > 0).to(prob.dtype).unsqueeze(1)
        intersection = (prob * y).sum(dim=(2, 3, 4))
        union = prob.sum(dim=(2, 3, 4)) + y.sum(dim=(2, 3, 4))
        dice = (2 * intersection + smooth) / (union + smooth)  # (B,1)
        return dice.mean(dim=0).view(1)

    # Multiclass
    prob = F.softmax(pred, dim=1)
    C = prob.shape[1]
    y_one_hot = F.one_hot(target.long(), num_classes=C).permute(0, 4, 1, 2, 3).to(prob.dtype)

    pred_flat = prob.reshape(prob.shape[0], C, -1)
    y_flat = y_one_hot.reshape(y_one_hot.shape[0], C, -1)

    intersection = (pred_flat * y_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + y_flat.sum(dim=2)
    dice = (2 * intersection + smooth) / (union + smooth)  # (B,C)
    dice = dice.mean(dim=0)  # (C,)

    if ignore_index is not None:
        dice = torch.cat([dice[:ignore_index], dice[ignore_index + 1 :]], dim=0)

    return dice


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    # Surface = voxels that differ from eroded version
    eroded = ndimage.binary_erosion(mask)
    return mask ^ eroded


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    pred/target: binary numpy arrays (D,H,W)
    """
    from scipy.spatial.distance import directed_hausdorff

    if pred.sum() == 0 or target.sum() == 0:
        return 1000.0

    pred_surface = _surface_voxels(pred)
    target_surface = _surface_voxels(target)

    pred_coords = np.column_stack(np.where(pred_surface))
    target_coords = np.column_stack(np.where(target_surface))

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 1000.0

    pred_coords = pred_coords * np.array(spacing)[None, :]
    target_coords = target_coords * np.array(spacing)[None, :]

    d1 = directed_hausdorff(pred_coords, target_coords)[0]
    d2 = directed_hausdorff(target_coords, pred_coords)[0]
    return float(np.percentile([d1, d2], 95))


def mean_surface_distance(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    pred/target: binary numpy arrays (D,H,W)
    """
    from scipy.spatial.distance import cdist

    if pred.sum() == 0 or target.sum() == 0:
        return 1000.0

    pred_surface = _surface_voxels(pred)
    target_surface = _surface_voxels(target)

    pred_coords = np.column_stack(np.where(pred_surface))
    target_coords = np.column_stack(np.where(target_surface))

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 1000.0

    pred_coords = pred_coords * np.array(spacing)[None, :]
    target_coords = target_coords * np.array(spacing)[None, :]

    dist_pred_to_target = cdist(pred_coords, target_coords).min(axis=1)
    dist_target_to_pred = cdist(target_coords, pred_coords).min(axis=1)

    return float(np.mean([dist_pred_to_target.mean(), dist_target_to_pred.mean()]))


def connected_components_count(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.5,
    connectivity: int = 1,
) -> Dict[str, int]:
    """
    pred/target: binary arrays (D,H,W) or float arrays.
    Returns counts for predicted objects and GT objects.
    """
    from scipy import ndimage

    pred_bin = pred > threshold if pred.dtype.kind != "b" else pred
    target_bin = target > threshold if target.dtype.kind != "b" else target

    structure = np.ones((3, 3, 3), dtype=np.int32)
    if connectivity == 0:
        structure = np.zeros((3, 3, 3), dtype=np.int32)
        structure[1, 1, 1] = 1

    pred_labeled, pred_n = ndimage.label(pred_bin, structure=structure)
    target_labeled, target_n = ndimage.label(target_bin, structure=structure)
    return {"pred_objects": int(pred_n), "gt_objects": int(target_n)}


__all__ = [
    "dice_coefficient",
    "hausdorff_distance_95",
    "mean_surface_distance",
    "connected_components_count",
]

