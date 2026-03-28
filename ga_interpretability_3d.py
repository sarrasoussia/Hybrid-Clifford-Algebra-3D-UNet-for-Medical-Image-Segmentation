"""
Geometric Algebra (Cl(3,0)) multivector interpretability utilities for 3D.

Assumes multivectors are stored in a channel-major tensor:
  (B, C * 8, D, H, W)
where the 8 components follow the basis ordering used by `CliffordConv3D`:
  [scalar, e1, e2, e3, e12, e13, e23, e123]
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch


def decompose_multivectors_Cl3_0(mv: torch.Tensor, *, multivector_dim: int = 8) -> Dict[str, torch.Tensor]:
    """
    Decompose multivector tensor into grade components.

    Args:
        mv: (B, C*8, D, H, W)
    Returns:
        dict with:
          - scalar:   (B, C, D, H, W)
          - vector:   (B, C, 3, D, H, W)
          - bivector: (B, C, 3, D, H, W)
          - trivector:(B, C, D, H, W)
    """
    if mv.ndim != 5:
        raise ValueError("mv must be (B, C*8, D, H, W)")
    B, CM, D, H, W = mv.shape
    if CM % multivector_dim != 0:
        raise ValueError(f"Expected channels multiple of {multivector_dim}, got {CM}")
    C = CM // multivector_dim

    mv = mv.view(B, C, multivector_dim, D, H, W)

    scalar = mv[:, :, 0, :, :, :]  # (B,C,D,H,W)
    vector = mv[:, :, 1:4, :, :, :]  # (B,C,3,D,H,W)
    bivector = mv[:, :, 4:7, :, :, :]  # (B,C,3,D,H,W)
    trivector = mv[:, :, 7, :, :, :]  # (B,C,D,H,W)

    return {
        "scalar": scalar,
        "vector": vector,
        "bivector": bivector,
        "trivector": trivector,
    }


def grade_magnitudes_Cl3_0(
    mv: torch.Tensor, *, eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """
    Compute magnitude maps per grade.

    Returns:
        - scalar_mag:   (B,C,D,H,W)
        - vector_mag:   (B,C,D,H,W)
        - bivector_mag: (B,C,D,H,W)
        - trivector_mag:(B,C,D,H,W)
    """
    parts = decompose_multivectors_Cl3_0(mv)

    scalar_mag = torch.abs(parts["scalar"])
    vector_mag = torch.sqrt((parts["vector"] ** 2).sum(dim=2) + eps)
    bivector_mag = torch.sqrt((parts["bivector"] ** 2).sum(dim=2) + eps)
    trivector_mag = torch.abs(parts["trivector"])

    return {
        "scalar_mag": scalar_mag,
        "vector_mag": vector_mag,
        "bivector_mag": bivector_mag,
        "trivector_mag": trivector_mag,
    }


def _normalize_to_uint8(img: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = img.detach().to(torch.float32)
    x = x - x.amin()
    x = x / (x.amax() + eps)
    return (x * 255.0).clamp(0, 255).to(torch.uint8)


def save_grade_slice_projections(
    mv: torch.Tensor,
    *,
    save_dir: str,
    prefix: str = "mv",
    slice_index: Optional[int] = None,
    axis: str = "D",
    aggregate_over_multivector_channels: str = "mean",
):
    """
    Save grade magnitudes for a single 2D slice (projection) from a multivector volume.

    This is meant for interpretability. It does not iterate over voxels; it uses slicing
    and reduction ops on tensors.
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    parts = grade_magnitudes_Cl3_0(mv)  # each (B,C,D,H,W)
    # Use first batch element
    sl = 0

    # Choose slice along depth (D) by default.
    _, C, D, H, W = parts["scalar_mag"].shape
    if axis == "D":
        max_len = D
        if slice_index is None:
            slice_index = D // 2
        scalar_2d = parts["scalar_mag"][sl, :, slice_index, :, :]  # (C,H,W)
        vector_2d = parts["vector_mag"][sl, :, slice_index, :, :]
        bivector_2d = parts["bivector_mag"][sl, :, slice_index, :, :]
        trivector_2d = parts["trivector_mag"][sl, :, slice_index, :, :]
    elif axis == "H":
        max_len = H
        if slice_index is None:
            slice_index = H // 2
        scalar_2d = parts["scalar_mag"][sl, :, :, slice_index, :]  # (C,D,W)
        vector_2d = parts["vector_mag"][sl, :, :, slice_index, :]
        bivector_2d = parts["bivector_mag"][sl, :, :, slice_index, :]
        trivector_2d = parts["trivector_mag"][sl, :, :, slice_index, :]
    elif axis == "W":
        max_len = W
        if slice_index is None:
            slice_index = W // 2
        scalar_2d = parts["scalar_mag"][sl, :, :, :, slice_index]  # (C,D,H)
        vector_2d = parts["vector_mag"][sl, :, :, :, slice_index]
        bivector_2d = parts["bivector_mag"][sl, :, :, :, slice_index]
        trivector_2d = parts["trivector_mag"][sl, :, :, :, slice_index]
    else:
        raise ValueError("axis must be one of {'D','H','W'}")

    def agg(x: torch.Tensor) -> torch.Tensor:
        if aggregate_over_multivector_channels == "mean":
            return x.mean(dim=0)
        if aggregate_over_multivector_channels == "max":
            return x.max(dim=0).values
        raise ValueError("aggregate_over_multivector_channels must be 'mean' or 'max'")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, img, title in [
        (axes[0], agg(scalar_2d), "Scalar |mag|"),
        (axes[1], agg(vector_2d), "Vector |mag|"),
        (axes[2], agg(bivector_2d), "Bivector |mag|"),
        (axes[3], agg(trivector_2d), "Trivector |mag|"),
    ]:
        ax.imshow(img.detach().cpu().numpy(), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")

    out_path = os.path.join(save_dir, f"{prefix}_{axis}{slice_index}_grades.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_basis_component_slice(
    mv: torch.Tensor,
    *,
    save_dir: str,
    prefix: str = "mv",
    slice_index: Optional[int] = None,
    axis: str = "D",
):
    """
    Save absolute values of individual basis components as separate 2D maps.

    Components follow:
      scalar,
      vector: e1,e2,e3,
      bivector: e12,e13,e23,
      trivector
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    parts = decompose_multivectors_Cl3_0(mv)
    sl = 0
    _, C, D, H, W = parts["scalar"].shape

    if axis == "D":
        if slice_index is None:
            slice_index = D // 2
        scalar_2d = torch.abs(parts["scalar"])[sl, :, slice_index]  # (C,H,W)
        vector_2d = torch.abs(parts["vector"])[sl, :, :, slice_index]  # (C,3,H,W)
        bivector_2d = torch.abs(parts["bivector"])[sl, :, :, slice_index]  # (C,3,H,W)
        trivector_2d = torch.abs(parts["trivector"])[sl, :, slice_index]  # (C,H,W)
    else:
        # Keep implementation minimal: only D-slices are provided.
        raise ValueError("save_basis_component_slice currently supports axis='D' only")

    # Aggregate across multivector channels (mean)
    scalar_agg = scalar_2d.mean(dim=0)
    vector_agg = vector_2d.mean(dim=0)  # (3,H,W)
    bivector_agg = bivector_2d.mean(dim=0)  # (3,H,W)
    trivector_agg = trivector_2d.mean(dim=0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    imgs = [
        (scalar_agg, "scalar"),
        (vector_agg[0], "e1"),
        (vector_agg[1], "e2"),
        (vector_agg[2], "e3"),
        (bivector_agg[0], "e12"),
        (bivector_agg[1], "e13"),
        (bivector_agg[2], "e23"),
        (trivector_agg, "e123"),
    ]
    for ax, (im, title) in zip(axes, imgs):
        ax.imshow(im.detach().cpu().numpy(), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")

    out_path = os.path.join(save_dir, f"{prefix}_D{slice_index}_components.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def overlay_vector_magnitude_on_image(
    image: torch.Tensor,
    mv: torch.Tensor,
    *,
    save_path: str,
    slice_index: Optional[int] = None,
    axis: str = "D",
    alpha: float = 0.4,
):
    """
    Overlay vector magnitude on top of a grayscale image slice.

    Args:
        image: (B, 1, D, H, W) or (B, C, D, H, W); if C>1, it uses channel0.
        mv: multivector tensor (B, C*8, D, H, W)
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if image.ndim != 5:
        raise ValueError("image must be (B, C, D, H, W)")
    if image.size(0) != mv.size(0):
        raise ValueError("image and mv batch sizes must match")

    parts = grade_magnitudes_Cl3_0(mv)
    vector_mag = parts["vector_mag"]  # (B,C,D,H,W)

    sl = 0
    img_ch0 = image[sl, 0]

    _, C, D, H, W = vector_mag.shape
    if axis == "D":
        if slice_index is None:
            slice_index = D // 2
        img_2d = img_ch0[slice_index]  # (H,W)
        v_2d = vector_mag[sl, :, slice_index].mean(dim=0)  # (H,W)
    elif axis == "H":
        if slice_index is None:
            slice_index = H // 2
        img_2d = img_ch0[:, slice_index, :]  # (D,W)
        v_2d = vector_mag[sl, :, :, slice_index].mean(dim=0)
    elif axis == "W":
        if slice_index is None:
            slice_index = W // 2
        img_2d = img_ch0[:, :, slice_index]  # (D,H)
        v_2d = vector_mag[sl, :, :, :, slice_index].mean(dim=0)
    else:
        raise ValueError("axis must be one of {'D','H','W'}")

    # Normalize overlay
    img_u8 = _normalize_to_uint8(img_2d)
    v_u8 = _normalize_to_uint8(v_2d)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_u8.cpu().numpy(), cmap="gray", interpolation="nearest")
    ax.imshow(v_u8.cpu().numpy(), cmap="magma", alpha=alpha, interpolation="nearest")
    ax.axis("off")
    ax.set_title("Vector magnitude overlay")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


__all__ = [
    "decompose_multivectors_Cl3_0",
    "grade_magnitudes_Cl3_0",
    "save_grade_slice_projections",
    "save_basis_component_slice",
    "overlay_vector_magnitude_on_image",
]

