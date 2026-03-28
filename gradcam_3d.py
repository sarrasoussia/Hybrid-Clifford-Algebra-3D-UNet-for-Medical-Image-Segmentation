"""
Grad-CAM utilities for 3D segmentation models.

Supports:
  - Standard Grad-CAM on a target 3D conv layer (heatmap: BxDxHxW)
  - Grade-wise Grad-CAM for multivector tensors shaped (B, C*8, D, H, W)
    using the same Cl(3,0) basis ordering:
      [scalar, e1, e2, e3, e12, e13, e23, e123]
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


class GradCAM3D:
    """
    Generic Grad-CAM for 3D CNNs.

    The model can return either:
      - seg_logits: (B,C,D,H,W)
      - (seg_logits, boundary_logits, ...) tuple/list
      - dict with 'seg_logits' key
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []

        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_module, _inp, output):
            self.activations = output

        def bwd_hook(_module, _grad_input, grad_output):
            # grad_output[0] matches activations shape
            self.gradients = grad_output[0]

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        # Use full backward hook if available (better for autograd).
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))
        else:
            self._handles.append(self.target_layer.register_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @staticmethod
    def _extract_seg_logits(outputs):
        if isinstance(outputs, dict):
            if "seg_logits" not in outputs:
                raise KeyError("Expected dict output to have 'seg_logits'")
            return outputs["seg_logits"]
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def generate(
        self,
        x: torch.Tensor,
        *,
        target_class: Optional[Union[int, torch.Tensor]] = None,
        output_normalize: bool = True,
        use_segmentation_output: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, Cin, D, H, W)
            target_class:
              - int: same class for whole batch
              - tensor (B,): per-item class index
              - None: uses argmax over classes
        Returns:
            heatmap: (B, D, H, W), normalized to [0,1] if output_normalize
        """
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        outputs = self.model(x) if use_segmentation_output else self.model(x)
        seg_logits = self._extract_seg_logits(outputs)
        if seg_logits.ndim != 5:
            raise ValueError("Expected seg_logits with shape (B,C,D,H,W)")

        B, C, D, H, W = seg_logits.shape

        if target_class is None:
            target_class = seg_logits.argmax(dim=1)  # (B,)

        if isinstance(target_class, int):
            score = seg_logits[:, target_class].sum()
        else:
            if target_class.shape != (B,):
                raise ValueError("target_class tensor must have shape (B,)")
            score = seg_logits.gather(1, target_class.view(B, 1, 1, 1, 1).expand(-1, 1, D, H, W)).sum()

        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        A = self.activations  # (B,C_feat,D,H,W)
        dY = self.gradients  # (B,C_feat,D,H,W)

        # weights: (B,C_feat,1,1,1)
        weights = dY.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * A).sum(dim=1)  # (B,D,H,W)
        cam = F.relu(cam)

        if output_normalize:
            cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
            cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()


def grade_wise_gradcam_from_multivector(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    *,
    multivector_dim: int = 8,
    eps: float = 1e-8,
) -> dict:
    """
    Compute grade-wise Grad-CAM heatmaps from multivector activations.

    Args:
        activations: (B, C_mv*8, D, H, W)
        gradients:   same shape as activations
    Returns:
        dict of heatmaps each (B, D, H, W):
          - scalar
          - vector
          - bivector
          - trivector
    """
    if activations.shape != gradients.shape:
        raise ValueError("activations and gradients must have the same shape")
    if activations.ndim != 5:
        raise ValueError("Expected (B,C*8,D,H,W)")

    B, CM, D, H, W = activations.shape
    if CM % multivector_dim != 0:
        raise ValueError("channels must be divisible by 8")
    C_mv = CM // multivector_dim

    A = activations.view(B, C_mv, multivector_dim, D, H, W)
    dY = gradients.view(B, C_mv, multivector_dim, D, H, W)

    # weights per multivector channel and component: mean over spatial dims
    weights = dY.mean(dim=(3, 4, 5), keepdim=True)  # (B,C_mv,8,1,1,1)

    def cam_for_indices(indices: Sequence[int]) -> torch.Tensor:
        idx = torch.tensor(indices, device=activations.device, dtype=torch.long)
        # Sum over multivector channels and selected component indices.
        # Result: (B, D, H, W)
        Aw = (weights[:, :, idx] * A[:, :, idx]).sum(dim=(1, 2))
        cam = F.relu(Aw)
        # normalize per-sample
        cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
        cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min + eps)

    return {
        "scalar": cam_for_indices([0]),
        "vector": cam_for_indices([1, 2, 3]),
        "bivector": cam_for_indices([4, 5, 6]),
        "trivector": cam_for_indices([7]),
    }


__all__ = ["GradCAM3D", "grade_wise_gradcam_from_multivector", "FeatureMapHook"]


class FeatureMapHook:
    """
    Minimal forward-hook helper to capture intermediate 3D feature maps.

    Useful for debugging / visualization.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.value: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _inp, output):
        self.value = output

    def close(self):
        self._handle.remove()


