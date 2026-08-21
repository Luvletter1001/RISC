"""Capped visual/text support fusion for FOCUS-EQText."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.detach().abs().max().cpu().item())
    return abs(float(value))


def _requires_grad(value: Any) -> bool:
    return bool(torch.is_tensor(value) and value.requires_grad)


class FocusDualSupportFusion(nn.Module):
    """Fuse native visual support with image-conditioned text support.

    The visual branch remains the exact fallback. When text is disabled or both
    visual/text residual alphas are zero, this module returns `e_visual`
    unchanged to keep baseline-equivalence tests strict.
    """

    def __init__(
            self,
            visual_weight_init: float = 0.9,
            text_weight_init: float = 0.1,
            max_text_weight: float = 0.2,
            eps: float = 1e-8) -> None:
        super().__init__()
        if visual_weight_init <= 0.0 or text_weight_init < 0.0:
            raise ValueError("fusion weights must be non-negative and visual > 0")
        if max_text_weight > 0.2:
            raise ValueError("max_text_weight must be <= 0.2")
        total = float(visual_weight_init) + float(text_weight_init)
        text_init = float(text_weight_init) / total
        visual_init = 1.0 - text_init
        self.visual_logit = nn.Parameter(torch.tensor(visual_init).log())
        self.text_logit = nn.Parameter(torch.tensor(max(text_init, eps)).log())
        self.max_text_weight = float(max_text_weight)
        self.eps = float(eps)

    def _weights(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        raw = torch.stack([
            self.visual_logit.to(device=device, dtype=dtype).exp(),
            self.text_logit.to(device=device, dtype=dtype).exp(),
        ])
        weights = raw / raw.sum().clamp_min(self.eps)
        text_weight = torch.clamp(weights[1], min=0.0, max=self.max_text_weight)
        visual_weight = 1.0 - text_weight
        return visual_weight, text_weight

    def _align_shapes(
            self,
            visual_support: Tensor,
            text_support: Tensor) -> Tuple[Tensor, Tensor]:
        if visual_support.shape == text_support.shape:
            return visual_support, text_support
        if (visual_support.dim() == 3
                and text_support.dim() == 4
                and visual_support.shape[0] == text_support.shape[0]
                and visual_support.shape[1:] == text_support.shape[2:]):
            visual_support = visual_support[:, None].expand_as(text_support)
            return visual_support, text_support
        if (visual_support.dim() == 4
                and text_support.dim() == 3
                and visual_support.shape[0] == text_support.shape[0]
                and visual_support.shape[2:] == text_support.shape[1:]):
            text_support = text_support[:, None].expand_as(visual_support)
            return visual_support, text_support
        return visual_support, text_support

    def forward(
            self,
            visual_support: Tensor,
            text_support: Optional[Tensor] = None,
            text_enabled: bool = True,
            visual_alpha: Optional[Any] = None,
            text_alpha: Optional[Any] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        if (not text_enabled) or text_support is None:
            return visual_support, {
                "dual_visual_weight": 1.0,
                "dual_text_weight": 0.0,
                "dual_fallback": "text_disabled",
            }

        text = text_support.to(
            device=visual_support.device, dtype=visual_support.dtype)
        visual_support, text = self._align_shapes(visual_support, text)
        if text.shape != visual_support.shape:
            raise ValueError(
                "visual and text support must have the same shape, got "
                f"{tuple(visual_support.shape)} vs {tuple(text.shape)}")
        visual_weight, text_weight = self._weights(
            visual_support.device, visual_support.dtype)
        fused = F.normalize(
            visual_support * visual_weight + text * text_weight,
            dim=-1)
        zero_alpha = (
            _as_float(visual_alpha) <= self.eps
            and _as_float(text_alpha) <= self.eps)
        alpha_trainable = (
            _requires_grad(visual_alpha)
            or _requires_grad(text_alpha))
        if zero_alpha:
            if torch.is_grad_enabled() and alpha_trainable:
                # Exact baseline value, fused-branch gradient for alpha=0 starts.
                straight_through = (
                    visual_support
                    + (fused - visual_support)
                    - (fused - visual_support).detach())
                return straight_through, {
                    "dual_visual_weight": float(
                        visual_weight.detach().cpu().item()),
                    "dual_text_weight": float(
                        text_weight.detach().cpu().item()),
                    "dual_fallback": "zero_alpha_straight_through",
                }
            return visual_support, {
                "dual_visual_weight": 1.0,
                "dual_text_weight": 0.0,
                "dual_fallback": "zero_alpha",
            }
        return fused, {
            "dual_visual_weight": float(visual_weight.detach().cpu().item()),
            "dual_text_weight": float(text_weight.detach().cpu().item()),
            "dual_fallback": "",
        }
