"""Conservative Fourier-conditioned text shadow adapter.

This module is intentionally shadow-only. It consumes already-mapped text
support and Fourier cues, returns a diagnostic conditioned support tensor, and
records geometry. It does not replace the detector support bank and does not
modify final logits.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _flatten_code(code: Tensor) -> Tensor:
    if code.dim() == 4:
        batch, height, width, dim = code.shape
        return code.reshape(batch, height * width, dim)
    if code.dim() == 3:
        return code
    raise ValueError(
        f"fourier cue must be [B,N,D] or [B,H,W,D], got {tuple(code.shape)}")


def _flatten_confidence(confidence: Optional[Tensor], code: Tensor) -> Tensor:
    if confidence is None:
        return code.new_ones(code.shape[:2])
    if confidence.dim() == 3:
        return confidence.reshape(confidence.shape[0], -1).to(code)
    if confidence.dim() == 2:
        return confidence.to(code)
    raise ValueError(
        "confidence must be [B,N] or [B,H,W], "
        f"got {tuple(confidence.shape)}")


class FourierShadowTextAdapter(nn.Module):
    """Zero-init residual adapter for shadow text diagnostics only."""

    def __init__(
            self,
            support_dim: int,
            code_dim: int,
            apply_to_class_ids: Optional[Iterable[int]] = None,
            class_names: Optional[Sequence[str]] = None,
            apply_to_classes: Iterable[str] = ("small-vehicle",),
            allow_all_classes: bool = False,
            low_rank: int = 16,
            alpha_t_init: float = 0.0,
            alpha_t_max: float = 0.01,
            max_delta_norm_ratio: float = 0.01,
            eps: float = 1e-8) -> None:
        super().__init__()
        if float(alpha_t_max) > 0.01:
            raise ValueError("alpha_t_max must be <= 0.01 for T-Safe")
        self.support_dim = int(support_dim)
        self.code_dim = int(code_dim)
        self.apply_to_class_ids = (
            {int(v) for v in apply_to_class_ids}
            if apply_to_class_ids is not None else None)
        self.class_names = list(class_names) if class_names is not None else None
        self.apply_to_classes = {
            str(v).lower().replace("_", "-") for v in apply_to_classes
        }
        self.allow_all_classes = bool(allow_all_classes)
        self.alpha_t_max = float(alpha_t_max)
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.eps = float(eps)

        hidden_dim = max(1, int(low_rank))
        self.delta = nn.Sequential(
            nn.Linear(self.code_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.support_dim),
        )
        self.alpha_t = nn.Parameter(torch.tensor(float(alpha_t_init)))

    @property
    def modifies_final_logits(self) -> bool:
        return False

    @property
    def replaces_support_bank(self) -> bool:
        return False

    def _expand_support(self, text_support: Tensor, batch: int,
                        positions: int) -> Tensor:
        if text_support.dim() == 2:
            return text_support[None, None].expand(
                batch, positions, text_support.shape[0], text_support.shape[1])
        if text_support.dim() == 3:
            if text_support.shape[0] == 1 and batch > 1:
                text_support = text_support.expand(batch, -1, -1)
            if text_support.shape[0] != batch:
                raise ValueError(
                    "text support batch does not match Fourier cue batch: "
                    f"{text_support.shape[0]} vs {batch}")
            return text_support[:, None].expand(
                batch, positions, text_support.shape[1], text_support.shape[2])
        if text_support.dim() == 4:
            if text_support.shape[0] != batch or text_support.shape[1] != positions:
                raise ValueError(
                    "text support [B,N,M,D] must match Fourier cue [B,N,D], "
                    f"got {tuple(text_support.shape)} and batch/positions "
                    f"{batch}/{positions}")
            return text_support
        raise ValueError(
            "text support must be [M,D], [B,M,D], or [B,N,M,D], "
            f"got {tuple(text_support.shape)}")

    def _class_mask_from_names(
            self,
            support_count: int,
            device: torch.device,
            class_names: Optional[Sequence[str]]) -> Tensor:
        if self.allow_all_classes:
            return torch.ones(support_count, dtype=torch.bool, device=device)
        names = list(class_names) if class_names is not None else self.class_names
        if names is None:
            return torch.ones(support_count, dtype=torch.bool, device=device)
        mask = torch.zeros(support_count, dtype=torch.bool, device=device)
        for idx, name in enumerate(names[:support_count]):
            clean = str(name).lower().replace("_", "-")
            if clean in self.apply_to_classes:
                mask[idx] = True
        return mask

    def _class_mask(
            self,
            support_count: int,
            batch: int,
            device: torch.device,
            class_ids: Optional[Tensor],
            class_names: Optional[Sequence[str]]) -> Tensor:
        if self.allow_all_classes:
            return torch.ones(support_count, dtype=torch.bool, device=device)
        if class_ids is None:
            return self._class_mask_from_names(
                support_count, device, class_names)
        ids = class_ids.to(device=device)
        if ids.dim() == 1:
            ids = ids[None, :].expand(batch, ids.shape[0])
        if ids.dim() != 2:
            raise ValueError(
                f"class_ids must be [M] or [B,M], got {tuple(ids.shape)}")
        if ids.shape[0] == 1 and batch > 1:
            ids = ids.expand(batch, ids.shape[1])
        if ids.shape[0] != batch or ids.shape[1] != support_count:
            raise ValueError(
                "class_ids must match support shape, got "
                f"{tuple(ids.shape)} for batch/support {batch}/{support_count}")
        if self.apply_to_class_ids is not None:
            target = torch.zeros_like(ids, dtype=torch.bool)
            for class_id in self.apply_to_class_ids:
                target = target | (ids == int(class_id))
            return target
        return self._class_mask_from_names(support_count, device, class_names)

    def _interclass_cos(self, support: Tensor) -> Tuple[Tensor, Tensor]:
        normalized = F.normalize(support, dim=-1)
        cos = normalized @ normalized.transpose(-1, -2)
        support_count = cos.shape[-1]
        if support_count <= 1:
            zero = cos.new_zeros(cos.shape[:-2])
            return zero, zero
        mask = ~torch.eye(support_count, dtype=torch.bool, device=cos.device)
        off_diag = cos[..., mask].reshape(*cos.shape[:-2], -1)
        return off_diag.mean(dim=-1), off_diag.max(dim=-1).values

    def forward(
            self,
            text_support: Tensor,
            fourier_cue: Tensor,
            confidence: Optional[Tensor] = None,
            class_ids: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None
    ) -> Tuple[Tensor, Dict[str, object]]:
        code = _flatten_code(fourier_cue)
        batch, positions, code_dim = code.shape
        if code_dim != self.code_dim:
            raise ValueError(f"expected code_dim={self.code_dim}, got {code_dim}")
        conf = _flatten_confidence(confidence, code).clamp(0.0, 1.0)
        base = self._expand_support(
            text_support.to(device=code.device, dtype=code.dtype),
            batch,
            positions)
        support_count = base.shape[-2]
        class_mask = self._class_mask(
            support_count, batch, code.device, class_ids, class_names)

        raw_delta = self.delta(code).unsqueeze(-2).expand_as(base)
        if class_mask.dim() == 1:
            raw_delta = raw_delta * class_mask.view(
                1, 1, support_count, 1).to(code.dtype)
        else:
            raw_delta = raw_delta * class_mask[:, None, :, None].to(code.dtype)

        alpha = torch.clamp(
            self.alpha_t, min=-self.alpha_t_max, max=self.alpha_t_max)
        gated_delta = raw_delta * alpha * conf[:, :, None, None]
        support_norm = base.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        max_delta = support_norm * self.max_delta_norm_ratio
        delta_norm = gated_delta.norm(dim=-1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(delta_norm), max_delta / (delta_norm + self.eps))
        applied_delta = gated_delta * scale
        conditioned = F.normalize(base + applied_delta, dim=-1)

        delta_ratio = applied_delta.norm(dim=-1) / support_norm.squeeze(
            -1).clamp_min(self.eps)
        cos_mean, cos_max = self._interclass_cos(conditioned)
        debug: Dict[str, object] = {
            "text_delta_norm": delta_ratio.detach(),
            "text_interclass_cos": cos_mean.detach(),
            "text_interclass_cos_max": cos_max.detach(),
            "alpha_t": alpha.detach(),
            "class_mask": class_mask[0].detach() if class_mask.dim() == 2 else class_mask.detach(),
            "final_logit_effect": False,
            "replaces_support_bank": False,
        }
        return conditioned, debug
