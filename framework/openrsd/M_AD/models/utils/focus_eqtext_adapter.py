"""Fourier-conditioned residual adapter for frozen text support."""

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


class FourierEquivariantTextAdapter(nn.Module):
    """Zero-init text residual conditioned by image-side Fourier phase.

    The adapter consumes already-mapped text support. It does not encode text
    and does not replace the support bank; it only adds a bounded residual in
    the detector support space.
    """

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
            alpha_t_max: float = 0.05,
            max_delta_norm_ratio: float = 0.05,
            eps: float = 1e-8) -> None:
        super().__init__()
        if float(alpha_t_max) > 0.05:
            raise ValueError("alpha_t_max must be <= 0.05")
        self.support_dim = int(support_dim)
        self.code_dim = int(code_dim)
        self.apply_to_class_ids = (
            {int(v) for v in apply_to_class_ids}
            if apply_to_class_ids is not None else None)
        self.class_names = list(class_names) if class_names is not None else None
        self.apply_to_classes = {str(v) for v in apply_to_classes}
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
        targets = {
            name.lower().replace("_", "-")
            for name in self.apply_to_classes
        }
        mask = torch.zeros(support_count, dtype=torch.bool, device=device)
        for idx, name in enumerate(names[:support_count]):
            if str(name).lower().replace("_", "-") in targets:
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

        name_mask = self._class_mask_from_names(
            max(int(ids[ids >= 0].max().item()) + 1 if torch.any(ids >= 0) else 1,
                len(class_names or self.class_names or []),
                support_count),
            device,
            class_names)
        safe = ids.long().clamp(min=0, max=name_mask.shape[0] - 1)
        valid = (ids >= 0) & (ids < name_mask.shape[0])
        return name_mask[safe] & valid

    def _interclass_cos(
            self,
            support: Tensor,
            labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if labels is not None and support.dim() == 4:
            labels = labels.to(device=support.device)
            if labels.dim() == 1:
                labels = labels[None, :].expand(support.shape[0], -1)
            if labels.shape[0] == 1 and support.shape[0] > 1:
                labels = labels.expand(support.shape[0], -1)
            if labels.shape[:2] == (support.shape[0], support.shape[2]):
                means = []
                maxes = []
                for batch_idx in range(support.shape[0]):
                    valid_labels = labels[batch_idx]
                    classes = torch.unique(valid_labels[valid_labels >= 0])
                    if classes.numel() <= 1:
                        zero = support.new_zeros(support.shape[1])
                        means.append(zero)
                        maxes.append(zero)
                        continue
                    protos = []
                    for class_id in classes:
                        mask = valid_labels == class_id
                        protos.append(support[batch_idx, :, mask].mean(dim=-2))
                    proto = F.normalize(torch.stack(protos, dim=-2), dim=-1)
                    cos = proto @ proto.transpose(-1, -2)
                    mask = ~torch.eye(
                        cos.shape[-1], dtype=torch.bool, device=cos.device)
                    off_diag = cos[..., mask].reshape(cos.shape[0], -1)
                    means.append(off_diag.mean(dim=-1))
                    maxes.append(off_diag.max(dim=-1).values)
                return torch.stack(means, dim=0), torch.stack(maxes, dim=0)

        support = F.normalize(support, dim=-1)
        cos = support @ support.transpose(-1, -2)
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
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
        labels = class_ids if class_ids is not None else support_labels
        class_mask = self._class_mask(
            support_count,
            batch,
            code.device,
            labels,
            class_names)

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
        anchor_loss_raw = F.mse_loss(
            conditioned,
            F.normalize(base, dim=-1),
            reduction="mean")
        delta_ratio = applied_delta.norm(dim=-1) / support_norm.squeeze(
            -1).clamp_min(self.eps)
        cos_mean, cos_max = self._interclass_cos(conditioned, labels=labels)
        debug = {
            "text_delta_norm": torch.nan_to_num(applied_delta.norm(dim=-1)),
            "text_delta_norm_ratio": torch.nan_to_num(delta_ratio),
            "text_interclass_cos": torch.nan_to_num(cos_mean.detach()),
            "text_interclass_cos_max": torch.nan_to_num(cos_max.detach()),
            "class_mask": (
                class_mask[0] if class_mask.dim() == 2 else class_mask
            ).detach().cpu(),
            "alpha_t": alpha.detach(),
            "text_anchor_loss_raw": anchor_loss_raw,
        }
        return conditioned, debug
