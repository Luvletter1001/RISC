from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _flatten_code(code: Tensor) -> Tuple[Tensor, Optional[Tuple[int, int]]]:
    if code.dim() == 4:
        batch, height, width, dim = code.shape
        return code.reshape(batch, height * width, dim), (height, width)
    if code.dim() == 3:
        return code, None
    raise ValueError(f"fourier_code must be [B,N,D] or [B,H,W,D], got {tuple(code.shape)}")


def _flatten_confidence(confidence: Tensor) -> Tensor:
    if confidence.dim() == 3:
        return confidence.reshape(confidence.shape[0], -1)
    if confidence.dim() == 2:
        return confidence
    raise ValueError(f"confidence must be [B,N] or [B,H,W], got {tuple(confidence.shape)}")


class FourierSupportResidualAdapter(nn.Module):
    """Zero-initialized residual modulation for native OpenRSD support."""

    def __init__(
            self,
            support_dim: int,
            code_dim: int,
            class_names: Optional[Sequence[str]] = None,
            apply_to_classes: Iterable[str] = ("small-vehicle",),
            allow_all_classes: bool = False,
            low_rank: int = 16,
            alpha_init: float = 0.0,
            alpha_max: float = 0.10,
            max_delta_norm_ratio: float = 0.05,
            normalize_after_residual: bool = True,
            eps: float = 1e-8) -> None:
        super().__init__()
        self.support_dim = int(support_dim)
        self.code_dim = int(code_dim)
        self.class_names = list(class_names) if class_names is not None else None
        self.apply_to_classes = {str(v) for v in apply_to_classes}
        self.allow_all_classes = bool(allow_all_classes)
        self.alpha_max = float(alpha_max)
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.normalize_after_residual = bool(normalize_after_residual)
        self.eps = float(eps)

        hidden_dim = max(1, int(low_rank))
        self.delta = nn.Sequential(
            nn.Linear(self.code_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.support_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _target_class_mask(
            self,
            num_classes: int,
            device: torch.device,
            class_names: Optional[Sequence[str]] = None) -> Tensor:
        if self.allow_all_classes:
            return torch.ones(num_classes, dtype=torch.bool, device=device)
        mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
        names = list(class_names) if class_names is not None else self.class_names
        if names is None:
            return mask
        normalized_targets = {
            name.lower().replace("_", "-") for name in self.apply_to_classes
        }
        for idx, name in enumerate(names[:num_classes]):
            normalized = str(name).lower().replace("_", "-")
            if normalized in normalized_targets:
                mask[idx] = True
        return mask

    def _support_mask(
            self,
            support_count: int,
            device: torch.device,
            batch: int,
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None) -> Tensor:
        if support_labels is None:
            return self._target_class_mask(
                support_count, device, class_names=class_names)

        labels = support_labels.to(device=device)
        if labels.dim() == 1:
            labels = labels[None, :].expand(batch, labels.shape[0])
        if labels.dim() != 2:
            raise ValueError(
                f"support_labels must be [M] or [B,M], got {tuple(labels.shape)}")
        if labels.shape[0] != batch:
            if labels.shape[0] == 1:
                labels = labels.expand(batch, labels.shape[1])
            else:
                raise ValueError(
                    f"support_labels batch={labels.shape[0]} does not match code batch={batch}")
        if labels.shape[1] != support_count:
            raise ValueError(
                f"support_labels count={labels.shape[1]} does not match support_count={support_count}")

        max_label = int(labels[labels >= 0].max().item()) + 1 if torch.any(labels >= 0) else 0
        num_classes = max(max_label, len(class_names or self.class_names or []), 1)
        target_class_mask = self._target_class_mask(
            num_classes, device, class_names=class_names)
        valid = (labels >= 0) & (labels < num_classes)
        safe_labels = labels.long().clamp(min=0, max=num_classes - 1)
        return target_class_mask[safe_labels] & valid

    def _expand_support(self, support_feats: Tensor, batch: int,
                        positions: int) -> Tensor:
        if support_feats.dim() == 2:
            return support_feats[None, None].expand(
                batch, positions, support_feats.shape[0], support_feats.shape[1])
        if support_feats.dim() == 3:
            return support_feats[:, None].expand(
                support_feats.shape[0], positions,
                support_feats.shape[1], support_feats.shape[2])
        raise ValueError(f"support_feats must be [C,D] or [B,C,D], got {tuple(support_feats.shape)}")

    def _inter_class_cosine(self, support: Tensor) -> Tensor:
        support = F.normalize(support, dim=-1)
        cos = support @ support.transpose(-1, -2)
        num_classes = cos.shape[-1]
        if num_classes <= 1:
            return cos.new_zeros(cos.shape[:-2])
        mask = ~torch.eye(num_classes, dtype=torch.bool, device=cos.device)
        return cos[..., mask].reshape(*cos.shape[:-2], -1).mean(dim=-1)

    def forward(
            self,
            support_feats: Tensor,
            fourier_code: Tensor,
            confidence: Tensor,
            support_labels: Optional[Tensor] = None,
            class_names: Optional[Sequence[str]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        code, _ = _flatten_code(fourier_code)
        confidence_flat = _flatten_confidence(confidence).to(code.dtype)
        batch, positions, code_dim = code.shape
        if code_dim != self.code_dim:
            raise ValueError(f"expected code_dim={self.code_dim}, got {code_dim}")

        base = self._expand_support(support_feats.to(code.device, code.dtype),
                                    batch, positions)
        support_count = base.shape[-2]
        class_mask = self._support_mask(
            support_count,
            code.device,
            batch,
            support_labels=support_labels,
            class_names=class_names)

        raw_delta = self.delta(code).unsqueeze(-2).expand_as(base)
        if class_mask.dim() == 1:
            raw_delta = raw_delta * class_mask.view(1, 1, support_count, 1).to(code.dtype)
        else:
            raw_delta = raw_delta * class_mask[:, None, :, None].to(code.dtype)

        alpha = torch.clamp(
            self.alpha, min=-self.alpha_max, max=self.alpha_max)
        gated_delta = raw_delta * alpha * confidence_flat.clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)

        support_norm = base.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        delta_norm = gated_delta.norm(dim=-1, keepdim=True)
        max_delta = support_norm * self.max_delta_norm_ratio
        scale = torch.minimum(torch.ones_like(delta_norm), max_delta / (delta_norm + self.eps))
        applied_delta = gated_delta * scale

        delta_norm_ratio = (applied_delta.norm(dim=-1) /
                            support_norm.squeeze(-1).clamp_min(self.eps))

        conditioned = base + applied_delta
        if (self.normalize_after_residual
                and applied_delta.abs().max().detach() != 0):
            conditioned = F.normalize(conditioned, dim=-1)

        debug = {
            "delta_norm": applied_delta.norm(dim=-1),
            "delta_norm_ratio": torch.nan_to_num(delta_norm_ratio, nan=0.0),
            "class_mask": (class_mask[0] if class_mask.dim() == 2 else class_mask).detach().cpu(),
            "inter_class_cos_before": self._inter_class_cosine(base).detach(),
            "inter_class_cos_after": self._inter_class_cosine(conditioned).detach(),
        }
        return conditioned, debug
