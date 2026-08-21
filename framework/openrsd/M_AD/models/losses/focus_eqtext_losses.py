"""Auxiliary losses for the FOCUS-EQText branch."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _zero_like(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _mask_values(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return values
    keep = mask.to(device=values.device, dtype=torch.bool)
    if keep.shape != values.shape[:keep.dim()]:
        keep = keep.reshape(values.shape[:keep.dim()])
    if not torch.any(keep):
        return values.reshape(-1)[:0]
    return values[keep]


def loss_focus_text_anchor(
        text_support_eq: Tensor,
        text_support_base: Tensor,
        weight: float = 1.0) -> Tensor:
    """Keep text residuals close to frozen mapped text support."""
    if text_support_eq.numel() == 0 or text_support_base.numel() == 0:
        return _zero_like(text_support_eq)
    base = text_support_base.to(
        device=text_support_eq.device, dtype=text_support_eq.dtype)
    while base.dim() < text_support_eq.dim():
        base = base.unsqueeze(1)
    base = base.expand_as(text_support_eq)
    if torch.equal(text_support_eq, base):
        return _zero_like(text_support_eq)
    return F.mse_loss(text_support_eq, base) * float(weight)


def loss_focus_eqtext_consistency(
        logits_a: Tensor,
        logits_b: Tensor,
        mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        weight: float = 1.0) -> Tensor:
    """Constrain text-conditioned logits to stay rotation-consistent."""
    if logits_a.numel() == 0 or logits_b.numel() == 0:
        return _zero_like(logits_a)
    other = logits_b.to(device=logits_a.device, dtype=logits_a.dtype)
    if logits_a.shape != other.shape:
        raise ValueError(
            f"logit shapes must match, got {tuple(logits_a.shape)} and "
            f"{tuple(other.shape)}")
    if torch.equal(logits_a, other):
        return _zero_like(logits_a)
    temp = max(float(temperature), 1e-6)
    a = logits_a / temp
    b = other / temp
    if mask is not None:
        keep = mask.to(device=logits_a.device, dtype=torch.bool)
        a = a[keep]
        b = b[keep]
        if a.numel() == 0:
            return _zero_like(logits_a)
    loss_ab = F.kl_div(
        F.log_softmax(a, dim=-1),
        F.softmax(b.detach(), dim=-1),
        reduction="batchmean")
    loss_ba = F.kl_div(
        F.log_softmax(b, dim=-1),
        F.softmax(a.detach(), dim=-1),
        reduction="batchmean")
    return 0.5 * (loss_ab + loss_ba) * (temp * temp) * float(weight)


def loss_focus_text_negative_margin(
        sv_text_logits: Tensor,
        negative_text_logits: Tensor,
        mask: Optional[Tensor] = None,
        margin: float = 0.1,
        weight: float = 1.0) -> Tensor:
    """Push corrected false-SV regions below auxiliary negative text prompts."""
    if sv_text_logits.numel() == 0 or negative_text_logits.numel() == 0:
        return _zero_like(sv_text_logits)
    neg = negative_text_logits.to(
        device=sv_text_logits.device, dtype=sv_text_logits.dtype)
    neg_max = neg.max(dim=-1).values
    sv = sv_text_logits
    if sv.shape != neg_max.shape:
        neg_max = neg_max.reshape_as(sv)
    violation = sv - neg_max + float(margin)
    violation = _mask_values(violation, mask)
    if violation.numel() == 0:
        return _zero_like(sv_text_logits)
    return violation.clamp_min(0.0).mean() * float(weight)


def loss_focus_text_proto_separation(
        text_prototypes: Tensor,
        margin: float = 0.9,
        weight: float = 1.0) -> Tensor:
    """Prevent text prototypes from collapsing into the same direction."""
    if text_prototypes.numel() == 0 or text_prototypes.shape[-2] <= 1:
        return _zero_like(text_prototypes)
    proto = F.normalize(text_prototypes, dim=-1)
    cos = proto @ proto.transpose(-1, -2)
    count = cos.shape[-1]
    mask = ~torch.eye(count, dtype=torch.bool, device=cos.device)
    off_diag = cos[..., mask].reshape(*cos.shape[:-2], -1)
    violation = (off_diag - float(margin)).clamp_min(0.0)
    if violation.numel() == 0:
        return _zero_like(text_prototypes)
    return violation.mean() * float(weight)
