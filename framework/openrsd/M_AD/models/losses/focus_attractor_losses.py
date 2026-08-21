from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _zero_like_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _broadcast_mask(mask: Tensor, logits: Tensor) -> Tensor:
    if mask.dim() == logits.dim() - 1:
        return mask.to(device=logits.device, dtype=torch.bool)
    if mask.dim() == logits.dim() and mask.shape[1] == 1:
        return mask[:, 0].to(device=logits.device, dtype=torch.bool)
    raise ValueError(
        f"mask shape {tuple(mask.shape)} is incompatible with logits {tuple(logits.shape)}")


def support_distill_loss(
        conditioned_support: Tensor,
        original_support: Tensor,
        weight: float = 1.0,
        margin: Optional[float] = None) -> Tensor:
    """Keep orientation-conditioned supports close to native support space."""
    if conditioned_support.numel() == 0 or original_support.numel() == 0:
        return _zero_like_loss(conditioned_support)
    original = original_support.to(
        device=conditioned_support.device,
        dtype=conditioned_support.dtype)
    if original.shape != conditioned_support.shape:
        while original.dim() < conditioned_support.dim():
            original = original.unsqueeze(1)
        original = original.expand_as(conditioned_support)

    diff = F.mse_loss(conditioned_support, original, reduction="none")
    loss = diff.mean(dim=-1)
    if margin is not None:
        loss = (loss - float(margin)).clamp_min(0.0)
    return loss.mean() * float(weight)


def anti_attractor_loss(
        logits: Tensor,
        sv_class_index: int,
        negative_mask: Optional[Tensor],
        margin: float = 0.0,
        weight: float = 1.0) -> Tensor:
    """Penalize small-vehicle activation on audited false-SV negatives."""
    if negative_mask is None or sv_class_index < 0:
        return _zero_like_loss(logits)
    if sv_class_index >= logits.shape[1]:
        return _zero_like_loss(logits)

    mask = _broadcast_mask(negative_mask, logits)
    if not torch.any(mask):
        return _zero_like_loss(logits)

    sv_logits = logits[:, sv_class_index]
    loss = F.softplus(sv_logits[mask] - float(margin))
    return loss.mean() * float(weight)


def preserve_loss(
        logits: Tensor,
        sv_class_index: int,
        positive_mask: Optional[Tensor],
        weight: float = 1.0) -> Tensor:
    """Preserve small-vehicle confidence on audited true/missing positives."""
    if positive_mask is None or sv_class_index < 0:
        return _zero_like_loss(logits)
    if sv_class_index >= logits.shape[1]:
        return _zero_like_loss(logits)

    mask = _broadcast_mask(positive_mask, logits)
    if not torch.any(mask):
        return _zero_like_loss(logits)

    targets = torch.full(
        (int(mask.sum().item()),),
        int(sv_class_index),
        dtype=torch.long,
        device=logits.device)
    point_logits = logits.permute(0, 2, 3, 1)[mask]
    return F.cross_entropy(point_logits, targets) * float(weight)


def orientation_consistency_loss(
        theta_a: Optional[Tensor],
        theta_b: Optional[Tensor],
        mask: Optional[Tensor] = None,
        weight: float = 1.0) -> Tensor:
    """Periodic consistency loss for two orientation fields in [0, pi)."""
    reference = theta_a if theta_a is not None else theta_b
    if theta_a is None or theta_b is None or reference is None:
        return torch.tensor(0.0)
    if theta_a.numel() == 0 or theta_b.numel() == 0:
        return _zero_like_loss(reference)

    diff = 1.0 - torch.cos(2.0 * (theta_a - theta_b))
    if mask is not None:
        mask = mask.to(device=diff.device, dtype=torch.bool)
        if not torch.any(mask):
            return _zero_like_loss(diff)
        diff = diff[mask]
    return diff.mean() * float(weight)


def migration_kl_loss(
        focus_logits: Tensor,
        baseline_logits: Tensor,
        mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        weight: float = 1.0) -> Tensor:
    """KL guardrail to prevent broad class-score migration."""
    if focus_logits.numel() == 0 or baseline_logits.numel() == 0:
        return _zero_like_loss(focus_logits)
    if focus_logits.shape == baseline_logits.shape and torch.equal(
            focus_logits, baseline_logits.to(
                device=focus_logits.device, dtype=focus_logits.dtype)):
        return _zero_like_loss(focus_logits)
    temp = max(float(temperature), 1e-6)
    focus = focus_logits / temp
    baseline = baseline_logits.to(
        device=focus_logits.device,
        dtype=focus_logits.dtype) / temp

    if mask is not None:
        keep = _broadcast_mask(mask, focus_logits)
        if not torch.any(keep):
            return _zero_like_loss(focus_logits)
        focus = focus.permute(0, 2, 3, 1)[keep]
        baseline = baseline.permute(0, 2, 3, 1)[keep]
        dim = -1
    else:
        dim = 1

    loss = F.kl_div(
        F.log_softmax(focus, dim=dim),
        F.softmax(baseline, dim=dim),
        reduction="batchmean")
    return loss * (temp * temp) * float(weight)
