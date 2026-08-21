"""Background SV de-hub margin loss for dense classification logits."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_background_sv_dehub_loss(
    cls_score: torch.Tensor,
    labels: torch.Tensor,
    sv_class_index: int,
    margin: float = 0.05,
    bg_class_ind: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Penalize high SV logit on background (negative) dense locations.

    Args:
        cls_score: (N, C) flattened class logits.
        labels: (N,) assignment labels; background uses bg_class_ind.
        sv_class_index: index of small-vehicle in class dimension.
        margin: hinge margin between sv logit and best non-sv logit.
        bg_class_ind: background label id; default num_classes from max label.

    Returns:
        (loss, stats dict with n_neg, mean_sv_logit, etc.)
    """
    stats = dict(n_neg=0, mean_sv_logit=0.0, mean_violation=0.0)
    if sv_class_index < 0 or cls_score.numel() == 0:
        return cls_score.new_tensor(0.0), stats

    n_cls = cls_score.shape[1]
    if sv_class_index >= n_cls:
        return cls_score.new_tensor(0.0), stats

    if bg_class_ind is None:
        bg_class_ind = int(labels.max().item()) if labels.numel() else n_cls

    neg_mask = labels >= bg_class_ind
    if not neg_mask.any():
        return cls_score.new_tensor(0.0), stats

    neg_logits = cls_score[neg_mask]
    sv_logit = neg_logits[:, sv_class_index]
    other_idx = [i for i in range(n_cls) if i != sv_class_index]
    if not other_idx:
        return cls_score.new_tensor(0.0), stats
    max_non_sv = neg_logits[:, other_idx].amax(dim=1)
    violation = F.relu(sv_logit - max_non_sv + margin)
    loss = violation.mean()
    stats['n_neg'] = int(neg_mask.sum().item())
    stats['mean_sv_logit'] = float(sv_logit.detach().mean().item())
    stats['mean_violation'] = float(violation.detach().mean().item())
    return loss, stats
