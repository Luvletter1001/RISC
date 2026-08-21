import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _masked_loss(loss_module: nn.Module, logits: Tensor, labels: Tensor,
                 weights: Tensor, mask: Tensor) -> Tensor:
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    selected_weights = weights[mask]
    return loss_module(
        selected_logits,
        selected_labels,
        selected_weights,
        avg_factor=max(int(mask.sum().item()), 1))


def balanced_group_classification_loss(
        loss_module: nn.Module,
        cls_scores: Tensor,
        labels: Tensor,
        query_weights: Tensor,
        valid_token_mask: Tensor,
        matched_query_mask: Tensor,
        matched_weight: float,
        unmatched_weight: float) -> Tuple[Tensor, Dict[str, int]]:
    """Average classification over matched and unmatched query groups."""
    if cls_scores.shape != labels.shape or cls_scores.shape != \
            valid_token_mask.shape:
        raise ValueError('scores, labels, and token mask must match')
    if query_weights.shape != cls_scores.shape[:2]:
        raise ValueError('query_weights must have shape [batch, queries]')
    if matched_query_mask.shape != cls_scores.shape[:2]:
        raise ValueError('matched mask must have shape [batch, queries]')

    token_weights = query_weights.unsqueeze(-1).expand_as(cls_scores)
    matched_tokens = valid_token_mask & matched_query_mask.unsqueeze(-1)
    unmatched_tokens = valid_token_mask & ~matched_query_mask.unsqueeze(-1)
    matched_loss = _masked_loss(
        loss_module, cls_scores, labels, token_weights, matched_tokens)
    unmatched_loss = _masked_loss(
        loss_module, cls_scores, labels, token_weights, unmatched_tokens)
    total = matched_weight * matched_loss + unmatched_weight * unmatched_loss
    stats = {
        'matched_queries': int(matched_query_mask.sum().item()),
        'unmatched_queries': int((~matched_query_mask).sum().item()),
        'matched_tokens': int(matched_tokens.sum().item()),
        'unmatched_tokens': int(unmatched_tokens.sum().item()),
    }
    return total, stats


def grounding_logits_to_class_log_scores(
        token_logits: Tensor, positive_map: dict) -> Tensor:
    """Aggregate token probabilities per class in the log domain."""
    if token_logits.ndim != 2:
        raise ValueError('token_logits must be [queries, tokens]')
    class_logs = []
    log_token_prob = F.logsigmoid(token_logits.float())
    for label_id in sorted(positive_map):
        token_ids = torch.as_tensor(
            positive_map[label_id], dtype=torch.long,
            device=token_logits.device)
        if token_ids.numel() == 0:
            raise ValueError('each class needs at least one positive token')
        selected = log_token_prob.index_select(-1, token_ids)
        class_logs.append(
            torch.logsumexp(selected, dim=-1) - math.log(token_ids.numel()))
    return torch.stack(class_logs, dim=-1)


def select_from_class_log_scores(class_log_scores: Tensor):
    """Select one class within every query row."""
    if class_log_scores.ndim != 2:
        raise ValueError('class_log_scores must be [queries, classes]')
    return class_log_scores.max(dim=-1)


def calibrate_selected_log_scores(
        selected_log_scores: Tensor,
        null_logits: Tensor = None,
        capacity: Tensor = None,
        temperature: float = 1.0,
        power: float = 1.0,
        log_residual: Tensor = None) -> Tensor:
    """Apply per-query calibration while retaining finite float32 scores."""
    if temperature <= 0 or power <= 0:
        raise ValueError('temperature and power must be positive')
    calibrated = selected_log_scores.float() * (power / temperature)
    if null_logits is not None:
        calibrated = calibrated + F.logsigmoid(-null_logits.float())
    if capacity is not None:
        calibrated = calibrated + capacity.float().clamp(min=1e-8).log()
    if log_residual is not None:
        if log_residual.shape != selected_log_scores.shape:
            raise ValueError('log_residual must match selected_log_scores')
        calibrated = calibrated + log_residual.float()
    return calibrated.clamp(min=-80.0, max=0.0).exp()
