from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ExplicitNullReservoir(nn.Module):
    """Continuous per-query probability of the explicit null state."""

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        if embed_dims <= 0:
            raise ValueError('embed_dims must be positive')
        self.null_head = nn.Linear(embed_dims, 1)
        nn.init.xavier_uniform_(self.null_head.weight)
        nn.init.zeros_(self.null_head.bias)

    def forward(self, query: Tensor) -> Tensor:
        if query.ndim != 3:
            raise ValueError('query must be [batch, queries, channels]')
        return self.null_head(query).squeeze(-1)


def _group_mean(values: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum()
    if count.item() == 0:
        return values.sum() * 0.0
    return values.masked_select(mask).mean()


def null_reservoir_losses(
        null_logits: Tensor,
        matched_mask: Tensor,
        target_count: Tensor,
        gate_strength: Optional[Tensor] = None,
        matched_weight: float = 1.0,
        unmatched_weight: float = 1.0,
        mass_weight: float = 0.1,
        gate_order_weight: float = 0.1,
        gate_margin: float = 0.0) -> Dict[str, Tensor]:
    """Compute null calibration losses without removing query rows."""
    if null_logits.shape != matched_mask.shape:
        raise ValueError('null logits and matched mask must match')
    target_null = (~matched_mask).to(null_logits.dtype)
    element_loss = F.binary_cross_entropy_with_logits(
        null_logits, target_null, reduction='none')
    matched_loss = _group_mean(element_loss, matched_mask)
    unmatched_loss = _group_mean(element_loss, ~matched_mask)
    non_null_mass = torch.sigmoid(-null_logits).sum(dim=-1)
    mass_loss = F.smooth_l1_loss(
        non_null_mass, target_count.to(null_logits.dtype))

    gate_order = null_logits.sum() * 0.0
    if gate_strength is not None and matched_mask.any() and \
            (~matched_mask).any():
        matched_gate = _group_mean(gate_strength, matched_mask)
        unmatched_gate = _group_mean(gate_strength, ~matched_mask)
        gate_order = F.relu(unmatched_gate - matched_gate + gate_margin)

    return {
        'loss_null_matched': matched_weight * matched_loss,
        'loss_null_unmatched': unmatched_weight * unmatched_loss,
        'loss_null_mass': mass_weight * mass_loss,
        'loss_gate_order': gate_order_weight * gate_order,
    }
