import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ExistenceResidual(nn.Linear):

    def __init__(self, embed_dims: int = 256):
        super().__init__(embed_dims, 1, bias=True)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()


def centered_existence_log_residual(logits: Tensor) -> Tensor:
    logits = logits.float()
    return F.logsigmoid(logits) - F.logsigmoid(torch.zeros_like(logits))


def grouped_existence_bce_loss(logits: Tensor,
                               matched_mask: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError('logits must have shape [batch, groups, queries]')
    if matched_mask.shape != logits.shape:
        raise ValueError('matched_mask must match logits')
    if matched_mask.dtype != torch.bool:
        raise TypeError('matched_mask must be boolean')
    if 0 in logits.shape:
        raise ValueError('logits dimensions must be non-empty')
    logits = logits.float()
    matched = matched_mask.to(dtype=logits.dtype)
    unmatched = 1.0 - matched
    matched_count = matched.sum(dim=-1)
    unmatched_count = unmatched.sum(dim=-1)
    signed_logits = torch.where(matched_mask, -logits, logits)
    element_loss = F.softplus(signed_logits)
    zero_loss = torch.zeros_like(element_loss)
    matched_mean = torch.where(
        matched_mask, element_loss, zero_loss).sum(
            dim=-1) / matched_count.clamp_min(1)
    unmatched_mean = torch.where(
        matched_mask, zero_loss, element_loss).sum(
            dim=-1) / unmatched_count.clamp_min(1)
    has_matched = matched_count > 0
    has_unmatched = unmatched_count > 0
    active_count = (
        has_matched.to(dtype=logits.dtype) +
        has_unmatched.to(dtype=logits.dtype)).clamp_min(1)
    group_loss = (
        torch.where(has_matched, matched_mean, torch.zeros_like(matched_mean)) +
        torch.where(has_unmatched, unmatched_mean,
                    torch.zeros_like(unmatched_mean))
    ) / active_count
    return group_loss.mean()
