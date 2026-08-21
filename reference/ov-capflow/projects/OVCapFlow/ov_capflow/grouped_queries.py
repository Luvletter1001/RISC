"""Pure tensor helpers for training-only grouped one-to-one queries."""

from typing import Callable, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor


def _validate_groups(groups: int) -> None:
    if not isinstance(groups, int) or groups < 1:
        raise ValueError('groups must be a positive integer')


def repeat_matching_queries(query: Tensor, references: Tensor,
                            groups: int) -> Tuple[Tensor, Tensor]:
    _validate_groups(groups)
    if query.ndim != 3 or references.ndim != 3:
        raise ValueError('query and references must be batched tensors')
    if query.shape[:2] != references.shape[:2]:
        raise ValueError('query and references must share batch/query axes')
    if groups == 1:
        return query, references
    return (torch.cat([query] * groups, dim=1),
            torch.cat([references] * groups, dim=1))


def expand_dn_attention_mask(mask: Tensor, num_dn: int,
                             queries_per_group: int,
                             groups: int) -> Tensor:
    _validate_groups(groups)
    if num_dn < 0 or queries_per_group < 1:
        raise ValueError('invalid DN or matching query count')
    expected = num_dn + queries_per_group
    if mask.ndim != 2 or mask.shape != (expected, expected):
        raise ValueError('DN attention mask has an unexpected shape')
    if groups == 1:
        return mask

    total = num_dn + queries_per_group * groups
    expanded = torch.ones(
        total, total, dtype=mask.dtype, device=mask.device)
    expanded[:num_dn, :num_dn] = mask[:num_dn, :num_dn]
    matching_block = mask[num_dn:, num_dn:]
    for group in range(groups):
        start = num_dn + group * queries_per_group
        end = start + queries_per_group
        expanded[:num_dn, start:end] = mask[:num_dn, num_dn:]
        expanded[start:end, :num_dn] = mask[num_dn:, :num_dn]
        expanded[start:end, start:end] = matching_block
    return expanded


def split_matching_groups(tensor: Tensor, queries_per_group: int,
                          groups: int) -> Sequence[Tensor]:
    _validate_groups(groups)
    if tensor.ndim < 2 or tensor.shape[-2] != queries_per_group * groups:
        raise ValueError('matching query axis does not match group geometry')
    return tensor.split(queries_per_group, dim=-2)


def average_matching_loss_dicts(
        losses: Sequence[Mapping[str, Tensor]]) -> Dict[str, Tensor]:
    if not losses:
        raise ValueError('at least one matching loss dictionary is required')
    keys = tuple(losses[0])
    if any(tuple(loss) != keys for loss in losses[1:]):
        raise ValueError('matching loss dictionaries need identical keys')
    return {
        key: torch.stack([loss[key] for loss in losses]).mean()
        for key in keys
    }


def grouped_matching_losses(
        cls_scores: Tensor,
        box_preds: Tensor,
        queries_per_group: int,
        groups: int,
        loss_fn: Callable[[Tensor, Tensor], Mapping[str, Tensor]],
        ) -> Dict[str, Tensor]:
    cls_groups = split_matching_groups(
        cls_scores, queries_per_group, groups)
    box_groups = split_matching_groups(
        box_preds, queries_per_group, groups)
    return average_matching_loss_dicts([
        loss_fn(group_cls, group_box)
        for group_cls, group_box in zip(cls_groups, box_groups)
    ])
