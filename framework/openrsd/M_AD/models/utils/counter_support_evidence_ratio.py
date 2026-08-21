"""Prompt-conditioned counter-support evidence ratio for OpenRSD logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CounterSupportEvidenceRatio(nn.Module):
    """Subtract learned counter evidence from existing class logits.

    The adapter is class-agnostic: it transforms the runtime support shots with
    one shared low-rank residual, so the same parameters apply to unseen class
    vocabularies.  A zero strength leaves the original logits bit-for-bit
    unchanged while retaining a non-zero gradient for the strength parameter.
    """

    def __init__(
            self,
            embed_dims: int,
            rank: int = 8,
            enable: bool = False,
            init_strength: float = 0.0) -> None:
        super().__init__()
        if int(embed_dims) <= 0:
            raise ValueError("embed_dims must be positive")
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        self.embed_dims = int(embed_dims)
        self.rank = int(rank)
        self.enable = bool(enable)
        self.support_down = nn.Linear(self.embed_dims, self.rank, bias=False)
        self.support_up = nn.Linear(self.rank, self.embed_dims, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))
        self.strength = nn.Parameter(
            torch.tensor(float(init_strength), dtype=torch.float32))
        nn.init.normal_(self.support_down.weight, std=0.02)
        nn.init.normal_(self.support_up.weight, std=0.02)
        self.last_debug: dict[str, float | int | bool] = {
            "enable": self.enable,
            "rank": self.rank,
        }

    def forward(
            self,
            positive_logits: Tensor,
            query_embeds: Tensor,
            support_feats: Tensor,
            support_labels: Tensor,
            visual_fc: nn.Module) -> Tensor:
        if not self.enable:
            return positive_logits
        if positive_logits.ndim != 4 or query_embeds.ndim != 4:
            raise ValueError("positive_logits and query_embeds must be BCHW")
        if support_feats.ndim != 3 or support_labels.ndim != 2:
            raise ValueError("support_feats must be BMD and labels must be BM")

        batch, num_classes, height, width = positive_logits.shape
        if query_embeds.shape[0] != batch:
            raise ValueError("query batch does not match positive logits")
        if support_feats.shape[:2] != support_labels.shape:
            raise ValueError("support features and labels do not align")
        if support_feats.shape[0] != batch:
            raise ValueError("support batch does not match positive logits")

        query = query_embeds.permute(0, 2, 3, 1).reshape(
            batch, height * width, -1)
        query = F.normalize(visual_fc(query), dim=-1)
        support = visual_fc(support_feats)
        counter_support = support + self.support_up(self.support_down(support))
        counter_support = F.normalize(counter_support, dim=-1)

        shot_logits = torch.matmul(
            query, counter_support.transpose(-1, -2))
        shot_logits = shot_logits * self.log_scale.exp() + self.bias
        labels = support_labels.to(device=shot_logits.device, dtype=torch.long)

        class_logits = []
        valid_classes = []
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            valid = class_mask.any(dim=1)
            masked = shot_logits.masked_fill(
                ~class_mask[:, None, :], float("-inf"))
            maximum = masked.max(dim=-1).values
            maximum = torch.where(valid[:, None], maximum, maximum.new_zeros(()))
            class_logits.append(maximum)
            valid_classes.append(valid)

        counter_logits = torch.stack(class_logits, dim=-1)
        valid_mask = torch.stack(valid_classes, dim=-1)[:, None, :]
        counter_evidence = F.softplus(counter_logits) * valid_mask.to(
            dtype=counter_logits.dtype)
        counter_evidence = counter_evidence.reshape(
            batch, height, width, num_classes).permute(0, 3, 1, 2)
        effective_strength = self.strength.clamp_min(0).to(
            dtype=positive_logits.dtype)
        adjusted = positive_logits - effective_strength * counter_evidence.to(
                dtype=positive_logits.dtype)

        detached = counter_evidence.detach()
        self.last_debug = {
            "enable": True,
            "rank": self.rank,
            "num_classes": int(num_classes),
            "num_valid_classes": int(valid_mask.sum().detach().cpu().item()),
            "strength": float(effective_strength.detach().cpu().item()),
            "counter_mean": float(detached.mean().cpu().item()),
            "counter_max": float(detached.max().cpu().item()),
        }
        return adjusted
