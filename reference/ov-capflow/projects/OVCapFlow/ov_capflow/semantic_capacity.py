from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SemanticEvidenceFusion(nn.Module):
    """Apply a gated native-semantic residual to the parent query."""

    def __init__(self,
                 embed_dims: int,
                 adapter_init: str = 'identity') -> None:
        super().__init__()
        if embed_dims <= 0:
            raise ValueError('embed_dims must be positive')
        if adapter_init not in {'identity', 'orthogonal'}:
            raise ValueError("adapter_init must be 'identity' or 'orthogonal'")

        self.embed_dims = embed_dims
        self.adapter_init = adapter_init
        self.evidence_adapter = nn.Linear(embed_dims, embed_dims, bias=False)
        self.gate = nn.Linear(embed_dims * 2, embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.adapter_init == 'identity':
            nn.init.eye_(self.evidence_adapter.weight)
        else:
            nn.init.orthogonal_(self.evidence_adapter.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self,
                native_query: Tensor,
                transported_evidence: Tensor,
                capacity: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if native_query.shape != transported_evidence.shape:
            raise ValueError(
                'native_query and transported_evidence must match')
        if native_query.shape[-1] != self.embed_dims:
            raise ValueError(
                'query channel dimension does not match embed_dims')

        native_residual = native_query - transported_evidence
        projected_residual = self.evidence_adapter(native_residual)
        gate = torch.tanh(
            self.gate(torch.cat([native_query, transported_evidence], dim=-1)))
        if capacity is not None:
            if capacity.shape != native_query.shape[:-1]:
                raise ValueError('capacity must have one value per query')
            gate = gate * capacity.unsqueeze(-1)
        fused_query = transported_evidence + gate * projected_residual
        return fused_query, gate


class ContinuousDensityCapacity(nn.Module):
    """Predict continuous per-query capacity and global scene count."""

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        if embed_dims <= 0:
            raise ValueError('embed_dims must be positive')
        self.embed_dims = embed_dims
        self.capacity_head = nn.Linear(embed_dims, 1)
        self.memory_norm = nn.LayerNorm(embed_dims)
        self.density_head = nn.Linear(embed_dims, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.capacity_head.weight)
        nn.init.zeros_(self.capacity_head.bias)
        nn.init.xavier_uniform_(self.density_head.weight)
        nn.init.zeros_(self.density_head.bias)

    def forward(self,
                query: Tensor,
                memory: Tensor,
                memory_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError(
                'query and memory must be [batch, tokens, channels]')
        if query.shape[0] != memory.shape[0]:
            raise ValueError('query and memory batch sizes must match')
        if query.shape[-1] != self.embed_dims or \
                memory.shape[-1] != self.embed_dims:
            raise ValueError('channel dimension does not match embed_dims')

        capacity = self.capacity_head(query).squeeze(-1).sigmoid()
        if memory_mask is None:
            pooled_memory = memory.mean(dim=1)
        else:
            if memory_mask.shape != memory.shape[:2]:
                raise ValueError('memory_mask must match memory token shape')
            valid = (~memory_mask).to(memory.dtype).unsqueeze(-1)
            pooled_memory = (memory * valid).sum(dim=1)
            pooled_memory = pooled_memory / valid.sum(dim=1).clamp(min=1)

        max_count = query.shape[1]
        predicted_count = max_count * self.density_head(
            self.memory_norm(pooled_memory)).squeeze(-1).sigmoid()
        return capacity, predicted_count


def density_capacity_losses(capacity: Tensor, predicted_count: Tensor,
                            target_count: Tensor) -> Dict[str, Tensor]:
    """Continuous mass and global count supervision without query pruning."""
    if capacity.ndim != 2:
        raise ValueError('capacity must be [batch, queries]')
    if predicted_count.shape != target_count.shape:
        raise ValueError('predicted_count and target_count shapes must match')
    if capacity.shape[0] != target_count.shape[0]:
        raise ValueError('capacity and target_count batch sizes must match')

    target_count = target_count.to(
        device=capacity.device, dtype=capacity.dtype)
    return {
        'loss_capacity_mass':
        F.smooth_l1_loss(capacity.sum(dim=-1), target_count),
        'loss_density_count': F.smooth_l1_loss(predicted_count, target_count),
    }
