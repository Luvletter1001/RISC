from typing import Optional, Tuple

import torch
from mmdet.models.layers.transformer.utils import MLP
from mmengine.model import ModuleList
from torch import Tensor, nn

from projects.GroundingDINO.groundingdino.grounding_dino_layers import (
    GroundingDinoTransformerDecoder,
    GroundingDinoTransformerDecoderLayer,
)

from .null_reservoir import ExplicitNullReservoir
from .semantic_capacity import (ContinuousDensityCapacity,
                                SemanticEvidenceFusion)


def apply_matching_query_interventions(
        native_query: Tensor,
        transported_query: Tensor,
        matching_query_count: int,
        fusion: Optional[SemanticEvidenceFusion],
        capacity: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply configured interventions to the matching-query suffix."""
    if native_query.shape != transported_query.shape:
        raise ValueError('native_query and transported_query must match')
    if matching_query_count <= 0 or \
            matching_query_count > native_query.shape[1]:
        raise ValueError('matching_query_count is outside the query range')
    if fusion is None:
        return transported_query, None
    if capacity is not None and capacity.shape != \
            native_query[:, -matching_query_count:].shape[:-1]:
        raise ValueError('capacity must describe the matching query suffix')

    matching_native = native_query[:, -matching_query_count:]
    matching_evidence = transported_query[:, -matching_query_count:]
    fused_matching, gate = fusion(
        matching_native, matching_evidence, capacity=capacity)
    prefix = transported_query[:, :-matching_query_count]
    if prefix.shape[1] == 0:
        return fused_matching, gate
    return torch.cat([prefix, fused_matching], dim=1), gate


def fuse_matching_query_suffix(
        native_query: Tensor, transported_query: Tensor, capacity: Tensor,
        matching_query_count: int,
        fusion: SemanticEvidenceFusion) -> Tuple[Tensor, Tensor]:
    """Fuse only matching queries and preserve a denoising prefix."""
    return apply_matching_query_interventions(
        native_query=native_query,
        transported_query=transported_query,
        matching_query_count=matching_query_count,
        fusion=fusion,
        capacity=capacity)


class OVCapFlowDecoderLayer(GroundingDinoTransformerDecoderLayer):
    """GroundingDINO layer with native semantic and density capacity state."""

    def __init__(self,
                 enable_semantic_fusion: bool = True,
                 enable_density_capacity: bool = True,
                 semantic_fusion_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        semantic_fusion_cfg = dict(semantic_fusion_cfg or {})
        self.enable_semantic_fusion = bool(enable_semantic_fusion)
        self.enable_density_capacity = bool(enable_density_capacity)
        self.semantic_fusion = (
            SemanticEvidenceFusion(
                embed_dims=self.embed_dims, **semantic_fusion_cfg)
            if self.enable_semantic_fusion else None)
        self.density_capacity = (
            ContinuousDensityCapacity(self.embed_dims)
            if self.enable_density_capacity else None)
        self.last_capacity = None
        self.last_predicted_count = None
        self.last_semantic_gate = None

    def forward(self,
                query: Tensor,
                value: Tensor = None,
                key_padding_mask: Tensor = None,
                native_query: Tensor = None,
                matching_query_count: int = None,
                **kwargs) -> Tensor:
        transported_query = super().forward(
            query=query,
            value=value,
            key_padding_mask=key_padding_mask,
            **kwargs)
        self.last_capacity = None
        self.last_predicted_count = None
        self.last_semantic_gate = None
        if native_query is None or matching_query_count is None:
            return transported_query

        capacity = None
        predicted_count = None
        if self.density_capacity is not None:
            matching_evidence = transported_query[:, -matching_query_count:]
            capacity, predicted_count = self.density_capacity(
                matching_evidence, value, memory_mask=key_padding_mask)
        fused_query, gate = apply_matching_query_interventions(
            native_query=native_query,
            transported_query=transported_query,
            capacity=capacity,
            matching_query_count=matching_query_count,
            fusion=self.semantic_fusion)
        self.last_capacity = capacity
        self.last_predicted_count = predicted_count
        self.last_semantic_gate = gate
        return fused_query


class OVCapFlowDecoder(GroundingDinoTransformerDecoder):
    """GroundingDINO decoder composed of OV-CapFlow decoder layers."""

    def __init__(self,
                 enable_null_reservoir=False,
                 null_reservoir_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.enable_null_reservoir = bool(enable_null_reservoir)
        null_reservoir_cfg = dict(null_reservoir_cfg or {})
        self.null_reservoir = (
            ExplicitNullReservoir(
                embed_dims=self.embed_dims, **null_reservoir_cfg)
            if self.enable_null_reservoir else None)
        self.last_null_logits = None

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            OVCapFlowDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError(
                'OVCapFlowDecoder does not use post normalization')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, *args, matching_query_count=None, **kwargs):
        inter_states, references = super().forward(
            *args, matching_query_count=matching_query_count, **kwargs)
        self.last_null_logits = None
        if self.null_reservoir is not None:
            if matching_query_count is None:
                raise ValueError('null reservoir requires matching query count')
            final_matching = inter_states[-1][:, -matching_query_count:]
            self.last_null_logits = self.null_reservoir(final_matching)
        return inter_states, references
