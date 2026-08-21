from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from mmdet.models.detectors.deformable_detr import DeformableDETR
from mmdet.models.detectors.dino import DINO
from mmdet.models.layers import (DeformableDetrTransformerEncoder,
                                 SinePositionalEncoding)
from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmrotate.registry import MODELS as MMROTATE_MODELS
from torch import Tensor, nn
from torch.nn.init import normal_

from M_AD.models.layers.transformer.deformable_detr_layers import (
    RotatedMultiScaleDeformableAttention,
)
from M_AD.models.layers.transformer.dinor_layersv2 import (
    CdnRQueryGenerator,
    DinoRTransformerDecoder,
)


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15SupportConditionedOrientedDINO(DINO):
    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        if dn_cfg is None:
            raise ValueError('P15SupportConditionedOrientedDINO requires dn_cfg')
        rotated_dn_cfg = dict(dn_cfg)
        horizontal_dn_cfg = dict(rotated_dn_cfg)
        horizontal_dn_cfg.pop('angle_noise_scale', None)
        super().__init__(*args, dn_cfg=horizontal_dn_cfg, **kwargs)

        rotated_dn_cfg['num_classes'] = self.bbox_head.num_classes
        rotated_dn_cfg['embed_dims'] = self.embed_dims
        rotated_dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnRQueryGenerator(**rotated_dn_cfg)

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoRTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for param in coder.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        for module in self.modules():
            if isinstance(module, RotatedMultiScaleDeformableAttention):
                module.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor,
                    batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_objectness = None
        if hasattr(self.bbox_head, 'obj_branches'):
            enc_outputs_objectness = self.bbox_head.obj_branches[
                self.decoder.num_layers](output_memory)
        enc_outputs_quality = None
        if hasattr(self.bbox_head, 'quality_branches'):
            enc_outputs_quality = self.bbox_head.quality_branches[
                self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        enc_outputs_angle = self.bbox_head.angle_branches[
            self.decoder.num_layers](output_memory)
        if hasattr(self.bbox_head, 'select_query_indices'):
            if enc_outputs_quality is not None:
                topk_indices = self.bbox_head.select_query_indices(
                    enc_outputs_class,
                    enc_outputs_objectness,
                    enc_outputs_quality,
                    num_queries=self.num_queries)
            else:
                topk_indices = self.bbox_head.select_query_indices(
                    enc_outputs_class,
                    enc_outputs_objectness,
                    num_queries=self.num_queries)
        elif hasattr(self.bbox_head, 'select_topk_scores'):
            if enc_outputs_quality is not None:
                topk_metric = self.bbox_head.select_topk_scores(
                    enc_outputs_class, enc_outputs_objectness,
                    enc_outputs_quality)
            else:
                topk_metric = self.bbox_head.select_topk_scores(
                    enc_outputs_class, enc_outputs_objectness)
            topk_indices = torch.topk(topk_metric, k=self.num_queries, dim=1)[1]
        else:
            topk_metric = enc_outputs_class.max(-1)[0]
            topk_indices = torch.topk(topk_metric, k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_objectness = None
        if enc_outputs_objectness is not None:
            topk_objectness = torch.gather(
                enc_outputs_objectness, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 1))
        topk_quality = None
        if enc_outputs_quality is not None:
            topk_quality = torch.gather(
                enc_outputs_quality, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 1))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_angles = torch.gather(
            enc_outputs_angle, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 1))

        batch_size = memory.shape[0]
        query = self.query_embedding.weight[None, :, :].repeat(
            batch_size, 1, 1)
        query = self.bbox_head.condition_matching_queries(
            query, topk_score, topk_objectness)
        if self.training:
            dn_label_query, dn_bbox_query, dn_angle_query, dn_mask, dn_meta = (
                self.dn_query_generator(batch_data_samples))
            if hasattr(self.bbox_head, 'condition_denoising_queries'):
                dn_label_query = self.bbox_head.condition_denoising_queries(
                    dn_label_query, batch_data_samples, dn_meta)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1)
            reference_angles = torch.cat([dn_angle_query, topk_angles], dim=1)
        else:
            reference_points = topk_coords_unact
            reference_angles = topk_angles
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()
        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            reference_angles=reference_angles,
            dn_mask=dn_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords_unact.sigmoid(),
            enc_outputs_angle=topk_angles,
            dn_meta=dn_meta) if self.training else dict()
        if self.training and topk_objectness is not None:
            head_inputs_dict['enc_outputs_objectness'] = topk_objectness
        if self.training and topk_quality is not None:
            head_inputs_dict['enc_outputs_quality'] = topk_quality
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        reference_angles: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        inter_states, references, inter_angles = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            reference_angles=reference_angles,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            angle_branches=self.bbox_head.angle_branches,
            **kwargs)
        if query.shape[1] == self.num_queries:
            inter_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0)
        return dict(
            hidden_states=inter_states,
            references=list(references),
            reference_angles=list(inter_angles))


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15EncoderBypassSupportConditionedOrientedDINO(
        P15SupportConditionedOrientedDINO):
    """P15N-B strict-E2E variant that bypasses the DINO encoder compute.

    The neck features are still flattened by ``pre_transformer`` and the
    existing two-stage proposal, support-conditioned query initializer, rotated
    decoder, and fixed-set posterior output remain unchanged. This isolates
    whether the deformable encoder is the runtime bottleneck without adding a
    dense detection head, NMS, or score-threshold fallback.
    """

    def __init__(self,
                 *args,
                 bypass_add_pos_to_memory: bool = False,
                 **kwargs) -> None:
        self.bypass_add_pos_to_memory = bool(bypass_add_pos_to_memory)
        super().__init__(*args, **kwargs)

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        del level_start_index, valid_ratios
        if getattr(self, 'bypass_add_pos_to_memory', False):
            memory = feat + feat_pos
        else:
            memory = feat
        return dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15MemoryAdapterSupportConditionedOrientedDINO(
        P15EncoderBypassSupportConditionedOrientedDINO):
    """Encoder-bypass variant with a cheap token-wise memory adapter.

    This keeps the strict E2E set-prediction path unchanged and avoids
    deformable encoder self-attention. It tests whether the encoder gain comes
    from lightweight memory normalization/projection or from cross-position
    context mixing that a token-wise adapter cannot provide.
    """

    def __init__(self,
                 *args,
                 memory_adapter_add_pos: bool = True,
                 memory_adapter_hidden_channels: int = 512,
                 memory_adapter_residual_scale: float = 0.5,
                 memory_adapter_norm: bool = True,
                 **kwargs) -> None:
        kwargs.pop('bypass_add_pos_to_memory', None)
        self.memory_adapter_add_pos = bool(memory_adapter_add_pos)
        self.memory_adapter_hidden_channels = int(
            memory_adapter_hidden_channels)
        self.memory_adapter_residual_scale = float(
            memory_adapter_residual_scale)
        self.memory_adapter_use_norm = bool(memory_adapter_norm)
        if self.memory_adapter_hidden_channels <= 0:
            raise ValueError('memory_adapter_hidden_channels must be positive')
        super().__init__(
            *args, bypass_add_pos_to_memory=False, **kwargs)

        self.memory_adapter_norm = (
            nn.LayerNorm(self.embed_dims)
            if self.memory_adapter_use_norm else nn.Identity())
        self.memory_adapter_fc1 = nn.Linear(
            self.embed_dims, self.memory_adapter_hidden_channels)
        self.memory_adapter_fc2 = nn.Linear(
            self.memory_adapter_hidden_channels, self.embed_dims)
        self.encoder_attitude_debug = dict(
            strict_e2e=True,
            uses_dense_detection_head=False,
            uses_nms=False,
            bypasses_deformable_encoder=True,
            uses_tokenwise_memory_adapter=True,
            memory_adapter_add_pos=self.memory_adapter_add_pos,
            memory_adapter_hidden_channels=self.memory_adapter_hidden_channels,
            memory_adapter_residual_scale=(
                self.memory_adapter_residual_scale))

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.xavier_uniform_(self.memory_adapter_fc1.weight)
        nn.init.xavier_uniform_(self.memory_adapter_fc2.weight)
        nn.init.zeros_(self.memory_adapter_fc1.bias)
        nn.init.zeros_(self.memory_adapter_fc2.bias)

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        del level_start_index, valid_ratios
        memory = feat + feat_pos if self.memory_adapter_add_pos else feat
        adapter_input = self.memory_adapter_norm(memory)
        adapter_delta = self.memory_adapter_fc2(
            torch.nn.functional.gelu(self.memory_adapter_fc1(adapter_input)))
        memory = memory + self.memory_adapter_residual_scale * adapter_delta
        return dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15DNStyleQueryPosEncoderBypassSupportConditionedOrientedDINO(
        P15EncoderBypassSupportConditionedOrientedDINO):
    """No-encoder P15 variant with DN/DAB-style initial query position prior.

    The rotated DINO decoder already builds a reference sine ``query_pos`` at
    every decoder layer. This variant additionally injects a projected sine
    encoding of the initial reference box into the query content before the
    decoder starts, including denoising queries during training. It keeps the
    strict E2E fixed-set output path unchanged: no dense head, no NMS, and no
    score-threshold fallback.
    """

    def __init__(self,
                 *args,
                 dn_style_query_pos_scale: float = 0.5,
                 dn_style_query_pos_temperature: int = 10000,
                 **kwargs) -> None:
        self.dn_style_query_pos_scale = float(dn_style_query_pos_scale)
        self.dn_style_query_pos_temperature = int(
            dn_style_query_pos_temperature)
        super().__init__(*args, **kwargs)
        if self.embed_dims % 2 != 0:
            raise ValueError('embed_dims must be even for sine box encoding')
        self.dn_style_query_pos_proj = MLP(
            self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)
        self.dn_style_query_pos_norm = nn.LayerNorm(self.embed_dims)
        self.dn_style_query_pos_debug = dict(
            strict_e2e=True,
            uses_dense_detection_head=False,
            uses_nms=False,
            bypasses_deformable_encoder=True,
            uses_dn_style_initial_query_pos=True,
            dn_style_query_pos_scale=self.dn_style_query_pos_scale,
            dn_style_query_pos_temperature=(
                self.dn_style_query_pos_temperature))

    def init_weights(self) -> None:
        super().init_weights()
        for param in self.dn_style_query_pos_proj.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def _dn_style_query_position_prior(self,
                                       reference_points: Tensor) -> Tensor:
        pos_embed = coordinate_to_encoding(
            reference_points,
            num_feats=self.embed_dims // 2,
            temperature=self.dn_style_query_pos_temperature)
        pos_embed = self.dn_style_query_pos_proj(pos_embed)
        return self.dn_style_query_pos_norm(pos_embed)

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor,
                    batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            batch_data_samples=batch_data_samples)
        if self.dn_style_query_pos_scale != 0.0:
            query_pos = self._dn_style_query_position_prior(
                decoder_inputs_dict['reference_points'])
            decoder_inputs_dict['query'] = (
            decoder_inputs_dict['query'] +
                self.dn_style_query_pos_scale * query_pos)
        return decoder_inputs_dict, head_inputs_dict


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P24MemoryAdapterDNStyleQueryPosSupportConditionedOrientedDINO(
        P15MemoryAdapterSupportConditionedOrientedDINO):
    """Strict-E2E Arbor variant combining memory adaptation and query priors.

    The model keeps the encoder-bypass fixed-set path, but adds two lightweight
    structure changes in different places: token-wise memory adaptation before
    proposal/query initialization, and a DN/DAB-style reference-box positional
    prior before the rotated decoder. This tests whether local memory
    recalibration and geometry-aware query content are complementary.
    """

    def __init__(self,
                 *args,
                 dn_style_query_pos_scale: float = 0.5,
                 dn_style_query_pos_temperature: int = 10000,
                 **kwargs) -> None:
        self.dn_style_query_pos_scale = float(dn_style_query_pos_scale)
        self.dn_style_query_pos_temperature = int(
            dn_style_query_pos_temperature)
        super().__init__(*args, **kwargs)
        if self.embed_dims % 2 != 0:
            raise ValueError('embed_dims must be even for sine box encoding')
        self.dn_style_query_pos_proj = MLP(
            self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)
        self.dn_style_query_pos_norm = nn.LayerNorm(self.embed_dims)
        self.encoder_attitude_debug.update(
            uses_dn_style_initial_query_pos=True,
            combines_memory_adapter_and_query_pos=True,
            dn_style_query_pos_scale=self.dn_style_query_pos_scale,
            dn_style_query_pos_temperature=(
                self.dn_style_query_pos_temperature))

    def init_weights(self) -> None:
        super().init_weights()
        for param in self.dn_style_query_pos_proj.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def _dn_style_query_position_prior(self,
                                       reference_points: Tensor) -> Tensor:
        pos_embed = coordinate_to_encoding(
            reference_points,
            num_feats=self.embed_dims // 2,
            temperature=self.dn_style_query_pos_temperature)
        pos_embed = self.dn_style_query_pos_proj(pos_embed)
        return self.dn_style_query_pos_norm(pos_embed)

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor,
                    batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            batch_data_samples=batch_data_samples)
        if self.dn_style_query_pos_scale != 0.0:
            query_pos = self._dn_style_query_position_prior(
                decoder_inputs_dict['reference_points'])
            decoder_inputs_dict['query'] = (
                decoder_inputs_dict['query'] +
                self.dn_style_query_pos_scale * query_pos)
        return decoder_inputs_dict, head_inputs_dict
