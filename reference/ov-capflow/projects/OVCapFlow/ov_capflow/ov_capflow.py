from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.layers import SinePositionalEncoding
from mmdet.structures import OptSampleList
from mmdet.structures import SampleList
from torch import Tensor

from mmrotate.registry import MODELS
from projects.GroundingDINO.groundingdino.grounding_dino import (
    RotatedGroundingDINO, )
from projects.GroundingDINO.groundingdino.grounding_dino_layers import (
    GroundingDinoTransformerEncoder, )

from .ov_capflow_layers import OVCapFlowDecoder
from .grouped_queries import (expand_dn_attention_mask,
                              repeat_matching_queries)
from .null_reservoir import null_reservoir_losses
from .query_initializer import FixedRotatedQueryInitializer
from .semantic_capacity import density_capacity_losses
from .freeze_except_hook import apply_freeze_except


@MODELS.register_module()
class OVCapFlow(RotatedGroundingDINO):
    """Strict-query substrate for open-vocabulary rotated set prediction.

    Unlike the upstream DINO initialization, matching queries and their 5-D
    references never originate from scored encoder proposals.
    """

    def __init__(self, density_loss_cfg=None, null_loss_cfg=None,
                 train_query_groups=1, query_init_mode='fixed',
                 freeze_except_patterns=None, **kwargs):
        if not isinstance(train_query_groups, int) or train_query_groups < 1:
            raise ValueError('train_query_groups must be a positive integer')
        if query_init_mode not in ('fixed', 'encoder_proposal'):
            raise ValueError(
                'query_init_mode must be fixed or encoder_proposal')
        if (query_init_mode == 'encoder_proposal' and
                kwargs.get('as_two_stage') is not True):
            raise ValueError(
                'encoder_proposal query initialization requires '
                'as_two_stage=True')
        self.density_loss_cfg = dict(density_loss_cfg or {})
        self.null_loss_cfg = dict(null_loss_cfg or {})
        self.train_query_groups = train_query_groups
        self.query_init_mode = query_init_mode
        super().__init__(**kwargs)
        if freeze_except_patterns is not None:
            apply_freeze_except(self, freeze_except_patterns)

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = OVCapFlowDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

        self.query_initializer = FixedRotatedQueryInitializer(
            num_queries=self.num_queries, embed_dims=self.embed_dims)

    @property
    def query_embedding(self):
        """Expose the DINO API without registering the module twice."""
        return self.query_initializer.query_embedding

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        """Prepare fixed matching queries without encoder proposal ranking."""
        batch_size = memory.shape[0]
        query, reference_points = self.query_initializer(batch_size)
        enc_outputs_class = None
        enc_outputs_coord = None
        if self.query_init_mode == 'encoder_proposal':
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            branch_index = self.decoder.num_layers
            encoder_class = self.bbox_head.cls_branches[branch_index](
                output_memory, memory_text, text_token_mask)
            encoder_coord_unact = (
                self.bbox_head.reg_branches[branch_index](output_memory)
                + output_proposals)
            topk_indices = torch.topk(
                encoder_class.max(-1)[0], k=self.num_queries, dim=1)[1]
            cls_features = encoder_class.shape[-1]
            enc_outputs_class = torch.gather(
                encoder_class, 1,
                topk_indices.unsqueeze(-1).expand(-1, -1, cls_features))
            topk_coord_unact = torch.gather(
                encoder_coord_unact, 1,
                topk_indices.unsqueeze(-1).expand(-1, -1, 5))
            enc_outputs_coord = topk_coord_unact.sigmoid()
            reference_points = topk_coord_unact.detach().sigmoid()
        matching_query_count = self.num_queries

        if self.training:
            query, reference_points = repeat_matching_queries(
                query, reference_points, self.train_query_groups)
            matching_query_count *= self.train_query_groups
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            num_dn = int(dn_meta['num_denoising_queries'])
            dn_mask = expand_dn_attention_mask(
                dn_mask,
                num_dn=num_dn,
                queries_per_group=self.num_queries,
                groups=self.train_query_groups)
            dn_meta = dict(dn_meta)
            dn_meta.update(
                num_matching_query_groups=self.train_query_groups,
                num_matching_queries_per_group=self.num_queries)
            query = torch.cat([dn_label_query, query], dim=1)
            matching_reference_logits = torch.logit(
                reference_points.clamp(min=1e-4, max=1 - 1e-4))
            reference_points = torch.cat(
                [dn_bbox_query, matching_reference_logits], dim=1).sigmoid()
        else:
            dn_mask, dn_meta = None, None

        decoder_inputs_dict = dict(
            query=query,
            native_query=query,
            matching_query_count=matching_query_count,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            dn_meta=dn_meta,
        ) if self.training else {}
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def _append_calibration_state(self, outputs: Dict) -> Dict:
        outputs = dict(outputs)
        last_layer = self.decoder.layers[-1]
        if self.decoder.last_null_logits is not None:
            outputs['null_logits'] = self.decoder.last_null_logits
        if last_layer.last_capacity is not None:
            outputs['capacity'] = last_layer.last_capacity
        if last_layer.last_semantic_gate is not None:
            outputs['semantic_gate'] = last_layer.last_semantic_gate
        return outputs

    def forward_decoder(self, *args, **kwargs) -> Dict:
        outputs = super().forward_decoder(*args, **kwargs)
        return self._append_calibration_state(outputs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        losses = super().loss(batch_inputs, batch_data_samples)
        last_layer = self.decoder.layers[-1]
        target_count = next(iter(losses.values())).new_tensor(
            [len(sample.gt_instances) for sample in batch_data_samples])
        if last_layer.last_capacity is not None:
            density_losses = density_capacity_losses(
                last_layer.last_capacity,
                last_layer.last_predicted_count,
                target_count)
            density_weight = float(
                self.density_loss_cfg.get('weight', 1.0))
            losses.update({
                name: density_weight * value
                for name, value in density_losses.items()
            })

        if self.decoder.last_null_logits is not None:
            matched = self.bbox_head.last_matching_mask
            if matched is None:
                raise RuntimeError(
                    'null loss requires final Hungarian matching mask')
            gate = last_layer.last_semantic_gate
            gate_strength = None if gate is None else gate.abs().mean(dim=-1)
            losses.update(null_reservoir_losses(
                self.decoder.last_null_logits,
                matched,
                target_count,
                gate_strength=gate_strength,
                matched_weight=float(
                    self.null_loss_cfg.get('matched_weight', 1.0)),
                unmatched_weight=float(
                    self.null_loss_cfg.get('unmatched_weight', 1.0)),
                mass_weight=float(
                    self.null_loss_cfg.get('mass_weight', 0.1)),
                gate_order_weight=float(
                    self.null_loss_cfg.get('gate_order_weight', 0.1)),
                gate_margin=float(
                    self.null_loss_cfg.get('gate_margin', 0.0))))
        return losses
