"""P13 dense support-fusion detection decoder head.

P13 is the B-route smoke line: support conditioning is allowed to emit class
logits and box/angle residuals directly from the fusion stage.  The inherited
RTMDet head is intentionally kept as a thin loss/decode wrapper for the smoke.
"""

from __future__ import annotations

import math

import torch
from mmengine.model import normal_init
from torch import Tensor, nn
import torch.nn.functional as F

from mmrotate.models.dense_heads.rotated_rtmdet_head import (
    RotatedRTMDetSepBNHead,
)
from mmrotate.registry import MODELS


def build_orthogonal_support_tokens(num_classes: int,
                                    embed_channels: int,
                                    scale: float = 1.0,
                                    tail_std: float = 0.0) -> Tensor:
    """Build class support tokens with orthogonal leading dimensions."""
    if num_classes <= 0:
        raise ValueError('num_classes must be positive')
    if embed_channels <= 0:
        raise ValueError('embed_channels must be positive')

    support = torch.zeros(num_classes, embed_channels, dtype=torch.float32)
    eye_dim = min(num_classes, embed_channels)
    support[:, :eye_dim] = torch.eye(num_classes, eye_dim)
    if embed_channels > eye_dim and tail_std > 0:
        support[:, eye_dim:] = torch.empty(
            num_classes, embed_channels - eye_dim).normal_(0.0, tail_std)
    return support * scale


@MODELS.register_module()
class P13FusionDecoderRotatedRTMDetSepBNHead(RotatedRTMDetSepBNHead):
    """Rotated RTMDet head with a dense support-fusion output decoder."""

    def __init__(self,
                 *args,
                 p13_decoder: dict | None = None,
                 **kwargs) -> None:
        self.p13_decoder_cfg = dict(p13_decoder or {})
        self.p13_embed_channels = int(
            self.p13_decoder_cfg.get('embed_channels', 64))
        self.p13_cls_weight = float(
            self.p13_decoder_cfg.get('cls_weight', 1.0))
        self.p13_objectness_weight = float(
            self.p13_decoder_cfg.get('objectness_weight', 0.25))
        self.p13_bbox_delta_weight = float(
            self.p13_decoder_cfg.get('bbox_delta_weight', 0.02))
        self.p13_angle_delta_weight = float(
            self.p13_decoder_cfg.get('angle_delta_weight', 0.02))
        super().__init__(*args, **kwargs)
        self._init_p13_support_tokens()
        self.p13_debug: dict[str, object] = {}

    def _init_layers(self) -> None:
        super()._init_layers()
        pad = self.pred_kernel_size // 2
        self.p13_query_proj = nn.ModuleList()
        self.p13_objectness = nn.ModuleList()
        self.p13_bbox_delta = nn.ModuleList()
        self.p13_angle_delta = nn.ModuleList()
        for idx in range(len(self.prior_generator.strides)):
            self.p13_query_proj.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.p13_embed_channels,
                    self.pred_kernel_size,
                    padding=pad))
            self.p13_objectness.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors,
                    self.pred_kernel_size,
                    padding=pad))
            self.p13_bbox_delta.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.rtm_reg[idx].out_channels,
                    self.pred_kernel_size,
                    padding=pad))
            self.p13_angle_delta.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.rtm_ang[idx].out_channels,
                    self.pred_kernel_size,
                    padding=pad))

    def _init_p13_support_tokens(self) -> None:
        support_init = str(
            self.p13_decoder_cfg.get('support_init', 'orthogonal'))
        support_scale = float(
            self.p13_decoder_cfg.get('support_scale', 1.0))
        support_std = float(
            self.p13_decoder_cfg.get('support_init_std', 0.02))

        if support_init == 'orthogonal':
            support = build_orthogonal_support_tokens(
                self.num_classes,
                self.p13_embed_channels,
                scale=support_scale,
                tail_std=support_std)
        elif support_init == 'normal':
            support = torch.empty(self.num_classes, self.p13_embed_channels)
            nn.init.normal_(support, mean=0.0, std=support_std)
        else:
            raise ValueError(f'Unsupported P13 support_init: {support_init}')

        self.p13_support_tokens = nn.Parameter(support)
        logit_scale_init = max(
            float(self.p13_decoder_cfg.get('logit_scale_init', 6.0)), 1e-4)
        self.p13_logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale_init), dtype=torch.float32))
        self.p13_class_bias = nn.Parameter(torch.zeros(self.num_classes))

    def init_weights(self) -> None:
        super().init_weights()
        for module_list in (self.p13_query_proj, self.p13_objectness):
            for module in module_list:
                normal_init(module, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)
        for module_list in (self.p13_bbox_delta, self.p13_angle_delta):
            for module in module_list:
                nn.init.constant_(module.weight, 0.0)
                nn.init.constant_(module.bias, 0.0)

    def _p13_cls_logits_from_maps(self, query: Tensor,
                                  objectness: Tensor) -> Tensor:
        batch_size, _, height, width = query.shape
        num_priors = self.num_base_priors
        embed_channels = self.p13_embed_channels

        query = query.reshape(batch_size, num_priors, embed_channels, height,
                              width)
        flat_query = query.permute(0, 1, 3, 4,
                                   2).reshape(-1, embed_channels)
        flat_query = F.normalize(flat_query, dim=-1)
        support = F.normalize(self.p13_support_tokens, dim=-1)

        logits = self.p13_cls_weight * (flat_query @ support.t())
        objectness = objectness.reshape(batch_size, num_priors, height, width)
        flat_objectness = objectness.permute(0, 1, 2, 3).reshape(-1, 1)
        logits = logits + self.p13_objectness_weight * flat_objectness
        logits = logits + self.p13_class_bias[None, :]
        scale = self.p13_logit_scale.exp().clamp(min=1e-4, max=100.0)
        logits = logits * scale

        logits = logits.reshape(batch_size, num_priors, height, width,
                                self.num_classes)
        return logits.permute(0, 1, 4, 2,
                              3).reshape(batch_size,
                                         num_priors * self.num_classes,
                                         height, width)

    def forward(self, feats: tuple[Tensor, ...]) -> tuple:
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            bbox_delta = self.p13_bbox_delta[idx](reg_feat).tanh() * stride[0]
            reg_dist = (
                reg_dist +
                self.p13_bbox_delta_weight * bbox_delta).clamp(min=1e-4)

            angle_pred = self.rtm_ang[idx](reg_feat)
            angle_delta = self.p13_angle_delta[idx](reg_feat).tanh()
            angle_pred = (
                angle_pred + self.p13_angle_delta_weight * angle_delta)

            query = self.p13_query_proj[idx](cls_feat)
            objectness = self.p13_objectness[idx](cls_feat)
            cls_score = self._p13_cls_logits_from_maps(query, objectness)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)

        self.p13_debug = {
            'used_rtm_cls': False,
            'direct_cls_from_fusion_decoder': True,
            'direct_bbox_residual_from_fusion_decoder': True,
            'cls_weight': self.p13_cls_weight,
            'objectness_weight': self.p13_objectness_weight,
            'bbox_delta_weight': self.p13_bbox_delta_weight,
            'angle_delta_weight': self.p13_angle_delta_weight,
        }
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)
