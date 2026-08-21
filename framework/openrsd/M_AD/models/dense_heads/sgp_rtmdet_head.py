"""SGP-Head: Gaussian product posterior RTMDet head.

This head deliberately avoids OpenRSD's support-fusion stack.  It keeps the
RTMDet regression/angle path but replaces the classification convolution with a
closed-form product score between query and class support Gaussians.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from mmengine.model import normal_init
from torch import Tensor, nn
import torch.nn.functional as F

from M_AD.models.utils.gaussian_semantic_scale import (
    load_class_geometry_priors,
)
from mmrotate.models.dense_heads.rotated_rtmdet_head import (
    RotatedRTMDetSepBNHead,
)
from mmrotate.registry import MODELS


def diagonal_gaussian_product_logit(query_mu: Tensor, query_logvar: Tensor,
                                    support_mu: Tensor,
                                    support_logvar: Tensor) -> Tensor:
    """Score query/support Gaussian overlap with diagonal covariance.

    Args:
        query_mu: Query means shaped ``(N, D)``.
        query_logvar: Query log variances shaped ``(N, D)``.
        support_mu: Class support means shaped ``(C, D)``.
        support_logvar: Class support log variances shaped ``(C, D)``.

    Returns:
        Product logits shaped ``(N, C)``.
    """
    if query_mu.ndim != 2 or query_logvar.ndim != 2:
        raise ValueError('query Gaussian tensors must be 2D')
    if support_mu.ndim != 2 or support_logvar.ndim != 2:
        raise ValueError('support Gaussian tensors must be 2D')
    if query_mu.shape != query_logvar.shape:
        raise ValueError('query_mu and query_logvar shape mismatch')
    if support_mu.shape != support_logvar.shape:
        raise ValueError('support_mu and support_logvar shape mismatch')
    if query_mu.shape[-1] != support_mu.shape[-1]:
        raise ValueError('query/support embedding dimensions differ')

    q_var = query_logvar.exp().clamp(min=1e-6, max=1e6)
    s_var = support_logvar.exp().clamp(min=1e-6, max=1e6)
    joint_var = q_var[:, None, :] + s_var[None, :, :]
    diff = query_mu[:, None, :] - support_mu[None, :, :]
    energy = diff.square() / joint_var + torch.log(joint_var)
    return -0.5 * energy.mean(dim=-1)


def geometry_logprob_from_distances(distances: Tensor,
                                    class_geometry_mean: Tensor,
                                    class_geometry_std: Tensor,
                                    valid_mask: Tensor | None = None,
                                    min_std: float = 1e-4) -> Tensor:
    """Class geometry log likelihood from RTMDet distance predictions.

    The geometry vector is ``[log(area), log(long_side / short_side)]``.
    Invalid class priors return zero contribution, so missing priors do not
    distort the semantic Gaussian product branch.
    """
    if distances.ndim != 2 or distances.shape[-1] != 4:
        raise ValueError('distances must be shaped (N, 4)')
    if class_geometry_mean.shape != class_geometry_std.shape:
        raise ValueError('geometry mean/std shape mismatch')
    if class_geometry_mean.ndim != 2 or class_geometry_mean.shape[-1] != 2:
        raise ValueError('class geometry priors must be shaped (C, 2)')

    distances = distances.to(dtype=class_geometry_mean.dtype)
    distances = distances.clamp(min=1e-4)
    width = distances[:, 0] + distances[:, 2]
    height = distances[:, 1] + distances[:, 3]
    long_side = torch.maximum(width, height).clamp(min=1e-4)
    short_side = torch.minimum(width, height).clamp(min=1e-4)
    geometry = torch.stack(
        [torch.log(width * height), torch.log(long_side / short_side)],
        dim=-1)

    means = class_geometry_mean.to(device=distances.device, dtype=distances.dtype)
    stds = class_geometry_std.to(device=distances.device,
                                 dtype=distances.dtype).clamp(min=min_std)
    z = (geometry[:, None, :] - means[None, :, :]) / stds[None, :, :]
    log_prob = (
        -0.5 * z.square()
        - torch.log(stds)[None, :, :]
        - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)

    if valid_mask is not None:
        valid = valid_mask.to(device=distances.device, dtype=torch.bool)
        log_prob = torch.where(valid[None, :], log_prob,
                               torch.zeros_like(log_prob))
    return log_prob


@MODELS.register_module()
class SGPRotatedRTMDetSepBNHead(RotatedRTMDetSepBNHead):
    """Rotated RTMDet head with Gaussian product posterior class logits."""

    def __init__(self,
                 *args,
                 sgp_head: dict | None = None,
                 **kwargs) -> None:
        self.sgp_head_cfg = dict(sgp_head or {})
        self.sgp_embed_channels = int(
            self.sgp_head_cfg.get('embed_channels', 64))
        self.sgp_semantic_weight = float(
            self.sgp_head_cfg.get('semantic_weight', 1.0))
        self.sgp_geometry_weight = float(
            self.sgp_head_cfg.get('geometry_weight', 0.0))
        self.sgp_objectness_weight = float(
            self.sgp_head_cfg.get('objectness_weight', 0.0))
        self.sgp_prototype_cosine_weight = float(
            self.sgp_head_cfg.get('prototype_cosine_weight', 0.0))
        self.sgp_query_logvar_min = float(
            self.sgp_head_cfg.get('query_logvar_min', -6.0))
        self.sgp_query_logvar_max = float(
            self.sgp_head_cfg.get('query_logvar_max', 4.0))
        self.sgp_support_logvar_min = float(
            self.sgp_head_cfg.get('support_logvar_min', -6.0))
        self.sgp_support_logvar_max = float(
            self.sgp_head_cfg.get('support_logvar_max', 4.0))
        self.sgp_query_logvar_bias_init = float(
            self.sgp_head_cfg.get('query_logvar_bias_init', 0.0))
        super().__init__(*args, **kwargs)
        self._init_sgp_support_distribution()
        self.sgp_debug: dict[str, object] = {}

    def _init_layers(self) -> None:
        super()._init_layers()
        pad = self.pred_kernel_size // 2
        self.sgp_query_mu = nn.ModuleList()
        self.sgp_query_logvar = nn.ModuleList()
        self.sgp_objectness = nn.ModuleList()
        for _ in range(len(self.prior_generator.strides)):
            self.sgp_query_mu.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.sgp_embed_channels,
                    self.pred_kernel_size,
                    padding=pad))
            self.sgp_query_logvar.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.sgp_embed_channels,
                    self.pred_kernel_size,
                    padding=pad))
            self.sgp_objectness.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors,
                    self.pred_kernel_size,
                    padding=pad))

    def _init_sgp_support_distribution(self) -> None:
        support_std = float(self.sgp_head_cfg.get('support_mu_init_std', 0.25))
        support_init = str(self.sgp_head_cfg.get('support_mu_init',
                                                 'normal'))
        support_scale = float(self.sgp_head_cfg.get('support_mu_scale', 1.0))
        support_logvar_init = float(
            self.sgp_head_cfg.get('support_logvar_init', 0.0))
        if support_init == 'orthogonal':
            support_mu = torch.zeros(self.num_classes,
                                     self.sgp_embed_channels)
            eye_dim = min(self.num_classes, self.sgp_embed_channels)
            support_mu[:, :eye_dim] = torch.eye(
                self.num_classes, eye_dim, dtype=torch.float32)
            if self.sgp_embed_channels > eye_dim and support_std > 0:
                support_mu[:, eye_dim:] = torch.empty(
                    self.num_classes,
                    self.sgp_embed_channels - eye_dim).normal_(
                        mean=0.0, std=support_std)
            support_mu = support_mu * support_scale
        elif support_init == 'normal':
            support_mu = torch.empty(self.num_classes,
                                     self.sgp_embed_channels)
            nn.init.normal_(support_mu, mean=0.0, std=support_std)
        else:
            raise ValueError(
                f'Unsupported SGP support_mu_init: {support_init}')
        support_logvar = torch.full_like(support_mu, support_logvar_init)
        self.sgp_support_mu = nn.Parameter(support_mu)
        self.sgp_support_logvar = nn.Parameter(support_logvar)

        logit_scale_init = max(
            float(self.sgp_head_cfg.get('logit_scale_init', 1.0)), 1e-4)
        self.sgp_logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale_init), dtype=torch.float32))
        self.sgp_class_bias = nn.Parameter(torch.zeros(self.num_classes))

        class_names = self._resolve_class_names(
            self.sgp_head_cfg.get('class_names'))
        priors_csv = self.sgp_head_cfg.get('class_geometry_priors_csv')
        if priors_csv:
            means, stds, valid, resolved_names = load_class_geometry_priors(
                priors_csv, class_names, self.num_classes)
        else:
            means = torch.zeros((self.num_classes, 2), dtype=torch.float32)
            stds = torch.ones((self.num_classes, 2), dtype=torch.float32)
            valid = torch.zeros(self.num_classes, dtype=torch.bool)
            resolved_names = list(class_names)
        self.sgp_class_names = list(resolved_names)
        self.register_buffer('sgp_geometry_mean', means)
        self.register_buffer('sgp_geometry_std', stds)
        self.register_buffer('sgp_geometry_valid', valid)

    def _resolve_class_names(self,
                             class_names: Sequence[str] | None) -> list[str]:
        if class_names is None:
            return [str(idx) for idx in range(self.num_classes)]
        names = [str(name) for name in class_names]
        if len(names) < self.num_classes:
            names.extend(str(idx) for idx in range(len(names),
                                                   self.num_classes))
        return names[:self.num_classes]

    def init_weights(self) -> None:
        super().init_weights()
        for module_list in (
                self.sgp_query_mu, self.sgp_query_logvar,
                self.sgp_objectness):
            for module in module_list:
                normal_init(module, mean=0.0, std=0.01)
        for module in self.sgp_query_logvar:
            nn.init.constant_(module.bias, self.sgp_query_logvar_bias_init)
        for module in self.sgp_objectness:
            nn.init.constant_(module.bias, 0.0)

    def _sgp_logits_from_maps(self, query_mu: Tensor, query_logvar: Tensor,
                              bbox_dist: Tensor, objectness: Tensor) -> Tensor:
        batch_size, _, height, width = query_mu.shape
        num_priors = self.num_base_priors
        embed_channels = self.sgp_embed_channels

        query_mu = query_mu.reshape(batch_size, num_priors, embed_channels,
                                    height, width)
        query_logvar = query_logvar.reshape(batch_size, num_priors,
                                            embed_channels, height, width)
        query_logvar = query_logvar.clamp(
            min=self.sgp_query_logvar_min, max=self.sgp_query_logvar_max)

        flat_mu = query_mu.permute(0, 1, 3, 4, 2).reshape(-1, embed_channels)
        flat_logvar = query_logvar.permute(0, 1, 3, 4,
                                           2).reshape(-1, embed_channels)
        support_logvar = self.sgp_support_logvar.clamp(
            min=self.sgp_support_logvar_min, max=self.sgp_support_logvar_max)
        semantic_logits = diagonal_gaussian_product_logit(
            flat_mu, flat_logvar, self.sgp_support_mu, support_logvar)
        logits = self.sgp_semantic_weight * semantic_logits

        if self.sgp_prototype_cosine_weight != 0.0:
            query_proto = F.normalize(flat_mu, dim=-1)
            support_proto = F.normalize(self.sgp_support_mu, dim=-1)
            prototype_logits = query_proto @ support_proto.t()
            logits = logits + self.sgp_prototype_cosine_weight * prototype_logits

        if self.sgp_geometry_weight != 0.0:
            flat_dist = bbox_dist.reshape(batch_size, num_priors, 4, height,
                                          width)
            flat_dist = flat_dist.permute(0, 1, 3, 4, 2).reshape(-1, 4)
            geometry_logits = geometry_logprob_from_distances(
                flat_dist,
                self.sgp_geometry_mean,
                self.sgp_geometry_std,
                valid_mask=self.sgp_geometry_valid)
            logits = logits + self.sgp_geometry_weight * geometry_logits

        if self.sgp_objectness_weight != 0.0:
            objectness = objectness.reshape(batch_size, num_priors, height,
                                            width)
            flat_objectness = objectness.permute(0, 1, 2,
                                                 3).reshape(-1, 1)
            logits = logits + self.sgp_objectness_weight * flat_objectness

        logits = logits + self.sgp_class_bias[None, :]
        scale = self.sgp_logit_scale.exp().clamp(min=1e-4, max=100.0)
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
            angle_pred = self.rtm_ang[idx](reg_feat)

            query_mu = self.sgp_query_mu[idx](cls_feat)
            query_logvar = self.sgp_query_logvar[idx](cls_feat)
            objectness = self.sgp_objectness[idx](cls_feat)
            cls_score = self._sgp_logits_from_maps(
                query_mu, query_logvar, reg_dist, objectness)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)

        self.sgp_debug = {
            'used_rtm_cls': False,
            'semantic_weight': self.sgp_semantic_weight,
            'geometry_weight': self.sgp_geometry_weight,
            'objectness_weight': self.sgp_objectness_weight,
            'prototype_cosine_weight': self.sgp_prototype_cosine_weight,
            'num_valid_geometry_priors': int(self.sgp_geometry_valid.sum()),
        }
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)
