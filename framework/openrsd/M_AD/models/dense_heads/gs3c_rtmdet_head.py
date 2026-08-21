"""Closed-set RTMDet head with Gaussian semantic-scale calibration."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.structures.bbox import cat_boxes, distance2bbox
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmrotate.models.dense_heads.rotated_rtmdet_head import RotatedRTMDetSepBNHead
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes, distance2obb
from torch import Tensor

from M_AD.models.utils.gaussian_semantic_scale import (
    GaussianScaleLogitAdapter,
    GaussianSemanticScaleDensityHead,
    gaussian_ap_constrained_support_ranking_projection_loss,
    build_classwise_gaussian_parameter,
    continuous_gaussian_logit_energy_delta,
    gaussian_ap_safe_support_projection_loss,
    gaussian_geometry_support_stats,
    gaussian_pos_hardneg_consistency_loss,
    gaussian_log_area_stats,
    gaussian_positive_scale_consistency_loss,
    gaussian_support_negative_focal_loss,
    load_class_geometry_priors,
    load_class_log_area_priors,
    semantic_scale_density_delta_outlier_loss,
    semantic_scale_density_nll_loss,
    semantic_scale_density_pair_margin_loss,
    semantic_scale_density_positive_delta_loss,
)
from M_AD.models.utils.ep2_path_probe_dump import (
    append_ep2_path_probe_rows,
    build_ep2_path_probe_rows,
)


@MODELS.register_module()
class GSRRotatedRTMDetSepBNHead(RotatedRTMDetSepBNHead):
    """RTMDet rotated dense head with class-conditional log-area modeling.

    This is the closed-set counterpart of G-S3C. It keeps the standard
    RTMDet training head intact and applies a non-positive logit correction
    after dense logits are produced and before ``filter_scores_and_topk``.
    """

    def __init__(self,
                 *args,
                 gaussian_semantic_scale: Optional[dict] = None,
                 **kwargs) -> None:
        self.gaussian_semantic_scale_cfg = dict(
            gaussian_semantic_scale or {})
        super().__init__(*args, **kwargs)
        self._init_gaussian_semantic_scale()
        self._init_ep2_path_probe_dump()

    def _init_ep2_path_probe_dump(self) -> None:
        cfg = dict(self.gaussian_semantic_scale_cfg.get(
            'ep2_path_probe', {}) or {})
        self.ep2_path_probe_enable = bool(cfg.get('enable', False))
        self.ep2_path_probe_output_csv = cfg.get('output_csv')
        self.ep2_path_probe_class_pairs = tuple(cfg.get('class_pairs', ()))
        self.ep2_path_probe_max_locations = int(
            cfg.get('max_locations_per_level', 128))

    def _dump_ep2_path_probe_logits(self,
                                    cls_logits: Tensor,
                                    bbox_pred: Tensor,
                                    angle_pred: Tensor,
                                    priors: Tensor,
                                    img_shape,
                                    img_meta=None,
                                    level_idx: int = -1) -> None:
        if (not getattr(self, 'ep2_path_probe_enable', False)
                or not getattr(self, 'ep2_path_probe_output_csv', None)
                or not getattr(self, 'ep2_path_probe_class_pairs', ())):
            return
        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
        decoded_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)
        decoded_bboxes = self.bbox_coder.decode(
            priors, decoded_pred, max_shape=img_shape)
        decoded_tensor = (
            decoded_bboxes.tensor
            if isinstance(decoded_bboxes, RotatedBoxes)
            else decoded_bboxes)
        rows = build_ep2_path_probe_rows(
            cls_logits=cls_logits[:, :int(self.num_classes)],
            decoded_bboxes=decoded_tensor,
            class_names=self.gaussian_semantic_scale_class_names,
            class_pairs=self.ep2_path_probe_class_pairs,
            log_area_mean=self.gaussian_semantic_log_area_mean,
            log_area_std=self.gaussian_semantic_log_area_std,
            valid_mask=self.gaussian_semantic_valid_mask,
            img_meta=img_meta,
            level_idx=level_idx,
            max_locations=self.ep2_path_probe_max_locations)
        append_ep2_path_probe_rows(self.ep2_path_probe_output_csv, rows)

    def _init_gaussian_semantic_scale(self) -> None:
        cfg = dict(self.gaussian_semantic_scale_cfg)
        self.gaussian_semantic_scale_enable = bool(cfg.get('enable', False))
        self.gaussian_semantic_scale_mode = str(
            cfg.get('mode', 'continuous_logit_energy'))
        if self.gaussian_semantic_scale_mode not in {
                'continuous_logit_energy', 'logit_adapter'}:
            raise ValueError(
                'gaussian_semantic_scale.mode must be '
                '"continuous_logit_energy" or "logit_adapter"')
        self.gaussian_semantic_scale_z0 = float(cfg.get('z0', 4.0))
        self.gaussian_semantic_scale_beta = float(
            cfg.get('beta', math.log(4.0)))
        self.gaussian_semantic_scale_domain_mode = str(
            cfg.get('domain_mode', cfg.get('domain', 'closed_set')))
        self.gaussian_semantic_scale_adapter_nonpositive_delta = bool(
            cfg.get('adapter_nonpositive_delta', True))
        self.gaussian_semantic_scale_adapter_max_delta_abs = float(
            cfg.get('adapter_max_delta_abs', 0.0) or 0.0)
        self.gaussian_semantic_scale_adapter_min_abs_z = float(
            cfg.get('adapter_min_abs_z', 0.0) or 0.0)
        self.gaussian_semantic_scale_adapter_min_score = float(
            cfg.get('adapter_min_score', 0.0) or 0.0)
        consistency_cfg = dict(cfg.get('consistency_loss', {}) or {})
        self.gaussian_semantic_consistency_enable = bool(
            consistency_cfg.get('enable', False))
        self.gaussian_semantic_consistency_weight = float(
            consistency_cfg.get('loss_weight', 0.05))
        self.gaussian_semantic_consistency_margin = float(
            consistency_cfg.get('margin', 0.5))
        self.gaussian_semantic_consistency_min_logprob_gap = float(
            consistency_cfg.get('min_logprob_gap', 1.0))
        self.gaussian_semantic_consistency_max_hardneg = int(
            consistency_cfg.get('max_hardneg', 1))
        self.gaussian_semantic_consistency_min_hardneg_logit = (
            consistency_cfg.get('min_hardneg_logit', None))
        if self.gaussian_semantic_consistency_min_hardneg_logit is not None:
            self.gaussian_semantic_consistency_min_hardneg_logit = float(
                self.gaussian_semantic_consistency_min_hardneg_logit)
        self.gaussian_semantic_consistency_max_gt_abs_z = (
            consistency_cfg.get('max_gt_abs_z', None))
        if self.gaussian_semantic_consistency_max_gt_abs_z is not None:
            self.gaussian_semantic_consistency_max_gt_abs_z = float(
                self.gaussian_semantic_consistency_max_gt_abs_z)
        self.gaussian_semantic_consistency_log_stats = bool(
            consistency_cfg.get('log_stats', False))
        density_cfg = dict(cfg.get('density_head', {}) or {})
        self.gaussian_semantic_density_enable = bool(
            density_cfg.get('enable', False))
        self.gaussian_semantic_density_loss_weight = float(
            density_cfg.get('loss_weight', 0.05))
        self.gaussian_semantic_density_hidden_channels = int(
            density_cfg.get('hidden_channels', cfg.get('adapter_hidden', 64)))
        self.gaussian_semantic_density_mean_residual_scale = float(
            density_cfg.get('mean_residual_scale', 1.0))
        self.gaussian_semantic_density_log_std_delta_limit = float(
            density_cfg.get('log_std_delta_limit', 1.0))
        self.gaussian_semantic_density_max_delta_abs = float(
            density_cfg.get('max_delta_abs', 0.5))
        self.gaussian_semantic_density_nonpositive_delta = bool(
            density_cfg.get('nonpositive_delta', True))
        self.gaussian_semantic_density_apply_logit_delta = bool(
            density_cfg.get('apply_logit_delta', False))
        self.gaussian_semantic_density_logit_delta_source = str(
            density_cfg.get('logit_delta_source', 'head'))
        if self.gaussian_semantic_density_logit_delta_source not in {
                'head', 'log_prob_energy'}:
            raise ValueError(
                'density_head.logit_delta_source must be "head" or '
                '"log_prob_energy"')
        self.gaussian_semantic_density_logprob_delta_beta = float(
            density_cfg.get('logprob_delta_beta', 1.0))
        self.gaussian_semantic_density_logprob_delta_threshold = float(
            density_cfg.get('logprob_delta_threshold', -8.0))
        self.gaussian_semantic_density_delta_loss_enable = bool(
            density_cfg.get('delta_loss_enable', False))
        self.gaussian_semantic_density_delta_loss_weight = float(
            density_cfg.get('delta_loss_weight', 0.02))
        self.gaussian_semantic_density_delta_loss_min_logprob_gap = float(
            density_cfg.get('delta_loss_min_logprob_gap', 1.0))
        self.gaussian_semantic_density_delta_loss_min_hardneg_score = (
            density_cfg.get('delta_loss_min_hardneg_score', None))
        if self.gaussian_semantic_density_delta_loss_min_hardneg_score is not None:
            self.gaussian_semantic_density_delta_loss_min_hardneg_score = float(
                self.gaussian_semantic_density_delta_loss_min_hardneg_score)
        self.gaussian_semantic_density_delta_loss_min_hardneg_logit = (
            density_cfg.get('delta_loss_min_hardneg_logit', None))
        if self.gaussian_semantic_density_delta_loss_min_hardneg_logit is not None:
            self.gaussian_semantic_density_delta_loss_min_hardneg_logit = float(
                self.gaussian_semantic_density_delta_loss_min_hardneg_logit)
        self.gaussian_semantic_density_delta_loss_target_negative_delta = float(
            density_cfg.get('delta_loss_target_negative_delta', 0.25))
        self.gaussian_semantic_density_delta_loss_gt_keep_weight = float(
            density_cfg.get('delta_loss_gt_keep_weight', 0.1))
        self.gaussian_semantic_density_delta_loss_max_hardneg = int(
            density_cfg.get('delta_loss_max_hardneg', 1))
        self.gaussian_semantic_density_positive_delta_loss_enable = bool(
            density_cfg.get('positive_delta_loss_enable', False))
        self.gaussian_semantic_density_positive_delta_loss_weight = float(
            density_cfg.get('positive_delta_loss_weight', 0.01))
        self.gaussian_semantic_density_positive_delta_loss_target = float(
            density_cfg.get('positive_delta_loss_target', 0.10))
        self.gaussian_semantic_density_positive_delta_loss_min_gt_logprob = (
            density_cfg.get('positive_delta_loss_min_gt_logprob', None))
        if self.gaussian_semantic_density_positive_delta_loss_min_gt_logprob is not None:
            self.gaussian_semantic_density_positive_delta_loss_min_gt_logprob = float(
                self.gaussian_semantic_density_positive_delta_loss_min_gt_logprob)
        self.gaussian_semantic_density_positive_delta_loss_min_gt_score = (
            density_cfg.get('positive_delta_loss_min_gt_score', None))
        if self.gaussian_semantic_density_positive_delta_loss_min_gt_score is not None:
            self.gaussian_semantic_density_positive_delta_loss_min_gt_score = float(
                self.gaussian_semantic_density_positive_delta_loss_min_gt_score)
        self.gaussian_semantic_density_pair_margin_loss_enable = bool(
            density_cfg.get('pair_margin_loss_enable', False))
        self.gaussian_semantic_density_pair_margin_loss_weight = float(
            density_cfg.get('pair_margin_loss_weight', 0.01))
        self.gaussian_semantic_density_pair_margin_loss_margin = float(
            density_cfg.get('pair_margin_loss_margin', 0.20))
        self.gaussian_semantic_density_pair_margin_loss_min_logprob_gap = float(
            density_cfg.get('pair_margin_loss_min_logprob_gap', 1.0))
        self.gaussian_semantic_density_pair_margin_loss_min_hardneg_score = (
            density_cfg.get('pair_margin_loss_min_hardneg_score', None))
        if self.gaussian_semantic_density_pair_margin_loss_min_hardneg_score is not None:
            self.gaussian_semantic_density_pair_margin_loss_min_hardneg_score = float(
                self.gaussian_semantic_density_pair_margin_loss_min_hardneg_score)
        self.gaussian_semantic_density_pair_margin_loss_max_hardneg = int(
            density_cfg.get('pair_margin_loss_max_hardneg', 1))
        self.gaussian_semantic_density_support_negative_loss_enable = bool(
            density_cfg.get('support_negative_loss_enable', False))
        self.gaussian_semantic_density_support_negative_loss_weight = float(
            density_cfg.get('support_negative_loss_weight', 0.01))
        support_negative_min_score = density_cfg.get(
            'support_negative_loss_min_score', 0.05)
        self.gaussian_semantic_density_support_negative_loss_min_score = (
            None if support_negative_min_score is None
            else float(support_negative_min_score))
        self.gaussian_semantic_density_support_negative_loss_min_logprob_gap = (
            float(density_cfg.get(
                'support_negative_loss_min_logprob_gap', 1.0)))
        self.gaussian_semantic_density_support_negative_loss_gamma = float(
            density_cfg.get('support_negative_loss_gamma', 2.0))
        self.gaussian_semantic_density_support_negative_loss_gap_scale = float(
            density_cfg.get('support_negative_loss_gap_scale', 8.0))
        self.gaussian_semantic_density_support_negative_loss_max_extra_weight = (
            float(density_cfg.get(
                'support_negative_loss_max_extra_weight', 2.0)))
        self.gaussian_semantic_density_support_negative_loss_max_hardneg = int(
            density_cfg.get('support_negative_loss_max_hardneg', 1))
        self.gaussian_semantic_density_scale_consistency_loss_enable = bool(
            density_cfg.get('scale_consistency_loss_enable', False))
        self.gaussian_semantic_density_scale_consistency_loss_weight = float(
            density_cfg.get('scale_consistency_loss_weight', 0.01))
        self.gaussian_semantic_density_scale_consistency_loss_max_logprob_drop = (
            float(density_cfg.get(
                'scale_consistency_loss_max_logprob_drop', 0.5)))
        scale_consistency_min_target_logprob = density_cfg.get(
            'scale_consistency_loss_min_target_logprob', None)
        self.gaussian_semantic_density_scale_consistency_loss_min_target_logprob = (
            None if scale_consistency_min_target_logprob is None
            else float(scale_consistency_min_target_logprob))
        level_routing_cfg = dict(density_cfg.get('level_routing', {}) or {})
        if not level_routing_cfg:
            level_routing_cfg = dict(cfg.get('level_routing', {}) or {})
        self.gaussian_semantic_level_routing_enable = bool(
            level_routing_cfg.get('enable', False))
        self.gaussian_semantic_level_routing_weight = float(
            level_routing_cfg.get('weight', 0.15))
        self.gaussian_semantic_level_routing_max_bias_abs = float(
            level_routing_cfg.get('max_bias_abs', 0.75))
        self.gaussian_semantic_level_routing_target_assignment_source = str(
            level_routing_cfg.get('target_assignment_source',
                                  'routed')).lower()
        if self.gaussian_semantic_level_routing_target_assignment_source not in {
                'routed', 'pre_routing'}:
            raise ValueError(
                'level_routing.target_assignment_source must be '
                '"routed" or "pre_routing".')
        self.gaussian_semantic_level_routing_learnable_gate_enable = bool(
            level_routing_cfg.get('learnable_gate_enable', False))
        self.gaussian_semantic_level_routing_learnable_gate_init = float(
            level_routing_cfg.get('learnable_gate_init', 1.0))
        self.gaussian_semantic_level_routing_projection_loss_enable = bool(
            level_routing_cfg.get('projection_loss_enable', False))
        self.gaussian_semantic_level_routing_projection_loss_weight = float(
            level_routing_cfg.get('projection_loss_weight', 0.01))
        self.gaussian_semantic_level_routing_projection_loss_max_drop = float(
            level_routing_cfg.get('projection_loss_max_drop', 0.05))
        self.gaussian_semantic_level_routing_projection_loss_min_assign_metric = (
            float(level_routing_cfg.get(
                'projection_loss_min_assign_metric', 0.0)))
        self.gaussian_semantic_level_routing_log_stats = bool(
            level_routing_cfg.get('log_stats', False))
        self.gaussian_semantic_level_routing_level_log_areas = tuple(
            self._resolve_gaussian_semantic_level_log_areas(
                level_routing_cfg))
        self.gaussian_semantic_density_use_geometry_logprob = bool(
            density_cfg.get('use_geometry_logprob', False))
        ap_safe_cfg = dict(
            density_cfg.get('ap_safe_support_projection', {}) or {})
        if not ap_safe_cfg:
            ap_safe_cfg = dict(
                cfg.get('ap_safe_support_projection', {}) or {})
        self.gaussian_semantic_ap_safe_support_projection_enable = bool(
            ap_safe_cfg.get('enable', False))
        self.gaussian_semantic_ap_safe_support_projection_loss_weight = float(
            ap_safe_cfg.get('loss_weight', 0.015))
        self.gaussian_semantic_ap_safe_support_projection_min_logprob_gap = (
            float(ap_safe_cfg.get('min_logprob_gap', 1.5)))
        self.gaussian_semantic_ap_safe_support_projection_protect_margin = (
            float(ap_safe_cfg.get('protect_margin', 0.20)))
        self.gaussian_semantic_ap_safe_support_projection_rank_margin = float(
            ap_safe_cfg.get('rank_margin', 0.05))
        self.gaussian_semantic_ap_safe_support_projection_margin_temperature = (
            float(ap_safe_cfg.get('margin_temperature', 0.25)))
        self.gaussian_semantic_ap_safe_support_projection_support_temperature = (
            float(ap_safe_cfg.get('support_temperature', 1.0)))
        ap_safe_min_score = ap_safe_cfg.get('min_hardneg_score', 0.05)
        self.gaussian_semantic_ap_safe_support_projection_min_hardneg_score = (
            None if ap_safe_min_score is None
            else float(ap_safe_min_score))
        ap_safe_min_logit = ap_safe_cfg.get('min_hardneg_logit', None)
        self.gaussian_semantic_ap_safe_support_projection_min_hardneg_logit = (
            None if ap_safe_min_logit is None
            else float(ap_safe_min_logit))
        self.gaussian_semantic_ap_safe_support_projection_max_hardneg = int(
            ap_safe_cfg.get('max_hardneg', 1))
        self.gaussian_semantic_ap_safe_support_projection_max_budget = float(
            ap_safe_cfg.get('max_budget', 1.0))
        self.gaussian_semantic_ap_safe_support_projection_assign_metric_power = (
            float(ap_safe_cfg.get('assign_metric_power', 1.0)))
        self.gaussian_semantic_ap_safe_support_projection_detach_gt_logit = bool(
            ap_safe_cfg.get('detach_gt_logit', True))
        ap_safe_use_geometry = ap_safe_cfg.get('use_geometry_logprob', None)
        self.gaussian_semantic_ap_safe_support_projection_use_geometry_logprob = (
            None if ap_safe_use_geometry is None
            else bool(ap_safe_use_geometry))
        self.gaussian_semantic_ap_safe_support_projection_log_stats = bool(
            ap_safe_cfg.get('log_stats', False))
        ranking_projection_cfg = dict(
            density_cfg.get('ap_safe_ranking_projection', {}) or {})
        if not ranking_projection_cfg:
            ranking_projection_cfg = dict(
                cfg.get('ap_safe_ranking_projection', {}) or {})
        self.gaussian_semantic_ap_safe_ranking_projection_enable = bool(
            ranking_projection_cfg.get('enable', False))
        self.gaussian_semantic_ap_safe_ranking_projection_loss_weight = float(
            ranking_projection_cfg.get('loss_weight', 0.015))
        self.gaussian_semantic_ap_safe_ranking_projection_min_logprob_gap = (
            float(ranking_projection_cfg.get('min_logprob_gap', 1.5)))
        self.gaussian_semantic_ap_safe_ranking_projection_protect_margin = (
            float(ranking_projection_cfg.get('protect_margin', 0.20)))
        self.gaussian_semantic_ap_safe_ranking_projection_margin_temperature = (
            float(ranking_projection_cfg.get('margin_temperature', 0.25)))
        self.gaussian_semantic_ap_safe_ranking_projection_support_temperature = (
            float(ranking_projection_cfg.get('support_temperature', 1.0)))
        ranking_min_score = ranking_projection_cfg.get(
            'min_hardneg_score', 0.05)
        self.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_score = (
            None if ranking_min_score is None
            else float(ranking_min_score))
        ranking_min_logit = ranking_projection_cfg.get(
            'min_hardneg_logit', None)
        self.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_logit = (
            None if ranking_min_logit is None
            else float(ranking_min_logit))
        self.gaussian_semantic_ap_safe_ranking_projection_max_hardneg = int(
            ranking_projection_cfg.get('max_hardneg', 1))
        self.gaussian_semantic_ap_safe_ranking_projection_max_budget = float(
            ranking_projection_cfg.get('max_budget', 1.0))
        self.gaussian_semantic_ap_safe_ranking_projection_assign_metric_power = (
            float(ranking_projection_cfg.get('assign_metric_power', 1.0)))
        ranking_use_geometry = ranking_projection_cfg.get(
            'use_geometry_logprob', None)
        self.gaussian_semantic_ap_safe_ranking_projection_use_geometry_logprob = (
            None if ranking_use_geometry is None
            else bool(ranking_use_geometry))
        self.gaussian_semantic_ap_safe_ranking_projection_log_stats = bool(
            ranking_projection_cfg.get('log_stats', False))
        self.gaussian_semantic_density_log_stats = bool(
            density_cfg.get('log_stats', False))
        self.gaussian_semantic_geometry_enable = bool(
            cfg.get('geometry_support_enable', False)
            or self.gaussian_semantic_density_use_geometry_logprob)

        class_names = tuple(
            cfg.get('class_names')
            or tuple(str(idx) for idx in range(int(self.num_classes))))
        if len(class_names) < int(self.num_classes):
            class_names = class_names + tuple(
                str(idx)
                for idx in range(len(class_names), int(self.num_classes)))
        priors_csv = cfg.get('class_area_priors_csv')
        geometry_priors_csv = (
            cfg.get('class_geometry_priors_csv')
            or cfg.get('geometry_priors_csv')
            or priors_csv)
        z0_per_class = build_classwise_gaussian_parameter(
            scalar_value=self.gaussian_semantic_scale_z0,
            classwise_value=cfg.get('classwise_z0'),
            class_names=class_names,
            num_classes=int(self.num_classes),
            name='z0',
            min_value=0.0)
        beta_per_class = build_classwise_gaussian_parameter(
            scalar_value=self.gaussian_semantic_scale_beta,
            classwise_value=cfg.get('classwise_beta'),
            class_names=class_names,
            num_classes=int(self.num_classes),
            name='beta',
            min_value=0.0)

        log_mean = torch.zeros(int(self.num_classes), dtype=torch.float32)
        log_std = torch.ones(int(self.num_classes), dtype=torch.float32)
        valid = torch.zeros(int(self.num_classes), dtype=torch.bool)
        geometry_mean = torch.zeros(
            (int(self.num_classes), 2), dtype=torch.float32)
        geometry_std = torch.ones(
            (int(self.num_classes), 2), dtype=torch.float32)
        geometry_valid = torch.zeros(int(self.num_classes), dtype=torch.bool)
        self.gaussian_semantic_scale_class_names = tuple(class_names)
        self.gaussian_semantic_scale_debug = {
            'enable': self.gaussian_semantic_scale_enable,
            'num_classes': int(self.num_classes),
            'num_valid_priors': 0,
            'domain_mode': self.gaussian_semantic_scale_domain_mode,
            'gaussian_energy_mode': self.gaussian_semantic_scale_mode,
        }

        if self.gaussian_semantic_scale_enable:
            if not priors_csv:
                raise ValueError(
                    'gaussian_semantic_scale.enable=True requires '
                    'class_area_priors_csv')
            log_mean, log_std, valid, loaded_names = load_class_log_area_priors(
                priors_csv, class_names, int(self.num_classes))
            self.gaussian_semantic_scale_class_names = tuple(loaded_names)
            self.gaussian_semantic_scale_debug.update({
                'class_area_priors_csv': str(priors_csv),
                'z0': self.gaussian_semantic_scale_z0,
                'beta': self.gaussian_semantic_scale_beta,
                'classwise_z0_keys': sorted(
                    (cfg.get('classwise_z0') or {}).keys())
                if isinstance(cfg.get('classwise_z0'), dict) else [],
                'classwise_beta_keys': sorted(
                    (cfg.get('classwise_beta') or {}).keys())
                if isinstance(cfg.get('classwise_beta'), dict) else [],
                'adapter_hidden': int(cfg.get('adapter_hidden', 64)),
                'adapter_nonpositive_delta': (
                    self.gaussian_semantic_scale_adapter_nonpositive_delta),
                'adapter_max_delta_abs': (
                    self.gaussian_semantic_scale_adapter_max_delta_abs),
                'adapter_min_abs_z': (
                    self.gaussian_semantic_scale_adapter_min_abs_z),
                'adapter_min_score': (
                    self.gaussian_semantic_scale_adapter_min_score),
                'consistency_loss_enable': (
                    self.gaussian_semantic_consistency_enable),
                'consistency_loss_weight': (
                    self.gaussian_semantic_consistency_weight),
                'consistency_margin': (
                    self.gaussian_semantic_consistency_margin),
                'consistency_min_logprob_gap': (
                    self.gaussian_semantic_consistency_min_logprob_gap),
                'consistency_max_hardneg': (
                    self.gaussian_semantic_consistency_max_hardneg),
                'consistency_min_hardneg_logit': (
                    self.gaussian_semantic_consistency_min_hardneg_logit),
                'consistency_max_gt_abs_z': (
                    self.gaussian_semantic_consistency_max_gt_abs_z),
                'consistency_log_stats': (
                    self.gaussian_semantic_consistency_log_stats),
                'density_head_enable': (
                    self.gaussian_semantic_density_enable),
                'density_head_loss_weight': (
                    self.gaussian_semantic_density_loss_weight),
                'density_head_hidden_channels': (
                    self.gaussian_semantic_density_hidden_channels),
                'density_head_apply_logit_delta': (
                    self.gaussian_semantic_density_apply_logit_delta),
                'density_head_logit_delta_source': (
                    self.gaussian_semantic_density_logit_delta_source),
                'density_head_logprob_delta_beta': (
                    self.gaussian_semantic_density_logprob_delta_beta),
                'density_head_logprob_delta_threshold': (
                    self.gaussian_semantic_density_logprob_delta_threshold),
                'density_head_delta_loss_enable': (
                    self.gaussian_semantic_density_delta_loss_enable),
                'density_head_delta_loss_weight': (
                    self.gaussian_semantic_density_delta_loss_weight),
                'density_head_delta_loss_min_logprob_gap': (
                    self.gaussian_semantic_density_delta_loss_min_logprob_gap),
                'density_head_delta_loss_min_hardneg_score': (
                    self.gaussian_semantic_density_delta_loss_min_hardneg_score),
                'density_head_delta_loss_min_hardneg_logit': (
                    self.gaussian_semantic_density_delta_loss_min_hardneg_logit),
                'density_head_delta_loss_target_negative_delta': (
                    self.gaussian_semantic_density_delta_loss_target_negative_delta),
                'density_head_delta_loss_gt_keep_weight': (
                    self.gaussian_semantic_density_delta_loss_gt_keep_weight),
                'density_head_delta_loss_max_hardneg': (
                    self.gaussian_semantic_density_delta_loss_max_hardneg),
                'density_head_positive_delta_loss_enable': (
                    self.gaussian_semantic_density_positive_delta_loss_enable),
                'density_head_positive_delta_loss_weight': (
                    self.gaussian_semantic_density_positive_delta_loss_weight),
                'density_head_positive_delta_loss_target': (
                    self.gaussian_semantic_density_positive_delta_loss_target),
                'density_head_positive_delta_loss_min_gt_logprob': (
                    self.gaussian_semantic_density_positive_delta_loss_min_gt_logprob),
                'density_head_positive_delta_loss_min_gt_score': (
                    self.gaussian_semantic_density_positive_delta_loss_min_gt_score),
                'density_head_pair_margin_loss_enable': (
                    self.gaussian_semantic_density_pair_margin_loss_enable),
                'density_head_pair_margin_loss_weight': (
                    self.gaussian_semantic_density_pair_margin_loss_weight),
                'density_head_pair_margin_loss_margin': (
                    self.gaussian_semantic_density_pair_margin_loss_margin),
                'density_head_pair_margin_loss_min_logprob_gap': (
                    self.gaussian_semantic_density_pair_margin_loss_min_logprob_gap),
                'density_head_pair_margin_loss_min_hardneg_score': (
                    self.gaussian_semantic_density_pair_margin_loss_min_hardneg_score),
                'density_head_pair_margin_loss_max_hardneg': (
                    self.gaussian_semantic_density_pair_margin_loss_max_hardneg),
                'density_head_support_negative_loss_enable': (
                    self.gaussian_semantic_density_support_negative_loss_enable),
                'density_head_support_negative_loss_weight': (
                    self.gaussian_semantic_density_support_negative_loss_weight),
                'density_head_support_negative_loss_min_score': (
                    self.gaussian_semantic_density_support_negative_loss_min_score),
                'density_head_support_negative_loss_min_logprob_gap': (
                    self.gaussian_semantic_density_support_negative_loss_min_logprob_gap),
                'density_head_support_negative_loss_gamma': (
                    self.gaussian_semantic_density_support_negative_loss_gamma),
                'density_head_support_negative_loss_gap_scale': (
                    self.gaussian_semantic_density_support_negative_loss_gap_scale),
                'density_head_support_negative_loss_max_extra_weight': (
                    self.gaussian_semantic_density_support_negative_loss_max_extra_weight),
                'density_head_support_negative_loss_max_hardneg': (
                    self.gaussian_semantic_density_support_negative_loss_max_hardneg),
                'density_head_scale_consistency_loss_enable': (
                    self.gaussian_semantic_density_scale_consistency_loss_enable),
                'density_head_scale_consistency_loss_weight': (
                    self.gaussian_semantic_density_scale_consistency_loss_weight),
                'density_head_scale_consistency_loss_max_logprob_drop': (
                    self.gaussian_semantic_density_scale_consistency_loss_max_logprob_drop),
                'density_head_scale_consistency_loss_min_target_logprob': (
                    self.gaussian_semantic_density_scale_consistency_loss_min_target_logprob),
                'density_head_use_geometry_logprob': (
                    self.gaussian_semantic_density_use_geometry_logprob),
                'ap_safe_support_projection_enable': (
                    self.gaussian_semantic_ap_safe_support_projection_enable),
                'ap_safe_support_projection_loss_weight': (
                    self.gaussian_semantic_ap_safe_support_projection_loss_weight),
                'ap_safe_support_projection_min_logprob_gap': (
                    self.gaussian_semantic_ap_safe_support_projection_min_logprob_gap),
                'ap_safe_support_projection_protect_margin': (
                    self.gaussian_semantic_ap_safe_support_projection_protect_margin),
                'ap_safe_support_projection_rank_margin': (
                    self.gaussian_semantic_ap_safe_support_projection_rank_margin),
                'ap_safe_support_projection_margin_temperature': (
                    self.gaussian_semantic_ap_safe_support_projection_margin_temperature),
                'ap_safe_support_projection_support_temperature': (
                    self.gaussian_semantic_ap_safe_support_projection_support_temperature),
                'ap_safe_support_projection_min_hardneg_score': (
                    self.gaussian_semantic_ap_safe_support_projection_min_hardneg_score),
                'ap_safe_support_projection_min_hardneg_logit': (
                    self.gaussian_semantic_ap_safe_support_projection_min_hardneg_logit),
                'ap_safe_support_projection_max_hardneg': (
                    self.gaussian_semantic_ap_safe_support_projection_max_hardneg),
                'ap_safe_support_projection_max_budget': (
                    self.gaussian_semantic_ap_safe_support_projection_max_budget),
                'ap_safe_support_projection_assign_metric_power': (
                    self.gaussian_semantic_ap_safe_support_projection_assign_metric_power),
                'ap_safe_support_projection_detach_gt_logit': (
                    self.gaussian_semantic_ap_safe_support_projection_detach_gt_logit),
                'ap_safe_support_projection_use_geometry_logprob': (
                    self.gaussian_semantic_ap_safe_support_projection_use_geometry_logprob),
                'ap_safe_support_projection_log_stats': (
                    self.gaussian_semantic_ap_safe_support_projection_log_stats),
                'ap_safe_ranking_projection_enable': (
                    self.gaussian_semantic_ap_safe_ranking_projection_enable),
                'ap_safe_ranking_projection_loss_weight': (
                    self.gaussian_semantic_ap_safe_ranking_projection_loss_weight),
                'ap_safe_ranking_projection_min_logprob_gap': (
                    self.gaussian_semantic_ap_safe_ranking_projection_min_logprob_gap),
                'ap_safe_ranking_projection_protect_margin': (
                    self.gaussian_semantic_ap_safe_ranking_projection_protect_margin),
                'ap_safe_ranking_projection_margin_temperature': (
                    self.gaussian_semantic_ap_safe_ranking_projection_margin_temperature),
                'ap_safe_ranking_projection_support_temperature': (
                    self.gaussian_semantic_ap_safe_ranking_projection_support_temperature),
                'ap_safe_ranking_projection_min_hardneg_score': (
                    self.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_score),
                'ap_safe_ranking_projection_min_hardneg_logit': (
                    self.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_logit),
                'ap_safe_ranking_projection_max_hardneg': (
                    self.gaussian_semantic_ap_safe_ranking_projection_max_hardneg),
                'ap_safe_ranking_projection_max_budget': (
                    self.gaussian_semantic_ap_safe_ranking_projection_max_budget),
                'ap_safe_ranking_projection_assign_metric_power': (
                    self.gaussian_semantic_ap_safe_ranking_projection_assign_metric_power),
                'ap_safe_ranking_projection_use_geometry_logprob': (
                    self.gaussian_semantic_ap_safe_ranking_projection_use_geometry_logprob),
                'ap_safe_ranking_projection_log_stats': (
                    self.gaussian_semantic_ap_safe_ranking_projection_log_stats),
                'geometry_support_enable': (
                    self.gaussian_semantic_geometry_enable),
                'class_geometry_priors_csv': (
                    str(geometry_priors_csv) if geometry_priors_csv else ''),
                'density_head_log_stats': (
                    self.gaussian_semantic_density_log_stats),
                'num_valid_priors': int(valid.sum().item()),
                'valid_classes': [
                    self.gaussian_semantic_scale_class_names[idx]
                    for idx in torch.nonzero(
                        valid, as_tuple=False).view(-1).tolist()
                ],
            })
            if self.gaussian_semantic_geometry_enable:
                if not geometry_priors_csv:
                    raise ValueError(
                        'geometry_support_enable=True requires '
                        'class_geometry_priors_csv or class_area_priors_csv')
                (geometry_mean, geometry_std, geometry_valid,
                 _) = load_class_geometry_priors(
                    geometry_priors_csv, class_names, int(self.num_classes))
                self.gaussian_semantic_scale_debug.update({
                    'num_valid_geometry_priors': int(
                        geometry_valid.sum().item()),
                    'valid_geometry_classes': [
                        self.gaussian_semantic_scale_class_names[idx]
                        for idx in torch.nonzero(
                            geometry_valid, as_tuple=False).view(-1).tolist()
                    ],
                })
        level_routing_bias = torch.zeros(
            (len(self.gaussian_semantic_level_routing_level_log_areas),
             int(self.num_classes)),
            dtype=torch.float32)
        if (self.gaussian_semantic_scale_enable
                and self.gaussian_semantic_level_routing_enable
                and len(self.gaussian_semantic_level_routing_level_log_areas)
                > 0):
            level_refs = torch.tensor(
                self.gaussian_semantic_level_routing_level_log_areas,
                dtype=torch.float32)
            level_stats = gaussian_log_area_stats(
                level_refs, log_mean, log_std, valid)
            log_prob = level_stats['log_prob']
            level_valid = level_stats['valid']
            masked_log_prob = torch.where(
                level_valid, log_prob,
                torch.full_like(log_prob, -1.0e9))
            best_log_prob = masked_log_prob.max(dim=0).values
            relative_log_prob = log_prob - best_log_prob[None, :]
            level_routing_bias = torch.where(
                level_valid, relative_log_prob,
                torch.zeros_like(relative_log_prob))
            level_routing_bias = (
                level_routing_bias
                * self.gaussian_semantic_level_routing_weight)
            level_routing_bias = torch.clamp(level_routing_bias, max=0.0)
            if self.gaussian_semantic_level_routing_max_bias_abs > 0:
                level_routing_bias = torch.clamp(
                    level_routing_bias,
                    min=-self.gaussian_semantic_level_routing_max_bias_abs)
        self.gaussian_semantic_scale_debug.update({
            'level_routing_enable': (
                self.gaussian_semantic_level_routing_enable),
            'level_routing_weight': (
                self.gaussian_semantic_level_routing_weight),
            'level_routing_max_bias_abs': (
                self.gaussian_semantic_level_routing_max_bias_abs),
            'level_routing_target_assignment_source': (
                self.gaussian_semantic_level_routing_target_assignment_source),
            'level_routing_learnable_gate_enable': (
                self.gaussian_semantic_level_routing_learnable_gate_enable),
            'level_routing_learnable_gate_init': (
                self.gaussian_semantic_level_routing_learnable_gate_init),
            'level_routing_projection_loss_enable': (
                self.gaussian_semantic_level_routing_projection_loss_enable),
            'level_routing_projection_loss_weight': (
                self.gaussian_semantic_level_routing_projection_loss_weight),
            'level_routing_level_log_areas': list(
                self.gaussian_semantic_level_routing_level_log_areas),
            'level_routing_num_levels': int(level_routing_bias.shape[0]),
        })

        self.register_buffer(
            'gaussian_semantic_log_area_mean', log_mean, persistent=False)
        self.register_buffer(
            'gaussian_semantic_log_area_std', log_std, persistent=False)
        self.register_buffer(
            'gaussian_semantic_valid_mask', valid, persistent=False)
        self.register_buffer(
            'gaussian_semantic_geometry_mean',
            geometry_mean,
            persistent=False)
        self.register_buffer(
            'gaussian_semantic_geometry_std',
            geometry_std,
            persistent=False)
        self.register_buffer(
            'gaussian_semantic_geometry_valid_mask',
            geometry_valid,
            persistent=False)
        self.register_buffer(
            'gaussian_semantic_scale_z0_per_class',
            z0_per_class,
            persistent=False)
        self.register_buffer(
            'gaussian_semantic_scale_beta_per_class',
            beta_per_class,
            persistent=False)
        self.register_buffer(
            'gaussian_semantic_level_routing_bias',
            level_routing_bias,
            persistent=False)
        if self.gaussian_semantic_level_routing_learnable_gate_enable:
            init_gate = min(
                max(self.gaussian_semantic_level_routing_learnable_gate_init,
                    1e-4), 1.0 - 1e-4)
            gate_logit = math.log(init_gate / (1.0 - init_gate))
            self.gaussian_semantic_level_routing_gate_logit = nn.Parameter(
                torch.tensor(gate_logit, dtype=torch.float32))
        self.gaussian_semantic_scale_adapter = None
        if (self.gaussian_semantic_scale_enable
                and self.gaussian_semantic_scale_mode == 'logit_adapter'):
            self.gaussian_semantic_scale_adapter = GaussianScaleLogitAdapter(
                hidden=int(cfg.get('adapter_hidden', 64)),
                nonpositive_delta=(
                    self.gaussian_semantic_scale_adapter_nonpositive_delta))
            adapter_checkpoint = cfg.get('adapter_checkpoint')
            if adapter_checkpoint:
                self._load_gaussian_semantic_scale_adapter_checkpoint(
                    adapter_checkpoint,
                    strict=bool(cfg.get('adapter_checkpoint_strict', True)))
        self.gaussian_semantic_density_head = None
        if (self.gaussian_semantic_scale_enable
                and self.gaussian_semantic_density_enable):
            self.gaussian_semantic_density_head = (
                GaussianSemanticScaleDensityHead(
                    in_channels=int(self.cls_out_channels),
                    num_classes=int(self.num_classes),
                    hidden_channels=(
                        self.gaussian_semantic_density_hidden_channels),
                    mean_residual_scale=(
                        self.gaussian_semantic_density_mean_residual_scale),
                    log_std_delta_limit=(
                        self.gaussian_semantic_density_log_std_delta_limit),
                    max_delta_abs=self.gaussian_semantic_density_max_delta_abs,
                    nonpositive_delta=(
                        self.gaussian_semantic_density_nonpositive_delta)))
        self._last_gaussian_semantic_scale_debug = dict(
            self.gaussian_semantic_scale_debug)
        self._last_gaussian_semantic_consistency_debug = {
            'enable': self.gaussian_semantic_consistency_enable,
            'loss_weight': self.gaussian_semantic_consistency_weight,
            'num_pos': 0,
            'num_hardneg': 0,
            'num_candidate_pairs': 0,
            'num_pos_with_hardneg': 0,
            'num_ap_sensitive_pos': 0,
            'active_violation_count': 0,
            'hardneg_per_pos': 0.0,
            'active_violation_rate': 0.0,
        }
        self._last_gaussian_semantic_density_debug = {
            'enable': self.gaussian_semantic_density_enable,
            'loss_weight': self.gaussian_semantic_density_loss_weight,
            'apply_logit_delta': (
                self.gaussian_semantic_density_apply_logit_delta),
            'logit_delta_source': (
                self.gaussian_semantic_density_logit_delta_source),
            'raw_loss': 0.0,
            'weighted_loss': 0.0,
            'num_pos': 0,
            'mean_gt_log_prob': 0.0,
            'delta_loss_enable': (
                self.gaussian_semantic_density_delta_loss_enable),
            'delta_loss_weight': (
                self.gaussian_semantic_density_delta_loss_weight),
            'delta_raw_loss': 0.0,
            'delta_weighted_loss': 0.0,
            'delta_num_hardneg': 0,
            'delta_candidate_pairs': 0,
            'delta_pos_with_hardneg': 0,
            'delta_score_gated_pairs': 0,
            'delta_mean_logprob_gap': 0.0,
            'delta_mean_hardneg_delta': 0.0,
            'positive_delta_loss_enable': (
                self.gaussian_semantic_density_positive_delta_loss_enable),
            'positive_delta_loss_weight': (
                self.gaussian_semantic_density_positive_delta_loss_weight),
            'positive_delta_raw_loss': 0.0,
            'positive_delta_weighted_loss': 0.0,
            'positive_delta_supported_pos': 0,
            'positive_delta_score_gated_pos': 0,
            'positive_delta_mean_gt_log_prob': 0.0,
            'positive_delta_mean_gt_delta': 0.0,
            'pair_margin_loss_enable': (
                self.gaussian_semantic_density_pair_margin_loss_enable),
            'pair_margin_loss_weight': (
                self.gaussian_semantic_density_pair_margin_loss_weight),
            'pair_margin_raw_loss': 0.0,
            'pair_margin_weighted_loss': 0.0,
            'pair_margin_num_hardneg': 0,
            'pair_margin_candidate_pairs': 0,
            'pair_margin_pos_with_hardneg': 0,
            'pair_margin_score_gated_pairs': 0,
            'pair_margin_active_violation_count': 0,
            'pair_margin_mean_logprob_gap': 0.0,
            'pair_margin_mean_pair_margin': 0.0,
            'support_negative_loss_enable': (
                self.gaussian_semantic_density_support_negative_loss_enable),
            'support_negative_loss_weight': (
                self.gaussian_semantic_density_support_negative_loss_weight),
            'support_negative_raw_loss': 0.0,
            'support_negative_weighted_loss': 0.0,
            'support_negative_negative_locations': 0,
            'support_negative_positive_locations': 0,
            'support_negative_candidate_pairs': 0,
            'support_negative_score_gated_pairs': 0,
            'support_negative_selected_pairs': 0,
            'support_negative_mean_score': 0.0,
            'support_negative_mean_logprob_gap': 0.0,
            'support_negative_mean_pair_weight': 0.0,
            'scale_consistency_loss_enable': (
                self.gaussian_semantic_density_scale_consistency_loss_enable),
            'scale_consistency_loss_weight': (
                self.gaussian_semantic_density_scale_consistency_loss_weight),
            'scale_consistency_raw_loss': 0.0,
            'scale_consistency_weighted_loss': 0.0,
            'scale_consistency_num_pos': 0,
            'scale_consistency_supported_pos': 0,
            'scale_consistency_active_violation_count': 0,
            'scale_consistency_mean_target_log_prob': 0.0,
            'scale_consistency_mean_pred_log_prob': 0.0,
            'scale_consistency_mean_logprob_drop': 0.0,
            'geometry_logprob_used': False,
        }
        self._last_gaussian_semantic_ap_safe_support_projection_debug = {
            'enable': (
                self.gaussian_semantic_ap_safe_support_projection_enable),
            'loss_weight': (
                self.gaussian_semantic_ap_safe_support_projection_loss_weight),
            'raw_loss': 0.0,
            'weighted_loss': 0.0,
            'num_pos': 0,
            'num_supported_pos': 0,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_pair_weight': 0.0,
            'mean_selected_score': 0.0,
            'geometry_logprob_used': False,
        }
        self._last_gaussian_semantic_ap_safe_ranking_projection_debug = {
            'enable': (
                self.gaussian_semantic_ap_safe_ranking_projection_enable),
            'loss_weight': (
                self.gaussian_semantic_ap_safe_ranking_projection_loss_weight),
            'raw_loss': 0.0,
            'weighted_loss': 0.0,
            'num_pos': 0,
            'num_supported_pos': 0,
            'num_candidate_pairs': 0,
            'num_score_gated_pairs': 0,
            'num_hardneg': 0,
            'num_pos_with_hardneg': 0,
            'active_projection_count': 0,
            'protected_positive_count': 0,
            'mean_logprob_gap': 0.0,
            'mean_gt_margin': 0.0,
            'mean_safe_budget': 0.0,
            'mean_projection_delta': 0.0,
            'mean_pair_weight': 0.0,
            'mean_selected_score': 0.0,
            'geometry_logprob_used': False,
        }

    def _resolve_gaussian_semantic_level_log_areas(
            self, level_routing_cfg: dict) -> tuple[float, ...]:
        """Resolve FPN-level reference log-areas for semantic routing."""
        explicit_log_areas = level_routing_cfg.get('level_log_areas')
        if explicit_log_areas is not None:
            return tuple(float(value) for value in explicit_log_areas)

        ref_sizes = level_routing_cfg.get('level_ref_sizes')
        if ref_sizes is None:
            stride_scale = float(level_routing_cfg.get(
                'stride_ref_scale', 8.0))
            strides = getattr(
                getattr(self, 'prior_generator', None), 'strides', ())
            ref_sizes = []
            for stride in strides:
                if isinstance(stride, (tuple, list)):
                    stride_value = float(stride[0])
                else:
                    stride_value = float(stride)
                ref_sizes.append(stride_value * stride_scale)
        return tuple(
            2.0 * math.log(max(float(size), 1e-6))
            for size in ref_sizes)

    def _apply_gaussian_semantic_level_routing(
            self, cls_score: Tensor, level_idx: int) -> Tensor:
        """Apply non-positive class-wise Gaussian FPN-level routing bias."""
        if (not getattr(self, 'gaussian_semantic_level_routing_enable', False)
                or cls_score.numel() == 0):
            return cls_score
        bias = getattr(self, 'gaussian_semantic_level_routing_bias', None)
        if (bias is None or bias.numel() == 0 or level_idx < 0
                or level_idx >= int(bias.shape[0])):
            return cls_score

        if cls_score.ndim == 4:
            num_classes = min(int(cls_score.shape[1]), int(bias.shape[1]))
            level_bias = bias[level_idx, :num_classes].to(
                device=cls_score.device, dtype=cls_score.dtype)
            if getattr(
                    self,
                    'gaussian_semantic_level_routing_learnable_gate_enable',
                    False):
                gate_logit = getattr(
                    self, 'gaussian_semantic_level_routing_gate_logit', None)
                if gate_logit is not None:
                    level_bias = (
                        level_bias
                        * gate_logit.to(
                            device=cls_score.device,
                            dtype=cls_score.dtype).sigmoid())
            routed = cls_score.clone()
            routed[:, :num_classes] = (
                routed[:, :num_classes]
                + level_bias.view(1, num_classes, 1, 1))
            return routed
        if cls_score.ndim == 3:
            num_classes = min(int(cls_score.shape[-1]), int(bias.shape[1]))
            level_bias = bias[level_idx, :num_classes].to(
                device=cls_score.device, dtype=cls_score.dtype)
            if getattr(
                    self,
                    'gaussian_semantic_level_routing_learnable_gate_enable',
                    False):
                gate_logit = getattr(
                    self, 'gaussian_semantic_level_routing_gate_logit', None)
                if gate_logit is not None:
                    level_bias = (
                        level_bias
                        * gate_logit.to(
                            device=cls_score.device,
                            dtype=cls_score.dtype).sigmoid())
            routed = cls_score.clone()
            routed[..., :num_classes] = (
                routed[..., :num_classes]
                + level_bias.view(1, 1, num_classes))
            return routed
        if cls_score.ndim == 2:
            num_classes = min(int(cls_score.shape[-1]), int(bias.shape[1]))
            level_bias = bias[level_idx, :num_classes].to(
                device=cls_score.device, dtype=cls_score.dtype)
            if getattr(
                    self,
                    'gaussian_semantic_level_routing_learnable_gate_enable',
                    False):
                gate_logit = getattr(
                    self, 'gaussian_semantic_level_routing_gate_logit', None)
                if gate_logit is not None:
                    level_bias = (
                        level_bias
                        * gate_logit.to(
                            device=cls_score.device,
                            dtype=cls_score.dtype).sigmoid())
            routed = cls_score.clone()
            routed[:, :num_classes] = (
                routed[:, :num_classes] + level_bias.view(1, num_classes))
            return routed
        return cls_score

    def _loss_gaussian_semantic_level_routing_projection(
            self,
            raw_cls_scores: List[Tensor],
            routed_cls_scores: List[Tensor],
            labels_list: List[Tensor],
            assign_metrics_list: List[Tensor]) -> Tensor:
        zero = routed_cls_scores[0].sum() * 0.0
        if (not getattr(self, 'gaussian_semantic_level_routing_enable', False)
                or not getattr(
                    self,
                    'gaussian_semantic_level_routing_projection_loss_enable',
                    False)):
            self._last_gaussian_semantic_level_routing_projection_debug = {
                'enable': bool(getattr(
                    self,
                    'gaussian_semantic_level_routing_projection_loss_enable',
                    False)),
                'loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_level_routing_projection_loss_weight',
                    0.0)),
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'num_pos': 0,
                'active_violation_count': 0,
                'mean_positive_logit_drop': 0.0,
            }
            return zero

        losses = []
        total_pos = 0
        total_active = 0
        drop_sum = 0.0
        max_drop = float(getattr(
            self,
            'gaussian_semantic_level_routing_projection_loss_max_drop',
            0.05))
        min_assign_metric = float(getattr(
            self,
            'gaussian_semantic_level_routing_projection_loss_min_assign_metric',
            0.0))
        for raw_score, routed_score, labels, assign_metrics in zip(
                raw_cls_scores, routed_cls_scores, labels_list,
                assign_metrics_list):
            if raw_score.numel() == 0 or routed_score.numel() == 0:
                continue
            flat_raw = raw_score.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels).contiguous()
            flat_routed = routed_score.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels).contiguous()
            num_classes = min(
                int(flat_raw.shape[-1]), int(flat_routed.shape[-1]),
                int(self.cls_out_channels))
            flat_labels = labels.reshape(-1).to(device=flat_raw.device)
            flat_assign = assign_metrics.reshape(-1).to(
                device=flat_raw.device, dtype=flat_raw.dtype)
            pos_mask = (
                (flat_labels >= 0)
                & (flat_labels < num_classes)
                & (flat_assign > min_assign_metric))
            if not bool(pos_mask.any()):
                continue
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
            pos_labels = flat_labels[pos_idx].long()
            raw_gt = flat_raw[pos_idx, pos_labels]
            routed_gt = flat_routed[pos_idx, pos_labels]
            positive_drop = raw_gt.detach() - routed_gt
            violation = F.relu(positive_drop - max_drop)
            pos_weights = flat_assign[pos_idx].clamp(min=0.0)
            denom = pos_weights.sum().clamp(min=1.0)
            losses.append((violation * pos_weights).sum() / denom)
            total_pos += int(pos_idx.numel())
            total_active += int(
                (violation.detach() > 0).sum().cpu().item())
            drop_sum += float(
                positive_drop.detach().sum().cpu().item())

        if not losses:
            self._last_gaussian_semantic_level_routing_projection_debug = {
                'enable': True,
                'loss_weight': (
                    self.gaussian_semantic_level_routing_projection_loss_weight),
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'num_pos': 0,
                'active_violation_count': 0,
                'mean_positive_logit_drop': 0.0,
            }
            return zero
        loss = torch.stack(losses).mean()
        weighted = (
            loss
            * self.gaussian_semantic_level_routing_projection_loss_weight)
        self._last_gaussian_semantic_level_routing_projection_debug = {
            'enable': True,
            'loss_weight': (
                self.gaussian_semantic_level_routing_projection_loss_weight),
            'raw_loss': float(loss.detach().cpu().item()),
            'weighted_loss': float(weighted.detach().cpu().item()),
            'num_pos': int(total_pos),
            'active_violation_count': int(total_active),
            'mean_positive_logit_drop': drop_sum / max(float(total_pos), 1.0),
        }
        return weighted

    def _loss_gaussian_semantic_ap_safe_support_projection(
            self,
            cls_scores: List[Tensor],
            labels_list: List[Tensor],
            bbox_targets_list: List[Tensor],
            assign_metrics_list: List[Tensor]) -> Tensor:
        zero = cls_scores[0].sum() * 0.0
        if not getattr(
                self,
                'gaussian_semantic_ap_safe_support_projection_enable',
                False):
            self._last_gaussian_semantic_ap_safe_support_projection_debug = {
                'enable': False,
                'loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_ap_safe_support_projection_loss_weight',
                    0.0)),
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'num_pos': 0,
                'num_supported_pos': 0,
                'num_candidate_pairs': 0,
                'num_score_gated_pairs': 0,
                'num_hardneg': 0,
                'num_pos_with_hardneg': 0,
                'active_projection_count': 0,
                'mean_logprob_gap': 0.0,
                'mean_gt_margin': 0.0,
                'mean_safe_budget': 0.0,
                'mean_pair_weight': 0.0,
                'mean_selected_score': 0.0,
                'geometry_logprob_used': False,
            }
            return zero

        use_geometry = getattr(
            self,
            'gaussian_semantic_ap_safe_support_projection_use_geometry_logprob',
            None)
        if use_geometry is None:
            use_geometry = bool(getattr(
                self, 'gaussian_semantic_density_use_geometry_logprob',
                False))

        losses = []
        total_pos = 0
        total_supported_pos = 0
        total_candidate_pairs = 0
        total_score_gated_pairs = 0
        total_hardneg = 0
        total_pos_with_hardneg = 0
        total_active = 0
        logprob_gap_sum = 0.0
        gt_margin_sum = 0.0
        safe_budget_sum = 0.0
        pair_weight_sum = 0.0
        selected_score_sum = 0.0
        geometry_logprob_used = False

        for cls_score, labels, bbox_targets, assign_metrics in zip(
                cls_scores, labels_list, bbox_targets_list,
                assign_metrics_list):
            flat_boxes = bbox_targets.reshape(
                -1, bbox_targets.shape[-1]).to(
                    device=cls_score.device, dtype=cls_score.dtype)
            if flat_boxes.numel() == 0 or flat_boxes.shape[-1] < 4:
                continue
            area = (flat_boxes[:, 2].abs() * flat_boxes[:, 3].abs()).clamp(
                min=1e-6)
            if (use_geometry
                    and getattr(self, 'gaussian_semantic_geometry_enable',
                                False)):
                long_side = torch.maximum(
                    flat_boxes[:, 2].abs(), flat_boxes[:, 3].abs())
                short_side = torch.minimum(
                    flat_boxes[:, 2].abs(), flat_boxes[:, 3].abs()).clamp(
                        min=1e-6)
                geometry = torch.stack(
                    [torch.log(area), torch.log(long_side / short_side)],
                    dim=-1)
                support_stats = gaussian_geometry_support_stats(
                    geometry=geometry,
                    means=self.gaussian_semantic_geometry_mean.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    stds=self.gaussian_semantic_geometry_std.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    valid_mask=self.gaussian_semantic_geometry_valid_mask.to(
                        device=cls_score.device))
                support_valid_mask = self.gaussian_semantic_geometry_valid_mask
                geometry_logprob_used = True
            else:
                support_stats = gaussian_log_area_stats(
                    torch.log(area),
                    self.gaussian_semantic_log_area_mean.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    self.gaussian_semantic_log_area_std.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    self.gaussian_semantic_valid_mask.to(
                        device=cls_score.device))
                support_valid_mask = self.gaussian_semantic_valid_mask

            loss, debug = gaussian_ap_safe_support_projection_loss(
                cls_logits=cls_score.permute(0, 2, 3, 1).reshape(
                    -1, self.cls_out_channels).contiguous(),
                labels=labels.reshape(-1),
                log_prob=support_stats['log_prob'],
                valid_mask=support_valid_mask,
                assign_metrics=assign_metrics.reshape(-1),
                min_logprob_gap=(
                    self.gaussian_semantic_ap_safe_support_projection_min_logprob_gap),
                protect_margin=(
                    self.gaussian_semantic_ap_safe_support_projection_protect_margin),
                rank_margin=(
                    self.gaussian_semantic_ap_safe_support_projection_rank_margin),
                margin_temperature=(
                    self.gaussian_semantic_ap_safe_support_projection_margin_temperature),
                support_temperature=(
                    self.gaussian_semantic_ap_safe_support_projection_support_temperature),
                min_hardneg_score=(
                    self.gaussian_semantic_ap_safe_support_projection_min_hardneg_score),
                min_hardneg_logit=(
                    self.gaussian_semantic_ap_safe_support_projection_min_hardneg_logit),
                max_hardneg=(
                    self.gaussian_semantic_ap_safe_support_projection_max_hardneg),
                max_budget=(
                    self.gaussian_semantic_ap_safe_support_projection_max_budget),
                assign_metric_power=(
                    self.gaussian_semantic_ap_safe_support_projection_assign_metric_power),
                detach_gt_logit=(
                    self.gaussian_semantic_ap_safe_support_projection_detach_gt_logit))
            losses.append(loss)
            hardneg = int(debug.get('num_hardneg', 0))
            total_pos += int(debug.get('num_pos', 0))
            total_supported_pos += int(debug.get('num_supported_pos', 0))
            total_candidate_pairs += int(debug.get('num_candidate_pairs', 0))
            total_score_gated_pairs += int(
                debug.get('num_score_gated_pairs', 0))
            total_hardneg += hardneg
            total_pos_with_hardneg += int(
                debug.get('num_pos_with_hardneg', 0))
            total_active += int(debug.get('active_projection_count', 0))
            logprob_gap_sum += (
                float(debug.get('mean_logprob_gap', 0.0)) * hardneg)
            gt_margin_sum += (
                float(debug.get('mean_gt_margin', 0.0)) * hardneg)
            safe_budget_sum += (
                float(debug.get('mean_safe_budget', 0.0)) * hardneg)
            pair_weight_sum += (
                float(debug.get('mean_pair_weight', 0.0)) * hardneg)
            selected_score_sum += (
                float(debug.get('mean_selected_score', 0.0)) * hardneg)

        if not losses:
            self._last_gaussian_semantic_ap_safe_support_projection_debug = {
                'enable': True,
                'loss_weight': (
                    self.gaussian_semantic_ap_safe_support_projection_loss_weight),
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'num_pos': 0,
                'num_supported_pos': 0,
                'num_candidate_pairs': 0,
                'num_score_gated_pairs': 0,
                'num_hardneg': 0,
                'num_pos_with_hardneg': 0,
                'active_projection_count': 0,
                'mean_logprob_gap': 0.0,
                'mean_gt_margin': 0.0,
                'mean_safe_budget': 0.0,
                'mean_pair_weight': 0.0,
                'mean_selected_score': 0.0,
                'geometry_logprob_used': bool(geometry_logprob_used),
            }
            return zero

        loss = torch.stack(losses).mean()
        weighted = (
            loss
            * self.gaussian_semantic_ap_safe_support_projection_loss_weight)
        self._last_gaussian_semantic_ap_safe_support_projection_debug = {
            'enable': True,
            'loss_weight': (
                self.gaussian_semantic_ap_safe_support_projection_loss_weight),
            'raw_loss': float(loss.detach().cpu().item()),
            'weighted_loss': float(weighted.detach().cpu().item()),
            'num_pos': int(total_pos),
            'num_supported_pos': int(total_supported_pos),
            'num_candidate_pairs': int(total_candidate_pairs),
            'num_score_gated_pairs': int(total_score_gated_pairs),
            'num_hardneg': int(total_hardneg),
            'num_pos_with_hardneg': int(total_pos_with_hardneg),
            'active_projection_count': int(total_active),
            'mean_logprob_gap': (
                logprob_gap_sum / max(float(total_hardneg), 1.0)),
            'mean_gt_margin': (
                gt_margin_sum / max(float(total_hardneg), 1.0)),
            'mean_safe_budget': (
                safe_budget_sum / max(float(total_hardneg), 1.0)),
            'mean_pair_weight': (
                pair_weight_sum / max(float(total_hardneg), 1.0)),
            'mean_selected_score': (
                selected_score_sum / max(float(total_hardneg), 1.0)),
            'geometry_logprob_used': bool(geometry_logprob_used),
        }
        return weighted

    def _loss_gaussian_semantic_ap_safe_ranking_projection(
            self,
            cls_scores: List[Tensor],
            labels_list: List[Tensor],
            bbox_targets_list: List[Tensor],
            assign_metrics_list: List[Tensor]) -> Tensor:
        zero = cls_scores[0].sum() * 0.0
        if not getattr(
                self,
                'gaussian_semantic_ap_safe_ranking_projection_enable',
                False):
            self._last_gaussian_semantic_ap_safe_ranking_projection_debug = {
                'enable': False,
                'loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_ap_safe_ranking_projection_loss_weight',
                    0.0)),
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'num_pos': 0,
                'num_supported_pos': 0,
                'num_candidate_pairs': 0,
                'num_score_gated_pairs': 0,
                'num_hardneg': 0,
                'num_pos_with_hardneg': 0,
                'active_projection_count': 0,
                'protected_positive_count': 0,
                'mean_logprob_gap': 0.0,
                'mean_gt_margin': 0.0,
                'mean_safe_budget': 0.0,
                'mean_projection_delta': 0.0,
                'mean_pair_weight': 0.0,
                'mean_selected_score': 0.0,
                'geometry_logprob_used': False,
            }
            return zero

        use_geometry = getattr(
            self,
            'gaussian_semantic_ap_safe_ranking_projection_use_geometry_logprob',
            None)
        if use_geometry is None:
            use_geometry = bool(getattr(
                self, 'gaussian_semantic_density_use_geometry_logprob',
                False))

        losses = []
        total_pos = 0
        total_supported_pos = 0
        total_candidate_pairs = 0
        total_score_gated_pairs = 0
        total_hardneg = 0
        total_pos_with_hardneg = 0
        total_active = 0
        total_protected = 0
        logprob_gap_sum = 0.0
        gt_margin_sum = 0.0
        safe_budget_sum = 0.0
        projection_delta_sum = 0.0
        pair_weight_sum = 0.0
        selected_score_sum = 0.0
        geometry_logprob_used = False

        for cls_score, labels, bbox_targets, assign_metrics in zip(
                cls_scores, labels_list, bbox_targets_list,
                assign_metrics_list):
            flat_boxes = bbox_targets.reshape(
                -1, bbox_targets.shape[-1]).to(
                    device=cls_score.device, dtype=cls_score.dtype)
            if flat_boxes.numel() == 0 or flat_boxes.shape[-1] < 4:
                continue
            area = (flat_boxes[:, 2].abs() * flat_boxes[:, 3].abs()).clamp(
                min=1e-6)
            if (use_geometry
                    and getattr(self, 'gaussian_semantic_geometry_enable',
                                False)):
                long_side = torch.maximum(
                    flat_boxes[:, 2].abs(), flat_boxes[:, 3].abs())
                short_side = torch.minimum(
                    flat_boxes[:, 2].abs(), flat_boxes[:, 3].abs()).clamp(
                        min=1e-6)
                geometry = torch.stack(
                    [torch.log(area), torch.log(long_side / short_side)],
                    dim=-1)
                support_stats = gaussian_geometry_support_stats(
                    geometry=geometry,
                    means=self.gaussian_semantic_geometry_mean.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    stds=self.gaussian_semantic_geometry_std.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    valid_mask=self.gaussian_semantic_geometry_valid_mask.to(
                        device=cls_score.device))
                support_valid_mask = self.gaussian_semantic_geometry_valid_mask
                geometry_logprob_used = True
            else:
                support_stats = gaussian_log_area_stats(
                    torch.log(area),
                    self.gaussian_semantic_log_area_mean.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    self.gaussian_semantic_log_area_std.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    self.gaussian_semantic_valid_mask.to(
                        device=cls_score.device))
                support_valid_mask = self.gaussian_semantic_valid_mask

            loss, _, debug = (
                gaussian_ap_constrained_support_ranking_projection_loss(
                    cls_logits=cls_score.permute(0, 2, 3, 1).reshape(
                        -1, self.cls_out_channels).contiguous(),
                    labels=labels.reshape(-1),
                    log_prob=support_stats['log_prob'],
                    valid_mask=support_valid_mask,
                    assign_metrics=assign_metrics.reshape(-1),
                    min_logprob_gap=(
                        self.gaussian_semantic_ap_safe_ranking_projection_min_logprob_gap),
                    protect_margin=(
                        self.gaussian_semantic_ap_safe_ranking_projection_protect_margin),
                    margin_temperature=(
                        self.gaussian_semantic_ap_safe_ranking_projection_margin_temperature),
                    support_temperature=(
                        self.gaussian_semantic_ap_safe_ranking_projection_support_temperature),
                    min_hardneg_score=(
                        self.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_score),
                    min_hardneg_logit=(
                        self.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_logit),
                    max_hardneg=(
                        self.gaussian_semantic_ap_safe_ranking_projection_max_hardneg),
                    max_budget=(
                        self.gaussian_semantic_ap_safe_ranking_projection_max_budget),
                    assign_metric_power=(
                        self.gaussian_semantic_ap_safe_ranking_projection_assign_metric_power)))
            losses.append(loss)
            hardneg = int(debug.get('num_hardneg', 0))
            total_pos += int(debug.get('num_pos', 0))
            total_supported_pos += int(debug.get('num_supported_pos', 0))
            total_candidate_pairs += int(debug.get('num_candidate_pairs', 0))
            total_score_gated_pairs += int(
                debug.get('num_score_gated_pairs', 0))
            total_hardneg += hardneg
            total_pos_with_hardneg += int(
                debug.get('num_pos_with_hardneg', 0))
            total_active += int(debug.get('active_projection_count', 0))
            total_protected += int(debug.get('protected_positive_count', 0))
            logprob_gap_sum += (
                float(debug.get('mean_logprob_gap', 0.0)) * hardneg)
            gt_margin_sum += (
                float(debug.get('mean_gt_margin', 0.0)) * hardneg)
            safe_budget_sum += (
                float(debug.get('mean_safe_budget', 0.0)) * hardneg)
            projection_delta_sum += (
                float(debug.get('mean_projection_delta', 0.0)) * hardneg)
            pair_weight_sum += (
                float(debug.get('mean_pair_weight', 0.0)) * hardneg)
            selected_score_sum += (
                float(debug.get('mean_selected_score', 0.0)) * hardneg)

        if not losses:
            self._last_gaussian_semantic_ap_safe_ranking_projection_debug = {
                'enable': True,
                'loss_weight': (
                    self.gaussian_semantic_ap_safe_ranking_projection_loss_weight),
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'num_pos': 0,
                'num_supported_pos': 0,
                'num_candidate_pairs': 0,
                'num_score_gated_pairs': 0,
                'num_hardneg': 0,
                'num_pos_with_hardneg': 0,
                'active_projection_count': 0,
                'protected_positive_count': 0,
                'mean_logprob_gap': 0.0,
                'mean_gt_margin': 0.0,
                'mean_safe_budget': 0.0,
                'mean_projection_delta': 0.0,
                'mean_pair_weight': 0.0,
                'mean_selected_score': 0.0,
                'geometry_logprob_used': bool(geometry_logprob_used),
            }
            return zero

        loss = torch.stack(losses).mean()
        weighted = (
            loss
            * self.gaussian_semantic_ap_safe_ranking_projection_loss_weight)
        self._last_gaussian_semantic_ap_safe_ranking_projection_debug = {
            'enable': True,
            'loss_weight': (
                self.gaussian_semantic_ap_safe_ranking_projection_loss_weight),
            'raw_loss': float(loss.detach().cpu().item()),
            'weighted_loss': float(weighted.detach().cpu().item()),
            'num_pos': int(total_pos),
            'num_supported_pos': int(total_supported_pos),
            'num_candidate_pairs': int(total_candidate_pairs),
            'num_score_gated_pairs': int(total_score_gated_pairs),
            'num_hardneg': int(total_hardneg),
            'num_pos_with_hardneg': int(total_pos_with_hardneg),
            'active_projection_count': int(total_active),
            'protected_positive_count': int(total_protected),
            'mean_logprob_gap': (
                logprob_gap_sum / max(float(total_hardneg), 1.0)),
            'mean_gt_margin': (
                gt_margin_sum / max(float(total_hardneg), 1.0)),
            'mean_safe_budget': (
                safe_budget_sum / max(float(total_hardneg), 1.0)),
            'mean_projection_delta': (
                projection_delta_sum / max(float(total_hardneg), 1.0)),
            'mean_pair_weight': (
                pair_weight_sum / max(float(total_hardneg), 1.0)),
            'mean_selected_score': (
                selected_score_sum / max(float(total_hardneg), 1.0)),
            'geometry_logprob_used': bool(geometry_logprob_used),
        }
        return weighted

    def _load_gaussian_semantic_scale_adapter_checkpoint(self,
                                                        checkpoint_path,
                                                        strict: bool = True
                                                        ) -> None:
        adapter = getattr(self, 'gaussian_semantic_scale_adapter', None)
        if adapter is None:
            raise RuntimeError(
                'Cannot load gaussian_semantic_scale.adapter_checkpoint '
                'without logit_adapter mode.')
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        prefix = 'gaussian_semantic_scale_adapter.'
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        incompatible = adapter.load_state_dict(state_dict, strict=strict)
        self.gaussian_semantic_scale_debug.update({
            'adapter_checkpoint': str(checkpoint_path),
            'adapter_checkpoint_strict': bool(strict),
            'adapter_checkpoint_missing_keys': list(incompatible.missing_keys),
            'adapter_checkpoint_unexpected_keys': list(
                incompatible.unexpected_keys),
        })

    def _apply_gaussian_semantic_scale(self,
                                       scores: Tensor,
                                       bbox_pred: Tensor,
                                       angle_pred: Tensor,
                                       priors: Tensor,
                                       img_shape,
                                       img_meta=None,
                                       level_idx: int = -1,
                                       density_features: Tensor | None = None
                                       ) -> Tensor:
        if (not self.gaussian_semantic_scale_enable
                or scores.numel() == 0):
            return scores
        num_score_classes = int(scores.shape[1])
        valid = self.gaussian_semantic_valid_mask.to(
            device=scores.device)[:num_score_classes]
        if not torch.any(valid):
            return scores

        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
        decoded_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)
        decoded_bboxes = self.bbox_coder.decode(
            priors, decoded_pred, max_shape=img_shape)
        decoded_tensor = (
            decoded_bboxes.tensor
            if isinstance(decoded_bboxes, RotatedBoxes)
            else decoded_bboxes)
        if decoded_tensor.numel() == 0 or decoded_tensor.shape[-1] < 4:
            return scores

        area = (decoded_tensor[:, 2].abs() * decoded_tensor[:, 3].abs()).clamp(
            min=1e-6)
        log_area = torch.log(area)
        means = self.gaussian_semantic_log_area_mean.to(
            device=scores.device, dtype=scores.dtype)[:num_score_classes]
        stds = self.gaussian_semantic_log_area_std.to(
            device=scores.device, dtype=scores.dtype)[:num_score_classes]
        stats = gaussian_log_area_stats(log_area.to(scores.dtype), means, stds,
                                        valid)
        z0 = getattr(
            self, 'gaussian_semantic_scale_z0_per_class',
            self.gaussian_semantic_scale_z0)
        beta = getattr(
            self, 'gaussian_semantic_scale_beta_per_class',
            self.gaussian_semantic_scale_beta)
        if torch.is_tensor(z0):
            z0 = z0.to(device=scores.device,
                       dtype=scores.dtype)[:num_score_classes]
        if torch.is_tensor(beta):
            beta = beta.to(device=scores.device,
                           dtype=scores.dtype)[:num_score_classes]
        delta = continuous_gaussian_logit_energy_delta(
            stats['z'],
            stats['valid'],
            z0=z0,
            beta=beta)

        if self.gaussian_semantic_scale_mode == 'logit_adapter':
            adapter = getattr(self, 'gaussian_semantic_scale_adapter', None)
            if adapter is not None:
                adapter_valid = stats['valid']
                adapter_min_abs_z = getattr(
                    self, 'gaussian_semantic_scale_adapter_min_abs_z', 0.0)
                if adapter_min_abs_z > 0:
                    adapter_valid = torch.logical_and(
                        adapter_valid, stats['abs_z'] >= adapter_min_abs_z)
                adapter_min_score = getattr(
                    self, 'gaussian_semantic_scale_adapter_min_score', 0.0)
                if adapter_min_score > 0:
                    adapter_valid = torch.logical_and(
                        adapter_valid, scores >= adapter_min_score)
                adapter_delta = adapter(
                    abs_z=stats['abs_z'],
                    signed_z=stats['z'],
                    gaussian_log_prob=stats['log_prob'],
                    log_std=stats['log_std'],
                    valid_mask=adapter_valid)
                if self.gaussian_semantic_scale_adapter_nonpositive_delta:
                    adapter_delta = torch.clamp(adapter_delta, max=0.0)
                if self.gaussian_semantic_scale_adapter_max_delta_abs > 0:
                    adapter_delta = torch.clamp(
                        adapter_delta,
                        min=-self.gaussian_semantic_scale_adapter_max_delta_abs)
                delta = delta + adapter_delta
        density_delta_valid_pairs = 0
        density_head = getattr(self, 'gaussian_semantic_density_head', None)
        if (getattr(self, 'gaussian_semantic_density_enable', False)
                and getattr(
                    self, 'gaussian_semantic_density_apply_logit_delta', False)
                and density_head is not None
                and density_features is not None):
            density = density_head(
                features=density_features.to(
                    device=scores.device, dtype=scores.dtype),
                log_area=log_area.to(scores.dtype),
                prior_mean=self.gaussian_semantic_log_area_mean,
                prior_std=self.gaussian_semantic_log_area_std,
                valid_mask=self.gaussian_semantic_valid_mask)
            density_delta_source = getattr(
                self, 'gaussian_semantic_density_logit_delta_source', 'head')
            if density_delta_source == 'log_prob_energy':
                threshold = scores.new_tensor(
                    getattr(
                        self,
                        'gaussian_semantic_density_logprob_delta_threshold',
                        -8.0))
                beta = max(
                    float(getattr(
                        self,
                        'gaussian_semantic_density_logprob_delta_beta',
                        1.0)),
                    0.0)
                density_delta = -beta * F.softplus(
                    threshold - density['log_prob'][:, :num_score_classes])
                if self.gaussian_semantic_density_max_delta_abs > 0:
                    density_delta = torch.clamp(
                        density_delta,
                        min=-self.gaussian_semantic_density_max_delta_abs)
            else:
                density_delta = density['logit_delta'][:, :num_score_classes]
            density_delta = torch.where(
                density['valid'][:, :num_score_classes],
                density_delta,
                torch.zeros_like(density_delta))
            delta = delta + density_delta.to(dtype=delta.dtype)
            density_delta_valid_pairs = int(
                density['valid'][:, :num_score_classes].sum().detach().cpu()
                .item())
        if self.gaussian_semantic_scale_adapter_nonpositive_delta:
            delta = torch.clamp(delta, max=0.0)

        eps = torch.finfo(scores.dtype).eps
        logits = torch.logit(scores.clamp(eps, 1 - eps))
        calibrated = torch.sigmoid(logits + delta.to(dtype=logits.dtype))
        valid_delta = delta[stats['valid']]
        if valid_delta.numel() > 0:
            penalty = -valid_delta.detach()
            penalty_mean = float(penalty.mean().cpu().item())
            penalty_max = float(penalty.max().cpu().item())
        else:
            penalty_mean = 0.0
            penalty_max = 0.0
        self._last_gaussian_semantic_scale_debug = {
            **self.gaussian_semantic_scale_debug,
            'domain_mode': self.gaussian_semantic_scale_domain_mode,
            'gaussian_energy_mode': self.gaussian_semantic_scale_mode,
            'last_num_locations': int(scores.shape[0]),
            'last_num_valid_pairs': int(
                stats['valid'].sum().detach().cpu().item()),
            'last_num_adapter_pairs': int(
                adapter_valid.sum().detach().cpu().item())
            if self.gaussian_semantic_scale_mode == 'logit_adapter'
            and 'adapter_valid' in locals() else 0,
            'last_num_density_delta_pairs': density_delta_valid_pairs,
            'density_logit_delta_source': (
                getattr(
                    self, 'gaussian_semantic_density_logit_delta_source',
                    'head')),
            'last_penalty_mean': penalty_mean,
            'last_penalty_max': penalty_max,
            'last_area_min': float(area.min().detach().cpu().item()),
            'last_area_max': float(area.max().detach().cpu().item()),
            'level_idx': int(level_idx),
            'image_id': (img_meta or {}).get('img_id'),
        }
        return calibrated

    def _loss_gaussian_semantic_consistency(
            self,
            cls_scores: List[Tensor],
            labels_list: List[Tensor],
            bbox_targets_list: List[Tensor],
            assign_metrics_list: List[Tensor]) -> Tensor:
        zero = cls_scores[0].sum() * 0.0
        if (not getattr(self, 'gaussian_semantic_consistency_enable', False)
                or not getattr(self, 'gaussian_semantic_scale_enable', False)):
            self._last_gaussian_semantic_consistency_debug = {
                'enable': bool(getattr(
                    self, 'gaussian_semantic_consistency_enable', False)),
                'loss_weight': self.gaussian_semantic_consistency_weight,
                'num_pos': 0,
                'num_hardneg': 0,
                'num_candidate_pairs': 0,
                'num_pos_with_hardneg': 0,
                'num_ap_sensitive_pos': 0,
                'active_violation_count': 0,
                'hardneg_per_pos': 0.0,
                'active_violation_rate': 0.0,
            }
            return zero

        losses = []
        total_pos = 0
        total_hardneg = 0
        total_candidate_pairs = 0
        total_pos_with_hardneg = 0
        total_ap_sensitive_pos = 0
        total_active_violation = 0
        gap_sum = 0.0
        for cls_score, labels, bbox_targets, assign_metrics in zip(
                cls_scores, labels_list, bbox_targets_list,
                assign_metrics_list):
            flat_cls = cls_score.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels).contiguous()
            loss, debug = gaussian_pos_hardneg_consistency_loss(
                cls_logits=flat_cls,
                labels=labels.reshape(-1),
                bbox_targets=bbox_targets.reshape(-1, 5),
                assign_metrics=assign_metrics.reshape(-1),
                log_area_mean=self.gaussian_semantic_log_area_mean,
                log_area_std=self.gaussian_semantic_log_area_std,
                valid_mask=self.gaussian_semantic_valid_mask,
                margin=self.gaussian_semantic_consistency_margin,
                min_logprob_gap=(
                    self.gaussian_semantic_consistency_min_logprob_gap),
                max_hardneg=self.gaussian_semantic_consistency_max_hardneg,
                min_hardneg_logit=getattr(
                    self,
                    'gaussian_semantic_consistency_min_hardneg_logit',
                    None),
                max_gt_abs_z=getattr(
                    self,
                    'gaussian_semantic_consistency_max_gt_abs_z',
                    None))
            losses.append(loss)
            total_pos += int(debug.get('num_pos', 0))
            hardneg = int(debug.get('num_hardneg', 0))
            total_hardneg += hardneg
            total_candidate_pairs += int(debug.get('num_candidate_pairs', 0))
            total_pos_with_hardneg += int(
                debug.get('num_pos_with_hardneg', 0))
            total_ap_sensitive_pos += int(
                debug.get('num_ap_sensitive_pos', 0))
            total_active_violation += int(
                debug.get('active_violation_count', 0))
            gap_sum += float(debug.get('mean_logprob_gap', 0.0)) * hardneg

        if not losses:
            return zero
        loss = torch.stack(losses).mean()
        weighted = loss * self.gaussian_semantic_consistency_weight
        self._last_gaussian_semantic_consistency_debug = {
            'enable': True,
            'loss_weight': self.gaussian_semantic_consistency_weight,
            'raw_loss': float(loss.detach().cpu().item()),
            'weighted_loss': float(weighted.detach().cpu().item()),
            'num_pos': int(total_pos),
            'num_hardneg': int(total_hardneg),
            'num_candidate_pairs': int(total_candidate_pairs),
            'num_pos_with_hardneg': int(total_pos_with_hardneg),
            'num_ap_sensitive_pos': int(total_ap_sensitive_pos),
            'active_violation_count': int(total_active_violation),
            'mean_logprob_gap': gap_sum / max(total_hardneg, 1),
            'hardneg_per_pos': (
                float(total_hardneg) / max(float(total_pos), 1.0)),
            'active_violation_rate': (
                float(total_active_violation)
                / max(float(total_hardneg), 1.0)),
        }
        return weighted

    def _loss_gaussian_scale_density(
            self,
            cls_scores: List[Tensor],
            labels_list: List[Tensor],
            bbox_targets_list: List[Tensor],
            assign_metrics_list: List[Tensor],
            pred_bboxes_list: List[Tensor] | None = None) -> Tensor:
        zero = cls_scores[0].sum() * 0.0
        density_head = getattr(self, 'gaussian_semantic_density_head', None)
        if (not getattr(self, 'gaussian_semantic_density_enable', False)
                or not getattr(self, 'gaussian_semantic_scale_enable', False)
                or density_head is None):
            self._last_gaussian_semantic_density_debug = {
                'enable': bool(getattr(
                    self, 'gaussian_semantic_density_enable', False)),
                'loss_weight': float(getattr(
                    self, 'gaussian_semantic_density_loss_weight', 0.0)),
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'num_pos': 0,
                'mean_gt_log_prob': 0.0,
                'delta_loss_enable': bool(getattr(
                    self,
                    'gaussian_semantic_density_delta_loss_enable',
                    False)),
                'delta_loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_density_delta_loss_weight',
                    0.0)),
                'delta_raw_loss': 0.0,
                'delta_weighted_loss': 0.0,
                'delta_num_hardneg': 0,
                'delta_candidate_pairs': 0,
                'delta_pos_with_hardneg': 0,
                'delta_score_gated_pairs': 0,
                'delta_mean_logprob_gap': 0.0,
                'delta_mean_hardneg_delta': 0.0,
                'positive_delta_loss_enable': bool(getattr(
                    self,
                    'gaussian_semantic_density_positive_delta_loss_enable',
                    False)),
                'positive_delta_loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_density_positive_delta_loss_weight',
                    0.0)),
                'positive_delta_raw_loss': 0.0,
                'positive_delta_weighted_loss': 0.0,
                'positive_delta_supported_pos': 0,
                'positive_delta_score_gated_pos': 0,
                'positive_delta_mean_gt_log_prob': 0.0,
                'positive_delta_mean_gt_delta': 0.0,
                'pair_margin_loss_enable': bool(getattr(
                    self,
                    'gaussian_semantic_density_pair_margin_loss_enable',
                    False)),
                'pair_margin_loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_density_pair_margin_loss_weight',
                    0.0)),
                'pair_margin_raw_loss': 0.0,
                'pair_margin_weighted_loss': 0.0,
                'pair_margin_num_hardneg': 0,
                'pair_margin_candidate_pairs': 0,
                'pair_margin_pos_with_hardneg': 0,
                'pair_margin_score_gated_pairs': 0,
                'pair_margin_active_violation_count': 0,
                'pair_margin_mean_logprob_gap': 0.0,
                'pair_margin_mean_pair_margin': 0.0,
                'support_negative_loss_enable': bool(getattr(
                    self,
                    'gaussian_semantic_density_support_negative_loss_enable',
                    False)),
                'support_negative_loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_density_support_negative_loss_weight',
                    0.0)),
                'support_negative_raw_loss': 0.0,
                'support_negative_weighted_loss': 0.0,
                'support_negative_negative_locations': 0,
                'support_negative_positive_locations': 0,
                'support_negative_candidate_pairs': 0,
                'support_negative_score_gated_pairs': 0,
                'support_negative_selected_pairs': 0,
                'support_negative_mean_score': 0.0,
                'support_negative_mean_logprob_gap': 0.0,
                'support_negative_mean_pair_weight': 0.0,
                'scale_consistency_loss_enable': bool(getattr(
                    self,
                    'gaussian_semantic_density_scale_consistency_loss_enable',
                    False)),
                'scale_consistency_loss_weight': float(getattr(
                    self,
                    'gaussian_semantic_density_scale_consistency_loss_weight',
                    0.0)),
                'scale_consistency_raw_loss': 0.0,
                'scale_consistency_weighted_loss': 0.0,
                'scale_consistency_num_pos': 0,
                'scale_consistency_supported_pos': 0,
                'scale_consistency_active_violation_count': 0,
                'scale_consistency_mean_target_log_prob': 0.0,
                'scale_consistency_mean_pred_log_prob': 0.0,
                'scale_consistency_mean_logprob_drop': 0.0,
                'geometry_logprob_used': False,
            }
            return zero

        losses = []
        delta_losses = []
        positive_delta_losses = []
        pair_margin_losses = []
        support_negative_losses = []
        scale_consistency_losses = []
        total_pos = 0
        logprob_sum = 0.0
        total_delta_hardneg = 0
        total_delta_candidate_pairs = 0
        total_delta_pos_with_hardneg = 0
        total_delta_score_gated_pairs = 0
        delta_gap_sum = 0.0
        delta_hardneg_delta_sum = 0.0
        total_positive_supported_pos = 0
        total_positive_score_gated_pos = 0
        positive_logprob_sum = 0.0
        positive_delta_sum = 0.0
        total_pair_margin_hardneg = 0
        total_pair_margin_candidate_pairs = 0
        total_pair_margin_pos_with_hardneg = 0
        total_pair_margin_score_gated_pairs = 0
        total_pair_margin_active_violation = 0
        pair_margin_gap_sum = 0.0
        pair_margin_margin_sum = 0.0
        total_support_negative_negative_locations = 0
        total_support_negative_positive_locations = 0
        total_support_negative_candidate_pairs = 0
        total_support_negative_score_gated_pairs = 0
        total_support_negative_selected_pairs = 0
        support_negative_score_sum = 0.0
        support_negative_gap_sum = 0.0
        support_negative_weight_sum = 0.0
        total_scale_consistency_pos = 0
        total_scale_consistency_supported_pos = 0
        total_scale_consistency_active_violation = 0
        scale_consistency_target_logprob_sum = 0.0
        scale_consistency_pred_logprob_sum = 0.0
        scale_consistency_drop_sum = 0.0
        geometry_logprob_used = False
        if pred_bboxes_list is None:
            pred_bboxes_iter = [None] * len(cls_scores)
        else:
            pred_bboxes_iter = pred_bboxes_list
        for cls_score, labels, bbox_targets, assign_metrics, pred_bboxes in zip(
                cls_scores, labels_list, bbox_targets_list,
                assign_metrics_list, pred_bboxes_iter):
            flat_boxes = bbox_targets.reshape(
                -1, bbox_targets.shape[-1]).to(
                    device=cls_score.device, dtype=cls_score.dtype)
            if flat_boxes.numel() == 0 or flat_boxes.shape[-1] < 4:
                continue
            area = (flat_boxes[:, 2].abs() * flat_boxes[:, 3].abs()).clamp(
                min=1e-6)
            density = density_head(
                features=cls_score,
                log_area=torch.log(area),
                prior_mean=self.gaussian_semantic_log_area_mean,
                prior_std=self.gaussian_semantic_log_area_std,
                valid_mask=self.gaussian_semantic_valid_mask)
            support_log_prob = density['log_prob']
            support_valid_mask = self.gaussian_semantic_valid_mask
            scale_target_log_prob = None
            scale_valid_mask = self.gaussian_semantic_valid_mask
            if (getattr(
                    self,
                    'gaussian_semantic_density_use_geometry_logprob',
                    False)
                    and getattr(
                        self, 'gaussian_semantic_geometry_enable', False)):
                long_side = torch.maximum(
                    flat_boxes[:, 2].abs(), flat_boxes[:, 3].abs())
                short_side = torch.minimum(
                    flat_boxes[:, 2].abs(), flat_boxes[:, 3].abs()).clamp(
                        min=1e-6)
                geometry = torch.stack(
                    [torch.log(area), torch.log(long_side / short_side)],
                    dim=-1)
                geometry_stats = gaussian_geometry_support_stats(
                    geometry=geometry,
                    means=self.gaussian_semantic_geometry_mean.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    stds=self.gaussian_semantic_geometry_std.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    valid_mask=self.gaussian_semantic_geometry_valid_mask.to(
                        device=cls_score.device))
                support_log_prob = geometry_stats['log_prob']
                support_valid_mask = self.gaussian_semantic_geometry_valid_mask
                scale_target_log_prob = geometry_stats['log_prob']
                scale_valid_mask = self.gaussian_semantic_geometry_valid_mask
                geometry_logprob_used = True
            if (scale_target_log_prob is None
                    and getattr(
                        self,
                        'gaussian_semantic_density_scale_consistency_loss_enable',
                        False)):
                target_area_stats = gaussian_log_area_stats(
                    torch.log(area),
                    self.gaussian_semantic_log_area_mean.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    self.gaussian_semantic_log_area_std.to(
                        device=cls_score.device, dtype=cls_score.dtype),
                    self.gaussian_semantic_valid_mask.to(
                        device=cls_score.device))
                scale_target_log_prob = target_area_stats['log_prob']
                scale_valid_mask = self.gaussian_semantic_valid_mask
            loss, debug = semantic_scale_density_nll_loss(
                log_prob=density['log_prob'],
                labels=labels.reshape(-1),
                assign_metrics=assign_metrics.reshape(-1),
                valid_mask=self.gaussian_semantic_valid_mask)
            losses.append(loss)
            num_pos = int(debug.get('num_pos', 0))
            total_pos += num_pos
            logprob_sum += (
                float(debug.get('mean_gt_log_prob', 0.0)) * num_pos)
            if getattr(
                    self,
                    'gaussian_semantic_density_delta_loss_enable',
                    False):
                delta_loss, delta_debug = (
                    semantic_scale_density_delta_outlier_loss(
                        logit_delta=density['logit_delta'],
                        log_prob=support_log_prob,
                        labels=labels.reshape(-1),
                        assign_metrics=assign_metrics.reshape(-1),
                        valid_mask=support_valid_mask,
                        cls_logits=cls_score.permute(0, 2, 3, 1).reshape(
                            -1, self.cls_out_channels).contiguous(),
                        min_hardneg_score=(
                            self.gaussian_semantic_density_delta_loss_min_hardneg_score),
                        min_hardneg_logit=(
                            self.gaussian_semantic_density_delta_loss_min_hardneg_logit),
                        min_logprob_gap=(
                            self.gaussian_semantic_density_delta_loss_min_logprob_gap),
                        target_negative_delta=(
                            self.gaussian_semantic_density_delta_loss_target_negative_delta),
                        gt_keep_weight=(
                            self.gaussian_semantic_density_delta_loss_gt_keep_weight),
                        max_hardneg=(
                            self.gaussian_semantic_density_delta_loss_max_hardneg)))
                delta_losses.append(delta_loss)
                num_hardneg = int(delta_debug.get('num_hardneg', 0))
                total_delta_hardneg += num_hardneg
                total_delta_candidate_pairs += int(
                    delta_debug.get('num_candidate_pairs', 0))
                total_delta_pos_with_hardneg += int(
                    delta_debug.get('num_pos_with_hardneg', 0))
                total_delta_score_gated_pairs += int(
                    delta_debug.get('num_score_gated_pairs', 0))
                delta_gap_sum += (
                    float(delta_debug.get('mean_logprob_gap', 0.0))
                    * num_hardneg)
                delta_hardneg_delta_sum += (
                    float(delta_debug.get('mean_hardneg_delta', 0.0))
                    * num_hardneg)
            if getattr(
                    self,
                    'gaussian_semantic_density_positive_delta_loss_enable',
                    False):
                positive_loss, positive_debug = (
                    semantic_scale_density_positive_delta_loss(
                        logit_delta=density['logit_delta'],
                        log_prob=support_log_prob,
                        labels=labels.reshape(-1),
                        assign_metrics=assign_metrics.reshape(-1),
                        valid_mask=support_valid_mask,
                        cls_logits=cls_score.permute(0, 2, 3, 1).reshape(
                            -1, self.cls_out_channels).contiguous(),
                        min_gt_logprob=(
                            self.gaussian_semantic_density_positive_delta_loss_min_gt_logprob),
                        min_gt_score=(
                            self.gaussian_semantic_density_positive_delta_loss_min_gt_score),
                        target_positive_delta=(
                            self.gaussian_semantic_density_positive_delta_loss_target)))
                positive_delta_losses.append(positive_loss)
                supported = int(positive_debug.get('num_supported_pos', 0))
                total_positive_supported_pos += supported
                total_positive_score_gated_pos += int(
                    positive_debug.get('num_score_gated_pos', 0))
                positive_logprob_sum += (
                    float(positive_debug.get('mean_gt_log_prob', 0.0))
                    * supported)
                positive_delta_sum += (
                    float(positive_debug.get('mean_gt_delta', 0.0))
                    * supported)
            if getattr(
                    self,
                    'gaussian_semantic_density_pair_margin_loss_enable',
                    False):
                pair_loss, pair_debug = (
                    semantic_scale_density_pair_margin_loss(
                        logit_delta=density['logit_delta'],
                        log_prob=support_log_prob,
                        labels=labels.reshape(-1),
                        assign_metrics=assign_metrics.reshape(-1),
                        valid_mask=support_valid_mask,
                        cls_logits=cls_score.permute(0, 2, 3, 1).reshape(
                            -1, self.cls_out_channels).contiguous(),
                        min_hardneg_score=(
                            self.gaussian_semantic_density_pair_margin_loss_min_hardneg_score),
                        min_logprob_gap=(
                            self.gaussian_semantic_density_pair_margin_loss_min_logprob_gap),
                        margin=(
                            self.gaussian_semantic_density_pair_margin_loss_margin),
                        max_hardneg=(
                            self.gaussian_semantic_density_pair_margin_loss_max_hardneg)))
                pair_margin_losses.append(pair_loss)
                pair_hardneg = int(pair_debug.get('num_hardneg', 0))
                total_pair_margin_hardneg += pair_hardneg
                total_pair_margin_candidate_pairs += int(
                    pair_debug.get('num_candidate_pairs', 0))
                total_pair_margin_pos_with_hardneg += int(
                    pair_debug.get('num_pos_with_hardneg', 0))
                total_pair_margin_score_gated_pairs += int(
                    pair_debug.get('num_score_gated_pairs', 0))
                total_pair_margin_active_violation += int(
                    pair_debug.get('active_violation_count', 0))
                pair_margin_gap_sum += (
                    float(pair_debug.get('mean_logprob_gap', 0.0))
                    * pair_hardneg)
                pair_margin_margin_sum += (
                    float(pair_debug.get('mean_pair_margin', 0.0))
                    * pair_hardneg)
            if ((getattr(
                    self,
                    'gaussian_semantic_density_support_negative_loss_enable',
                    False)
                    or getattr(
                        self,
                        'gaussian_semantic_density_scale_consistency_loss_enable',
                        False))
                    and pred_bboxes is not None):
                flat_pred_boxes = pred_bboxes.reshape(
                    -1, pred_bboxes.shape[-1]).to(
                        device=cls_score.device, dtype=cls_score.dtype)
                if flat_pred_boxes.numel() > 0 and flat_pred_boxes.shape[-1] >= 4:
                    pred_area = (
                        flat_pred_boxes[:, 2].abs()
                        * flat_pred_boxes[:, 3].abs()).clamp(min=1e-6)
                    pred_support_valid_mask = self.gaussian_semantic_valid_mask
                    if (getattr(
                            self,
                            'gaussian_semantic_density_use_geometry_logprob',
                            False)
                            and getattr(
                                self,
                                'gaussian_semantic_geometry_enable',
                                False)):
                        pred_long_side = torch.maximum(
                            flat_pred_boxes[:, 2].abs(),
                            flat_pred_boxes[:, 3].abs())
                        pred_short_side = torch.minimum(
                            flat_pred_boxes[:, 2].abs(),
                            flat_pred_boxes[:, 3].abs()).clamp(min=1e-6)
                        pred_geometry = torch.stack(
                            [
                                torch.log(pred_area),
                                torch.log(pred_long_side / pred_short_side),
                            ],
                            dim=-1)
                        pred_geometry_stats = gaussian_geometry_support_stats(
                            geometry=pred_geometry,
                            means=self.gaussian_semantic_geometry_mean.to(
                                device=cls_score.device,
                                dtype=cls_score.dtype),
                            stds=self.gaussian_semantic_geometry_std.to(
                                device=cls_score.device,
                                dtype=cls_score.dtype),
                            valid_mask=(
                                self.gaussian_semantic_geometry_valid_mask.to(
                                    device=cls_score.device)))
                        pred_support_log_prob = pred_geometry_stats[
                            'log_prob']
                        pred_support_valid_mask = (
                            self.gaussian_semantic_geometry_valid_mask)
                        geometry_logprob_used = True
                    else:
                        pred_area_stats = gaussian_log_area_stats(
                            torch.log(pred_area),
                            self.gaussian_semantic_log_area_mean.to(
                                device=cls_score.device,
                                dtype=cls_score.dtype),
                            self.gaussian_semantic_log_area_std.to(
                                device=cls_score.device,
                                dtype=cls_score.dtype),
                            self.gaussian_semantic_valid_mask.to(
                                device=cls_score.device))
                        pred_support_log_prob = pred_area_stats['log_prob']
                    if (getattr(
                            self,
                            'gaussian_semantic_density_scale_consistency_loss_enable',
                            False)
                            and scale_target_log_prob is not None):
                        scale_loss, scale_debug = (
                            gaussian_positive_scale_consistency_loss(
                                target_log_prob=scale_target_log_prob,
                                pred_log_prob=pred_support_log_prob,
                                labels=labels.reshape(-1),
                                assign_metrics=assign_metrics.reshape(-1),
                                valid_mask=scale_valid_mask,
                                max_logprob_drop=(
                                    self.gaussian_semantic_density_scale_consistency_loss_max_logprob_drop),
                                min_target_logprob=(
                                    self.gaussian_semantic_density_scale_consistency_loss_min_target_logprob)))
                        scale_consistency_losses.append(scale_loss)
                        scale_pos = int(scale_debug.get('num_pos', 0))
                        scale_supported = int(
                            scale_debug.get('num_supported_pos', 0))
                        total_scale_consistency_pos += scale_pos
                        total_scale_consistency_supported_pos += (
                            scale_supported)
                        total_scale_consistency_active_violation += int(
                            scale_debug.get(
                                'active_violation_count', 0))
                        scale_consistency_target_logprob_sum += (
                            float(scale_debug.get(
                                'mean_target_log_prob', 0.0))
                            * scale_supported)
                        scale_consistency_pred_logprob_sum += (
                            float(scale_debug.get(
                                'mean_pred_log_prob', 0.0))
                            * scale_supported)
                        scale_consistency_drop_sum += (
                            float(scale_debug.get(
                                'mean_logprob_drop', 0.0))
                            * scale_supported)
                    if getattr(
                            self,
                            'gaussian_semantic_density_support_negative_loss_enable',
                            False):
                        support_negative_loss, support_negative_debug = (
                            gaussian_support_negative_focal_loss(
                                cls_logits=cls_score.permute(
                                    0, 2, 3, 1).reshape(
                                        -1,
                                        self.cls_out_channels).contiguous(),
                                labels=labels.reshape(-1),
                                log_prob=pred_support_log_prob,
                                valid_mask=pred_support_valid_mask,
                                min_score=(
                                    self.gaussian_semantic_density_support_negative_loss_min_score),
                                min_logprob_gap=(
                                    self.gaussian_semantic_density_support_negative_loss_min_logprob_gap),
                                gamma=(
                                    self.gaussian_semantic_density_support_negative_loss_gamma),
                                gap_scale=(
                                    self.gaussian_semantic_density_support_negative_loss_gap_scale),
                                max_extra_weight=(
                                    self.gaussian_semantic_density_support_negative_loss_max_extra_weight),
                                max_hardneg=(
                                    self.gaussian_semantic_density_support_negative_loss_max_hardneg)))
                        support_negative_losses.append(
                            support_negative_loss)
                        selected_pairs = int(
                            support_negative_debug.get(
                                'num_selected_pairs', 0))
                        total_support_negative_negative_locations += int(
                            support_negative_debug.get(
                                'num_negative_locations', 0))
                        total_support_negative_positive_locations += int(
                            support_negative_debug.get(
                                'num_positive_locations', 0))
                        total_support_negative_candidate_pairs += int(
                            support_negative_debug.get(
                                'num_candidate_pairs', 0))
                        total_support_negative_score_gated_pairs += int(
                            support_negative_debug.get(
                                'num_score_gated_pairs', 0))
                        total_support_negative_selected_pairs += (
                            selected_pairs)
                        support_negative_score_sum += (
                            float(support_negative_debug.get(
                                'mean_selected_score', 0.0))
                            * selected_pairs)
                        support_negative_gap_sum += (
                            float(support_negative_debug.get(
                                'mean_logprob_gap', 0.0))
                            * selected_pairs)
                        support_negative_weight_sum += (
                            float(support_negative_debug.get(
                                'mean_pair_weight', 0.0))
                            * selected_pairs)

        if not losses:
            return zero
        loss = torch.stack(losses).mean()
        nll_weighted = loss * self.gaussian_semantic_density_loss_weight
        if delta_losses:
            delta_raw_loss = torch.stack(delta_losses).mean()
        else:
            delta_raw_loss = loss * 0.0
        delta_weighted = delta_raw_loss * float(getattr(
            self,
            'gaussian_semantic_density_delta_loss_weight',
            0.0))
        if positive_delta_losses:
            positive_delta_raw_loss = torch.stack(
                positive_delta_losses).mean()
        else:
            positive_delta_raw_loss = loss * 0.0
        positive_delta_weighted = positive_delta_raw_loss * float(getattr(
            self,
            'gaussian_semantic_density_positive_delta_loss_weight',
            0.0))
        if pair_margin_losses:
            pair_margin_raw_loss = torch.stack(pair_margin_losses).mean()
        else:
            pair_margin_raw_loss = loss * 0.0
        pair_margin_weighted = pair_margin_raw_loss * float(getattr(
            self,
            'gaussian_semantic_density_pair_margin_loss_weight',
            0.0))
        if support_negative_losses:
            support_negative_raw_loss = torch.stack(
                support_negative_losses).mean()
        else:
            support_negative_raw_loss = loss * 0.0
        support_negative_weighted = support_negative_raw_loss * float(getattr(
            self,
            'gaussian_semantic_density_support_negative_loss_weight',
            0.0))
        if scale_consistency_losses:
            scale_consistency_raw_loss = torch.stack(
                scale_consistency_losses).mean()
        else:
            scale_consistency_raw_loss = loss * 0.0
        scale_consistency_weighted = scale_consistency_raw_loss * float(
            getattr(
                self,
                'gaussian_semantic_density_scale_consistency_loss_weight',
                0.0))
        weighted = (
            nll_weighted + delta_weighted + positive_delta_weighted
            + pair_margin_weighted + support_negative_weighted
            + scale_consistency_weighted)
        self._last_gaussian_semantic_density_debug = {
            'enable': True,
            'loss_weight': self.gaussian_semantic_density_loss_weight,
            'raw_loss': float(loss.detach().cpu().item()),
            'weighted_loss': float(weighted.detach().cpu().item()),
            'nll_weighted_loss': float(nll_weighted.detach().cpu().item()),
            'num_pos': int(total_pos),
            'mean_gt_log_prob': logprob_sum / max(float(total_pos), 1.0),
            'delta_loss_enable': bool(getattr(
                self,
                'gaussian_semantic_density_delta_loss_enable',
                False)),
            'delta_loss_weight': float(getattr(
                self,
                'gaussian_semantic_density_delta_loss_weight',
                0.0)),
            'delta_raw_loss': float(delta_raw_loss.detach().cpu().item()),
            'delta_weighted_loss': float(
                delta_weighted.detach().cpu().item()),
            'delta_num_hardneg': int(total_delta_hardneg),
            'delta_candidate_pairs': int(total_delta_candidate_pairs),
            'delta_pos_with_hardneg': int(total_delta_pos_with_hardneg),
            'delta_score_gated_pairs': int(total_delta_score_gated_pairs),
            'delta_mean_logprob_gap': (
                delta_gap_sum / max(float(total_delta_hardneg), 1.0)),
            'delta_mean_hardneg_delta': (
                delta_hardneg_delta_sum
                / max(float(total_delta_hardneg), 1.0)),
            'positive_delta_loss_enable': bool(getattr(
                self,
                'gaussian_semantic_density_positive_delta_loss_enable',
                False)),
            'positive_delta_loss_weight': float(getattr(
                self,
                'gaussian_semantic_density_positive_delta_loss_weight',
                0.0)),
            'positive_delta_raw_loss': float(
                positive_delta_raw_loss.detach().cpu().item()),
            'positive_delta_weighted_loss': float(
                positive_delta_weighted.detach().cpu().item()),
            'positive_delta_supported_pos': int(
                total_positive_supported_pos),
            'positive_delta_score_gated_pos': int(
                total_positive_score_gated_pos),
            'positive_delta_mean_gt_log_prob': (
                positive_logprob_sum
                / max(float(total_positive_supported_pos), 1.0)),
            'positive_delta_mean_gt_delta': (
                positive_delta_sum
                / max(float(total_positive_supported_pos), 1.0)),
            'pair_margin_loss_enable': bool(getattr(
                self,
                'gaussian_semantic_density_pair_margin_loss_enable',
                False)),
            'pair_margin_loss_weight': float(getattr(
                self,
                'gaussian_semantic_density_pair_margin_loss_weight',
                0.0)),
            'pair_margin_raw_loss': float(
                pair_margin_raw_loss.detach().cpu().item()),
            'pair_margin_weighted_loss': float(
                pair_margin_weighted.detach().cpu().item()),
            'pair_margin_num_hardneg': int(total_pair_margin_hardneg),
            'pair_margin_candidate_pairs': int(
                total_pair_margin_candidate_pairs),
            'pair_margin_pos_with_hardneg': int(
                total_pair_margin_pos_with_hardneg),
            'pair_margin_score_gated_pairs': int(
                total_pair_margin_score_gated_pairs),
            'pair_margin_active_violation_count': int(
                total_pair_margin_active_violation),
            'pair_margin_mean_logprob_gap': (
                pair_margin_gap_sum
                / max(float(total_pair_margin_hardneg), 1.0)),
            'pair_margin_mean_pair_margin': (
                pair_margin_margin_sum
                / max(float(total_pair_margin_hardneg), 1.0)),
            'support_negative_loss_enable': bool(getattr(
                self,
                'gaussian_semantic_density_support_negative_loss_enable',
                False)),
            'support_negative_loss_weight': float(getattr(
                self,
                'gaussian_semantic_density_support_negative_loss_weight',
                0.0)),
            'support_negative_raw_loss': float(
                support_negative_raw_loss.detach().cpu().item()),
            'support_negative_weighted_loss': float(
                support_negative_weighted.detach().cpu().item()),
            'support_negative_negative_locations': int(
                total_support_negative_negative_locations),
            'support_negative_positive_locations': int(
                total_support_negative_positive_locations),
            'support_negative_candidate_pairs': int(
                total_support_negative_candidate_pairs),
            'support_negative_score_gated_pairs': int(
                total_support_negative_score_gated_pairs),
            'support_negative_selected_pairs': int(
                total_support_negative_selected_pairs),
            'support_negative_mean_score': (
                support_negative_score_sum / max(
                    float(total_support_negative_selected_pairs), 1.0)),
            'support_negative_mean_logprob_gap': (
                support_negative_gap_sum / max(
                    float(total_support_negative_selected_pairs), 1.0)),
            'support_negative_mean_pair_weight': (
                support_negative_weight_sum / max(
                    float(total_support_negative_selected_pairs), 1.0)),
            'scale_consistency_loss_enable': bool(getattr(
                self,
                'gaussian_semantic_density_scale_consistency_loss_enable',
                False)),
            'scale_consistency_loss_weight': float(getattr(
                self,
                'gaussian_semantic_density_scale_consistency_loss_weight',
                0.0)),
            'scale_consistency_raw_loss': float(
                scale_consistency_raw_loss.detach().cpu().item()),
            'scale_consistency_weighted_loss': float(
                scale_consistency_weighted.detach().cpu().item()),
            'scale_consistency_num_pos': int(total_scale_consistency_pos),
            'scale_consistency_supported_pos': int(
                total_scale_consistency_supported_pos),
            'scale_consistency_active_violation_count': int(
                total_scale_consistency_active_violation),
            'scale_consistency_mean_target_log_prob': (
                scale_consistency_target_logprob_sum / max(
                    float(total_scale_consistency_supported_pos), 1.0)),
            'scale_consistency_mean_pred_log_prob': (
                scale_consistency_pred_logprob_sum / max(
                    float(total_scale_consistency_supported_pos), 1.0)),
            'scale_consistency_mean_logprob_drop': (
                scale_consistency_drop_sum / max(
                    float(total_scale_consistency_supported_pos), 1.0)),
            'geometry_logprob_used': bool(geometry_logprob_used),
        }
        return weighted

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        raw_cls_scores = cls_scores
        cls_scores = [
            self._apply_gaussian_semantic_level_routing(
                cls_score, level_idx)
            for level_idx, cls_score in enumerate(cls_scores)
        ]
        assignment_cls_scores = cls_scores
        if getattr(
                self,
                'gaussian_semantic_level_routing_target_assignment_source',
                'routed') == 'pre_routing':
            assignment_cls_scores = raw_cls_scores

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in assignment_cls_scores
        ], 1)

        decoded_bboxes = []
        decoded_hbboxes = []
        angle_preds_list = []
        for anchor, bbox_pred, angle_pred in zip(anchor_list[0], bbox_preds,
                                                 angle_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.angle_coder.encode_size)

            if self.use_hbbox_loss:
                hbbox_pred = distance2bbox(anchor, bbox_pred)
                decoded_hbboxes.append(hbbox_pred)

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)
            bbox_pred = distance2obb(
                anchor, bbox_pred, angle_version=self.angle_version)
            decoded_bboxes.append(bbox_pred)
            angle_preds_list.append(angle_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)
        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        if self.use_hbbox_loss:
            decoded_bboxes = decoded_hbboxes

        (losses_cls, losses_bbox, losses_angle, cls_avg_factors,
         bbox_avg_factors, angle_avg_factors) = multi_apply(
            self.loss_by_feat_single, cls_scores, decoded_bboxes,
            angle_preds_list, labels_list, label_weights_list,
            bbox_targets_list, assign_metrics_list,
            self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses = {}
        if self.loss_angle is not None:
            angle_avg_factors = reduce_mean(
                sum(angle_avg_factors)).clamp_(min=1).item()
            losses_angle = list(
                map(lambda x: x / angle_avg_factors, losses_angle))
            losses.update(dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_angle=losses_angle))
        else:
            losses.update(dict(loss_cls=losses_cls, loss_bbox=losses_bbox))

        loss_g3 = self._loss_gaussian_semantic_consistency(
            cls_scores, labels_list, bbox_targets_list, assign_metrics_list)
        if getattr(self, 'gaussian_semantic_consistency_enable', False):
            losses['loss_gs3c_g3'] = loss_g3
            if getattr(self, 'gaussian_semantic_consistency_log_stats', False):
                debug = self._last_gaussian_semantic_consistency_debug
                metric_base = loss_g3.detach()
                losses.update({
                    'gs3c_g3_raw': metric_base.new_tensor(
                        float(debug.get('raw_loss', 0.0))),
                    'gs3c_g3_weighted': metric_base.new_tensor(
                        float(debug.get('weighted_loss', 0.0))),
                    'gs3c_g3_num_pos': metric_base.new_tensor(
                        float(debug.get('num_pos', 0))),
                    'gs3c_g3_num_hardneg': metric_base.new_tensor(
                        float(debug.get('num_hardneg', 0))),
                    'gs3c_g3_candidate_pairs': metric_base.new_tensor(
                        float(debug.get('num_candidate_pairs', 0))),
                    'gs3c_g3_pos_with_hardneg': metric_base.new_tensor(
                        float(debug.get('num_pos_with_hardneg', 0))),
                    'gs3c_g3_ap_sensitive_pos': metric_base.new_tensor(
                        float(debug.get('num_ap_sensitive_pos', 0))),
                    'gs3c_g3_active_violation_count': metric_base.new_tensor(
                        float(debug.get('active_violation_count', 0))),
                    'gs3c_g3_hardneg_per_pos': metric_base.new_tensor(
                        float(debug.get('hardneg_per_pos', 0.0))),
                    'gs3c_g3_active_violation_rate': metric_base.new_tensor(
                        float(debug.get('active_violation_rate', 0.0))),
                })
        loss_density = self._loss_gaussian_scale_density(
            cls_scores,
            labels_list,
            bbox_targets_list,
            assign_metrics_list,
            pred_bboxes_list=decoded_bboxes)
        if getattr(self, 'gaussian_semantic_density_enable', False):
            losses['loss_gs3c_density'] = loss_density
            if getattr(self, 'gaussian_semantic_density_log_stats', False):
                debug = self._last_gaussian_semantic_density_debug
                metric_base = loss_density.detach()
                losses.update({
                    'gs3c_density_raw': metric_base.new_tensor(
                        float(debug.get('raw_loss', 0.0))),
                    'gs3c_density_weighted': metric_base.new_tensor(
                        float(debug.get('weighted_loss', 0.0))),
                    'gs3c_density_nll_weighted': metric_base.new_tensor(
                        float(debug.get('nll_weighted_loss', 0.0))),
                    'gs3c_density_num_pos': metric_base.new_tensor(
                        float(debug.get('num_pos', 0))),
                    'gs3c_density_mean_gt_log_prob': metric_base.new_tensor(
                        float(debug.get('mean_gt_log_prob', 0.0))),
                    'gs3c_density_delta_raw': metric_base.new_tensor(
                        float(debug.get('delta_raw_loss', 0.0))),
                    'gs3c_density_delta_weighted': metric_base.new_tensor(
                        float(debug.get('delta_weighted_loss', 0.0))),
                    'gs3c_density_delta_num_hardneg': metric_base.new_tensor(
                        float(debug.get('delta_num_hardneg', 0))),
                    'gs3c_density_delta_candidate_pairs': (
                        metric_base.new_tensor(float(debug.get(
                            'delta_candidate_pairs', 0)))),
                    'gs3c_density_delta_pos_with_hardneg': (
                        metric_base.new_tensor(float(debug.get(
                            'delta_pos_with_hardneg', 0)))),
                    'gs3c_density_delta_score_gated_pairs': (
                        metric_base.new_tensor(float(debug.get(
                            'delta_score_gated_pairs', 0)))),
                    'gs3c_density_delta_mean_logprob_gap': (
                        metric_base.new_tensor(float(debug.get(
                            'delta_mean_logprob_gap', 0.0)))),
                    'gs3c_density_delta_mean_hardneg_delta': (
                        metric_base.new_tensor(float(debug.get(
                            'delta_mean_hardneg_delta', 0.0)))),
                    'gs3c_density_positive_delta_raw': (
                        metric_base.new_tensor(float(debug.get(
                            'positive_delta_raw_loss', 0.0)))),
                    'gs3c_density_positive_delta_weighted': (
                        metric_base.new_tensor(float(debug.get(
                            'positive_delta_weighted_loss', 0.0)))),
                    'gs3c_density_positive_delta_supported_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'positive_delta_supported_pos', 0)))),
                    'gs3c_density_positive_delta_score_gated_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'positive_delta_score_gated_pos', 0)))),
                    'gs3c_density_positive_delta_mean_gt_log_prob': (
                        metric_base.new_tensor(float(debug.get(
                            'positive_delta_mean_gt_log_prob', 0.0)))),
                    'gs3c_density_positive_delta_mean_gt_delta': (
                        metric_base.new_tensor(float(debug.get(
                            'positive_delta_mean_gt_delta', 0.0)))),
                    'gs3c_density_pair_margin_raw': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_raw_loss', 0.0)))),
                    'gs3c_density_pair_margin_weighted': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_weighted_loss', 0.0)))),
                    'gs3c_density_pair_margin_num_hardneg': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_num_hardneg', 0)))),
                    'gs3c_density_pair_margin_candidate_pairs': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_candidate_pairs', 0)))),
                    'gs3c_density_pair_margin_pos_with_hardneg': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_pos_with_hardneg', 0)))),
                    'gs3c_density_pair_margin_score_gated_pairs': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_score_gated_pairs', 0)))),
                    'gs3c_density_pair_margin_active_violation': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_active_violation_count', 0)))),
                    'gs3c_density_pair_margin_mean_logprob_gap': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_mean_logprob_gap', 0.0)))),
                    'gs3c_density_pair_margin_mean_pair_margin': (
                        metric_base.new_tensor(float(debug.get(
                            'pair_margin_mean_pair_margin', 0.0)))),
                    'gs3c_density_support_negative_raw': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_raw_loss', 0.0)))),
                    'gs3c_density_support_negative_weighted': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_weighted_loss', 0.0)))),
                    'gs3c_density_support_negative_neg_locs': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_negative_locations', 0)))),
                    'gs3c_density_support_negative_pos_locs': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_positive_locations', 0)))),
                    'gs3c_density_support_negative_candidate_pairs': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_candidate_pairs', 0)))),
                    'gs3c_density_support_negative_score_gated': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_score_gated_pairs', 0)))),
                    'gs3c_density_support_negative_selected_pairs': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_selected_pairs', 0)))),
                    'gs3c_density_support_negative_mean_score': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_mean_score', 0.0)))),
                    'gs3c_density_support_negative_mean_logprob_gap': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_mean_logprob_gap', 0.0)))),
                    'gs3c_density_support_negative_mean_pair_weight': (
                        metric_base.new_tensor(float(debug.get(
                            'support_negative_mean_pair_weight', 0.0)))),
                    'gs3c_density_scale_consistency_raw': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_raw_loss', 0.0)))),
                    'gs3c_density_scale_consistency_weighted': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_weighted_loss', 0.0)))),
                    'gs3c_density_scale_consistency_num_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_num_pos', 0)))),
                    'gs3c_density_scale_consistency_supported_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_supported_pos', 0)))),
                    'gs3c_density_scale_consistency_active': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_active_violation_count',
                            0)))),
                    'gs3c_density_scale_consistency_mean_target_log_prob': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_mean_target_log_prob',
                            0.0)))),
                    'gs3c_density_scale_consistency_mean_pred_log_prob': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_mean_pred_log_prob',
                            0.0)))),
                    'gs3c_density_scale_consistency_mean_logprob_drop': (
                        metric_base.new_tensor(float(debug.get(
                            'scale_consistency_mean_logprob_drop', 0.0)))),
                })
        loss_level_projection = (
            self._loss_gaussian_semantic_level_routing_projection(
                raw_cls_scores,
                cls_scores,
                labels_list,
                assign_metrics_list))
        if getattr(
                self,
                'gaussian_semantic_level_routing_projection_loss_enable',
                False):
            losses['loss_gs3c_level_routing_projection'] = (
                loss_level_projection)
            if getattr(
                    self, 'gaussian_semantic_level_routing_log_stats',
                    False):
                debug = (
                    self._last_gaussian_semantic_level_routing_projection_debug)
                metric_base = loss_level_projection.detach()
                losses.update({
                    'gs3c_level_routing_projection_raw': (
                        metric_base.new_tensor(float(debug.get(
                            'raw_loss', 0.0)))),
                    'gs3c_level_routing_projection_weighted': (
                        metric_base.new_tensor(float(debug.get(
                            'weighted_loss', 0.0)))),
                    'gs3c_level_routing_projection_num_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'num_pos', 0)))),
                    'gs3c_level_routing_projection_active': (
                        metric_base.new_tensor(float(debug.get(
                            'active_violation_count', 0)))),
                    'gs3c_level_routing_projection_mean_drop': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_positive_logit_drop', 0.0)))),
                })
        loss_ap_safe_projection = (
            self._loss_gaussian_semantic_ap_safe_support_projection(
                cls_scores,
                labels_list,
                bbox_targets_list,
                assign_metrics_list))
        if getattr(
                self,
                'gaussian_semantic_ap_safe_support_projection_enable',
                False):
            losses['loss_gs3c_ap_safe_support_projection'] = (
                loss_ap_safe_projection)
            if getattr(
                    self,
                    'gaussian_semantic_ap_safe_support_projection_log_stats',
                    False):
                debug = (
                    self
                    ._last_gaussian_semantic_ap_safe_support_projection_debug)
                metric_base = loss_ap_safe_projection.detach()
                losses.update({
                    'gs3c_ap_safe_support_projection_raw': (
                        metric_base.new_tensor(float(debug.get(
                            'raw_loss', 0.0)))),
                    'gs3c_ap_safe_support_projection_weighted': (
                        metric_base.new_tensor(float(debug.get(
                            'weighted_loss', 0.0)))),
                    'gs3c_ap_safe_support_projection_num_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'num_pos', 0)))),
                    'gs3c_ap_safe_support_projection_supported_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'num_supported_pos', 0)))),
                    'gs3c_ap_safe_support_projection_candidates': (
                        metric_base.new_tensor(float(debug.get(
                            'num_candidate_pairs', 0)))),
                    'gs3c_ap_safe_support_projection_score_gated': (
                        metric_base.new_tensor(float(debug.get(
                            'num_score_gated_pairs', 0)))),
                    'gs3c_ap_safe_support_projection_hardneg': (
                        metric_base.new_tensor(float(debug.get(
                            'num_hardneg', 0)))),
                    'gs3c_ap_safe_support_projection_pos_with_hardneg': (
                        metric_base.new_tensor(float(debug.get(
                            'num_pos_with_hardneg', 0)))),
                    'gs3c_ap_safe_support_projection_active': (
                        metric_base.new_tensor(float(debug.get(
                            'active_projection_count', 0)))),
                    'gs3c_ap_safe_support_projection_mean_logprob_gap': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_logprob_gap', 0.0)))),
                    'gs3c_ap_safe_support_projection_mean_gt_margin': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_gt_margin', 0.0)))),
                    'gs3c_ap_safe_support_projection_mean_safe_budget': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_safe_budget', 0.0)))),
                    'gs3c_ap_safe_support_projection_mean_pair_weight': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_pair_weight', 0.0)))),
                    'gs3c_ap_safe_support_projection_mean_selected_score': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_selected_score', 0.0)))),
                })
        loss_ap_safe_ranking_projection = (
            self._loss_gaussian_semantic_ap_safe_ranking_projection(
                cls_scores,
                labels_list,
                bbox_targets_list,
                assign_metrics_list))
        if getattr(
                self,
                'gaussian_semantic_ap_safe_ranking_projection_enable',
                False):
            losses['loss_gs3c_ap_safe_ranking_projection'] = (
                loss_ap_safe_ranking_projection)
            if getattr(
                    self,
                    'gaussian_semantic_ap_safe_ranking_projection_log_stats',
                    False):
                debug = (
                    self
                    ._last_gaussian_semantic_ap_safe_ranking_projection_debug)
                metric_base = loss_ap_safe_ranking_projection.detach()
                losses.update({
                    'gs3c_ap_safe_ranking_projection_raw': (
                        metric_base.new_tensor(float(debug.get(
                            'raw_loss', 0.0)))),
                    'gs3c_ap_safe_ranking_projection_weighted': (
                        metric_base.new_tensor(float(debug.get(
                            'weighted_loss', 0.0)))),
                    'gs3c_ap_safe_ranking_projection_num_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'num_pos', 0)))),
                    'gs3c_ap_safe_ranking_projection_supported_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'num_supported_pos', 0)))),
                    'gs3c_ap_safe_ranking_projection_candidates': (
                        metric_base.new_tensor(float(debug.get(
                            'num_candidate_pairs', 0)))),
                    'gs3c_ap_safe_ranking_projection_score_gated': (
                        metric_base.new_tensor(float(debug.get(
                            'num_score_gated_pairs', 0)))),
                    'gs3c_ap_safe_ranking_projection_hardneg': (
                        metric_base.new_tensor(float(debug.get(
                            'num_hardneg', 0)))),
                    'gs3c_ap_safe_ranking_projection_pos_with_hardneg': (
                        metric_base.new_tensor(float(debug.get(
                            'num_pos_with_hardneg', 0)))),
                    'gs3c_ap_safe_ranking_projection_active': (
                        metric_base.new_tensor(float(debug.get(
                            'active_projection_count', 0)))),
                    'gs3c_ap_safe_ranking_projection_protected_pos': (
                        metric_base.new_tensor(float(debug.get(
                            'protected_positive_count', 0)))),
                    'gs3c_ap_safe_ranking_projection_mean_logprob_gap': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_logprob_gap', 0.0)))),
                    'gs3c_ap_safe_ranking_projection_mean_gt_margin': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_gt_margin', 0.0)))),
                    'gs3c_ap_safe_ranking_projection_mean_safe_budget': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_safe_budget', 0.0)))),
                    'gs3c_ap_safe_ranking_projection_mean_projection_delta': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_projection_delta', 0.0)))),
                    'gs3c_ap_safe_ranking_projection_mean_pair_weight': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_pair_weight', 0.0)))),
                    'gs3c_ap_safe_ranking_projection_mean_selected_score': (
                        metric_base.new_tensor(float(debug.get(
                            'mean_selected_score', 0.0)))),
                })
        return losses

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            density_features = (
                cls_score.unsqueeze(0)
                if cls_score.dim() == 3 else cls_score)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            self._dump_ep2_path_probe_logits(
                cls_logits=cls_score,
                bbox_pred=bbox_pred,
                angle_pred=angle_pred,
                priors=priors,
                img_shape=img_shape,
                img_meta=img_meta,
                level_idx=level_idx)
            cls_score = self._apply_gaussian_semantic_level_routing(
                cls_score, level_idx)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            scores = self._apply_gaussian_semantic_scale(
                scores=scores,
                bbox_pred=bbox_pred,
                angle_pred=angle_pred,
                priors=priors,
                img_shape=img_shape,
                img_meta=img_meta,
                level_idx=level_idx,
                density_features=density_features)

            score_thr = cfg.get('score_thr', 0)
            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
