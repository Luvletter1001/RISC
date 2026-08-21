# Copyright (c) OpenMMLab. All rights reserved.
import copy
import csv
import json
import os
from typing import List, Optional, Tuple

import torch
from mmcv.cnn import ConvModule, Scale, is_norm
from mmdet.models import inverse_sigmoid
from mmdet.models.dense_heads import RTMDetHead
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, cat_boxes, distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn
import numpy as np

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, distance2obb
from mmengine.model import BaseModule
from mmcv.cnn.bricks import build_norm_layer

import torch.nn.functional as F
from mmrotate.models.dense_heads.rotated_rtmdet_head import RotatedRTMDetHead, RotatedRTMDetSepBNHead
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl,
                                unpack_gt_instances)
from mmdet.structures import OptSampleList, SampleList
from copy import deepcopy
import math

from mmdet.models.utils import (images_to_levels, multi_apply, sigmoid_geometric_mean,
                                unmap)
from M_AD.models.utils.focus_contrastive_embed import OrientationConditionedContrastiveEmbed
from M_AD.models.utils.focus_dual_support_fusion import FocusDualSupportFusion
from M_AD.models.utils.focus_mess_fourier_text_branch import (
    build_focus_text_adapter,
)
from M_AD.models.utils.focus_fourier_head_gate import FourierHeadLogitAdapter
from M_AD.models.utils.focus_fourier_orientation import FourierOrientationLearner
from M_AD.models.utils.focus_loss_targets import (
    normalize_focus_losses_config,
    validate_focus_loss_targets_config,
)
from M_AD.models.utils.focus_support_adapter import FourierSupportResidualAdapter
from M_AD.models.utils.focus_text_anchor_calibration import (
    DEFAULT_DOTA2_CLASSES,
    TextAnchorCalibration,
)
from M_AD.models.utils.focus_text_logit_mixer import FocusTextLogitMixer
from M_AD.models.utils.gaussian_semantic_scale import (
    GaussianScaleLogitAdapter,
    build_classwise_gaussian_parameter,
    continuous_gaussian_logit_energy_delta,
    gaussian_log_area_stats,
    load_class_log_area_priors,
)
from M_AD.models.utils.ep2_path_probe_dump import (
    append_ep2_path_probe_rows,
    build_ep2_path_probe_rows,
)
from M_AD.models.losses.focus_attractor_losses import (
    anti_attractor_loss,
    migration_kl_loss,
    preserve_loss,
)
from M_AD.models.utils.counter_support_evidence_ratio import (
    CounterSupportEvidenceRatio,
)

"""
纯预训练
"""


def compute_ccl_loss(
        dense_embeds: Tensor,
        assigned_labels: Tensor,
        support_feats: Tensor,
        support_labels: Tensor,
        num_classes: int,
        temperature: float = 0.1) -> Tensor:
    """Contextual consistency loss over dense locations and class supports."""
    zero = dense_embeds.sum() * 0.0
    if dense_embeds.numel() == 0 or support_feats.numel() == 0:
        return zero

    temperature = max(float(temperature), 1e-6)
    losses = []
    batch_size = dense_embeds.shape[0]
    for batch_idx in range(batch_size):
        labels_b = assigned_labels[batch_idx].long()
        support_labels_b = support_labels[batch_idx].long()
        support_feats_b = support_feats[batch_idx]

        valid_support = ((support_labels_b >= 0)
                         & (support_labels_b < int(num_classes)))
        if not torch.any(valid_support):
            continue

        proto_labels = torch.unique(support_labels_b[valid_support])
        prototypes = []
        for label in proto_labels:
            class_mask = valid_support & (support_labels_b == label)
            prototypes.append(support_feats_b[class_mask].mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)

        pos_mask = ((labels_b >= 0) & (labels_b < int(num_classes)))
        if not torch.any(pos_mask):
            continue

        pos_embeds = dense_embeds[batch_idx][pos_mask]
        pos_labels = labels_b[pos_mask]
        target_matches = pos_labels[:, None] == proto_labels[None, :]
        keep = target_matches.any(dim=1)
        if not torch.any(keep):
            continue

        pos_embeds = F.normalize(pos_embeds[keep], dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        targets = target_matches[keep].float().argmax(dim=1)
        logits = pos_embeds @ prototypes.transpose(0, 1) / temperature
        losses.append(F.cross_entropy(logits, targets))

    if not losses:
        return zero
    return torch.stack(losses).mean()


class InstanceAlignmentEmbed(nn.Module):

    def __init__(self, embed_dims=256):
        super().__init__()

    def forward(self,
                q, k,
                q_label, k_label,
                visual_fc, text_fc,
                **kwargs):
        """

        :param q:
        :param k:
        :param q_label: 分配的label，只有正样本
        :param k_label: support labels，包含负样本(label为-1)
        :param visual_fc:
        :param text_fc:
        :param kwargs:
        :return:
        """
        q_ = visual_fc(q)
        k_ = visual_fc(k)
        temperature = torch.Tensor([0.1]).to(q_.device)
        loss_ct = self.loss_nearest_contrastive(q_, k_, q_label, k_label, temperature)

        return loss_ct


    def loss_nearest_contrastive(self, q_org, k_org, q_label_org, k_label, temperature):
        ### ----- 对每个k (word) 寻找最近的q（同个标签），实现一一匹配
        cosine_sims = torch.einsum('nc,mc->nm',
                                   [F.normalize(k_org, dim=-1),
                                    F.normalize(q_org, dim=-1)])
        if len(q_label_org) == 0 or len(k_label) == 0:
            print('Zero Contrastive: ')
            print(cosine_sims.shape, q_org.shape, q_label_org)
            return torch.Tensor([0.]).to(cosine_sims.device)

        # label_eq = torch.eq(k_m_label.reshape(-1, 1),
        #                     q_label.reshape(1, -1))
        # cosine_sims[~label_eq] = float('-inf')
        nearest_sims, nearest_ids = torch.max(cosine_sims, dim=1)

        q = q_org[nearest_ids]
        q_label = q_label_org[nearest_ids]

        ################# --- 计算监督对比损失(loss_supervised_contrastive)
        ### --- 只保留region to word的损失函数
        # ----- 原因：负样本word与任何一个region都不匹配，因此应当为全0，而损失计算的是softmax损失
        q = F.normalize(q, dim=-1)
        k = F.normalize(k_org, dim=-1)

        logits_12 = torch.einsum('nc,mc->nm', [q, k]) / temperature
        exp_12 = torch.exp(logits_12)

        # positive assignment mask, N x M and M x N
        mask_12 = torch.eq(q_label.reshape(-1, 1),
                           k_label.reshape(1, -1)).int()

        # probabilities
        prob_12 = exp_12 / torch.sum(exp_12, dim=1, keepdim=True).clamp(min=1e-6)

        # log first and then average
        l_sup_12 = - torch.sum(mask_12 * torch.log(prob_12), dim=1) / \
                   torch.sum(mask_12, dim=1).clamp(min=1)

        # average all
        loss = 2 * temperature * torch.mean(l_sup_12)
        return loss

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BaseClsHead(nn.Module):

    def __init__(self):
        super().__init__()

    def get_matching_scores(self,
                            matching_scores,
                            support_labels,
                            **kwargs):
        """

        :param matching_scores: B, N, M
        :param support_labels:  B, M
        :param kwargs:
        :return:
        """
        # ----- padding的score设置为-inf，因此在计算分类损失的时候这些特征不会计算分类损失
        B, N, M = matching_scores.shape
        support_labels = support_labels[:, None, :].expand([B, N, M])
        matching_scores[support_labels < 0] = -10 # float('-inf')

        # ----- 只有两种情况，相差为1，包括一个out_class，或者相差为0，不包括out_class
        num_classes = kwargs['num_classes']
        num_in_classes = kwargs['num_in_classes']
        support_shot = kwargs['support_shot']
        # assert (num_classes - num_in_classes) in [0, 1]

        len_in_classes = num_in_classes * support_shot
        cls_scores = torch.full([B, N, num_classes],
                                float('-inf'),
                                device=matching_scores.device)
        # ----- in_class的分类分数，由于每个类数量是一致的，因此直接取max
        in_match_scores = (matching_scores[:, :, :len_in_classes].
                           reshape(B, N, num_in_classes, support_shot))
        in_cls_scores = torch.max(in_match_scores, dim=-1)[0]
        cls_scores[:, :, :num_in_classes] = in_cls_scores

        # ----- out_class的分类分数，也是取max
        if M > len_in_classes:
            out_match_scores = matching_scores[:, :, len_in_classes:]
            out_cls_scores = torch.max(out_match_scores, dim=-1)[0]
            cls_scores[:, :, -1] = out_cls_scores

        return cls_scores

class ContrastiveEmbed(BaseClsHead):

    def __init__(self):
        super().__init__()
        # ------------ 结果缩放
        self.log_scale_t1 = nn.Parameter(
            torch.Tensor([-float(1.0)]), requires_grad=True)
        self.bias_t1 = nn.Parameter(
            torch.Tensor([-float(4.0)]), requires_grad=True)

    def forward(self,
                pred_embeds,
                support_feats,
                support_labels,
                visual_fc,
                text_fc,
                **kwargs
                ) -> (Tensor, Tensor):
        # ----- Semantic mapping
        B, D, H, W = pred_embeds.shape
        x = (pred_embeds.permute(0, 2, 3, 1).reshape(B, H * W, D) +
             sum(x.view(-1)[0]for x in self.parameters()) * 0.)
        align_style = kwargs['align_style']
        if align_style in ['labelled', 'pure_img']:             # ---- support是visual embeds
            bias = self.bias_t1
            log_scale = self.log_scale_t1
            x = visual_fc(x)
            support_feats = visual_fc(support_feats)
        else:
            raise Exception(f'Unknown alignment style {align_style}')

        w = F.normalize(support_feats, dim=-1)
        # ----- Get matching scores: B N D x B M D -> B N M
        match_logit = x @ w.transpose(-1, -2)
        scaled_logit = match_logit * log_scale.exp() + bias
        # ----- Get class logits: B N M -> B N C -> B C H W
        cls_logits = self.get_matching_scores(scaled_logit, support_labels, **kwargs)
        cls_logits = (cls_logits.reshape(B, H, W, cls_logits.shape[-1]).
                      permute(0, 3, 1, 2).contiguous())

        return cls_logits

from M_AD.models.utils.transformer_modular_yolo import TwoWayTransformerModularYOLO

class DummyCrossAttention(nn.Module):
    def __init__(
            self) -> None:
        super().__init__()

    def forward(self, queries, keys):
        return queries, keys

@MODELS.register_module()
class OpenRotatedRTMDetSepBNHead(RotatedRTMDetSepBNHead):
    """
    继承自E_Rrtmdet_head_v2.py
    添加显式Contrastive Alignment

    结论：
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 with_obj_align=False,
                 ## ---
                 with_slot_embed=True,
                 cross_mlp_dim=2048,
                 cross_num_layers=3,
                 use_sv_dehub_loss: bool = False,
                 sv_dehub_margin: float = 0.05,
                 sv_dehub_loss_weight: float = 0.05,
                 use_ccl_loss: bool = False,
                 ccl_loss_weight: float = 0.05,
                 ccl_temperature: float = 0.1,
                 use_focus_ovd: bool = False,
                 focus_ovd: Optional[dict] = None,
                 focus_losses: Optional[dict] = None,
                 focus_text_anchor_calibration: Optional[dict] = None,
                 focus_fourier_head_gate: Optional[dict] = None,
                 focus_text_logit_mixer: Optional[dict] = None,
                 scale_semantic_calibration: Optional[dict] = None,
                 gaussian_semantic_scale: Optional[dict] = None,
                 counter_support_ratio: Optional[dict] = None,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_sv_dehub_loss = use_sv_dehub_loss
        self.sv_dehub_margin = sv_dehub_margin
        self.sv_dehub_loss_weight = sv_dehub_loss_weight
        self.use_ccl_loss = use_ccl_loss
        self.ccl_loss_weight = ccl_loss_weight
        self.ccl_temperature = ccl_temperature
        self.focus_ovd_cfg = dict(focus_ovd or {})
        self.focus_text_anchor_calibration_cfg = dict(
            focus_text_anchor_calibration or {})
        self.focus_fourier_head_gate_cfg = dict(
            focus_fourier_head_gate or {})
        self.focus_text_logit_mixer_cfg = dict(
            focus_text_logit_mixer or {})
        self.scale_semantic_calibration_cfg = dict(
            scale_semantic_calibration or {})
        self.gaussian_semantic_scale_cfg = dict(
            gaussian_semantic_scale or {})
        self.counter_support_ratio_cfg = dict(counter_support_ratio or {})
        self.focus_losses = normalize_focus_losses_config(focus_losses)
        self.focus_loss_config_status = validate_focus_loss_targets_config(
            self.focus_losses)
        self.use_focus_ovd = bool(
            use_focus_ovd or self.focus_ovd_cfg.get('enable', False))
        super().__init__(*args, **kwargs)

        self.with_obj_align = with_obj_align
        self.text_fc = nn.Identity()
        self.visual_fc = nn.Sequential(
            nn.Linear(embed_dims, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, embed_dims)
        )
        # --------- 显式的对齐
        if self.with_obj_align:
            self.obj_align_branch = InstanceAlignmentEmbed()
        ###########################
        self.cross_input = nn.Linear(self.feat_channels, embed_dims)
        self.cross_norm = nn.LayerNorm(embed_dims)
        self.cross_attention = TwoWayTransformerModularYOLO(embedding_dim=embed_dims,
                                                            num_heads=8,
                                                            mlp_dim=cross_mlp_dim,
                                                            with_query_self_attn=False,   # 不利于拓展
                                                            with_cross_query_to_key=True,
                                                            with_cross_key_to_query=True,
                                                            depth=cross_num_layers)
        self.cross_output = nn.Linear(embed_dims, self.feat_channels)
        ###########################
        self.with_slot_embed = with_slot_embed
        if self.with_slot_embed:
            self.class_slot_embeds = (
                nn.Embedding(num_embeddings=256, embedding_dim=embed_dims))
        self._init_focus_ovd_modules()
        self._init_focus_text_anchor_calibration()
        self._init_focus_fourier_head_gate()
        self._init_focus_text_logit_mixer()
        self._init_scale_semantic_calibration()
        self._init_gaussian_semantic_scale()
        self._init_counter_support_ratio()
        self._init_ep2_path_probe_dump()

    def _init_counter_support_ratio(self) -> None:
        cfg = dict(self.counter_support_ratio_cfg)
        self.counter_support_ratio = CounterSupportEvidenceRatio(
            embed_dims=self.embed_dims,
            rank=int(cfg.get('rank', 8)),
            enable=bool(cfg.get('enable', False)),
            init_strength=float(cfg.get('init_strength', 0.0)),
        )

    def _apply_counter_support_ratio(
            self,
            positive_logits: Tensor,
            query_embeds: Tensor,
            support_feats: Tensor,
            support_labels: Tensor) -> Tensor:
        return self.counter_support_ratio(
            positive_logits=positive_logits,
            query_embeds=query_embeds,
            support_feats=support_feats,
            support_labels=support_labels,
            visual_fc=self.visual_fc,
        )

    def _init_ep2_path_probe_dump(self) -> None:
        cfg = (
            self.gaussian_semantic_scale_cfg.get('ep2_path_probe')
            or self.scale_semantic_calibration_cfg.get('ep2_path_probe')
            or {})
        cfg = dict(cfg)
        self.ep2_path_probe_enable = bool(cfg.get('enable', False))
        self.ep2_path_probe_output_csv = cfg.get('output_csv')
        self.ep2_path_probe_class_pairs = tuple(cfg.get('class_pairs', ()))
        self.ep2_path_probe_max_locations = int(
            cfg.get('max_locations_per_level', 128))

    def _ep2_path_probe_prior_source(self):
        gaussian_valid = getattr(
            self, 'gaussian_semantic_valid_mask',
            torch.zeros(0, dtype=torch.bool))
        if bool(torch.any(gaussian_valid).detach().cpu().item()):
            return (
                self.gaussian_semantic_scale_class_names,
                self.gaussian_semantic_log_area_mean,
                self.gaussian_semantic_log_area_std,
                self.gaussian_semantic_valid_mask,
            )
        return (
            self.scale_semantic_calibration_class_names,
            self.scale_semantic_log_area_mean,
            self.scale_semantic_log_area_std,
            self.scale_semantic_valid_mask,
        )

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
        class_names, means, stds, valid = self._ep2_path_probe_prior_source()
        rows = build_ep2_path_probe_rows(
            cls_logits=cls_logits[:, :int(self.num_classes)],
            decoded_bboxes=decoded_tensor,
            class_names=class_names,
            class_pairs=self.ep2_path_probe_class_pairs,
            log_area_mean=means,
            log_area_std=stds,
            valid_mask=valid,
            img_meta=img_meta,
            level_idx=level_idx,
            max_locations=self.ep2_path_probe_max_locations)
        append_ep2_path_probe_rows(self.ep2_path_probe_output_csv, rows)

    def _init_scale_semantic_calibration(self) -> None:
        """Initialize optional source-excluded scale-semantic calibration."""
        cfg = dict(self.scale_semantic_calibration_cfg)
        self.scale_semantic_calibration_enable = bool(
            cfg.get('enable', False))
        self.scale_semantic_calibration_z_margin = float(
            cfg.get('z_margin', cfg.get('log_area_z_margin', 4.0)))
        self.scale_semantic_calibration_lambda = float(
            cfg.get('downweight_lambda', cfg.get('lambda', 0.25)))
        self.scale_semantic_calibration_mode = str(
            cfg.get('mode', 'score_multiply'))
        self.scale_semantic_calibration_min_score = float(
            cfg.get('min_score', 0.0))
        self.scale_semantic_dump_topk_jsonl = cfg.get('dump_topk_jsonl')
        self.scale_semantic_dump_topk_k = int(cfg.get('dump_topk_k', 0) or 0)
        self.scale_semantic_calibration_debug = {
            'enable': self.scale_semantic_calibration_enable,
            'num_classes': int(self.num_classes),
            'num_valid_priors': 0,
        }
        if self.scale_semantic_dump_topk_jsonl:
            os.makedirs(
                os.path.dirname(self.scale_semantic_dump_topk_jsonl) or '.',
                exist_ok=True)

        class_names = tuple(
            cfg.get('class_names')
            or DEFAULT_DOTA2_CLASSES[:int(self.num_classes)])
        if len(class_names) < int(self.num_classes):
            class_names = class_names + tuple(
                str(idx) for idx in range(len(class_names), int(self.num_classes)))
        self.scale_semantic_calibration_class_names = class_names

        log_mean = torch.zeros(int(self.num_classes), dtype=torch.float32)
        log_std = torch.ones(int(self.num_classes), dtype=torch.float32)
        valid = torch.zeros(int(self.num_classes), dtype=torch.bool)

        priors_csv = cfg.get('class_area_priors_csv')
        if self.scale_semantic_calibration_enable:
            if not priors_csv:
                raise ValueError(
                    'scale_semantic_calibration.enable=True requires '
                    'class_area_priors_csv')
            name_to_idx = {
                str(name): idx for idx, name in enumerate(class_names)
            }
            with open(priors_csv, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    cls_name = row.get('class')
                    if cls_name not in name_to_idx:
                        continue
                    cls_idx = name_to_idx[cls_name]
                    mean = float(row.get('log_area_mean') or 0.0)
                    std = max(float(row.get('log_area_std') or 0.0), 1e-6)
                    log_mean[cls_idx] = mean
                    log_std[cls_idx] = std
                    valid[cls_idx] = True
            self.scale_semantic_calibration_debug.update({
                'class_area_priors_csv': str(priors_csv),
                'z_margin': self.scale_semantic_calibration_z_margin,
                'downweight_lambda': self.scale_semantic_calibration_lambda,
                'mode': self.scale_semantic_calibration_mode,
                'min_score': self.scale_semantic_calibration_min_score,
                'dump_topk_jsonl': self.scale_semantic_dump_topk_jsonl,
                'dump_topk_k': self.scale_semantic_dump_topk_k,
                'num_valid_priors': int(valid.sum().item()),
                'valid_classes': [
                    class_names[idx]
                    for idx in torch.nonzero(valid, as_tuple=False).view(-1).tolist()
                ],
            })

        self.register_buffer(
            'scale_semantic_log_area_mean', log_mean, persistent=False)
        self.register_buffer(
            'scale_semantic_log_area_std', log_std, persistent=False)
        self.register_buffer(
            'scale_semantic_valid_mask', valid, persistent=False)
        self._last_scale_semantic_calibration_debug = dict(
            self.scale_semantic_calibration_debug)

    def _init_gaussian_semantic_scale(self) -> None:
        """Initialize G-S3C class-conditional Gaussian log-area modeling."""
        cfg = dict(self.gaussian_semantic_scale_cfg)
        self.gaussian_semantic_scale_enable = bool(
            cfg.get('enable', False))
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
            cfg.get('domain_mode', cfg.get('domain', 'ovd')))
        self.gaussian_semantic_scale_adapter_nonpositive_delta = bool(
            cfg.get('adapter_nonpositive_delta', True))
        self.gaussian_semantic_scale_preserve_s3c_guard = bool(
            cfg.get('preserve_s3c_guard', True))
        self.gaussian_semantic_scale_guard_lambda = float(
            cfg.get('guard_lambda', 0.25))

        class_names = tuple(
            cfg.get('class_names')
            or self.scale_semantic_calibration_cfg.get('class_names')
            or DEFAULT_DOTA2_CLASSES[:int(self.num_classes)])
        if len(class_names) < int(self.num_classes):
            class_names = class_names + tuple(
                str(idx)
                for idx in range(len(class_names), int(self.num_classes)))
        priors_csv = (
            cfg.get('class_area_priors_csv')
            or self.scale_semantic_calibration_cfg.get('class_area_priors_csv'))
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
        self.gaussian_semantic_scale_class_names = tuple(class_names)

        self.gaussian_semantic_scale_debug = {
            'enable': self.gaussian_semantic_scale_enable,
            'num_classes': int(self.num_classes),
            'num_valid_priors': 0,
            'domain_mode': self.gaussian_semantic_scale_domain_mode,
            'gaussian_energy_mode': self.gaussian_semantic_scale_mode,
            'preserve_s3c_guard': (
                self.gaussian_semantic_scale_preserve_s3c_guard),
        }
        if self.gaussian_semantic_scale_enable:
            if not priors_csv:
                raise ValueError(
                    'gaussian_semantic_scale.enable=True requires '
                    'class_area_priors_csv or scale_semantic_calibration '
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
                'guard_lambda': self.gaussian_semantic_scale_guard_lambda,
                'num_valid_priors': int(valid.sum().item()),
                'valid_classes': [
                    self.gaussian_semantic_scale_class_names[idx]
                    for idx in torch.nonzero(
                        valid, as_tuple=False).view(-1).tolist()
                ],
            })

        self.register_buffer(
            'gaussian_semantic_log_area_mean', log_mean, persistent=False)
        self.register_buffer(
            'gaussian_semantic_log_area_std', log_std, persistent=False)
        self.register_buffer(
            'gaussian_semantic_valid_mask', valid, persistent=False)
        self.register_buffer(
            'gaussian_semantic_scale_z0_per_class',
            z0_per_class,
            persistent=False)
        self.register_buffer(
            'gaussian_semantic_scale_beta_per_class',
            beta_per_class,
            persistent=False)
        self.gaussian_semantic_scale_adapter = None
        if (self.gaussian_semantic_scale_enable
                and self.gaussian_semantic_scale_mode == 'logit_adapter'):
            self.gaussian_semantic_scale_adapter = GaussianScaleLogitAdapter(
                hidden=int(cfg.get('adapter_hidden', 64)),
                nonpositive_delta=(
                    self.gaussian_semantic_scale_adapter_nonpositive_delta))
        self._last_gaussian_semantic_scale_debug = dict(
            self.gaussian_semantic_scale_debug)

    def _apply_gaussian_semantic_scale(self,
                                       scores: Tensor,
                                       bbox_pred: Tensor,
                                       angle_pred: Tensor,
                                       priors: Tensor,
                                       img_shape,
                                       img_meta=None,
                                       level_idx: int = -1) -> Tensor:
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
        if isinstance(decoded_bboxes, RotatedBoxes):
            decoded_tensor = decoded_bboxes.tensor
        else:
            decoded_tensor = decoded_bboxes
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
                adapter_delta = adapter(
                    abs_z=stats['abs_z'],
                    signed_z=stats['z'],
                    gaussian_log_prob=stats['log_prob'],
                    log_std=stats['log_std'],
                    valid_mask=stats['valid'])
                if self.gaussian_semantic_scale_adapter_nonpositive_delta:
                    adapter_delta = torch.clamp(adapter_delta, max=0.0)
                delta = delta + adapter_delta
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
            'last_num_valid_pairs': int(stats['valid'].sum().detach().cpu().item()),
            'last_penalty_mean': penalty_mean,
            'last_penalty_max': penalty_max,
            'last_area_min': float(area.min().detach().cpu().item()),
            'last_area_max': float(area.max().detach().cpu().item()),
            'level_idx': int(level_idx),
            'image_id': (img_meta or {}).get('img_id'),
        }
        return calibrated

    def _apply_scale_semantic_calibration(self,
                                          scores: Tensor,
                                          bbox_pred: Tensor,
                                          angle_pred: Tensor,
                                          priors: Tensor,
                                          img_shape,
                                          img_meta=None,
                                          level_idx: int = -1) -> Tensor:
        if (not self.scale_semantic_calibration_enable
                or scores.numel() == 0):
            return scores
        num_score_classes = int(scores.shape[1])
        valid = self.scale_semantic_valid_mask.to(
            device=scores.device)[:num_score_classes]
        if not torch.any(valid):
            return scores

        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
        decoded_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)
        decoded_bboxes = self.bbox_coder.decode(
            priors, decoded_pred, max_shape=img_shape)
        if isinstance(decoded_bboxes, RotatedBoxes):
            decoded_tensor = decoded_bboxes.tensor
        else:
            decoded_tensor = decoded_bboxes
        if decoded_tensor.numel() == 0 or decoded_tensor.shape[-1] < 4:
            return scores

        area = (decoded_tensor[:, 2].abs() * decoded_tensor[:, 3].abs()).clamp(
            min=1e-6)
        log_area = torch.log(area)
        means = self.scale_semantic_log_area_mean.to(
            device=scores.device, dtype=scores.dtype)[:num_score_classes]
        stds = self.scale_semantic_log_area_std.to(
            device=scores.device, dtype=scores.dtype)[:num_score_classes].clamp(
                min=1e-6)
        z = (log_area[:, None].to(scores.dtype) - means[None, :]) / stds[None, :]
        mask = (
            valid[None, :]
            & (torch.abs(z) >= self.scale_semantic_calibration_z_margin))
        if self.scale_semantic_calibration_min_score > 0:
            mask = mask & (scores >= self.scale_semantic_calibration_min_score)
        if not torch.any(mask):
            self._last_scale_semantic_calibration_debug = {
                **self.scale_semantic_calibration_debug,
                'last_num_locations': int(scores.shape[0]),
                'last_num_flags': 0,
            }
            self._dump_scale_semantic_topk(
                before_scores=scores,
                after_scores=scores,
                area=area,
                z=z,
                mask=mask,
                img_meta=img_meta,
                level_idx=level_idx)
            return scores

        lam = max(min(float(self.scale_semantic_calibration_lambda), 1.0), 0.0)
        calibrated = scores.clone()
        if self.scale_semantic_calibration_mode == 'logit_shift':
            eps = torch.finfo(scores.dtype).eps
            logits = torch.logit(calibrated.clamp(eps, 1 - eps))
            logits = torch.where(
                mask, logits + math.log(max(lam, eps)), logits)
            calibrated = torch.sigmoid(logits)
        else:
            calibrated = torch.where(mask, calibrated * lam, calibrated)

        self._last_scale_semantic_calibration_debug = {
            **self.scale_semantic_calibration_debug,
            'last_num_locations': int(scores.shape[0]),
            'last_num_flags': int(mask.sum().detach().cpu().item()),
            'last_flag_rate': float(mask.float().mean().detach().cpu().item()),
            'last_area_min': float(area.min().detach().cpu().item()),
            'last_area_max': float(area.max().detach().cpu().item()),
        }
        self._dump_scale_semantic_topk(
            before_scores=scores,
            after_scores=calibrated,
            area=area,
            z=z,
            mask=mask,
            img_meta=img_meta,
            level_idx=level_idx)
        return calibrated

    def _dump_scale_semantic_topk(self,
                                  before_scores: Tensor,
                                  after_scores: Tensor,
                                  area: Tensor,
                                  z: Tensor,
                                  mask: Tensor,
                                  img_meta=None,
                                  level_idx: int = -1) -> None:
        path = self.scale_semantic_dump_topk_jsonl
        k = int(self.scale_semantic_dump_topk_k)
        if not path or k <= 0 or before_scores.numel() == 0:
            return
        try:
            rows = []
            for stage, score_tensor in (
                    ('before', before_scores), ('after', after_scores)):
                flat = score_tensor.reshape(-1)
                topk = min(k, int(flat.numel()))
                values, flat_indices = torch.topk(flat, topk)
                for rank, (value, flat_idx) in enumerate(
                        zip(values.detach().cpu().tolist(),
                            flat_indices.detach().cpu().tolist()), start=1):
                    loc_idx = int(flat_idx) // int(score_tensor.shape[1])
                    class_idx = int(flat_idx) % int(score_tensor.shape[1])
                    class_name = (
                        self.scale_semantic_calibration_class_names[class_idx]
                        if class_idx < len(
                            self.scale_semantic_calibration_class_names)
                        else str(class_idx))
                    rows.append({
                        'stage': stage,
                        'rank': rank,
                        'process_rank': int(os.environ.get('RANK', '0') or 0),
                        'local_rank': int(
                            os.environ.get('LOCAL_RANK', '0') or 0),
                        'image_id': (img_meta or {}).get('img_id'),
                        'img_path': (img_meta or {}).get('img_path'),
                        'level_idx': int(level_idx),
                        'location_idx': loc_idx,
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'score': float(value),
                        'area': float(area[loc_idx].detach().cpu().item()),
                        'log_area_z': float(
                            z[loc_idx, class_idx].detach().cpu().item()),
                        's3c_flagged': bool(
                            mask[loc_idx, class_idx].detach().cpu().item()),
                    })
            with open(path, 'a', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
        except Exception as exc:  # noqa: BLE001
            self._last_scale_semantic_calibration_debug = {
                **getattr(self, '_last_scale_semantic_calibration_debug', {}),
                'dump_topk_error': repr(exc),
            }

    def _init_focus_text_anchor_calibration(self) -> None:
        cfg = dict(self.focus_text_anchor_calibration_cfg)
        class_names = cfg.get(
            'class_names',
            DEFAULT_DOTA2_CLASSES[:self.num_classes])
        tac_num_classes = int(cfg.get(
            'num_classes',
            len(class_names) if class_names else self.num_classes))
        self.focus_text_anchor_calibration = TextAnchorCalibration(
            num_classes=tac_num_classes,
            enable=cfg.get('enable', False),
            use_alpha=cfg.get('use_alpha', True),
            use_beta=cfg.get('use_beta', False),
            alpha_init=cfg.get('alpha_init', 0.0),
            beta_init=cfg.get('beta_init', 0.0),
            alpha_bound=cfg.get('alpha_bound', 0.05),
            beta_bound=cfg.get('beta_bound', 0.20),
            apply_to_classes=cfg.get('apply_to_classes', None),
            class_names=class_names,
            anchor_weight=cfg.get('anchor_weight', 0.01),
            log_debug=cfg.get('log_debug', False))
        self._last_focus_tac_debug = (
            self.focus_text_anchor_calibration.debug_state())

    def _init_focus_fourier_head_gate(self) -> None:
        cfg = dict(self.focus_fourier_head_gate_cfg)
        class_names = cfg.get(
            'class_names',
            DEFAULT_DOTA2_CLASSES[:self.num_classes])
        gate_num_classes = int(cfg.get(
            'num_classes',
            len(class_names) if class_names else self.num_classes))
        self.focus_fourier_head_gate = FourierHeadLogitAdapter(
            num_classes=gate_num_classes,
            enable=cfg.get('enable', False),
            feature_mode=cfg.get('feature_mode', 'confidence'),
            harmonic_order=cfg.get('harmonic_order', 2),
            use_alpha=cfg.get('use_alpha', True),
            use_beta=cfg.get('use_beta', False),
            alpha_init=cfg.get('alpha_init', 0.0),
            beta_init=cfg.get('beta_init', 0.0),
            alpha_bound=cfg.get('alpha_bound', 0.03),
            beta_bound=cfg.get('beta_bound', 0.05),
            apply_to_classes=cfg.get('apply_to_classes', None),
            class_names=class_names,
            anchor_weight=cfg.get('anchor_weight', 0.01))
        self._last_focus_fourier_head_gate_debug = (
            self.focus_fourier_head_gate.debug_state())

    def _init_focus_text_logit_mixer(self) -> None:
        cfg = dict(self.focus_text_logit_mixer_cfg)
        class_names = cfg.get(
            'class_names',
            DEFAULT_DOTA2_CLASSES[:self.num_classes])
        mixer_num_classes = int(cfg.get(
            'num_classes',
            len(class_names) if class_names else self.num_classes))
        self.focus_text_logit_mixer = FocusTextLogitMixer(
            num_classes=mixer_num_classes,
            enable=cfg.get('enable', False),
            mode=cfg.get('mode', 'smooth'),
            use_gamma=cfg.get('use_gamma', True),
            use_beta=cfg.get('use_beta', False),
            gamma_init=cfg.get('gamma_init', 0.0),
            beta_init=cfg.get('beta_init', 0.0),
            gamma_bound=cfg.get('gamma_bound', 0.03),
            beta_bound=cfg.get('beta_bound', 0.05),
            apply_to_classes=cfg.get('apply_to_classes', None),
            class_names=class_names,
            sim_threshold=cfg.get('sim_threshold', 0.0),
            sim_power=cfg.get('sim_power', 1.0),
            topk=cfg.get('topk', 0),
            anchor_weight=cfg.get('anchor_weight', 0.01))
        self._last_focus_text_logit_mixer_debug = (
            self.focus_text_logit_mixer.debug_state())

    def _init_focus_ovd_modules(self) -> None:
        self.focus_ovd_enable = bool(
            self.use_focus_ovd and self.focus_ovd_cfg.get('enable', True))
        self.focus_orientation_learner = None
        self.focus_support_adapter = None
        self.focus_text_adapter = None
        self.focus_dual_support_fusion = None
        self.focus_cls_head = None
        self.focus_head_residual_enabled = True
        self.focus_eqtext_enabled = False
        self.focus_dual_fusion_enabled = False
        if not self.focus_ovd_enable:
            return

        orientation_cfg = dict(
            self.focus_ovd_cfg.get(
                'orientation',
                self.focus_ovd_cfg.get('orientation_cfg', {})))
        adapter_cfg = dict(
            self.focus_ovd_cfg.get(
                'adapter',
                self.focus_ovd_cfg.get('adapter_cfg', {})))
        eqtext_cfg = dict(self.focus_ovd_cfg.get('eqtext', {}))
        dual_cfg = dict(self.focus_ovd_cfg.get('dual_fusion', {}))
        harmonic_orders = tuple(
            orientation_cfg.get('harmonic_orders', (2, 4, 6)))
        self.focus_orientation_learner = FourierOrientationLearner(
            patch_size=orientation_cfg.get('patch_size', 7),
            num_angle_bins=orientation_cfg.get('num_angle_bins', 36),
            harmonic_orders=harmonic_orders,
            confidence_type=orientation_cfg.get(
                'confidence_type', 'peak_entropy'),
            min_confidence=orientation_cfg.get('min_confidence', 0.15),
            detach_orientation=orientation_cfg.get(
                'detach_orientation', True))
        self.focus_head_residual_enabled = bool(
            adapter_cfg.get('enable', True))
        self.focus_support_adapter = FourierSupportResidualAdapter(
            support_dim=self.embed_dims,
            code_dim=2 * len(harmonic_orders),
            class_names=None,
            apply_to_classes=adapter_cfg.get(
                'apply_to_classes', ('small-vehicle',)),
            allow_all_classes=adapter_cfg.get('allow_all_classes', False),
            low_rank=adapter_cfg.get('low_rank', 16),
            alpha_init=adapter_cfg.get('alpha_init', 0.0),
            alpha_max=adapter_cfg.get('alpha_max', 0.10),
            max_delta_norm_ratio=adapter_cfg.get(
                'max_delta_norm_ratio', 0.05),
            normalize_after_residual=adapter_cfg.get(
                'normalize_after_residual', True))
        self.focus_eqtext_enabled = bool(eqtext_cfg.get('enable', False))
        if self.focus_eqtext_enabled:
            self.focus_text_adapter = build_focus_text_adapter(
                eqtext_cfg=eqtext_cfg,
                support_dim=self.embed_dims,
                code_dim=2 * len(harmonic_orders),
                class_names=None)
        self.focus_dual_fusion_enabled = bool(dual_cfg.get('enable', False))
        if self.focus_dual_fusion_enabled:
            self.focus_dual_support_fusion = FocusDualSupportFusion(
                visual_weight_init=dual_cfg.get('visual_weight_init', 0.9),
                text_weight_init=dual_cfg.get('text_weight_init', 0.1),
                max_text_weight=dual_cfg.get('max_text_weight', 0.2))
        self.focus_cls_head = OrientationConditionedContrastiveEmbed()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.rtm_cls = nn.ModuleList()
        self.rtm_cls_heads = nn.ModuleList()
        for n in range(len(self.prior_generator.strides)):
            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.embed_dims,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_cls_heads.append(ContrastiveEmbed())


    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()

    def loss(self,
             x: Tuple[Tensor],
             batch_data_samples: SampleList,
             support_feats,
             support_labels,
             support_slot_labels,
             obj_embeds_list,
             obj_labels_list,
             return_outs=False,
             **kwargs) -> dict:
        # len(support_feats_list[0]) = support_shot * num_in_classes + num_out_instance(或许没有)
        device = x[0].device
        num_classes = kwargs['num_classes']
        num_in_classes = kwargs['num_in_classes']
        support_shot = kwargs['support_shot']
        ###          ---- 对support进行padding
        max_obj_embed_len = max([len(e) for e in obj_embeds_list])
        obj_embeds = []
        obj_labels = []
        for f, l in zip(obj_embeds_list, obj_labels_list):
            if len(f) == max_obj_embed_len:
                obj_embeds.append(f)
                obj_labels.append(l)
                continue
            n_pad = max_obj_embed_len - len(f)
            pad_feats = torch.zeros([n_pad, f.shape[-1]]).float().to(device)
            pad_labels = torch.ones(n_pad).long().to(device) * -1
            obj_embeds.append(torch.cat([f, pad_feats]))
            obj_labels.append(torch.cat([l, pad_labels]))
        obj_embeds = torch.stack(obj_embeds)
        obj_labels = torch.stack(obj_labels)

        ###################################
        losses = dict()
        outs = self(x,
                    support_feats,
                    support_labels,
                    support_slot_labels,
                    obj_embeds,
                    **kwargs)
        out_obj_embeds = outs[-1]
        out_support_feats = outs[-2]
        outs = outs[:-2]

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore,
                              out_obj_embeds,
                              obj_labels,
                              support_feats,
                              support_labels,
                              support_slot_labels,
                              )
        losses_head = self.loss_by_feat(*loss_inputs, **kwargs)
        losses.update(losses_head)
        if not return_outs:
            return losses
        else:
            return losses, outs

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                support_feats,
                support_labels,
                support_slot_labels,
                rescale: bool = False,
                **kwargs) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        device = x[0].device
        ###############
        outs = self(x,
                    support_feats,
                    support_labels,
                    support_slot_labels,
                    **kwargs)
        out_support_feats = outs[-1]
        pred_embeds = outs[-2]
        outs = outs[:-2]
        ###############
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def forward(self,
                feats: Tuple[Tensor, ...],
                support_feats,
                support_labels,
                support_slot_labels,
                obj_embeds=None,
                **kwargs) -> tuple:
        device = feats[0].device
        ################# support feats 和 obj_embeddings进行映射
        out_obj_embeds = obj_embeds
        #######################
        spatial_shapes = [f.shape[2:] for f in feats]
        flatten_feats = torch.cat([f.flatten(2) for f in feats], dim=-1).permute(0, 2, 1)

        B, N, C = flatten_feats.shape
        in_flatten_feats = self.cross_norm(self.cross_input(flatten_feats))
        ########### ----- 建立label到[0, n_label]的随机映射，并获得position embeddings
        ########### 2, 2, 1, 1, 0, 0 -> [1, 2, 0] -> 1, 1, 0, 0, 2, 2
        if self.with_slot_embed:
            label_set = torch.unique(support_slot_labels).detach().cpu().numpy().astype(np.int64)
            label_set = np.random.permutation(label_set).tolist()
            permuted_labels = torch.zeros_like(support_slot_labels).long().to(device)
            for p_l, l in enumerate(label_set):
                permuted_labels[support_slot_labels == l] = p_l
            permuted_labels = permuted_labels.contiguous()
            slot_embeds = self.class_slot_embeds(permuted_labels)
            support_feats = support_feats + slot_embeds

        out_support_feats, out_flatten_feats = self.cross_attention(support_feats, in_flatten_feats)
        out_flatten_feats = self.cross_output(out_flatten_feats)
        out_flatten_feats = out_flatten_feats.permute(0, 2, 1)
        out_feats = []
        start = 0
        for spatial_shape in spatial_shapes:
            H, W = spatial_shape
            out_feats.append(out_flatten_feats[:, :, start: start + H*W].
                                   reshape(B, C, H, W))
            start += H*W
        #######################
        pred_embeds = []
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        focus_debug_by_level = []
        fourier_head_gate_debug_by_level = []
        for idx, (x, stride) in enumerate(
                zip(out_feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            pred_embed = self.rtm_cls[idx](cls_feat)
            if bool(getattr(self, 'focus_ovd_enable', False)
                    and kwargs.get('focus_ovd_enable', True)):
                focus_kwargs = dict(kwargs)
                text_support_feats = focus_kwargs.pop(
                    'text_support_feats', None)
                visual_support_feats = focus_kwargs.pop(
                    'visual_support_feats',
                    focus_kwargs.pop('cls_support_feats', None))
                cls_logit = self.focus_cls_head(
                    pred_embed,
                    out_support_feats,
                    support_labels,
                    visual_fc=self.visual_fc,
                    text_fc=self.text_fc,
                    log_scale=self.rtm_cls_heads[idx].log_scale_t1,
                    bias=self.rtm_cls_heads[idx].bias_t1,
                    orientation_learner=self.focus_orientation_learner,
                    support_adapter=self.focus_support_adapter,
                    text_support_adapter=self.focus_text_adapter,
                    dual_support_fusion=self.focus_dual_support_fusion,
                    visual_support_feats=visual_support_feats,
                    text_support_feats=text_support_feats,
                    eqtext_enabled=self.focus_eqtext_enabled,
                    dual_fusion_enabled=self.focus_dual_fusion_enabled,
                    head_residual_enabled=self.focus_head_residual_enabled,
                    focus_enabled=True,
                    **focus_kwargs)
                focus_debug_by_level.append(
                    dict(self.focus_cls_head.last_focus_debug or {}))
            else:
                cls_logit = self.rtm_cls_heads[idx](pred_embed,
                                                    out_support_feats,
                                                    support_labels,
                                                    visual_fc=self.visual_fc,
                                                    text_fc=self.text_fc,
                                                    **kwargs)
                focus_debug_by_level.append(None)
            cls_score = self._apply_counter_support_ratio(
                cls_logit,
                pred_embed,
                out_support_feats,
                support_labels,
            )
            mixer = getattr(self, 'focus_text_logit_mixer', None)
            if mixer is not None and getattr(mixer, 'enable', False):
                with torch.no_grad():
                    mixer_support_feats = self.visual_fc(out_support_feats)
                cls_score = mixer(
                    cls_score,
                    mixer_support_feats,
                    support_labels)
            gate = getattr(self, 'focus_fourier_head_gate', None)
            if gate is not None and getattr(gate, 'enable', False):
                level_debug = focus_debug_by_level[-1] or {}
                orientation_debug = level_debug.get('orientation') or {}
                cls_score = gate(
                    cls_score,
                    theta=orientation_debug.get('theta'),
                    confidence=orientation_debug.get('confidence'))
                fourier_head_gate_debug_by_level.append(
                    dict(gate.last_debug or {}))
            else:
                fourier_head_gate_debug_by_level.append(None)
            if getattr(self, 'focus_text_anchor_calibration', None) is not None:
                cls_score = self.focus_text_anchor_calibration(cls_score)
            ########################################
            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)


            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            angle_pred = self.rtm_ang[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
            pred_embeds.append(pred_embed)
        self._last_focus_debug_by_level = focus_debug_by_level
        self._last_focus_fourier_head_gate_debug_by_level = (
            fourier_head_gate_debug_by_level)
        if getattr(self, 'focus_text_anchor_calibration', None) is not None:
            self._last_focus_tac_debug = (
                self.focus_text_anchor_calibration.debug_state())
        if getattr(self, 'focus_text_logit_mixer', None) is not None:
            self._last_focus_text_logit_mixer_debug = (
                self.focus_text_logit_mixer.debug_state())
        if out_obj_embeds is not None:
            return (tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(pred_embeds),
                    out_support_feats, out_obj_embeds)
        else:
            return (tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(pred_embeds),
                    out_support_feats)

    ############## --------------- Loss 计算相关 ---------------
    def _focus_zero_loss(self, cls_scores: List[Tensor]) -> Tensor:
        if cls_scores:
            return cls_scores[0].sum() * 0.0
        return torch.tensor(0.0)

    def _focus_mask_for_level(self, masks, level_idx: int,
                              cls_score: Tensor) -> Optional[Tensor]:
        if masks is None:
            return None
        if isinstance(masks, dict):
            masks = masks.get(level_idx, masks.get(str(level_idx), None))
        elif isinstance(masks, (list, tuple)):
            if level_idx >= len(masks):
                return None
            masks = masks[level_idx]
        if masks is None:
            return None

        mask = torch.as_tensor(
            masks, device=cls_score.device, dtype=torch.bool)
        batch, _, height, width = cls_score.shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(batch, -1, -1)
        elif mask.dim() == 3:
            if mask.shape[0] == 1 and batch > 1:
                mask = mask.expand(batch, -1, -1)
        elif mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
            if mask.shape[0] == 1 and batch > 1:
                mask = mask.expand(batch, -1, -1)
        else:
            raise ValueError(
                f"focus target mask level {level_idx} has incompatible "
                f"shape {tuple(mask.shape)} for cls_score {tuple(cls_score.shape)}")
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(
                mask.float().unsqueeze(1),
                size=(height, width),
                mode='nearest')[:, 0].to(dtype=torch.bool)
        return mask

    def loss_focus_by_feat(self,
                           cls_scores: List[Tensor],
                           focus_target_masks: Optional[dict] = None,
                           baseline_cls_scores: Optional[List[Tensor]] = None,
                           sv_class_index: int = -1,
                           **kwargs) -> dict:
        cfg = self.focus_losses
        zero = self._focus_zero_loss(cls_scores)
        losses = {
            'loss_focus_support_distill': zero,
            'loss_focus_text_anchor': zero,
            'loss_focus_anti': zero,
            'loss_focus_preserve': zero,
            'loss_focus_migration': zero,
        }
        assignment_debug = kwargs.get('focus_assignment_debug', {}) or {}
        point_debug = {
            'sv_class_index': int(sv_class_index),
            'num_focus_images_with_targets': int(
                assignment_debug.get('num_focus_images_with_targets', 0) or 0),
            'num_focus_anti_targets': int(
                assignment_debug.get('num_focus_anti_targets', 0) or 0),
            'num_focus_preserve_targets': int(
                assignment_debug.get('num_focus_preserve_targets', 0) or 0),
            'num_focus_anti_points': 0,
            'num_focus_preserve_points': 0,
            'focus_assignment_coverage': float(
                assignment_debug.get('focus_assignment_coverage', 0.0) or 0.0),
            'no_focus_targets': True,
        }
        if not cfg.get('enable', False):
            losses['loss_focus_total'] = zero
            self._last_focus_loss_debug = point_debug
            return losses

        support_weight = float(cfg.get('support_distill_weight', 0.0))
        if support_weight > 0.0:
            debug_losses = []
            for debug in getattr(self, '_last_focus_debug_by_level', []) or []:
                if not debug or not debug.get('enabled', False):
                    continue
                adapter_debug = debug.get('adapter') or {}
                delta_norm = adapter_debug.get('delta_norm')
                if delta_norm is not None and hasattr(delta_norm, 'mean'):
                    debug_losses.append(delta_norm.mean())
            if debug_losses:
                losses['loss_focus_support_distill'] = (
                    torch.stack(debug_losses).mean() * support_weight)

        text_anchor_weight = float(cfg.get('text_anchor_weight', 0.0))
        if text_anchor_weight > 0.0:
            text_anchor_losses = []
            for debug in getattr(self, '_last_focus_debug_by_level', []) or []:
                if not debug or not debug.get('enabled', False):
                    continue
                eqtext_debug = debug.get('eqtext') or {}
                anchor_loss = eqtext_debug.get('text_anchor_loss_raw')
                if anchor_loss is not None and hasattr(anchor_loss, 'mean'):
                    text_anchor_losses.append(anchor_loss.mean())
            if text_anchor_losses:
                losses['loss_focus_text_anchor'] = (
                    torch.stack(text_anchor_losses).mean()
                    * text_anchor_weight)

        mode = str(cfg.get('target_mapping_mode', 'none'))
        use_targets = mode in {'spatial_region', 'pre_nms_provenance'}
        sv_idx = int(sv_class_index)
        if use_targets and focus_target_masks is not None:
            anti_masks = focus_target_masks.get('anti_mask_by_level')
            preserve_masks = focus_target_masks.get('preserve_mask_by_level')
            anti_weight = float(cfg.get('anti_attractor_weight', 0.0))
            preserve_weight = float(cfg.get('preserve_weight', 0.0))
            anti_terms = []
            preserve_terms = []
            for level_idx, cls_score in enumerate(cls_scores):
                if anti_weight > 0.0:
                    anti_mask = self._focus_mask_for_level(
                        anti_masks, level_idx, cls_score)
                    if anti_mask is not None:
                        point_debug['num_focus_anti_points'] += int(
                            anti_mask.sum().detach().item())
                    anti_terms.append(anti_attractor_loss(
                        cls_score,
                        sv_idx,
                        anti_mask,
                        margin=float(cfg.get('anti_margin', 0.0)),
                        weight=anti_weight))
                if preserve_weight > 0.0:
                    preserve_mask = self._focus_mask_for_level(
                        preserve_masks, level_idx, cls_score)
                    if preserve_mask is not None:
                        point_debug['num_focus_preserve_points'] += int(
                            preserve_mask.sum().detach().item())
                    preserve_terms.append(preserve_loss(
                        cls_score,
                        sv_idx,
                        preserve_mask,
                        weight=preserve_weight))
            if anti_terms:
                losses['loss_focus_anti'] = torch.stack(anti_terms).mean()
            if preserve_terms:
                losses['loss_focus_preserve'] = torch.stack(
                    preserve_terms).mean()

        migration_weight = float(cfg.get('migration_weight', 0.0))
        if migration_weight > 0.0 and baseline_cls_scores is not None:
            migration_terms = []
            temperature = float(cfg.get('migration_temperature', 1.0))
            for cls_score, baseline in zip(cls_scores, baseline_cls_scores):
                migration_terms.append(migration_kl_loss(
                    cls_score,
                    baseline,
                    temperature=temperature,
                    weight=migration_weight))
            if migration_terms:
                losses['loss_focus_migration'] = torch.stack(
                    migration_terms).mean()

        losses['loss_focus_total'] = (
            losses['loss_focus_support_distill']
            + losses['loss_focus_text_anchor']
            + losses['loss_focus_anti']
            + losses['loss_focus_preserve']
            + losses['loss_focus_migration'])
        point_debug['no_focus_targets'] = not (
            point_debug['num_focus_anti_points'] > 0
            or point_debug['num_focus_preserve_points'] > 0)
        point_debug['loss_focus_support_distill'] = float(
            losses['loss_focus_support_distill'].detach().item())
        point_debug['loss_focus_text_anchor'] = float(
            losses['loss_focus_text_anchor'].detach().item())
        point_debug['loss_focus_anti'] = float(
            losses['loss_focus_anti'].detach().item())
        point_debug['loss_focus_preserve'] = float(
            losses['loss_focus_preserve'].detach().item())
        point_debug['loss_focus_migration'] = float(
            losses['loss_focus_migration'].detach().item())
        point_debug['loss_focus_total'] = float(
            losses['loss_focus_total'].detach().item())
        self._last_focus_loss_debug = point_debug
        return losses

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            assign_metrics: Tensor, stride: List[int],
                            **kwargs):
        num_classes = kwargs['num_classes']
        num_in_classes = kwargs['num_in_classes']
        #######################################
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_out_channels = cls_score.shape[1]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, cls_out_channels).contiguous()

        if self.use_hbbox_loss:
            bbox_pred = bbox_pred.reshape(-1, 4)
        else:
            bbox_pred = bbox_pred.reshape(-1, 5)
        bbox_targets = bbox_targets.reshape(-1, 5)

        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        # ---- 如果包含SAM Obj，则num_classes=21, num_in_classes=20, cls_score的长度为21，sam_obj的label为20, bg为21
        bg_class_ind = num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets
            if self.use_hbbox_loss:
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(
                    pos_bbox_targets[:, :4])

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_angle = angle_pred.sum() * 0
            if self.loss_angle is not None:
                angle_pred = angle_pred.reshape(-1,
                                                self.angle_coder.encode_size)
                pos_angle_pred = angle_pred[pos_inds]
                pos_angle_target = pos_bbox_targets[:, 4:5]
                pos_angle_target = self.angle_coder.encode(pos_angle_target)
                if pos_angle_target.dim() == 2:
                    pos_angle_weight = pos_bbox_weight.unsqueeze(-1)
                else:
                    pos_angle_weight = pos_bbox_weight
                loss_angle = self.loss_angle(
                    pos_angle_pred,
                    pos_angle_target,
                    weight=pos_angle_weight,
                    avg_factor=1.0)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
            loss_angle = angle_pred.sum() * 0

        return (loss_cls, loss_bbox, loss_angle, assign_metrics.sum(),
                pos_bbox_weight.sum(), pos_bbox_weight.sum())

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     embed_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     obj_embeds=None,
                     obj_labels=None,
                     support_feats=None,
                     support_labels=None,
                     support_slot_labels=None,
                     **kwargs):
        losses = dict()
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_out_channels = cls_scores[0].shape[1]
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  cls_out_channels)
            for cls_score in cls_scores
        ], 1)

        decoded_bboxes = []
        decoded_hbboxes = []
        angle_preds_list = []
        for anchor, bbox_pred, angle_pred in zip(anchor_list[0], bbox_preds,
                                                 angle_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
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

        # flatten_bboxes is rbox, for target assign
        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, ins_labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets
        ##############################################################################
        ############## 计算contrastive loss
        need_dense_embeds = self.with_obj_align or self.use_ccl_loss
        if need_dense_embeds:
            # ----- n_level x [(B, D, H_i, W_i)] -> B x n_position x D
            all_pred_embeds = torch.cat([
                embed.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.embed_dims)
                for embed in embed_preds
            ], 1)
            # ----- n_level x [(B, H_i*W_i)] -> B x n_position
            all_labels = torch.cat(labels_list, dim=1)

        if self.with_obj_align:
            loss_align = torch.Tensor([0.0]).to(device)

            for i in range(num_imgs):
                assigned_labels = all_labels[i]
                pred_embeds = all_pred_embeds[i]
                s_labels = support_labels[i]
                s_feats = support_feats[i]

                bg_class_ind = kwargs['num_classes']
                pos_inds = ((assigned_labels >= 0)
                            & (assigned_labels < bg_class_ind)).nonzero().squeeze(1)

                # ----- Positive的预测embeddings
                pos_pred_embeds = pred_embeds[pos_inds]
                pos_labels = assigned_labels[pos_inds]
                # ----- 计算对齐损失
                loss_align_ = self.obj_align_branch(pos_pred_embeds,
                                                   s_feats,
                                                   pos_labels,
                                                   s_labels,
                                                   visual_fc=self.visual_fc,
                                                   text_fc=self.text_fc,
                                                   **kwargs)
                loss_align = loss_align + loss_align_
            loss_align = loss_align / num_imgs
            losses['loss_aln'] = loss_align
        if self.use_ccl_loss:
            loss_ccl = compute_ccl_loss(
                dense_embeds=self.visual_fc(all_pred_embeds),
                assigned_labels=all_labels,
                support_feats=self.visual_fc(support_feats),
                support_labels=support_labels,
                num_classes=kwargs['num_classes'],
                temperature=self.ccl_temperature)
            losses['loss_ccl'] = loss_ccl * self.ccl_loss_weight
        ##############################################################################

        if self.use_hbbox_loss:
            decoded_bboxes = decoded_hbboxes
        # ----loss_by_feat_single对每个层级的特征单独计算特征
        (losses_cls, losses_bbox, losses_angle, cls_avg_factors,
         bbox_avg_factors, angle_avg_factors) = multi_apply(
            self.loss_by_feat_single, cls_scores, decoded_bboxes,
            angle_preds_list, labels_list, label_weights_list,
            bbox_targets_list, assign_metrics_list,
            self.prior_generator.strides, **kwargs)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
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
            losses.update(dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox))

        if self.use_sv_dehub_loss and kwargs.get('sv_class_index', -1) >= 0:
            from M_AD.models.losses.sv_dehub_loss import compute_background_sv_dehub_loss
            sv_idx = int(kwargs['sv_class_index'])
            margin = float(kwargs.get('sv_dehub_margin', self.sv_dehub_margin))
            weight = float(kwargs.get('sv_dehub_loss_weight', self.sv_dehub_loss_weight))
            bg_ind = int(kwargs['num_classes'])
            dehub_sum = cls_scores[0].new_tensor(0.0)
            n_scale = 0
            for cls_score, labels in zip(cls_scores, labels_list):
                flat = cls_score.permute(0, 2, 3, 1).reshape(-1, cls_score.shape[1])
                lab = labels.reshape(-1)
                dloss, _ = compute_background_sv_dehub_loss(
                    flat, lab, sv_idx, margin=margin, bg_class_ind=bg_ind)
                dehub_sum = dehub_sum + dloss
                n_scale += 1
            if n_scale > 0:
                losses['loss_dehub'] = dehub_sum / n_scale * weight
            if not getattr(self, '_dbg_dehub_applied', False):
                self._dbg_dehub_applied = True
                try:
                    from tools.exp_sv_dehub_lite_train_gpu89.debug_log import dbg
                    dbg('C', 'Flex_Rrtmdet_head_v3_1.py:loss_by_feat',
                        'dehub loss branch',
                        {'applied': True, 'sv_idx': sv_idx,
                         'dehub_val': float(losses['loss_dehub'].detach())})
                except Exception:
                    pass
        elif self.use_sv_dehub_loss and not getattr(self, '_dbg_dehub_skipped', False):
            self._dbg_dehub_skipped = True
            try:
                from tools.exp_sv_dehub_lite_train_gpu89.debug_log import dbg
                dbg('C', 'Flex_Rrtmdet_head_v3_1.py:loss_by_feat',
                    'dehub skipped', {'sv_class_index': kwargs.get('sv_class_index', -1)})
            except Exception:
                pass
        if self.focus_losses.get('enable', False):
            focus_kwargs = dict(kwargs)
            focus_target_masks = focus_kwargs.pop('focus_target_masks', None)
            baseline_cls_scores = focus_kwargs.pop('baseline_cls_scores', None)
            sv_class_index = focus_kwargs.pop('sv_class_index', -1)
            losses.update(self.loss_focus_by_feat(
                list(cls_scores),
                focus_target_masks=focus_target_masks,
                baseline_cls_scores=baseline_cls_scores,
                sv_class_index=sv_class_index,
                **focus_kwargs))
        if getattr(self, 'focus_text_anchor_calibration', None) is not None:
            losses['loss_focus_tac_anchor'] = (
                self.focus_text_anchor_calibration.anchor_regularizer(
                    cls_scores[0]))
            self._last_focus_tac_debug = (
                self.focus_text_anchor_calibration.debug_state())
        gate = getattr(self, 'focus_fourier_head_gate', None)
        if gate is not None and getattr(gate, 'enable', False):
            losses['loss_focus_fourier_head_anchor'] = (
                gate.anchor_regularizer(cls_scores[0]))
            self._last_focus_fourier_head_gate_debug = gate.debug_state()
        mixer = getattr(self, 'focus_text_logit_mixer', None)
        if mixer is not None and getattr(mixer, 'enable', False):
            losses['loss_focus_text_logit_mixer_anchor'] = (
                mixer.anchor_regularizer(cls_scores[0]))
            self._last_focus_text_logit_mixer_debug = mixer.debug_state()
        return losses

    def get_targets(self,
                    cls_scores: Tensor,
                    bbox_preds: Tensor,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs=True):
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])
        ######## ------------- 每张图片单独计算target，返回了实例标签
        (all_anchors, all_labels, all_ins_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics, sampling_results_list) = multi_apply(
            self._get_targets_single,
            cls_scores.detach(),
            bbox_preds.detach(),
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        ins_labels_list = images_to_levels(all_ins_labels, num_level_anchors)

        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        assign_metrics_list = images_to_levels(all_assign_metrics,
                                               num_level_anchors)

        return (anchors_list, labels_list, ins_labels_list, label_weights_list,
                bbox_targets_list, assign_metrics_list, sampling_results_list)

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs=True):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors)

        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        """
        sampling_result.pos_assigned_gt_inds: 每个pos分配到的gt, N_pos
        sampling_result.pos_inds：pos的anchor的id, N_pos
        sampling_result.neg_inds：neg的anchor的id, N_neg
        sampling_result.pos_is_gt: 可以忽略，因为sampling是一个伪采样器

        """

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((*anchors.size()[:-1], 5))
        labels = anchors.new_full((num_valid_anchors,),
                                  cls_scores.shape[-1],
                                  dtype=torch.long)
        ################ instance的标签
        ins_labels = anchors.new_full((num_valid_anchors,),
                                      -1,
                                      dtype=torch.long)
        ################
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            pos_bbox_targets = pos_bbox_targets.regularize_boxes(
                self.angle_version)
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            ################
            ins_gt_labels = gt_instances.ins_labels
            ins_labels[pos_inds] = ins_gt_labels[sampling_result.pos_assigned_gt_inds]
            ################
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # ---- 分配上的gt（理应每个gt都被分配到）
        # ---- assign_metrics中只有pos的>0，neg=0
        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[
                gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=cls_scores.shape[-1])
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors,
                                   inside_flags)
        return (anchors, labels, ins_labels, label_weights, bbox_targets, assign_metrics,
                sampling_result)

    ################################ Predict #################################

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

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
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
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

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_out_channels = cls_score.shape[0]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, cls_out_channels)
            self._dump_ep2_path_probe_logits(
                cls_logits=cls_score,
                bbox_pred=bbox_pred,
                angle_pred=angle_pred,
                priors=priors,
                img_shape=img_shape,
                img_meta=img_meta,
                level_idx=level_idx)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]
            scores = self._apply_gaussian_semantic_scale(
                scores=scores,
                bbox_pred=bbox_pred,
                angle_pred=angle_pred,
                priors=priors,
                img_shape=img_shape,
                img_meta=img_meta,
                level_idx=level_idx)
            if (not getattr(self, 'gaussian_semantic_scale_enable', False)
                    or getattr(
                        self,
                        'gaussian_semantic_scale_preserve_s3c_guard',
                        True)):
                scores = self._apply_scale_semantic_calibration(
                    scores=scores,
                    bbox_pred=bbox_pred,
                    angle_pred=angle_pred,
                    priors=priors,
                    img_shape=img_shape,
                    img_meta=img_meta,
                    level_idx=level_idx)

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
