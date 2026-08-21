from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.layers.transformer.utils import MLP, inverse_sigmoid
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures import SampleList
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from mmrotate.registry import MODELS as MMROTATE_MODELS
from mmrotate.structures import RotatedBoxes
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


def build_orthogonal_support_tokens(num_classes: int,
                                    embed_dims: int,
                                    scale: float = 1.0,
                                    tail_std: float = 0.0) -> Tensor:
    if num_classes <= 0:
        raise ValueError('num_classes must be positive')
    if embed_dims <= 0:
        raise ValueError('embed_dims must be positive')
    tokens = torch.zeros(num_classes, embed_dims, dtype=torch.float32)
    eye_dim = min(num_classes, embed_dims)
    tokens[:, :eye_dim] = torch.eye(num_classes, eye_dim)
    if embed_dims > eye_dim and tail_std > 0:
        tokens[:, eye_dim:] = torch.empty(
            num_classes, embed_dims - eye_dim).normal_(0.0, tail_std)
    return tokens * float(scale)


def _periodic_l1(pred: Tensor,
                 target: Tensor,
                 period: float = math.pi) -> Tensor:
    delta = (pred - target + period / 2) % period - period / 2
    return delta.abs()


def _rbox_to_gaussian(boxes: Tensor) -> Tuple[Tensor, Tensor]:
    shape = boxes.shape
    xy = boxes[..., :2]
    wh = boxes[..., 2:4].clamp(min=1e-6).reshape(-1, 2)
    angle = boxes[..., 4].reshape(-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rotation = torch.stack(
        (cos_a, -sin_a, sin_a, cos_a), dim=-1).reshape(-1, 2, 2)
    scale = 0.5 * torch.diag_embed(wh)
    sigma = rotation.bmm(scale.square()).bmm(
        rotation.transpose(1, 2)).reshape(shape[:-1] + (2, 2))
    return xy, sigma


def _det2x2(matrix: Tensor) -> Tensor:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - (
        matrix[..., 0, 1] * matrix[..., 1, 0])


def gaussian_wasserstein_loss(pred_boxes: Tensor,
                              target_boxes: Tensor,
                              eps: float = 1e-6) -> Tensor:
    xy_p, sigma_p = _rbox_to_gaussian(pred_boxes)
    xy_t, sigma_t = _rbox_to_gaussian(target_boxes)
    xy_distance = (xy_p - xy_t).square().sum(dim=-1)
    trace_p = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_t = sigma_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    sigma_cross = torch.matmul(sigma_p, sigma_t)
    trace_cross = sigma_cross.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)
    det_cross = (_det2x2(sigma_p) * _det2x2(sigma_t)).clamp(
        min=eps).sqrt()
    sigma_distance = (
        trace_p + trace_t -
        2.0 * (trace_cross + 2.0 * det_cross).clamp(min=eps).sqrt())
    distance = (xy_distance + sigma_distance.clamp(min=0.0)).clamp(
        min=eps).sqrt()
    return torch.log1p(distance)


class P15SupportQueryConditioner(nn.Module):
    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 support_scale: float = 1.0,
                 support_init_std: float = 0.02) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dims = int(embed_dims)
        self.support_scale = float(support_scale)
        support = build_orthogonal_support_tokens(
            self.num_classes,
            self.embed_dims,
            scale=1.0,
            tail_std=float(support_init_std))
        self.support_tokens = nn.Parameter(support)
        self.support_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.debug: Dict[str, object] = {
            'support_source': 'learned_orthogonal',
            'support_conditioned_query_initializer': True,
            'support_conditioning_strength': 0.0,
        }

    def forward(self,
                query: Tensor,
                class_logits: Tensor,
                query_gate: Optional[Tensor] = None) -> Tensor:
        if self.support_scale == 0:
            self.debug['support_conditioning_strength'] = 0.0
            return query
        weights = class_logits.sigmoid().softmax(dim=-1)
        context = weights @ self.support_tokens.to(query.dtype)
        delta = self.support_proj(context)
        if query_gate is not None:
            if query_gate.dim() == 2:
                query_gate = query_gate.unsqueeze(-1)
            delta = delta * query_gate.to(device=query.device, dtype=query.dtype)
        out = query + float(self.support_scale) * delta
        self.debug['support_conditioning_strength'] = float(
            delta.detach().norm(dim=-1).mean().cpu())
        return out


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15OrientedDINOSetHead(BaseModule):
    def __init__(self,
                 num_classes: int,
                 embed_dims: int = 256,
                 num_reg_fcs: int = 2,
                 num_pred_layer: int = 7,
                 num_queries: int = 300,
                 share_pred_layer: bool = False,
                 as_two_stage: bool = True,
                 sync_cls_avg_factor: bool = True,
                 support_query_scale: float = 1.0,
                 support_init_std: float = 0.02,
                 loss_cls_weight: float = 2.0,
                 loss_bbox_weight: float = 5.0,
                 loss_angle_weight: float = 1.0,
                 loss_gwd_weight: float = 1.0,
                 max_per_img: int = 300,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = int(num_classes)
        self.embed_dims = int(embed_dims)
        self.num_pred_layer = int(num_pred_layer)
        self.num_queries = int(num_queries)
        self.share_pred_layer = bool(share_pred_layer)
        self.as_two_stage = bool(as_two_stage)
        self.sync_cls_avg_factor = bool(sync_cls_avg_factor)
        self.loss_cls_weight = float(loss_cls_weight)
        self.loss_bbox_weight = float(loss_bbox_weight)
        self.loss_angle_weight = float(loss_angle_weight)
        self.loss_gwd_weight = float(loss_gwd_weight)
        self.max_per_img = int(max_per_img)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.query_conditioner = P15SupportQueryConditioner(
            num_classes=self.num_classes,
            embed_dims=self.embed_dims,
            support_scale=float(support_query_scale),
            support_init_std=float(support_init_std))
        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()
        self.angle_branches = nn.ModuleList()
        for _ in range(self.num_pred_layer):
            self.cls_branches.append(
                nn.Linear(self.embed_dims, self.num_classes))
            self.reg_branches.append(
                MLP(self.embed_dims, self.embed_dims, 4, int(num_reg_fcs)))
            self.angle_branches.append(
                MLP(self.embed_dims, self.embed_dims, 1, int(num_reg_fcs)))
        self.e2e_debug: Dict[str, object] = {
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_traditional_dense_head': False,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
            'support_conditioned_query_initializer': True,
            'teacher_training_only': True,
        }

    def init_weights(self) -> None:
        super().init_weights()
        bias = -math.log((1 - 0.01) / 0.01)
        for cls_branch in self.cls_branches:
            nn.init.constant_(cls_branch.bias, bias)
        for reg_branch in self.reg_branches:
            constant_init(reg_branch.layers[-1], 0, bias=0)
            nn.init.constant_(reg_branch.layers[-1].bias.data[2:], -2.0)
        for angle_branch in self.angle_branches:
            constant_init(angle_branch.layers[-1], 0, bias=0)

    def condition_matching_queries(
            self,
            query: Tensor,
            topk_class_logits: Tensor,
            topk_objectness: Optional[Tensor] = None) -> Tensor:
        conditioned = self.query_conditioner(query, topk_class_logits)
        self.e2e_debug.update(self.query_conditioner.debug)
        return conditioned

    def forward(self, hidden_states: Tensor, references: List[Tensor],
                reference_angles: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        all_cls_scores = []
        all_bbox_preds = []
        all_angle_preds = []
        for layer_id in range(hidden_states.shape[0]):
            hidden_state = hidden_states[layer_id]
            reference = inverse_sigmoid(references[layer_id], eps=1e-3)
            outputs_class = self.cls_branches[layer_id](hidden_state)
            reg_delta = self.reg_branches[layer_id](hidden_state)
            angle_delta = self.angle_branches[layer_id](hidden_state)
            bbox_pred = (reg_delta + reference).sigmoid()
            angle_pred = angle_delta + reference_angles[layer_id]
            all_cls_scores.append(outputs_class)
            all_bbox_preds.append(bbox_pred)
            all_angle_preds.append(angle_pred)
        return (
            torch.stack(all_cls_scores),
            torch.stack(all_bbox_preds),
            torch.stack(all_angle_preds),
        )

    def _gt_tensor(self, gt_instances: InstanceData) -> Tensor:
        bboxes = gt_instances.bboxes
        if isinstance(bboxes, RotatedBoxes):
            return bboxes.tensor
        return bboxes

    def _scale_boxes_to_pixels(self, boxes: Tensor, img_meta: dict) -> Tensor:
        img_h, img_w = img_meta['img_shape']
        factor = boxes.new_tensor([img_w, img_h, img_w, img_h])
        return boxes * factor

    def _assign_single(self, cls_score: Tensor, bbox_pred: Tensor,
                       angle_pred: Tensor, gt_instances: InstanceData,
                       img_meta: dict) -> Tuple[Tensor, Tensor]:
        num_queries = cls_score.shape[0]
        gt_boxes = self._gt_tensor(gt_instances)
        assigned_gt = cls_score.new_full((num_queries,), -1, dtype=torch.long)
        assigned_labels = cls_score.new_full(
            (num_queries,), -1, dtype=torch.long)
        if gt_boxes.numel() == 0:
            return assigned_gt, assigned_labels
        pred_xywh = self._scale_boxes_to_pixels(bbox_pred, img_meta)
        pred_rbox = torch.cat([pred_xywh, angle_pred], dim=-1)
        cls_prob = cls_score.sigmoid()
        cls_cost = -cls_prob[:, gt_instances.labels]
        img_h, img_w = img_meta['img_shape']
        norm = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
        gt_norm = gt_boxes[:, :4] / norm
        bbox_cost = torch.cdist(bbox_pred, gt_norm, p=1)
        angle_cost = torch.cdist(angle_pred, gt_boxes[:, 4:5], p=1)
        gwd_cost = gaussian_wasserstein_loss(pred_rbox[:, None, :],
                                             gt_boxes[None, :, :])
        total_cost = (
            2.0 * cls_cost + 5.0 * bbox_cost + 1.0 * angle_cost +
            1.0 * gwd_cost)
        row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu())
        row_ind = torch.as_tensor(
            row_ind, dtype=torch.long, device=cls_score.device)
        col_ind = torch.as_tensor(
            col_ind, dtype=torch.long, device=cls_score.device)
        assigned_gt[row_ind] = col_ind
        assigned_labels[row_ind] = gt_instances.labels[col_ind]
        return assigned_gt, assigned_labels

    def _loss_single(self, cls_score: Tensor, bbox_pred: Tensor,
                     angle_pred: Tensor, batch_data_samples: SampleList,
                     prefix: str = '') -> Dict[str, Tensor]:
        cls_losses = []
        bbox_losses = []
        angle_losses = []
        gwd_losses = []
        num_pos = cls_score.new_tensor(0.0)
        for img_id, sample in enumerate(batch_data_samples):
            gt_instances = sample.gt_instances
            img_meta = sample.metainfo
            assigned_gt, assigned_labels = self._assign_single(
                cls_score[img_id], bbox_pred[img_id], angle_pred[img_id],
                gt_instances, img_meta)
            cls_target = cls_score.new_zeros(cls_score[img_id].shape)
            pos_mask = assigned_gt >= 0
            if pos_mask.any():
                cls_target[pos_mask, assigned_labels[pos_mask]] = 1.0
            cls_loss = py_sigmoid_focal_loss(
                cls_score[img_id],
                cls_target,
                weight=None,
                gamma=2.0,
                alpha=0.25,
                reduction='sum')
            cls_losses.append(cls_loss)
            if pos_mask.any():
                pos_gt = assigned_gt[pos_mask]
                gt_boxes = self._gt_tensor(gt_instances)[pos_gt]
                img_h, img_w = img_meta['img_shape']
                norm = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
                gt_norm = gt_boxes[:, :4] / norm
                pos_bbox = bbox_pred[img_id][pos_mask]
                pos_angle = angle_pred[img_id][pos_mask]
                bbox_losses.append(
                    F.l1_loss(pos_bbox, gt_norm, reduction='sum'))
                angle_losses.append(
                    _periodic_l1(pos_angle, gt_boxes[:, 4:5]).sum())
                pred_rbox = torch.cat([
                    self._scale_boxes_to_pixels(pos_bbox, img_meta),
                    pos_angle,
                ],
                                       dim=-1)
                gwd_losses.append(
                    gaussian_wasserstein_loss(pred_rbox, gt_boxes).sum())
                num_pos += pos_mask.sum()
        avg_factor = num_pos.clamp(min=1.0)
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(avg_factor)
        zero = cls_score.sum() * 0.0
        losses = {
            f'{prefix}loss_cls':
            sum(cls_losses) / avg_factor * self.loss_cls_weight,
            f'{prefix}loss_bbox':
            (sum(bbox_losses) if bbox_losses else zero) / avg_factor *
            self.loss_bbox_weight,
            f'{prefix}loss_angle':
            (sum(angle_losses) if angle_losses else zero) / avg_factor *
            self.loss_angle_weight,
            f'{prefix}loss_gwd':
            (sum(gwd_losses) if gwd_losses else zero) / avg_factor *
            self.loss_gwd_weight,
        }
        return losses

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             reference_angles: List[Tensor], enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor, enc_outputs_angle: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Optional[Dict[str, int]] = None) -> Dict[str, Tensor]:
        del dn_meta
        all_cls_scores, all_bbox_preds, all_angle_preds = self(
            hidden_states, references, reference_angles)
        losses = self._loss_single(
            all_cls_scores[-1], all_bbox_preds[-1], all_angle_preds[-1],
            batch_data_samples)
        if enc_outputs_class is not None:
            enc_losses = self._loss_single(
                enc_outputs_class, enc_outputs_coord, enc_outputs_angle,
                batch_data_samples, prefix='enc_')
            losses.update(enc_losses)
        for layer_id in range(all_cls_scores.shape[0] - 1):
            layer_losses = self._loss_single(
                all_cls_scores[layer_id],
                all_bbox_preds[layer_id],
                all_angle_preds[layer_id],
                batch_data_samples,
                prefix=f'd{layer_id}.')
            losses.update(layer_losses)
        return losses

    def _predict_single(self, cls_score: Tensor, bbox_pred: Tensor,
                        angle_pred: Tensor, img_meta: dict,
                        rescale: bool = True) -> InstanceData:
        scores, labels = cls_score.sigmoid().max(dim=-1)
        boxes = self._scale_boxes_to_pixels(bbox_pred, img_meta)
        rboxes = torch.cat([boxes, angle_pred], dim=-1)
        if rescale and 'scale_factor' in img_meta:
            scale_factor = rboxes.new_tensor(img_meta['scale_factor']).repeat(2)
            rboxes[:, :4] = rboxes[:, :4] / scale_factor
        if rboxes.shape[0] != self.num_queries:
            raise RuntimeError(
                f'P15 expected {self.num_queries} fixed queries, got '
                f'{rboxes.shape[0]}')
        results = InstanceData()
        results.bboxes = RotatedBoxes(rboxes)
        results.scores = scores
        results.labels = labels
        return results

    def predict(self, hidden_states: Tensor, references: List[Tensor],
                reference_angles: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> List[InstanceData]:
        all_cls_scores, all_bbox_preds, all_angle_preds = self(
            hidden_states, references, reference_angles)
        cls_score = all_cls_scores[-1]
        bbox_pred = all_bbox_preds[-1]
        angle_pred = all_angle_preds[-1]
        return [
            self._predict_single(cls_score[i], bbox_pred[i], angle_pred[i],
                                 batch_data_samples[i].metainfo, rescale)
            for i in range(cls_score.shape[0])
        ]


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15BOrientedDINOSetHead(P15OrientedDINOSetHead):
    """P15B strict-E2E set head with background and objectness posterior."""

    def __init__(self,
                 *args,
                 bg_cls_weight: float = 0.1,
                 loss_obj_weight: float = 1.0,
                 topk_objectness_power: float = 1.0,
                 matching_objectness_power: float = 1.0,
                 objectness_prior_floor: float = 0.0,
                 support_objectness_gate_power: float = 0.0,
                 support_objectness_gate_warmup_iters: int = 0,
                 support_objectness_gate_floor_start: float = 0.0,
                 support_objectness_gate_floor_end: float = 0.0,
                 support_dn_query_scale: float = 0.0,
                 support_dn_query_logit: float = 6.0,
                 support_dn_query_warmup_iters: int = 0,
                 train_class_balanced_topk_per_class: int = 0,
                 aux_one2many_topk: int = 0,
                 aux_one2many_loss_weight: float = 0.0,
                 aux_one2many_warmup_iters: int = 0,
                 duplicate_rank_topk: int = 0,
                 duplicate_rank_loss_weight: float = 0.0,
                 duplicate_rank_margin: float = 0.15,
                 duplicate_rank_warmup_iters: int = 0,
                 duplicate_rank_max_cost: float = float('inf'),
                 matching_duplicate_neg_topk: int = 0,
                 matching_duplicate_neg_loss_weight: float = 0.0,
                 matching_duplicate_neg_warmup_iters: int = 0,
                 matching_duplicate_neg_decay_start_iters: int = 0,
                 matching_duplicate_neg_decay_end_iters: int = 0,
                 matching_duplicate_neg_decay_final_mult: float = 1.0,
                 matching_duplicate_neg_max_cost: float = float('inf'),
                 cardinality_loss_weight: float = 0.0,
                 cardinality_warmup_iters: int = 0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.background_class_index = self.num_classes
        self.cls_out_channels = self.num_classes + 1
        self.bg_cls_weight = float(bg_cls_weight)
        self.loss_obj_weight = float(loss_obj_weight)
        self.topk_objectness_power = float(topk_objectness_power)
        self.matching_objectness_power = float(matching_objectness_power)
        self.objectness_prior_floor = float(objectness_prior_floor)
        self.support_objectness_gate_power = float(
            support_objectness_gate_power)
        self.support_objectness_gate_warmup_iters = int(
            support_objectness_gate_warmup_iters)
        self.support_objectness_gate_floor_start = float(
            support_objectness_gate_floor_start)
        self.support_objectness_gate_floor_end = float(
            support_objectness_gate_floor_end)
        self.support_dn_query_scale = float(support_dn_query_scale)
        self.support_dn_query_logit = float(support_dn_query_logit)
        self.support_dn_query_warmup_iters = int(
            support_dn_query_warmup_iters)
        self.train_class_balanced_topk_per_class = int(
            train_class_balanced_topk_per_class)
        self.aux_one2many_topk = int(aux_one2many_topk)
        self.aux_one2many_loss_weight = float(aux_one2many_loss_weight)
        self.aux_one2many_warmup_iters = int(aux_one2many_warmup_iters)
        self.duplicate_rank_topk = int(duplicate_rank_topk)
        self.duplicate_rank_loss_weight = float(duplicate_rank_loss_weight)
        self.duplicate_rank_margin = float(duplicate_rank_margin)
        self.duplicate_rank_warmup_iters = int(duplicate_rank_warmup_iters)
        self.duplicate_rank_max_cost = float(duplicate_rank_max_cost)
        self.matching_duplicate_neg_topk = int(matching_duplicate_neg_topk)
        self.matching_duplicate_neg_loss_weight = float(
            matching_duplicate_neg_loss_weight)
        self.matching_duplicate_neg_warmup_iters = int(
            matching_duplicate_neg_warmup_iters)
        self.matching_duplicate_neg_decay_start_iters = int(
            matching_duplicate_neg_decay_start_iters)
        self.matching_duplicate_neg_decay_end_iters = int(
            matching_duplicate_neg_decay_end_iters)
        self.matching_duplicate_neg_decay_final_mult = float(
            matching_duplicate_neg_decay_final_mult)
        self.matching_duplicate_neg_max_cost = float(
            matching_duplicate_neg_max_cost)
        self.cardinality_loss_weight = float(cardinality_loss_weight)
        self.cardinality_warmup_iters = int(cardinality_warmup_iters)
        if self.topk_objectness_power < 0:
            raise ValueError('topk_objectness_power must be non-negative')
        if self.matching_objectness_power < 0:
            raise ValueError('matching_objectness_power must be non-negative')
        if self.support_objectness_gate_power < 0:
            raise ValueError(
                'support_objectness_gate_power must be non-negative')
        if self.support_objectness_gate_warmup_iters < 0:
            raise ValueError(
                'support_objectness_gate_warmup_iters must be non-negative')
        if self.support_dn_query_scale < 0:
            raise ValueError('support_dn_query_scale must be non-negative')
        if self.support_dn_query_warmup_iters < 0:
            raise ValueError(
                'support_dn_query_warmup_iters must be non-negative')
        if self.train_class_balanced_topk_per_class < 0:
            raise ValueError(
                'train_class_balanced_topk_per_class must be non-negative')
        if self.aux_one2many_topk < 0:
            raise ValueError('aux_one2many_topk must be non-negative')
        if self.aux_one2many_loss_weight < 0:
            raise ValueError('aux_one2many_loss_weight must be non-negative')
        if self.aux_one2many_warmup_iters < 0:
            raise ValueError('aux_one2many_warmup_iters must be non-negative')
        if self.duplicate_rank_topk < 0:
            raise ValueError('duplicate_rank_topk must be non-negative')
        if self.duplicate_rank_loss_weight < 0:
            raise ValueError(
                'duplicate_rank_loss_weight must be non-negative')
        if self.duplicate_rank_margin < 0:
            raise ValueError('duplicate_rank_margin must be non-negative')
        if self.duplicate_rank_warmup_iters < 0:
            raise ValueError(
                'duplicate_rank_warmup_iters must be non-negative')
        if math.isnan(self.duplicate_rank_max_cost):
            raise ValueError('duplicate_rank_max_cost must not be NaN')
        if self.matching_duplicate_neg_topk < 0:
            raise ValueError('matching_duplicate_neg_topk must be non-negative')
        if self.matching_duplicate_neg_loss_weight < 0:
            raise ValueError(
                'matching_duplicate_neg_loss_weight must be non-negative')
        if self.matching_duplicate_neg_warmup_iters < 0:
            raise ValueError(
                'matching_duplicate_neg_warmup_iters must be non-negative')
        if self.matching_duplicate_neg_decay_start_iters < 0:
            raise ValueError(
                'matching_duplicate_neg_decay_start_iters must be non-negative')
        if self.matching_duplicate_neg_decay_end_iters < 0:
            raise ValueError(
                'matching_duplicate_neg_decay_end_iters must be non-negative')
        if not 0.0 <= self.matching_duplicate_neg_decay_final_mult <= 1.0:
            raise ValueError(
                'matching_duplicate_neg_decay_final_mult must be in [0, 1]')
        if (self.matching_duplicate_neg_decay_final_mult != 1.0
                and self.matching_duplicate_neg_decay_end_iters
                <= self.matching_duplicate_neg_decay_start_iters):
            raise ValueError(
                'matching_duplicate_neg_decay_end_iters must be greater than '
                'matching_duplicate_neg_decay_start_iters when decay is active')
        if math.isnan(self.matching_duplicate_neg_max_cost):
            raise ValueError('matching_duplicate_neg_max_cost must not be NaN')
        if self.cardinality_loss_weight < 0:
            raise ValueError('cardinality_loss_weight must be non-negative')
        if self.cardinality_warmup_iters < 0:
            raise ValueError('cardinality_warmup_iters must be non-negative')
        if not 0.0 <= self.objectness_prior_floor < 1.0:
            raise ValueError('objectness_prior_floor must be in [0, 1)')
        for name, floor in (
                ('support_objectness_gate_floor_start',
                 self.support_objectness_gate_floor_start),
                ('support_objectness_gate_floor_end',
                 self.support_objectness_gate_floor_end)):
            if not 0.0 <= floor < 1.0:
                raise ValueError(f'{name} must be in [0, 1)')
        self.register_buffer(
            '_support_gate_step',
            torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            '_support_dn_step',
            torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            '_aux_one2many_step',
            torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            '_duplicate_rank_step',
            torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            '_matching_duplicate_neg_step',
            torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            '_cardinality_step',
            torch.zeros((), dtype=torch.long),
            persistent=True)

        self.cls_branches = nn.ModuleList()
        self.obj_branches = nn.ModuleList()
        for _ in range(self.num_pred_layer):
            self.cls_branches.append(
                nn.Linear(self.embed_dims, self.cls_out_channels))
            self.obj_branches.append(nn.Linear(self.embed_dims, 1))
        self.e2e_debug.update({
            'uses_background_softmax': True,
            'uses_query_objectness_posterior': True,
            'uses_dn_loss_split': True,
            'topk_objectness_power': self.topk_objectness_power,
            'matching_objectness_power': self.matching_objectness_power,
            'support_objectness_gate_power':
            self.support_objectness_gate_power,
            'support_objectness_gate_warmup_iters':
            self.support_objectness_gate_warmup_iters,
            'support_objectness_gate_floor_start':
            self.support_objectness_gate_floor_start,
            'support_objectness_gate_floor_end':
            self.support_objectness_gate_floor_end,
            'objectness_prior_floor': self.objectness_prior_floor,
            'support_dn_query_scale': self.support_dn_query_scale,
            'support_dn_query_logit': self.support_dn_query_logit,
            'support_dn_query_warmup_iters':
            self.support_dn_query_warmup_iters,
            'train_class_balanced_topk_per_class':
            self.train_class_balanced_topk_per_class,
            'aux_one2many_topk': self.aux_one2many_topk,
            'aux_one2many_loss_weight': self.aux_one2many_loss_weight,
            'aux_one2many_warmup_iters': self.aux_one2many_warmup_iters,
            'duplicate_rank_topk': self.duplicate_rank_topk,
            'duplicate_rank_loss_weight': self.duplicate_rank_loss_weight,
            'duplicate_rank_margin': self.duplicate_rank_margin,
            'duplicate_rank_warmup_iters':
            self.duplicate_rank_warmup_iters,
            'duplicate_rank_max_cost': self.duplicate_rank_max_cost,
            'matching_duplicate_neg_topk':
            self.matching_duplicate_neg_topk,
            'matching_duplicate_neg_loss_weight':
            self.matching_duplicate_neg_loss_weight,
            'matching_duplicate_neg_warmup_iters':
            self.matching_duplicate_neg_warmup_iters,
            'matching_duplicate_neg_decay_start_iters':
            self.matching_duplicate_neg_decay_start_iters,
            'matching_duplicate_neg_decay_end_iters':
            self.matching_duplicate_neg_decay_end_iters,
            'matching_duplicate_neg_decay_final_mult':
            self.matching_duplicate_neg_decay_final_mult,
            'matching_duplicate_neg_max_cost':
            self.matching_duplicate_neg_max_cost,
            'cardinality_loss_weight': self.cardinality_loss_weight,
            'cardinality_warmup_iters': self.cardinality_warmup_iters,
            'uses_tempered_objectness_prior':
            self.topk_objectness_power != 1.0
            or self.matching_objectness_power != 1.0
            or self.objectness_prior_floor != 0.0,
            'uses_objectness_gated_support_conditioning':
            self.support_objectness_gate_power != 0.0,
            'uses_support_gate_warmup_floor':
            self.support_objectness_gate_power != 0.0
            and self.support_objectness_gate_warmup_iters > 0,
            'uses_support_conditioned_dn_queries':
            self.support_dn_query_scale != 0.0,
            'uses_train_class_balanced_query_selection':
            self.train_class_balanced_topk_per_class > 0,
            'uses_aux_one2many_primary_supervision':
            self.aux_one2many_topk > 0
            and self.aux_one2many_loss_weight > 0.0,
            'uses_duplicate_rank_suppression':
            self.duplicate_rank_topk > 0
            and self.duplicate_rank_loss_weight > 0.0,
            'uses_matching_duplicate_objectness_suppression':
            self.matching_duplicate_neg_topk > 0
            and self.matching_duplicate_neg_loss_weight > 0.0,
            'uses_objectness_cardinality_budget':
            self.cardinality_loss_weight > 0.0,
            'final_score_formula': 'objectness_sigmoid_x_foreground_softmax',
        })

    def init_weights(self) -> None:
        super().init_weights()
        for cls_branch in self.cls_branches:
            nn.init.constant_(cls_branch.bias[:self.num_classes], -2.0)
            nn.init.constant_(cls_branch.bias[self.background_class_index],
                              2.0)
        for obj_branch in self.obj_branches:
            constant_init(obj_branch, 0, bias=-2.0)

    def condition_matching_queries(
            self,
            query: Tensor,
            topk_class_logits: Tensor,
            topk_objectness: Optional[Tensor] = None) -> Tensor:
        foreground_logits = topk_class_logits[..., :self.num_classes]
        query_gate = None
        if (topk_objectness is not None
                and self.support_objectness_gate_power != 0.0):
            query_gate = self._support_query_gate(topk_objectness)
        conditioned = self.query_conditioner(
            query, foreground_logits, query_gate=query_gate)
        self.e2e_debug.update(self.query_conditioner.debug)
        if self.training and query_gate is not None:
            self._support_gate_step.add_(1)
        return conditioned

    def _support_dn_scale_schedule(self) -> Tuple[float, float]:
        if self.support_dn_query_warmup_iters <= 0:
            return self.support_dn_query_scale, 1.0
        progress = min(
            float(self._support_dn_step.item()) /
            float(self.support_dn_query_warmup_iters),
            1.0)
        return self.support_dn_query_scale * progress, progress

    def condition_denoising_queries(
            self,
            dn_query: Tensor,
            batch_data_samples: SampleList,
            dn_meta: Optional[Dict[str, int]]) -> Tensor:
        if self.support_dn_query_scale == 0.0 or dn_meta is None:
            self.e2e_debug.update({
                'uses_support_conditioned_dn_queries': False,
                'support_dn_positive_slots': 0,
            })
            return dn_query
        num_groups = int(dn_meta['num_denoising_groups'])
        num_denoising_queries = int(dn_meta['num_denoising_queries'])
        if num_groups <= 0 or num_denoising_queries % num_groups != 0:
            raise ValueError('invalid dn_meta for support-conditioned DN')
        num_queries_each_group = num_denoising_queries // num_groups
        max_num_target = num_queries_each_group // 2
        if max_num_target <= 0:
            return dn_query

        class_logits = dn_query.new_zeros(
            dn_query.shape[0], dn_query.shape[1], self.num_classes)
        query_gate = dn_query.new_zeros(dn_query.shape[:2] + (1, ))
        positive_slots = 0
        for img_id, sample in enumerate(batch_data_samples):
            gt_labels = sample.gt_instances.labels
            if gt_labels.numel() == 0:
                continue
            num_targets = min(int(gt_labels.numel()), max_num_target)
            target_labels = gt_labels[:num_targets].to(dtype=torch.long)
            for group_id in range(num_groups):
                start = group_id * num_queries_each_group
                pos_inds = start + torch.arange(
                    num_targets, device=dn_query.device)
                class_logits[img_id, pos_inds, target_labels] = (
                    self.support_dn_query_logit)
                query_gate[img_id, pos_inds, 0] = 1.0
                positive_slots += num_targets

        scale, progress = self._support_dn_scale_schedule()
        if positive_slots == 0:
            conditioned = dn_query
        else:
            support_conditioned = self.query_conditioner(
                dn_query, class_logits, query_gate=query_gate)
            conditioned = dn_query + float(scale) * (
                support_conditioned - dn_query)
            self.e2e_debug.update(self.query_conditioner.debug)
        self.e2e_debug.update({
            'uses_support_conditioned_dn_queries': True,
            'support_dn_query_current_scale': float(scale),
            'support_dn_query_progress': float(progress),
            'support_dn_query_step': int(self._support_dn_step.item()),
            'support_dn_positive_slots': int(positive_slots),
        })
        if self.training:
            self._support_dn_step.add_(1)
        return conditioned

    def select_topk_scores(self,
                           class_logits: Tensor,
                           objectness_logits: Optional[Tensor] = None) -> Tensor:
        foreground_prob = class_logits.softmax(dim=-1)[..., :self.num_classes]
        topk_score = foreground_prob.max(dim=-1)[0]
        if objectness_logits is not None:
            topk_score = topk_score * self._objectness_prior(
                objectness_logits, self.topk_objectness_power)
        return topk_score

    def _classwise_query_scores(
            self,
            class_logits: Tensor,
            objectness_logits: Optional[Tensor] = None,
            quality_logits: Optional[Tensor] = None) -> Tensor:
        scores = class_logits.softmax(dim=-1)[..., :self.num_classes]
        if objectness_logits is not None:
            scores = scores * self._objectness_prior(
                objectness_logits, self.topk_objectness_power).unsqueeze(-1)
        if quality_logits is not None:
            scores = scores * quality_logits.sigmoid()
        return scores

    def select_query_indices(
            self,
            class_logits: Tensor,
            objectness_logits: Optional[Tensor] = None,
            quality_logits: Optional[Tensor] = None,
            num_queries: Optional[int] = None) -> Tensor:
        if num_queries is None:
            num_queries = self.num_queries
        if quality_logits is not None:
            global_scores = self.select_topk_scores(
                class_logits, objectness_logits, quality_logits)
        else:
            global_scores = self.select_topk_scores(
                class_logits, objectness_logits)
        global_indices = torch.topk(global_scores, k=num_queries, dim=1)[1]
        per_class = self.train_class_balanced_topk_per_class
        if (not self.training) or per_class <= 0:
            self.e2e_debug.update({
                'uses_train_class_balanced_query_selection': False,
                'train_class_balanced_reserved_queries': 0,
            })
            return global_indices

        class_scores = self._classwise_query_scores(
            class_logits, objectness_logits, quality_logits)
        per_class = min(int(per_class), class_scores.shape[1])
        balanced_indices = global_indices.new_empty(global_indices.shape)
        reserved = 0
        for img_id in range(class_scores.shape[0]):
            candidates = []
            for class_id in range(self.num_classes):
                class_topk = torch.topk(
                    class_scores[img_id, :, class_id],
                    k=per_class,
                    dim=0)[1]
                candidates.extend(class_topk.detach().cpu().tolist())
            candidates.extend(global_indices[img_id].detach().cpu().tolist())
            selected = []
            seen = set()
            for idx in candidates:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                if len(selected) == num_queries:
                    break
            balanced_indices[img_id] = torch.as_tensor(
                selected, dtype=torch.long, device=global_indices.device)
            reserved += min(
                self.num_classes * per_class,
                len(selected),
            )
        self.e2e_debug.update({
            'uses_train_class_balanced_query_selection': True,
            'train_class_balanced_reserved_queries': int(reserved),
        })
        return balanced_indices

    def _objectness_prior(self, objectness_logits: Tensor,
                          power: float) -> Tensor:
        prior = objectness_logits.sigmoid().squeeze(-1)
        if self.objectness_prior_floor > 0.0:
            prior = prior.clamp(min=self.objectness_prior_floor)
        if float(power) == 1.0:
            return prior
        return prior.pow(float(power))

    def _support_gate_schedule(self) -> Tuple[float, float, float]:
        if self.support_objectness_gate_warmup_iters <= 0:
            return (
                self.support_objectness_gate_power,
                self.support_objectness_gate_floor_end,
                1.0,
            )
        progress = min(
            float(self._support_gate_step.item()) /
            float(self.support_objectness_gate_warmup_iters),
            1.0)
        power = self.support_objectness_gate_power * progress
        floor = (
            self.support_objectness_gate_floor_start +
            (self.support_objectness_gate_floor_end -
             self.support_objectness_gate_floor_start) * progress)
        return power, floor, progress

    def _support_query_gate(self, objectness_logits: Tensor) -> Tensor:
        power, floor, progress = self._support_gate_schedule()
        gate = objectness_logits.sigmoid().squeeze(-1)
        if float(power) != 1.0:
            gate = gate.pow(float(power))
        if floor > 0.0:
            gate = float(floor) + (1.0 - float(floor)) * gate
        self.e2e_debug.update({
            'support_objectness_gate_current_power': float(power),
            'support_objectness_gate_current_floor': float(floor),
            'support_objectness_gate_progress': float(progress),
            'support_objectness_gate_step':
            int(self._support_gate_step.item()),
        })
        return gate

    def forward(self, hidden_states: Tensor, references: List[Tensor],
                reference_angles: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor,
                                                         Tensor]:
        all_cls_scores = []
        all_obj_logits = []
        all_bbox_preds = []
        all_angle_preds = []
        for layer_id in range(hidden_states.shape[0]):
            hidden_state = hidden_states[layer_id]
            reference = inverse_sigmoid(references[layer_id], eps=1e-3)
            outputs_class = self.cls_branches[layer_id](hidden_state)
            outputs_objectness = self.obj_branches[layer_id](hidden_state)
            reg_delta = self.reg_branches[layer_id](hidden_state)
            angle_delta = self.angle_branches[layer_id](hidden_state)
            bbox_pred = (reg_delta + reference).sigmoid()
            angle_pred = angle_delta + reference_angles[layer_id]
            all_cls_scores.append(outputs_class)
            all_obj_logits.append(outputs_objectness)
            all_bbox_preds.append(bbox_pred)
            all_angle_preds.append(angle_pred)
        return (
            torch.stack(all_cls_scores),
            torch.stack(all_obj_logits),
            torch.stack(all_bbox_preds),
            torch.stack(all_angle_preds),
        )

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_obj_logits: Tensor,
                      all_layers_bbox_preds: Tensor,
                      all_layers_angle_preds: Tensor,
                      dn_meta: Optional[Dict[str, int]]) -> Tuple[
                          Tensor, Tensor, Tensor, Tensor, Optional[Tensor],
                          Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if dn_meta is None:
            return (all_layers_cls_scores, all_layers_obj_logits,
                    all_layers_bbox_preds, all_layers_angle_preds, None, None,
                    None, None)
        num_denoising_queries = dn_meta['num_denoising_queries']
        return (
            all_layers_cls_scores[:, :, num_denoising_queries:, :],
            all_layers_obj_logits[:, :, num_denoising_queries:, :],
            all_layers_bbox_preds[:, :, num_denoising_queries:, :],
            all_layers_angle_preds[:, :, num_denoising_queries:, :],
            all_layers_cls_scores[:, :, :num_denoising_queries, :],
            all_layers_obj_logits[:, :, :num_denoising_queries, :],
            all_layers_bbox_preds[:, :, :num_denoising_queries, :],
            all_layers_angle_preds[:, :, :num_denoising_queries, :],
        )

    def _matching_cost_matrix(self,
                              cls_score: Tensor,
                              bbox_pred: Tensor,
                              angle_pred: Tensor,
                              gt_instances: InstanceData,
                              img_meta: dict,
                              obj_logit: Optional[Tensor] = None) -> Tensor:
        gt_boxes = self._gt_tensor(gt_instances)
        if gt_boxes.numel() == 0:
            return cls_score.new_zeros((cls_score.shape[0], 0))
        pred_xywh = self._scale_boxes_to_pixels(bbox_pred, img_meta)
        pred_rbox = torch.cat([pred_xywh, angle_pred], dim=-1)
        cls_prob = cls_score.softmax(dim=-1)[..., :self.num_classes]
        if obj_logit is not None:
            cls_prob = cls_prob * self._objectness_prior(
                obj_logit, self.matching_objectness_power)[:, None]
        cls_cost = -cls_prob[:, gt_instances.labels]
        img_h, img_w = img_meta['img_shape']
        norm = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
        gt_norm = gt_boxes[:, :4] / norm
        bbox_cost = torch.cdist(bbox_pred, gt_norm, p=1)
        angle_cost = torch.cdist(angle_pred, gt_boxes[:, 4:5], p=1)
        gwd_cost = gaussian_wasserstein_loss(pred_rbox[:, None, :],
                                             gt_boxes[None, :, :])
        return (
            2.0 * cls_cost + 5.0 * bbox_cost + 1.0 * angle_cost +
            1.0 * gwd_cost)

    def _assign_from_cost(self, total_cost: Tensor,
                          gt_labels: Tensor) -> Tuple[Tensor, Tensor]:
        num_queries = total_cost.shape[0]
        assigned_gt = total_cost.new_full(
            (num_queries,), -1, dtype=torch.long)
        assigned_labels = total_cost.new_full(
            (num_queries,), -1, dtype=torch.long)
        if total_cost.shape[1] == 0:
            return assigned_gt, assigned_labels
        row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu())
        row_ind = torch.as_tensor(
            row_ind, dtype=torch.long, device=total_cost.device)
        col_ind = torch.as_tensor(
            col_ind, dtype=torch.long, device=total_cost.device)
        assigned_gt[row_ind] = col_ind
        assigned_labels[row_ind] = gt_labels[col_ind]
        return assigned_gt, assigned_labels

    def _aux_one2many_loss_scale(self) -> Tensor:
        scale = self._aux_one2many_step.new_tensor(
            self.aux_one2many_loss_weight, dtype=torch.float32)
        if (self.aux_one2many_topk <= 0
                or self.aux_one2many_loss_weight <= 0.0):
            return scale * 0.0
        if self.aux_one2many_warmup_iters <= 0:
            progress = scale.new_tensor(1.0)
        else:
            progress = (
                self._aux_one2many_step.float() /
                float(self.aux_one2many_warmup_iters)).clamp(max=1.0)
        self.e2e_debug.update({
            'aux_one2many_current_loss_scale':
            float((scale * progress).detach().cpu()),
            'aux_one2many_step': int(self._aux_one2many_step.item()),
        })
        return scale * progress

    def _duplicate_rank_loss_scale(self) -> Tensor:
        scale = self._duplicate_rank_step.new_tensor(
            self.duplicate_rank_loss_weight, dtype=torch.float32)
        if (self.duplicate_rank_topk <= 0
                or self.duplicate_rank_loss_weight <= 0.0):
            return scale * 0.0
        if self.duplicate_rank_warmup_iters <= 0:
            progress = scale.new_tensor(1.0)
        else:
            progress = (
                self._duplicate_rank_step.float() /
                float(self.duplicate_rank_warmup_iters)).clamp(max=1.0)
        self.e2e_debug.update({
            'duplicate_rank_current_loss_scale':
            float((scale * progress).detach().cpu()),
            'duplicate_rank_step': int(self._duplicate_rank_step.item()),
        })
        return scale * progress

    def _matching_duplicate_neg_loss_scale(self) -> Tensor:
        scale = self._matching_duplicate_neg_step.new_tensor(
            self.matching_duplicate_neg_loss_weight, dtype=torch.float32)
        if (self.matching_duplicate_neg_topk <= 0
                or self.matching_duplicate_neg_loss_weight <= 0.0):
            return scale * 0.0
        if self.matching_duplicate_neg_warmup_iters <= 0:
            progress = scale.new_tensor(1.0)
        else:
            progress = (
                self._matching_duplicate_neg_step.float() /
                float(self.matching_duplicate_neg_warmup_iters)).clamp(max=1.0)
        decay_mult = scale.new_tensor(1.0)
        if self.matching_duplicate_neg_decay_final_mult != 1.0:
            decay_start = float(self.matching_duplicate_neg_decay_start_iters)
            decay_end = float(self.matching_duplicate_neg_decay_end_iters)
            decay_progress = (
                (self._matching_duplicate_neg_step.float() - decay_start) /
                (decay_end - decay_start)).clamp(min=0.0, max=1.0)
            final_mult = scale.new_tensor(
                self.matching_duplicate_neg_decay_final_mult)
            decay_mult = 1.0 + (final_mult - 1.0) * decay_progress
        self.e2e_debug.update({
            'matching_duplicate_neg_current_loss_scale':
            float((scale * progress * decay_mult).detach().cpu()),
            'matching_duplicate_neg_current_decay_mult':
            float(decay_mult.detach().cpu()),
            'matching_duplicate_neg_step':
            int(self._matching_duplicate_neg_step.item()),
        })
        return scale * progress * decay_mult

    def _cardinality_loss_scale(self) -> Tensor:
        scale = self._cardinality_step.new_tensor(
            self.cardinality_loss_weight, dtype=torch.float32)
        if self.cardinality_loss_weight <= 0.0:
            return scale * 0.0
        if self.cardinality_warmup_iters <= 0:
            progress = scale.new_tensor(1.0)
        else:
            progress = (
                self._cardinality_step.float() /
                float(self.cardinality_warmup_iters)).clamp(max=1.0)
        self.e2e_debug.update({
            'cardinality_current_loss_scale':
            float((scale * progress).detach().cpu()),
            'cardinality_step': int(self._cardinality_step.item()),
        })
        return scale * progress

    def _select_aux_one2many_pairs(self, total_cost: Tensor,
                                   assigned_gt: Tensor) -> Tuple[Tensor, Tensor]:
        if (self.aux_one2many_topk <= 0 or total_cost.numel() == 0
                or total_cost.shape[1] == 0):
            empty = assigned_gt.new_empty(0)
            return empty, empty
        available = torch.where(assigned_gt < 0)[0]
        if available.numel() == 0:
            empty = assigned_gt.new_empty(0)
            return empty, empty
        selected_queries = []
        selected_gts = []
        for gt_id in range(total_cost.shape[1]):
            if available.numel() == 0:
                break
            k = min(self.aux_one2many_topk, int(available.numel()))
            local_rank = torch.topk(
                total_cost[available, gt_id], k=k, largest=False)[1]
            chosen = available[local_rank]
            selected_queries.append(chosen)
            selected_gts.append(
                assigned_gt.new_full((chosen.numel(), ), gt_id))
            keep = torch.ones_like(available, dtype=torch.bool)
            keep[local_rank] = False
            available = available[keep]
        if not selected_queries:
            empty = assigned_gt.new_empty(0)
            return empty, empty
        return torch.cat(selected_queries), torch.cat(selected_gts)

    def _select_duplicate_rank_pairs(
            self, total_cost: Tensor,
            assigned_gt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if (self.duplicate_rank_topk <= 0 or total_cost.numel() == 0
                or total_cost.shape[1] == 0):
            empty = assigned_gt.new_empty(0)
            return empty, empty, empty
        available = torch.where(assigned_gt < 0)[0]
        if available.numel() == 0:
            empty = assigned_gt.new_empty(0)
            return empty, empty, empty
        pos_queries = []
        dup_queries = []
        dup_gts = []
        for gt_id in range(total_cost.shape[1]):
            matched = torch.where(assigned_gt == gt_id)[0]
            if matched.numel() == 0 or available.numel() == 0:
                continue
            candidate_cost = total_cost[available, gt_id]
            if math.isfinite(self.duplicate_rank_max_cost):
                keep_cost = candidate_cost <= self.duplicate_rank_max_cost
                if not keep_cost.any():
                    continue
                candidate_indices = available[keep_cost]
                candidate_cost = candidate_cost[keep_cost]
            else:
                candidate_indices = available
            k = min(self.duplicate_rank_topk, int(candidate_indices.numel()))
            local_rank = torch.topk(
                candidate_cost, k=k, largest=False)[1]
            chosen = candidate_indices[local_rank]
            pos_queries.append(matched[:1].expand_as(chosen))
            dup_queries.append(chosen)
            dup_gts.append(assigned_gt.new_full((chosen.numel(), ), gt_id))
            keep_available = torch.ones_like(available, dtype=torch.bool)
            for chosen_id in chosen:
                keep_available &= available != chosen_id
            available = available[keep_available]
        if not dup_queries:
            empty = assigned_gt.new_empty(0)
            return empty, empty, empty
        return (
            torch.cat(pos_queries),
            torch.cat(dup_queries),
            torch.cat(dup_gts),
        )

    def _select_matching_duplicate_negatives(
            self, total_cost: Tensor, assigned_gt: Tensor) -> Tensor:
        if (self.matching_duplicate_neg_topk <= 0 or total_cost.numel() == 0
                or total_cost.shape[1] == 0):
            return assigned_gt.new_empty(0)
        available = torch.where(assigned_gt < 0)[0]
        if available.numel() == 0:
            return assigned_gt.new_empty(0)
        selected_queries = []
        for gt_id in range(total_cost.shape[1]):
            if available.numel() == 0:
                break
            candidate_cost = total_cost[available, gt_id]
            if math.isfinite(self.matching_duplicate_neg_max_cost):
                keep_cost = candidate_cost <= self.matching_duplicate_neg_max_cost
                if not keep_cost.any():
                    continue
                candidate_indices = available[keep_cost]
                candidate_cost = candidate_cost[keep_cost]
            else:
                candidate_indices = available
            k = min(self.matching_duplicate_neg_topk,
                    int(candidate_indices.numel()))
            local_rank = torch.topk(
                candidate_cost, k=k, largest=False)[1]
            chosen = candidate_indices[local_rank]
            selected_queries.append(chosen)
            keep_available = torch.ones_like(available, dtype=torch.bool)
            for chosen_id in chosen:
                keep_available &= available != chosen_id
            available = available[keep_available]
        if not selected_queries:
            return assigned_gt.new_empty(0)
        return torch.cat(selected_queries)

    def _assign_single(self,
                       cls_score: Tensor,
                       bbox_pred: Tensor,
                       angle_pred: Tensor,
                       gt_instances: InstanceData,
                       img_meta: dict,
                       obj_logit: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        total_cost = self._matching_cost_matrix(
            cls_score, bbox_pred, angle_pred, gt_instances, img_meta,
            obj_logit)
        return self._assign_from_cost(total_cost, gt_instances.labels)

    def _loss_single(self, cls_score: Tensor, obj_logit: Tensor,
                     bbox_pred: Tensor, angle_pred: Tensor,
                     batch_data_samples: SampleList,
                     prefix: str = '') -> Dict[str, Tensor]:
        cls_losses = []
        obj_losses = []
        bbox_losses = []
        angle_losses = []
        gwd_losses = []
        aux_cls_losses = []
        aux_obj_losses = []
        aux_bbox_losses = []
        aux_angle_losses = []
        aux_gwd_losses = []
        duplicate_rank_losses = []
        matching_duplicate_neg_losses = []
        cardinality_losses = []
        num_pos = cls_score.new_tensor(0.0)
        aux_num_pos = cls_score.new_tensor(0.0)
        aux_selected = 0
        duplicate_rank_pairs = 0
        matching_duplicate_neg_selected = 0
        aux_enabled = (
            prefix == ''
            and self.aux_one2many_topk > 0
            and self.aux_one2many_loss_weight > 0.0)
        aux_scale = (
            self._aux_one2many_loss_scale()
            if aux_enabled else cls_score.new_tensor(0.0))
        duplicate_rank_enabled = (
            prefix == ''
            and self.duplicate_rank_topk > 0
            and self.duplicate_rank_loss_weight > 0.0)
        duplicate_rank_scale = (
            self._duplicate_rank_loss_scale()
            if duplicate_rank_enabled else cls_score.new_tensor(0.0))
        matching_duplicate_neg_enabled = (
            prefix == ''
            and self.matching_duplicate_neg_topk > 0
            and self.matching_duplicate_neg_loss_weight > 0.0)
        matching_duplicate_neg_scale = (
            self._matching_duplicate_neg_loss_scale()
            if matching_duplicate_neg_enabled
            else cls_score.new_tensor(0.0))
        cardinality_enabled = (
            prefix == '' and self.cardinality_loss_weight > 0.0)
        cardinality_scale = (
            self._cardinality_loss_scale()
            if cardinality_enabled else cls_score.new_tensor(0.0))
        for img_id, sample in enumerate(batch_data_samples):
            gt_instances = sample.gt_instances
            img_meta = sample.metainfo
            total_cost = self._matching_cost_matrix(
                cls_score[img_id], bbox_pred[img_id], angle_pred[img_id],
                gt_instances, img_meta, obj_logit[img_id])
            assigned_gt, assigned_labels = self._assign_from_cost(
                total_cost, gt_instances.labels)
            pos_mask = assigned_gt >= 0
            cls_target = cls_score.new_full(
                (cls_score.shape[1], ),
                self.background_class_index,
                dtype=torch.long)
            cls_weight = cls_score.new_ones(cls_target.shape)
            cls_weight[~pos_mask] = self.bg_cls_weight
            if pos_mask.any():
                cls_target[pos_mask] = assigned_labels[pos_mask]
            cls_loss = F.cross_entropy(
                cls_score[img_id],
                cls_target,
                reduction='none')
            cls_losses.append((cls_loss * cls_weight).sum())

            obj_target = obj_logit.new_zeros(obj_logit[img_id].shape[:-1])
            obj_target[pos_mask] = 1.0
            obj_weight = obj_logit.new_ones(obj_target.shape)
            obj_weight[~pos_mask] = self.bg_cls_weight
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_logit[img_id].squeeze(-1),
                obj_target,
                reduction='none')
            obj_losses.append((obj_loss * obj_weight).sum())
            if cardinality_enabled:
                pred_count = obj_logit[img_id].sigmoid().sum()
                target_count = obj_logit.new_tensor(
                    float(len(gt_instances.labels)))
                count_normalizer = target_count.clamp(min=1.0)
                cardinality_losses.append(
                    F.smooth_l1_loss(
                        pred_count / count_normalizer,
                        target_count / count_normalizer,
                        reduction='sum'))

            if pos_mask.any():
                pos_gt = assigned_gt[pos_mask]
                gt_boxes = self._gt_tensor(gt_instances)[pos_gt]
                img_h, img_w = img_meta['img_shape']
                norm = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
                gt_norm = gt_boxes[:, :4] / norm
                pos_bbox = bbox_pred[img_id][pos_mask]
                pos_angle = angle_pred[img_id][pos_mask]
                bbox_losses.append(
                    F.l1_loss(pos_bbox, gt_norm, reduction='sum'))
                angle_losses.append(
                    _periodic_l1(pos_angle, gt_boxes[:, 4:5]).sum())
                pred_rbox = torch.cat([
                    self._scale_boxes_to_pixels(pos_bbox, img_meta),
                    pos_angle,
                ],
                                       dim=-1)
                gwd_losses.append(
                    gaussian_wasserstein_loss(pred_rbox, gt_boxes).sum())
                num_pos += pos_mask.sum()
            if aux_enabled and total_cost.shape[1] > 0:
                aux_query, aux_gt = self._select_aux_one2many_pairs(
                    total_cost.detach(), assigned_gt)
                if aux_query.numel() > 0:
                    gt_boxes = self._gt_tensor(gt_instances)[aux_gt]
                    img_h, img_w = img_meta['img_shape']
                    norm = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
                    gt_norm = gt_boxes[:, :4] / norm
                    aux_labels = gt_instances.labels[aux_gt]
                    aux_bbox = bbox_pred[img_id][aux_query]
                    aux_angle = angle_pred[img_id][aux_query]
                    aux_cls_losses.append(
                        F.cross_entropy(
                            cls_score[img_id][aux_query],
                            aux_labels,
                            reduction='sum'))
                    aux_obj_losses.append(
                        F.binary_cross_entropy_with_logits(
                            obj_logit[img_id][aux_query].squeeze(-1),
                            obj_logit.new_ones(aux_query.shape),
                            reduction='sum'))
                    aux_bbox_losses.append(
                        F.l1_loss(aux_bbox, gt_norm, reduction='sum'))
                    aux_angle_losses.append(
                        _periodic_l1(aux_angle, gt_boxes[:, 4:5]).sum())
                    pred_rbox = torch.cat([
                        self._scale_boxes_to_pixels(aux_bbox, img_meta),
                        aux_angle,
                    ],
                                           dim=-1)
                    aux_gwd_losses.append(
                        gaussian_wasserstein_loss(pred_rbox, gt_boxes).sum())
                    aux_num_pos += aux_query.numel()
                    aux_selected += int(aux_query.numel())
            if duplicate_rank_enabled and total_cost.shape[1] > 0:
                pos_query, dup_query, dup_gt = (
                    self._select_duplicate_rank_pairs(
                        total_cost.detach(), assigned_gt))
                if dup_query.numel() > 0:
                    gt_labels = gt_instances.labels[dup_gt].to(
                        dtype=torch.long)
                    foreground_probs = cls_score[img_id].softmax(
                        dim=-1)[..., :self.num_classes]
                    objectness = obj_logit[img_id].sigmoid().squeeze(-1)
                    pos_scores = (
                        foreground_probs[pos_query, gt_labels] *
                        objectness[pos_query])
                    dup_scores = (
                        foreground_probs[dup_query, gt_labels] *
                        objectness[dup_query])
                    duplicate_rank_losses.append(
                        F.relu(
                            self.duplicate_rank_margin + dup_scores -
                            pos_scores).sum())
                    duplicate_rank_pairs += int(dup_query.numel())
            if matching_duplicate_neg_enabled and total_cost.shape[1] > 0:
                duplicate_neg_query = (
                    self._select_matching_duplicate_negatives(
                        total_cost.detach(), assigned_gt))
                if duplicate_neg_query.numel() > 0:
                    matching_duplicate_neg_losses.append(
                        F.binary_cross_entropy_with_logits(
                            obj_logit[img_id][duplicate_neg_query].squeeze(-1),
                            obj_logit.new_zeros(duplicate_neg_query.shape),
                            reduction='sum'))
                    matching_duplicate_neg_selected += int(
                        duplicate_neg_query.numel())
        avg_factor = num_pos.clamp(min=1.0)
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(avg_factor)
        zero = cls_score.sum() * 0.0
        losses = {
            f'{prefix}loss_cls':
            sum(cls_losses) / avg_factor * self.loss_cls_weight,
            f'{prefix}loss_obj':
            sum(obj_losses) / avg_factor * self.loss_obj_weight,
            f'{prefix}loss_bbox':
            (sum(bbox_losses) if bbox_losses else zero) / avg_factor *
            self.loss_bbox_weight,
            f'{prefix}loss_angle':
            (sum(angle_losses) if angle_losses else zero) / avg_factor *
            self.loss_angle_weight,
            f'{prefix}loss_gwd':
            (sum(gwd_losses) if gwd_losses else zero) / avg_factor *
            self.loss_gwd_weight,
        }
        if aux_enabled:
            aux_avg_factor = aux_num_pos.clamp(min=1.0)
            if self.sync_cls_avg_factor:
                aux_avg_factor = reduce_mean(aux_avg_factor)
            losses.update({
                'loss_aux_one2many_cls':
                (sum(aux_cls_losses) if aux_cls_losses else zero) /
                aux_avg_factor * self.loss_cls_weight * aux_scale,
                'loss_aux_one2many_obj':
                (sum(aux_obj_losses) if aux_obj_losses else zero) /
                aux_avg_factor * self.loss_obj_weight * aux_scale,
                'loss_aux_one2many_bbox':
                (sum(aux_bbox_losses) if aux_bbox_losses else zero) /
                aux_avg_factor * self.loss_bbox_weight * aux_scale,
                'loss_aux_one2many_angle':
                (sum(aux_angle_losses) if aux_angle_losses else zero) /
                aux_avg_factor * self.loss_angle_weight * aux_scale,
                'loss_aux_one2many_gwd':
                (sum(aux_gwd_losses) if aux_gwd_losses else zero) /
                aux_avg_factor * self.loss_gwd_weight * aux_scale,
            })
            self.e2e_debug.update({
                'aux_one2many_selected_pairs': aux_selected,
                'aux_one2many_current_loss_scale':
                float(aux_scale.detach().cpu()),
            })
            if self.training:
                self._aux_one2many_step += 1
        if duplicate_rank_enabled:
            duplicate_rank_avg_factor = cls_score.new_tensor(
                float(max(duplicate_rank_pairs, 1)))
            if self.sync_cls_avg_factor:
                duplicate_rank_avg_factor = reduce_mean(
                    duplicate_rank_avg_factor)
            losses['loss_duplicate_rank'] = (
                (sum(duplicate_rank_losses)
                 if duplicate_rank_losses else zero) /
                duplicate_rank_avg_factor * duplicate_rank_scale)
            self.e2e_debug.update({
                'duplicate_rank_selected_pairs': duplicate_rank_pairs,
                'duplicate_rank_current_loss_scale':
                float(duplicate_rank_scale.detach().cpu()),
            })
            if self.training:
                self._duplicate_rank_step += 1
        if matching_duplicate_neg_enabled:
            neg_avg_factor = cls_score.new_tensor(
                float(max(matching_duplicate_neg_selected, 1)))
            if self.sync_cls_avg_factor:
                neg_avg_factor = reduce_mean(neg_avg_factor)
            losses['loss_matching_dup_obj_neg'] = (
                (sum(matching_duplicate_neg_losses)
                 if matching_duplicate_neg_losses else zero) /
                neg_avg_factor * matching_duplicate_neg_scale)
            self.e2e_debug.update({
                'matching_duplicate_neg_selected_queries':
                matching_duplicate_neg_selected,
                'matching_duplicate_neg_current_loss_scale':
                float(matching_duplicate_neg_scale.detach().cpu()),
            })
            if self.training:
                self._matching_duplicate_neg_step += 1
        if cardinality_enabled:
            cardinality_avg_factor = cls_score.new_tensor(
                float(max(len(cardinality_losses), 1)))
            if self.sync_cls_avg_factor:
                cardinality_avg_factor = reduce_mean(cardinality_avg_factor)
            losses['loss_cardinality_obj'] = (
                (sum(cardinality_losses) if cardinality_losses else zero) /
                cardinality_avg_factor * cardinality_scale)
            self.e2e_debug.update({
                'cardinality_current_loss_scale':
                float(cardinality_scale.detach().cpu()),
            })
            if self.training:
                self._cardinality_step += 1
        return losses

    def _get_dn_targets_single(self, gt_instances: InstanceData, img_meta: dict,
                               dn_meta: Dict[str, int]) -> Tuple[Tensor, ...]:
        gt_boxes = self._gt_tensor(gt_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        max_num_target = num_queries_each_group // 2
        labels = gt_labels.new_full((num_denoising_queries, ),
                                    self.background_class_index)
        cls_weights = gt_boxes.new_full((num_denoising_queries, ),
                                        self.bg_cls_weight)
        obj_targets = gt_boxes.new_zeros(num_denoising_queries)
        bbox_targets = gt_boxes.new_zeros(num_denoising_queries, 4)
        bbox_weights = gt_boxes.new_zeros(num_denoising_queries, 4)
        angle_targets = gt_boxes.new_zeros(num_denoising_queries, 1)
        angle_weights = gt_boxes.new_zeros(num_denoising_queries, 1)
        rbox_targets = gt_boxes.new_zeros(num_denoising_queries, 5)
        pos_inds = gt_labels.new_empty(0)
        pos_assigned_gt_inds = gt_labels.new_empty(0)
        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            group_offsets = (
                torch.arange(num_groups, dtype=torch.long, device=device) *
                num_queries_each_group)
            pos_inds = (group_offsets[:, None] + t).flatten()
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            cls_weights[pos_inds] = 1.0
            obj_targets[pos_inds] = 1.0
            img_h, img_w = img_meta['img_shape']
            norm = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
            bbox_targets[pos_inds] = (
                gt_boxes[pos_assigned_gt_inds, :4] / norm)
            bbox_weights[pos_inds] = 1.0
            angle_targets[pos_inds] = gt_boxes[pos_assigned_gt_inds, 4:5]
            angle_weights[pos_inds] = 1.0
            rbox_targets[pos_inds] = gt_boxes[pos_assigned_gt_inds]
        return (labels, cls_weights, obj_targets, bbox_targets, bbox_weights,
                angle_targets, angle_weights, rbox_targets, pos_inds)

    def _loss_dn_single(self, dn_cls_score: Tensor, dn_obj_logit: Tensor,
                        dn_bbox_pred: Tensor, dn_angle_pred: Tensor,
                        batch_data_samples: SampleList) -> Dict[str, Tensor]:
        dn_meta = self._active_dn_meta
        cls_losses = []
        obj_losses = []
        bbox_losses = []
        angle_losses = []
        gwd_losses = []
        num_pos = dn_cls_score.new_tensor(0.0)
        for img_id, sample in enumerate(batch_data_samples):
            targets = self._get_dn_targets_single(sample.gt_instances,
                                                  sample.metainfo, dn_meta)
            (labels, cls_weights, obj_targets, bbox_targets, bbox_weights,
             angle_targets, angle_weights, rbox_targets, pos_inds) = targets
            cls_loss = F.cross_entropy(
                dn_cls_score[img_id], labels, reduction='none')
            cls_losses.append((cls_loss * cls_weights).sum())
            obj_loss = F.binary_cross_entropy_with_logits(
                dn_obj_logit[img_id].squeeze(-1),
                obj_targets,
                reduction='none')
            obj_weights = torch.where(obj_targets > 0,
                                      torch.ones_like(obj_targets),
                                      torch.full_like(obj_targets,
                                                      self.bg_cls_weight))
            obj_losses.append((obj_loss * obj_weights).sum())
            if pos_inds.numel() > 0:
                bbox_losses.append(
                    (F.l1_loss(
                        dn_bbox_pred[img_id],
                        bbox_targets,
                        reduction='none') * bbox_weights).sum())
                angle_losses.append(
                    (_periodic_l1(dn_angle_pred[img_id], angle_targets) *
                     angle_weights).sum())
                pred_rbox = torch.cat([
                    self._scale_boxes_to_pixels(
                        dn_bbox_pred[img_id][pos_inds],
                        sample.metainfo),
                    dn_angle_pred[img_id][pos_inds],
                ],
                                       dim=-1)
                gwd_losses.append(
                    gaussian_wasserstein_loss(
                        pred_rbox, rbox_targets[pos_inds]).sum())
                num_pos += pos_inds.numel()
        avg_factor = num_pos.clamp(min=1.0)
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(avg_factor)
        zero = dn_cls_score.sum() * 0.0
        return {
            'loss_cls': sum(cls_losses) / avg_factor * self.loss_cls_weight,
            'loss_obj': sum(obj_losses) / avg_factor * self.loss_obj_weight,
            'loss_bbox': (sum(bbox_losses) if bbox_losses else zero) /
            avg_factor * self.loss_bbox_weight,
            'loss_angle': (sum(angle_losses) if angle_losses else zero) /
            avg_factor * self.loss_angle_weight,
            'loss_gwd': (sum(gwd_losses) if gwd_losses else zero) /
            avg_factor * self.loss_gwd_weight,
        }

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_obj_logits: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_angle_preds: Tensor,
                batch_data_samples: SampleList,
                dn_meta: Dict[str, int]) -> List[Dict[str, Tensor]]:
        self._active_dn_meta = dn_meta
        return [
            self._loss_dn_single(cls_score, obj_logit, bbox_pred, angle_pred,
                                 batch_data_samples)
            for cls_score, obj_logit, bbox_pred, angle_pred in zip(
                all_layers_denoising_cls_scores,
                all_layers_denoising_obj_logits,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_angle_preds)
        ]

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             reference_angles: List[Tensor], enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor, enc_outputs_angle: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Optional[Dict[str, int]] = None,
             enc_outputs_objectness: Optional[Tensor] = None) -> Dict[str, Tensor]:
        (all_cls_scores, all_obj_logits, all_bbox_preds,
         all_angle_preds) = self(hidden_states, references, reference_angles)
        (matching_cls_scores, matching_obj_logits, matching_bbox_preds,
         matching_angle_preds, dn_cls_scores, dn_obj_logits, dn_bbox_preds,
         dn_angle_preds) = self.split_outputs(all_cls_scores, all_obj_logits,
                                              all_bbox_preds, all_angle_preds,
                                              dn_meta)
        losses = self._loss_single(
            matching_cls_scores[-1], matching_obj_logits[-1],
            matching_bbox_preds[-1], matching_angle_preds[-1],
            batch_data_samples)
        if enc_outputs_class is not None:
            if enc_outputs_objectness is None:
                enc_outputs_objectness = enc_outputs_class.new_zeros(
                    enc_outputs_class.shape[:-1] + (1, ))
            enc_losses = self._loss_single(
                enc_outputs_class, enc_outputs_objectness, enc_outputs_coord,
                enc_outputs_angle, batch_data_samples, prefix='enc_')
            losses.update(enc_losses)
        for layer_id in range(matching_cls_scores.shape[0] - 1):
            layer_losses = self._loss_single(
                matching_cls_scores[layer_id], matching_obj_logits[layer_id],
                matching_bbox_preds[layer_id], matching_angle_preds[layer_id],
                batch_data_samples, prefix=f'd{layer_id}.')
            losses.update(layer_losses)
        if dn_cls_scores is not None:
            dn_losses = self.loss_dn(dn_cls_scores, dn_obj_logits,
                                     dn_bbox_preds, dn_angle_preds,
                                     batch_data_samples, dn_meta)
            final_dn = dn_losses[-1]
            for name, value in final_dn.items():
                losses[f'dn_{name}'] = value
            for layer_id, layer_loss in enumerate(dn_losses[:-1]):
                for name, value in layer_loss.items():
                    losses[f'd{layer_id}.dn_{name}'] = value
        return losses

    def _predict_single(self, cls_score: Tensor, obj_logit: Tensor,
                        bbox_pred: Tensor, angle_pred: Tensor, img_meta: dict,
                        rescale: bool = True) -> InstanceData:
        foreground_probs = cls_score.softmax(dim=-1)[..., :self.num_classes]
        scores, labels = foreground_probs.max(dim=-1)
        scores = scores * obj_logit.sigmoid().squeeze(-1)
        boxes = self._scale_boxes_to_pixels(bbox_pred, img_meta)
        rboxes = torch.cat([boxes, angle_pred], dim=-1)
        if rescale and 'scale_factor' in img_meta:
            scale_factor = rboxes.new_tensor(img_meta['scale_factor']).repeat(2)
            rboxes[:, :4] = rboxes[:, :4] / scale_factor
        if rboxes.shape[0] != self.num_queries:
            raise RuntimeError(
                f'P15B expected {self.num_queries} fixed queries, got '
                f'{rboxes.shape[0]}')
        results = InstanceData()
        results.bboxes = RotatedBoxes(rboxes)
        results.scores = scores
        results.labels = labels
        return results

    def predict(self, hidden_states: Tensor, references: List[Tensor],
                reference_angles: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> List[InstanceData]:
        (all_cls_scores, all_obj_logits, all_bbox_preds,
         all_angle_preds) = self(hidden_states, references, reference_angles)
        cls_score = all_cls_scores[-1]
        obj_logit = all_obj_logits[-1]
        bbox_pred = all_bbox_preds[-1]
        angle_pred = all_angle_preds[-1]
        return [
            self._predict_single(cls_score[i], obj_logit[i], bbox_pred[i],
                                 angle_pred[i],
                                 batch_data_samples[i].metainfo, rescale)
            for i in range(cls_score.shape[0])
        ]


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15CGaussianQualityDINOSetHead(P15BOrientedDINOSetHead):
    """P15C adds a Gaussian box-quality posterior to P15B set prediction."""

    def __init__(self,
                 *args,
                 loss_quality_weight: float = 1.0,
                 quality_gwd_sigma: float = 4.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_quality_weight = float(loss_quality_weight)
        self.quality_gwd_sigma = float(quality_gwd_sigma)
        self.quality_branches = nn.ModuleList()
        for _ in range(self.num_pred_layer):
            self.quality_branches.append(nn.Linear(self.embed_dims, 1))
        self.e2e_debug.update({
            'uses_gaussian_quality_posterior': True,
            'uses_quality_matching_warmup': True,
            'final_score_formula':
            'objectness_sigmoid_x_gaussian_quality_x_foreground_softmax',
        })

    def init_weights(self) -> None:
        super().init_weights()
        for quality_branch in self.quality_branches:
            constant_init(quality_branch, 0, bias=-2.0)

    def select_topk_scores(self,
                           class_logits: Tensor,
                           objectness_logits: Optional[Tensor] = None,
                           quality_logits: Optional[Tensor] = None) -> Tensor:
        topk_score = super().select_topk_scores(class_logits,
                                                objectness_logits)
        if quality_logits is not None:
            topk_score = topk_score * quality_logits.sigmoid().squeeze(-1)
        return topk_score

    def forward(self, hidden_states: Tensor, references: List[Tensor],
                reference_angles: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor,
                                                         Tensor, Tensor]:
        all_cls_scores = []
        all_obj_logits = []
        all_quality_logits = []
        all_bbox_preds = []
        all_angle_preds = []
        for layer_id in range(hidden_states.shape[0]):
            hidden_state = hidden_states[layer_id]
            reference = inverse_sigmoid(references[layer_id], eps=1e-3)
            outputs_class = self.cls_branches[layer_id](hidden_state)
            outputs_objectness = self.obj_branches[layer_id](hidden_state)
            outputs_quality = self.quality_branches[layer_id](hidden_state)
            reg_delta = self.reg_branches[layer_id](hidden_state)
            angle_delta = self.angle_branches[layer_id](hidden_state)
            bbox_pred = (reg_delta + reference).sigmoid()
            angle_pred = angle_delta + reference_angles[layer_id]
            all_cls_scores.append(outputs_class)
            all_obj_logits.append(outputs_objectness)
            all_quality_logits.append(outputs_quality)
            all_bbox_preds.append(bbox_pred)
            all_angle_preds.append(angle_pred)
        return (
            torch.stack(all_cls_scores),
            torch.stack(all_obj_logits),
            torch.stack(all_quality_logits),
            torch.stack(all_bbox_preds),
            torch.stack(all_angle_preds),
        )

    @staticmethod
    def split_outputs(
        all_layers_cls_scores: Tensor,
        all_layers_obj_logits: Tensor,
        all_layers_quality_logits: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_angle_preds: Tensor,
        dn_meta: Optional[Dict[str, int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor],
               Optional[Tensor], Optional[Tensor], Optional[Tensor],
               Optional[Tensor]]:
        if dn_meta is None:
            return (all_layers_cls_scores, all_layers_obj_logits,
                    all_layers_quality_logits, all_layers_bbox_preds,
                    all_layers_angle_preds, None, None, None, None, None)
        num_denoising_queries = dn_meta['num_denoising_queries']
        return (
            all_layers_cls_scores[:, :, num_denoising_queries:, :],
            all_layers_obj_logits[:, :, num_denoising_queries:, :],
            all_layers_quality_logits[:, :, num_denoising_queries:, :],
            all_layers_bbox_preds[:, :, num_denoising_queries:, :],
            all_layers_angle_preds[:, :, num_denoising_queries:, :],
            all_layers_cls_scores[:, :, :num_denoising_queries, :],
            all_layers_obj_logits[:, :, :num_denoising_queries, :],
            all_layers_quality_logits[:, :, :num_denoising_queries, :],
            all_layers_bbox_preds[:, :, :num_denoising_queries, :],
            all_layers_angle_preds[:, :, :num_denoising_queries, :],
        )

    def _gaussian_quality_target(self, pred_rbox: Tensor,
                                 gt_rbox: Tensor) -> Tensor:
        sigma = max(self.quality_gwd_sigma, 1e-4)
        quality = torch.exp(
            -gaussian_wasserstein_loss(pred_rbox, gt_rbox) / sigma)
        return quality.clamp(min=0.0, max=1.0)

    def _loss_single(self, cls_score: Tensor, obj_logit: Tensor,
                     quality_logit: Tensor, bbox_pred: Tensor,
                     angle_pred: Tensor, batch_data_samples: SampleList,
                     prefix: str = '') -> Dict[str, Tensor]:
        cls_losses = []
        obj_losses = []
        quality_losses = []
        bbox_losses = []
        angle_losses = []
        gwd_losses = []
        num_pos = cls_score.new_tensor(0.0)
        for img_id, sample in enumerate(batch_data_samples):
            gt_instances = sample.gt_instances
            img_meta = sample.metainfo
            assigned_gt, assigned_labels = super()._assign_single(
                cls_score[img_id], bbox_pred[img_id], angle_pred[img_id],
                gt_instances, img_meta, obj_logit[img_id])
            pos_mask = assigned_gt >= 0
            cls_target = cls_score.new_full(
                (cls_score.shape[1], ),
                self.background_class_index,
                dtype=torch.long)
            cls_weight = cls_score.new_ones(cls_target.shape)
            cls_weight[~pos_mask] = self.bg_cls_weight
            if pos_mask.any():
                cls_target[pos_mask] = assigned_labels[pos_mask]
            cls_loss = F.cross_entropy(
                cls_score[img_id],
                cls_target,
                reduction='none')
            cls_losses.append((cls_loss * cls_weight).sum())

            obj_target = obj_logit.new_zeros(obj_logit[img_id].shape[:-1])
            obj_target[pos_mask] = 1.0
            obj_weight = obj_logit.new_ones(obj_target.shape)
            obj_weight[~pos_mask] = self.bg_cls_weight
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_logit[img_id].squeeze(-1),
                obj_target,
                reduction='none')
            obj_losses.append((obj_loss * obj_weight).sum())

            quality_target = quality_logit.new_zeros(
                quality_logit[img_id].shape[:-1])
            quality_weight = quality_logit.new_ones(quality_target.shape)
            quality_weight[~pos_mask] = self.bg_cls_weight

            if pos_mask.any():
                pos_gt = assigned_gt[pos_mask]
                gt_boxes = self._gt_tensor(gt_instances)[pos_gt]
                img_h, img_w = img_meta['img_shape']
                norm = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
                gt_norm = gt_boxes[:, :4] / norm
                pos_bbox = bbox_pred[img_id][pos_mask]
                pos_angle = angle_pred[img_id][pos_mask]
                bbox_losses.append(
                    F.l1_loss(pos_bbox, gt_norm, reduction='sum'))
                angle_losses.append(
                    _periodic_l1(pos_angle, gt_boxes[:, 4:5]).sum())
                pred_rbox = torch.cat([
                    self._scale_boxes_to_pixels(pos_bbox, img_meta),
                    pos_angle,
                ],
                                       dim=-1)
                gwd = gaussian_wasserstein_loss(pred_rbox, gt_boxes)
                gwd_losses.append(gwd.sum())
                quality_target[pos_mask] = torch.exp(
                    -gwd / max(self.quality_gwd_sigma, 1e-4)).clamp(
                        min=0.0, max=1.0)
                num_pos += pos_mask.sum()

            quality_loss = F.binary_cross_entropy_with_logits(
                quality_logit[img_id].squeeze(-1),
                quality_target,
                reduction='none')
            quality_losses.append((quality_loss * quality_weight).sum())

        avg_factor = num_pos.clamp(min=1.0)
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(avg_factor)
        zero = cls_score.sum() * 0.0
        return {
            f'{prefix}loss_cls':
            sum(cls_losses) / avg_factor * self.loss_cls_weight,
            f'{prefix}loss_obj':
            sum(obj_losses) / avg_factor * self.loss_obj_weight,
            f'{prefix}loss_quality':
            sum(quality_losses) / avg_factor * self.loss_quality_weight,
            f'{prefix}loss_bbox':
            (sum(bbox_losses) if bbox_losses else zero) / avg_factor *
            self.loss_bbox_weight,
            f'{prefix}loss_angle':
            (sum(angle_losses) if angle_losses else zero) / avg_factor *
            self.loss_angle_weight,
            f'{prefix}loss_gwd':
            (sum(gwd_losses) if gwd_losses else zero) / avg_factor *
            self.loss_gwd_weight,
        }

    def _loss_dn_single(self, dn_cls_score: Tensor, dn_obj_logit: Tensor,
                        dn_quality_logit: Tensor, dn_bbox_pred: Tensor,
                        dn_angle_pred: Tensor,
                        batch_data_samples: SampleList) -> Dict[str, Tensor]:
        dn_meta = self._active_dn_meta
        cls_losses = []
        obj_losses = []
        quality_losses = []
        bbox_losses = []
        angle_losses = []
        gwd_losses = []
        num_pos = dn_cls_score.new_tensor(0.0)
        for img_id, sample in enumerate(batch_data_samples):
            targets = self._get_dn_targets_single(sample.gt_instances,
                                                  sample.metainfo, dn_meta)
            (labels, cls_weights, obj_targets, bbox_targets, bbox_weights,
             angle_targets, angle_weights, rbox_targets, pos_inds) = targets
            cls_loss = F.cross_entropy(
                dn_cls_score[img_id], labels, reduction='none')
            cls_losses.append((cls_loss * cls_weights).sum())
            obj_loss = F.binary_cross_entropy_with_logits(
                dn_obj_logit[img_id].squeeze(-1),
                obj_targets,
                reduction='none')
            obj_weights = torch.where(obj_targets > 0,
                                      torch.ones_like(obj_targets),
                                      torch.full_like(obj_targets,
                                                      self.bg_cls_weight))
            obj_losses.append((obj_loss * obj_weights).sum())

            quality_targets = dn_quality_logit.new_zeros(
                dn_quality_logit[img_id].shape[:-1])
            quality_weights = torch.where(
                obj_targets > 0, torch.ones_like(obj_targets),
                torch.full_like(obj_targets, self.bg_cls_weight))
            if pos_inds.numel() > 0:
                bbox_losses.append(
                    (F.l1_loss(
                        dn_bbox_pred[img_id],
                        bbox_targets,
                        reduction='none') * bbox_weights).sum())
                angle_losses.append(
                    (_periodic_l1(dn_angle_pred[img_id], angle_targets) *
                     angle_weights).sum())
                pred_rbox = torch.cat([
                    self._scale_boxes_to_pixels(
                        dn_bbox_pred[img_id][pos_inds],
                        sample.metainfo),
                    dn_angle_pred[img_id][pos_inds],
                ],
                                       dim=-1)
                gwd = gaussian_wasserstein_loss(pred_rbox,
                                                rbox_targets[pos_inds])
                gwd_losses.append(gwd.sum())
                quality_targets[pos_inds] = torch.exp(
                    -gwd / max(self.quality_gwd_sigma, 1e-4)).clamp(
                        min=0.0, max=1.0)
                num_pos += pos_inds.numel()
            quality_loss = F.binary_cross_entropy_with_logits(
                dn_quality_logit[img_id].squeeze(-1),
                quality_targets,
                reduction='none')
            quality_losses.append((quality_loss * quality_weights).sum())

        avg_factor = num_pos.clamp(min=1.0)
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(avg_factor)
        zero = dn_cls_score.sum() * 0.0
        return {
            'loss_cls': sum(cls_losses) / avg_factor * self.loss_cls_weight,
            'loss_obj': sum(obj_losses) / avg_factor * self.loss_obj_weight,
            'loss_quality':
            sum(quality_losses) / avg_factor * self.loss_quality_weight,
            'loss_bbox': (sum(bbox_losses) if bbox_losses else zero) /
            avg_factor * self.loss_bbox_weight,
            'loss_angle': (sum(angle_losses) if angle_losses else zero) /
            avg_factor * self.loss_angle_weight,
            'loss_gwd': (sum(gwd_losses) if gwd_losses else zero) /
            avg_factor * self.loss_gwd_weight,
        }

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_obj_logits: Tensor,
                all_layers_denoising_quality_logits: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_angle_preds: Tensor,
                batch_data_samples: SampleList,
                dn_meta: Dict[str, int]) -> List[Dict[str, Tensor]]:
        self._active_dn_meta = dn_meta
        return [
            self._loss_dn_single(cls_score, obj_logit, quality_logit,
                                 bbox_pred, angle_pred, batch_data_samples)
            for cls_score, obj_logit, quality_logit, bbox_pred, angle_pred in
            zip(all_layers_denoising_cls_scores,
                all_layers_denoising_obj_logits,
                all_layers_denoising_quality_logits,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_angle_preds)
        ]

    def loss(self,
             hidden_states: Tensor,
             references: List[Tensor],
             reference_angles: List[Tensor],
             enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor,
             enc_outputs_angle: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Optional[Dict[str, int]] = None,
             enc_outputs_objectness: Optional[Tensor] = None,
             enc_outputs_quality: Optional[Tensor] = None) -> Dict[str, Tensor]:
        (all_cls_scores, all_obj_logits, all_quality_logits, all_bbox_preds,
         all_angle_preds) = self(hidden_states, references, reference_angles)
        (matching_cls_scores, matching_obj_logits, matching_quality_logits,
         matching_bbox_preds, matching_angle_preds, dn_cls_scores,
         dn_obj_logits, dn_quality_logits, dn_bbox_preds,
         dn_angle_preds) = self.split_outputs(all_cls_scores, all_obj_logits,
                                              all_quality_logits,
                                              all_bbox_preds, all_angle_preds,
                                              dn_meta)
        losses = self._loss_single(
            matching_cls_scores[-1], matching_obj_logits[-1],
            matching_quality_logits[-1], matching_bbox_preds[-1],
            matching_angle_preds[-1], batch_data_samples)
        if enc_outputs_class is not None:
            if enc_outputs_objectness is None:
                enc_outputs_objectness = enc_outputs_class.new_zeros(
                    enc_outputs_class.shape[:-1] + (1, ))
            if enc_outputs_quality is None:
                enc_outputs_quality = enc_outputs_class.new_zeros(
                    enc_outputs_class.shape[:-1] + (1, ))
            enc_losses = self._loss_single(
                enc_outputs_class, enc_outputs_objectness,
                enc_outputs_quality, enc_outputs_coord, enc_outputs_angle,
                batch_data_samples, prefix='enc_')
            losses.update(enc_losses)
        for layer_id in range(matching_cls_scores.shape[0] - 1):
            layer_losses = self._loss_single(
                matching_cls_scores[layer_id], matching_obj_logits[layer_id],
                matching_quality_logits[layer_id],
                matching_bbox_preds[layer_id], matching_angle_preds[layer_id],
                batch_data_samples, prefix=f'd{layer_id}.')
            losses.update(layer_losses)
        if dn_cls_scores is not None:
            dn_losses = self.loss_dn(dn_cls_scores, dn_obj_logits,
                                     dn_quality_logits, dn_bbox_preds,
                                     dn_angle_preds, batch_data_samples,
                                     dn_meta)
            final_dn = dn_losses[-1]
            for name, value in final_dn.items():
                losses[f'dn_{name}'] = value
            for layer_id, layer_loss in enumerate(dn_losses[:-1]):
                for name, value in layer_loss.items():
                    losses[f'd{layer_id}.dn_{name}'] = value
        return losses

    def _predict_single(self, cls_score: Tensor, obj_logit: Tensor,
                        quality_logit: Tensor, bbox_pred: Tensor,
                        angle_pred: Tensor, img_meta: dict,
                        rescale: bool = True) -> InstanceData:
        foreground_probs = cls_score.softmax(dim=-1)[..., :self.num_classes]
        scores, labels = foreground_probs.max(dim=-1)
        scores = (
            scores * obj_logit.sigmoid().squeeze(-1) *
            quality_logit.sigmoid().squeeze(-1))
        boxes = self._scale_boxes_to_pixels(bbox_pred, img_meta)
        rboxes = torch.cat([boxes, angle_pred], dim=-1)
        if rescale and 'scale_factor' in img_meta:
            scale_factor = rboxes.new_tensor(img_meta['scale_factor']).repeat(2)
            rboxes[:, :4] = rboxes[:, :4] / scale_factor
        if rboxes.shape[0] != self.num_queries:
            raise RuntimeError(
                f'P15C expected {self.num_queries} fixed queries, got '
                f'{rboxes.shape[0]}')
        results = InstanceData()
        results.bboxes = RotatedBoxes(rboxes)
        results.scores = scores
        results.labels = labels
        return results

    def predict(self, hidden_states: Tensor, references: List[Tensor],
                reference_angles: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> List[InstanceData]:
        (all_cls_scores, all_obj_logits, all_quality_logits, all_bbox_preds,
         all_angle_preds) = self(hidden_states, references, reference_angles)
        cls_score = all_cls_scores[-1]
        obj_logit = all_obj_logits[-1]
        quality_logit = all_quality_logits[-1]
        bbox_pred = all_bbox_preds[-1]
        angle_pred = all_angle_preds[-1]
        return [
            self._predict_single(cls_score[i], obj_logit[i],
                                 quality_logit[i], bbox_pred[i],
                                 angle_pred[i],
                                 batch_data_samples[i].metainfo, rescale)
            for i in range(cls_score.shape[0])
        ]


@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15DLogitFusedGaussianQualityDINOSetHead(
        P15CGaussianQualityDINOSetHead):
    """P15D fuses Gaussian quality into objectness logits before sigmoid."""

    def __init__(self,
                 *args,
                 quality_logit_alpha: float = 0.5,
                 learnable_quality_logit_alpha: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.learnable_quality_logit_alpha = bool(
            learnable_quality_logit_alpha)
        if self.learnable_quality_logit_alpha:
            self.quality_logit_alpha = nn.Parameter(
                torch.tensor(float(quality_logit_alpha)))
        else:
            self.register_buffer(
                'quality_logit_alpha',
                torch.tensor(float(quality_logit_alpha)),
                persistent=False)
        self.e2e_debug.update({
            'uses_logit_fused_gaussian_quality_posterior': True,
            'learnable_quality_logit_alpha':
            self.learnable_quality_logit_alpha,
            'final_score_formula':
            'foreground_softmax_x_sigmoid_objectness_plus_alpha_quality',
        })

    def _fused_objectness_logit(self, obj_logit: Tensor,
                                quality_logit: Tensor) -> Tensor:
        alpha = self.quality_logit_alpha.to(
            device=obj_logit.device, dtype=obj_logit.dtype)
        return obj_logit.squeeze(-1) + alpha * quality_logit.squeeze(-1)

    def select_topk_scores(self,
                           class_logits: Tensor,
                           objectness_logits: Optional[Tensor] = None,
                           quality_logits: Optional[Tensor] = None) -> Tensor:
        foreground_prob = class_logits.softmax(dim=-1)[..., :self.num_classes]
        topk_score = foreground_prob.max(dim=-1)[0]
        if objectness_logits is None:
            return topk_score
        if quality_logits is None:
            fused_logit = objectness_logits.squeeze(-1)
        else:
            fused_logit = self._fused_objectness_logit(
                objectness_logits, quality_logits)
        return topk_score * fused_logit.sigmoid()

    def _predict_single(self, cls_score: Tensor, obj_logit: Tensor,
                        quality_logit: Tensor, bbox_pred: Tensor,
                        angle_pred: Tensor, img_meta: dict,
                        rescale: bool = True) -> InstanceData:
        foreground_probs = cls_score.softmax(dim=-1)[..., :self.num_classes]
        scores, labels = foreground_probs.max(dim=-1)
        scores = scores * self._fused_objectness_logit(
            obj_logit, quality_logit).sigmoid()
        boxes = self._scale_boxes_to_pixels(bbox_pred, img_meta)
        rboxes = torch.cat([boxes, angle_pred], dim=-1)
        if rescale and 'scale_factor' in img_meta:
            scale_factor = rboxes.new_tensor(img_meta['scale_factor']).repeat(2)
            rboxes[:, :4] = rboxes[:, :4] / scale_factor
        if rboxes.shape[0] != self.num_queries:
            raise RuntimeError(
                f'P15D expected {self.num_queries} fixed queries, got '
                f'{rboxes.shape[0]}')
        results = InstanceData()
        results.bboxes = RotatedBoxes(rboxes)
        results.scores = scores
        results.labels = labels
        return results
