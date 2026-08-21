"""P13E end-to-end dense-seed Gaussian set decoder.

This head is intentionally not a dense RTMDet head.  Dense feature maps are
used only to initialize a fixed set of object queries.  Final predictions are a
one-to-one set of rotated boxes with no anchor assignment and no NMS.
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Sequence

import torch
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures.bbox import get_box_tensor
from mmengine.model import BaseModule, normal_init
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
import torch.nn.functional as F

from mmrotate.registry import MODELS


def build_support_tokens(num_classes: int,
                         embed_channels: int,
                         scale: float = 1.0,
                         tail_std: float = 0.0) -> Tensor:
    if num_classes <= 0:
        raise ValueError('num_classes must be positive')
    if embed_channels <= 0:
        raise ValueError('embed_channels must be positive')
    tokens = torch.zeros(num_classes, embed_channels, dtype=torch.float32)
    eye_dim = min(num_classes, embed_channels)
    tokens[:, :eye_dim] = torch.eye(num_classes, eye_dim)
    if embed_channels > eye_dim and tail_std > 0:
        tokens[:, eye_dim:] = torch.empty(
            num_classes, embed_channels - eye_dim).normal_(0.0, tail_std)
    return tokens * scale


def _rbox_to_gaussian(boxes: Tensor) -> tuple[Tensor, Tensor]:
    shape = boxes.shape
    xy = boxes[..., :2]
    wh = boxes[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    angle = boxes[..., 4].reshape(-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rotation = torch.stack(
        (cos_a, -sin_a, sin_a, cos_a), dim=-1).reshape(-1, 2, 2)
    scale = 0.5 * torch.diag_embed(wh)
    sigma = rotation.bmm(scale.square()).bmm(
        rotation.permute(0, 2, 1)).reshape(shape[:-1] + (2, 2))
    return xy, sigma


def _gaussian_wasserstein_loss(pred_boxes: Tensor,
                               target_boxes: Tensor,
                               alpha: float = 1.0,
                               normalize: bool = True) -> Tensor:
    xy_p, sigma_p = _rbox_to_gaussian(pred_boxes)
    xy_t, sigma_t = _rbox_to_gaussian(target_boxes)
    xy_distance = (xy_p - xy_t).square().sum(dim=-1)
    whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)
    trace_term = (sigma_p.bmm(sigma_t)).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)
    det_term = (sigma_p.det() * sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance - 2.0 * (
        trace_term + 2.0 * det_term).clamp(1e-7).sqrt()
    distance = (xy_distance + alpha * alpha * whr_distance).clamp(
        min=1e-7).sqrt()
    if normalize:
        scale = 2.0 * det_term.clamp(1e-7).sqrt().clamp(
            min=1e-7).sqrt()
        distance = distance / scale.clamp(min=1e-7)
    return torch.log1p(distance)


@MODELS.register_module()
class P13EDenseSeedGaussianSetHead(BaseModule):
    """Fixed-query rotated detection head with one-to-one matching."""

    def __init__(self,
                 num_classes: int,
                 in_channels: int = 256,
                 feat_channels: int = 128,
                 num_queries: int = 200,
                 num_decoder_layers: int = 2,
                 num_heads: int = 8,
                 strides: Sequence[int] = (8, 16, 32),
                 image_size: Sequence[int] = (800, 800),
                 support_scale: float = 1.5,
                 support_init_std: float = 0.02,
                 logit_scale_init: float = 6.0,
                 cls_loss_weight: float = 2.0,
                 bbox_loss_weight: float = 5.0,
                 angle_loss_weight: float = 1.0,
                 quality_loss_weight: float = 1.0,
                 seed_loss_weight: float = 0.25,
                 match_cls_cost: float = 2.0,
                 match_bbox_cost: float = 5.0,
                 match_angle_cost: float = 1.0,
                 query_class_prior_weight: float = 0.0,
                 own_class_logit_bias: float = 0.0,
                 dn_loss_weight: float = 0.0,
                 dn_noise_scale: float = 0.02,
                 dn_max_gt: int = 100,
                 box_delta_scale: float = 2.0,
                 score_thr: float = 0.05,
                 max_per_img: int | None = None,
                 train_cfg: dict | None = None,
                 test_cfg: dict | None = None,
                 init_cfg: dict | None = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.num_queries = int(num_queries)
        self.strides = tuple(int(s) for s in strides)
        self.image_size = tuple(int(v) for v in image_size)
        self.cls_loss_weight = float(cls_loss_weight)
        self.bbox_loss_weight = float(bbox_loss_weight)
        self.angle_loss_weight = float(angle_loss_weight)
        self.quality_loss_weight = float(quality_loss_weight)
        self.seed_loss_weight = float(seed_loss_weight)
        self.match_cls_cost = float(match_cls_cost)
        self.match_bbox_cost = float(match_bbox_cost)
        self.match_angle_cost = float(match_angle_cost)
        self.query_class_prior_weight = float(query_class_prior_weight)
        self.own_class_logit_bias = float(own_class_logit_bias)
        self.dn_loss_weight = float(dn_loss_weight)
        self.dn_noise_scale = float(dn_noise_scale)
        self.dn_max_gt = int(dn_max_gt)
        self.box_delta_scale = float(box_delta_scale)
        self.score_thr = float(score_thr)
        self.max_per_img = int(max_per_img or num_queries)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        self.input_proj = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.feat_channels, 1)
            for _ in self.strides
        ])
        self.seed_logits = nn.ModuleList([
            nn.Conv2d(self.feat_channels, 1, 1) for _ in self.strides
        ])
        self.level_embed = nn.Parameter(
            torch.zeros(len(self.strides), self.feat_channels))
        self.coord_proj = nn.Linear(3, self.feat_channels)
        self.query_embed = nn.Parameter(
            torch.zeros(self.num_queries, self.feat_channels))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feat_channels,
            nhead=int(num_heads),
            dim_feedforward=self.feat_channels * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(num_decoder_layers))

        self.semantic_proj = nn.Linear(self.feat_channels, self.feat_channels)
        self.box_delta = nn.Linear(self.feat_channels, 5)
        self.quality = nn.Linear(self.feat_channels, 1)

        support = build_support_tokens(
            self.num_classes,
            self.feat_channels,
            scale=support_scale,
            tail_std=support_init_std)
        self.support_tokens = nn.Parameter(support)
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(max(float(logit_scale_init), 1e-4))))
        self.e2e_debug: dict[str, object] = {
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_gaussian_query_prior': self.query_class_prior_weight > 0,
            'uses_denoising_queries': self.dn_loss_weight > 0,
        }

    def init_weights(self) -> None:
        super().init_weights()
        for module in self.input_proj:
            normal_init(module, mean=0.0, std=0.01)
        for module in self.seed_logits:
            normal_init(module, mean=0.0, std=0.01)
            nn.init.constant_(module.bias, -4.0)
        nn.init.normal_(self.level_embed, mean=0.0, std=0.01)
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        normal_init(self.coord_proj, mean=0.0, std=0.01)
        normal_init(self.semantic_proj, mean=0.0, std=0.01)
        normal_init(self.box_delta, mean=0.0, std=0.01)
        nn.init.constant_(self.box_delta.bias, 0.0)
        normal_init(self.quality, mean=0.0, std=0.01)
        nn.init.constant_(self.quality.bias, -2.0)

    def _level_coords(self, feat: Tensor, level_idx: int) -> tuple[Tensor, Tensor]:
        _, _, height, width = feat.shape
        device = feat.device
        dtype = feat.dtype
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        level = torch.full_like(xx, float(level_idx) / max(len(self.strides) - 1, 1))
        coords = torch.stack([xx, yy, level], dim=-1).reshape(-1, 3)
        stride = torch.full((height * width, 1), float(self.strides[level_idx]),
                            device=device, dtype=dtype)
        return coords, stride

    def _flatten_features(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, Tensor,
                                                                    Tensor, Tensor]:
        memories = []
        coords = []
        strides = []
        seed_logits = []
        for idx, feat in enumerate(feats[:len(self.strides)]):
            projected = self.input_proj[idx](feat)
            projected = projected + self.level_embed[idx].view(1, -1, 1, 1)
            seed_logit = self.seed_logits[idx](projected)
            batch_size, _, height, width = projected.shape
            level_coords, level_strides = self._level_coords(projected, idx)
            memories.append(projected.flatten(2).transpose(1, 2))
            coords.append(level_coords)
            strides.append(level_strides)
            seed_logits.append(seed_logit.reshape(batch_size, -1))
        memory = torch.cat(memories, dim=1)
        coord_tensor = torch.cat(coords, dim=0)
        stride_tensor = torch.cat(strides, dim=0)
        seed_logit_tensor = torch.cat(seed_logits, dim=1)
        return memory, coord_tensor, stride_tensor, seed_logit_tensor

    def _query_class_ids(self, num_queries: int, device: torch.device) -> Tensor:
        return torch.arange(num_queries, device=device) % self.num_classes

    def _query_embed_slice(self, num_queries: int, device: torch.device) -> Tensor:
        if num_queries <= self.num_queries:
            return self.query_embed[:num_queries]
        repeat = math.ceil(num_queries / self.num_queries)
        return self.query_embed.repeat(repeat, 1)[:num_queries].to(device)

    def _decode_queries(self,
                        memory: Tensor,
                        query_memory: Tensor,
                        query_coords: Tensor,
                        query_strides: Tensor,
                        query_class_ids: Tensor | None = None,
                        base_wh_override: Tensor | None = None
                        ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = query_memory.shape[1]
        coord_embed = self.coord_proj(query_coords)
        query_embed = self._query_embed_slice(
            num_queries, query_memory.device).unsqueeze(0)
        query = query_memory + coord_embed + query_embed
        if query_class_ids is not None and self.query_class_prior_weight != 0:
            class_prior = self.support_tokens[query_class_ids].unsqueeze(0)
            query = query + self.query_class_prior_weight * class_prior
        decoded = self.decoder(query, memory)

        semantic = F.normalize(self.semantic_proj(decoded), dim=-1)
        support = F.normalize(self.support_tokens, dim=-1)
        cls_logits = semantic @ support.t()
        cls_logits = cls_logits + self.class_bias.view(1, 1, -1)
        if query_class_ids is not None and self.own_class_logit_bias != 0:
            class_bias = F.one_hot(
                query_class_ids,
                num_classes=self.num_classes).to(dtype=cls_logits.dtype)
            cls_logits = cls_logits + self.own_class_logit_bias * class_bias
        cls_logits = cls_logits * self.logit_scale.exp().clamp(1e-4, 100.0)

        raw_box = self.box_delta(decoded)
        image_w = float(self.image_size[0])
        image_h = float(self.image_size[1])
        if base_wh_override is None:
            base_wh = torch.cat([
                (query_strides / image_w).clamp(min=1.0 / image_w),
                (query_strides / image_h).clamp(min=1.0 / image_h)
            ], dim=-1)
        else:
            base_wh = base_wh_override
        center = (query_coords[..., :2] + 0.20 * raw_box[..., :2].tanh()).clamp(
            min=1e-4, max=1.0 - 1e-4)
        wh = (base_wh * torch.exp(
            self.box_delta_scale * raw_box[..., 2:4].tanh())).clamp(
            min=1e-4, max=1.0)
        angle = raw_box[..., 4:5].tanh() * (math.pi / 2.0)
        box_preds = torch.cat([center, wh, angle], dim=-1)
        quality_logits = self.quality(decoded).squeeze(-1)
        return cls_logits, box_preds, quality_logits

    def _select_query_seeds(self, memory: Tensor, coords: Tensor,
                            strides: Tensor, seed_logits: Tensor
                            ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        _, num_tokens, channels = memory.shape
        num_queries = min(self.num_queries, num_tokens)

        _, topk_inds = seed_logits.topk(num_queries, dim=1)
        gather_idx = topk_inds.unsqueeze(-1).expand(-1, -1, channels)
        seed_memory = memory.gather(1, gather_idx)
        seed_coords = coords[topk_inds]
        seed_strides = strides[topk_inds]
        query_class_ids = None
        if self.query_class_prior_weight != 0 or self.own_class_logit_bias != 0:
            query_class_ids = self._query_class_ids(
                num_queries, memory.device)
        return seed_memory, seed_coords, seed_strides, query_class_ids

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, Tensor,
                                                         Tensor, Tensor]:
        memory, coords, strides, seed_logits = self._flatten_features(feats)
        seed_memory, seed_coords, seed_strides, query_class_ids = (
            self._select_query_seeds(memory, coords, strides, seed_logits))
        cls_logits, box_preds, quality_logits = self._decode_queries(
            memory,
            seed_memory,
            seed_coords,
            seed_strides,
            query_class_ids=query_class_ids)
        return cls_logits, box_preds, quality_logits, seed_logits

    def _gt_tensor(self, gt_instances: InstanceData, img_meta: dict) -> Tensor:
        if len(gt_instances) == 0:
            return torch.empty(0, 5, device=self.class_bias.device)
        gt_bboxes = get_box_tensor(gt_instances.bboxes).to(self.class_bias.device)
        img_h, img_w = img_meta['img_shape'][:2]
        norm = gt_bboxes.clone()
        norm[:, 0] = norm[:, 0] / float(img_w)
        norm[:, 1] = norm[:, 1] / float(img_h)
        norm[:, 2] = norm[:, 2] / float(img_w)
        norm[:, 3] = norm[:, 3] / float(img_h)
        return norm

    def _match_single(self, cls_logits: Tensor, box_pred: Tensor,
                      gt_bboxes: Tensor, gt_labels: Tensor) -> tuple[Tensor, Tensor]:
        if gt_bboxes.numel() == 0 or cls_logits.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=cls_logits.device)
            return empty, empty
        prob = cls_logits.sigmoid()
        cls_cost = -prob[:, gt_labels].clamp(min=1e-6).log()
        bbox_cost = torch.cdist(box_pred[:, :4], gt_bboxes[:, :4], p=1)
        angle_cost = torch.cdist(box_pred[:, 4:5], gt_bboxes[:, 4:5], p=1)
        cost = (
            self.match_cls_cost * cls_cost +
            self.match_bbox_cost * bbox_cost +
            self.match_angle_cost * angle_cost)
        row, col = linear_sum_assignment(cost.detach().cpu())
        return (torch.as_tensor(row, dtype=torch.long, device=cls_logits.device),
                torch.as_tensor(col, dtype=torch.long, device=cls_logits.device))

    def _seed_loss_single(self, seed_logits: Tensor, gt_bboxes: Tensor,
                          coords: Tensor) -> Tensor:
        target = seed_logits.new_zeros(seed_logits.shape)
        if gt_bboxes.numel() > 0:
            dist = torch.cdist(coords[:, :2], gt_bboxes[:, :2], p=2)
            nearest = dist.argmin(dim=0)
            target[nearest] = 1.0
        return py_sigmoid_focal_loss(
            seed_logits,
            target,
            alpha=0.25,
            gamma=2.0,
            reduction='mean')

    def _quality_target_for_matches(self, matched_boxes: Tensor,
                                    matched_gt: Tensor) -> Tensor:
        return torch.ones(
            matched_boxes.shape[0],
            device=matched_boxes.device,
            dtype=matched_boxes.dtype)

    def _dn_quality_target(self, box_preds: Tensor, gt_bboxes: Tensor) -> Tensor:
        return torch.ones(
            box_preds.shape[0],
            device=box_preds.device,
            dtype=box_preds.dtype)

    def _denoising_loss_single(self, memory: Tensor, gt_bboxes: Tensor,
                               gt_labels: Tensor) -> tuple[Tensor, Tensor,
                                                           Tensor, Tensor]:
        zero = memory.sum() * 0.0
        if self.dn_loss_weight <= 0 or gt_bboxes.numel() == 0:
            return zero, zero, zero, zero

        max_gt = min(int(gt_bboxes.shape[0]), self.dn_max_gt)
        gt_bboxes = gt_bboxes[:max_gt]
        gt_labels = gt_labels[:max_gt]
        noisy_xy = gt_bboxes[:, :2]
        if self.dn_noise_scale > 0:
            noisy_xy = noisy_xy + torch.randn_like(noisy_xy) * self.dn_noise_scale
        noisy_xy = noisy_xy.clamp(min=1e-4, max=1.0 - 1e-4)
        level = torch.full(
            (max_gt, 1), 0.5, device=memory.device, dtype=memory.dtype)
        query_coords = torch.cat([noisy_xy, level], dim=-1).unsqueeze(0)
        image_w = float(self.image_size[0])
        query_strides = (
            gt_bboxes[:, 2:3].clamp(min=1.0 / image_w) * image_w).unsqueeze(0)
        base_wh = gt_bboxes[:, 2:4].clamp(min=1e-4, max=1.0).unsqueeze(0)
        query_memory = memory.mean(dim=1, keepdim=True).expand(-1, max_gt, -1)

        cls_logits, box_preds, quality_logits = self._decode_queries(
            memory,
            query_memory,
            query_coords,
            query_strides,
            query_class_ids=gt_labels,
            base_wh_override=base_wh)
        cls_logits = cls_logits[0]
        box_preds = box_preds[0]
        quality_logits = quality_logits[0]

        cls_target = torch.zeros_like(cls_logits)
        cls_target[torch.arange(max_gt, device=memory.device), gt_labels] = 1.0
        dn_cls = py_sigmoid_focal_loss(
            cls_logits, cls_target, alpha=0.25, gamma=2.0, reduction='sum')
        dn_bbox = F.l1_loss(
            box_preds[:, :4], gt_bboxes[:, :4], reduction='sum')
        dn_angle = F.l1_loss(
            box_preds[:, 4:5], gt_bboxes[:, 4:5], reduction='sum')
        dn_quality = F.binary_cross_entropy_with_logits(
            quality_logits,
            self._dn_quality_target(box_preds, gt_bboxes),
            reduction='sum')
        normalizer = max(max_gt, 1)
        return (
            dn_cls / normalizer,
            dn_bbox / normalizer,
            dn_angle / normalizer,
            dn_quality / normalizer,
        )

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        cls_logits, box_preds, quality_logits, seed_logits = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, _, _ = self._flatten_features(x)

        total_cls = cls_logits.sum() * 0.0
        total_bbox = box_preds.sum() * 0.0
        total_angle = box_preds.sum() * 0.0
        total_quality = quality_logits.sum() * 0.0
        total_seed = seed_logits.sum() * 0.0
        total_dn_cls = cls_logits.sum() * 0.0
        total_dn_bbox = box_preds.sum() * 0.0
        total_dn_angle = box_preds.sum() * 0.0
        total_dn_quality = quality_logits.sum() * 0.0
        total_pos = 0

        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            matched_q, matched_gt = self._match_single(
                cls_logits[img_idx], box_preds[img_idx], gt_bboxes, gt_labels)

            cls_target = torch.zeros_like(cls_logits[img_idx])
            quality_target = torch.zeros_like(quality_logits[img_idx])
            if matched_q.numel() > 0:
                cls_target[matched_q, gt_labels[matched_gt]] = 1.0
                quality_target[matched_q] = self._quality_target_for_matches(
                    box_preds[img_idx, matched_q], gt_bboxes[matched_gt])
                total_bbox = total_bbox + F.l1_loss(
                    box_preds[img_idx, matched_q, :4],
                    gt_bboxes[matched_gt, :4],
                    reduction='sum')
                total_angle = total_angle + F.l1_loss(
                    box_preds[img_idx, matched_q, 4:5],
                    gt_bboxes[matched_gt, 4:5],
                    reduction='sum')
                total_pos += int(matched_q.numel())

            total_cls = total_cls + py_sigmoid_focal_loss(
                cls_logits[img_idx],
                cls_target,
                alpha=0.25,
                gamma=2.0,
                reduction='sum')
            total_quality = total_quality + F.binary_cross_entropy_with_logits(
                quality_logits[img_idx], quality_target, reduction='sum')
            total_seed = total_seed + self._seed_loss_single(
                seed_logits[img_idx], gt_bboxes, coords)
            dn_cls, dn_bbox, dn_angle, dn_quality = self._denoising_loss_single(
                memory[img_idx:img_idx + 1], gt_bboxes, gt_labels)
            total_dn_cls = total_dn_cls + dn_cls
            total_dn_bbox = total_dn_bbox + dn_bbox
            total_dn_angle = total_dn_angle + dn_angle
            total_dn_quality = total_dn_quality + dn_quality

        avg_pos = max(total_pos, 1)
        batch_size = max(len(batch_gt_instances), 1)
        losses = {
            'loss_cls': total_cls * self.cls_loss_weight / avg_pos,
            'loss_bbox': total_bbox * self.bbox_loss_weight / avg_pos,
            'loss_angle': total_angle * self.angle_loss_weight / avg_pos,
            'loss_quality': total_quality * self.quality_loss_weight /
            (batch_size * self.num_queries),
            'loss_seed': total_seed * self.seed_loss_weight / batch_size,
        }
        if self.dn_loss_weight > 0:
            losses.update({
                'loss_dn_cls':
                total_dn_cls * self.dn_loss_weight * self.cls_loss_weight /
                batch_size,
                'loss_dn_bbox':
                total_dn_bbox * self.dn_loss_weight * self.bbox_loss_weight /
                batch_size,
                'loss_dn_angle':
                total_dn_angle * self.dn_loss_weight *
                self.angle_loss_weight / batch_size,
                'loss_dn_quality':
                total_dn_quality * self.dn_loss_weight *
                self.quality_loss_weight / batch_size,
            })
        return losses

    def _boxes_to_pixels(self, boxes: Tensor, img_meta: dict,
                         rescale: bool) -> Tensor:
        img_h, img_w = img_meta['img_shape'][:2]
        out = boxes.clone()
        out[:, 0] = out[:, 0] * float(img_w)
        out[:, 1] = out[:, 1] * float(img_h)
        out[:, 2] = out[:, 2] * float(img_w)
        out[:, 3] = out[:, 3] * float(img_h)
        if rescale and 'scale_factor' in img_meta:
            scale = img_meta['scale_factor']
            if not torch.is_tensor(scale):
                scale = out.new_tensor(scale)
            scale = scale.to(device=out.device, dtype=out.dtype).flatten()
            if scale.numel() >= 2:
                out[:, 0] = out[:, 0] / scale[0]
                out[:, 2] = out[:, 2] / scale[0]
                out[:, 1] = out[:, 1] / scale[1]
                out[:, 3] = out[:, 3] / scale[1]
        return out

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            scores_per_class = cls_logits[img_idx].sigmoid()
            quality = quality_logits[img_idx].sigmoid().unsqueeze(-1)
            scores_per_class = scores_per_class * quality
            scores, labels = scores_per_class.max(dim=-1)
            keep = scores >= self.score_thr
            if keep.sum() == 0:
                keep = scores.topk(min(1, scores.numel())).indices
            else:
                keep = keep.nonzero().squeeze(1)
            if keep.numel() > self.max_per_img:
                top = scores[keep].topk(self.max_per_img).indices
                keep = keep[top]
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, keep],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores[keep]
            inst.labels = labels[keep]
            results.append(inst)
        self.e2e_debug = {
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_gaussian_query_prior': self.query_class_prior_weight > 0,
            'uses_denoising_queries': self.dn_loss_weight > 0,
            'num_queries': self.num_queries,
            'max_per_img': self.max_per_img,
        }
        return results


@MODELS.register_module()
class P13FGaussianPriorSetHead(P13EDenseSeedGaussianSetHead):
    """P13F light: class-Gaussian query prior plus denoising supervision."""

    def __init__(self,
                 *args,
                 query_class_prior_weight: float = 1.0,
                 own_class_logit_bias: float = 0.5,
                 dn_loss_weight: float = 1.0,
                 dn_noise_scale: float = 0.015,
                 dn_max_gt: int = 80,
                 **kwargs) -> None:
        super().__init__(
            *args,
            query_class_prior_weight=query_class_prior_weight,
            own_class_logit_bias=own_class_logit_bias,
            dn_loss_weight=dn_loss_weight,
            dn_noise_scale=dn_noise_scale,
            dn_max_gt=dn_max_gt,
            **kwargs)
        self.e2e_debug.update({
            'p13_variant': 'P13F Gaussian Prior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_gaussian_query_prior': True,
            'uses_denoising_queries': True,
        })


@MODELS.register_module()
class P13GCalibratedGaussianSetHead(P13FGaussianPriorSetHead):
    """P13G: calibrated strict-E2E variant for reducing P13F FP flooding."""

    def __init__(self,
                 *args,
                 seed_gaussian_sigma: float = 0.035,
                 quality_center_sigma: float = 0.12,
                 per_class_query_layout: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed_gaussian_sigma = float(seed_gaussian_sigma)
        self.quality_center_sigma = float(quality_center_sigma)
        self.per_class_query_layout = bool(per_class_query_layout)
        self.e2e_debug.update({
            'p13_variant': 'P13G Calibrated Gaussian Set Decoder',
            'uses_gaussian_seed_heatmap': True,
            'uses_calibrated_quality': True,
            'uses_per_class_query_layout': self.per_class_query_layout,
        })

    def _select_query_seeds(self, memory: Tensor, coords: Tensor,
                            strides: Tensor, seed_logits: Tensor
                            ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if not self.per_class_query_layout:
            return super()._select_query_seeds(
                memory, coords, strides, seed_logits)

        batch_size, num_tokens, channels = memory.shape
        num_queries = min(self.num_queries, num_tokens * self.num_classes)
        per_class = max(1, math.ceil(num_queries / self.num_classes))
        per_class = min(per_class, num_tokens)
        _, base_inds = seed_logits.topk(per_class, dim=1)

        class_ids = torch.arange(
            self.num_classes, device=memory.device).repeat_interleave(per_class)
        class_ids = class_ids[:num_queries]
        repeat = math.ceil(num_queries / per_class)
        query_inds = base_inds.repeat(1, repeat)[:, :num_queries]

        gather_idx = query_inds.unsqueeze(-1).expand(-1, -1, channels)
        seed_memory = memory.gather(1, gather_idx)
        seed_coords = coords[query_inds]
        seed_strides = strides[query_inds]
        if class_ids.numel() < num_queries:
            pad = self._query_class_ids(
                num_queries - class_ids.numel(), memory.device)
            class_ids = torch.cat([class_ids, pad], dim=0)
        return seed_memory, seed_coords, seed_strides, class_ids

    def _seed_loss_single(self, seed_logits: Tensor, gt_bboxes: Tensor,
                          coords: Tensor) -> Tensor:
        target = seed_logits.new_zeros(seed_logits.shape)
        if gt_bboxes.numel() > 0:
            dist = torch.cdist(coords[:, :2], gt_bboxes[:, :2], p=2)
            sigma = max(self.seed_gaussian_sigma, 1e-4)
            heatmap = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
            target = heatmap.max(dim=1).values.clamp(max=1.0)
        return F.binary_cross_entropy_with_logits(
            seed_logits, target, reduction='mean')

    def _center_quality(self, pred_boxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        if pred_boxes.numel() == 0:
            return pred_boxes.new_zeros(0)
        dist = torch.linalg.vector_norm(
            pred_boxes[:, :2] - gt_bboxes[:, :2], dim=-1)
        sigma = max(self.quality_center_sigma, 1e-4)
        return torch.exp(-(dist * dist) / (2.0 * sigma * sigma)).clamp(
            min=0.0, max=1.0)

    def _quality_target_for_matches(self, matched_boxes: Tensor,
                                    matched_gt: Tensor) -> Tensor:
        return self._center_quality(matched_boxes, matched_gt)

    def _dn_quality_target(self, box_preds: Tensor, gt_bboxes: Tensor) -> Tensor:
        return self._center_quality(box_preds, gt_bboxes)

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p13_variant': 'P13G Calibrated Gaussian Set Decoder',
            'uses_gaussian_seed_heatmap': True,
            'uses_calibrated_quality': True,
            'uses_per_class_query_layout': self.per_class_query_layout,
        })
        return results

@MODELS.register_module()
class P13HRankedGaussianPosteriorHead(P13GCalibratedGaussianSetHead):
    """P13H: strict-E2E ranked Gaussian posterior set decoder.

    This variant keeps the P13E/F/G no-NMS fixed-query contract, but changes the
    scoring target from independent class/quality BCE into a sortable posterior.
    The extra losses are training-only; inference still returns one set of boxes
    directly from queries.
    """

    def __init__(self,
                 *args,
                 ranking_loss_weight: float = 0.5,
                 ranking_margin: float = 0.15,
                 aux_recall_loss_weight: float = 0.5,
                 aux_recall_sigma: float = 0.08,
                 size_prior_init: Sequence[float] = (0.07, 0.07),
                 size_prior_mix: float = 0.65,
                 size_prior_sigma: float = 0.55,
                 size_likelihood_weight: float = 0.25,
                 positive_quality_floor: float = 0.35,
                 quality_size_sigma: float = 0.55,
                 quality_angle_sigma: float = 0.50,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_margin = float(ranking_margin)
        self.aux_recall_loss_weight = float(aux_recall_loss_weight)
        self.aux_recall_sigma = float(aux_recall_sigma)
        self.size_prior_mix = float(size_prior_mix)
        self.size_prior_sigma = float(size_prior_sigma)
        self.size_likelihood_weight = float(size_likelihood_weight)
        self.positive_quality_floor = float(positive_quality_floor)
        self.quality_size_sigma = float(quality_size_sigma)
        self.quality_angle_sigma = float(quality_angle_sigma)

        if len(size_prior_init) != 2:
            raise ValueError('size_prior_init must contain normalized w and h')
        init_wh = torch.tensor(size_prior_init, dtype=torch.float32).clamp(
            min=1e-4, max=1.0)
        self.class_log_wh = nn.Parameter(
            init_wh.log().repeat(self.num_classes, 1))
        self.e2e_debug.update({
            'p13_variant': 'P13H Ranked Gaussian Posterior Set Decoder',
            'uses_ranked_posterior': True,
            'uses_size_prior': True,
            'uses_one_to_many_aux': self.aux_recall_loss_weight > 0,
        })

    def _prior_base_wh(self, query_strides: Tensor,
                       query_class_ids: Tensor | None) -> Tensor | None:
        if query_class_ids is None:
            return None
        image_w = float(self.image_size[0])
        image_h = float(self.image_size[1])
        stride_base = torch.cat([
            (query_strides / image_w).clamp(min=1.0 / image_w),
            (query_strides / image_h).clamp(min=1.0 / image_h)
        ], dim=-1)
        prior = self.class_log_wh[query_class_ids].exp().clamp(
            min=1e-4, max=1.0)
        prior = prior.unsqueeze(0).expand_as(stride_base)
        mix = min(max(self.size_prior_mix, 0.0), 1.0)
        return (1.0 - mix) * stride_base + mix * prior

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, Tensor,
                                                         Tensor, Tensor]:
        memory, coords, strides, seed_logits = self._flatten_features(feats)
        seed_memory, seed_coords, seed_strides, query_class_ids = (
            self._select_query_seeds(memory, coords, strides, seed_logits))
        cls_logits, box_preds, quality_logits = self._decode_queries(
            memory,
            seed_memory,
            seed_coords,
            seed_strides,
            query_class_ids=query_class_ids,
            base_wh_override=self._prior_base_wh(seed_strides, query_class_ids))
        return cls_logits, box_preds, quality_logits, seed_logits

    def _class_size_likelihood(self, box_preds: Tensor) -> Tensor:
        log_wh = box_preds[..., 2:4].clamp(min=1e-4).log().unsqueeze(-2)
        prior = self.class_log_wh.view(1, 1, self.num_classes, 2)
        diff = log_wh - prior
        sigma = max(self.size_prior_sigma, 1e-4)
        likelihood = torch.exp(
            -(diff * diff).sum(dim=-1) / (2.0 * sigma * sigma))
        if self.size_likelihood_weight != 1.0:
            likelihood = likelihood.clamp(min=1e-6).pow(
                max(self.size_likelihood_weight, 0.0))
        return likelihood.clamp(min=0.0, max=1.0)

    def _posterior_scores(self, cls_logits: Tensor, box_preds: Tensor,
                          quality_logits: Tensor) -> Tensor:
        class_score = cls_logits.sigmoid()
        object_score = quality_logits.sigmoid().unsqueeze(-1)
        size_likelihood = self._class_size_likelihood(box_preds)
        return class_score * object_score * size_likelihood

    def _joint_geometry_quality(self, pred_boxes: Tensor,
                                gt_bboxes: Tensor) -> Tensor:
        if pred_boxes.numel() == 0:
            return pred_boxes.new_zeros(0)
        center = self._center_quality(pred_boxes, gt_bboxes)
        log_diff = (
            pred_boxes[:, 2:4].clamp(min=1e-4).log() -
            gt_bboxes[:, 2:4].clamp(min=1e-4).log())
        size_sigma = max(self.quality_size_sigma, 1e-4)
        size_score = torch.exp(
            -(log_diff * log_diff).sum(dim=-1) /
            (2.0 * size_sigma * size_sigma))
        angle_diff = torch.atan2(
            torch.sin(pred_boxes[:, 4] - gt_bboxes[:, 4]),
            torch.cos(pred_boxes[:, 4] - gt_bboxes[:, 4]))
        angle_sigma = max(self.quality_angle_sigma, 1e-4)
        angle_score = torch.exp(
            -(angle_diff * angle_diff) / (2.0 * angle_sigma * angle_sigma))
        target = (center * size_score * angle_score).clamp(min=0.0, max=1.0)
        floor = min(max(self.positive_quality_floor, 0.0), 1.0)
        return floor + (1.0 - floor) * target

    def _quality_target_for_matches(self, matched_boxes: Tensor,
                                    matched_gt: Tensor) -> Tensor:
        return self._joint_geometry_quality(matched_boxes, matched_gt)

    def _dn_quality_target(self, box_preds: Tensor, gt_bboxes: Tensor) -> Tensor:
        return self._joint_geometry_quality(box_preds, gt_bboxes)

    def _ranking_loss_single(self, cls_logits: Tensor, box_preds: Tensor,
                             quality_logits: Tensor, matched_q: Tensor,
                             matched_gt: Tensor, gt_labels: Tensor) -> Tensor:
        zero = cls_logits.sum() * 0.0
        if matched_q.numel() == 0:
            return zero
        posterior = self._posterior_scores(
            cls_logits.unsqueeze(0),
            box_preds.unsqueeze(0),
            quality_logits.unsqueeze(0))[0]
        pos_scores = posterior[matched_q, gt_labels[matched_gt]]
        unmatched = torch.ones(
            posterior.shape[0], dtype=torch.bool, device=posterior.device)
        unmatched[matched_q] = False
        if unmatched.sum() == 0:
            return zero
        neg_scores = posterior[unmatched].amax(dim=-1)
        hard_neg = neg_scores.topk(
            min(int(neg_scores.numel()), max(int(pos_scores.numel()) * 4, 1))
        ).values
        return F.relu(
            self.ranking_margin - pos_scores[:, None] +
            hard_neg[None, :]).mean()

    def _aux_recall_loss_single(self, cls_logits: Tensor, box_preds: Tensor,
                                quality_logits: Tensor, gt_bboxes: Tensor,
                                gt_labels: Tensor) -> tuple[Tensor, Tensor]:
        zero = cls_logits.sum() * 0.0
        if gt_bboxes.numel() == 0 or self.aux_recall_loss_weight <= 0:
            return zero, zero

        dist = torch.cdist(box_preds[:, :2], gt_bboxes[:, :2], p=2)
        sigma = max(self.aux_recall_sigma, 1e-4)
        heatmap = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
        soft_obj, nearest_gt = heatmap.max(dim=1)
        soft_obj = soft_obj.clamp(min=0.0, max=1.0)

        cls_target = torch.zeros_like(cls_logits)
        nearest_labels = gt_labels[nearest_gt]
        cls_target[torch.arange(
            cls_logits.shape[0], device=cls_logits.device),
                   nearest_labels] = soft_obj
        aux_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / max(int(gt_bboxes.shape[0]), 1)
        aux_quality = F.binary_cross_entropy_with_logits(
            quality_logits, soft_obj, reduction='mean')
        return aux_cls, aux_quality

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        cls_logits, box_preds, quality_logits, seed_logits = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, _, _ = self._flatten_features(x)

        total_cls = cls_logits.sum() * 0.0
        total_bbox = box_preds.sum() * 0.0
        total_angle = box_preds.sum() * 0.0
        total_quality = quality_logits.sum() * 0.0
        total_seed = seed_logits.sum() * 0.0
        total_rank = cls_logits.sum() * 0.0
        total_aux_cls = cls_logits.sum() * 0.0
        total_aux_quality = quality_logits.sum() * 0.0
        total_dn_cls = cls_logits.sum() * 0.0
        total_dn_bbox = box_preds.sum() * 0.0
        total_dn_angle = box_preds.sum() * 0.0
        total_dn_quality = quality_logits.sum() * 0.0
        total_pos = 0

        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            matched_q, matched_gt = self._match_single(
                cls_logits[img_idx], box_preds[img_idx], gt_bboxes, gt_labels)

            cls_target = torch.zeros_like(cls_logits[img_idx])
            quality_target = torch.zeros_like(quality_logits[img_idx])
            if matched_q.numel() > 0:
                cls_target[matched_q, gt_labels[matched_gt]] = 1.0
                quality_target[matched_q] = self._quality_target_for_matches(
                    box_preds[img_idx, matched_q], gt_bboxes[matched_gt])
                total_bbox = total_bbox + F.l1_loss(
                    box_preds[img_idx, matched_q, :4],
                    gt_bboxes[matched_gt, :4],
                    reduction='sum')
                total_angle = total_angle + F.l1_loss(
                    box_preds[img_idx, matched_q, 4:5],
                    gt_bboxes[matched_gt, 4:5],
                    reduction='sum')
                total_rank = total_rank + self._ranking_loss_single(
                    cls_logits[img_idx],
                    box_preds[img_idx],
                    quality_logits[img_idx],
                    matched_q,
                    matched_gt,
                    gt_labels)
                total_pos += int(matched_q.numel())

            total_cls = total_cls + py_sigmoid_focal_loss(
                cls_logits[img_idx],
                cls_target,
                alpha=0.25,
                gamma=2.0,
                reduction='sum')
            total_quality = total_quality + F.binary_cross_entropy_with_logits(
                quality_logits[img_idx], quality_target, reduction='sum')
            total_seed = total_seed + self._seed_loss_single(
                seed_logits[img_idx], gt_bboxes, coords)
            aux_cls, aux_quality = self._aux_recall_loss_single(
                cls_logits[img_idx],
                box_preds[img_idx],
                quality_logits[img_idx],
                gt_bboxes,
                gt_labels)
            total_aux_cls = total_aux_cls + aux_cls
            total_aux_quality = total_aux_quality + aux_quality
            dn_cls, dn_bbox, dn_angle, dn_quality = self._denoising_loss_single(
                memory[img_idx:img_idx + 1], gt_bboxes, gt_labels)
            total_dn_cls = total_dn_cls + dn_cls
            total_dn_bbox = total_dn_bbox + dn_bbox
            total_dn_angle = total_dn_angle + dn_angle
            total_dn_quality = total_dn_quality + dn_quality

        avg_pos = max(total_pos, 1)
        batch_size = max(len(batch_gt_instances), 1)
        losses = {
            'loss_cls': total_cls * self.cls_loss_weight / avg_pos,
            'loss_bbox': total_bbox * self.bbox_loss_weight / avg_pos,
            'loss_angle': total_angle * self.angle_loss_weight / avg_pos,
            'loss_quality': total_quality * self.quality_loss_weight /
            (batch_size * self.num_queries),
            'loss_seed': total_seed * self.seed_loss_weight / batch_size,
            'loss_rank': total_rank * self.ranking_loss_weight / batch_size,
            'loss_aux_recall_cls':
            total_aux_cls * self.aux_recall_loss_weight / batch_size,
            'loss_aux_recall_quality':
            total_aux_quality * self.aux_recall_loss_weight / batch_size,
        }
        if self.dn_loss_weight > 0:
            losses.update({
                'loss_dn_cls':
                total_dn_cls * self.dn_loss_weight * self.cls_loss_weight /
                batch_size,
                'loss_dn_bbox':
                total_dn_bbox * self.dn_loss_weight * self.bbox_loss_weight /
                batch_size,
                'loss_dn_angle':
                total_dn_angle * self.dn_loss_weight *
                self.angle_loss_weight / batch_size,
                'loss_dn_quality':
                total_dn_quality * self.dn_loss_weight *
                self.quality_loss_weight / batch_size,
            })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            scores_per_class = self._posterior_scores(
                cls_logits[img_idx:img_idx + 1],
                box_preds[img_idx:img_idx + 1],
                quality_logits[img_idx:img_idx + 1])[0]
            scores, labels = scores_per_class.max(dim=-1)
            keep = scores >= self.score_thr
            if keep.sum() == 0:
                keep = scores.topk(min(1, scores.numel())).indices
            else:
                keep = keep.nonzero().squeeze(1)
            if keep.numel() > self.max_per_img:
                top = scores[keep].topk(self.max_per_img).indices
                keep = keep[top]
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, keep],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores[keep]
            inst.labels = labels[keep]
            results.append(inst)
        self.e2e_debug.update({
            'p13_variant': 'P13H Ranked Gaussian Posterior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_ranked_posterior': True,
            'uses_size_prior': True,
            'uses_one_to_many_aux': self.aux_recall_loss_weight > 0,
        })
        return results


@MODELS.register_module()
class P13IStratifiedLatticePosteriorHead(P13HRankedGaussianPosteriorHead):
    """P13I: strict-E2E lattice posterior with one-to-many box pull.

    P13H/H2 showed that score ranking can train while query centers still miss
    objects.  This variant removes seed top-k as the query bottleneck: each
    class receives the same deterministic spatial lattice, then Gaussian
    auxiliary targets pull nearby queries toward GT boxes.
    """

    def __init__(self,
                 *args,
                 aux_box_loss_weight: float = 1.0,
                 aux_box_angle_loss_weight: float = 0.5,
                 aux_box_min_weight: float = 0.05,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aux_box_loss_weight = float(aux_box_loss_weight)
        self.aux_box_angle_loss_weight = float(aux_box_angle_loss_weight)
        self.aux_box_min_weight = float(aux_box_min_weight)
        self.e2e_debug.update({
            'p13_variant': 'P13I Stratified Lattice Posterior Set Decoder',
            'uses_stratified_lattice_queries': True,
            'uses_auxiliary_box_pull': self.aux_box_loss_weight > 0,
        })

    def _lattice_base_indices(self, coords: Tensor, per_class: int) -> Tensor:
        device = coords.device
        dtype = coords.dtype
        cols = max(1, int(math.ceil(math.sqrt(per_class))))
        rows = max(1, int(math.ceil(per_class / cols)))
        xs = (torch.arange(cols, device=device, dtype=dtype) + 0.5) / cols
        ys = (torch.arange(rows, device=device, dtype=dtype) + 0.5) / rows
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        lattice = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        lattice = lattice[:per_class]
        dist = torch.cdist(lattice, coords[:, :2], p=2)
        return dist.argmin(dim=1)

    def _select_query_seeds(self, memory: Tensor, coords: Tensor,
                            strides: Tensor, seed_logits: Tensor
                            ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        batch_size, num_tokens, channels = memory.shape
        num_queries = min(self.num_queries, num_tokens * self.num_classes)
        per_class = max(1, math.ceil(num_queries / self.num_classes))
        per_class = min(per_class, num_tokens)

        base = self._lattice_base_indices(coords, per_class)
        base = base.view(1, -1).expand(batch_size, -1)
        repeat = math.ceil(num_queries / per_class)
        query_inds = base.repeat(1, repeat)[:, :num_queries]

        class_ids = torch.arange(
            self.num_classes, device=memory.device).repeat_interleave(per_class)
        class_ids = class_ids[:num_queries]
        if class_ids.numel() < num_queries:
            pad = self._query_class_ids(
                num_queries - class_ids.numel(), memory.device)
            class_ids = torch.cat([class_ids, pad], dim=0)

        gather_idx = query_inds.unsqueeze(-1).expand(-1, -1, channels)
        seed_memory = memory.gather(1, gather_idx)
        seed_coords = coords[query_inds]
        seed_strides = strides[query_inds]
        return seed_memory, seed_coords, seed_strides, class_ids

    def _aux_recall_loss_single(self, cls_logits: Tensor, box_preds: Tensor,
                                quality_logits: Tensor, gt_bboxes: Tensor,
                                gt_labels: Tensor
                                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        zero = cls_logits.sum() * 0.0
        if gt_bboxes.numel() == 0 or self.aux_recall_loss_weight <= 0:
            return zero, zero, zero, zero

        dist = torch.cdist(box_preds[:, :2], gt_bboxes[:, :2], p=2)
        sigma = max(self.aux_recall_sigma, 1e-4)
        heatmap = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
        soft_obj, nearest_gt = heatmap.max(dim=1)
        soft_obj = soft_obj.clamp(min=0.0, max=1.0)

        cls_target = torch.zeros_like(cls_logits)
        nearest_labels = gt_labels[nearest_gt]
        cls_target[torch.arange(
            cls_logits.shape[0], device=cls_logits.device),
                   nearest_labels] = soft_obj
        aux_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / max(int(gt_bboxes.shape[0]), 1)
        aux_quality = F.binary_cross_entropy_with_logits(
            quality_logits, soft_obj, reduction='mean')

        weights = soft_obj.detach()
        if self.aux_box_min_weight > 0:
            weights = weights * (weights >= self.aux_box_min_weight).to(
                dtype=weights.dtype)
        normalizer = weights.sum().clamp_min(1.0)
        target_boxes = gt_bboxes[nearest_gt]
        bbox_l1 = (box_preds[:, :4] - target_boxes[:, :4]).abs().sum(dim=-1)
        angle_diff = torch.atan2(
            torch.sin(box_preds[:, 4] - target_boxes[:, 4]),
            torch.cos(box_preds[:, 4] - target_boxes[:, 4])).abs()
        aux_bbox = (bbox_l1 * weights).sum() / normalizer
        aux_angle = (angle_diff * weights).sum() / normalizer
        return aux_cls, aux_quality, aux_bbox, aux_angle

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        cls_logits, box_preds, quality_logits, seed_logits = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, _, _ = self._flatten_features(x)

        total_cls = cls_logits.sum() * 0.0
        total_bbox = box_preds.sum() * 0.0
        total_angle = box_preds.sum() * 0.0
        total_quality = quality_logits.sum() * 0.0
        total_seed = seed_logits.sum() * 0.0
        total_rank = cls_logits.sum() * 0.0
        total_aux_cls = cls_logits.sum() * 0.0
        total_aux_quality = quality_logits.sum() * 0.0
        total_aux_bbox = box_preds.sum() * 0.0
        total_aux_angle = box_preds.sum() * 0.0
        total_dn_cls = cls_logits.sum() * 0.0
        total_dn_bbox = box_preds.sum() * 0.0
        total_dn_angle = box_preds.sum() * 0.0
        total_dn_quality = quality_logits.sum() * 0.0
        total_pos = 0

        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            matched_q, matched_gt = self._match_single(
                cls_logits[img_idx], box_preds[img_idx], gt_bboxes, gt_labels)

            cls_target = torch.zeros_like(cls_logits[img_idx])
            quality_target = torch.zeros_like(quality_logits[img_idx])
            if matched_q.numel() > 0:
                cls_target[matched_q, gt_labels[matched_gt]] = 1.0
                quality_target[matched_q] = self._quality_target_for_matches(
                    box_preds[img_idx, matched_q], gt_bboxes[matched_gt])
                total_bbox = total_bbox + F.l1_loss(
                    box_preds[img_idx, matched_q, :4],
                    gt_bboxes[matched_gt, :4],
                    reduction='sum')
                total_angle = total_angle + F.l1_loss(
                    box_preds[img_idx, matched_q, 4:5],
                    gt_bboxes[matched_gt, 4:5],
                    reduction='sum')
                total_rank = total_rank + self._ranking_loss_single(
                    cls_logits[img_idx],
                    box_preds[img_idx],
                    quality_logits[img_idx],
                    matched_q,
                    matched_gt,
                    gt_labels)
                total_pos += int(matched_q.numel())

            total_cls = total_cls + py_sigmoid_focal_loss(
                cls_logits[img_idx],
                cls_target,
                alpha=0.25,
                gamma=2.0,
                reduction='sum')
            total_quality = total_quality + F.binary_cross_entropy_with_logits(
                quality_logits[img_idx], quality_target, reduction='sum')
            total_seed = total_seed + self._seed_loss_single(
                seed_logits[img_idx], gt_bboxes, coords)
            aux_cls, aux_quality, aux_bbox, aux_angle = (
                self._aux_recall_loss_single(
                    cls_logits[img_idx],
                    box_preds[img_idx],
                    quality_logits[img_idx],
                    gt_bboxes,
                    gt_labels))
            total_aux_cls = total_aux_cls + aux_cls
            total_aux_quality = total_aux_quality + aux_quality
            total_aux_bbox = total_aux_bbox + aux_bbox
            total_aux_angle = total_aux_angle + aux_angle
            dn_cls, dn_bbox, dn_angle, dn_quality = self._denoising_loss_single(
                memory[img_idx:img_idx + 1], gt_bboxes, gt_labels)
            total_dn_cls = total_dn_cls + dn_cls
            total_dn_bbox = total_dn_bbox + dn_bbox
            total_dn_angle = total_dn_angle + dn_angle
            total_dn_quality = total_dn_quality + dn_quality

        avg_pos = max(total_pos, 1)
        batch_size = max(len(batch_gt_instances), 1)
        losses = {
            'loss_cls': total_cls * self.cls_loss_weight / avg_pos,
            'loss_bbox': total_bbox * self.bbox_loss_weight / avg_pos,
            'loss_angle': total_angle * self.angle_loss_weight / avg_pos,
            'loss_quality': total_quality * self.quality_loss_weight /
            (batch_size * self.num_queries),
            'loss_seed': total_seed * self.seed_loss_weight / batch_size,
            'loss_rank': total_rank * self.ranking_loss_weight / batch_size,
            'loss_aux_recall_cls':
            total_aux_cls * self.aux_recall_loss_weight / batch_size,
            'loss_aux_recall_quality':
            total_aux_quality * self.aux_recall_loss_weight / batch_size,
            'loss_aux_recall_bbox':
            total_aux_bbox * self.aux_box_loss_weight / batch_size,
            'loss_aux_recall_angle':
            total_aux_angle * self.aux_box_angle_loss_weight / batch_size,
        }
        if self.dn_loss_weight > 0:
            losses.update({
                'loss_dn_cls':
                total_dn_cls * self.dn_loss_weight * self.cls_loss_weight /
                batch_size,
                'loss_dn_bbox':
                total_dn_bbox * self.dn_loss_weight * self.bbox_loss_weight /
                batch_size,
                'loss_dn_angle':
                total_dn_angle * self.dn_loss_weight *
                self.angle_loss_weight / batch_size,
                'loss_dn_quality':
                total_dn_quality * self.dn_loss_weight *
                self.quality_loss_weight / batch_size,
            })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p13_variant': 'P13I Stratified Lattice Posterior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_stratified_lattice_queries': True,
            'uses_auxiliary_box_pull': self.aux_box_loss_weight > 0,
        })
        return results


@MODELS.register_module()
class P13JAssignedLatticePosteriorHead(P13IStratifiedLatticePosteriorHead):
    """P13J: strict-E2E lattice posterior with explicit query-GT binding.

    P13I still used predicted box centers to decide how strongly an auxiliary
    target should pull each query. That creates a cold-start loop: a query must
    already localize a GT before receiving strong localization supervision.
    P13J assigns responsibility from deterministic lattice coordinates instead.
    Inference remains a fixed query set with no dense detection head and no NMS.
    """

    def __init__(self,
                 *args,
                 assigned_lattice_loss_weight: float = 1.0,
                 assigned_lattice_sigma: float = 0.14,
                 assigned_lattice_min_weight: float = 0.03,
                 assigned_lattice_topk_floor: float = 0.35,
                 assigned_topk_per_gt: int = 3,
                 assigned_rank_loss_weight: float = 0.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.assigned_lattice_loss_weight = float(assigned_lattice_loss_weight)
        self.assigned_lattice_sigma = float(assigned_lattice_sigma)
        self.assigned_lattice_min_weight = float(assigned_lattice_min_weight)
        self.assigned_lattice_topk_floor = float(assigned_lattice_topk_floor)
        self.assigned_topk_per_gt = int(assigned_topk_per_gt)
        self.assigned_rank_loss_weight = float(assigned_rank_loss_weight)
        self.e2e_debug.update({
            'p13_variant': 'P13J Assigned Lattice Posterior Set Decoder',
            'uses_auxiliary_box_pull': False,
            'uses_assigned_lattice_box_pull': self.aux_box_loss_weight > 0,
            'uses_assigned_lattice_gt_binding': True,
            'uses_predicted_center_for_aux_assignment': False,
        })

    def _assigned_lattice_targets(self, query_coords: Tensor,
                                  query_class_ids: Tensor,
                                  gt_bboxes: Tensor,
                                  gt_labels: Tensor) -> tuple[Tensor, Tensor]:
        soft_obj = query_coords.new_zeros(query_coords.shape[0])
        nearest_gt = torch.zeros(
            query_coords.shape[0], dtype=torch.long, device=query_coords.device)
        if gt_bboxes.numel() == 0 or query_coords.numel() == 0:
            return soft_obj, nearest_gt

        sigma = max(self.assigned_lattice_sigma, 1e-4)
        topk_per_gt = max(int(self.assigned_topk_per_gt), 0)
        topk_floor = min(max(self.assigned_lattice_topk_floor, 0.0), 1.0)
        for class_id in gt_labels.unique():
            gt_mask = gt_labels == class_id
            query_mask = query_class_ids == class_id
            if gt_mask.sum() == 0 or query_mask.sum() == 0:
                continue
            gt_idx = gt_mask.nonzero().squeeze(1)
            query_idx = query_mask.nonzero().squeeze(1)
            dist = torch.cdist(
                query_coords[query_idx, :2], gt_bboxes[gt_idx, :2], p=2)
            heatmap = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))

            class_weight, local_nearest = heatmap.max(dim=1)
            update = class_weight > soft_obj[query_idx]
            if update.any():
                update_idx = query_idx[update]
                soft_obj[update_idx] = class_weight[update]
                nearest_gt[update_idx] = gt_idx[local_nearest[update]]

            if topk_per_gt <= 0:
                continue
            k = min(topk_per_gt, int(query_idx.numel()))
            for local_gt, global_gt in enumerate(gt_idx):
                top_vals, top_rows = heatmap[:, local_gt].topk(k)
                top_vals = top_vals.clamp_min(topk_floor)
                top_query_idx = query_idx[top_rows]
                improve = top_vals > soft_obj[top_query_idx]
                if improve.any():
                    improved_q = top_query_idx[improve]
                    soft_obj[improved_q] = top_vals[improve]
                    nearest_gt[improved_q] = global_gt

        return soft_obj.clamp(min=0.0, max=1.0), nearest_gt

    def _assigned_quality_loss(self, quality_logits: Tensor,
                               soft_obj: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            quality_logits, soft_obj, reduction='mean')

    def _assigned_rank_loss(self, cls_logits: Tensor, box_preds: Tensor,
                            quality_logits: Tensor, query_class_ids: Tensor,
                            soft_obj: Tensor, weights: Tensor) -> Tensor:
        zero = cls_logits.sum() * 0.0
        posterior = self._posterior_scores(
            cls_logits.unsqueeze(0),
            box_preds.unsqueeze(0),
            quality_logits.unsqueeze(0))[0]
        own_scores = posterior[
            torch.arange(cls_logits.shape[0], device=cls_logits.device),
            query_class_ids]
        pos = weights > 0
        neg = soft_obj <= 1e-6
        if pos.any() and neg.any():
            pos_scores = own_scores[pos]
            neg_scores = own_scores[neg]
            hard_neg = neg_scores.topk(
                min(int(neg_scores.numel()),
                    max(int(pos_scores.numel()) * 4, 1))).values
            return F.relu(
                self.ranking_margin - pos_scores[:, None] +
                hard_neg[None, :]).mean()
        return zero

    def _assigned_lattice_loss_single(
            self, cls_logits: Tensor, box_preds: Tensor, quality_logits: Tensor,
            query_coords: Tensor, query_class_ids: Tensor, gt_bboxes: Tensor,
            gt_labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        zero = cls_logits.sum() * 0.0
        if gt_bboxes.numel() == 0 or self.assigned_lattice_loss_weight <= 0:
            return zero, zero, zero, zero, zero

        soft_obj, nearest_gt = self._assigned_lattice_targets(
            query_coords, query_class_ids, gt_bboxes, gt_labels)
        cls_target = torch.zeros_like(cls_logits)
        cls_target[
            torch.arange(cls_logits.shape[0], device=cls_logits.device),
            query_class_ids] = soft_obj
        assigned_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / max(int(gt_bboxes.shape[0]), 1)
        assigned_quality = self._assigned_quality_loss(
            quality_logits, soft_obj)

        weights = soft_obj.detach()
        min_weight = max(self.assigned_lattice_min_weight, 0.0)
        if min_weight > 0:
            weights = weights * (weights >= min_weight).to(dtype=weights.dtype)
        normalizer = weights.sum().clamp_min(1.0)
        target_boxes = gt_bboxes[nearest_gt]
        bbox_l1 = (box_preds[:, :4] - target_boxes[:, :4]).abs().sum(dim=-1)
        angle_diff = torch.atan2(
            torch.sin(box_preds[:, 4] - target_boxes[:, 4]),
            torch.cos(box_preds[:, 4] - target_boxes[:, 4])).abs()
        assigned_bbox = (bbox_l1 * weights).sum() / normalizer
        assigned_angle = (angle_diff * weights).sum() / normalizer

        assigned_rank = self._assigned_rank_loss(
            cls_logits, box_preds, quality_logits, query_class_ids, soft_obj,
            weights)

        return (assigned_cls, assigned_quality, assigned_bbox, assigned_angle,
                assigned_rank)

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        cls_logits, box_preds, quality_logits, seed_logits = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, query_coords, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)

        total_cls = cls_logits.sum() * 0.0
        total_bbox = box_preds.sum() * 0.0
        total_angle = box_preds.sum() * 0.0
        total_quality = quality_logits.sum() * 0.0
        total_seed = seed_logits.sum() * 0.0
        total_rank = cls_logits.sum() * 0.0
        total_assigned_cls = cls_logits.sum() * 0.0
        total_assigned_quality = quality_logits.sum() * 0.0
        total_assigned_bbox = box_preds.sum() * 0.0
        total_assigned_angle = box_preds.sum() * 0.0
        total_assigned_rank = cls_logits.sum() * 0.0
        total_dn_cls = cls_logits.sum() * 0.0
        total_dn_bbox = box_preds.sum() * 0.0
        total_dn_angle = box_preds.sum() * 0.0
        total_dn_quality = quality_logits.sum() * 0.0
        total_pos = 0

        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            matched_q, matched_gt = self._match_single(
                cls_logits[img_idx], box_preds[img_idx], gt_bboxes, gt_labels)

            cls_target = torch.zeros_like(cls_logits[img_idx])
            quality_target = torch.zeros_like(quality_logits[img_idx])
            if matched_q.numel() > 0:
                cls_target[matched_q, gt_labels[matched_gt]] = 1.0
                quality_target[matched_q] = self._quality_target_for_matches(
                    box_preds[img_idx, matched_q], gt_bboxes[matched_gt])
                total_bbox = total_bbox + F.l1_loss(
                    box_preds[img_idx, matched_q, :4],
                    gt_bboxes[matched_gt, :4],
                    reduction='sum')
                total_angle = total_angle + F.l1_loss(
                    box_preds[img_idx, matched_q, 4:5],
                    gt_bboxes[matched_gt, 4:5],
                    reduction='sum')
                total_rank = total_rank + self._ranking_loss_single(
                    cls_logits[img_idx],
                    box_preds[img_idx],
                    quality_logits[img_idx],
                    matched_q,
                    matched_gt,
                    gt_labels)
                total_pos += int(matched_q.numel())

            total_cls = total_cls + py_sigmoid_focal_loss(
                cls_logits[img_idx],
                cls_target,
                alpha=0.25,
                gamma=2.0,
                reduction='sum')
            total_quality = total_quality + F.binary_cross_entropy_with_logits(
                quality_logits[img_idx], quality_target, reduction='sum')
            total_seed = total_seed + self._seed_loss_single(
                seed_logits[img_idx], gt_bboxes, coords)
            assigned = self._assigned_lattice_loss_single(
                cls_logits[img_idx],
                box_preds[img_idx],
                quality_logits[img_idx],
                query_coords[img_idx],
                query_class_ids,
                gt_bboxes,
                gt_labels)
            total_assigned_cls = total_assigned_cls + assigned[0]
            total_assigned_quality = total_assigned_quality + assigned[1]
            total_assigned_bbox = total_assigned_bbox + assigned[2]
            total_assigned_angle = total_assigned_angle + assigned[3]
            total_assigned_rank = total_assigned_rank + assigned[4]
            dn_cls, dn_bbox, dn_angle, dn_quality = self._denoising_loss_single(
                memory[img_idx:img_idx + 1], gt_bboxes, gt_labels)
            total_dn_cls = total_dn_cls + dn_cls
            total_dn_bbox = total_dn_bbox + dn_bbox
            total_dn_angle = total_dn_angle + dn_angle
            total_dn_quality = total_dn_quality + dn_quality

        avg_pos = max(total_pos, 1)
        batch_size = max(len(batch_gt_instances), 1)
        assigned_weight = self.assigned_lattice_loss_weight
        losses = {
            'loss_cls': total_cls * self.cls_loss_weight / avg_pos,
            'loss_bbox': total_bbox * self.bbox_loss_weight / avg_pos,
            'loss_angle': total_angle * self.angle_loss_weight / avg_pos,
            'loss_quality': total_quality * self.quality_loss_weight /
            (batch_size * self.num_queries),
            'loss_seed': total_seed * self.seed_loss_weight / batch_size,
            'loss_rank': total_rank * self.ranking_loss_weight / batch_size,
            'loss_assigned_lattice_cls':
            total_assigned_cls * assigned_weight / batch_size,
            'loss_assigned_lattice_quality':
            total_assigned_quality * assigned_weight / batch_size,
            'loss_assigned_lattice_bbox':
            total_assigned_bbox * assigned_weight * self.aux_box_loss_weight /
            batch_size,
            'loss_assigned_lattice_angle':
            total_assigned_angle * assigned_weight *
            self.aux_box_angle_loss_weight / batch_size,
            'loss_assigned_lattice_rank':
            total_assigned_rank * self.assigned_rank_loss_weight / batch_size,
        }
        if self.dn_loss_weight > 0:
            losses.update({
                'loss_dn_cls':
                total_dn_cls * self.dn_loss_weight * self.cls_loss_weight /
                batch_size,
                'loss_dn_bbox':
                total_dn_bbox * self.dn_loss_weight * self.bbox_loss_weight /
                batch_size,
                'loss_dn_angle':
                total_dn_angle * self.dn_loss_weight *
                self.angle_loss_weight / batch_size,
                'loss_dn_quality':
                total_dn_quality * self.dn_loss_weight *
                self.quality_loss_weight / batch_size,
            })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p13_variant': 'P13J Assigned Lattice Posterior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_stratified_lattice_queries': True,
            'uses_auxiliary_box_pull': False,
            'uses_assigned_lattice_box_pull': self.aux_box_loss_weight > 0,
            'uses_assigned_lattice_gt_binding': True,
            'uses_predicted_center_for_aux_assignment': False,
        })
        return results


@MODELS.register_module()
class P13KBalancedAssignedPosteriorHead(P13JAssignedLatticePosteriorHead):
    """P13K: strict-E2E assigned posterior with positive quality balancing.

    P13J proved that query-coordinate GT binding can train localization, but
    full-threshold inference collapsed because positive quality/rank signals
    were diluted by the fixed query background. P13K keeps the same no-NMS
    fixed set output and changes only the posterior training bias: positive
    assigned queries receive stronger quality pressure, and query centers can
    move farther from the lattice anchor during early training.
    """

    def __init__(self,
                 *args,
                 assigned_positive_quality_gain: float = 3.0,
                 center_delta_scale: float = 0.35,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.assigned_positive_quality_gain = float(
            assigned_positive_quality_gain)
        self.center_delta_scale = float(center_delta_scale)
        self.e2e_debug.update({
            'p13_variant': 'P13K Balanced Assigned Posterior Set Decoder',
            'uses_balanced_assigned_posterior': True,
            'uses_expanded_center_delta': self.center_delta_scale != 0.20,
        })

    def _assigned_quality_loss(self, quality_logits: Tensor,
                               soft_obj: Tensor) -> Tensor:
        per_query = F.binary_cross_entropy_with_logits(
            quality_logits, soft_obj, reduction='none')
        gain = max(self.assigned_positive_quality_gain, 0.0)
        weights = 1.0 + gain * soft_obj.detach()
        return (per_query * weights).sum() / weights.sum().clamp_min(1.0)

    def _decode_queries(self,
                        memory: Tensor,
                        query_memory: Tensor,
                        query_coords: Tensor,
                        query_strides: Tensor,
                        query_class_ids: Tensor | None = None,
                        base_wh_override: Tensor | None = None
                        ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = query_memory.shape[1]
        coord_embed = self.coord_proj(query_coords)
        query_embed = self._query_embed_slice(
            num_queries, query_memory.device).unsqueeze(0)
        query = query_memory + coord_embed + query_embed
        if query_class_ids is not None and self.query_class_prior_weight != 0:
            class_prior = self.support_tokens[query_class_ids].unsqueeze(0)
            query = query + self.query_class_prior_weight * class_prior
        decoded = self.decoder(query, memory)

        semantic = F.normalize(self.semantic_proj(decoded), dim=-1)
        support = F.normalize(self.support_tokens, dim=-1)
        cls_logits = semantic @ support.t()
        cls_logits = cls_logits + self.class_bias.view(1, 1, -1)
        if query_class_ids is not None and self.own_class_logit_bias != 0:
            class_bias = F.one_hot(
                query_class_ids,
                num_classes=self.num_classes).to(dtype=cls_logits.dtype)
            cls_logits = cls_logits + self.own_class_logit_bias * class_bias
        cls_logits = cls_logits * self.logit_scale.exp().clamp(1e-4, 100.0)

        raw_box = self.box_delta(decoded)
        image_w = float(self.image_size[0])
        image_h = float(self.image_size[1])
        if base_wh_override is None:
            base_wh = torch.cat([
                (query_strides / image_w).clamp(min=1.0 / image_w),
                (query_strides / image_h).clamp(min=1.0 / image_h)
            ], dim=-1)
        else:
            base_wh = base_wh_override
        center = (
            query_coords[..., :2] +
            self.center_delta_scale * raw_box[..., :2].tanh()).clamp(
                min=1e-4, max=1.0 - 1e-4)
        wh = (base_wh * torch.exp(
            self.box_delta_scale * raw_box[..., 2:4].tanh())).clamp(
            min=1e-4, max=1.0)
        angle = raw_box[..., 4:5].tanh() * (math.pi / 2.0)
        box_preds = torch.cat([center, wh, angle], dim=-1)
        quality_logits = self.quality(decoded).squeeze(-1)
        return cls_logits, box_preds, quality_logits

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p13_variant': 'P13K Balanced Assigned Posterior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_stratified_lattice_queries': True,
            'uses_auxiliary_box_pull': False,
            'uses_assigned_lattice_box_pull': self.aux_box_loss_weight > 0,
            'uses_assigned_lattice_gt_binding': True,
            'uses_predicted_center_for_aux_assignment': False,
            'uses_balanced_assigned_posterior': True,
            'uses_expanded_center_delta': self.center_delta_scale != 0.20,
        })
        return results


@MODELS.register_module()
class P13LClassLockedBalancedPosteriorHead(
        P13KBalancedAssignedPosteriorHead):
    """P13L: class-locked no-NMS output for stratified query sets.

    P13K still allowed each class-stratified query to choose the maximum over
    all classes at inference time. That is inconsistent with the assigned
    lattice target, which supervises each query's own class id. P13L locks the
    output label to the query class id and applies a per-class quota before the
    final global cap. It remains a fixed set decoder with no dense head/NMS.
    """

    def __init__(self,
                 *args,
                 per_class_max_per_img: int = 12,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.per_class_max_per_img = int(per_class_max_per_img)
        self.e2e_debug.update({
            'p13_variant': 'P13L Class-Locked Balanced Posterior Set Decoder',
            'uses_class_locked_inference': True,
            'uses_per_class_output_quota': self.per_class_max_per_img > 0,
        })

    def _stratified_query_class_ids_for_count(
            self, num_queries: int, device: torch.device) -> Tensor:
        per_class = max(1, math.ceil(num_queries / self.num_classes))
        class_ids = torch.arange(
            self.num_classes, device=device).repeat_interleave(per_class)
        class_ids = class_ids[:num_queries]
        if class_ids.numel() < num_queries:
            pad = self._query_class_ids(
                num_queries - class_ids.numel(), device)
            class_ids = torch.cat([class_ids, pad], dim=0)
        return class_ids

    def _class_locked_select(
            self, scores_per_class: Tensor,
            query_class_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        keep_parts = []
        score_parts = []
        label_parts = []
        per_class_limit = max(self.per_class_max_per_img, 1)
        for class_id in range(self.num_classes):
            class_mask = query_class_ids == class_id
            if not class_mask.any():
                continue
            class_query_idx = class_mask.nonzero().squeeze(1)
            class_scores = scores_per_class[class_query_idx, class_id]
            valid = class_scores >= self.score_thr
            if valid.any():
                class_query_idx = class_query_idx[valid]
                class_scores = class_scores[valid]
            elif self.score_thr > 0:
                continue
            k = min(per_class_limit, int(class_scores.numel()))
            if k <= 0:
                continue
            top = class_scores.topk(k).indices
            selected_idx = class_query_idx[top]
            selected_scores = class_scores[top]
            keep_parts.append(selected_idx)
            score_parts.append(selected_scores)
            label_parts.append(
                torch.full_like(selected_idx, class_id, dtype=torch.long))

        if keep_parts:
            keep = torch.cat(keep_parts, dim=0)
            scores = torch.cat(score_parts, dim=0)
            labels = torch.cat(label_parts, dim=0)
        else:
            own_scores = scores_per_class[
                torch.arange(
                    query_class_ids.numel(), device=query_class_ids.device),
                query_class_ids]
            keep = own_scores.topk(min(1, own_scores.numel())).indices
            scores = own_scores[keep]
            labels = query_class_ids[keep]

        if keep.numel() > self.max_per_img:
            top = scores.topk(self.max_per_img).indices
            keep = keep[top]
            scores = scores[top]
            labels = labels[top]
        return keep, scores, labels

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        query_class_ids = self._stratified_query_class_ids_for_count(
            cls_logits.shape[1], cls_logits.device)
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            scores_per_class = self._posterior_scores(
                cls_logits[img_idx:img_idx + 1],
                box_preds[img_idx:img_idx + 1],
                quality_logits[img_idx:img_idx + 1])[0]
            keep, scores, labels = self._class_locked_select(
                scores_per_class, query_class_ids)
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, keep],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores
            inst.labels = labels
            results.append(inst)
        self.e2e_debug.update({
            'p13_variant': 'P13L Class-Locked Balanced Posterior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_stratified_lattice_queries': True,
            'uses_auxiliary_box_pull': False,
            'uses_assigned_lattice_box_pull': self.aux_box_loss_weight > 0,
            'uses_assigned_lattice_gt_binding': True,
            'uses_predicted_center_for_aux_assignment': False,
            'uses_balanced_assigned_posterior': True,
            'uses_expanded_center_delta': self.center_delta_scale != 0.20,
            'uses_class_locked_inference': True,
            'uses_per_class_output_quota': self.per_class_max_per_img > 0,
        })
        return results


@MODELS.register_module()
class P13MClasswiseRankLockedPosteriorHead(
        P13LClassLockedBalancedPosteriorHead):
    """P13M: class-locked output with classwise assigned ranking.

    Positive assigned queries only compete with hard negatives from the same
    query class. This matches P13L inference, where each class forms an
    independent quota before the final no-NMS set merge.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e2e_debug.update({
            'p13_variant': 'P13M Classwise Rank Locked Posterior Set Decoder',
            'uses_classwise_assigned_rank': True,
        })

    def _assigned_rank_loss(self, cls_logits: Tensor, box_preds: Tensor,
                            quality_logits: Tensor, query_class_ids: Tensor,
                            soft_obj: Tensor, weights: Tensor) -> Tensor:
        zero = cls_logits.sum() * 0.0
        posterior = self._posterior_scores(
            cls_logits.unsqueeze(0),
            box_preds.unsqueeze(0),
            quality_logits.unsqueeze(0))[0]
        own_scores = posterior[
            torch.arange(cls_logits.shape[0], device=cls_logits.device),
            query_class_ids]
        pos_mask = weights > 0
        if not pos_mask.any():
            return zero

        rank_terms = []
        for class_id in query_class_ids[pos_mask].unique():
            same_class = query_class_ids == class_id
            pos = pos_mask & same_class
            neg = (soft_obj <= 1e-6) & same_class
            if not pos.any() or not neg.any():
                continue
            pos_scores = own_scores[pos]
            neg_scores = own_scores[neg]
            hard_neg = neg_scores.topk(
                min(int(neg_scores.numel()),
                    max(int(pos_scores.numel()) * 4, 1))).values
            rank_terms.append(
                F.relu(self.ranking_margin - pos_scores[:, None] +
                       hard_neg[None, :]).mean())
        if not rank_terms:
            return zero
        return torch.stack(rank_terms).mean()

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p13_variant': 'P13M Classwise Rank Locked Posterior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_class_locked_inference': True,
            'uses_per_class_output_quota': self.per_class_max_per_img > 0,
            'uses_classwise_assigned_rank': True,
        })
        return results


@MODELS.register_module()
class P13NTeacherAnchoredPosteriorHead(
        P13MClasswiseRankLockedPosteriorHead):
    """P13N: strict fixed-set posterior trained with teacher fields.

    P13N keeps the inference contract stricter than P13L/M: all fixed queries
    are returned directly as posterior detections. There is no score-threshold
    filtering, quota selection, NMS, or empty-result fallback. A P13C/dense
    teacher may be loaded only for training targets; it is not called at
    inference.
    """

    def __init__(self,
                 *args,
                 teacher_predictions_path: str | None = None,
                 teacher_loss_weight: float = 1.0,
                 teacher_rank_loss_weight: float = 1.0,
                 teacher_gaussian_sigma: float = 0.12,
                 teacher_score_thr: float = 0.15,
                 teacher_topk_per_img: int = 160,
                 teacher_score_power: float = 1.0,
                 teacher_min_weight: float = 0.02,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_predictions_path = teacher_predictions_path
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.teacher_rank_loss_weight = float(teacher_rank_loss_weight)
        self.teacher_gaussian_sigma = float(teacher_gaussian_sigma)
        self.teacher_score_thr = float(teacher_score_thr)
        self.teacher_topk_per_img = int(teacher_topk_per_img)
        self.teacher_score_power = float(teacher_score_power)
        self.teacher_min_weight = float(teacher_min_weight)
        self._teacher_by_img_id = self._load_teacher_predictions(
            teacher_predictions_path)
        self.e2e_debug.update({
            'p13_variant': 'P13N Teacher-Anchored Posterior Set Decoder',
            'uses_teacher_training_only': True,
            'uses_inference_fallback': False,
            'uses_score_threshold_postprocess': False,
            'uses_per_class_output_quota': False,
            'outputs_fixed_query_set': True,
        })

    def _load_teacher_predictions(self, path: str | None) -> dict:
        if not path:
            return {}
        if not os.path.exists(path):
            return {}
        with open(path, 'rb') as f:
            predictions = pickle.load(f)
        mapping = {}
        for item in predictions:
            if not isinstance(item, dict) or 'pred_instances' not in item:
                continue
            keys = []
            for key in ('img_id', 'img_path'):
                if key in item:
                    keys.extend(self._teacher_key_variants(item[key]))
            for key in keys:
                mapping.setdefault(key, item['pred_instances'])
        return mapping

    def _teacher_key_variants(self, value) -> list:
        variants = []
        if value is None:
            return variants
        variants.append(value)
        variants.append(str(value))
        if isinstance(value, str):
            base = os.path.basename(value)
            stem, _ = os.path.splitext(base)
            variants.extend([base, stem])
        return variants

    def _teacher_instances_for_sample(
            self, data_sample) -> tuple[Tensor, Tensor, Tensor]:
        device = self.class_bias.device
        empty_bboxes = torch.empty(0, 5, device=device)
        empty_labels = torch.empty(0, dtype=torch.long, device=device)
        empty_scores = torch.empty(0, device=device)
        if not self._teacher_by_img_id:
            return empty_bboxes, empty_labels, empty_scores

        img_meta = data_sample.metainfo
        inst = None
        for meta_key in ('img_id', 'img_path'):
            if meta_key not in img_meta:
                continue
            for key in self._teacher_key_variants(img_meta[meta_key]):
                if key in self._teacher_by_img_id:
                    inst = self._teacher_by_img_id[key]
                    break
            if inst is not None:
                break
        if inst is None or not hasattr(inst, 'bboxes'):
            return empty_bboxes, empty_labels, empty_scores

        bboxes = get_box_tensor(inst.bboxes).to(device=device).float()
        labels = inst.labels.to(device=device).long()
        scores = inst.scores.to(device=device).float()
        if bboxes.numel() == 0:
            return empty_bboxes, empty_labels, empty_scores

        keep = scores >= self.teacher_score_thr
        if keep.any():
            bboxes = bboxes[keep]
            labels = labels[keep]
            scores = scores[keep]
        else:
            return empty_bboxes, empty_labels, empty_scores
        if self.teacher_topk_per_img > 0 and scores.numel(
        ) > self.teacher_topk_per_img:
            top = scores.topk(self.teacher_topk_per_img).indices
            bboxes = bboxes[top]
            labels = labels[top]
            scores = scores[top]

        if bboxes[:, :4].max() > 2.0:
            img_h, img_w = img_meta['img_shape'][:2]
            bboxes = bboxes.clone()
            bboxes[:, 0] = bboxes[:, 0] / float(img_w)
            bboxes[:, 1] = bboxes[:, 1] / float(img_h)
            bboxes[:, 2] = bboxes[:, 2] / float(img_w)
            bboxes[:, 3] = bboxes[:, 3] / float(img_h)
        return bboxes, labels, scores

    def _teacher_augmented_targets(
            self, query_coords: Tensor, query_class_ids: Tensor,
            gt_bboxes: Tensor, gt_labels: Tensor, teacher_bboxes: Tensor,
            teacher_labels: Tensor, teacher_scores: Tensor
            ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = int(query_coords.shape[0])
        soft_obj = query_coords.new_zeros(num_queries)
        target_boxes = query_coords.new_zeros(num_queries, 5)
        target_labels = query_class_ids.clone()
        sigma = max(self.teacher_gaussian_sigma, 1e-4)

        def apply_sources(source_bboxes: Tensor, source_labels: Tensor,
                          source_scores: Tensor) -> None:
            if source_bboxes.numel() == 0:
                return
            source_scores_local = source_scores.to(
                device=query_coords.device, dtype=query_coords.dtype).clamp(
                    min=0.0, max=1.0)
            if self.teacher_score_power != 1.0:
                source_scores_local = source_scores_local.pow(
                    max(self.teacher_score_power, 0.0))
            for class_id in source_labels.unique():
                query_mask = query_class_ids == class_id
                source_mask = source_labels == class_id
                if not query_mask.any() or not source_mask.any():
                    continue
                query_idx = query_mask.nonzero().squeeze(1)
                source_idx = source_mask.nonzero().squeeze(1)
                dist = torch.cdist(
                    query_coords[query_idx, :2],
                    source_bboxes[source_idx, :2],
                    p=2)
                heatmap = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
                heatmap = heatmap * source_scores_local[source_idx].view(1, -1)
                class_soft, nearest = heatmap.max(dim=1)
                update = class_soft > soft_obj[query_idx]
                if update.any():
                    update_query = query_idx[update]
                    update_source = source_idx[nearest[update]]
                    soft_obj[update_query] = class_soft[update]
                    target_boxes[update_query] = source_bboxes[update_source]
                    target_labels[update_query] = source_labels[update_source]

        apply_sources(teacher_bboxes, teacher_labels, teacher_scores)
        gt_scores = gt_bboxes.new_ones(gt_bboxes.shape[0])
        apply_sources(gt_bboxes, gt_labels, gt_scores)
        return soft_obj.clamp(min=0.0, max=1.0), target_boxes, target_labels

    def _teacher_anchored_loss_single(
            self, cls_logits: Tensor, box_preds: Tensor, quality_logits: Tensor,
            query_coords: Tensor, query_class_ids: Tensor, gt_bboxes: Tensor,
            gt_labels: Tensor, teacher_bboxes: Tensor, teacher_labels: Tensor,
            teacher_scores: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor,
                                             Tensor]:
        zero = cls_logits.sum() * 0.0
        if (gt_bboxes.numel() == 0 and teacher_bboxes.numel() == 0
                or self.teacher_loss_weight <= 0):
            return zero, zero, zero, zero, zero

        soft_obj, target_boxes, _ = self._teacher_augmented_targets(
            query_coords, query_class_ids, gt_bboxes, gt_labels,
            teacher_bboxes, teacher_labels, teacher_scores)
        cls_target = torch.zeros_like(cls_logits)
        cls_target[
            torch.arange(cls_logits.shape[0], device=cls_logits.device),
            query_class_ids] = soft_obj
        teacher_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / max(
                int(gt_bboxes.shape[0] + teacher_bboxes.shape[0]), 1)
        teacher_quality = F.binary_cross_entropy_with_logits(
            quality_logits, soft_obj, reduction='mean')

        weights = soft_obj.detach()
        if self.teacher_min_weight > 0:
            weights = weights * (weights >= self.teacher_min_weight).to(
                dtype=weights.dtype)
        normalizer = weights.sum().clamp_min(1.0)
        bbox_l1 = (box_preds[:, :4] - target_boxes[:, :4]).abs().sum(dim=-1)
        angle_diff = torch.atan2(
            torch.sin(box_preds[:, 4] - target_boxes[:, 4]),
            torch.cos(box_preds[:, 4] - target_boxes[:, 4])).abs()
        teacher_bbox = (bbox_l1 * weights).sum() / normalizer
        teacher_angle = (angle_diff * weights).sum() / normalizer
        teacher_rank = self._assigned_rank_loss(
            cls_logits, box_preds, quality_logits, query_class_ids, soft_obj,
            weights)
        return (teacher_cls, teacher_quality, teacher_bbox, teacher_angle,
                teacher_rank)

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = super().loss(x, batch_data_samples)
        if self.teacher_loss_weight <= 0:
            return losses

        cls_logits, box_preds, quality_logits, seed_logits = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, query_coords, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)

        total_teacher_cls = cls_logits.sum() * 0.0
        total_teacher_quality = quality_logits.sum() * 0.0
        total_teacher_bbox = box_preds.sum() * 0.0
        total_teacher_angle = box_preds.sum() * 0.0
        total_teacher_rank = cls_logits.sum() * 0.0

        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            teacher_bboxes, teacher_labels, teacher_scores = (
                self._teacher_instances_for_sample(batch_data_samples[img_idx]))
            teacher = self._teacher_anchored_loss_single(
                cls_logits[img_idx],
                box_preds[img_idx],
                quality_logits[img_idx],
                query_coords[img_idx],
                query_class_ids,
                gt_bboxes,
                gt_labels,
                teacher_bboxes,
                teacher_labels,
                teacher_scores)
            total_teacher_cls = total_teacher_cls + teacher[0]
            total_teacher_quality = total_teacher_quality + teacher[1]
            total_teacher_bbox = total_teacher_bbox + teacher[2]
            total_teacher_angle = total_teacher_angle + teacher[3]
            total_teacher_rank = total_teacher_rank + teacher[4]

        batch_size = max(len(batch_gt_instances), 1)
        losses.update({
            'loss_teacher_cls':
            total_teacher_cls * self.teacher_loss_weight / batch_size,
            'loss_teacher_quality':
            total_teacher_quality * self.teacher_loss_weight / batch_size,
            'loss_teacher_bbox':
            total_teacher_bbox * self.teacher_loss_weight *
            self.aux_box_loss_weight / batch_size,
            'loss_teacher_angle':
            total_teacher_angle * self.teacher_loss_weight *
            self.aux_box_angle_loss_weight / batch_size,
            'loss_teacher_rank':
            total_teacher_rank * self.teacher_rank_loss_weight / batch_size,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        query_class_ids = self._stratified_query_class_ids_for_count(
            cls_logits.shape[1], cls_logits.device)
        query_arange = torch.arange(
            cls_logits.shape[1], device=cls_logits.device)
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            scores_per_class = self._posterior_scores(
                cls_logits[img_idx:img_idx + 1],
                box_preds[img_idx:img_idx + 1],
                quality_logits[img_idx:img_idx + 1])[0]
            scores = scores_per_class[query_arange, query_class_ids]
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores
            inst.labels = query_class_ids.clone()
            results.append(inst)
        self.e2e_debug.update({
            'p13_variant': 'P13N Teacher-Anchored Posterior Set Decoder',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_teacher_training_only': True,
            'uses_inference_fallback': False,
            'uses_score_threshold_postprocess': False,
            'uses_per_class_output_quota': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14RotatedDINOQueryTransportHead(P13EDenseSeedGaussianSetHead):
    """P14-A: strict set posterior with hard rotated box-query transport.

    This is the first minimal implementation of the literature-first rescue
    route.  It intentionally avoids P13N's soft Gaussian teacher field.  During
    training, GT boxes and optional teacher boxes are converted into hard
    one-to-one query transport targets.  Inference returns the fixed query set
    directly with no NMS, no score threshold filtering, and no teacher model.
    """

    def __init__(self,
                 *args,
                 teacher_predictions_path: str | None = None,
                 query_transport_loss_weight: float = 1.0,
                 query_transport_topk: int = 1,
                 teacher_score_thr: float = 0.20,
                 teacher_topk_per_img: int = 120,
                 teacher_score_power: float = 1.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_predictions_path = teacher_predictions_path
        self.query_transport_loss_weight = float(query_transport_loss_weight)
        self.query_transport_topk = max(int(query_transport_topk), 1)
        self.teacher_score_thr = float(teacher_score_thr)
        self.teacher_topk_per_img = int(teacher_topk_per_img)
        self.teacher_score_power = float(teacher_score_power)
        self._p14_teacher_by_img_id = self._load_transport_teacher_predictions(
            teacher_predictions_path)
        self.e2e_debug.update({
            'p14_variant': 'P14 Rotated DINO Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_hungarian_set_matching': True,
            'uses_dino_style_denoising': self.dn_loss_weight > 0,
            'uses_teacher_training_only': bool(teacher_predictions_path),
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })

    def _load_transport_teacher_predictions(self, path: str | None) -> dict:
        if not path or not os.path.exists(path):
            return {}
        with open(path, 'rb') as f:
            predictions = pickle.load(f)
        mapping = {}
        for item in predictions:
            if not isinstance(item, dict) or 'pred_instances' not in item:
                continue
            keys = []
            for key in ('img_id', 'img_path'):
                if key in item:
                    keys.extend(self._transport_key_variants(item[key]))
            for key in keys:
                mapping.setdefault(key, item['pred_instances'])
        return mapping

    def _transport_key_variants(self, value) -> list:
        if value is None:
            return []
        variants = [value, str(value)]
        if isinstance(value, str):
            base = os.path.basename(value)
            stem, _ = os.path.splitext(base)
            variants.extend([base, stem])
        return variants

    def _teacher_instances_for_transport(
            self, data_sample) -> tuple[Tensor, Tensor, Tensor]:
        device = self.class_bias.device
        empty_bboxes = torch.empty(0, 5, device=device)
        empty_labels = torch.empty(0, dtype=torch.long, device=device)
        empty_scores = torch.empty(0, device=device)
        if not self._p14_teacher_by_img_id:
            return empty_bboxes, empty_labels, empty_scores

        img_meta = data_sample.metainfo
        inst = None
        for meta_key in ('img_id', 'img_path'):
            if meta_key not in img_meta:
                continue
            for key in self._transport_key_variants(img_meta[meta_key]):
                if key in self._p14_teacher_by_img_id:
                    inst = self._p14_teacher_by_img_id[key]
                    break
            if inst is not None:
                break
        if inst is None or not hasattr(inst, 'bboxes'):
            return empty_bboxes, empty_labels, empty_scores

        bboxes = get_box_tensor(inst.bboxes).to(device=device).float()
        labels = inst.labels.to(device=device).long()
        scores = inst.scores.to(device=device).float()
        if bboxes.numel() == 0:
            return empty_bboxes, empty_labels, empty_scores

        keep = scores >= self.teacher_score_thr
        if not keep.any():
            return empty_bboxes, empty_labels, empty_scores
        bboxes = bboxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        if self.teacher_topk_per_img > 0 and scores.numel(
        ) > self.teacher_topk_per_img:
            top = scores.topk(self.teacher_topk_per_img).indices
            bboxes = bboxes[top]
            labels = labels[top]
            scores = scores[top]

        if self.teacher_score_power != 1.0:
            scores = scores.clamp(min=0.0, max=1.0).pow(
                max(self.teacher_score_power, 0.0))
        if bboxes[:, :4].max() > 2.0:
            img_h, img_w = img_meta['img_shape'][:2]
            bboxes = bboxes.clone()
            bboxes[:, 0] = bboxes[:, 0] / float(img_w)
            bboxes[:, 1] = bboxes[:, 1] / float(img_h)
            bboxes[:, 2] = bboxes[:, 2] / float(img_w)
            bboxes[:, 3] = bboxes[:, 3] / float(img_h)
        return bboxes, labels, scores

    def _box_query_transport_targets(
            self,
            query_boxes: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            teacher_bboxes: Tensor,
            teacher_labels: Tensor,
            teacher_scores: Tensor,
            query_class_ids: Tensor | None = None) -> dict[str, Tensor]:
        """Build hard query-to-box transport targets.

        The assignment is one-to-one over source boxes.  GT boxes get priority
        because they are concatenated before teacher boxes.  No Gaussian field
        is constructed here.
        """
        device = query_boxes.device
        dtype = query_boxes.dtype
        gt_bboxes = gt_bboxes.to(device=device, dtype=dtype)
        gt_labels = gt_labels.to(device=device, dtype=torch.long)
        teacher_bboxes = teacher_bboxes.to(device=device, dtype=dtype)
        teacher_labels = teacher_labels.to(device=device, dtype=torch.long)
        teacher_scores = teacher_scores.to(device=device, dtype=dtype)

        num_queries = int(query_boxes.shape[0])
        positive_mask = torch.zeros(num_queries, dtype=torch.bool, device=device)
        target_boxes = torch.zeros(num_queries, 5, dtype=dtype, device=device)
        target_labels = torch.zeros(num_queries, dtype=torch.long, device=device)
        target_scores = torch.zeros(num_queries, dtype=dtype, device=device)

        source_boxes = []
        source_labels = []
        source_scores = []
        if gt_bboxes.numel() > 0:
            source_boxes.append(gt_bboxes)
            source_labels.append(gt_labels)
            source_scores.append(gt_bboxes.new_ones(gt_bboxes.shape[0]))
        if teacher_bboxes.numel() > 0:
            source_boxes.append(teacher_bboxes)
            source_labels.append(teacher_labels)
            source_scores.append(teacher_scores.clamp(min=0.0, max=1.0))
        if not source_boxes or num_queries == 0:
            return dict(
                positive_mask=positive_mask,
                target_boxes=target_boxes,
                target_labels=target_labels,
                target_scores=target_scores)

        boxes = torch.cat(source_boxes, dim=0)
        labels = torch.cat(source_labels, dim=0)
        scores = torch.cat(source_scores, dim=0)
        if boxes.shape[0] > num_queries:
            keep = scores.topk(num_queries).indices
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

        center_cost = torch.cdist(query_boxes[:, :2], boxes[:, :2], p=1)
        size_cost = torch.cdist(query_boxes[:, 2:4], boxes[:, 2:4], p=1)
        angle_diff = torch.atan2(
            torch.sin(query_boxes[:, 4:5] - boxes[:, 4].view(1, -1)),
            torch.cos(query_boxes[:, 4:5] - boxes[:, 4].view(1, -1))).abs()
        cost = center_cost + 0.5 * size_cost + 0.25 * angle_diff
        if query_class_ids is not None:
            query_class_ids = query_class_ids.to(
                device=device, dtype=torch.long)
            mismatch = query_class_ids.view(-1, 1) != labels.view(1, -1)
            cost = cost + mismatch.to(dtype=dtype) * 1e6
        row, col = linear_sum_assignment(cost.detach().cpu())
        row = torch.as_tensor(row, dtype=torch.long, device=device)
        col = torch.as_tensor(col, dtype=torch.long, device=device)
        positive_mask[row] = True
        target_boxes[row] = boxes[col]
        target_labels[row] = labels[col]
        target_scores[row] = scores[col]
        return dict(
            positive_mask=positive_mask,
            target_boxes=target_boxes,
            target_labels=target_labels,
            target_scores=target_scores)

    def _query_transport_loss_single(
            self,
            cls_logits: Tensor,
            box_preds: Tensor,
            quality_logits: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            teacher_bboxes: Tensor,
            teacher_labels: Tensor,
            teacher_scores: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        zero = cls_logits.sum() * 0.0
        if self.query_transport_loss_weight <= 0:
            return zero, zero, zero, zero

        targets = self._box_query_transport_targets(
            box_preds.detach(), gt_bboxes, gt_labels, teacher_bboxes,
            teacher_labels, teacher_scores)
        positive = targets['positive_mask']
        cls_target = torch.zeros_like(cls_logits)
        quality_target = targets['target_scores'].to(dtype=quality_logits.dtype)
        if positive.any():
            cls_target[
                positive,
                targets['target_labels'][positive]] = targets[
                    'target_scores'][positive].to(dtype=cls_logits.dtype)
        normalizer = positive.sum().clamp_min(1).to(dtype=cls_logits.dtype)
        loss_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / normalizer
        loss_quality = F.binary_cross_entropy_with_logits(
            quality_logits, quality_target, reduction='sum') / max(
                int(cls_logits.shape[0]), 1)

        if not positive.any():
            return loss_cls, zero, zero, loss_quality
        loss_bbox = F.l1_loss(
            box_preds[positive, :4],
            targets['target_boxes'][positive, :4],
            reduction='sum') / normalizer
        angle_diff = torch.atan2(
            torch.sin(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4]),
            torch.cos(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4])).abs()
        loss_angle = angle_diff.sum() / normalizer
        return loss_cls, loss_bbox, loss_angle, loss_quality

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = super().loss(x, batch_data_samples)
        cls_logits, box_preds, quality_logits, _ = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)

        total_qt_cls = cls_logits.sum() * 0.0
        total_qt_bbox = box_preds.sum() * 0.0
        total_qt_angle = box_preds.sum() * 0.0
        total_qt_quality = quality_logits.sum() * 0.0
        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            teacher_bboxes, teacher_labels, teacher_scores = (
                self._teacher_instances_for_transport(
                    batch_data_samples[img_idx]))
            qt_losses = self._query_transport_loss_single(
                cls_logits[img_idx],
                box_preds[img_idx],
                quality_logits[img_idx],
                gt_bboxes,
                gt_labels,
                teacher_bboxes,
                teacher_labels,
                teacher_scores)
            total_qt_cls = total_qt_cls + qt_losses[0]
            total_qt_bbox = total_qt_bbox + qt_losses[1]
            total_qt_angle = total_qt_angle + qt_losses[2]
            total_qt_quality = total_qt_quality + qt_losses[3]

        batch_size = max(len(batch_gt_instances), 1)
        losses.update({
            'loss_query_transport_cls':
            total_qt_cls * self.query_transport_loss_weight / batch_size,
            'loss_query_transport_bbox':
            total_qt_bbox * self.query_transport_loss_weight *
            self.bbox_loss_weight / batch_size,
            'loss_query_transport_angle':
            total_qt_angle * self.query_transport_loss_weight *
            self.angle_loss_weight / batch_size,
            'loss_query_transport_quality':
            total_qt_quality * self.query_transport_loss_weight / batch_size,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            scores_per_class = cls_logits[img_idx].sigmoid()
            quality = quality_logits[img_idx].sigmoid().unsqueeze(-1)
            scores_per_class = scores_per_class * quality
            scores, labels = scores_per_class.max(dim=-1)
            keep = torch.arange(scores.numel(), device=scores.device)
            if keep.numel() > self.max_per_img:
                keep = scores.topk(self.max_per_img).indices
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, keep],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores[keep]
            inst.labels = labels[keep]
            results.append(inst)
        self.e2e_debug.update({
            'p14_variant': 'P14 Rotated DINO Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dino_style_denoising': self.dn_loss_weight > 0,
            'uses_teacher_training_only': bool(self.teacher_predictions_path),
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14BReferenceBoxQueryTransportHead(P14RotatedDINOQueryTransportHead):
    """P14-B: DAB-style reference-box query transport.

    P14-A still decoded width/height from FPN stride references, which is too
    small for HRRSD's large rotated objects.  P14-B keeps the strict fixed-set
    no-NMS contract but changes each query into an explicit reference box:
    class-level width/height priors initialize the posterior and the decoder
    only predicts residual corrections around those references.
    """

    def __init__(self,
                 *args,
                 class_wh_priors: Sequence[Sequence[float]] | None = None,
                 learn_reference_wh: bool = True,
                 reference_wh_scale: float = 1.0,
                 center_delta_scale: float = 0.40,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.learn_reference_wh = bool(learn_reference_wh)
        self.reference_wh_scale = float(reference_wh_scale)
        self.center_delta_scale = float(center_delta_scale)
        priors = self._build_reference_wh_priors(class_wh_priors)
        if self.learn_reference_wh:
            logits = priors.clamp(min=1e-4, max=1.0 - 1e-4).logit()
            self.reference_wh_logits = nn.Parameter(logits)
        else:
            self.register_buffer('reference_wh_priors', priors)
        self.e2e_debug.update({
            'p14_variant': 'P14B Reference-Box Query Transport',
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'learn_reference_wh': self.learn_reference_wh,
            'center_delta_scale': self.center_delta_scale,
        })

    def _build_reference_wh_priors(
            self, class_wh_priors: Sequence[Sequence[float]] | None) -> Tensor:
        if class_wh_priors is None:
            priors = torch.full((self.num_classes, 2), 0.20)
        else:
            priors = torch.as_tensor(class_wh_priors, dtype=torch.float32)
            if priors.ndim != 2 or priors.shape[1] != 2:
                raise ValueError('class_wh_priors must have shape [C, 2]')
            if priors.shape[0] < self.num_classes:
                repeat = math.ceil(self.num_classes / priors.shape[0])
                priors = priors.repeat(repeat, 1)
            priors = priors[:self.num_classes].clone()
            if bool((priors > 1.0).any()):
                image_w = float(self.image_size[0])
                image_h = float(self.image_size[1])
                priors[:, 0] = priors[:, 0] / image_w
                priors[:, 1] = priors[:, 1] / image_h
        return priors.clamp(min=1e-4, max=1.0 - 1e-4)

    def _reference_wh_for_queries(self, query_class_ids: Tensor | None,
                                  num_queries: int, device: torch.device,
                                  dtype: torch.dtype) -> Tensor:
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(num_queries, device)
        else:
            query_class_ids = query_class_ids.to(device=device, dtype=torch.long)
        if hasattr(self, 'reference_wh_logits'):
            priors = self.reference_wh_logits.sigmoid()
        else:
            priors = self.reference_wh_priors
        priors = priors.to(device=device, dtype=dtype)
        wh = priors[query_class_ids.clamp(min=0, max=self.num_classes - 1)]
        wh = wh * self.reference_wh_scale
        return wh.clamp(min=1e-4, max=1.0)

    def _decode_queries(self,
                        memory: Tensor,
                        query_memory: Tensor,
                        query_coords: Tensor,
                        query_strides: Tensor,
                        query_class_ids: Tensor | None = None,
                        base_wh_override: Tensor | None = None
                        ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = query_memory.shape[1]
        coord_embed = self.coord_proj(query_coords)
        query_embed = self._query_embed_slice(
            num_queries, query_memory.device).unsqueeze(0)
        query = query_memory + coord_embed + query_embed
        if query_class_ids is not None and self.query_class_prior_weight != 0:
            class_prior = self.support_tokens[query_class_ids].unsqueeze(0)
            query = query + self.query_class_prior_weight * class_prior
        decoded = self.decoder(query, memory)

        semantic = F.normalize(self.semantic_proj(decoded), dim=-1)
        support = F.normalize(self.support_tokens, dim=-1)
        cls_logits = semantic @ support.t()
        cls_logits = cls_logits + self.class_bias.view(1, 1, -1)
        if query_class_ids is not None and self.own_class_logit_bias != 0:
            class_bias = F.one_hot(
                query_class_ids,
                num_classes=self.num_classes).to(dtype=cls_logits.dtype)
            cls_logits = cls_logits + self.own_class_logit_bias * class_bias
        cls_logits = cls_logits * self.logit_scale.exp().clamp(1e-4, 100.0)

        raw_box = self.box_delta(decoded)
        if base_wh_override is None:
            base_wh = self._reference_wh_for_queries(
                query_class_ids,
                num_queries,
                query_memory.device,
                query_memory.dtype).unsqueeze(0)
        else:
            base_wh = base_wh_override
        center = (
            query_coords[..., :2] +
            self.center_delta_scale * raw_box[..., :2].tanh()).clamp(
                min=1e-4, max=1.0 - 1e-4)
        wh = (base_wh * torch.exp(
            self.box_delta_scale * raw_box[..., 2:4].tanh())).clamp(
            min=1e-4, max=1.0)
        angle = raw_box[..., 4:5].tanh() * (math.pi / 2.0)
        box_preds = torch.cat([center, wh, angle], dim=-1)
        quality_logits = self.quality(decoded).squeeze(-1)
        return cls_logits, box_preds, quality_logits

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14B Reference-Box Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14CClassConsistentQueryTransportHead(
        P14BReferenceBoxQueryTransportHead):
    """P14-C: class-consistent transport with same-class posterior ranking."""

    def __init__(self,
                 *args,
                 transport_rank_loss_weight: float = 0.5,
                 ranking_margin: float = 0.10,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transport_rank_loss_weight = float(transport_rank_loss_weight)
        self.ranking_margin = float(ranking_margin)
        self.e2e_debug.update({
            'p14_variant': 'P14C Class-Consistent Query Transport',
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
        })

    def _query_transport_loss_and_targets(
            self,
            cls_logits: Tensor,
            box_preds: Tensor,
            quality_logits: Tensor,
            query_class_ids: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            teacher_bboxes: Tensor,
            teacher_labels: Tensor,
            teacher_scores: Tensor
            ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        zero = cls_logits.sum() * 0.0
        targets = self._box_query_transport_targets(
            box_preds.detach(), gt_bboxes, gt_labels, teacher_bboxes,
            teacher_labels, teacher_scores, query_class_ids=query_class_ids)
        if self.query_transport_loss_weight <= 0:
            return zero, zero, zero, zero, targets

        positive = targets['positive_mask']
        cls_target = torch.zeros_like(cls_logits)
        quality_target = targets['target_scores'].to(dtype=quality_logits.dtype)
        if positive.any():
            cls_target[
                positive,
                targets['target_labels'][positive]] = targets[
                    'target_scores'][positive].to(dtype=cls_logits.dtype)
        normalizer = positive.sum().clamp_min(1).to(dtype=cls_logits.dtype)
        loss_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / normalizer
        loss_quality = F.binary_cross_entropy_with_logits(
            quality_logits, quality_target, reduction='sum') / max(
                int(cls_logits.shape[0]), 1)

        if not positive.any():
            return loss_cls, zero, zero, loss_quality, targets
        loss_bbox = F.l1_loss(
            box_preds[positive, :4],
            targets['target_boxes'][positive, :4],
            reduction='sum') / normalizer
        angle_diff = torch.atan2(
            torch.sin(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4]),
            torch.cos(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4])).abs()
        loss_angle = angle_diff.sum() / normalizer
        return loss_cls, loss_bbox, loss_angle, loss_quality, targets

    def _transport_rank_loss(self, cls_logits: Tensor,
                             quality_logits: Tensor,
                             query_class_ids: Tensor,
                             targets: dict[str, Tensor]) -> Tensor:
        zero = cls_logits.sum() * 0.0
        if self.transport_rank_loss_weight <= 0:
            return zero
        positive = targets['positive_mask']
        if not positive.any():
            return zero
        posterior = cls_logits.sigmoid() * quality_logits.sigmoid().unsqueeze(-1)
        own_scores = posterior[
            torch.arange(cls_logits.shape[0], device=cls_logits.device),
            query_class_ids]
        rank_terms = []
        for class_id in query_class_ids.unique():
            same_class = query_class_ids == class_id
            pos = positive & same_class & (
                targets['target_labels'] == class_id)
            neg = (~positive) & same_class
            if not pos.any() or not neg.any():
                continue
            pos_scores = own_scores[pos]
            neg_scores = own_scores[neg]
            hard_neg = neg_scores.topk(
                min(int(neg_scores.numel()),
                    max(int(pos_scores.numel()) * 4, 1))).values
            rank_terms.append(
                F.relu(self.ranking_margin - pos_scores[:, None] +
                       hard_neg[None, :]).mean())
        if not rank_terms:
            return zero
        return torch.stack(rank_terms).mean()

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = P13EDenseSeedGaussianSetHead.loss(
            self, x, batch_data_samples)
        cls_logits, box_preds, quality_logits, _ = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, _, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)
        else:
            query_class_ids = query_class_ids.to(device=cls_logits.device)

        total_qt_cls = cls_logits.sum() * 0.0
        total_qt_bbox = box_preds.sum() * 0.0
        total_qt_angle = box_preds.sum() * 0.0
        total_qt_quality = quality_logits.sum() * 0.0
        total_qt_rank = cls_logits.sum() * 0.0
        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            teacher_bboxes, teacher_labels, teacher_scores = (
                self._teacher_instances_for_transport(
                    batch_data_samples[img_idx]))
            qt = self._query_transport_loss_and_targets(
                cls_logits[img_idx],
                box_preds[img_idx],
                quality_logits[img_idx],
                query_class_ids,
                gt_bboxes,
                gt_labels,
                teacher_bboxes,
                teacher_labels,
                teacher_scores)
            total_qt_cls = total_qt_cls + qt[0]
            total_qt_bbox = total_qt_bbox + qt[1]
            total_qt_angle = total_qt_angle + qt[2]
            total_qt_quality = total_qt_quality + qt[3]
            total_qt_rank = total_qt_rank + self._transport_rank_loss(
                cls_logits[img_idx],
                quality_logits[img_idx],
                query_class_ids,
                qt[4])

        batch_size = max(len(batch_gt_instances), 1)
        losses.update({
            'loss_query_transport_cls':
            total_qt_cls * self.query_transport_loss_weight / batch_size,
            'loss_query_transport_bbox':
            total_qt_bbox * self.query_transport_loss_weight *
            self.bbox_loss_weight / batch_size,
            'loss_query_transport_angle':
            total_qt_angle * self.query_transport_loss_weight *
            self.angle_loss_weight / batch_size,
            'loss_query_transport_quality':
            total_qt_quality * self.query_transport_loss_weight / batch_size,
            'loss_query_transport_rank':
            total_qt_rank * self.transport_rank_loss_weight / batch_size,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        query_class_ids = self._query_class_ids(
            cls_logits.shape[1], cls_logits.device)
        query_arange = torch.arange(
            cls_logits.shape[1], device=cls_logits.device)
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            posterior = cls_logits[img_idx].sigmoid()
            quality = quality_logits[img_idx].sigmoid()
            scores = posterior[query_arange, query_class_ids] * quality
            keep = query_arange
            if keep.numel() > self.max_per_img:
                keep = scores.topk(self.max_per_img).indices
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, keep],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores[keep]
            inst.labels = query_class_ids[keep].clone()
            results.append(inst)
        self.e2e_debug.update({
            'p14_variant': 'P14C Class-Consistent Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14DStratifiedClassQueryTransportHead(
        P14CClassConsistentQueryTransportHead):
    """P14-D: class-stratified spatial query coverage for P14-C."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e2e_debug.update({
            'p14_variant': 'P14D Stratified Class Query Transport',
            'uses_stratified_class_queries': True,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
        })

    def _query_class_ids(self, num_queries: int,
                         device: torch.device) -> Tensor:
        per_class = max(1, math.ceil(num_queries / self.num_classes))
        class_ids = torch.arange(
            self.num_classes, device=device).repeat_interleave(per_class)
        if class_ids.numel() < num_queries:
            repeat = math.ceil(num_queries / max(class_ids.numel(), 1))
            class_ids = class_ids.repeat(repeat)
        return class_ids[:num_queries]

    def _select_query_seeds(self, memory: Tensor, coords: Tensor,
                            strides: Tensor, seed_logits: Tensor
                            ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        batch_size, num_tokens, channels = memory.shape
        num_queries = min(self.num_queries, num_tokens * self.num_classes)
        per_class = max(1, math.ceil(num_queries / self.num_classes))
        per_class = min(per_class, num_tokens)
        _, base_inds = seed_logits.topk(per_class, dim=1)

        repeat = math.ceil(num_queries / per_class)
        query_inds = base_inds.repeat(1, repeat)[:, :num_queries]
        gather_idx = query_inds.unsqueeze(-1).expand(-1, -1, channels)
        seed_memory = memory.gather(1, gather_idx)
        seed_coords = coords[query_inds]
        seed_strides = strides[query_inds]
        query_class_ids = self._query_class_ids(num_queries, memory.device)
        return seed_memory, seed_coords, seed_strides, query_class_ids

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14D Stratified Class Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_stratified_class_queries': True,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14EQualityCalibratedQueryTransportHead(
        P14CClassConsistentQueryTransportHead):
    """P14-E: class-consistent transport with geometry quality calibration."""

    def __init__(self,
                 *args,
                 geometry_quality_loss_weight: float = 1.0,
                 geometry_quality_cost_scale: float = 0.75,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.geometry_quality_loss_weight = float(
            geometry_quality_loss_weight)
        self.geometry_quality_cost_scale = max(
            float(geometry_quality_cost_scale), 1e-4)
        self.e2e_debug.update({
            'p14_variant': 'P14E Quality-Calibrated Query Transport',
            'uses_geometry_quality_calibration':
            self.geometry_quality_loss_weight > 0,
            'geometry_quality_cost_scale': self.geometry_quality_cost_scale,
        })

    def _geometry_quality_targets(
            self,
            query_boxes: Tensor,
            query_class_ids: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            teacher_bboxes: Tensor,
            teacher_labels: Tensor,
            teacher_scores: Tensor) -> Tensor:
        device = query_boxes.device
        dtype = query_boxes.dtype
        query_class_ids = query_class_ids.to(device=device, dtype=torch.long)
        gt_bboxes = gt_bboxes.to(device=device, dtype=dtype)
        gt_labels = gt_labels.to(device=device, dtype=torch.long)
        teacher_bboxes = teacher_bboxes.to(device=device, dtype=dtype)
        teacher_labels = teacher_labels.to(device=device, dtype=torch.long)
        teacher_scores = teacher_scores.to(device=device, dtype=dtype)

        source_boxes = []
        source_labels = []
        source_scores = []
        if gt_bboxes.numel() > 0:
            source_boxes.append(gt_bboxes)
            source_labels.append(gt_labels)
            source_scores.append(gt_bboxes.new_ones(gt_bboxes.shape[0]))
        if teacher_bboxes.numel() > 0:
            source_boxes.append(teacher_bboxes)
            source_labels.append(teacher_labels)
            source_scores.append(teacher_scores.clamp(min=0.0, max=1.0))
        quality = query_boxes.new_zeros(query_boxes.shape[0])
        if not source_boxes or query_boxes.numel() == 0:
            return quality

        boxes = torch.cat(source_boxes, dim=0)
        labels = torch.cat(source_labels, dim=0)
        scores = torch.cat(source_scores, dim=0)
        center_cost = torch.cdist(query_boxes[:, :2], boxes[:, :2], p=1)
        size_cost = torch.cdist(query_boxes[:, 2:4], boxes[:, 2:4], p=1)
        angle_cost = torch.atan2(
            torch.sin(query_boxes[:, 4:5] - boxes[:, 4].view(1, -1)),
            torch.cos(query_boxes[:, 4:5] - boxes[:, 4].view(1, -1))).abs()
        cost = center_cost + 0.5 * size_cost + 0.25 * angle_cost / (
            math.pi / 2.0)
        same_class = query_class_ids.view(-1, 1) == labels.view(1, -1)
        raw_quality = (1.0 - cost / self.geometry_quality_cost_scale).clamp(
            min=0.0, max=1.0) * scores.view(1, -1)
        raw_quality = raw_quality * same_class.to(dtype=dtype)
        if raw_quality.numel() > 0:
            quality = raw_quality.max(dim=1).values
        return quality.clamp(min=0.0, max=1.0)

    def _geometry_quality_loss(self, quality_logits: Tensor,
                               quality_targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            quality_logits,
            quality_targets.to(dtype=quality_logits.dtype),
            reduction='mean')

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = super().loss(x, batch_data_samples)
        if self.geometry_quality_loss_weight <= 0:
            return losses

        cls_logits, box_preds, quality_logits, _ = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, _, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)
        else:
            query_class_ids = query_class_ids.to(device=cls_logits.device)

        total_geo_quality = quality_logits.sum() * 0.0
        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            teacher_bboxes, teacher_labels, teacher_scores = (
                self._teacher_instances_for_transport(
                    batch_data_samples[img_idx]))
            quality_targets = self._geometry_quality_targets(
                box_preds[img_idx].detach(),
                query_class_ids,
                gt_bboxes,
                gt_labels,
                teacher_bboxes,
                teacher_labels,
                teacher_scores)
            total_geo_quality = total_geo_quality + self._geometry_quality_loss(
                quality_logits[img_idx], quality_targets)

        batch_size = max(len(batch_gt_instances), 1)
        losses.update({
            'loss_query_transport_geo_quality':
            total_geo_quality * self.geometry_quality_loss_weight / batch_size,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14E Quality-Calibrated Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_geometry_quality_calibration':
            self.geometry_quality_loss_weight > 0,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14FDetachedQualityQueryTransportHead(
        P14EQualityCalibratedQueryTransportHead):
    """P14-F: train quality calibration without moving query geometry."""

    def __init__(self, *args, detach_quality_features: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.detach_quality_features = bool(detach_quality_features)
        self.e2e_debug.update({
            'p14_variant': 'P14F Detached-Quality Query Transport',
            'uses_detached_quality_features': self.detach_quality_features,
        })

    def _decode_queries(self,
                        memory: Tensor,
                        query_memory: Tensor,
                        query_coords: Tensor,
                        query_strides: Tensor,
                        query_class_ids: Tensor | None = None,
                        base_wh_override: Tensor | None = None
                        ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = query_memory.shape[1]
        coord_embed = self.coord_proj(query_coords)
        query_embed = self._query_embed_slice(
            num_queries, query_memory.device).unsqueeze(0)
        query = query_memory + coord_embed + query_embed
        if query_class_ids is not None and self.query_class_prior_weight != 0:
            class_prior = self.support_tokens[query_class_ids].unsqueeze(0)
            query = query + self.query_class_prior_weight * class_prior
        decoded = self.decoder(query, memory)

        semantic = F.normalize(self.semantic_proj(decoded), dim=-1)
        support = F.normalize(self.support_tokens, dim=-1)
        cls_logits = semantic @ support.t()
        cls_logits = cls_logits + self.class_bias.view(1, 1, -1)
        if query_class_ids is not None and self.own_class_logit_bias != 0:
            class_bias = F.one_hot(
                query_class_ids,
                num_classes=self.num_classes).to(dtype=cls_logits.dtype)
            cls_logits = cls_logits + self.own_class_logit_bias * class_bias
        cls_logits = cls_logits * self.logit_scale.exp().clamp(1e-4, 100.0)

        raw_box = self.box_delta(decoded)
        if base_wh_override is None:
            base_wh = self._reference_wh_for_queries(
                query_class_ids,
                num_queries,
                query_memory.device,
                query_memory.dtype).unsqueeze(0)
        else:
            base_wh = base_wh_override
        center = (
            query_coords[..., :2] +
            self.center_delta_scale * raw_box[..., :2].tanh()).clamp(
                min=1e-4, max=1.0 - 1e-4)
        wh = (base_wh * torch.exp(
            self.box_delta_scale * raw_box[..., 2:4].tanh())).clamp(
            min=1e-4, max=1.0)
        angle = raw_box[..., 4:5].tanh() * (math.pi / 2.0)
        box_preds = torch.cat([center, wh, angle], dim=-1)
        quality_features = decoded.detach(
        ) if self.detach_quality_features else decoded
        quality_logits = self.quality(quality_features).squeeze(-1)
        return cls_logits, box_preds, quality_logits

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14F Detached-Quality Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_geometry_quality_calibration':
            self.geometry_quality_loss_weight > 0,
            'uses_detached_quality_features': self.detach_quality_features,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14GGaussianSeedObjectnessQueryTransportHead(
        P14CClassConsistentQueryTransportHead):
    """P14-G: fuse Gaussian encoder objectness into fixed-query posterior.

    P14-C used encoder seed logits only to choose query locations.  DINO-style
    two-stage query selection keeps that encoder objectness as part of the
    object posterior, so this variant trains seed logits with a Gaussian center
    heatmap and multiplies the class-locked query posterior by the selected
    seed object's probability.  It remains a fixed set predictor: no dense
    detection head, no NMS, and no score-threshold fallback.
    """

    def __init__(self,
                 *args,
                 seed_gaussian_sigma: float = 0.045,
                 seed_score_power: float = 0.50,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed_gaussian_sigma = max(float(seed_gaussian_sigma), 1e-4)
        self.seed_score_power = max(float(seed_score_power), 0.0)
        self.e2e_debug.update({
            'p14_variant': 'P14G Gaussian Seed Objectness Query Transport',
            'uses_gaussian_seed_heatmap': True,
            'uses_encoder_seed_objectness_posterior': True,
            'seed_score_power': self.seed_score_power,
        })

    def _gaussian_seed_targets(self, coords: Tensor,
                               gt_bboxes: Tensor) -> Tensor:
        target = coords.new_zeros(coords.shape[0])
        if gt_bboxes.numel() == 0 or coords.numel() == 0:
            return target
        dist = torch.cdist(coords[:, :2], gt_bboxes[:, :2], p=2)
        heatmap = torch.exp(-0.5 * (dist / self.seed_gaussian_sigma).pow(2))
        return heatmap.max(dim=1).values.clamp(min=0.0, max=1.0)

    def _seed_loss_single(self, seed_logits: Tensor, gt_bboxes: Tensor,
                          coords: Tensor) -> Tensor:
        target = self._gaussian_seed_targets(coords, gt_bboxes)
        return py_sigmoid_focal_loss(
            seed_logits,
            target,
            alpha=0.25,
            gamma=2.0,
            reduction='mean')

    def _select_query_seeds_with_scores(
            self, memory: Tensor, coords: Tensor, strides: Tensor,
            seed_logits: Tensor
            ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_tokens, channels = memory.shape
        num_queries = min(self.num_queries, num_tokens)
        _, topk_inds = seed_logits.topk(num_queries, dim=1)
        gather_idx = topk_inds.unsqueeze(-1).expand(-1, -1, channels)
        seed_memory = memory.gather(1, gather_idx)
        seed_coords = coords[topk_inds]
        seed_strides = strides[topk_inds]
        selected_seed_logits = seed_logits.gather(1, topk_inds)
        query_class_ids = self._query_class_ids(num_queries, memory.device)
        return (seed_memory, seed_coords, seed_strides, query_class_ids,
                selected_seed_logits)

    def _select_query_seeds(self, memory: Tensor, coords: Tensor,
                            strides: Tensor, seed_logits: Tensor
                            ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        seed_memory, seed_coords, seed_strides, query_class_ids, _ = (
            self._select_query_seeds_with_scores(
                memory, coords, strides, seed_logits))
        return seed_memory, seed_coords, seed_strides, query_class_ids

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, _, _, query_class_ids, selected_seed_logits = (
            self._select_query_seeds_with_scores(
                memory, coords, strides, flat_seed_logits))
        query_class_ids = query_class_ids.to(device=cls_logits.device)
        query_arange = torch.arange(
            cls_logits.shape[1], device=cls_logits.device)
        selected_seed_logits = selected_seed_logits.to(device=cls_logits.device)

        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            posterior = cls_logits[img_idx].sigmoid()
            quality = quality_logits[img_idx].sigmoid()
            seed_objectness = selected_seed_logits[img_idx].sigmoid().clamp(
                min=1e-4, max=1.0).pow(self.seed_score_power)
            scores = (
                posterior[query_arange, query_class_ids] * quality *
                seed_objectness)
            keep = query_arange
            if keep.numel() > self.max_per_img:
                keep = scores.topk(self.max_per_img).indices
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, keep],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores[keep]
            inst.labels = query_class_ids[keep].clone()
            results.append(inst)

        self.e2e_debug.update({
            'p14_variant': 'P14G Gaussian Seed Objectness Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_gaussian_seed_heatmap': True,
            'uses_encoder_seed_objectness_posterior': True,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14HDecoupledQueryTransportHead(P14CClassConsistentQueryTransportHead):
    """P14-H: D2Q-style decoupled cls/reg query features for P14-C."""

    def __init__(self, *args, decouple_init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decouple_init_std = float(decouple_init_std)
        self.cls_query_proj = nn.Linear(self.feat_channels, self.feat_channels)
        self.reg_query_proj = nn.Linear(self.feat_channels, self.feat_channels)
        self.e2e_debug.update({
            'p14_variant': 'P14H Decoupled Query Transport',
            'uses_decoupled_query_features': True,
            'quality_branch_uses_cls_features': True,
            'box_branch_uses_reg_features': True,
        })

    def init_weights(self) -> None:
        super().init_weights()
        normal_init(self.cls_query_proj, mean=0.0, std=self.decouple_init_std)
        normal_init(self.reg_query_proj, mean=0.0, std=self.decouple_init_std)

    def _decode_queries(self,
                        memory: Tensor,
                        query_memory: Tensor,
                        query_coords: Tensor,
                        query_strides: Tensor,
                        query_class_ids: Tensor | None = None,
                        base_wh_override: Tensor | None = None
                        ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = query_memory.shape[1]
        coord_embed = self.coord_proj(query_coords)
        query_embed = self._query_embed_slice(
            num_queries, query_memory.device).unsqueeze(0)
        query = query_memory + coord_embed + query_embed
        if query_class_ids is not None and self.query_class_prior_weight != 0:
            class_prior = self.support_tokens[query_class_ids].unsqueeze(0)
            query = query + self.query_class_prior_weight * class_prior
        decoded = self.decoder(query, memory)

        cls_features = decoded + self.cls_query_proj(decoded)
        reg_features = decoded + self.reg_query_proj(decoded)

        semantic = F.normalize(self.semantic_proj(cls_features), dim=-1)
        support = F.normalize(self.support_tokens, dim=-1)
        cls_logits = semantic @ support.t()
        cls_logits = cls_logits + self.class_bias.view(1, 1, -1)
        if query_class_ids is not None and self.own_class_logit_bias != 0:
            class_bias = F.one_hot(
                query_class_ids,
                num_classes=self.num_classes).to(dtype=cls_logits.dtype)
            cls_logits = cls_logits + self.own_class_logit_bias * class_bias
        cls_logits = cls_logits * self.logit_scale.exp().clamp(1e-4, 100.0)

        raw_box = self.box_delta(reg_features)
        if base_wh_override is None:
            base_wh = self._reference_wh_for_queries(
                query_class_ids,
                num_queries,
                query_memory.device,
                query_memory.dtype).unsqueeze(0)
        else:
            base_wh = base_wh_override
        center = (
            query_coords[..., :2] +
            self.center_delta_scale * raw_box[..., :2].tanh()).clamp(
                min=1e-4, max=1.0 - 1e-4)
        wh = (base_wh * torch.exp(
            self.box_delta_scale * raw_box[..., 2:4].tanh())).clamp(
            min=1e-4, max=1.0)
        angle = raw_box[..., 4:5].tanh() * (math.pi / 2.0)
        box_preds = torch.cat([center, wh, angle], dim=-1)
        quality_logits = self.quality(cls_features).squeeze(-1)
        return cls_logits, box_preds, quality_logits

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14H Decoupled Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_decoupled_query_features': True,
            'quality_branch_uses_cls_features': True,
            'box_branch_uses_reg_features': True,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14IGaussianReferenceQueryTransportHead(
        P14CClassConsistentQueryTransportHead):
    """P14-I: inject rotated-box Gaussian geometry into query features."""

    def __init__(self, *args, gaussian_geometry_weight: float = 0.50,
                 gaussian_geometry_init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gaussian_geometry_weight = float(gaussian_geometry_weight)
        self.gaussian_geometry_init_std = float(gaussian_geometry_init_std)
        self.gaussian_geometry_proj = nn.Linear(6, self.feat_channels)
        self.e2e_debug.update({
            'p14_variant': 'P14I Gaussian Reference Query Transport',
            'uses_gaussian_reference_geometry': True,
            'gaussian_geometry_weight': self.gaussian_geometry_weight,
        })

    def init_weights(self) -> None:
        super().init_weights()
        normal_init(
            self.gaussian_geometry_proj,
            mean=0.0,
            std=self.gaussian_geometry_init_std)

    def _gaussian_reference_features(self, ref_boxes: Tensor) -> Tensor:
        cx = ref_boxes[..., 0]
        cy = ref_boxes[..., 1]
        half_w = ref_boxes[..., 2].clamp(min=1e-4) * 0.5
        half_h = ref_boxes[..., 3].clamp(min=1e-4) * 0.5
        angle = ref_boxes[..., 4]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        var_w = half_w.pow(2)
        var_h = half_h.pow(2)
        cov_xx = cos_a.pow(2) * var_w + sin_a.pow(2) * var_h
        cov_yy = sin_a.pow(2) * var_w + cos_a.pow(2) * var_h
        cov_xy = sin_a * cos_a * (var_w - var_h)
        area_scale = (var_w * var_h).sqrt()
        return torch.stack([cx, cy, cov_xx, cov_yy, cov_xy, area_scale], dim=-1)

    def _reference_boxes_for_queries(
            self,
            query_coords: Tensor,
            query_class_ids: Tensor | None,
            num_queries: int,
            device: torch.device,
            dtype: torch.dtype,
            base_wh_override: Tensor | None = None) -> Tensor:
        if base_wh_override is None:
            base_wh = self._reference_wh_for_queries(
                query_class_ids, num_queries, device, dtype)
            while base_wh.ndim < query_coords.ndim:
                base_wh = base_wh.unsqueeze(0)
        else:
            base_wh = base_wh_override
        if base_wh.shape[:-1] != query_coords.shape[:-1]:
            base_wh = base_wh.expand(query_coords.shape[:-1] + (2, ))
        angle = query_coords.new_zeros(query_coords.shape[:-1] + (1, ))
        return torch.cat([query_coords[..., :2], base_wh, angle], dim=-1)

    def _decode_queries(self,
                        memory: Tensor,
                        query_memory: Tensor,
                        query_coords: Tensor,
                        query_strides: Tensor,
                        query_class_ids: Tensor | None = None,
                        base_wh_override: Tensor | None = None
                        ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = query_memory.shape[1]
        ref_boxes = self._reference_boxes_for_queries(
            query_coords,
            query_class_ids,
            num_queries,
            query_memory.device,
            query_memory.dtype,
            base_wh_override=base_wh_override)
        gaussian_embed = self.gaussian_geometry_proj(
            self._gaussian_reference_features(ref_boxes))

        coord_embed = self.coord_proj(query_coords)
        query_embed = self._query_embed_slice(
            num_queries, query_memory.device).unsqueeze(0)
        query = (
            query_memory + coord_embed + query_embed +
            self.gaussian_geometry_weight * gaussian_embed)
        if query_class_ids is not None and self.query_class_prior_weight != 0:
            class_prior = self.support_tokens[query_class_ids].unsqueeze(0)
            query = query + self.query_class_prior_weight * class_prior
        decoded = self.decoder(query, memory)

        semantic = F.normalize(self.semantic_proj(decoded), dim=-1)
        support = F.normalize(self.support_tokens, dim=-1)
        cls_logits = semantic @ support.t()
        cls_logits = cls_logits + self.class_bias.view(1, 1, -1)
        if query_class_ids is not None and self.own_class_logit_bias != 0:
            class_bias = F.one_hot(
                query_class_ids,
                num_classes=self.num_classes).to(dtype=cls_logits.dtype)
            cls_logits = cls_logits + self.own_class_logit_bias * class_bias
        cls_logits = cls_logits * self.logit_scale.exp().clamp(1e-4, 100.0)

        raw_box = self.box_delta(decoded)
        base_wh = ref_boxes[..., 2:4]
        center = (
            query_coords[..., :2] +
            self.center_delta_scale * raw_box[..., :2].tanh()).clamp(
                min=1e-4, max=1.0 - 1e-4)
        wh = (base_wh * torch.exp(
            self.box_delta_scale * raw_box[..., 2:4].tanh())).clamp(
            min=1e-4, max=1.0)
        angle = raw_box[..., 4:5].tanh() * (math.pi / 2.0)
        box_preds = torch.cat([center, wh, angle], dim=-1)
        quality_logits = self.quality(decoded).squeeze(-1)
        return cls_logits, box_preds, quality_logits

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14I Gaussian Reference Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_gaussian_reference_geometry': True,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14JOneToManyQueryTransportHead(P14CClassConsistentQueryTransportHead):
    """P14-J: add one-to-many auxiliary positives for class-consistent queries."""

    def __init__(self,
                 *args,
                 aux_otm_loss_weight: float = 0.50,
                 aux_otm_topk_per_gt: int = 2,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aux_otm_loss_weight = float(aux_otm_loss_weight)
        self.aux_otm_topk_per_gt = max(int(aux_otm_topk_per_gt), 1)
        self.e2e_debug.update({
            'p14_variant': 'P14J One-to-Many Query Transport',
            'uses_one_to_many_auxiliary_transport':
            self.aux_otm_loss_weight > 0,
            'aux_otm_topk_per_gt': self.aux_otm_topk_per_gt,
        })

    def _one_to_many_transport_targets(
            self, query_boxes: Tensor, query_class_ids: Tensor,
            gt_bboxes: Tensor, gt_labels: Tensor) -> dict[str, Tensor]:
        device = query_boxes.device
        dtype = query_boxes.dtype
        query_class_ids = query_class_ids.to(device=device, dtype=torch.long)
        gt_bboxes = gt_bboxes.to(device=device, dtype=dtype)
        gt_labels = gt_labels.to(device=device, dtype=torch.long)
        positive_mask = torch.zeros(
            query_boxes.shape[0], dtype=torch.bool, device=device)
        target_boxes = torch.zeros_like(query_boxes)
        target_labels = torch.zeros(
            query_boxes.shape[0], dtype=torch.long, device=device)
        best_cost = torch.full(
            (query_boxes.shape[0], ), float('inf'), dtype=dtype, device=device)
        if query_boxes.numel() == 0 or gt_bboxes.numel() == 0:
            return dict(
                positive_mask=positive_mask,
                target_boxes=target_boxes,
                target_labels=target_labels)

        for gt_idx in range(gt_bboxes.shape[0]):
            same_class = query_class_ids == gt_labels[gt_idx]
            if not same_class.any():
                continue
            query_idx = torch.nonzero(same_class, as_tuple=False).flatten()
            boxes = query_boxes[query_idx]
            center_cost = torch.cdist(
                boxes[:, :2], gt_bboxes[gt_idx:gt_idx + 1, :2], p=1).squeeze(1)
            size_cost = torch.cdist(
                boxes[:, 2:4], gt_bboxes[gt_idx:gt_idx + 1, 2:4],
                p=1).squeeze(1)
            angle_cost = torch.atan2(
                torch.sin(boxes[:, 4] - gt_bboxes[gt_idx, 4]),
                torch.cos(boxes[:, 4] - gt_bboxes[gt_idx, 4])).abs()
            cost = center_cost + 0.5 * size_cost + 0.25 * angle_cost
            topk = min(self.aux_otm_topk_per_gt, int(cost.numel()))
            selected = query_idx[cost.topk(topk, largest=False).indices]
            selected_cost = cost[cost.topk(topk, largest=False).indices]
            improve = selected_cost < best_cost[selected]
            selected = selected[improve]
            selected_cost = selected_cost[improve]
            if selected.numel() == 0:
                continue
            positive_mask[selected] = True
            target_boxes[selected] = gt_bboxes[gt_idx]
            target_labels[selected] = gt_labels[gt_idx]
            best_cost[selected] = selected_cost
        return dict(
            positive_mask=positive_mask,
            target_boxes=target_boxes,
            target_labels=target_labels)

    def _one_to_many_transport_loss_single(
            self, cls_logits: Tensor, box_preds: Tensor,
            quality_logits: Tensor, query_class_ids: Tensor, gt_bboxes: Tensor,
            gt_labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        zero = cls_logits.sum() * 0.0
        if self.aux_otm_loss_weight <= 0:
            return zero, zero, zero, zero
        targets = self._one_to_many_transport_targets(
            box_preds.detach(), query_class_ids, gt_bboxes, gt_labels)
        positive = targets['positive_mask']
        cls_target = torch.zeros_like(cls_logits)
        quality_target = torch.zeros_like(quality_logits)
        if positive.any():
            cls_target[
                positive,
                targets['target_labels'][positive]] = 1.0
            quality_target[positive] = 1.0
        normalizer = positive.sum().clamp_min(1).to(dtype=cls_logits.dtype)
        loss_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / normalizer
        loss_quality = F.binary_cross_entropy_with_logits(
            quality_logits, quality_target, reduction='sum') / max(
                int(quality_logits.shape[0]), 1)
        if not positive.any():
            return loss_cls, zero, zero, loss_quality
        loss_bbox = F.l1_loss(
            box_preds[positive, :4],
            targets['target_boxes'][positive, :4],
            reduction='sum') / normalizer
        angle_diff = torch.atan2(
            torch.sin(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4]),
            torch.cos(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4])).abs()
        loss_angle = angle_diff.sum() / normalizer
        return loss_cls, loss_bbox, loss_angle, loss_quality

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = super().loss(x, batch_data_samples)
        if self.aux_otm_loss_weight <= 0:
            return losses

        cls_logits, box_preds, quality_logits, _ = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, _, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)
        else:
            query_class_ids = query_class_ids.to(device=cls_logits.device)

        total_cls = cls_logits.sum() * 0.0
        total_bbox = box_preds.sum() * 0.0
        total_angle = box_preds.sum() * 0.0
        total_quality = quality_logits.sum() * 0.0
        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            otm = self._one_to_many_transport_loss_single(
                cls_logits[img_idx], box_preds[img_idx],
                quality_logits[img_idx], query_class_ids, gt_bboxes,
                gt_labels)
            total_cls = total_cls + otm[0]
            total_bbox = total_bbox + otm[1]
            total_angle = total_angle + otm[2]
            total_quality = total_quality + otm[3]

        batch_size = max(len(batch_gt_instances), 1)
        losses.update({
            'loss_query_transport_otm_cls':
            total_cls * self.aux_otm_loss_weight / batch_size,
            'loss_query_transport_otm_bbox':
            total_bbox * self.aux_otm_loss_weight * self.bbox_loss_weight /
            batch_size,
            'loss_query_transport_otm_angle':
            total_angle * self.aux_otm_loss_weight * self.angle_loss_weight /
            batch_size,
            'loss_query_transport_otm_quality':
            total_quality * self.aux_otm_loss_weight / batch_size,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14J One-to-Many Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_one_to_many_auxiliary_transport':
            self.aux_otm_loss_weight > 0,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14KGroupWiseQueryTransportHead(P14CClassConsistentQueryTransportHead):
    """P14-K: Group DETR-style training-only grouped query transport.

    Each query group receives one-to-one class-consistent transport supervision.
    Inference keeps a single fixed query group, so the output remains an
    end-to-end set with no dense head and no NMS.
    """

    def __init__(self,
                 *args,
                 group_transport_loss_weight: float = 0.50,
                 query_group_count: int = 2,
                 inference_single_group: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.group_transport_loss_weight = float(group_transport_loss_weight)
        self.query_group_count = max(int(query_group_count), 1)
        self.inference_single_group = bool(inference_single_group)
        self.e2e_debug.update({
            'p14_variant': 'P14K Group-Wise Query Transport',
            'uses_groupwise_query_transport':
            self.group_transport_loss_weight > 0 and
            self.query_group_count > 1,
            'query_group_count': self.query_group_count,
            'inference_single_group': self.inference_single_group,
        })

    def _query_group_slices(self, num_queries: int) -> list[slice]:
        if self.query_group_count <= 1:
            return [slice(0, num_queries)]
        group_size = max(num_queries // self.query_group_count, 1)
        slices = []
        for group_idx in range(self.query_group_count):
            start = group_idx * group_size
            end = num_queries if group_idx == self.query_group_count - 1 else (
                group_idx + 1) * group_size
            if start < num_queries and end > start:
                slices.append(slice(start, end))
        return slices

    def _primary_query_slice(self, num_queries: int) -> slice:
        if not self.inference_single_group or self.query_group_count <= 1:
            return slice(0, num_queries)
        return self._query_group_slices(num_queries)[0]

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = super().loss(x, batch_data_samples)
        if self.group_transport_loss_weight <= 0 or self.query_group_count <= 1:
            return losses

        cls_logits, box_preds, quality_logits, _ = self(x)
        group_slices = self._query_group_slices(cls_logits.shape[1])
        if len(group_slices) <= 1:
            return losses

        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, _, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)
        else:
            query_class_ids = query_class_ids.to(device=cls_logits.device)

        total_cls = cls_logits.sum() * 0.0
        total_bbox = box_preds.sum() * 0.0
        total_angle = box_preds.sum() * 0.0
        total_quality = quality_logits.sum() * 0.0
        total_rank = cls_logits.sum() * 0.0
        count = 0
        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            teacher_bboxes, teacher_labels, teacher_scores = (
                self._teacher_instances_for_transport(
                    batch_data_samples[img_idx]))
            for group_slice in group_slices:
                group_class_ids = query_class_ids[group_slice]
                qt = self._query_transport_loss_and_targets(
                    cls_logits[img_idx, group_slice],
                    box_preds[img_idx, group_slice],
                    quality_logits[img_idx, group_slice],
                    group_class_ids,
                    gt_bboxes,
                    gt_labels,
                    teacher_bboxes,
                    teacher_labels,
                    teacher_scores)
                total_cls = total_cls + qt[0]
                total_bbox = total_bbox + qt[1]
                total_angle = total_angle + qt[2]
                total_quality = total_quality + qt[3]
                total_rank = total_rank + self._transport_rank_loss(
                    cls_logits[img_idx, group_slice],
                    quality_logits[img_idx, group_slice],
                    group_class_ids,
                    qt[4])
                count += 1

        normalizer = max(count, 1)
        losses.update({
            'loss_query_transport_group_cls':
            total_cls * self.group_transport_loss_weight / normalizer,
            'loss_query_transport_group_bbox':
            total_bbox * self.group_transport_loss_weight *
            self.bbox_loss_weight / normalizer,
            'loss_query_transport_group_angle':
            total_angle * self.group_transport_loss_weight *
            self.angle_loss_weight / normalizer,
            'loss_query_transport_group_quality':
            total_quality * self.group_transport_loss_weight / normalizer,
            'loss_query_transport_group_rank':
            total_rank * self.group_transport_loss_weight *
            self.transport_rank_loss_weight / normalizer,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, quality_logits, _ = self(x)
        group_slice = self._primary_query_slice(cls_logits.shape[1])
        query_class_ids = self._query_class_ids(
            cls_logits.shape[1], cls_logits.device)[group_slice]
        query_arange = torch.arange(
            cls_logits.shape[1], device=cls_logits.device)[group_slice]
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            posterior = cls_logits[img_idx, group_slice].sigmoid()
            quality = quality_logits[img_idx, group_slice].sigmoid()
            local_arange = torch.arange(
                query_class_ids.shape[0], device=cls_logits.device)
            scores = posterior[local_arange, query_class_ids] * quality
            keep = torch.arange(scores.numel(), device=scores.device)
            if keep.numel() > self.max_per_img:
                keep = scores.topk(self.max_per_img).indices
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, query_arange[keep]],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores[keep]
            inst.labels = query_class_ids[keep].clone()
            results.append(inst)
        self.e2e_debug.update({
            'p14_variant': 'P14K Group-Wise Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_groupwise_query_transport':
            self.group_transport_loss_weight > 0 and
            self.query_group_count > 1,
            'query_group_count': self.query_group_count,
            'inference_single_group': self.inference_single_group,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14LGaussianGeometryQueryTransportHead(
        P14CClassConsistentQueryTransportHead):
    """P14-L: train-only Gaussian geometry loss for transport positives."""

    def __init__(self,
                 *args,
                 gwd_loss_weight: float = 4.0,
                 gwd_alpha: float = 1.0,
                 gwd_normalize: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gwd_loss_weight = float(gwd_loss_weight)
        self.gwd_alpha = float(gwd_alpha)
        self.gwd_normalize = bool(gwd_normalize)
        self.e2e_debug.update({
            'p14_variant': 'P14L Gaussian Geometry Query Transport',
            'uses_gaussian_transport_geometry_loss': self.gwd_loss_weight > 0,
            'gwd_loss_weight': self.gwd_loss_weight,
        })

    def _transport_gwd_loss(self, box_preds: Tensor,
                            targets: dict[str, Tensor]) -> Tensor:
        zero = box_preds.sum() * 0.0
        if self.gwd_loss_weight <= 0:
            return zero
        positive = targets['positive_mask']
        if not positive.any():
            return zero
        losses = _gaussian_wasserstein_loss(
            box_preds[positive],
            targets['target_boxes'][positive],
            alpha=self.gwd_alpha,
            normalize=self.gwd_normalize)
        return losses.sum() / positive.sum().clamp_min(1).to(
            dtype=box_preds.dtype)

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = super().loss(x, batch_data_samples)
        if self.gwd_loss_weight <= 0:
            return losses

        cls_logits, box_preds, quality_logits, _ = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, _, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)
        else:
            query_class_ids = query_class_ids.to(device=cls_logits.device)

        total_gwd = box_preds.sum() * 0.0
        for img_idx, gt_instances in enumerate(batch_gt_instances):
            gt_bboxes = self._gt_tensor(gt_instances, batch_img_metas[img_idx])
            gt_labels = gt_instances.labels.to(cls_logits.device)
            teacher_bboxes, teacher_labels, teacher_scores = (
                self._teacher_instances_for_transport(
                    batch_data_samples[img_idx]))
            qt = self._query_transport_loss_and_targets(
                cls_logits[img_idx],
                box_preds[img_idx],
                quality_logits[img_idx],
                query_class_ids,
                gt_bboxes,
                gt_labels,
                teacher_bboxes,
                teacher_labels,
                teacher_scores)
            total_gwd = total_gwd + self._transport_gwd_loss(
                box_preds[img_idx], qt[4])

        batch_size = max(len(batch_gt_instances), 1)
        losses.update({
            'loss_query_transport_gwd':
            total_gwd * self.gwd_loss_weight / batch_size,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14L Gaussian Geometry Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_gaussian_transport_geometry_loss': self.gwd_loss_weight > 0,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14MIterativeRefineQueryTransportHead(
        P14CClassConsistentQueryTransportHead):
    """P14-M: DAB/DINO-style iterative reference-box refinement.

    Final inference remains a fixed query set.  The structural change is inside
    the decoder: each layer predicts a residual around the current reference
    box, then the next layer receives the refined reference geometry.
    """

    def __init__(self,
                 *args,
                 intermediate_refine_loss_weight: float = 0.50,
                 iterative_center_delta_scale: float = 0.20,
                 iterative_angle_delta_scale: float = math.pi / 4.0,
                 detach_refine_between_layers: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.intermediate_refine_loss_weight = float(
            intermediate_refine_loss_weight)
        self.iterative_center_delta_scale = float(iterative_center_delta_scale)
        self.iterative_angle_delta_scale = float(iterative_angle_delta_scale)
        self.detach_refine_between_layers = bool(detach_refine_between_layers)
        self._p14m_last_intermediate: dict[str, list[Tensor]] = {
            'cls_logits': [],
            'boxes': [],
            'quality_logits': [],
        }
        self.e2e_debug.update({
            'p14_variant': 'P14M Iterative Reference Refinement',
            'uses_iterative_reference_refinement': True,
            'uses_intermediate_refine_loss':
            self.intermediate_refine_loss_weight > 0,
            'detach_refine_between_layers': self.detach_refine_between_layers,
        })

    def _normalize_angle_le90(self, angle: Tensor) -> Tensor:
        return torch.remainder(angle + math.pi / 2.0, math.pi) - math.pi / 2.0

    def _initial_reference_boxes(self,
                                 query_coords: Tensor,
                                 query_class_ids: Tensor | None,
                                 base_wh_override: Tensor | None = None
                                 ) -> Tensor:
        batch_size, num_queries, _ = query_coords.shape
        if base_wh_override is None:
            base_wh = self._reference_wh_for_queries(
                query_class_ids,
                num_queries,
                query_coords.device,
                query_coords.dtype).unsqueeze(0).expand(
                    batch_size, -1, -1)
        else:
            base_wh = base_wh_override
            if base_wh.dim() == 2:
                base_wh = base_wh.unsqueeze(0)
            base_wh = base_wh.to(
                device=query_coords.device, dtype=query_coords.dtype)
            if base_wh.shape[0] == 1 and batch_size > 1:
                base_wh = base_wh.expand(batch_size, -1, -1)
        angle = query_coords.new_zeros(batch_size, num_queries, 1)
        return torch.cat([
            query_coords[..., :2].clamp(min=1e-4, max=1.0 - 1e-4),
            base_wh.clamp(min=1e-4, max=1.0),
            angle,
        ], dim=-1)

    def _layer_logits_and_quality(
            self, decoded: Tensor,
            query_class_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        semantic = F.normalize(self.semantic_proj(decoded), dim=-1)
        support = F.normalize(self.support_tokens, dim=-1)
        cls_logits = semantic @ support.t()
        cls_logits = cls_logits + self.class_bias.view(1, 1, -1)
        if query_class_ids is not None and self.own_class_logit_bias != 0:
            class_bias = F.one_hot(
                query_class_ids,
                num_classes=self.num_classes).to(dtype=cls_logits.dtype)
            cls_logits = cls_logits + self.own_class_logit_bias * class_bias
        cls_logits = cls_logits * self.logit_scale.exp().clamp(1e-4, 100.0)
        quality_logits = self.quality(decoded).squeeze(-1)
        return cls_logits, quality_logits

    def _refine_reference_boxes(self, reference_boxes: Tensor,
                                raw_box: Tensor) -> Tensor:
        center = (
            reference_boxes[..., :2] +
            self.iterative_center_delta_scale * raw_box[..., :2].tanh()
        ).clamp(min=1e-4, max=1.0 - 1e-4)
        wh = (
            reference_boxes[..., 2:4] *
            torch.exp(self.box_delta_scale * raw_box[..., 2:4].tanh())
        ).clamp(min=1e-4, max=1.0)
        angle = self._normalize_angle_le90(
            reference_boxes[..., 4:5] +
            self.iterative_angle_delta_scale * raw_box[..., 4:5].tanh())
        return torch.cat([center, wh, angle], dim=-1)

    def _decode_queries(self,
                        memory: Tensor,
                        query_memory: Tensor,
                        query_coords: Tensor,
                        query_strides: Tensor,
                        query_class_ids: Tensor | None = None,
                        base_wh_override: Tensor | None = None
                        ) -> tuple[Tensor, Tensor, Tensor]:
        num_queries = query_memory.shape[1]
        query_embed = self._query_embed_slice(
            num_queries, query_memory.device).unsqueeze(0)
        reference_boxes = self._initial_reference_boxes(
            query_coords, query_class_ids, base_wh_override)
        level_coord = query_coords[..., 2:3]
        decoded = query_memory

        cls_layers = []
        box_layers = []
        quality_layers = []
        for layer in self.decoder.layers:
            refined_coords = torch.cat(
                [reference_boxes[..., :2], level_coord], dim=-1)
            query = decoded + self.coord_proj(refined_coords) + query_embed
            if query_class_ids is not None and self.query_class_prior_weight != 0:
                class_prior = self.support_tokens[query_class_ids].unsqueeze(0)
                query = query + self.query_class_prior_weight * class_prior

            decoded = layer(query, memory)
            raw_box = self.box_delta(decoded)
            box_preds = self._refine_reference_boxes(reference_boxes, raw_box)
            cls_logits, quality_logits = self._layer_logits_and_quality(
                decoded, query_class_ids)
            cls_layers.append(cls_logits)
            box_layers.append(box_preds)
            quality_layers.append(quality_logits)
            if self.detach_refine_between_layers:
                reference_boxes = box_preds.detach()
            else:
                reference_boxes = box_preds

        self._p14m_last_intermediate = {
            'cls_logits': cls_layers,
            'boxes': box_layers,
            'quality_logits': quality_layers,
        }
        return cls_layers[-1], box_layers[-1], quality_layers[-1]

    def loss(self, x: tuple[Tensor, ...], batch_data_samples) -> dict:
        losses = super().loss(x, batch_data_samples)
        if self.intermediate_refine_loss_weight <= 0:
            return losses

        cls_logits, _, _, _ = self(x)
        intermediates = self._p14m_last_intermediate
        if len(intermediates['boxes']) <= 1:
            return losses

        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        memory, coords, strides, flat_seed_logits = self._flatten_features(x)
        _, _, _, query_class_ids = self._select_query_seeds(
            memory, coords, strides, flat_seed_logits)
        if query_class_ids is None:
            query_class_ids = self._query_class_ids(
                cls_logits.shape[1], cls_logits.device)
        else:
            query_class_ids = query_class_ids.to(device=cls_logits.device)

        total_cls = cls_logits.sum() * 0.0
        total_bbox = cls_logits.sum() * 0.0
        total_angle = cls_logits.sum() * 0.0
        total_quality = cls_logits.sum() * 0.0
        total_rank = cls_logits.sum() * 0.0
        count = 0
        aux_layers = zip(intermediates['cls_logits'][:-1],
                         intermediates['boxes'][:-1],
                         intermediates['quality_logits'][:-1])
        for layer_cls, layer_boxes, layer_quality in aux_layers:
            for img_idx, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = self._gt_tensor(gt_instances,
                                            batch_img_metas[img_idx])
                gt_labels = gt_instances.labels.to(cls_logits.device)
                teacher_bboxes, teacher_labels, teacher_scores = (
                    self._teacher_instances_for_transport(
                        batch_data_samples[img_idx]))
                qt = self._query_transport_loss_and_targets(
                    layer_cls[img_idx],
                    layer_boxes[img_idx],
                    layer_quality[img_idx],
                    query_class_ids,
                    gt_bboxes,
                    gt_labels,
                    teacher_bboxes,
                    teacher_labels,
                    teacher_scores)
                total_cls = total_cls + qt[0]
                total_bbox = total_bbox + qt[1]
                total_angle = total_angle + qt[2]
                total_quality = total_quality + qt[3]
                total_rank = total_rank + self._transport_rank_loss(
                    layer_cls[img_idx],
                    layer_quality[img_idx],
                    query_class_ids,
                    qt[4])
                count += 1

        normalizer = max(count, 1)
        losses.update({
            'loss_query_refine_aux_cls':
            total_cls * self.intermediate_refine_loss_weight *
            self.query_transport_loss_weight / normalizer,
            'loss_query_refine_aux_bbox':
            total_bbox * self.intermediate_refine_loss_weight *
            self.query_transport_loss_weight * self.bbox_loss_weight /
            normalizer,
            'loss_query_refine_aux_angle':
            total_angle * self.intermediate_refine_loss_weight *
            self.query_transport_loss_weight * self.angle_loss_weight /
            normalizer,
            'loss_query_refine_aux_quality':
            total_quality * self.intermediate_refine_loss_weight *
            self.query_transport_loss_weight / normalizer,
            'loss_query_refine_aux_rank':
            total_rank * self.intermediate_refine_loss_weight *
            self.transport_rank_loss_weight / normalizer,
        })
        return losses

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14M Iterative Reference Refinement',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_soft_gaussian_teacher_field': False,
            'uses_dab_reference_boxes': True,
            'uses_stride_width_reference': False,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_class_locked_posterior': True,
            'uses_iterative_reference_refinement': True,
            'uses_intermediate_refine_loss':
            self.intermediate_refine_loss_weight > 0,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14NRankQualityQueryTransportHead(P14CClassConsistentQueryTransportHead):
    """P14-N: rank-quality posterior for strict E2E query transport.

    P14-C can place useful same-class queries near GT boxes, but the final AP
    collapses when those queries are scored below poorer boxes.  This variant
    keeps the fixed-query, no-NMS inference contract and changes only the
    posterior supervision: positive class/quality targets are continuous
    geometry-quality scores, so better localized queries are trained to rank
    higher.
    """

    def __init__(self,
                 *args,
                 rank_quality_floor: float = 0.25,
                 rank_quality_center_sigma: float = 0.45,
                 rank_quality_size_sigma: float = 0.65,
                 rank_quality_angle_sigma: float = 0.60,
                 rank_quality_margin_scale: float = 0.10,
                 rank_quality_cls_target_blend: float = 1.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rank_quality_floor = float(rank_quality_floor)
        self.rank_quality_center_sigma = max(
            float(rank_quality_center_sigma), 1e-4)
        self.rank_quality_size_sigma = max(
            float(rank_quality_size_sigma), 1e-4)
        self.rank_quality_angle_sigma = max(
            float(rank_quality_angle_sigma), 1e-4)
        self.rank_quality_margin_scale = float(rank_quality_margin_scale)
        self.rank_quality_cls_target_blend = min(
            max(float(rank_quality_cls_target_blend), 0.0), 1.0)
        self.e2e_debug.update({
            'p14_variant': 'P14N Rank-Quality Query Transport',
            'uses_rank_quality_posterior': True,
            'uses_geometry_aware_score_targets': True,
            'rank_quality_cls_target_blend':
            self.rank_quality_cls_target_blend,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })

    def _rank_quality_from_boxes(self, pred_boxes: Tensor,
                                 target_boxes: Tensor) -> Tensor:
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes.new_zeros((pred_boxes.shape[0], ))
        target_wh = target_boxes[:, 2:4].clamp(min=1e-4)
        center_delta = (pred_boxes[:, :2] - target_boxes[:, :2]) / target_wh
        center_quality = torch.exp(
            -0.5 * center_delta.square().sum(dim=-1) /
            (self.rank_quality_center_sigma**2))
        size_delta = torch.log(
            pred_boxes[:, 2:4].clamp(min=1e-4) / target_wh).abs()
        size_quality = torch.exp(
            -size_delta.mean(dim=-1) / self.rank_quality_size_sigma)
        angle_delta = torch.atan2(
            torch.sin(pred_boxes[:, 4] - target_boxes[:, 4]),
            torch.cos(pred_boxes[:, 4] - target_boxes[:, 4])).abs()
        angle_quality = torch.exp(
            -angle_delta / self.rank_quality_angle_sigma)
        quality = center_quality * size_quality * angle_quality
        floor = min(max(self.rank_quality_floor, 0.0), 1.0)
        return (floor + (1.0 - floor) * quality).clamp(min=0.0, max=1.0)

    def _rank_quality_class_target(self, source_scores: Tensor,
                                   rank_quality: Tensor) -> Tensor:
        blend = min(max(self.rank_quality_cls_target_blend, 0.0), 1.0)
        return (
            (1.0 - blend) * source_scores +
            blend * rank_quality).clamp(min=0.0, max=1.0)

    def _query_transport_loss_and_targets(
            self,
            cls_logits: Tensor,
            box_preds: Tensor,
            quality_logits: Tensor,
            query_class_ids: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            teacher_bboxes: Tensor,
            teacher_labels: Tensor,
            teacher_scores: Tensor
            ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        zero = cls_logits.sum() * 0.0
        targets = self._box_query_transport_targets(
            box_preds.detach(), gt_bboxes, gt_labels, teacher_bboxes,
            teacher_labels, teacher_scores, query_class_ids=query_class_ids)
        positive = targets['positive_mask']
        rank_quality = quality_logits.new_zeros(quality_logits.shape)
        if positive.any():
            source_scores = targets['target_scores'][positive].to(
                dtype=quality_logits.dtype)
            geometry_quality = self._rank_quality_from_boxes(
                box_preds[positive].detach(),
                targets['target_boxes'][positive])
            rank_quality[positive] = (geometry_quality *
                                      source_scores).clamp(min=0.0, max=1.0)
        targets['rank_quality_scores'] = rank_quality

        if self.query_transport_loss_weight <= 0:
            return zero, zero, zero, zero, targets

        cls_target = torch.zeros_like(cls_logits)
        quality_target = rank_quality.to(dtype=quality_logits.dtype)
        if positive.any():
            cls_scores = self._rank_quality_class_target(
                targets['target_scores'][positive].to(dtype=cls_logits.dtype),
                quality_target[positive].to(dtype=cls_logits.dtype))
            cls_target[
                positive,
                targets['target_labels'][positive]] = cls_scores
        normalizer = positive.sum().clamp_min(1).to(dtype=cls_logits.dtype)
        loss_cls = py_sigmoid_focal_loss(
            cls_logits,
            cls_target,
            alpha=0.25,
            gamma=2.0,
            reduction='sum') / normalizer
        loss_quality = F.binary_cross_entropy_with_logits(
            quality_logits, quality_target, reduction='sum') / max(
                int(cls_logits.shape[0]), 1)

        if not positive.any():
            return loss_cls, zero, zero, loss_quality, targets
        loss_bbox = F.l1_loss(
            box_preds[positive, :4],
            targets['target_boxes'][positive, :4],
            reduction='sum') / normalizer
        angle_diff = torch.atan2(
            torch.sin(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4]),
            torch.cos(box_preds[positive, 4] -
                      targets['target_boxes'][positive, 4])).abs()
        loss_angle = angle_diff.sum() / normalizer
        return loss_cls, loss_bbox, loss_angle, loss_quality, targets

    def _transport_rank_loss(self, cls_logits: Tensor,
                             quality_logits: Tensor,
                             query_class_ids: Tensor,
                             targets: dict[str, Tensor]) -> Tensor:
        zero = cls_logits.sum() * 0.0
        if self.transport_rank_loss_weight <= 0:
            return zero
        positive = targets['positive_mask']
        if not positive.any():
            return zero
        own_scores = self._rank_own_scores(
            cls_logits, quality_logits, query_class_ids)
        rank_quality = targets.get('rank_quality_scores')
        if rank_quality is None:
            rank_quality = positive.to(dtype=own_scores.dtype)
        rank_quality = rank_quality.to(device=own_scores.device,
                                       dtype=own_scores.dtype)

        rank_terms = []
        for class_id in query_class_ids.unique():
            same_class = query_class_ids == class_id
            pos = positive & same_class & (
                targets['target_labels'] == class_id)
            neg = (~positive) & same_class
            if pos.any() and neg.any():
                pos_scores = own_scores[pos]
                pos_quality = rank_quality[pos]
                neg_scores = own_scores[neg]
                hard_neg = neg_scores.topk(
                    min(int(neg_scores.numel()),
                        max(int(pos_scores.numel()) * 4, 1))).values
                margin = (
                    self.ranking_margin +
                    self.rank_quality_margin_scale * pos_quality).view(-1, 1)
                rank_terms.append(
                    (pos_quality.view(-1, 1) *
                     F.relu(margin - pos_scores[:, None] +
                            hard_neg[None, :])).mean())

            pos_idx = torch.where(pos)[0]
            if pos_idx.numel() > 1:
                pos_scores = own_scores[pos_idx]
                pos_quality = rank_quality[pos_idx]
                quality_gap = pos_quality[:, None] - pos_quality[None, :]
                ordered = quality_gap > 0.05
                if ordered.any():
                    pair_loss = F.relu(
                        self.ranking_margin +
                        pos_scores[None, :] - pos_scores[:, None])
                    rank_terms.append(
                        (pair_loss[ordered] *
                         quality_gap[ordered].detach()).mean())
        if not rank_terms:
            return zero
        return torch.stack(rank_terms).mean()

    def _rank_own_scores(self, cls_logits: Tensor, quality_logits: Tensor,
                         query_class_ids: Tensor) -> Tensor:
        posterior = cls_logits.sigmoid() * quality_logits.sigmoid().unsqueeze(
            -1)
        return posterior[
            torch.arange(cls_logits.shape[0], device=cls_logits.device),
            query_class_ids]

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        results = super().predict(x, batch_data_samples, rescale=rescale)
        self.e2e_debug.update({
            'p14_variant': 'P14N Rank-Quality Query Transport',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_dab_reference_boxes': True,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_rank_quality_posterior': True,
            'uses_geometry_aware_score_targets': True,
            'rank_quality_cls_target_blend':
            self.rank_quality_cls_target_blend,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


@MODELS.register_module()
class P14OQualityIntegratedClassPosteriorHead(
        P14NRankQualityQueryTransportHead):
    """P14-O: GFL/VFNet-style quality-integrated class posterior.

    P14-N trained both the class target and the quality branch with geometry
    quality, then multiplied them at inference.  Full HRRSD training showed a
    low-score collapse.  This variant keeps strict fixed-query E2E inference
    and treats the class logit as the final quality-aware posterior score.
    The quality branch remains a train-time auxiliary target, but it is not a
    second inference multiplier.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e2e_debug.update({
            'p14_variant': 'P14O Quality-Integrated Class Posterior',
            'uses_quality_integrated_class_posterior': True,
            'uses_quality_score_product': False,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })

    def _rank_own_scores(self, cls_logits: Tensor, quality_logits: Tensor,
                         query_class_ids: Tensor) -> Tensor:
        del quality_logits
        class_scores = cls_logits.sigmoid()
        return class_scores[
            torch.arange(cls_logits.shape[0], device=cls_logits.device),
            query_class_ids]

    def predict(self, x: tuple[Tensor, ...], batch_data_samples,
                rescale: bool = False) -> list[InstanceData]:
        cls_logits, box_preds, _, _ = self(x)
        query_class_ids = self._query_class_ids(
            cls_logits.shape[1], cls_logits.device)
        query_arange = torch.arange(
            cls_logits.shape[1], device=cls_logits.device)
        results = []
        for img_idx, data_sample in enumerate(batch_data_samples):
            posterior = cls_logits[img_idx].sigmoid()
            scores = posterior[query_arange, query_class_ids]
            keep = query_arange
            if keep.numel() > self.max_per_img:
                keep = scores.topk(self.max_per_img).indices
            bboxes = self._boxes_to_pixels(
                box_preds[img_idx, keep],
                data_sample.metainfo,
                rescale=rescale)
            inst = InstanceData()
            inst.bboxes = bboxes
            inst.scores = scores[keep]
            inst.labels = query_class_ids[keep].clone()
            results.append(inst)
        self.e2e_debug.update({
            'p14_variant': 'P14O Quality-Integrated Class Posterior',
            'strict_e2e': True,
            'uses_nms': False,
            'uses_dense_detection_head': False,
            'uses_box_query_transport': True,
            'uses_dab_reference_boxes': True,
            'uses_class_consistent_transport': True,
            'uses_transport_rank_loss': self.transport_rank_loss_weight > 0,
            'uses_rank_quality_posterior': True,
            'uses_geometry_aware_score_targets': True,
            'uses_quality_integrated_class_posterior': True,
            'uses_quality_score_product': False,
            'uses_score_threshold_postprocess': False,
            'outputs_fixed_query_set': True,
        })
        return results


def _register_p13_p14_heads_to_mmdet() -> None:
    """Expose these strict set heads to detectors built under mmdet scope."""
    for name, cls in list(globals().items()):
        if (name.startswith(('P13', 'P14')) and isinstance(cls, type)
                and issubclass(cls, BaseModule)):
            MMDET_MODELS.register_module(module=cls, force=True)


_register_p13_p14_heads_to_mmdet()
