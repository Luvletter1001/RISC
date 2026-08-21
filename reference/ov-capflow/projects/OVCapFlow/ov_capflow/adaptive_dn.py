"""Open-vocabulary positive-Hungarian denoising targets.

This adapts RHINO's positive-Hungarian denoising rule to GroundingDINO's
token-positive maps. A noisy positive remains a regression/classification
positive only when the joint Hungarian assignment chooses its original GT.
"""

from typing import Dict

import torch
from mmdet.models.losses import QualityFocalLoss
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.structures import InstanceData
from torch import Tensor


def build_adaptive_dn_targets_single(
        dn_cls_score: Tensor,
        dn_bbox_pred: Tensor,
        matching_cls_score: Tensor,
        matching_bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        dn_meta: Dict[str, int],
        dn_assigner,
        max_text_len: int,
        angle_factor: float,
        angle_cfg: dict):
    """Build one image's token-level adaptive denoising targets."""
    num_groups = int(dn_meta['num_denoising_groups'])
    num_denoising_queries = int(dn_meta['num_denoising_queries'])
    if num_groups < 1:
        raise ValueError('num_denoising_groups must be positive')
    if num_denoising_queries % num_groups:
        raise ValueError(
            'num_denoising_queries must be divisible by its groups')
    if dn_cls_score.shape[0] != num_denoising_queries:
        raise ValueError('denoising predictions do not match dn_meta')
    if dn_cls_score.shape[-1] != max_text_len:
        raise ValueError('denoising token width must equal max_text_len')

    gt_instances.bboxes.regularize_boxes(**angle_cfg)
    gt_bboxes = gt_instances.bboxes.tensor
    device = gt_bboxes.device
    num_gt = len(gt_instances)
    queries_each_group = num_denoising_queries // num_groups
    if num_gt > queries_each_group // 2:
        raise ValueError('denoising group has too few positive slots for GT')

    if num_gt:
        positive_maps = gt_instances.positive_maps
        if positive_maps.shape != (num_gt, max_text_len):
            raise ValueError('GT positive_maps must match max_text_len')
        gt_order = torch.arange(
            num_gt, dtype=torch.long, device=device).repeat(num_groups)
        group_offsets = torch.arange(
            num_groups, dtype=torch.long, device=device) * queries_each_group
        pos_inds = (group_offsets[:, None] + torch.arange(
            num_gt, dtype=torch.long, device=device)[None, :]).flatten()

        img_h, img_w = img_meta['img_shape'][:2]
        factor = gt_bboxes.new_tensor(
            [img_w, img_h, img_w, img_h, angle_factor]).unsqueeze(0)
        pred_instances = InstanceData(
            scores=matching_cls_score,
            bboxes=matching_bbox_pred * factor)
        pos_dn_instances = InstanceData(
            scores=dn_cls_score[pos_inds],
            bboxes=dn_bbox_pred[pos_inds] * factor)
        assign_result = dn_assigner.assign(
            pred_instances=pred_instances,
            dn_instances=pos_dn_instances,
            gt_instances=gt_instances,
            dn_meta=dn_meta,
            img_meta=img_meta)
        retained = gt_order == (assign_result.gt_inds - 1)
    else:
        factor = gt_bboxes.new_tensor(
            [img_meta['img_shape'][1], img_meta['img_shape'][0],
             img_meta['img_shape'][1], img_meta['img_shape'][0],
             angle_factor]).unsqueeze(0)
        gt_order = pos_inds = torch.empty(
            0, dtype=torch.long, device=device)
        retained = torch.empty(0, dtype=torch.bool, device=device)

    original_negative_inds = pos_inds + queries_each_group // 2
    rejected_positive_inds = pos_inds[~retained]
    neg_inds = torch.cat(
        [rejected_positive_inds, original_negative_inds], dim=0)
    pos_inds = pos_inds[retained]
    assigned_gt_inds = gt_order[retained]

    labels = gt_bboxes.new_zeros(
        (num_denoising_queries, max_text_len), dtype=torch.float32)
    if pos_inds.numel():
        labels[pos_inds] = gt_instances.positive_maps[assigned_gt_inds]
    label_weights = gt_bboxes.new_ones(num_denoising_queries)

    bbox_targets = gt_bboxes.new_zeros((num_denoising_queries, 5))
    bbox_weights = gt_bboxes.new_zeros((num_denoising_queries, 5))
    bbox_weights[pos_inds] = 1.0
    if pos_inds.numel():
        bbox_targets[pos_inds] = (gt_bboxes / factor)[assigned_gt_inds]

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


class OpenVocabularyAdaptiveDNMixin:
    """RHINO positive-Hungarian DN with token-level positive maps."""

    def adaptive_loss_dn(self, all_dn_cls_scores, all_dn_bbox_preds,
                         all_matching_cls_scores, all_matching_bbox_preds,
                         batch_gt_instances, batch_img_metas, dn_meta):
        return multi_apply(
            self._adaptive_loss_dn_single,
            all_dn_cls_scores,
            all_dn_bbox_preds,
            all_matching_cls_scores,
            all_matching_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def get_adaptive_dn_targets(self, dn_cls_scores, dn_bbox_preds,
                                matching_cls_scores, matching_bbox_preds,
                                batch_gt_instances, batch_img_metas, dn_meta):
        num_images = dn_cls_scores.size(0)
        target_lists = multi_apply(
            self._get_adaptive_dn_targets_single,
            [dn_cls_scores[index] for index in range(num_images)],
            [dn_bbox_preds[index] for index in range(num_images)],
            [matching_cls_scores[index] for index in range(num_images)],
            [matching_bbox_preds[index] for index in range(num_images)],
            batch_gt_instances,
            batch_img_metas,
            dn_meta=dn_meta)
        (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
         neg_inds) = target_lists
        num_total_pos = sum(indices.numel() for indices in pos_inds)
        num_total_neg = sum(indices.numel() for indices in neg_inds)
        return (labels, label_weights, bbox_targets, bbox_weights,
                num_total_pos, num_total_neg)

    def _get_adaptive_dn_targets_single(
            self, dn_cls_score, dn_bbox_pred, matching_cls_score,
            matching_bbox_pred, gt_instances, img_meta, dn_meta):
        return build_adaptive_dn_targets_single(
            dn_cls_score=dn_cls_score,
            dn_bbox_pred=dn_bbox_pred,
            matching_cls_score=matching_cls_score,
            matching_bbox_pred=matching_bbox_pred,
            gt_instances=gt_instances,
            img_meta=img_meta,
            dn_meta=dn_meta,
            dn_assigner=self.dn_assigner,
            max_text_len=self.max_text_len,
            angle_factor=self.angle_factor,
            angle_cfg=self.angle_cfg)

    def _adaptive_loss_dn_single(
            self, dn_cls_scores, dn_bbox_preds, matching_cls_scores,
            matching_bbox_preds, batch_gt_instances, batch_img_metas,
            dn_meta):
        targets = self.get_adaptive_dn_targets(
            dn_cls_scores, dn_bbox_preds, matching_cls_scores,
            matching_bbox_preds, batch_gt_instances, batch_img_metas,
            dn_meta)
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        if self.text_masks.dim() != 2:
            raise ValueError('text_masks must have shape [batch, tokens]')
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1).repeat(
            1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(
            dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[..., None].repeat(
            1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        cls_avg_factor = (
            num_total_pos + num_total_neg * self.bg_cls_weight)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        if cls_scores.numel():
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError(
                    'QualityFocalLoss is not supported for adaptive DN')
            loss_cls = self.loss_cls(
                cls_scores,
                labels,
                label_weights,
                avg_factor=cls_avg_factor)
        else:
            loss_cls = cls_scores.new_zeros(1)

        normalized_pos = loss_cls.new_tensor([num_total_pos])
        normalized_pos = torch.clamp(
            reduce_mean(normalized_pos), min=1).item()
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape'][:2]
            factor = bbox_pred.new_tensor([
                img_w, img_h, img_w, img_h, self.angle_factor
            ]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        flat_bbox_preds = dn_bbox_preds.reshape(-1, 5)
        scaled_bbox_preds = flat_bbox_preds * factors
        scaled_bbox_targets = bbox_targets * factors
        loss_iou = self.loss_iou(
            scaled_bbox_preds,
            scaled_bbox_targets,
            bbox_weights,
            avg_factor=normalized_pos)
        loss_bbox = self.loss_bbox(
            flat_bbox_preds,
            bbox_targets,
            bbox_weights,
            avg_factor=normalized_pos)
        return loss_cls, loss_bbox, loss_iou
