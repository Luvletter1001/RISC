import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.dense_heads import DeformableDETRHead
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures.bbox import rbbox_overlaps
from projects.GroundingDINO.groundingdino.grounding_dino_head import (
    RotatedGroundingDINOHead, )

from .calibration import (balanced_group_classification_loss,
                          calibrate_selected_log_scores,
                          grounding_logits_to_class_log_scores,
                          select_from_class_log_scores)
from .adaptive_dn import OpenVocabularyAdaptiveDNMixin
from .grouped_queries import (grouped_matching_losses,
                              split_matching_groups)
from .existence_residual import (ExistenceResidual,
                                 centered_existence_log_residual,
                                 grouped_existence_bce_loss)


def scale_positive_maps_by_rotated_iou(
        labels: Tensor,
        bbox_pred: Tensor,
        bbox_targets: Tensor,
        pos_inds: Tensor,
        img_meta: dict,
        angle_factor: float) -> Tensor:
    """Scale matched token maps by detached aligned rotated IoU."""
    if labels.ndim != 2:
        raise ValueError('labels must have shape [num_queries, tokens]')
    if (bbox_pred.ndim != 2 or bbox_pred.shape[-1] != 5 or
            bbox_targets.shape != bbox_pred.shape):
        raise ValueError(
            'bbox_pred and bbox_targets must share shape [num_queries, 5]')
    if labels.shape[0] != bbox_pred.shape[0]:
        raise ValueError('labels and boxes must share the query dimension')
    if pos_inds.ndim != 1 or pos_inds.dtype != torch.long:
        raise ValueError('pos_inds must be a one-dimensional long tensor')
    if not torch.is_floating_point(labels):
        raise ValueError('labels must have a floating-point dtype')
    if not torch.isfinite(labels).all():
        raise ValueError('labels must be finite')
    if pos_inds.numel() == 0:
        return labels
    if pos_inds.min().item() < 0 or pos_inds.max().item() >= labels.shape[0]:
        raise ValueError('pos_inds contains an out-of-range query index')

    img_h, img_w = img_meta['img_shape'][:2]
    factor = bbox_pred.new_tensor(
        [img_w, img_h, img_w, img_h, angle_factor],
        dtype=torch.float32)
    positive_pred = bbox_pred.index_select(0, pos_inds).detach().float()
    positive_target = bbox_targets.index_select(
        0, pos_inds).detach().float()
    positive_pred = positive_pred * factor
    positive_target = positive_target * factor
    if (not torch.isfinite(positive_pred).all() or
            not torch.isfinite(positive_target).all()):
        raise ValueError('positive rotated boxes must be finite')

    quality = rbbox_overlaps(
        positive_pred, positive_target, is_aligned=True).detach()
    if quality.shape != pos_inds.shape:
        raise ValueError('aligned IoU count must match positive indices')
    if not torch.isfinite(quality).all():
        raise ValueError('aligned rotated IoU must be finite')
    quality = quality.clamp(0, 1).to(dtype=labels.dtype)

    scaled_labels = labels.clone()
    scaled_labels[pos_inds] = (
        scaled_labels.index_select(0, pos_inds) * quality.unsqueeze(-1))
    if not torch.isfinite(scaled_labels).all():
        raise ValueError('scaled labels must be finite')
    return scaled_labels


def select_one_class_per_query(class_scores: Tensor,
                               boxes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Choose one class for every query without ranking query rows."""
    if class_scores.ndim != 2:
        raise ValueError('class_scores must have shape [num_queries, classes]')
    if boxes.ndim != 2 or boxes.shape[-1] != 5:
        raise ValueError('boxes must have shape [num_queries, 5]')
    if class_scores.shape[0] != boxes.shape[0]:
        raise ValueError(
            'class_scores and boxes need the same number of queries')

    scores, labels = class_scores.max(dim=-1)
    return scores, labels, boxes


@MODELS.register_module()
class OVCapFlowHead(OpenVocabularyAdaptiveDNMixin,
                    RotatedGroundingDINOHead):
    """Grounding head that emits exactly one prediction per matching query."""

    def __init__(self, balanced_cfg=None, readout_cfg=None,
                 adaptive_dn_cfg=None, matching_query_groups=1,
                 position_supervised_cfg=None,
                 existence_loss_weight=None, **kwargs):
        if (not isinstance(matching_query_groups, int) or
                matching_query_groups < 1):
            raise ValueError(
                'matching_query_groups must be a positive integer')
        self.balanced_cfg = dict(balanced_cfg or {})
        if position_supervised_cfg is None:
            position_supervised_cfg = {}
        if not isinstance(position_supervised_cfg, dict):
            raise TypeError('position_supervised_cfg must be a dict')
        unknown_keys = set(position_supervised_cfg) - {'enabled'}
        if unknown_keys:
            raise ValueError(
                'position_supervised_cfg contains unknown keys')
        position_supervised_enabled = position_supervised_cfg.get(
            'enabled', False)
        if type(position_supervised_enabled) is not bool:
            raise TypeError('position_supervised_cfg.enabled must be a bool')
        self.position_supervised_cfg = {
            'enabled': position_supervised_enabled,
        }
        self.position_supervised_enabled = position_supervised_enabled
        if (self.position_supervised_enabled and
                self.balanced_cfg.get('enabled', False)):
            raise ValueError(
                'position supervision is incompatible with balanced loss')
        self.readout_cfg = dict(readout_cfg or {})
        self.adaptive_dn_cfg = dict(adaptive_dn_cfg or {})
        self.adaptive_dn_enabled = bool(
            self.adaptive_dn_cfg.get('enabled', False))
        self.matching_query_groups = matching_query_groups
        self.last_matching_mask = None
        self.last_balance_stats = None
        if existence_loss_weight is not None:
            if type(existence_loss_weight) is not float:
                raise TypeError(
                    'existence_loss_weight must be None or float 0.0/1.0')
            if (not math.isfinite(existence_loss_weight) or
                    existence_loss_weight not in (0.0, 1.0) or
                    (existence_loss_weight == 0.0 and math.copysign(
                        1.0, existence_loss_weight) < 0)):
                raise ValueError(
                    'existence_loss_weight must be exactly 0.0 or 1.0')
        self.existence_loss_weight = existence_loss_weight
        super().__init__(**kwargs)
        self._capture_matching_targets = False
        self._suppress_matching_target_capture = False
        self._captured_matching_mask = None
        self._loss_matching_group_masks = []
        self.last_existence_logits = None
        self.last_existence_residual = None
        self.last_matching_group_masks = None
        self.last_matching_group_matched_counts = None
        self.last_existence_logit_quantiles = None
        self.last_existence_residual_quantiles = None
        self.last_existence_clamp_hit_counts = None
        if existence_loss_weight is not None:
            if self.matching_query_groups != 3:
                raise ValueError(
                    'D13-N matching_query_groups must be exactly 3')
            if getattr(self, 'embed_dims', None) != 256:
                raise ValueError('D13-N embed_dims must be exactly 256')
            self.existence_residual = ExistenceResidual(256)
        if self.adaptive_dn_enabled:
            train_cfg = kwargs.get('train_cfg')
            if train_cfg is None or 'dn_assigner' not in train_cfg:
                raise ValueError(
                    'adaptive DN requires train_cfg.dn_assigner')
            self.dn_assigner = TASK_UTILS.build(train_cfg['dn_assigner'])

    @property
    def _existence_enabled(self):
        return hasattr(self, 'existence_residual')

    @staticmethod
    def _group_quantiles(values):
        if values.ndim != 3:
            raise ValueError('telemetry values must have shape [B,G,Q]')
        grouped = values.detach().float().permute(1, 0, 2).reshape(
            values.shape[1], -1)
        quantiles = torch.tensor(
            [0.05, 0.5, 0.95], device=grouped.device,
            dtype=grouped.dtype)
        return torch.quantile(grouped, quantiles, dim=1).transpose(0, 1)

    def _cache_existence_training_state(self, logits, residual, masks,
                                        selected_parent_log_scores=None):
        self.last_existence_logits = logits.detach()
        self.last_existence_residual = residual.detach()
        self.last_matching_group_masks = masks.detach()
        self.last_matching_mask = self.last_matching_group_masks[:, 0]
        self.last_matching_group_matched_counts = masks.sum(dim=-1).detach()
        self.last_existence_logit_quantiles = self._group_quantiles(logits)
        self.last_existence_residual_quantiles = self._group_quantiles(
            residual)
        if selected_parent_log_scores is None:
            hits = logits.new_zeros(2, dtype=torch.long)
        else:
            calibrated = selected_parent_log_scores.detach().float()
            calibrated = calibrated * (
                float(self.readout_cfg.get('power', 1.0)) /
                float(self.readout_cfg.get('temperature', 1.0)))
            would_be = calibrated + residual.detach().float()
            hits = torch.stack([
                (would_be < -80.0).sum(), (would_be > 0.0).sum()
            ])
        self.last_existence_clamp_hit_counts = hits.detach()

    def _selected_training_log_scores(self, all_layers_cls_scores,
                                      all_layers_bbox_preds, dn_meta,
                                      batch_data_samples):
        matching_cls, _, _, _ = self.split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        final_cls = matching_cls[-1]
        groups = dn_meta['num_matching_query_groups']
        queries_per_group = dn_meta['num_matching_queries_per_group']
        expected_shape = (
            len(batch_data_samples), groups * queries_per_group,
            final_cls.shape[-1])
        if final_cls.shape != expected_shape:
            raise ValueError(
                'final class scores do not match matching query geometry')
        final_cls = final_cls.reshape(
            len(batch_data_samples), groups, queries_per_group,
            final_cls.shape[-1])
        selected_batch = []
        with torch.no_grad():
            for image_index, sample in enumerate(batch_data_samples):
                positive_map = getattr(sample, 'token_positive_map', None)
                selected_groups = []
                for group_index in range(groups):
                    group_cls = final_cls[image_index, group_index]
                    if positive_map is None:
                        class_log_scores = F.logsigmoid(
                            group_cls.float()).amax(dim=-1, keepdim=True)
                    else:
                        class_log_scores = grounding_logits_to_class_log_scores(
                            group_cls, positive_map)
                    selected, _ = select_from_class_log_scores(
                        class_log_scores)
                    selected_groups.append(selected)
                selected_batch.append(torch.stack(selected_groups, dim=0))
        return torch.stack(selected_batch, dim=0).detach()

    def get_targets(self, cls_scores_list, bbox_preds_list,
                    batch_gt_instances, batch_img_metas):
        targets = super().get_targets(
            cls_scores_list, bbox_preds_list, batch_gt_instances,
            batch_img_metas)
        if (self._capture_matching_targets and
                not self._suppress_matching_target_capture):
            bbox_weights_list = targets[3]
            masks = []
            for weights in bbox_weights_list:
                if weights.ndim != 2 or weights.shape[-1] != 5:
                    raise ValueError(
                        'matching bbox weights must have shape [queries,5]')
                masks.append(weights.any(dim=-1))
            captured = torch.stack(masks, dim=0).to(dtype=torch.bool)
            self._captured_matching_mask = captured.detach()
        return targets

    def _get_targets_single(self, cls_score, bbox_pred, gt_instances,
                            img_meta):
        parent_targets = super()._get_targets_single(
            cls_score, bbox_pred, gt_instances, img_meta)
        if not self.position_supervised_enabled:
            return parent_targets
        (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
         neg_inds) = parent_targets
        if pos_inds.numel() == 0:
            return parent_targets
        labels = scale_positive_maps_by_rotated_iou(
            labels, bbox_pred, bbox_targets, pos_inds, img_meta,
            self.angle_factor)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def _matching_loss_by_feat(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore=None):
        if not self._existence_enabled:
            return super(DeformableDETRHead, self).loss_by_feat(
                cls_scores,
                bbox_preds,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore)
        if self._capture_matching_targets:
            raise RuntimeError('matching target capture cannot be nested')
        self._captured_matching_mask = None
        self._capture_matching_targets = True
        try:
            losses = super(DeformableDETRHead, self).loss_by_feat(
                cls_scores,
                bbox_preds,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore)
        finally:
            self._capture_matching_targets = False
        if self._captured_matching_mask is None:
            raise RuntimeError('parent matching loss produced no assignments')
        self._loss_matching_group_masks.append(
            self._captured_matching_mask.detach())
        return losses

    def _denoising_loss_dict(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, dn_meta):
        dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
            cls_scores,
            bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)
        losses = {
            'dn_loss_cls': dn_losses_cls[-1],
            'dn_loss_bbox': dn_losses_bbox[-1],
            'dn_loss_iou': dn_losses_iou[-1],
        }
        for layer, (loss_cls, loss_bbox, loss_iou) in enumerate(zip(
                dn_losses_cls[:-1], dn_losses_bbox[:-1],
                dn_losses_iou[:-1])):
            losses[f'd{layer}.dn_loss_cls'] = loss_cls
            losses[f'd{layer}.dn_loss_bbox'] = loss_bbox
            losses[f'd{layer}.dn_loss_iou'] = loss_iou
        return losses

    def _adaptive_denoising_loss_dict(
            self, cls_scores, bbox_preds, matching_cls_scores,
            matching_bbox_preds, batch_gt_instances, batch_img_metas,
            dn_meta):
        dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.adaptive_loss_dn(
            cls_scores,
            bbox_preds,
            matching_cls_scores,
            matching_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)
        losses = {
            'dn_loss_cls': dn_losses_cls[-1],
            'dn_loss_bbox': dn_losses_bbox[-1],
            'dn_loss_iou': dn_losses_iou[-1],
        }
        for layer, (loss_cls, loss_bbox, loss_iou) in enumerate(zip(
                dn_losses_cls[:-1], dn_losses_bbox[:-1],
                dn_losses_iou[:-1])):
            losses[f'd{layer}.dn_loss_cls'] = loss_cls
            losses[f'd{layer}.dn_loss_bbox'] = loss_bbox
            losses[f'd{layer}.dn_loss_iou'] = loss_iou
        return losses

    def loss_by_feat(
            self, all_layers_cls_scores, all_layers_bbox_preds,
            enc_cls_scores, enc_bbox_preds, batch_gt_instances,
            batch_img_metas, dn_meta,
            batch_gt_instances_ignore=None):
        """Match each training query group independently and average losses."""
        if (getattr(self, 'position_supervised_enabled', False) and
                (enc_cls_scores is not None or enc_bbox_preds is not None)):
            raise ValueError(
                'position supervision excludes encoder output supervision')
        if self._existence_enabled:
            self._loss_matching_group_masks = []
            self.last_matching_group_masks = None
            if not isinstance(dn_meta, dict):
                raise ValueError('D13-N training requires dn_meta')
            metadata_groups = dn_meta.get('num_matching_query_groups')
            queries_per_group = dn_meta.get(
                'num_matching_queries_per_group')
            if (type(metadata_groups) is not int or metadata_groups != 3 or
                    type(queries_per_group) is not int or
                    queries_per_group != 600):
                raise ValueError(
                    'D13-N matching query geometry must be exactly 3x600')
        if (self.matching_query_groups == 1 and
                not self._existence_enabled and
                not getattr(self, 'adaptive_dn_enabled', False)):
            return super().loss_by_feat(
                all_layers_cls_scores,
                all_layers_bbox_preds,
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances,
                batch_img_metas,
                dn_meta,
                batch_gt_instances_ignore)
        if dn_meta is None:
            raise ValueError('custom matching requires denoising metadata')
        metadata_groups = int(dn_meta.get('num_matching_query_groups', 1))
        queries_per_group = int(
            dn_meta.get('num_matching_queries_per_group', 0))
        if metadata_groups != self.matching_query_groups:
            raise ValueError(
                'head and detector matching query groups do not agree')
        if queries_per_group < 1:
            raise ValueError('missing matching queries per group metadata')

        (matching_cls, matching_bbox, denoising_cls,
         denoising_bbox) = self.split_outputs(
             all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        if self.matching_query_groups == 1:
            matching_losses = self._matching_loss_by_feat(
                matching_cls,
                matching_bbox,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore)
        else:
            matching_losses = grouped_matching_losses(
                matching_cls,
                matching_bbox,
                queries_per_group=queries_per_group,
                groups=self.matching_query_groups,
                loss_fn=lambda group_cls, group_bbox:
                self._matching_loss_by_feat(
                    group_cls,
                    group_bbox,
                    batch_gt_instances,
                    batch_img_metas,
                    batch_gt_instances_ignore))

        if self._existence_enabled:
            if len(self._loss_matching_group_masks) != \
                    self.matching_query_groups:
                raise RuntimeError(
                    'actual assignment capture count does not match groups')
            self.last_matching_group_masks = torch.stack(
                self._loss_matching_group_masks, dim=1).detach()
            self.last_matching_mask = self.last_matching_group_masks[:, 0]

        if enc_cls_scores is not None:
            enc_loss_cls, enc_loss_bbox, enc_loss_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores,
                    enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            matching_losses.update(
                enc_loss_cls=enc_loss_cls,
                enc_loss_bbox=enc_loss_bbox,
                enc_loss_iou=enc_loss_iou)
        if denoising_cls is not None:
            if getattr(self, 'adaptive_dn_enabled', False):
                primary_cls = split_matching_groups(
                    matching_cls, queries_per_group,
                    self.matching_query_groups)[0]
                primary_bbox = split_matching_groups(
                    matching_bbox, queries_per_group,
                    self.matching_query_groups)[0]
                matching_losses.update(self._adaptive_denoising_loss_dict(
                    denoising_cls,
                    denoising_bbox,
                    primary_cls,
                    primary_bbox,
                    batch_gt_instances,
                    batch_img_metas,
                    dn_meta))
            else:
                matching_losses.update(self._denoising_loss_dict(
                    denoising_cls,
                    denoising_bbox,
                    batch_gt_instances,
                    batch_img_metas,
                    dn_meta))
        return matching_losses

    def loss_by_feat_single(self, cls_scores, bbox_preds,
                            batch_gt_instances, batch_img_metas):
        parent_cls, loss_bbox, loss_iou = super().loss_by_feat_single(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)
        if not self.balanced_cfg.get('enabled', False):
            self.last_balance_stats = None
            return parent_cls, loss_bbox, loss_iou

        cls_scores_list = [cls_scores[i] for i in range(cls_scores.size(0))]
        bbox_preds_list = [bbox_preds[i] for i in range(bbox_preds.size(0))]
        previous_suppression = self._suppress_matching_target_capture
        self._suppress_matching_target_capture = True
        try:
            with torch.no_grad():
                target_data = self.get_targets(
                    cls_scores_list, bbox_preds_list,
                    batch_gt_instances, batch_img_metas)
        finally:
            self._suppress_matching_target_capture = previous_suppression
        labels_list, label_weights_list, _, bbox_weights_list, _, _ = \
            target_data
        labels = torch.stack(labels_list, dim=0)
        query_weights = torch.stack(label_weights_list, dim=0)
        matched_query_mask = torch.stack(
            [weights.any(dim=-1) for weights in bbox_weights_list])

        padded_text_mask = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        padded_text_mask[:, :self.text_masks.size(1)] = self.text_masks
        valid_token_mask = (padded_text_mask > 0).unsqueeze(1).expand_as(labels)
        loss_cls, stats = balanced_group_classification_loss(
            loss_module=self.loss_cls,
            cls_scores=cls_scores,
            labels=labels,
            query_weights=query_weights,
            valid_token_mask=valid_token_mask,
            matched_query_mask=matched_query_mask,
            matched_weight=float(
                self.balanced_cfg.get('matched_weight', 1.0)),
            unmatched_weight=float(
                self.balanced_cfg.get('unmatched_weight', 1.0)))
        self.last_balance_stats = stats
        return loss_cls, loss_bbox, loss_iou

    def loss(self,
             hidden_states,
             references,
             memory_text,
             text_token_mask,
             enc_outputs_class,
             enc_outputs_coord,
             batch_data_samples,
             dn_meta,
             null_logits=None,
             capacity=None,
             semantic_gate=None):
        """Compute upstream losses and cache final matching assignments."""
        batch_gt_instances = [
            sample.gt_instances for sample in batch_data_samples]
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        losses = self.loss_by_feat(
            *outs,
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances,
            batch_img_metas,
            dn_meta)

        if self._existence_enabled:
            if not isinstance(dn_meta, dict):
                raise ValueError(
                    'existence training requires denoising metadata')
            groups = dn_meta.get('num_matching_query_groups')
            queries_per_group = dn_meta.get(
                'num_matching_queries_per_group')
            num_dn = dn_meta.get('num_denoising_queries')
            if (type(groups) is not int or type(queries_per_group) is not int
                    or type(num_dn) is not int or groups < 1 or
                    queries_per_group < 1 or num_dn < 0):
                raise ValueError(
                    'matching query geometry is missing from dn_meta')
            if groups != self.matching_query_groups:
                raise ValueError(
                    'head and metadata matching groups do not agree')
            if groups != 3 or queries_per_group != 600:
                raise ValueError(
                    'D13-N matching query geometry must be exactly 3x600')
            final_hidden = hidden_states[-1]
            expected_queries = groups * queries_per_group
            if expected_queries != 1800:
                raise ValueError(
                    'D13-N training requires exactly 1800 matching queries')
            if (final_hidden.ndim != 3 or
                    final_hidden.shape[1] != num_dn + expected_queries):
                raise ValueError(
                    'hidden matching query axis does not match dn_meta')
            if final_hidden.shape[-1] != self.existence_residual.in_features:
                raise ValueError(
                    'hidden embed dimension does not match existence head')
            matching_hidden = final_hidden[:, -expected_queries:, :]
            matching_hidden = matching_hidden.reshape(
                final_hidden.shape[0], groups, queries_per_group,
                self.existence_residual.in_features)
            existence_logits = self.existence_residual(
                matching_hidden.detach()).squeeze(-1)
            masks = self.last_matching_group_masks
            if (masks is None or masks.shape != existence_logits.shape or
                    masks.dtype != torch.bool):
                raise RuntimeError(
                    'actual matching masks do not match existence logits')
            residual = centered_existence_log_residual(existence_logits)
            raw_loss = grouped_existence_bce_loss(existence_logits, masks)
            if self.existence_loss_weight == 0.0:
                losses['loss_existence'] = (
                    self.existence_residual.weight.sum() * 0.0 +
                    self.existence_residual.bias.sum() * 0.0)
            else:
                losses['loss_existence'] = raw_loss
            selected_parent_log_scores = self._selected_training_log_scores(
                outs[0], outs[1], dn_meta, batch_data_samples)
            self._cache_existence_training_state(
                existence_logits, residual, masks,
                selected_parent_log_scores=selected_parent_log_scores)

        if not self._existence_enabled:
            matching_cls, matching_box, _, _ = self.split_outputs(
                outs[0], outs[1], dn_meta)
            final_cls = matching_cls[-1]
            final_box = matching_box[-1]
            if self.matching_query_groups > 1:
                queries_per_group = int(
                    dn_meta['num_matching_queries_per_group'])
                final_cls = split_matching_groups(
                    final_cls, queries_per_group,
                    self.matching_query_groups)[0]
                final_box = split_matching_groups(
                    final_box, queries_per_group,
                    self.matching_query_groups)[0]
            with torch.no_grad():
                target_data = self.get_targets(
                    [final_cls[i] for i in range(final_cls.size(0))],
                    [final_box[i] for i in range(final_box.size(0))],
                    batch_gt_instances,
                    batch_img_metas)
                bbox_weights_list = target_data[3]
                self.last_matching_mask = torch.stack(
                    [weights.any(dim=-1) for weights in bbox_weights_list])
        return losses

    def predict(self,
                hidden_states,
                references,
                memory_text,
                text_token_mask,
                batch_data_samples,
                rescale=True,
                null_logits=None,
                capacity=None,
                semantic_gate=None):
        """Predict one calibrated result for every matching query."""
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        positive_maps = [sample.token_positive_map
                         for sample in batch_data_samples]
        cls_scores, bbox_preds = self(
            hidden_states, references, memory_text, text_token_mask)
        cls_scores = cls_scores[-1]
        bbox_preds = bbox_preds[-1]
        existence_residual = None
        if self._existence_enabled:
            num_queries = 600
            final_hidden = hidden_states[-1]
            if (cls_scores.ndim != 3 or bbox_preds.ndim != 3 or
                    cls_scores.shape[1] != num_queries or
                    bbox_preds.shape[1] != num_queries):
                raise ValueError('D13-N inference requires exactly 600 rows')
            if (final_hidden.ndim != 3 or
                    final_hidden.shape[0] != cls_scores.shape[0] or
                    final_hidden.shape[1] < num_queries or
                    final_hidden.shape[2] !=
                    self.existence_residual.in_features):
                raise ValueError(
                    'inference hidden states do not contain the 600-row tail')
            matching_hidden = final_hidden[:, -num_queries:, :]
            existence_logits = self.existence_residual(
                matching_hidden.detach()).squeeze(-1)
            existence_residual = centered_existence_log_residual(
                existence_logits)
            self.last_existence_logits = existence_logits.detach()
            self.last_existence_residual = existence_residual.detach()
            self.last_existence_logit_quantiles = self._group_quantiles(
                existence_logits.unsqueeze(1))
            self.last_existence_residual_quantiles = self._group_quantiles(
                existence_residual.unsqueeze(1))
            self.last_existence_clamp_hit_counts = existence_logits.new_zeros(
                2, dtype=torch.long)
        results = []
        for index, (cls_score, bbox_pred, positive_map, img_meta) in enumerate(
                zip(cls_scores, bbox_preds, positive_maps, batch_img_metas)):
            image_null = None if null_logits is None else null_logits[index]
            image_capacity = None if capacity is None else capacity[index]
            image_residual = (None if existence_residual is None else
                              existence_residual[index])
            results.append(self._predict_by_feat_single(
                cls_score,
                bbox_pred,
                positive_map,
                img_meta,
                rescale=rescale,
                null_logits=image_null,
                capacity=image_capacity,
                existence_log_residual=image_residual))
        return results

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: Dict,
                                img_meta: dict,
                                rescale: bool = True,
                                null_logits: Tensor = None,
                                capacity: Tensor = None,
                                existence_log_residual: Tensor = None
                                ) -> InstanceData:
        num_queries = cls_score.shape[0]
        if bbox_pred.shape != (num_queries, 5):
            raise ValueError('bbox_pred must contain one 5-D box per query')
        if null_logits is not None and null_logits.shape != (num_queries, ):
            raise ValueError('null_logits must have one value per query')
        if capacity is not None and capacity.shape != (num_queries, ):
            raise ValueError('capacity must have one value per query')
        if (existence_log_residual is not None and
                existence_log_residual.shape != (num_queries, )):
            raise ValueError(
                'existence_log_residual must have one value per query')

        if token_positive_maps is not None:
            class_log_scores = grounding_logits_to_class_log_scores(
                cls_score, token_positive_maps)
        else:
            class_log_scores = F.logsigmoid(cls_score.float()).amax(
                dim=-1, keepdim=True)
        selected_log_scores, det_labels = select_from_class_log_scores(
            class_log_scores)
        if existence_log_residual is not None:
            preclamp = selected_log_scores.float() * (
                float(self.readout_cfg.get('power', 1.0)) /
                float(self.readout_cfg.get('temperature', 1.0)))
            if null_logits is not None:
                preclamp = preclamp + F.logsigmoid(-null_logits.float())
            if (capacity is not None and
                    self.readout_cfg.get('use_capacity', False)):
                preclamp = preclamp + capacity.float().clamp(
                    min=1e-8).log()
            preclamp = preclamp + existence_log_residual.float()
            clamp_hits = torch.stack([
                (preclamp < -80.0).sum(), (preclamp > 0.0).sum()
            ]).detach()
            if self.last_existence_clamp_hit_counts is None:
                self.last_existence_clamp_hit_counts = clamp_hits
            else:
                self.last_existence_clamp_hit_counts = (
                    self.last_existence_clamp_hit_counts + clamp_hits).detach()
        scores = calibrate_selected_log_scores(
            selected_log_scores,
            null_logits=null_logits,
            capacity=(capacity if self.readout_cfg.get('use_capacity', False)
                      else None),
            temperature=float(self.readout_cfg.get('temperature', 1.0)),
            power=float(self.readout_cfg.get('power', 1.0)),
            log_residual=existence_log_residual)

        # Do not mutate decoder outputs retained for auxiliary losses/debugging.
        det_bboxes = bbox_pred.clone()
        img_h, img_w = img_meta['img_shape'][:2]
        det_bboxes[:, 0:4:2] *= img_w
        det_bboxes[:, 1:4:2] *= img_h
        det_bboxes[:, 4] *= self.angle_factor
        det_bboxes[:, 0:4:2].clamp_(min=0, max=img_w)
        det_bboxes[:, 1:4:2].clamp_(min=0, max=img_h)

        if rescale:
            scale_factor = det_bboxes.new_tensor(img_meta['scale_factor'])
            if scale_factor.numel() == 2:
                scale_factor = scale_factor.repeat(2)
            if scale_factor.numel() != 4:
                raise ValueError(
                    'scale_factor must contain two or four values')
            scale_factor = torch.cat([scale_factor, scale_factor.new_ones(1)])
            det_bboxes /= scale_factor

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results
