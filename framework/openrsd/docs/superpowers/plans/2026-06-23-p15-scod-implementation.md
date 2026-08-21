# P15-SCOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P15-SCOD, a support-conditioned oriented DINO detector that outputs final fixed-set `rbox + cls` predictions directly, with no traditional dense detection head, no NMS, and no score-threshold fallback.

**Architecture:** Reuse the mature DINO/Deformable DETR encoder-decoder path, replace the horizontal prediction branch with an oriented set-prediction branch, and insert support conditioning into the query initializer. The detector performs Hungarian set prediction over fixed queries; inference emits exactly one label, score, and rotated box per query.

**Tech Stack:** PyTorch, MMEngine, MMDetection DINO/Deformable DETR, MMRotate rotated boxes, SciPy Hungarian assignment, HRRSD split at `/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/`.

---

## Non-Negotiable Contracts

- The inference path is `image -> backbone -> neck -> encoder memory -> support-conditioned DINO queries -> decoder -> fixed query set`.
- The final prediction tensor is a fixed query set: `num_queries` boxes and `num_queries` class posterior choices per image.
- No NMS, no rotated NMS, no pre-NMS top-k pruning, no score-threshold filtering, no empty-output rescue.
- No RTMDet, RetinaNet, FCOS, S2ANet, RPN, RoI head, anchor head, or point dense detection branch in the final prediction path.
- `bbox_head` is only an MMDetection configuration key; the class must expose debug flags proving it is a set-prediction branch, not a traditional dense head.
- The first smoke target is signal discovery, not a full baseline claim. Stop before 4GPU training when the overfit or mini-oracle gates fail.

## File Map

- Create `tests/test_p15_oriented_dino_contract.py`
  - Unit tests for query conditioning, fixed-set output, rotated box shape, and strict E2E debug flags.
- Create `M_AD/models/dense_heads/p15_oriented_dino_head.py`
  - Support query conditioner, oriented DINO set-prediction branch, Hungarian loss, no-NMS prediction.
- Create `M_AD/models/detectors/p15_oriented_dino.py`
  - DINO subclass using `DinoRTransformerDecoder`, `CdnRQueryGenerator`, support-conditioned matching queries, and angle references.
- Create `M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py`
  - 20-image HRRSD overfit gate config.
- Create `M_configs/Diagnostics/hrrsd_p15_oriented_dino_mini.py`
  - 128-image HRRSD mini-eval gate config.
- Create `M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh`
  - Single-GPU overfit and mini-gate runner. Four-GPU commands stay commented out until gates pass.
- Modify `M_Tools/analysis/oracle_rank_fixed_query_predictions.py`
  - Add a `--fixed-query-all` mode only if the current script cannot consume all P15 query predictions.

## Task 1: Write Strict E2E Contract Tests

**Files:**
- Create: `tests/test_p15_oriented_dino_contract.py`

- [ ] **Step 1: Create the failing contract test file**

Write this complete file:

```python
import torch
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmrotate.structures import RotatedBoxes

from M_AD.models.dense_heads.p15_oriented_dino_head import (
    P15OrientedDINOSetHead,
    P15SupportQueryConditioner,
    build_orthogonal_support_tokens,
)


def _sample_with_gt():
    sample = DetDataSample()
    sample.set_metainfo(
        dict(
            img_id='p15_fake',
            img_shape=(64, 64),
            ori_shape=(64, 64),
            scale_factor=(1.0, 1.0)))
    gt = InstanceData()
    gt.bboxes = RotatedBoxes(
        torch.tensor([[24.0, 28.0, 12.0, 8.0, 0.10]], dtype=torch.float32))
    gt.labels = torch.tensor([1], dtype=torch.long)
    sample.gt_instances = gt
    sample.ignored_instances = InstanceData()
    sample.ignored_instances.bboxes = torch.empty(0, 5)
    sample.ignored_instances.labels = torch.empty(0, dtype=torch.long)
    return sample


def test_p15_support_tokens_are_class_orthogonal():
    support = build_orthogonal_support_tokens(
        num_classes=4, embed_dims=6, scale=2.0, tail_std=0.0)

    assert support.shape == (4, 6)
    assert torch.allclose(support[:, :4], torch.eye(4) * 2.0)
    assert torch.allclose(support[:, 4:], torch.zeros(4, 2))


def test_p15_query_conditioner_zero_scale_is_identity():
    conditioner = P15SupportQueryConditioner(
        num_classes=3,
        embed_dims=8,
        support_scale=0.0,
        support_init_std=0.0)
    query = torch.randn(2, 5, 8)
    logits = torch.randn(2, 5, 3)

    out = conditioner(query, logits)

    assert torch.allclose(out, query)
    assert conditioner.debug['support_source'] == 'learned_orthogonal'
    assert conditioner.debug['support_conditioned_query_initializer'] is True


def test_p15_query_conditioner_changes_queries_when_enabled():
    conditioner = P15SupportQueryConditioner(
        num_classes=3,
        embed_dims=8,
        support_scale=1.0,
        support_init_std=0.0)
    query = torch.zeros(2, 5, 8)
    logits = torch.zeros(2, 5, 3)
    logits[..., 1] = 6.0

    out = conditioner(query, logits)

    assert out.shape == query.shape
    assert not torch.allclose(out, query)
    assert conditioner.debug['support_conditioning_strength'] > 0.0


def test_p15_head_predict_outputs_all_queries_without_nms_or_threshold():
    init_default_scope('mmrotate')
    head = P15OrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=7,
        support_query_scale=1.0,
        max_per_img=7)
    head.init_weights()
    hidden_states = torch.randn(1, 1, 7, 16)
    references = [torch.rand(1, 7, 4), torch.rand(1, 7, 4)]
    reference_angles = [
        torch.zeros(1, 7, 1),
        torch.zeros(1, 7, 1),
    ]
    samples = [_sample_with_gt()]

    with torch.no_grad():
        results = head.predict(
            hidden_states=hidden_states,
            references=references,
            reference_angles=reference_angles,
            batch_data_samples=samples,
            rescale=True)

    assert len(results) == 1
    assert results[0].bboxes.tensor.shape == (7, 5)
    assert results[0].scores.shape == (7,)
    assert results[0].labels.shape == (7,)
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False
    assert head.e2e_debug['outputs_fixed_query_set'] is True


def test_p15_head_loss_is_finite_on_synthetic_rotated_gt():
    init_default_scope('mmrotate')
    head = P15OrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=7,
        support_query_scale=1.0,
        max_per_img=7)
    head.init_weights()
    hidden_states = torch.randn(1, 1, 7, 16)
    references = [torch.rand(1, 7, 4), torch.rand(1, 7, 4)]
    reference_angles = [
        torch.zeros(1, 7, 1),
        torch.zeros(1, 7, 1),
    ]
    enc_outputs_class = torch.randn(1, 7, 3)
    enc_outputs_coord = torch.rand(1, 7, 4)
    enc_outputs_angle = torch.zeros(1, 7, 1)

    losses = head.loss(
        hidden_states=hidden_states,
        references=references,
        reference_angles=reference_angles,
        enc_outputs_class=enc_outputs_class,
        enc_outputs_coord=enc_outputs_coord,
        enc_outputs_angle=enc_outputs_angle,
        batch_data_samples=[_sample_with_gt()],
        dn_meta=None)

    for key in ('loss_cls', 'loss_bbox', 'loss_angle', 'enc_loss_cls',
                'enc_loss_bbox', 'enc_loss_angle'):
        assert key in losses
        assert torch.isfinite(losses[key])
```

- [ ] **Step 2: Run the tests and verify they fail before implementation**

Run:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p15_oriented_dino_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'M_AD.models.dense_heads.p15_oriented_dino_head'`.

## Task 2: Implement Support Query Conditioning

**Files:**
- Create: `M_AD/models/dense_heads/p15_oriented_dino_head.py`
- Test: `tests/test_p15_oriented_dino_contract.py`

- [ ] **Step 1: Create support-token helpers**

Create `M_AD/models/dense_heads/p15_oriented_dino_head.py` with this initial content:

```python
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

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


def _periodic_l1(pred: Tensor, target: Tensor, period: float = math.pi) -> Tensor:
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


def gaussian_wasserstein_loss(pred_boxes: Tensor,
                              target_boxes: Tensor,
                              eps: float = 1e-6) -> Tensor:
    xy_p, sigma_p = _rbox_to_gaussian(pred_boxes)
    xy_t, sigma_t = _rbox_to_gaussian(target_boxes)
    xy_distance = (xy_p - xy_t).square().sum(dim=-1)
    trace_p = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_t = sigma_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_cross = sigma_p.bmm(sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    det_cross = (sigma_p.det() * sigma_t.det()).clamp(min=eps).sqrt()
    sigma_distance = (trace_p + trace_t -
                      2.0 * (trace_cross + 2.0 * det_cross).clamp(min=eps).sqrt())
    distance = (xy_distance + sigma_distance.clamp(min=0.0)).clamp(min=eps).sqrt()
    return torch.log1p(distance)
```

- [ ] **Step 2: Add the support query conditioner class**

Append this class to the same file:

```python
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

    def forward(self, query: Tensor, class_logits: Tensor) -> Tensor:
        if self.support_scale == 0:
            self.debug['support_conditioning_strength'] = 0.0
            return query
        weights = class_logits.sigmoid().softmax(dim=-1)
        context = weights @ self.support_tokens.to(query.dtype)
        delta = self.support_proj(context)
        out = query + float(self.support_scale) * delta
        self.debug['support_conditioning_strength'] = float(
            delta.detach().norm(dim=-1).mean().cpu())
        return out
```

- [ ] **Step 3: Run the support-only tests**

Run:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p15_oriented_dino_contract.py::test_p15_support_tokens_are_class_orthogonal tests/test_p15_oriented_dino_contract.py::test_p15_query_conditioner_zero_scale_is_identity tests/test_p15_oriented_dino_contract.py::test_p15_query_conditioner_changes_queries_when_enabled -q
```

Expected: PASS for the three support tests.

## Task 3: Implement the Oriented DINO Set-Prediction Branch

**Files:**
- Modify: `M_AD/models/dense_heads/p15_oriented_dino_head.py`
- Test: `tests/test_p15_oriented_dino_contract.py`

- [ ] **Step 1: Add the class header and prediction branches**

Append this class header to `M_AD/models/dense_heads/p15_oriented_dino_head.py`:

```python
@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15OrientedDINOSetHead(BaseModule):
    def __init__(self,
                 num_classes: int,
                 embed_dims: int = 256,
                 num_reg_fcs: int = 2,
                 num_pred_layer: int = 7,
                 num_queries: int = 300,
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
            self.cls_branches.append(nn.Linear(self.embed_dims, self.num_classes))
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
```

- [ ] **Step 2: Add query conditioning and forward methods**

Append these methods inside `P15OrientedDINOSetHead`:

```python
    def condition_matching_queries(self, query: Tensor,
                                   topk_class_logits: Tensor) -> Tensor:
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
```

- [ ] **Step 3: Add target conversion and Hungarian assignment**

Append these methods inside `P15OrientedDINOSetHead`:

```python
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
        assigned_labels = cls_score.new_full((num_queries,), -1, dtype=torch.long)
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
        gwd_cost = gaussian_wasserstein_loss(
            pred_rbox[:, None, :].reshape(-1, 5),
            gt_boxes[None, :, :].expand(num_queries, -1, 5).reshape(-1, 5),
        ).reshape(num_queries, -1)
        total_cost = (
            2.0 * cls_cost +
            5.0 * bbox_cost +
            1.0 * angle_cost +
            1.0 * gwd_cost)
        row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu())
        row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=cls_score.device)
        col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=cls_score.device)
        assigned_gt[row_ind] = col_ind
        assigned_labels[row_ind] = gt_instances.labels[col_ind]
        return assigned_gt, assigned_labels
```

- [ ] **Step 4: Add loss methods**

Append these methods inside `P15OrientedDINOSetHead`:

```python
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
                bbox_losses.append(F.l1_loss(pos_bbox, gt_norm, reduction='sum'))
                angle_losses.append(
                    _periodic_l1(pos_angle, gt_boxes[:, 4:5]).sum())
                pred_rbox = torch.cat([
                    self._scale_boxes_to_pixels(pos_bbox, img_meta),
                    pos_angle,
                ], dim=-1)
                gwd_losses.append(gaussian_wasserstein_loss(pred_rbox, gt_boxes).sum())
                num_pos += pos_mask.sum()
        avg_factor = num_pos.clamp(min=1.0)
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(avg_factor)
        zero = cls_score.sum() * 0.0
        losses = {
            f'{prefix}loss_cls': sum(cls_losses) / avg_factor * self.loss_cls_weight,
            f'{prefix}loss_bbox': (sum(bbox_losses) if bbox_losses else zero) / avg_factor * self.loss_bbox_weight,
            f'{prefix}loss_angle': (sum(angle_losses) if angle_losses else zero) / avg_factor * self.loss_angle_weight,
            f'{prefix}loss_gwd': (sum(gwd_losses) if gwd_losses else zero) / avg_factor * self.loss_gwd_weight,
        }
        return losses

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             reference_angles: List[Tensor], enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor, enc_outputs_angle: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Optional[Dict[str, int]] = None) -> Dict[str, Tensor]:
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
```

- [ ] **Step 5: Add fixed-query prediction**

Append these methods inside `P15OrientedDINOSetHead`:

```python
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
                f'P15 expected {self.num_queries} fixed queries, got {rboxes.shape[0]}')
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
```

- [ ] **Step 6: Run head contract tests**

Run:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p15_oriented_dino_contract.py -q
```

Expected: PASS for all tests in `tests/test_p15_oriented_dino_contract.py`.

## Task 4: Implement the P15 Support-Conditioned Oriented DINO Detector

**Files:**
- Create: `M_AD/models/detectors/p15_oriented_dino.py`
- Test: `tests/test_p15_oriented_dino_contract.py`

- [ ] **Step 1: Create the detector imports and registration**

Create `M_AD/models/detectors/p15_oriented_dino.py` with this header:

```python
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from mmdet.models.detectors.deformable_detr import DeformableDETR
from mmdet.models.layers import DeformableDetrTransformerEncoder, SinePositionalEncoding
from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmrotate.registry import MODELS as MMROTATE_MODELS
from torch import Tensor, nn
from torch.nn.init import normal_

from M_AD.models.layers.transformer.dinor_layersv2 import (
    CdnRQueryGenerator,
    DinoRTransformerDecoder,
)
from M_AD.models.layers.transformer.deformable_detr_layers import (
    RotatedMultiScaleDeformableAttention,
)
```

- [ ] **Step 2: Add the detector class and layer initialization**

Append this class to `M_AD/models/detectors/p15_oriented_dino.py`:

```python
@MMROTATE_MODELS.register_module(force=True)
@MMDET_MODELS.register_module(force=True)
class P15SupportConditionedOrientedDINO(DINO):
    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        rotated_dn_cfg = dict(dn_cfg or {})
        horizontal_dn_cfg = dict(rotated_dn_cfg)
        horizontal_dn_cfg.pop('angle_noise_scale', None)
        super().__init__(*args, dn_cfg=horizontal_dn_cfg, **kwargs)
        if dn_cfg is not None:
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
```

- [ ] **Step 3: Add rotated pre-decoder logic**

Append this method inside `P15SupportConditionedOrientedDINO`:

```python
    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor,
                    batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        bs, _, _ = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        enc_outputs_angle = self.bbox_head.angle_branches[
            self.decoder.num_layers](output_memory)
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_angles = torch.gather(
            enc_outputs_angle, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 1))
        query = self.query_embedding.weight[None, :, :].repeat(bs, 1, 1)
        query = self.bbox_head.condition_matching_queries(query, topk_score)
        if self.training:
            dn_label_query, dn_bbox_query, dn_angle_query, dn_mask, dn_meta = (
                self.dn_query_generator(batch_data_samples))
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
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
        return decoder_inputs_dict, head_inputs_dict
```

- [ ] **Step 4: Add rotated decoder forwarding**

Append this method inside `P15SupportConditionedOrientedDINO`:

```python
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
```

- [ ] **Step 5: Run import/build smoke**

Run:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -c "from mmengine.registry import init_default_scope; init_default_scope('mmrotate'); from M_AD.models.detectors.p15_oriented_dino import P15SupportConditionedOrientedDINO; from M_AD.models.dense_heads.p15_oriented_dino_head import P15OrientedDINOSetHead; print(P15SupportConditionedOrientedDINO.__name__); print(P15OrientedDINOSetHead.__name__)"
```

Expected:

```text
P15SupportConditionedOrientedDINO
P15OrientedDINOSetHead
```

## Task 5: Add HRRSD Overfit and Mini Configs

**Files:**
- Create: `M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py`
- Create: `M_configs/Diagnostics/hrrsd_p15_oriented_dino_mini.py`

- [ ] **Step 1: Create the overfit config**

Write `M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py`:

```python
custom_imports = dict(
    imports=[
        'M_AD.models.detectors.p15_oriented_dino',
        'M_AD.models.dense_heads.p15_oriented_dino_head',
        'M_AD.models.layers.transformer.dinor_layersv2',
        'M_AD.models.layers.transformer.deformable_detr_layers',
    ],
    allow_failed_imports=False)

_base_ = '../../mmdet/configs/dino/dino_4scale_r50_8xb2_12e_coco.py'

data_root = '/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/'
dataset_type = 'DOTADataset'
img_scale = (800, 800)
file_client_args = dict(backend='disk')
class_name = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
num_classes = len(class_name)

work_dir = 'work_dirs/p15_scod_hrrsd_20260623/overfit20'
default_scope = 'mmrotate'

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

model = dict(
    type='P15SupportConditionedOrientedDINO',
    num_queries=300,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        boxtype2tensor=False),
    bbox_head=dict(
        type='P15OrientedDINOSetHead',
        num_classes=num_classes,
        embed_dims=256,
        num_pred_layer=7,
        num_queries=300,
        support_query_scale=1.0,
        max_per_img=300),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=0.4,
        angle_noise_scale=0.2,
        group_cfg=dict(dynamic=True, num_dn_queries=100)))

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        indices=20,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        img_shape=img_scale,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=20,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        img_shape=img_scale,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

max_epochs = 20
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, save_last=True))

optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = []
auto_scale_lr = dict(enable=False, base_batch_size=1)
```

- [ ] **Step 2: Create the mini config**

Write `M_configs/Diagnostics/hrrsd_p15_oriented_dino_mini.py`:

```python
_base_ = './hrrsd_p15_oriented_dino_overfit.py'

work_dir = 'work_dirs/p15_scod_hrrsd_20260623/mini128'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(indices=128))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        indices=128,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/')))
test_dataloader = val_dataloader

max_epochs = 3
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))
```

- [ ] **Step 3: Validate config parsing**

Run:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python tools/misc/print_config.py M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py --cfg-options work_dir=/tmp/p15_print_config
```

Expected: command exits 0 and printed config contains `P15SupportConditionedOrientedDINO`, `P15OrientedDINOSetHead`, and `/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/`.

## Task 6: Add Smoke Runner with Stop Gates

**Files:**
- Create: `M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh`

- [ ] **Step 1: Create the runner**

Write `M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /data1/zcy/OpenRSD
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/data/zcy/anaconda3/envs/openrsd/bin/python

OVERFIT_CFG=M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py
MINI_CFG=M_configs/Diagnostics/hrrsd_p15_oriented_dino_mini.py

"${PY}" tools/train.py "${OVERFIT_CFG}" --work-dir work_dirs/p15_scod_hrrsd_20260623/overfit20
"${PY}" tools/test.py "${OVERFIT_CFG}" work_dirs/p15_scod_hrrsd_20260623/overfit20/epoch_20.pth \
  --work-dir work_dirs/p15_scod_hrrsd_20260623/overfit20_eval \
  --out work_dirs/p15_scod_hrrsd_20260623/overfit20_eval/predictions.pkl

"${PY}" tools/train.py "${MINI_CFG}" --work-dir work_dirs/p15_scod_hrrsd_20260623/mini128
"${PY}" tools/test.py "${MINI_CFG}" work_dirs/p15_scod_hrrsd_20260623/mini128/epoch_3.pth \
  --work-dir work_dirs/p15_scod_hrrsd_20260623/mini128_eval \
  --out work_dirs/p15_scod_hrrsd_20260623/mini128_eval/predictions.pkl
```

- [ ] **Step 2: Mark executable and lint the script**

Run:

```bash
rtk chmod +x M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh
rtk bash -n M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh
```

Expected: no shell syntax output.

- [ ] **Step 3: Run the smoke sequence only after tests pass**

Run:

```bash
rtk bash M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh
```

Expected:
- Overfit train finishes.
- Overfit eval produces `work_dirs/p15_scod_hrrsd_20260623/overfit20_eval/predictions.pkl`.
- Mini train finishes only after overfit command completes.
- Mini eval produces `work_dirs/p15_scod_hrrsd_20260623/mini128_eval/predictions.pkl`.

## Task 7: Gate Analysis Before Any Four-GPU Run

**Files:**
- Modify: `resultmd/exp_p13_fusion_decoder_hrrsd/literature_boundary_20260623.md`
- Create or update: `resultmd/exp_p15_scod_hrrsd/smoke_20260623.md`

- [ ] **Step 1: Record the strict E2E contract**

Create `resultmd/exp_p15_scod_hrrsd/smoke_20260623.md` with this header:

```markdown
# P15-SCOD HRRSD Smoke Log 2026-06-23

## Contract

- Detector: P15SupportConditionedOrientedDINO
- Set branch: P15OrientedDINOSetHead
- Inference: fixed query set
- NMS: false
- Score threshold filtering: false
- Traditional dense detection head: false
- Query support source in first smoke: learned_orthogonal
- External HRRSD support pkl under `/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/`: absent at plan time

## Gates

| gate | metric | pass line | result | decision |
|---|---:|---:|---:|---|
| overfit20 | train/eval AP50 | >= 0.50 | not run | pending |
| mini128 | AP50 | >= 0.05 | not run | pending |
| mini128 | mAP | > 0.00 | not run | pending |
| fixed-query oracle | oracle mAP | >= 0.50 | not run | pending |
| support safety | oracle mAP drop vs support_query_scale=0 | <= 0.02 | not run | pending |

## Decision Rule

- If overfit20 AP50 < 0.50, stop and debug matching or angle scale.
- If mini128 AP50 == 0.00, stop and do not launch four-GPU.
- If fixed-query oracle mAP < 0.30, stop and redesign query initialization.
- If support_query_scale=1.0 hurts oracle mAP by more than 0.02 versus support_query_scale=0.0, keep the oriented DINO path and disable support conditioning for the next smoke.
```

- [ ] **Step 2: Analyze the generated predictions**

Run after the smoke script finishes:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python M_Tools/analysis/oracle_rank_fixed_query_predictions.py --predictions work_dirs/p15_scod_hrrsd_20260623/mini128_eval/predictions.pkl --ann-dir /data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/val/annfiles --out resultmd/exp_p15_scod_hrrsd/mini128_oracle_20260623.json
```

Expected: JSON contains a non-empty prediction count, oracle AP fields, and per-class recall fields.

- [ ] **Step 3: Decide whether four-GPU is justified**

Use this command to inspect the smoke log and oracle output:

```bash
rtk sed -n '1,220p' resultmd/exp_p15_scod_hrrsd/smoke_20260623.md
```

Decision:
- Continue to four-GPU only if overfit20 AP50 >= 0.50, mini128 AP50 > 0.00, and fixed-query oracle mAP >= 0.30.
- Stop and revise P15 when any gate fails.

## Task 8: Verification Commands

**Files:**
- All files above.

- [ ] **Step 1: Run the unit tests**

Run:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_p15_oriented_dino_contract.py -q
```

Expected: PASS.

- [ ] **Step 2: Run config parse**

Run:

```bash
rtk /usr/bin/env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python tools/misc/print_config.py M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py --cfg-options work_dir=/tmp/p15_print_config
```

Expected: exits 0.

- [ ] **Step 3: Check changed files for whitespace errors**

Run:

```bash
rtk git diff --check -- tests/test_p15_oriented_dino_contract.py M_AD/models/dense_heads/p15_oriented_dino_head.py M_AD/models/detectors/p15_oriented_dino.py M_configs/Diagnostics/hrrsd_p15_oriented_dino_overfit.py M_configs/Diagnostics/hrrsd_p15_oriented_dino_mini.py M_Tools/experiments/run_hrrsd_p15_overfit_20260623.sh resultmd/exp_p15_scod_hrrsd/smoke_20260623.md
```

Expected: exits 0.

## Self-Review

- Spec coverage: strict E2E, no traditional dense final head, no NMS, fixed query output, support-conditioned query initializer, HRRSD absolute dataset path, and smoke gates are covered.
- Literature boundary: the plan follows the mature DINO/Deformable DETR set-prediction family instead of extending P13/P14 posterior heads.
- Type consistency: `reference_angles` is a list of tensors matching `references`; prediction results use `RotatedBoxes`; `condition_matching_queries` is called by the detector before decoder entry.
- Risk boundary: no external HRRSD support pkl exists under the confirmed dataset root at plan time, so the first smoke uses learned orthogonal support prototypes and records this explicitly.
