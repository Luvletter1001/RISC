# DOTA-v2 A1 Rotated-IoU Position Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved A1 target-only rotated-IoU position supervision, prove that every non-target path remains unchanged, and leave the immutable 1,600/400 proxy config ready for a separately authorized run.

**Architecture:** Add one pure tensor helper and one default-disabled `_get_targets_single()` override in `OVCapFlowHead`. The override reuses the parent Hungarian assignment, scales only matched positive token maps by detached aligned rotated IoU, and returns all other targets unchanged. A single inherited proxy config enables the feature; focused tests lock the disabled rollback path, DN/encoder/inference exclusions, exact config diff, and zero-update real-batch behavior.

**Tech Stack:** Python 3.8, PyTorch, MMDetection/MMRotate, `mmrotate.structures.bbox.rbbox_overlaps`, MMEngine configs, pytest, Git.

---

## Execution contract

- The approved design is `docs/superpowers/specs/2026-07-22-dotav2-rotated-iou-position-supervision-design.md` at commit `7a2d5d5981dc873934407305c8e8eb977a0e87fb`.
- Use `/data/zcy/anaconda3/envs/mmdet/bin/python` with `PYTHONNOUSERSITE=1` and `PYTHONPATH=/data1/zcy/OV-CapFlow` for every Python/pytest command.
- Prefix every shell command with `rtk`, as required by the repository instructions.
- Preserve all pre-existing untracked files. Stage and commit only the paths named in the current task.
- Do not change matching costs, regression losses, DN/adaptive-DN targets, encoder supervision, inference, Q600, three matching groups, six decoder layers, token aggregation, or the scale-1024 rare4x recipe.
- Do not add a quality floor, exponent, epsilon rescaling, confidence term, new parameter, or new loss type.
- Stage 0 permits builds, unit tests, one real-batch forward/backward, and inference with no optimizer step. It does not permit proxy/full training or checkpoint evaluation.
- Any later GPU command is restricted to physical GPU `2`, `3`, `8`, or `9`. This plan uses physical GPU `2` only after an idle check; if it is occupied, stop and report rather than selecting an unapproved GPU.
- Do not create a full-data A1 config. That artifact is gated on a Stage-1 proxy pass.

### Task 1: Specify and implement the pure rotated-IoU target transform

**Files:**

- Create: `tests/test_projects/ov_capflow/test_position_supervision.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`

- [x] **Step 1: Write the helper tests before the helper exists**

Create the focused test module with imports, a reusable partial-overlap fixture, and tests for the numerical rule, no-positive fast path, perfect/degenerate boxes, non-finite failure, and gradient isolation:

```python
import math

import pytest
import torch
import torch.nn.functional as F

import projects.OVCapFlow.ov_capflow.ov_capflow_head as head_module
from projects.OVCapFlow.ov_capflow.ov_capflow_head import (
    OVCapFlowHead, scale_positive_maps_by_rotated_iou)


def _partial_overlap_inputs():
    labels = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ])
    bbox_pred = torch.tensor([
        [0.5, 0.5, 0.4, 0.4, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.0],
    ], requires_grad=True)
    bbox_targets = torch.tensor([
        [0.6, 0.5, 0.4, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    return labels, bbox_pred, bbox_targets


def test_partial_overlap_scales_original_map_and_preserves_unmatched_rows():
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()

    scaled = scale_positive_maps_by_rotated_iou(
        labels=labels,
        bbox_pred=bbox_pred,
        bbox_targets=bbox_targets,
        pos_inds=torch.tensor([0]),
        img_meta=dict(img_shape=(100, 100)),
        angle_factor=math.pi)

    torch.testing.assert_close(
        scaled, torch.tensor([[0.3, 0.3, 0.0], [0.0, 0.0, 0.0]]),
        rtol=1e-5, atol=1e-6)
    assert scaled[0, 0] / scaled[0, 1] == 1


def test_perfect_and_near_degenerate_overlap_are_finite():
    labels = torch.tensor([[0.25, 0.75]])
    boxes = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])

    scaled = scale_positive_maps_by_rotated_iou(
        labels, boxes, boxes.clone(), torch.tensor([0]),
        dict(img_shape=(100, 100)), math.pi)

    assert torch.isfinite(scaled).all()
    torch.testing.assert_close(scaled, labels)


def test_zero_positive_returns_same_tensor_without_calling_iou(monkeypatch):
    labels = torch.zeros(3, 4)
    monkeypatch.setattr(
        head_module, 'rbbox_overlaps',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('IoU must not run')))

    scaled = scale_positive_maps_by_rotated_iou(
        labels=labels,
        bbox_pred=torch.zeros(3, 5),
        bbox_targets=torch.zeros(3, 5),
        pos_inds=torch.empty(0, dtype=torch.long),
        img_meta=dict(img_shape=(100, 100)),
        angle_factor=math.pi)

    assert scaled is labels


def test_nonfinite_positive_box_fails_instead_of_repairing():
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()
    bbox_pred = bbox_pred.detach()
    bbox_pred[0, 0] = float('nan')

    with pytest.raises(ValueError, match='finite'):
        scale_positive_maps_by_rotated_iou(
            labels, bbox_pred, bbox_targets, torch.tensor([0]),
            dict(img_shape=(100, 100)), math.pi)


def test_nonfinite_iou_output_fails_instead_of_repairing(monkeypatch):
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()
    monkeypatch.setattr(
        head_module, 'rbbox_overlaps',
        lambda *args, **kwargs: torch.tensor([float('nan')]))

    with pytest.raises(ValueError, match='aligned rotated IoU'):
        scale_positive_maps_by_rotated_iou(
            labels, bbox_pred, bbox_targets, torch.tensor([0]),
            dict(img_shape=(100, 100)), math.pi)


def test_quality_target_has_no_gradient_path_to_bbox_prediction():
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()
    target = scale_positive_maps_by_rotated_iou(
        labels, bbox_pred, bbox_targets, torch.tensor([0]),
        dict(img_shape=(100, 100)), math.pi)
    logits = torch.zeros_like(target, requires_grad=True)

    F.binary_cross_entropy_with_logits(logits, target).backward()

    assert logits.grad is not None
    assert bbox_pred.grad is None
    assert target.grad_fn is None
```

- [x] **Step 2: Run the focused module and observe the import failure**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_position_supervision.py -q
```

Expected: collection fails because `scale_positive_maps_by_rotated_iou` does not yet exist. A pass means the test was not written against the intended API; stop and inspect.

- [x] **Step 3: Add the minimal pure helper**

In `ov_capflow_head.py`, import `rbbox_overlaps`:

```python
from mmrotate.structures.bbox import rbbox_overlaps
```

Add this module-level helper immediately before `select_one_class_per_query`:

```python
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
    return scaled_labels
```

Do not add `nan_to_num`, a minimum quality, or any trainable state.

- [x] **Step 4: Run the helper tests to green**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_position_supervision.py -q
```

Expected: `6 passed`.

- [x] **Step 5: Commit only the helper and focused test**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_position_supervision.py
rtk git diff --cached --check
rtk git commit -m "feat: add rotated IoU position targets"
```

Expected: one commit containing only the two named files.

### Task 2: Integrate the default-disabled head override and seal excluded paths

**Files:**

- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `tests/test_projects/ov_capflow/test_position_supervision.py`

- [x] **Step 1: Add failing constructor-contract tests**

Append:

```python
from projects.GroundingDINO.groundingdino.grounding_dino_head import (
    RotatedGroundingDINOHead)


def _construct_head(monkeypatch, **kwargs):
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '__init__',
        lambda self, **parent_kwargs: None)
    return OVCapFlowHead(**kwargs)


def test_position_supervision_is_default_disabled(monkeypatch):
    head = _construct_head(monkeypatch)
    assert head.position_supervised_cfg == {'enabled': False}
    assert head.position_supervised_enabled is False


@pytest.mark.parametrize('bad_cfg', [
    {'enabled': 1},
    {'enabled': False, 'quality_floor': 0.2},
    ['enabled'],
])
def test_position_supervision_rejects_invalid_config(monkeypatch, bad_cfg):
    with pytest.raises((TypeError, ValueError)):
        _construct_head(monkeypatch, position_supervised_cfg=bad_cfg)


def test_position_supervision_rejects_balanced_classification(monkeypatch):
    with pytest.raises(ValueError, match='balanced'):
        _construct_head(
            monkeypatch,
            balanced_cfg=dict(enabled=True),
            position_supervised_cfg=dict(enabled=True))
```

- [x] **Step 2: Add failing target-override and encoder/DN boundary tests**

Append the following. The parent target fixture deliberately returns the same tuple object so disabled identity can be checked, not just numerical equality.

```python
def _parent_targets():
    return (
        torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]),
        torch.ones(2),
        torch.tensor([
            [0.6, 0.5, 0.4, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        torch.tensor([0]),
        torch.tensor([1]),
    )


def test_disabled_target_builder_returns_parent_tuple_unchanged(monkeypatch):
    parent_targets = _parent_targets()
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '_get_targets_single',
        lambda self, *args, **kwargs: parent_targets)
    monkeypatch.setattr(
        head_module, 'scale_positive_maps_by_rotated_iou',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('disabled mode must not run IoU scaling')))
    head = object.__new__(OVCapFlowHead)
    head.position_supervised_enabled = False
    head.angle_factor = math.pi

    result = head._get_targets_single(
        torch.zeros(2, 3), torch.zeros(2, 5), object(),
        dict(img_shape=(100, 100)))

    assert result is parent_targets


def test_enabled_empty_gt_returns_parent_tuple_without_iou(monkeypatch):
    parent_targets = (
        torch.zeros(2, 3),
        torch.ones(2),
        torch.zeros(2, 5),
        torch.zeros(2, 5),
        torch.empty(0, dtype=torch.long),
        torch.tensor([0, 1]),
    )
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '_get_targets_single',
        lambda self, *args, **kwargs: parent_targets)
    monkeypatch.setattr(
        head_module, 'scale_positive_maps_by_rotated_iou',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('empty GT must not run IoU scaling')))
    head = object.__new__(OVCapFlowHead)
    head.position_supervised_enabled = True

    result = head._get_targets_single(
        torch.zeros(2, 3), torch.zeros(2, 5), object(),
        dict(img_shape=(100, 100)))

    assert result is parent_targets
    assert result[0].shape == (2, 3)
    assert torch.isfinite(result[0]).all()


def test_enabled_target_builder_changes_only_positive_labels(monkeypatch):
    parent_targets = _parent_targets()
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '_get_targets_single',
        lambda self, *args, **kwargs: parent_targets)
    head = object.__new__(OVCapFlowHead)
    head.position_supervised_enabled = True
    head.angle_factor = math.pi
    bbox_pred = torch.tensor([
        [0.5, 0.5, 0.4, 0.4, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.0],
    ])

    result = head._get_targets_single(
        torch.zeros(2, 3), bbox_pred, object(),
        dict(img_shape=(100, 100)))

    torch.testing.assert_close(
        result[0], torch.tensor([[0.3, 0.3, 0.0], [0.0, 0.0, 0.0]]),
        rtol=1e-5, atol=1e-6)
    for index in range(1, 6):
        assert result[index] is parent_targets[index]


@pytest.mark.parametrize('enc_cls,enc_bbox', [
    (torch.zeros(1), None),
    (None, torch.zeros(1)),
    (torch.zeros(1), torch.zeros(1)),
])
def test_enabled_position_supervision_rejects_encoder_outputs(
        enc_cls, enc_bbox):
    head = object.__new__(OVCapFlowHead)
    head.position_supervised_enabled = True

    with pytest.raises(ValueError, match='encoder'):
        head.loss_by_feat(
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 5),
            enc_cls,
            enc_bbox,
            [],
            [],
            None)


def test_position_supervision_does_not_override_dn_target_builder():
    assert '_get_dn_targets_single' not in OVCapFlowHead.__dict__


class _RotatedBoxesStub:

    def __init__(self, tensor):
        self.tensor = tensor

    def regularize_boxes(self, **kwargs):
        return self


def test_fixed_dn_loss_route_is_identical_when_a1_flag_changes():
    from mmdet.models.losses import FocalLoss

    head = object.__new__(OVCapFlowHead)
    torch.nn.Module.__init__(head)
    head.angle_cfg = {}
    head.angle_factor = math.pi
    head.max_text_len = 3
    head.text_masks = torch.ones(1, 3, dtype=torch.bool)
    head.bg_cls_weight = 0.1
    head.sync_cls_avg_factor = False
    head.loss_cls = FocalLoss(
        use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
    head.loss_bbox = (
        lambda pred, target, weight, avg_factor: pred.sum() * 0)
    head.loss_iou = (
        lambda pred, target, weight, avg_factor: pred.sum() * 0)
    gt_instances = type('GT', (), {})()
    gt_instances.bboxes = _RotatedBoxesStub(
        torch.tensor([[50.0, 50.0, 40.0, 40.0, 0.0]]))
    gt_instances.labels = torch.tensor([0])
    gt_instances.positive_maps = torch.tensor([[0.5, 0.5, 0.0]])
    dn_meta = dict(num_denoising_queries=4, num_denoising_groups=2)

    logits = torch.tensor([[[
        [0.2, -0.4, 0.1],
        [-0.3, 0.5, -0.2],
        [0.7, -0.1, -0.5],
        [-0.6, 0.4, 0.3],
    ]]])
    boxes = torch.full((1, 1, 4, 5), 0.5)
    boxes[..., 4] = 0.0

    head.position_supervised_enabled = False
    disabled = head._denoising_loss_dict(
        logits, boxes, [gt_instances], [dict(img_shape=(100, 100))],
        dn_meta)
    head.position_supervised_enabled = True
    enabled = head._denoising_loss_dict(
        logits, boxes, [gt_instances], [dict(img_shape=(100, 100))],
        dn_meta)

    assert disabled.keys() == enabled.keys()
    for key in disabled:
        torch.testing.assert_close(disabled[key], enabled[key])


def test_six_layers_and_three_groups_use_the_shared_target_rule(monkeypatch):
    parent_targets = _parent_targets()
    parent_targets[2][0, 0] = 0.5
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '_get_targets_single',
        lambda self, *args, **kwargs: parent_targets)
    seen_labels = []

    def loss_by_feat_single(self, cls_scores, bbox_preds,
                            batch_gt_instances, batch_img_metas):
        targets = self.get_targets(
            [cls_scores[0]], [bbox_preds[0]],
            batch_gt_instances, batch_img_metas)
        seen_labels.append(targets[0][0][0].clone())
        zero = cls_scores.sum() * 0
        return zero, zero, zero

    monkeypatch.setattr(
        OVCapFlowHead, 'loss_by_feat_single', loss_by_feat_single)
    monkeypatch.setattr(
        OVCapFlowHead, '_denoising_loss_dict',
        lambda self, *args, **kwargs: {})
    head = object.__new__(OVCapFlowHead)
    head.position_supervised_enabled = True
    head.adaptive_dn_enabled = False
    head.matching_query_groups = 3
    head.angle_factor = math.pi
    bbox_preds = torch.zeros(6, 1, 6, 5)
    for layer in range(6):
        for group in range(3):
            positive = group * 2
            offset = layer * 3 + group
            bbox_preds[layer, 0, positive] = torch.tensor(
                [0.5 + offset / 100.0, 0.5, 0.4, 0.4, 0.0])
            bbox_preds[layer, 0, positive + 1] = torch.tensor(
                [0.2, 0.2, 0.1, 0.1, 0.0])

    head.loss_by_feat(
        torch.zeros(6, 1, 6, 3),
        bbox_preds,
        None,
        None,
        [object()],
        [dict(img_shape=(100, 100))],
        dict(
            num_denoising_queries=0,
            num_denoising_groups=1,
            num_matching_query_groups=3,
            num_matching_queries_per_group=2))

    assert len(seen_labels) == 18
    expected_qualities = [
        (40.0 - (layer * 3 + group)) /
        (40.0 + (layer * 3 + group))
        for group in range(3)
        for layer in range(6)
    ]
    for labels, quality in zip(seen_labels, expected_qualities):
        torch.testing.assert_close(
            labels, torch.tensor([0.5 * quality, 0.5 * quality, 0.0]),
            rtol=1e-5, atol=1e-6)
```

- [x] **Step 3: Run the focused tests and verify the new tests fail**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_position_supervision.py -q
```

Expected: helper tests pass; constructor/override tests fail because the new configuration and override are absent.

- [x] **Step 4: Validate and store the opt-in configuration in `__init__`**

Change the signature to accept `position_supervised_cfg=None`. Immediately after `self.balanced_cfg` is set, add:

```python
        if position_supervised_cfg is None:
            position_supervised_cfg = {}
        if not isinstance(position_supervised_cfg, dict):
            raise TypeError('position_supervised_cfg must be a dictionary')
        unknown_keys = set(position_supervised_cfg) - {'enabled'}
        if unknown_keys:
            raise ValueError(
                'unknown position_supervised_cfg keys: '
                f'{sorted(unknown_keys)}')
        enabled = position_supervised_cfg.get('enabled', False)
        if not isinstance(enabled, bool):
            raise TypeError('position_supervised_cfg.enabled must be bool')
        self.position_supervised_cfg = {'enabled': enabled}
        self.position_supervised_enabled = enabled
        if (self.position_supervised_enabled and
                self.balanced_cfg.get('enabled', False)):
            raise ValueError(
                'position supervision is incompatible with balanced '
                'classification')
```

Keep default-disabled construction compatible with every existing config.

- [x] **Step 5: Add the single parent-preserving target override**

Add this method after `__init__`:

```python
    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        targets = super()._get_targets_single(
            cls_score, bbox_pred, gt_instances, img_meta)
        if not self.position_supervised_enabled:
            return targets

        (labels, label_weights, bbox_targets, bbox_weights,
         pos_inds, neg_inds) = targets
        if pos_inds.numel() == 0:
            return targets
        labels = scale_positive_maps_by_rotated_iou(
            labels=labels,
            bbox_pred=bbox_pred,
            bbox_targets=bbox_targets,
            pos_inds=pos_inds,
            img_meta=img_meta,
            angle_factor=self.angle_factor)
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)
```

Call the parent exactly once. Do not recompute the assignment and do not alter any object other than the returned labels tensor in enabled mode.

- [x] **Step 6: Add the encoder-scope guard before all loss routing**

At the first executable line of `loss_by_feat`, before the one-group early return, add:

```python
        if (getattr(self, 'position_supervised_enabled', False) and
                (enc_cls_scores is not None or enc_bbox_preds is not None)):
            raise ValueError(
                'position supervision excludes encoder output supervision')
```

Using `getattr` preserves existing lightweight `__new__`-based tests while normal construction always defines the field.

- [x] **Step 7: Run focused and adjacent contract tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_position_supervision.py tests/test_projects/ov_capflow/test_grouped_queries.py tests/test_projects/ov_capflow/test_adaptive_dn.py tests/test_projects/ov_capflow/test_strict_head.py -q
```

Expected: all tests pass. The grouped-query tests retain three independent calls and arithmetic-mean aggregation; the adaptive-DN and strict inference tests remain unchanged.

- [x] **Step 8: Commit the integration**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_position_supervision.py
rtk git diff --cached --check
rtk git commit -m "feat: enable opt-in position supervision"
```

Expected: only the two named files are committed.

### Task 3: Add the immutable A1 proxy config and exact-diff audit

**Files:**

- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py`
- Modify: `tests/test_projects/ov_capflow/test_grouped_queries.py`

- [x] **Step 1: Add names and a failing exact-diff config test**

Add these constants beside the existing scale-1024 batch-1 constants:

```python
S1_GROUPED_SCALE1024_BATCH1_RARE4X = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py')
S1_GROUPED_SCALE1024_BATCH1_RARE4X_POSITION_SUPERVISED = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x_position_supervised.py')
```

Add this test after the batch-1 config tests:

```python
def test_a1_position_supervision_changes_only_target_rule_and_paths():
    control = _load(S1_GROUPED_SCALE1024_BATCH1_RARE4X).to_dict()
    candidate = _load(
        S1_GROUPED_SCALE1024_BATCH1_RARE4X_POSITION_SUPERVISED).to_dict()

    assert candidate['a1_parent'] == '8-T6-R-E12'
    assert candidate['a1_only_scientific_delta'] == (
        'rotated_iou_positive_classification_target')
    assert candidate['model']['bbox_head']['position_supervised_cfg'] == {
        'enabled': True}
    assert candidate['model']['num_queries'] == 600
    assert candidate['model']['train_query_groups'] == 3
    assert candidate['model']['bbox_head']['matching_query_groups'] == 3
    assert candidate['model']['decoder']['num_layers'] == 6
    assert not candidate['model']['bbox_head'].get(
        'balanced_cfg', {}).get('enabled', False)
    assert candidate['model']['bbox_head']['loss_cls'] == {
        'type': 'mmdet.FocalLoss',
        'use_sigmoid': True,
        'gamma': 2.0,
        'alpha': 0.25,
        'loss_weight': 1.0,
    }
    assert candidate['model']['bbox_head']['loss_bbox'] == {
        'type': 'mmdet.L1Loss', 'loss_weight': 5.0}
    assert candidate['model']['bbox_head']['loss_iou'] == {
        'type': 'GDLoss',
        'loss_type': 'kld',
        'fun': 'log1p',
        'tau': 1,
        'sqrt': False,
        'loss_weight': 2.0,
    }
    assert candidate['model']['train_cfg']['assigner']['match_costs'] == [
        {'type': 'mmdet.BinaryFocalLossCost', 'weight': 2.0},
        {'type': 'RBoxL1Cost', 'weight': 5.0, 'box_format': 'xywha'},
        {
            'type': 'GDCost',
            'loss_type': 'kld',
            'fun': 'log1p',
            'tau': 1,
            'sqrt': False,
            'weight': 2.0,
        },
    ]
    assert candidate['model']['type'] == 'OVCapFlow'
    assert candidate['resume'] is False
    assert candidate['randomness']['seed'] == 20260716
    assert candidate['train_cfg']['max_epochs'] == 12
    assert candidate['work_dir'].endswith(
        's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_'
        'position_supervised')

    candidate.pop('a1_parent')
    candidate.pop('a1_only_scientific_delta')
    candidate['model']['bbox_head'].pop('position_supervised_cfg')
    candidate['train_dataloader']['batch_sampler']['audit_path'] = control[
        'train_dataloader']['batch_sampler']['audit_path']
    candidate['work_dir'] = control['work_dir']
    assert candidate == control
```

- [x] **Step 2: Run only the new test and observe the missing-config failure**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py::test_a1_position_supervision_changes_only_target_rule_and_paths -q
```

Expected: fail because the candidate config file does not exist.

- [x] **Step 3: Pin the fixed-query encoder exclusion in the production pre-decoder test**

In `test_pre_decoder_groups_only_during_training`, immediately after the
training `head_inputs` DN metadata assertions, add:

```python
    assert head_inputs['enc_outputs_class'] is None
    assert head_inputs['enc_outputs_coord'] is None
```

The candidate config test pins `model.type == 'OVCapFlow'` and three groups;
these assertions pin that class's live training pre-decoder output. The
enabled real-batch test later closes the loop because its head guard fails if
either encoder output is non-`None`.

- [x] **Step 4: Create the candidate config with one scientific delta**

Create the file with exactly:

```python
_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py'
]

a1_parent = '8-T6-R-E12'
a1_only_scientific_delta = (
    'rotated_iou_positive_classification_target')

model = dict(
    bbox_head=dict(position_supervised_cfg=dict(enabled=True)))
train_dataloader = dict(
    batch_sampler=dict(
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_'
            'position_supervised_sampler.json')))
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_'
    'position_supervised')
resume = False
```

- [x] **Step 5: Run the exact-diff test, the full config module, and the pre-decoder contract**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py::test_a1_position_supervision_changes_only_target_rule_and_paths -q
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_grouped_queries.py::test_pre_decoder_groups_only_during_training -q
```

Expected: all three commands pass. The equality assertion proves that the resolved
candidate differs from the repository's rare4x control config file only by
the A1 flag and bookkeeping paths. Reuse of the historical `8-T6-R-E12` run
as the scientific control remains separately gated by Task 6's artifact hash
audit.

- [x] **Step 6: Commit the candidate config and audits**

Run:

```bash
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py tests/test_projects/ov_capflow/test_grouped_queries.py
rtk git diff --cached --check
rtk git commit -m "config: add A1 position supervision proxy"
```

Expected: only the new config, its exact-diff test, and the two encoder-output
assertions are committed.

### Task 4: Strengthen the zero-update real-batch audit

**Files:**

- Modify: `tests/test_projects/ov_capflow/test_dotav2_real_batch.py`
- Modify: `tests/test_projects/ov_capflow/test_strict_head.py`

- [x] **Step 1: Add candidate-specific scope and inference-finiteness assertions**

Inside `test_real_dotav2_batch_forward_backward`, immediately after loading `cfg`, add:

```python
    position_supervised = bool(
        cfg.model.bbox_head.get(
            'position_supervised_cfg', {}).get('enabled', False))
    if position_supervised:
        assert cfg.model.num_queries == 600
        assert cfg.model.train_query_groups == 3
        assert cfg.model.bbox_head.matching_query_groups == 3
        assert cfg.model.decoder.num_layers == 6
        assert not cfg.model.bbox_head.get(
            'balanced_cfg', {}).get('enabled', False)
```

After `losses = model.loss(...)`, extend the finite-loss checks with:

```python
    if position_supervised:
        for key in ('loss_cls', 'dn_loss_cls'):
            assert key in losses
            assert torch.is_tensor(losses[key])
            assert torch.isfinite(losses[key]).all()
            assert losses[key].requires_grad
```

Immediately after the existing `total = sum(...)` statement and before
`total.backward()`, add:

```python
    assert torch.isfinite(total).all()
```

After the `active` gradient-name assertion block, add:

```python
    finite_gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert finite_gradients
    assert torch.stack(finite_gradients).all()
```

After the existing prediction shape assertions, add:

```python
    assert all(
        sample.pred_instances.scores.shape ==
        sample.pred_instances.labels.shape == (cfg.model.num_queries, )
        for sample in predictions)
    assert all(
        sample.pred_instances.bboxes.shape ==
        (cfg.model.num_queries, 5)
        for sample in predictions)
    assert all(
        torch.isfinite(sample.pred_instances.scores).all()
        for sample in predictions)
    assert all(
        torch.isfinite(sample.pred_instances.bboxes).all()
        for sample in predictions)
```

The test already enforces the exact clean-start checkpoint missing-key set, no unexpected state-dict keys, finite losses, successful backward, and 600 output rows. Do not add an optimizer or call `step()`.

- [x] **Step 2: Add a non-monotonic query-order inference contract**

Append to `test_strict_head.py`:

```python
def test_predict_preserves_nonmonotonic_query_order():
    head = object.__new__(OVCapFlowHead)
    head.readout_cfg = dict(temperature=1.0, power=1.0)
    head.angle_factor = 3.141592653589793
    token_logits = torch.tensor([
        [0.0, -4.0],
        [4.0, -4.0],
        [-4.0, 2.0],
    ])
    boxes = torch.tensor([
        [0.1, 0.1, 0.1, 0.1, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.0],
        [0.3, 0.3, 0.1, 0.1, 0.0],
    ])

    results = head._predict_by_feat_single(
        token_logits,
        boxes,
        token_positive_maps={1: [0], 2: [1]},
        img_meta=dict(img_shape=(100, 100)),
        rescale=False)

    torch.testing.assert_close(
        results.scores, torch.sigmoid(torch.tensor([0.0, 4.0, 2.0])))
    assert results.labels.tolist() == [0, 0, 1]
    torch.testing.assert_close(
        results.bboxes[:, 0], torch.tensor([10.0, 20.0, 30.0]))
```

The deliberately non-monotonic scores and distinct boxes fail if any part of
the live prediction path sorts query rows by score.

- [x] **Step 3: Run the tests in skipped/CPU mode as a collection and order check**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_strict_head.py -q
```

Expected: the opt-in GPU tests collect successfully and are skipped, and all
strict-head tests pass; there must be no import or config collection error.

- [x] **Step 4: Commit the Stage-0 assertions**

Run:

```bash
rtk git add tests/test_projects/ov_capflow/test_dotav2_real_batch.py tests/test_projects/ov_capflow/test_strict_head.py
rtk git diff --cached --check
rtk git commit -m "test: audit A1 real-batch invariants"
```

Expected: only the real-batch and strict-head tests are committed.

### Task 5: Run CPU regressions and the single-GPU zero-update smoke

**Files:**

- Verify only; no planned source changes.

- [x] **Step 1: Run the focused CPU contract suite**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_position_supervision.py tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py tests/test_projects/ov_capflow/test_grouped_queries.py tests/test_projects/ov_capflow/test_adaptive_dn.py tests/test_projects/ov_capflow/test_strict_head.py tests/test_projects/ov_capflow/test_strict_detector.py -q
```

Expected: all selected tests pass; no xfail is accepted for A1 tests.

- [x] **Step 2: Run the complete non-opt-in OVCapFlow project suite**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow -q
```

Expected: all ordinary tests pass. Only tests explicitly skipped because their documented opt-in environment variable is absent may be skipped.

- [x] **Step 3: Check the approved GPU set and physical GPU 2 occupancy**

Run these read-only commands:

```bash
rtk nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
rtk pgrep -af '(tools/[t]rain.py|tools/[t]est.py|torch.distributed.[r]un|torch.distributed.[l]aunch).*rare4x_position_supervised'
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -c "from pathlib import Path; p=Path('work_dirs/dotav2_cleanstart/s1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_position_supervised'); print(sorted(str(path) for path in p.rglob('*.pth')) if p.exists() else [])"
```

Expected: physical GPU `2` is present and has no compute process, `pgrep`
prints no matching CPU/GPU launcher, and the candidate checkpoint snapshot is
`[]`. If GPU 2 is occupied or a matching process exists, stop this task and
report the PID/process; do not kill it and do not use any GPU outside
`2,3,8,9`. If the checkpoint snapshot is not empty, preserve it and stop for
artifact provenance review before the smoke.

- [x] **Step 4: Run one candidate real-batch forward/backward and inference on physical GPU 2**

Run exactly one opt-in test process:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow CUDA_VISIBLE_DEVICES=2 OVCAPFLOW_RUN_DOTA2_REAL_BATCH=1 OVCAPFLOW_DOTA2_CONFIG=configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py::test_real_dotav2_batch_forward_backward -q -s
```

Expected: one test passes with finite matching and DN losses, finite gradients, unchanged checkpoint key coverage, and exactly 600 finite inference rows per image. The test performs backward but no optimizer update.

- [x] **Step 5: Confirm the smoke exited without creating a candidate checkpoint**

Run:

```bash
rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
rtk pgrep -af '(tools/[t]rain.py|tools/[t]est.py|torch.distributed.[r]un|torch.distributed.[l]aunch).*rare4x_position_supervised'
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -c "from pathlib import Path; p=Path('work_dirs/dotav2_cleanstart/s1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_position_supervised'); print(sorted(str(path) for path in p.rglob('*.pth')) if p.exists() else [])"
rtk git status --short
```

Expected: the pytest process has exited, neither `nvidia-smi` nor `pgrep`
finds an A1 training/evaluation process, the candidate checkpoint snapshot
remains `[]`, and the working tree has no new tracked modifications.
Pre-existing untracked user files may still appear and must remain untouched.

Execution note (2026-07-23): the final focused suite passed 62 tests and the
final complete non-opt-in suite passed 666 tests with 16 documented opt-in
skips. Two opt-in pytest processes were attempted on physical GPU 2, not one:
the first stopped before backward/inference because its randomly selected
batch had no GT and upstream GroundingDINO intentionally returned a detached
zero DN loss for zero DN queries. Systematic diagnosis showed that matching
losses still had gradients and A1 does not touch DN. Commit `3daf221` changed
only the smoke fixture to select the first non-empty-GT batch with a bounded
scan while retaining the strict matching/DN gradient checks. The second
process was the only complete effective smoke and passed. Neither process ran
an optimizer step or created a checkpoint; postflight showed GPU 2 idle, no
A1 process, an empty candidate checkpoint snapshot, and no new tracked files.

### Task 6: Independent review, final verification, and gated handoff

**Files:**

- Review: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Review: `tests/test_projects/ov_capflow/test_position_supervision.py`
- Review: `tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py`
- Review: `tests/test_projects/ov_capflow/test_grouped_queries.py`
- Review: `tests/test_projects/ov_capflow/test_dotav2_real_batch.py`
- Review: `tests/test_projects/ov_capflow/test_strict_head.py`
- Review: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py`

- [x] **Step 1: Dispatch a fresh review subagent**

Give the reviewer the approved spec, the commit range `7a2d5d5981dc873934407305c8e8eb977a0e87fb..HEAD`, and this exact brief:

```text
Review A1 for spec compliance and scientific isolation. Check the aligned
rotated-IoU math, float32/detach/clamp/finiteness behavior, normalized token
map preservation, disabled tensor-for-tensor rollback, all-layer/all-group
coverage, DN/encoder/Hungarian/regression/inference exclusions, constructor
validation, no learned state, exact proxy config diff, Q600/no-NMS/no-top-k
integrity, and real-batch zero-update coverage. Report findings by severity
with file:line evidence. Do not modify files.
```

Expected: a written review with either no critical/important finding or a concrete list to fix. Do not proceed past an unresolved critical or important finding.

- [x] **Step 2: Apply the independent-review gate**

Expected: the review reports no critical or important finding. If it reports
either severity, stop without modifying files and return the evidence to the
plan author for a new explicit plan revision. Do not improvise a fix inside
this task. Minor editorial findings may be reported for later cleanup but
cannot broaden A1 or change the scientific diff.

- [x] **Step 3: Run fresh final verification**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_position_supervision.py tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py tests/test_projects/ov_capflow/test_grouped_queries.py tests/test_projects/ov_capflow/test_adaptive_dn.py tests/test_projects/ov_capflow/test_strict_head.py tests/test_projects/ov_capflow/test_strict_detector.py -q
rtk git diff 7a2d5d5981dc873934407305c8e8eb977a0e87fb..HEAD --check
rtk git status --short
```

Expected: all focused tests pass, the commit-range diff is whitespace-clean, and only pre-existing unrelated untracked files remain.

- [x] **Step 4: Audit the final scientific diff and commit list**

Run:

```bash
rtk git diff --stat 7a2d5d5981dc873934407305c8e8eb977a0e87fb..HEAD
rtk git diff 7a2d5d5981dc873934407305c8e8eb977a0e87fb..HEAD -- projects/OVCapFlow/ov_capflow/ov_capflow_head.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py
rtk git log --oneline 7a2d5d5981dc873934407305c8e8eb977a0e87fb..HEAD
```

Expected: the source diff contains only the default-disabled A1 helper/config/override/guard; the proxy config contains only the A1 flag and bookkeeping; no full-data config, training output, checkpoint, evaluator change, or data change exists.

- [x] **Step 5: Audit the historical `8-T6-R-E12` control artifacts**

Compare the normalized candidate config to the config actually saved by the
historical control run. The only historical-launch-only field permitted is
`launcher`; the candidate-only A1 and bookkeeping fields are normalized away:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -c "from mmengine import Config; import hashlib,json; candidate=Config.fromfile('configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py').to_dict(); historical=Config.fromfile('work_dirs/dotav2_cleanstart/s1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x.py').to_dict(); historical.pop('launcher',None); candidate.pop('a1_parent'); candidate.pop('a1_only_scientific_delta'); candidate['model']['bbox_head'].pop('position_supervised_cfg'); candidate['train_dataloader']['batch_sampler']['audit_path']=historical['train_dataloader']['batch_sampler']['audit_path']; candidate['work_dir']=historical['work_dir']; encode=lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode(); assert candidate==historical; print(hashlib.sha256(encode(candidate)).hexdigest()); print(hashlib.sha256(encode(historical)).hexdigest())"
rtk sha256sum work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_compatible.pth work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/train/manifest.json work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.sha256
```

Expected: the assertion passes, the two normalized resolved-config hashes
printed by the first command are both
`4449be70d7d2872f44d64a275d8c7e7fc4979f914084f8251520063d3ad46f3b`.
The second command must record checkpoint hash
`e4b612bedf7ce0d78064bfacdf5b3641e265aaf1a75c0f3c2d82a924bd337d16`,
rare4x train-manifest hash
`1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290`,
and proxy manifest-record hash
`99b46b9871370ae529f11f8322ca6fdd52113bde1ee5129fbc809b718c61eddb`
in the Stage-0 report.

If the saved historical config is absent, normalization equality fails, any
artifact hash is unavailable, or the recorded historical hashes disagree,
the existing `8-T6-R-E12` result is not an admissible control. Stop and
request separate authorization for a paired control; never launch only the
A1 candidate under a failed control audit.

- [x] **Step 6: Stop and request explicit Stage-1 run authorization**

Report:

- focused and full CPU test counts;
- real-batch GPU ID and result;
- checkpoint missing/unexpected-key audit result;
- independent-review disposition;
- commit hashes;
- candidate config path;
- the immutable Stage-1 metric and mechanism gates from the approved spec.

Include this reviewed-but-not-executed Stage-1 launch command in the report:

```bash
rtk env CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=41981 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x_position_supervised.py --launcher pytorch
```

Before any later authorized launch, recheck GPUs `8,9`, port `41981`, the
resolved-config hash, the generic clean-start checkpoint hash, and the
immutable train/validation manifests. The Stage-1 endpoint gate remains
`AP50 >= 0.4790`, `novel4 >= 0.3675`, and `base14 >= 0.464643`; the mechanism
gate remains score--rotated-IoU Spearman improvement `>= +0.03`, total-recall
drop `<= 0.005`, and reduced fixed-recall ranking headroom. All gates are
conjunctive and use Epoch 12 only.

Do not launch `tools/train.py`, `torchrun`, proxy evaluation, a repeat, or a full-data run. The next action requires the user's explicit authorization after reviewing Stage-0 evidence.

Execution note (2026-07-23): the independent spec and final quality reviews
reported no Critical or Important findings. Fresh focused verification passed
62 tests; the full non-opt-in suite passed 666 tests with 16 opt-in skips.
Commit-range whitespace and tracked-worktree checks were clean. The historical
control normalization produced the expected config hash twice, and the
checkpoint, rare4x manifest, and proxy-manifest-record SHA256 values matched
the three preregistered values above. The quality review recorded two Minor
monitoring risks only: possible throughput cost from CUDA synchronization in
the finite/input checks and non-replayability of the exact shuffled smoke
fixture. Before Task 6 completed, the user explicitly authorized all remaining
in-scope work and instructed autonomous duty for ten hours while unavailable.
That later explicit authorization satisfies this handoff gate and supersedes
the stop/wait instruction above; it does not relax any metric, mechanism,
artifact, GPU, or no-rescue gate.

## Completion boundary

This implementation plan is complete when Tasks 1–6 pass and the candidate is waiting at the Stage-1 authorization gate. It does **not** complete the persistent AP70 research goal. That goal remains complete only when a canonical raw 13,833-image Q600 run records both official `dota/mAP >= 0.7000` and `dota/AP50 >= 0.7000` under the frozen rotated end-to-end protocol.
