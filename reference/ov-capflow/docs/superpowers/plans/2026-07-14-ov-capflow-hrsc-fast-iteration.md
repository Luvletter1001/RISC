# OV-CapFlow HRSC Fast-Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible HRSC2016 structural gate for OV-CapFlow, covering a strict fixed-query parent, identity-preserving evidence fusion, balanced matched/unmatched supervision, explicit null calibration, and stable all-query readout.

**Architecture:** Keep the existing CastDet/RHINO substrate isolated under `projects/OVCapFlow`. Add explicit config switches at decoder/head boundaries, keep balanced classification and null calibration in focused modules, route final matching masks through the head, and expose final capacity/null state through the detector. Train matched HRSC anchors first, then freeze the strict parent and run a five-epoch causal adapter matrix with `.lab` as the experiment ledger.

**Tech Stack:** Python 3.8.19, PyTorch 1.12.1+cu113, MMCV 2.1.0, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate 1.0.0rc1, pytest, four NVIDIA A40 GPUs.

---

## File map

- `projects/OVCapFlow/ov_capflow/calibration.py`: balanced matched/unmatched classification reduction and log-domain score calibration.
- `projects/OVCapFlow/ov_capflow/null_reservoir.py`: continuous per-query null logits and null/gate/mass losses.
- `projects/OVCapFlow/ov_capflow/ov_capflow_layers.py`: optional semantic/density interventions and final decoder-level null state.
- `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`: balanced matching loss, final Hungarian mask capture, stable readout, and null-aware score calibration.
- `projects/OVCapFlow/ov_capflow/ov_capflow.py`: explicit loss configuration, decoder-state routing, and auxiliary loss aggregation.
- `projects/OVCapFlow/ov_capflow/freeze_except_hook.py`: fail-closed regex-based adapter freezing for H2.
- `projects/OVCapFlow/ov_capflow/__init__.py`: registry/export surface for new modules and hook.
- `configs/ov_capflow/hrsc/grounding_dino_swin-t_hrsc_parent_10e.py`: upstream H1 parent.
- `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c0_native_10e.py`: strict H1 parent.
- `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_fusion_5e.py`: H2 fusion arm.
- `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c2_balanced_5e.py`: H2 balanced arm.
- `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c3_null_5e.py`: H2 null arm.
- `tests/test_projects/ov_capflow/test_calibration.py`: balanced loss and stable-readout contracts.
- `tests/test_projects/ov_capflow/test_null_reservoir.py`: null probability, group losses, mass, and gate-order tests.
- `tests/test_projects/ov_capflow/test_hrsc_config.py`: HRSC dataset/config/switch tests.
- `tests/test_projects/ov_capflow/test_hrsc_real_batch.py`: opt-in CUDA real-batch H0 test.
- `projects/OVCapFlow/tools/audit_checkpoint_load.py`: checkpoint missing/unexpected/effective-switch JSON audit.
- `projects/OVCapFlow/tools/audit_strict_inference.py`: forbidden-operation and prediction-count audit.
- `.gitignore`: keep `.lab/` and `run.log` outside version control.
- `.lab/*`: untracked researcher configuration, results, branch ledger, and run helpers.

## Task 1: Freeze the existing OV-CapFlow substrate

**Files:**
- Add: `projects/OVCapFlow/**`
- Add: `configs/ov_capflow/ov_capflow_swin-t_strict_visdrone.py`
- Add: `tests/test_projects/ov_capflow/**`
- Add: `docs/superpowers/specs/2026-07-14-ov-capflow-design.md`
- Add: `docs/superpowers/plans/2026-07-14-ov-capflow-strict-substrate.md`
- Add: `docs/superpowers/plans/2026-07-14-ov-capflow-semantic-capacity.md`

- [ ] **Step 1: Verify the recorded environment**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -c "import torch, mmcv, mmengine, mmdet, mmrotate; print(torch.__version__, mmcv.__version__, mmengine.__version__, mmdet.__version__, mmrotate.__version__)"
```

Expected: `1.12.1+cu113 2.1.0 0.10.4 3.3.0 1.0.0rc1`.

- [ ] **Step 2: Run the existing focused suite**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Expected: `17 passed` and only the known MMCV dropout deprecation warnings.

- [ ] **Step 3: Scan the executable project path**

Run:

```bash
rtk rg -n "torch\.topk|\.topk\(|multiclass_nms|rotated_nms|\bnms\b|minAreaRect" projects/OVCapFlow/ov_capflow configs/ov_capflow -g '*.py'
```

Expected: no executable match. Comments that mention the invariant are reviewed manually and do not call a forbidden operation.

- [ ] **Step 4: Commit only the current substrate assets**

Run:

```bash
rtk git add projects/OVCapFlow configs/ov_capflow/ov_capflow_swin-t_strict_visdrone.py tests/test_projects/ov_capflow docs/superpowers/specs/2026-07-14-ov-capflow-design.md docs/superpowers/plans/2026-07-14-ov-capflow-strict-substrate.md docs/superpowers/plans/2026-07-14-ov-capflow-semantic-capacity.md
rtk git commit -m "feat: freeze OV-CapFlow strict substrate"
```

Expected: one commit containing the existing 17-test substrate; unrelated untracked project-history files remain untouched.

## Task 2: Add explicit semantic and density switches

**Files:**
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_layers.py`
- Modify: `tests/test_projects/ov_capflow/test_semantic_capacity.py`

- [ ] **Step 1: Write failing switch tests**

Append:

```python
def test_disabled_semantic_fusion_returns_transported_query():
    native = torch.randn(1, 5, 4)
    transported = torch.randn(1, 5, 4)
    output, gate = apply_matching_query_interventions(
        native_query=native,
        transported_query=transported,
        matching_query_count=3,
        fusion=None,
        capacity=None)
    torch.testing.assert_close(output, transported)
    assert gate is None


def test_fusion_without_density_uses_no_capacity_mask():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    with torch.no_grad():
        fusion.gate.bias.fill_(1.0)
    native = torch.randn(1, 3, 4)
    transported = torch.randn(1, 3, 4)
    output, gate = apply_matching_query_interventions(
        native_query=native,
        transported_query=transported,
        matching_query_count=3,
        fusion=fusion,
        capacity=None)
    expected, expected_gate = fusion(native, transported, capacity=None)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(gate, expected_gate)


def test_decoder_layer_omits_disabled_modules():
    layer = OVCapFlowDecoderLayer(
        enable_semantic_fusion=False,
        enable_density_capacity=False,
        self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
        cross_attn_text_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
        cross_attn_cfg=dict(
            embed_dims=8, num_heads=2, num_levels=1, dropout=0.0),
        ffn_cfg=dict(embed_dims=8, feedforward_channels=16, ffn_drop=0.0))
    assert layer.semantic_fusion is None
    assert layer.density_capacity is None
```

Update the test import to include `apply_matching_query_interventions`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_semantic_capacity.py -q
```

Expected: collection fails because `apply_matching_query_interventions` and the enable flags do not exist.

- [ ] **Step 3: Implement the switchable helper and layer constructor**

Add this helper above `OVCapFlowDecoderLayer`:

```python
def apply_matching_query_interventions(
        native_query: Tensor,
        transported_query: Tensor,
        matching_query_count: int,
        fusion: Optional[SemanticEvidenceFusion],
        capacity: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
    if native_query.shape != transported_query.shape:
        raise ValueError('native_query and transported_query must match')
    if matching_query_count <= 0 or \
            matching_query_count > native_query.shape[1]:
        raise ValueError('matching_query_count is outside the query range')
    if fusion is None:
        return transported_query, None
    if capacity is not None and capacity.shape != \
            native_query[:, -matching_query_count:].shape[:-1]:
        raise ValueError('capacity must describe the matching query suffix')

    matching_native = native_query[:, -matching_query_count:]
    matching_evidence = transported_query[:, -matching_query_count:]
    fused_matching, gate = fusion(
        matching_native, matching_evidence, capacity=capacity)
    prefix = transported_query[:, :-matching_query_count]
    if prefix.shape[1] == 0:
        return fused_matching, gate
    return torch.cat([prefix, fused_matching], dim=1), gate
```

Replace the layer constructor with:

```python
def __init__(self,
             enable_semantic_fusion: bool = True,
             enable_density_capacity: bool = True,
             semantic_fusion_cfg: Optional[dict] = None,
             **kwargs) -> None:
    super().__init__(**kwargs)
    semantic_fusion_cfg = dict(semantic_fusion_cfg or {})
    self.enable_semantic_fusion = bool(enable_semantic_fusion)
    self.enable_density_capacity = bool(enable_density_capacity)
    self.semantic_fusion = (
        SemanticEvidenceFusion(
            embed_dims=self.embed_dims, **semantic_fusion_cfg)
        if self.enable_semantic_fusion else None)
    self.density_capacity = (
        ContinuousDensityCapacity(self.embed_dims)
        if self.enable_density_capacity else None)
    self.last_capacity = None
    self.last_predicted_count = None
    self.last_semantic_gate = None
```

In `forward`, clear cached state at the start, compute capacity only when the module exists, and call `apply_matching_query_interventions`. When both interventions are disabled, return the upstream transported query unchanged.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_semantic_capacity.py -q
```

Expected: all semantic/capacity tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow_layers.py tests/test_projects/ov_capflow/test_semantic_capacity.py
rtk git commit -m "feat: make OV-CapFlow interventions switchable"
```

## Task 3: Add balanced matched/unmatched classification reduction

**Files:**
- Create: `projects/OVCapFlow/ov_capflow/calibration.py`
- Create: `tests/test_projects/ov_capflow/test_calibration.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`

- [ ] **Step 1: Write failing balanced-reduction tests**

Create:

```python
import torch
from mmdet.models.losses import FocalLoss

from projects.OVCapFlow.ov_capflow.calibration import (
    balanced_group_classification_loss,
)


def _loss_module():
    return FocalLoss(
        use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean')


def test_balanced_loss_is_invariant_to_repeated_unmatched_queries():
    logits = torch.tensor([[[2.0], [-2.0], [-2.0]]], requires_grad=True)
    labels = torch.tensor([[[1.0], [0.0], [0.0]]])
    matched = torch.tensor([[True, False, False]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    weights = torch.ones(1, 3)
    loss_a, stats_a = balanced_group_classification_loss(
        _loss_module(), logits, labels, weights, valid, matched, 1.0, 1.0)

    logits_b = torch.cat([logits, logits[:, 1:].repeat(1, 4, 1)], dim=1)
    labels_b = torch.cat([labels, labels[:, 1:].repeat(1, 4, 1)], dim=1)
    matched_b = torch.cat(
        [matched, torch.zeros(1, 8, dtype=torch.bool)], dim=1)
    valid_b = torch.ones_like(labels_b, dtype=torch.bool)
    weights_b = torch.ones(1, 11)
    loss_b, stats_b = balanced_group_classification_loss(
        _loss_module(), logits_b, labels_b, weights_b, valid_b, matched_b,
        1.0, 1.0)

    torch.testing.assert_close(loss_a, loss_b)
    assert stats_a['matched_queries'] == stats_b['matched_queries'] == 1
    assert stats_b['unmatched_queries'] == 10


def test_balanced_loss_handles_empty_group_and_backpropagates():
    logits = torch.tensor([[[-1.0], [-2.0]]], requires_grad=True)
    labels = torch.zeros_like(logits)
    matched = torch.zeros(1, 2, dtype=torch.bool)
    valid = torch.ones_like(labels, dtype=torch.bool)
    loss, stats = balanced_group_classification_loss(
        _loss_module(), logits, labels, torch.ones(1, 2), valid, matched,
        1.0, 1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert stats['matched_queries'] == 0
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_calibration.py -q
```

Expected: import fails because `calibration.py` does not exist.

- [ ] **Step 3: Implement the balanced loss helper**

Create:

```python
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def _masked_loss(loss_module: nn.Module, logits: Tensor, labels: Tensor,
                 weights: Tensor, mask: Tensor) -> Tensor:
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    selected_weights = weights[mask]
    return loss_module(
        selected_logits,
        selected_labels,
        selected_weights,
        avg_factor=max(int(mask.sum().item()), 1))


def balanced_group_classification_loss(
        loss_module: nn.Module,
        cls_scores: Tensor,
        labels: Tensor,
        query_weights: Tensor,
        valid_token_mask: Tensor,
        matched_query_mask: Tensor,
        matched_weight: float,
        unmatched_weight: float) -> Tuple[Tensor, Dict[str, int]]:
    if cls_scores.shape != labels.shape or cls_scores.shape != \
            valid_token_mask.shape:
        raise ValueError('scores, labels, and token mask must match')
    if query_weights.shape != cls_scores.shape[:2]:
        raise ValueError('query_weights must have shape [batch, queries]')
    if matched_query_mask.shape != cls_scores.shape[:2]:
        raise ValueError('matched mask must have shape [batch, queries]')

    token_weights = query_weights.unsqueeze(-1).expand_as(cls_scores)
    matched_tokens = valid_token_mask & matched_query_mask.unsqueeze(-1)
    unmatched_tokens = valid_token_mask & ~matched_query_mask.unsqueeze(-1)
    matched_loss = _masked_loss(
        loss_module, cls_scores, labels, token_weights, matched_tokens)
    unmatched_loss = _masked_loss(
        loss_module, cls_scores, labels, token_weights, unmatched_tokens)
    total = matched_weight * matched_loss + unmatched_weight * unmatched_loss
    stats = {
        'matched_queries': int(matched_query_mask.sum().item()),
        'unmatched_queries': int((~matched_query_mask).sum().item()),
        'matched_tokens': int(matched_tokens.sum().item()),
        'unmatched_tokens': int(unmatched_tokens.sum().item()),
    }
    return total, stats
```

- [ ] **Step 4: Add the head configuration and matching-loss override**

Add to `OVCapFlowHead.__init__`:

```python
def __init__(self, balanced_cfg=None, readout_cfg=None, **kwargs):
    self.balanced_cfg = dict(balanced_cfg or {})
    self.readout_cfg = dict(readout_cfg or {})
    self.last_matching_mask = None
    self.last_balance_stats = None
    super().__init__(**kwargs)
```

Import `balanced_group_classification_loss`, then add this complete override.
It reuses the parent call for the unchanged rotated box losses and recomputes
Hungarian targets only for the balanced classification mask:

```python
def loss_by_feat_single(self, cls_scores, bbox_preds,
                        batch_gt_instances, batch_img_metas):
    parent_cls, loss_bbox, loss_iou = super().loss_by_feat_single(
        cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)
    if not self.balanced_cfg.get('enabled', False):
        self.last_balance_stats = None
        return parent_cls, loss_bbox, loss_iou

    cls_scores_list = [cls_scores[i] for i in range(cls_scores.size(0))]
    bbox_preds_list = [bbox_preds[i] for i in range(bbox_preds.size(0))]
    with torch.no_grad():
        target_data = self.get_targets(
            cls_scores_list, bbox_preds_list,
            batch_gt_instances, batch_img_metas)
    labels_list, label_weights_list, _, bbox_weights_list, _, _ = target_data
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
        matched_weight=float(self.balanced_cfg.get('matched_weight', 1.0)),
        unmatched_weight=float(
            self.balanced_cfg.get('unmatched_weight', 1.0)))
    self.last_balance_stats = stats
    return loss_cls, loss_bbox, loss_iou
```

Do not change `_loss_dn_single`; denoising remains upstream and unbalanced.

- [ ] **Step 5: Run calibration and regression tests**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_calibration.py tests/test_projects/ov_capflow/test_strict_head.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/calibration.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_calibration.py
rtk git commit -m "feat: balance matched and unmatched classification"
```

## Task 4: Add stable log-domain all-query readout

**Files:**
- Modify: `projects/OVCapFlow/ov_capflow/calibration.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `tests/test_projects/ov_capflow/test_calibration.py`
- Modify: `tests/test_projects/ov_capflow/test_strict_head.py`

- [ ] **Step 1: Write failing numerical-stability tests**

Append:

```python
from projects.OVCapFlow.ov_capflow.calibration import (
    calibrate_selected_log_scores,
    grounding_logits_to_class_log_scores,
    select_from_class_log_scores,
)


def test_log_readout_is_label_invariant_to_common_power():
    logits = torch.tensor([
        [-80.0, -79.0, -120.0, -121.0],
        [-5.0, -6.0, -4.0, -7.0],
    ], dtype=torch.float16)
    positive_map = {1: [0, 1], 2: [2, 3]}
    log_scores = grounding_logits_to_class_log_scores(logits, positive_map)
    _, labels_a = select_from_class_log_scores(log_scores)
    _, labels_b = select_from_class_log_scores(log_scores * 32.0)
    torch.testing.assert_close(labels_a, labels_b)
    assert torch.isfinite(log_scores).all()


def test_calibration_remains_finite_with_null_and_capacity():
    selected = torch.tensor([-120.0, -2.0], dtype=torch.float16)
    null_logits = torch.tensor([20.0, -20.0], dtype=torch.float16)
    capacity = torch.tensor([1e-4, 0.9], dtype=torch.float16)
    scores = calibrate_selected_log_scores(
        selected, null_logits=null_logits, capacity=capacity,
        temperature=1.0, power=32.0)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
    assert torch.all(scores > 0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_calibration.py -q
```

Expected: imports fail for the three new helpers.

- [ ] **Step 3: Implement stable token-to-class conversion and calibration**

Append to `calibration.py`:

```python
import math
import torch.nn.functional as F


def grounding_logits_to_class_log_scores(
        token_logits: Tensor, positive_map: dict) -> Tensor:
    if token_logits.ndim != 2:
        raise ValueError('token_logits must be [queries, tokens]')
    class_logs = []
    log_token_prob = F.logsigmoid(token_logits.float())
    for label_id in sorted(positive_map):
        token_ids = token_logits.new_tensor(
            positive_map[label_id], dtype=torch.long)
        if token_ids.numel() == 0:
            raise ValueError('each class needs at least one positive token')
        selected = log_token_prob.index_select(-1, token_ids)
        class_logs.append(
            torch.logsumexp(selected, dim=-1) - math.log(token_ids.numel()))
    return torch.stack(class_logs, dim=-1)


def select_from_class_log_scores(class_log_scores: Tensor):
    if class_log_scores.ndim != 2:
        raise ValueError('class_log_scores must be [queries, classes]')
    return class_log_scores.max(dim=-1)


def calibrate_selected_log_scores(
        selected_log_scores: Tensor,
        null_logits: Tensor = None,
        capacity: Tensor = None,
        temperature: float = 1.0,
        power: float = 1.0) -> Tensor:
    if temperature <= 0 or power <= 0:
        raise ValueError('temperature and power must be positive')
    calibrated = selected_log_scores.float() * (power / temperature)
    if null_logits is not None:
        calibrated = calibrated + F.logsigmoid(-null_logits.float())
    if capacity is not None:
        calibrated = calibrated + capacity.float().clamp(min=1e-8).log()
    return calibrated.clamp(min=-80.0, max=0.0).exp()
```

- [ ] **Step 4: Route stable readout through `OVCapFlowHead`**

Import `torch.nn.functional as F` and the three calibration helpers. Replace
`_predict_by_feat_single` with this complete implementation:

```python
def _predict_by_feat_single(self,
                            cls_score,
                            bbox_pred,
                            token_positive_maps,
                            img_meta,
                            rescale=True,
                            null_logits=None,
                            capacity=None):
    num_queries = cls_score.shape[0]
    if bbox_pred.shape != (num_queries, 5):
        raise ValueError('bbox_pred must contain one 5-D box per query')
    if null_logits is not None and null_logits.shape != (num_queries, ):
        raise ValueError('null_logits must have one value per query')
    if capacity is not None and capacity.shape != (num_queries, ):
        raise ValueError('capacity must have one value per query')

    if token_positive_maps is not None:
        class_log_scores = grounding_logits_to_class_log_scores(
            cls_score, token_positive_maps)
    else:
        class_log_scores = F.logsigmoid(cls_score.float()).amax(
            dim=-1, keepdim=True)
    selected_log_scores, det_labels = select_from_class_log_scores(
        class_log_scores)
    scores = calibrate_selected_log_scores(
        selected_log_scores,
        null_logits=null_logits,
        capacity=(capacity if self.readout_cfg.get('use_capacity', False)
                  else None),
        temperature=float(self.readout_cfg.get('temperature', 1.0)),
        power=float(self.readout_cfg.get('power', 1.0)))

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
            raise ValueError('scale_factor must contain two or four values')
        scale_factor = torch.cat(
            [scale_factor, scale_factor.new_ones(1)])
        det_bboxes /= scale_factor

    results = InstanceData()
    results.bboxes = det_bboxes
    results.scores = scores
    results.labels = det_labels
    return results
```

- [ ] **Step 5: Verify all head tests**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_calibration.py tests/test_projects/ov_capflow/test_strict_head.py -q
```

Expected: all tests pass; extreme fp16 scores are finite and labels are power-invariant.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/calibration.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_calibration.py tests/test_projects/ov_capflow/test_strict_head.py
rtk git commit -m "feat: add stable log-domain query readout"
```

## Task 5: Implement the explicit null reservoir

**Files:**
- Create: `projects/OVCapFlow/ov_capflow/null_reservoir.py`
- Create: `tests/test_projects/ov_capflow/test_null_reservoir.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_layers.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`

- [ ] **Step 1: Write failing null module and loss tests**

Create:

```python
import torch

from projects.OVCapFlow.ov_capflow.null_reservoir import (
    ExplicitNullReservoir,
    null_reservoir_losses,
)


def test_null_probability_is_bounded_and_keeps_every_query():
    module = ExplicitNullReservoir(embed_dims=8)
    query = torch.randn(2, 5, 8)
    logits = module(query)
    probability = logits.sigmoid()
    assert logits.shape == (2, 5)
    assert torch.all((probability > 0) & (probability < 1))


def test_null_losses_are_finite_for_mixed_and_empty_groups():
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    matched = torch.tensor([[True, False, False]])
    target_count = torch.tensor([1.0])
    gate = torch.tensor([[0.5, 0.2, 0.1]], requires_grad=True)
    losses = null_reservoir_losses(
        null_logits=logits,
        matched_mask=matched,
        target_count=target_count,
        gate_strength=gate,
        matched_weight=1.0,
        unmatched_weight=1.0,
        mass_weight=0.1,
        gate_order_weight=0.1,
        gate_margin=0.0)
    sum(losses.values()).backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert logits.grad is not None
    assert gate.grad is not None

    empty_matched = torch.zeros(1, 3, dtype=torch.bool)
    empty_losses = null_reservoir_losses(
        null_logits=logits.detach(),
        matched_mask=empty_matched,
        target_count=torch.tensor([0.0]))
    assert all(torch.isfinite(value) for value in empty_losses.values())
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_null_reservoir.py -q
```

Expected: import fails because `null_reservoir.py` does not exist.

- [ ] **Step 3: Implement the null module and group-balanced losses**

Create:

```python
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ExplicitNullReservoir(nn.Module):
    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        if embed_dims <= 0:
            raise ValueError('embed_dims must be positive')
        self.null_head = nn.Linear(embed_dims, 1)
        nn.init.xavier_uniform_(self.null_head.weight)
        nn.init.zeros_(self.null_head.bias)

    def forward(self, query: Tensor) -> Tensor:
        if query.ndim != 3:
            raise ValueError('query must be [batch, queries, channels]')
        return self.null_head(query).squeeze(-1)


def _group_mean(values: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum()
    if count.item() == 0:
        return values.sum() * 0.0
    return values.masked_select(mask).mean()


def null_reservoir_losses(
        null_logits: Tensor,
        matched_mask: Tensor,
        target_count: Tensor,
        gate_strength: Optional[Tensor] = None,
        matched_weight: float = 1.0,
        unmatched_weight: float = 1.0,
        mass_weight: float = 0.1,
        gate_order_weight: float = 0.1,
        gate_margin: float = 0.0) -> Dict[str, Tensor]:
    if null_logits.shape != matched_mask.shape:
        raise ValueError('null logits and matched mask must match')
    target_null = (~matched_mask).to(null_logits.dtype)
    element_loss = F.binary_cross_entropy_with_logits(
        null_logits, target_null, reduction='none')
    matched_loss = _group_mean(element_loss, matched_mask)
    unmatched_loss = _group_mean(element_loss, ~matched_mask)
    non_null_mass = torch.sigmoid(-null_logits).sum(dim=-1)
    mass_loss = F.smooth_l1_loss(
        non_null_mass, target_count.to(null_logits.dtype))

    gate_order = null_logits.sum() * 0.0
    if gate_strength is not None and matched_mask.any() and \
            (~matched_mask).any():
        matched_gate = _group_mean(gate_strength, matched_mask)
        unmatched_gate = _group_mean(gate_strength, ~matched_mask)
        gate_order = F.relu(unmatched_gate - matched_gate + gate_margin)

    return {
        'loss_null_matched': matched_weight * matched_loss,
        'loss_null_unmatched': unmatched_weight * unmatched_loss,
        'loss_null_mass': mass_weight * mass_loss,
        'loss_gate_order': gate_order_weight * gate_order,
    }
```

- [ ] **Step 4: Attach one null reservoir to the decoder output**

Import `ExplicitNullReservoir`, then add this constructor and forward method to
`OVCapFlowDecoder`:

```python
def __init__(self,
             enable_null_reservoir=False,
             null_reservoir_cfg=None,
             **kwargs):
    super().__init__(**kwargs)
    self.enable_null_reservoir = bool(enable_null_reservoir)
    null_reservoir_cfg = dict(null_reservoir_cfg or {})
    self.null_reservoir = (
        ExplicitNullReservoir(
            embed_dims=self.embed_dims, **null_reservoir_cfg)
        if self.enable_null_reservoir else None)
    self.last_null_logits = None


def forward(self, *args, matching_query_count=None, **kwargs):
    inter_states, references = super().forward(
        *args, matching_query_count=matching_query_count, **kwargs)
    self.last_null_logits = None
    if self.null_reservoir is not None:
        if matching_query_count is None:
            raise ValueError('null reservoir requires matching query count')
        final_matching = inter_states[-1][:, -matching_query_count:]
        self.last_null_logits = self.null_reservoir(final_matching)
    return inter_states, references
```

This creates one final-layer null head, not six unused per-layer heads.

- [ ] **Step 5: Run null and decoder tests**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_null_reservoir.py tests/test_projects/ov_capflow/test_semantic_capacity.py -q
```

Expected: all tests pass and the decoder still constructs with null disabled.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/null_reservoir.py projects/OVCapFlow/ov_capflow/ov_capflow_layers.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_null_reservoir.py
rtk git commit -m "feat: add explicit continuous null reservoir"
```

## Task 6: Route matching masks and calibration state through training and inference

**Files:**
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `tests/test_projects/ov_capflow/test_strict_detector.py`
- Modify: `tests/test_projects/ov_capflow/test_strict_head.py`
- Modify: `tests/test_projects/ov_capflow/test_null_reservoir.py`

- [ ] **Step 1: Write failing routing tests**

Add `from types import SimpleNamespace` and append these tests:

```python
def test_forward_decoder_exports_only_available_calibration_state():
    model = OVCapFlow.__new__(OVCapFlow)
    nn.Module.__init__(model)
    model.decoder = SimpleNamespace(
        last_null_logits=torch.randn(2, 4),
        layers=[SimpleNamespace(
            last_capacity=None,
            last_predicted_count=None,
            last_semantic_gate=torch.randn(2, 4, 8))])
    output = model._append_calibration_state({'hidden_states': torch.empty(0)})
    assert output['null_logits'].shape == (2, 4)
    assert 'capacity' not in output
    assert output['semantic_gate'].shape == (2, 4, 8)


def test_null_score_calibration_keeps_all_query_rows():
    head = object.__new__(OVCapFlowHead)
    head.readout_cfg = dict(temperature=1.0, power=1.0)
    head.angle_factor = 3.141592653589793
    token_logits = torch.tensor([
        [2.0, -2.0],
        [1.0, -1.0],
        [0.5, -0.5],
    ])
    boxes = torch.randn(3, 5)
    null_logits = torch.tensor([-2.0, 0.0, 2.0])
    results = head._predict_by_feat_single(
        token_logits,
        boxes,
        token_positive_maps={1: [0], 2: [1]},
        img_meta=dict(img_shape=(100, 100), scale_factor=(1.0, 1.0)),
        rescale=True,
        null_logits=null_logits)
    assert results.scores.shape == results.labels.shape == (3, )
    assert results.bboxes.shape == (3, 5)
    assert results.scores[0] > results.scores[2]
```

- [ ] **Step 2: Run routing tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_strict_detector.py tests/test_projects/ov_capflow/test_strict_head.py tests/test_projects/ov_capflow/test_null_reservoir.py -q
```

Expected: failures for missing routing helper and readout arguments.

- [ ] **Step 3: Capture the final matching mask in `OVCapFlowHead.loss`**

Import `List`, `Optional`, `SampleList`, and implement the complete loss method:

```python
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

    matching_cls, matching_box, _, _ = self.split_outputs(
        outs[0], outs[1], dn_meta)
    final_cls = matching_cls[-1]
    final_box = matching_box[-1]
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
```

The mask is final-layer matching-query only and never includes DN queries.

- [ ] **Step 4: Export decoder calibration state**

Add to `OVCapFlow`:

```python
def _append_calibration_state(self, outputs: Dict) -> Dict:
    outputs = dict(outputs)
    last_layer = self.decoder.layers[-1]
    if self.decoder.last_null_logits is not None:
        outputs['null_logits'] = self.decoder.last_null_logits
    if last_layer.last_capacity is not None:
        outputs['capacity'] = last_layer.last_capacity
    if last_layer.last_semantic_gate is not None:
        outputs['semantic_gate'] = last_layer.last_semantic_gate
    return outputs


def forward_decoder(self, *args, **kwargs) -> Dict:
    outputs = super().forward_decoder(*args, **kwargs)
    return self._append_calibration_state(outputs)
```

Add complete prediction routing methods to `OVCapFlowHead`:

```python
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
    batch_img_metas = [sample.metainfo for sample in batch_data_samples]
    positive_maps = [sample.token_positive_map
                     for sample in batch_data_samples]
    cls_scores, bbox_preds = self(
        hidden_states, references, memory_text, text_token_mask)
    cls_scores = cls_scores[-1]
    bbox_preds = bbox_preds[-1]
    results = []
    for index, (cls_score, bbox_pred, positive_map, img_meta) in enumerate(
            zip(cls_scores, bbox_preds, positive_maps, batch_img_metas)):
        image_null = None if null_logits is None else null_logits[index]
        image_capacity = None if capacity is None else capacity[index]
        results.append(self._predict_by_feat_single(
            cls_score,
            bbox_pred,
            positive_map,
            img_meta,
            rescale=rescale,
            null_logits=image_null,
            capacity=image_capacity))
    return results
```

Training ignores raw capacity/gate arguments after the detector has cached them;
inference uses null/capacity only for per-query calibration and never deletes a
row.

- [ ] **Step 5: Aggregate density and null losses in `OVCapFlow.loss`**

Add a detector constructor that stores loss configs before the parent build:

```python
def __init__(self, density_loss_cfg=None, null_loss_cfg=None, **kwargs):
    self.density_loss_cfg = dict(density_loss_cfg or {})
    self.null_loss_cfg = dict(null_loss_cfg or {})
    super().__init__(**kwargs)
```

After `super().loss`, use the final layer and captured mask:

```python
last_layer = self.decoder.layers[-1]
target_count = next(iter(losses.values())).new_tensor(
    [len(sample.gt_instances) for sample in batch_data_samples])
if last_layer.last_capacity is not None:
    density_losses = density_capacity_losses(
        last_layer.last_capacity,
        last_layer.last_predicted_count,
        target_count)
    density_weight = float(self.density_loss_cfg.get('weight', 1.0))
    losses.update({name: density_weight * value
                   for name, value in density_losses.items()})

if self.decoder.last_null_logits is not None:
    matched = self.bbox_head.last_matching_mask
    if matched is None:
        raise RuntimeError('null loss requires final Hungarian matching mask')
    gate = last_layer.last_semantic_gate
    gate_strength = None if gate is None else gate.abs().mean(dim=-1)
    losses.update(null_reservoir_losses(
        self.decoder.last_null_logits,
        matched,
        target_count,
        gate_strength=gate_strength,
        matched_weight=float(
            self.null_loss_cfg.get('matched_weight', 1.0)),
        unmatched_weight=float(
            self.null_loss_cfg.get('unmatched_weight', 1.0)),
        mass_weight=float(self.null_loss_cfg.get('mass_weight', 0.1)),
        gate_order_weight=float(
            self.null_loss_cfg.get('gate_order_weight', 0.1)),
        gate_margin=float(self.null_loss_cfg.get('gate_margin', 0.0))))
```

- [ ] **Step 6: Run the focused suite**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Expected: all previous and new tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_strict_detector.py tests/test_projects/ov_capflow/test_strict_head.py tests/test_projects/ov_capflow/test_null_reservoir.py
rtk git commit -m "feat: route null and capacity calibration state"
```

## Task 7: Add HRSC H1/H2 configurations

**Files:**
- Create: `configs/ov_capflow/hrsc/grounding_dino_swin-t_hrsc_parent_10e.py`
- Create: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c0_native_10e.py`
- Create: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_fusion_5e.py`
- Create: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c2_balanced_5e.py`
- Create: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c3_null_5e.py`
- Create: `tests/test_projects/ov_capflow/test_hrsc_config.py`

- [ ] **Step 1: Write failing HRSC config tests**

Create:

```python
from pathlib import Path

import pytest
from mmengine import Config


CONFIG_DIR = Path('configs/ov_capflow/hrsc')


@pytest.mark.parametrize('name,model_type', [
    ('grounding_dino_swin-t_hrsc_parent_10e.py', 'RotatedGroundingDINO'),
    ('ov_capflow_swin-t_hrsc_c0_native_10e.py', 'OVCapFlow'),
    ('ov_capflow_swin-t_hrsc_c1_fusion_5e.py', 'OVCapFlow'),
    ('ov_capflow_swin-t_hrsc_c2_balanced_5e.py', 'OVCapFlow'),
    ('ov_capflow_swin-t_hrsc_c3_null_5e.py', 'OVCapFlow'),
])
def test_hrsc_configs_share_fixed_protocol(name, model_type):
    cfg = Config.fromfile(CONFIG_DIR / name)
    assert cfg.model.type == model_type
    assert cfg.model.num_queries == 600
    assert cfg.train_dataloader.batch_size == 3
    assert cfg.train_dataloader.dataset.ann_file == 'ImageSets/trainval.txt'
    assert cfg.val_dataloader.dataset.ann_file == 'ImageSets/test.txt'
    assert cfg.val_evaluator.type == 'DOTAMetric'
    assert cfg.val_evaluator.iou_thrs == 0.5
    assert cfg.randomness.seed == 20260712


def test_causal_switches_are_single_direction():
    c0 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c0_native_10e.py')
    c1 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c1_fusion_5e.py')
    c2 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c2_balanced_5e.py')
    c3 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c3_null_5e.py')
    assert not c0.model.decoder.layer_cfg.enable_semantic_fusion
    assert c1.model.decoder.layer_cfg.enable_semantic_fusion
    assert not c1.model.bbox_head.balanced_cfg.enabled
    assert c2.model.bbox_head.balanced_cfg.enabled
    assert not c2.model.decoder.enable_null_reservoir
    assert c3.model.decoder.enable_null_reservoir
    assert not c3.model.decoder.layer_cfg.enable_density_capacity
```

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_hrsc_config.py -q
```

Expected: config files are missing.

- [ ] **Step 3: Create the upstream HRSC parent config**

Use the Oriented GroundingDINO VisDrone config as base and override the full
dataset/training protocol:

```python
_base_ = [
    '../../../projects/GroundingDINO/configs/'
    'grounding_dino_swin-t_visdrone_base-set_adamw.py'
]

data_root = '/data/zcy/dataset/HRSC_unzip/'
num_queries = 600
batch_size = 3

model = dict(
    num_queries=num_queries,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data/zcy/swin_tiny_patch4_window7_224.pth')),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(max_per_img=num_queries))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip', prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='HRSCDataset', data_root=data_root,
        ann_file='ImageSets/trainval.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline, return_classes=True))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='HRSCDataset', data_root=data_root,
        ann_file='ImageSets/test.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True, pipeline=val_pipeline, return_classes=True))
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=10, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=10, by_epoch=True,
        milestones=[8], gamma=0.1),
]
optim_wrapper = dict(accumulative_counts=2)
val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=0.5)
test_evaluator = val_evaluator
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=5, max_keep_ckpts=3,
        save_best='dota/mAP', rule='greater', save_last=True))
log_processor = dict(by_epoch=True)
randomness = dict(seed=20260712, deterministic=False, diff_rank_seed=False)
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_p0_parent_10e'
```

- [ ] **Step 4: Create the strict and causal configs**

`C0` inherits P0 and replaces only the detector/head/decoder switches:

```python
_base_ = './grounding_dino_swin-t_hrsc_parent_10e.py'
custom_imports = dict(
    imports=['projects.OVCapFlow.ov_capflow'], allow_failed_imports=False)
model = dict(
    type='OVCapFlow',
    decoder=dict(
        enable_null_reservoir=False,
        layer_cfg=dict(
            enable_semantic_fusion=False,
            enable_density_capacity=False)),
    bbox_head=dict(
        type='OVCapFlowHead',
        balanced_cfg=dict(enabled=False),
        readout_cfg=dict(temperature=1.0, power=1.0, use_capacity=False)),
    density_loss_cfg=dict(weight=0.0),
    null_loss_cfg=dict(),
    test_cfg=dict(_delete_=True))
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e'
```

Create `C1` with this exact content:

```python
_base_ = './ov_capflow_swin-t_hrsc_c0_native_10e.py'

model = dict(
    decoder=dict(
        enable_null_reservoir=False,
        layer_cfg=dict(
            enable_semantic_fusion=True,
            enable_density_capacity=False,
            semantic_fusion_cfg=dict(adapter_init='identity'))),
    bbox_head=dict(
        balanced_cfg=dict(
            enabled=False, matched_weight=1.0, unmatched_weight=1.0)))
load_from = 'work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e/epoch_10.pth'
train_cfg = dict(max_epochs=5, val_interval=5)
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=5, by_epoch=True,
        milestones=[4], gamma=0.1),
]
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        trainable_patterns=[
            r'^decoder\.layers\.\d+\.semantic_fusion\.',
        ])
]
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c1_fusion_5e'
```

Create `C2`:

```python
_base_ = './ov_capflow_swin-t_hrsc_c1_fusion_5e.py'
model = dict(
    bbox_head=dict(
        balanced_cfg=dict(
            enabled=True, matched_weight=1.0, unmatched_weight=1.0)))
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c2_balanced_5e'
```

Create `C3`:

```python
_base_ = './ov_capflow_swin-t_hrsc_c2_balanced_5e.py'
model = dict(
    decoder=dict(
        enable_null_reservoir=True,
        null_reservoir_cfg=dict()),
    null_loss_cfg=dict(
        matched_weight=1.0,
        unmatched_weight=1.0,
        mass_weight=0.1,
        gate_order_weight=0.1,
        gate_margin=0.0))
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        trainable_patterns=[
            r'^decoder\.layers\.\d+\.semantic_fusion\.',
            r'^decoder\.null_reservoir\.',
        ])
]
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c3_null_5e'
```

All three inherit density disabled from C0 and load the same C0 checkpoint.

- [ ] **Step 5: Run config tests and dataset build**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_hrsc_config.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python tools/misc/browse_dataset.py configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c0_native_10e.py --output-dir /tmp/ovcapflow_hrsc_browse --not-show
```

Expected: config tests pass and the browser processes HRSC samples without missing files, missing text metadata, or box-conversion errors.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add configs/ov_capflow/hrsc tests/test_projects/ov_capflow/test_hrsc_config.py
rtk git commit -m "feat: add matched HRSC causal configs"
```

## Task 8: Add fail-closed adapter freezing and checkpoint audit

**Files:**
- Create: `projects/OVCapFlow/ov_capflow/freeze_except_hook.py`
- Create: `tests/test_projects/ov_capflow/test_freeze_except_hook.py`
- Create: `projects/OVCapFlow/tools/audit_checkpoint_load.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Modify: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_fusion_5e.py`
- Modify: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c2_balanced_5e.py`
- Modify: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c3_null_5e.py`

- [ ] **Step 1: Write failing freeze-hook tests**

Create the complete test module:

```python
import pytest
import torch.nn as nn

from projects.OVCapFlow.ov_capflow.freeze_except_hook import FreezeExceptHook


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_fusion = nn.Linear(4, 4)


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(), ToyLayer()])
        self.null_reservoir = nn.Linear(4, 1)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.decoder = ToyDecoder()


def test_freeze_except_hook_is_fail_closed():
    model = ToyModel()
    hook = FreezeExceptHook(
        trainable_patterns=[r'^decoder\.layers\.\d+\.semantic_fusion\.'])
    hook.apply(model)
    trainable = [name for name, value in model.named_parameters()
                 if value.requires_grad]
    assert trainable
    assert all('.semantic_fusion.' in name for name in trainable)


def test_freeze_except_hook_rejects_empty_match():
    with pytest.raises(RuntimeError, match='matched no parameters'):
        FreezeExceptHook([r'^missing\.']).apply(ToyModel())
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_freeze_except_hook.py -q
```

Expected: import fails because the hook does not exist.

- [ ] **Step 3: Implement the hook**

Create a registered MMEngine hook with an `apply(model)` method:

```python
import re

from mmengine.hooks import Hook
from mmrotate.registry import HOOKS


@HOOKS.register_module()
class FreezeExceptHook(Hook):
    def __init__(self, trainable_patterns):
        self.patterns = [re.compile(pattern) for pattern in trainable_patterns]

    def apply(self, model):
        matched = []
        for name, parameter in model.named_parameters():
            enabled = any(pattern.search(name) for pattern in self.patterns)
            parameter.requires_grad_(enabled)
            if enabled:
                matched.append(name)
        if not matched:
            raise RuntimeError('trainable patterns matched no parameters')
        return matched

    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') \
            else runner.model
        matched = self.apply(model)
        runner.logger.info('trainable_parameters=%s', matched)
```

- [ ] **Step 4: Configure exact trainable patterns**

Use these common patterns for C1/C2/C3:

```python
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        trainable_patterns=[
            r'^decoder\.layers\.\d+\.semantic_fusion\.',
        ])
]
```

C3 adds `r'^decoder\.null_reservoir\.'`. Balanced reduction adds no parameter;
its gradients act through the same semantic-fusion parameters.

- [ ] **Step 5: Implement checkpoint audit output**

Create `audit_checkpoint_load.py`:

```python
import argparse
import json
import re
from pathlib import Path

import torch
from mmengine import Config
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import HOOKS, MODELS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def unwrap_state_dict(checkpoint):
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    return {re.sub(r'^module\.', '', key): value
            for key, value in state.items()}


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    incompatible = model.load_state_dict(
        unwrap_state_dict(checkpoint), strict=False)

    layer_cfg = cfg.model.decoder.layer_cfg
    semantic_enabled = bool(layer_cfg.enable_semantic_fusion)
    density_enabled = bool(layer_cfg.enable_density_capacity)
    null_enabled = bool(cfg.model.decoder.enable_null_reservoir)
    allowed_missing = []
    if semantic_enabled:
        allowed_missing.append(re.compile(
            r'^decoder\.layers\.\d+\.semantic_fusion\.'))
    if density_enabled:
        allowed_missing.append(re.compile(
            r'^decoder\.layers\.\d+\.density_capacity\.'))
    if null_enabled:
        allowed_missing.append(re.compile(r'^decoder\.null_reservoir\.'))
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not any(pattern.search(key) for pattern in allowed_missing)]

    trainable = []
    for hook_cfg in cfg.get('custom_hooks', []):
        if hook_cfg.get('type') == 'FreezeExceptHook':
            trainable = HOOKS.build(hook_cfg).apply(model)

    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'invalid_missing_keys': invalid_missing,
        'enable_semantic_fusion': semantic_enabled,
        'enable_density_capacity': density_enabled,
        'enable_null_reservoir': null_enabled,
        'balanced_enabled': bool(
            cfg.model.bbox_head.balanced_cfg.enabled),
        'trainable_parameters': trainable,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    if invalid_missing or incompatible.unexpected_keys:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
```

It writes JSON with this schema:

```json
{
  "checkpoint": ".../epoch_10.pth",
  "missing_keys": [],
  "unexpected_keys": [],
  "enable_semantic_fusion": true,
  "enable_density_capacity": false,
  "enable_null_reservoir": false,
  "balanced_enabled": false,
  "trainable_parameters": []
}
```

The script exits non-zero when an unexpected key exists or a missing key is not
owned by a newly enabled intervention.

- [ ] **Step 6: Run tests and audit the config build**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_freeze_except_hook.py tests/test_projects/ov_capflow/test_hrsc_config.py -q
```

Expected: tests pass. Checkpoint audit is run after H1 creates `epoch_10.pth`.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/freeze_except_hook.py projects/OVCapFlow/ov_capflow/__init__.py projects/OVCapFlow/tools/audit_checkpoint_load.py tests/test_projects/ov_capflow/test_freeze_except_hook.py configs/ov_capflow/hrsc
rtk git commit -m "feat: freeze and audit HRSC adapter parameters"
```

## Task 9: Add the real-HRSC H0 integration gate

**Files:**
- Create: `tests/test_projects/ov_capflow/test_hrsc_real_batch.py`
- Create: `projects/OVCapFlow/tools/audit_strict_inference.py`

- [ ] **Step 1: Write the opt-in integration test**

Create the complete CUDA test guarded by `RUN_OVCAPFLOW_INTEGRATION=1`:

```python
import os
from pathlib import Path

import pytest
import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


@pytest.mark.skipif(
    os.getenv('RUN_OVCAPFLOW_INTEGRATION') != '1',
    reason='real HRSC integration is opt-in')
@pytest.mark.parametrize('config_name', [
    'ov_capflow_swin-t_hrsc_c0_native_10e.py',
    'ov_capflow_swin-t_hrsc_c1_fusion_5e.py',
    'ov_capflow_swin-t_hrsc_c2_balanced_5e.py',
    'ov_capflow_swin-t_hrsc_c3_null_5e.py',
])
def test_real_hrsc_batch_forward_backward(config_name):
    assert torch.cuda.is_available()
    cfg = Config.fromfile(Path('configs/ov_capflow/hrsc') / config_name)
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model).cuda().train()
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    batch = next(iter(dataloader))
    batch = model.data_preprocessor(batch, training=True)
    losses = model.loss(batch['inputs'], batch['data_samples'])
    total = sum(value for value in losses.values()
                if torch.is_tensor(value) and value.requires_grad)
    assert losses
    assert all(torch.isfinite(value).all() for value in losses.values()
               if torch.is_tensor(value))
    total.backward()
    active = [name for name, parameter in model.named_parameters()
              if parameter.requires_grad and parameter.grad is not None]
    assert active
    if 'c0_native' in config_name:
        assert not any(name.startswith('loss_null_') for name in losses)
        assert 'loss_capacity_mass' not in losses
    if 'c1_fusion' in config_name:
        assert model.bbox_head.last_balance_stats is None
    if 'c2_balanced' in config_name:
        assert model.bbox_head.last_balance_stats is not None
    if 'c3_null' in config_name:
        assert {'loss_null_matched', 'loss_null_unmatched',
                'loss_null_mass', 'loss_gate_order'} <= set(losses)
```

- [ ] **Step 2: Run the portable suite**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Expected: all unit/config tests pass and four integration cases are skipped.

- [ ] **Step 3: Run the real CUDA gate on GPU4**

Run:

```bash
rtk bash -lc 'CUDA_VISIBLE_DEVICES=4 RUN_OVCAPFLOW_INTEGRATION=1 MPLCONFIGDIR=/tmp/ovcapflow-mpl /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_hrsc_real_batch.py -q'
```

Expected: four cases pass; every active loss is finite and every enabled
intervention has a gradient.

- [ ] **Step 4: Implement and run the strict audit**

Create `audit_strict_inference.py`:

```python
import argparse
import ast
import json
from pathlib import Path

import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


FORBIDDEN = {
    'topk', 'nms', 'rotated_nms', 'multiclass_nms', 'minAreaRect'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def scan_calls(root):
    hits = []
    for path in sorted(Path(root).rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) \
                else function.id if isinstance(function, ast.Name) else ''
            if name in FORBIDDEN:
                hits.append({'path': str(path), 'line': node.lineno,
                             'call': name})
    return hits


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    model = MODELS.build(cfg.model).cuda().eval()
    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    batch = next(iter(dataloader))
    batch = model.data_preprocessor(batch, training=False)
    with torch.no_grad():
        predictions = model.predict(
            batch['inputs'], batch['data_samples'], rescale=True)
    counts = [len(sample.pred_instances) for sample in predictions]
    hits = scan_calls('projects/OVCapFlow/ov_capflow')
    report = {
        'pass': not hits and counts == [cfg.model.num_queries] * len(counts),
        'num_matching_queries': int(cfg.model.num_queries),
        'predictions_per_image': counts,
        'forbidden_calls': hits,
        'uses_nms': any('nms' in hit['call'] for hit in hits),
        'uses_topk': any(hit['call'] == 'topk' for hit in hits),
        'uses_min_area_rect': any(
            hit['call'] == 'minAreaRect' for hit in hits),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
```

Run:

```bash
rtk bash -lc 'CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/ovcapflow-mpl /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_strict_inference.py configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c0_native_10e.py --output /tmp/ovcapflow_hrsc_strict_audit.json'
```

Expected JSON contains `"pass": true`, `"predictions_per_image": [600]`, and
all forbidden-operation booleans are false.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add tests/test_projects/ov_capflow/test_hrsc_real_batch.py projects/OVCapFlow/tools/audit_strict_inference.py
rtk git commit -m "test: gate OV-CapFlow on a real HRSC batch"
```

## Task 10: Initialize the researcher ledger and branch

**Files:**
- Modify: `.gitignore`
- Create untracked: `.lab/config.md`
- Create untracked: `.lab/results.tsv`
- Create untracked: `.lab/log.md`
- Create untracked: `.lab/branches.md`
- Create untracked: `.lab/parking-lot.md`
- Create untracked: `.lab/bin/run`
- Create untracked: `.lab/bin/measure`
- Create untracked: `.lab/workspace/`

- [ ] **Step 1: Add the ledger exclusions**

Append to `.gitignore`:

```gitignore
.lab/
run.log
```

- [ ] **Step 2: Commit the ignore rule before creating `.lab`**

Run:

```bash
rtk git add .gitignore
rtk git commit -m "chore: ignore local experiment ledger"
```

- [ ] **Step 3: Create the research branch**

Run:

```bash
rtk git switch -c research/hrsc-calibrated-capacity
```

Expected: the active branch is `research/hrsc-calibrated-capacity`.

- [ ] **Step 4: Create the `.lab` files with the confirmed contract**

`.lab/config.md` records:

```markdown
# HRSC OV-CapFlow Research Configuration

- Objective: improve strict HRSC AP50 through query-preserving fusion,
  balanced supervision, and explicit null calibration.
- Primary metric: `dota/mAP` at IoU 0.5; higher is better.
- Secondary metrics: recall, GT coverage, duplicate extras/GT, null Brier,
  matched/unmatched gate gap, peak memory, iteration time.
- Scope: `projects/OVCapFlow`, `configs/ov_capflow/hrsc`,
  `tests/test_projects/ov_capflow`, `.lab`.
- Off-limits: old GSOVD code/checkpoints as trainable parents, DOTA2,
  teacher pseudo labels, NMS/top-k/dense inference, density sweep.
- Per-run timeout: 7200 seconds.
- First seed: 20260712.
- Repeat seed: 20260713.
- Target: AP50 delta >= 0.002 over C0R, or AP tie within 0.001 plus two
  pre-registered mediator improvements.
- Termination: reproducible H3 winner, researcher discard/plateau guardrail,
  or user interruption.
```

`results.tsv` header is:

```text
experiment	branch	parent	commit	metric	secondary_metrics	status	duration_s	description
```

`branches.md` header is:

```markdown
| Branch | Forked from | Status | Experiments | Best metric | Notes |
|---|---|---|---:|---:|---|
| research/hrsc-calibrated-capacity | baseline | active | 0 | n/a | H1/H2 canonical line |
```

- [ ] **Step 5: Create robust run and measure helpers**

`.lab/bin/run` accepts GPU, config, and work directory, then executes:

```bash
#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
config="$2"
work_dir="$3"
mkdir -p "$work_dir"
CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/ovcapflow-mpl \
  /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py "$config" \
  --work-dir "$work_dir" 2>&1 | tee "$work_dir/run.log"
```

`.lab/bin/measure` fails if the metric is absent and emits one stripped number:

```bash
#!/usr/bin/env bash
set -euo pipefail
log_file="$1"
value=$(rg -o "dota/mAP[^0-9]*[0-9]+\.[0-9]+" "$log_file" \
  | tail -n 1 | rg -o "[0-9]+\.[0-9]+")
test -n "$value"
printf '%s' "$value" | tr -d '[:space:]'
```

Run `rtk chmod +x .lab/bin/run .lab/bin/measure`.

- [ ] **Step 6: Record H0 as experiment 0**

Record commit SHA, focused-test count, real-batch result, strict audit path,
runtime, and status `keep` in `.lab/results.tsv` and `.lab/log.md`. H0's metric
is `n/a`; H1 establishes the first AP baseline.

## Task 11: Run H1 anchors and validate C0

**Files:**
- Write untracked: `.lab/log.md`
- Write untracked: `.lab/results.tsv`
- Write ignored: `work_dirs/ov_capflow_hrsc/hrsc_p0_parent_10e/**`
- Write ignored: `work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e/**`

- [ ] **Step 1: Log the H1 hypotheses before running**

Add `THINK` entries stating:

- P0 tests whether the public Oriented GroundingDINO shell trains correctly on
  the local HRSC mouth.
- C0 tests whether fixed learned queries and references form a viable strict
  substrate without semantic/density/null interventions.
- P0 and C0 are parallel anchors, not a direct architecture delta.

- [ ] **Step 2: Launch P0 and C0 on GPUs 4 and 5**

Run each through the researcher helper in a separate managed session:

```bash
rtk .lab/bin/run 4 configs/ov_capflow/hrsc/grounding_dino_swin-t_hrsc_parent_10e.py work_dirs/ov_capflow_hrsc/hrsc_p0_parent_10e
rtk .lab/bin/run 5 configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c0_native_10e.py work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e
```

Expected: validation at epochs 5 and 10, no NaN/Inf/OOM, and checkpoints saved.
Each run is terminated at 7200 seconds if incomplete.

- [ ] **Step 3: Measure and log both anchors**

Run:

```bash
rtk .lab/bin/measure work_dirs/ov_capflow_hrsc/hrsc_p0_parent_10e/run.log
rtk .lab/bin/measure work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e/run.log
```

Log metric, recall, runtime, peak memory, final loss, checkpoint path, and SHA256
before any decision.

- [ ] **Step 4: Re-evaluate C0 twice**

Run the same `epoch_10.pth` twice with `tools/test.py --out` into separate
directories. Expected absolute AP50 difference: at most `0.001`.

- [ ] **Step 5: Apply the H1 gate**

C0 passes when AP50 is at least `0.50`, strict audit passes, all 600 query rows
are emitted, and repeated evaluation is stable. If it fails, log
`invalid-engineering` for routing/config errors or `discard` for a valid weak
substrate; do not start H2.

- [ ] **Step 6: Audit C1/C2/C3 checkpoint loading**

Run `audit_checkpoint_load.py` for each adapter config against C0 epoch 10.
Expected: only semantic fusion and, for C3, null-reservoir keys are missing;
there are no unexplained unexpected keys.

## Task 12: Run the H2 causal matrix and H3 repeat

**Files:**
- Write untracked: `.lab/log.md`
- Write untracked: `.lab/results.tsv`
- Write untracked: `.lab/branches.md`
- Write ignored: `work_dirs/ov_capflow_hrsc/hrsc_c1_fusion_5e/**`
- Write ignored: `work_dirs/ov_capflow_hrsc/hrsc_c2_balanced_5e/**`
- Write ignored: `work_dirs/ov_capflow_hrsc/hrsc_c3_null_5e/**`

- [ ] **Step 1: Replay C0 without updates**

Evaluate C0 epoch 10 on GPU4 before launching interventions. Log the AP50 as
`HRSC-C0R`. Re-evaluate after the intervention runs to detect evaluator drift.

- [ ] **Step 2: Commit each experiment config before running**

For any config-only correction, use the researcher commit format:

```text
experiment #N: <short hypothesis>

Branch: research/hrsc-calibrated-capacity
Parent: #M
Hypothesis: <one-line causal claim>
```

No modified repository file may be executed before its commit exists.

- [ ] **Step 3: Launch C1/C2/C3 on GPUs 5/6/7**

Run:

```bash
rtk .lab/bin/run 5 configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_fusion_5e.py work_dirs/ov_capflow_hrsc/hrsc_c1_fusion_5e
rtk .lab/bin/run 6 configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c2_balanced_5e.py work_dirs/ov_capflow_hrsc/hrsc_c2_balanced_5e
rtk .lab/bin/run 7 configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c3_null_5e.py work_dirs/ov_capflow_hrsc/hrsc_c3_null_5e
```

Expected: each completes within two hours with identical data budget and only
the pre-registered trainable parameters.

- [ ] **Step 4: Log before deciding**

For each arm record AP50, recall, GT coverage, duplicate extras/GT, null Brier,
matched/unmatched gate means, gate gap, peak memory, duration, commit, parent,
checkpoint SHA256, and effective switches. Update `.lab/results.tsv` and
`.lab/log.md` before reset/keep decisions.

- [ ] **Step 5: Apply pre-registered promotion rules**

Promote when AP50 delta over C0R is at least `+0.002`. A tie within `0.001` may
be `interesting` only when at least two mediator gates pass: null Brier -10%,
positive gate gap -20% or non-positive, duplicate extras/GT -10%, or GT coverage
`+0.005`, with recall drop no worse than `0.005`.

- [ ] **Step 6: Repeat a promoted winner**

Create a seed-20260713 config inheriting the winner, commit it, and run on the
first available GPU. If the repeat retains the gate, continue that arm to epoch
10 with validation at every epoch and calibration early stopping. Otherwise log
the original as single-seed/parked.

- [ ] **Step 7: Produce the HRSC transfer report**

Create `docs/project_history/exp_20260714_hrsc_fast_iteration/fres_hrsc_ov_capflow_fast_iteration_zh.md` containing:

- exact mouth and environment;
- H0 audit;
- P0/C0 AP50 at epochs 5/10;
- C0 replay consistency;
- C1/C2/C3 primary and mediator table;
- seed repeat;
- keep/park/stop decisions;
- the single mechanism, if any, allowed to transfer to DOTA2;
- explicit statement that HRSC does not prove open-vocabulary transfer.

- [ ] **Step 8: Re-validate and close the researcher run**

Re-run the global best checkpoint, write `.lab/summary.md`, checkout the branch
containing the global best, and commit repository state with:

```bash
rtk git add docs/project_history/exp_20260714_hrsc_fast_iteration configs/ov_capflow/hrsc projects/OVCapFlow tests/test_projects/ov_capflow
rtk git commit -m "research complete: validate OV-CapFlow on HRSC"
```

Expected: the active branch contains only kept implementation/experiment
changes; `.lab` retains the full keep/discard/crash history outside Git.
