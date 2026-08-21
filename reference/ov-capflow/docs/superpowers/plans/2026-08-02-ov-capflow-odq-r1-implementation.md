# OV-CapFlow ODQ-R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest possible oriented distributional query (ODQ) residual branch to the existing rotated-box E2E OV-CapFlow pipeline, preserve the parent T7 behavior exactly at zero step, and produce a five-check audit that must pass before any training is allowed.

**Architecture:** Keep every existing parent regression module and checkpoint key unchanged. Add one parallel four-coordinate distribution branch per decoder layer, expose non-registered combined regression callables to the existing decoder, and reuse the already-computed rotated Hungarian assignments to supervise positive x/y/w/h residual distributions. The angle, Q600 mouth, grouped-query structure, DN path, matching costs, GDLoss, and all-row E2E readout remain unchanged.

**Tech Stack:** Python 3.8, PyTorch 1.12.1, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate project modules, pytest, two physical A40 GPUs numbered 8 and 9 for later training.

---

## Frozen scientific contract

- Parent recipe: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py` at the implementation commit's recorded SHA.
- Query mouth: exactly Q600 at inference and the existing grouped/DN expansion at training; never add a query selector.
- Geometry: only x, y, w, h receive ODQ residuals in inverse-sigmoid coordinates. The fifth rotated-box angle coordinate is bitwise inherited from the parent branch.
- Distribution: 17 symmetric support points on `[-1, 1]`, radii `(2.0, 2.0, 2.0, 2.0)`, centered expectation, and zero-initialized final logits.
- S0 control: `alpha=0.0`, distribution loss weight `0.0`.
- R1 candidate: `alpha=1.0`, distribution loss weight `1.0`.
- First cycle exclusions: no distribution-aware matching, no independent quality head, no score calibration, no NMS, no top-k, no post-hoc filtering, no encoder-proposal path, and no semantic-first branch.
- Compatibility boundary: ODQ may coexist with the T7 grouped matching and DN paths. It must reject balanced matching, position-supervised matching, existence loss, and adaptive-DN variants until separately specified.
- Training authorization boundary: completing this plan does not authorize training. Training may start only after the five-check report is `PASS` and the overnight supervisor plan is implemented and dry-run audited.

## Environment and command contract

All commands run from `/data1/zcy/OV-CapFlow`. Every shell command is prefixed by `rtk`. Use the established project interpreter rather than the user-site Python:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -c "import torch, mmcv, mmengine, mmdet; print(torch.__version__, mmcv.__version__, mmengine.__version__, mmdet.__version__)"
```

Expected environment identity:

```text
1.12.1+cu113 2.1.0 0.10.4 3.3.0
```

GPU checks or later real-batch audits must expose only physical GPU 8:

```bash
rtk env CUDA_VISIBLE_DEVICES=8 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_stage0.py -q
```

## Task 1: Implement and unit-test the centered distribution mathematics

**Files:**

- Create: `projects/OVCapFlow/ov_capflow/oriented_distribution.py`
- Create: `tests/test_projects/ov_capflow/test_oriented_distribution.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`

- [ ] **Step 1: Write the failing tests for configuration and exact centered expectation**

Add focused tests with these assertions:

```python
import pytest
import torch

from projects.OVCapFlow.ov_capflow.oriented_distribution import (
    OrientedResidualDistributionBranch,
    centered_expectation,
    distribution_residual_loss,
    soft_bin_targets,
    symmetric_support,
)


def test_support_requires_odd_bin_count_and_is_symmetric():
    support = symmetric_support(17, device=torch.device('cpu'), dtype=torch.float32)
    assert support.shape == (17,)
    assert torch.equal(support, -support.flip(0))
    assert support[8].item() == 0.0
    with pytest.raises(ValueError, match='odd'):
        symmetric_support(16, device=torch.device('cpu'), dtype=torch.float32)


def test_zero_logits_have_bitwise_zero_centered_expectation():
    logits = torch.zeros(2, 600, 4, 17, dtype=torch.float32)
    residual = centered_expectation(logits, radii=(2.0, 2.0, 2.0, 2.0))
    assert residual.shape == (2, 600, 4)
    assert torch.count_nonzero(residual).item() == 0


def test_positive_edge_mass_moves_only_selected_coordinate():
    logits = torch.zeros(1, 1, 4, 17)
    logits[0, 0, 2, -1] = 12.0
    residual = centered_expectation(logits, radii=(2.0, 2.0, 2.0, 2.0))
    assert residual[0, 0, 2].item() > 1.9
    assert torch.count_nonzero(residual[0, 0, [0, 1, 3]]).item() == 0
```

- [ ] **Step 2: Run the tests and confirm the import failure**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_oriented_distribution.py -q
```

Expected: `ModuleNotFoundError` for `oriented_distribution`.

- [ ] **Step 3: Implement validation, support, and centered expectation**

Implement the public core with explicit validation and no NaN repair:

```python
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validated_radii(radii: Sequence[float]) -> Tuple[float, float, float, float]:
    values = tuple(float(value) for value in radii)
    if len(values) != 4 or any(value <= 0.0 for value in values):
        raise ValueError('radii must contain four positive values')
    return values


def symmetric_support(num_bins: int, *, device, dtype) -> Tensor:
    if num_bins < 3 or num_bins % 2 == 0:
        raise ValueError('num_bins must be odd and at least three')
    return torch.linspace(-1.0, 1.0, num_bins, device=device, dtype=dtype)


def centered_expectation(logits: Tensor, radii: Sequence[float]) -> Tensor:
    if logits.ndim < 3 or logits.shape[-2] != 4:
        raise ValueError('logits must end in [4, num_bins]')
    if not logits.is_floating_point():
        raise TypeError('logits must be floating point')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('logits contain non-finite values')
    values = _validated_radii(radii)
    support = symmetric_support(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    probabilities = logits.softmax(dim=-1)
    centered = probabilities - torch.zeros_like(logits).softmax(dim=-1)
    expectation = (centered * support).sum(dim=-1)
    radius = logits.new_tensor(values)
    return expectation * radius
```

- [ ] **Step 4: Add failing tests for soft targets and positive-only loss**

Cover exact-bin, interpolated-bin, clipping, detached parent targets, empty positives, and non-finite input:

```python
def test_soft_bin_targets_linearly_interpolate_between_neighbors():
    values = torch.tensor([-1.0, -0.9375, 0.0, 0.9375, 1.0, 3.0])
    targets = soft_bin_targets(values, num_bins=17)
    assert targets.shape == (6, 17)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(6))
    assert targets[0, 0].item() == 1.0
    assert targets[2, 8].item() == 1.0
    assert targets[-1, -1].item() == 1.0
    assert torch.count_nonzero(targets[1]).item() == 2


def test_distribution_loss_uses_positive_xywh_rows_only():
    logits = torch.zeros(1, 3, 4, 17, requires_grad=True)
    parent = torch.full((1, 3, 5), 0.5, requires_grad=True)
    target = parent.detach().clone()
    target[0, 0, 0] = 0.6
    weights = torch.zeros_like(target)
    weights[0, 0] = 1.0
    loss, telemetry = distribution_residual_loss(
        logits, parent, target, weights, radii=(2.0, 2.0, 2.0, 2.0))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0
    assert parent.grad is None
    assert telemetry['num_positive'].item() == 1


def test_distribution_loss_empty_positive_set_is_graph_connected_zero():
    logits = torch.randn(2, 5, 4, 17, requires_grad=True)
    parent = torch.full((2, 5, 5), 0.5)
    target = parent.clone()
    weights = torch.zeros_like(parent)
    loss, telemetry = distribution_residual_loss(
        logits, parent, target, weights, radii=(2.0, 2.0, 2.0, 2.0))
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0
    assert telemetry['num_positive'].item() == 0
```

- [ ] **Step 5: Run the targeted tests and confirm missing-function failures**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_oriented_distribution.py -q
```

Expected: failures name `soft_bin_targets` or `distribution_residual_loss`.

- [ ] **Step 6: Implement soft targets and positive-only cross entropy**

Use inverse-sigmoid residuals only for x/y/w/h and detach the parent boxes when constructing targets:

```python
def soft_bin_targets(values: Tensor, num_bins: int) -> Tensor:
    support = symmetric_support(num_bins, device=values.device, dtype=values.dtype)
    clipped = values.clamp(min=-1.0, max=1.0)
    scaled = (clipped - support[0]) * (num_bins - 1) / 2.0
    lower = scaled.floor().long().clamp(max=num_bins - 1)
    upper = (lower + 1).clamp(max=num_bins - 1)
    upper_weight = scaled - lower.to(scaled.dtype)
    lower_weight = 1.0 - upper_weight
    result = values.new_zeros(values.shape + (num_bins,))
    result.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    result.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return result


def _inverse_sigmoid(value: Tensor, eps: float) -> Tensor:
    clipped = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(clipped / (1.0 - clipped))


def distribution_residual_loss(logits: Tensor, parent_boxes: Tensor,
                               bbox_targets: Tensor, bbox_weights: Tensor,
                               radii: Sequence[float], eps: float = 1e-4):
    if logits.shape[:-2] != parent_boxes.shape[:-1] or logits.shape[-2] != 4:
        raise ValueError('ODQ logits and parent boxes have incompatible shapes')
    if parent_boxes.shape != bbox_targets.shape or parent_boxes.shape != bbox_weights.shape:
        raise ValueError('parent, target, and weight boxes must have identical shapes')
    values = _validated_radii(radii)
    positive = bbox_weights[..., :4].sum(dim=-1) > 0
    count = positive.sum()
    if count.item() == 0:
        zero = logits.sum() * 0.0
        return zero, {'num_positive': count.detach(),
                      'boundary_hit_rate': zero.detach(),
                      'mean_abs_normalized_residual': zero.detach()}
    parent_xywh = parent_boxes.detach()[..., :4][positive]
    target_xywh = bbox_targets[..., :4][positive]
    radius = logits.new_tensor(values)
    normalized = (_inverse_sigmoid(target_xywh, eps) -
                  _inverse_sigmoid(parent_xywh, eps)) / radius
    soft_targets = soft_bin_targets(normalized, logits.shape[-1])
    selected_logits = logits[positive]
    loss = -(soft_targets * F.log_softmax(selected_logits, dim=-1)).sum(dim=-1).mean()
    telemetry = {
        'num_positive': count.detach(),
        'boundary_hit_rate': (normalized.abs() >= 1.0).float().mean().detach(),
        'mean_abs_normalized_residual': normalized.abs().mean().detach(),
    }
    return loss, telemetry
```

- [ ] **Step 7: Add the branch and verify zero initialization**

The branch produces exactly `[... , 4, 17]`; its last linear layer is zero-initialized:

```python
class OrientedResidualDistributionBranch(nn.Module):
    def __init__(self, embed_dims: int = 256, num_fcs: int = 2,
                 num_bins: int = 17,
                 radii: Sequence[float] = (2.0, 2.0, 2.0, 2.0)):
        super().__init__()
        if num_fcs < 1:
            raise ValueError('num_fcs must be positive')
        self.num_bins = int(num_bins)
        self.radii = _validated_radii(radii)
        symmetric_support(self.num_bins, device=torch.device('cpu'), dtype=torch.float32)
        layers = []
        for _ in range(num_fcs - 1):
            layers.extend((nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True)))
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(embed_dims, 4 * self.num_bins)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        logits = self.output(self.hidden(hidden_states))
        return logits.unflatten(-1, (4, self.num_bins))
```

Add a test that all final parameters are zero, the output shape is `(2, 600, 4, 17)`, and a backward pass produces finite gradients.

- [ ] **Step 8: Run the complete core test file**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_oriented_distribution.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit the core**

```bash
rtk git add projects/OVCapFlow/ov_capflow/oriented_distribution.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_oriented_distribution.py
rtk git diff --cached --check
rtk git commit -m "feat: add oriented residual distribution core"
```

## Task 2: Integrate ODQ into iterative decoder refinement without changing parent keys

**Files:**

- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow.py`
- Create: `tests/test_projects/ov_capflow/test_odq_head.py`

- [ ] **Step 1: Write failing tests for state keys and the combined delta**

Construct a minimal head fixture using the existing test conventions. Assert:

```python
def test_parent_regression_state_keys_are_unchanged(parent_head, odq_head):
    parent_keys = {key for key in parent_head.state_dict() if key.startswith('reg_branches.')}
    odq_parent_keys = {key for key in odq_head.state_dict() if key.startswith('reg_branches.')}
    assert odq_parent_keys == parent_keys
    new_keys = set(odq_head.state_dict()) - set(parent_head.state_dict())
    assert new_keys
    assert all(key.startswith('odq_branches.') for key in new_keys)


def test_alpha_zero_returns_bitwise_parent_delta(odq_s0_head):
    hidden = torch.randn(2, 600, odq_s0_head.embed_dims)
    parent = odq_s0_head.reg_branches[0](hidden)
    combined = odq_s0_head.combined_regression_delta(0, hidden)
    assert torch.equal(combined, parent)


def test_zero_initialized_alpha_one_keeps_xywha_exact(odq_r1_head):
    hidden = torch.randn(2, 600, odq_r1_head.embed_dims)
    parent = odq_r1_head.reg_branches[0](hidden)
    combined = odq_r1_head.combined_regression_delta(0, hidden)
    assert torch.equal(combined, parent)
    assert torch.equal(combined[..., 4], parent[..., 4])
```

Also assert `decoder_reg_branches()` returns a plain tuple, has one callable per decoder layer, and is absent from `named_modules()`.

- [ ] **Step 2: Run the integration tests and observe missing ODQ configuration**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_head.py -q
```

Expected: head construction or `combined_regression_delta` fails.

- [ ] **Step 3: Add strict ODQ configuration and parallel modules to the head**

Extend `OVCapFlowHead.__init__` with `odq_cfg=None`. Validate the exact first-cycle schema and reject incompatible experimental branches:

```python
def _init_odq(self, odq_cfg):
    self.odq_enabled = odq_cfg is not None
    if not self.odq_enabled:
        self.odq_alpha = 0.0
        self.odq_loss_weight = 0.0
        return
    allowed = {'num_bins', 'radii', 'alpha', 'loss_weight', 'num_fcs'}
    unknown = set(odq_cfg) - allowed
    if unknown:
        raise ValueError(f'unknown odq_cfg keys: {sorted(unknown)}')
    if self.balanced_cfg is not None or self.position_supervised_cfg is not None:
        raise ValueError('ODQ first cycle is incompatible with balanced or position supervision')
    if self.existence_loss_weight > 0 or self.adaptive_dn_cfg is not None:
        raise ValueError('ODQ first cycle is incompatible with existence or adaptive DN')
    self.odq_alpha = float(odq_cfg['alpha'])
    self.odq_loss_weight = float(odq_cfg['loss_weight'])
    self.odq_radii = tuple(float(value) for value in odq_cfg['radii'])
    self.odq_num_bins = int(odq_cfg['num_bins'])
    self.odq_branches = nn.ModuleList([
        OrientedResidualDistributionBranch(
            embed_dims=self.embed_dims,
            num_fcs=int(odq_cfg.get('num_fcs', 2)),
            num_bins=self.odq_num_bins,
            radii=self.odq_radii)
        for _ in range(self.num_pred_layer)
    ])
```

Use the head's actual initialized attribute names when inserting the validation; do not rename existing parent attributes or alter their defaults.

- [ ] **Step 4: Implement combined callables and an explicit detector decoder call**

The combined delta keeps the parent fifth coordinate exactly. Decoder-refinement
calls use this function without populating the loss cache:

```python
from functools import partial


def combined_regression_delta(self, layer_id: int, hidden_states: Tensor) -> Tensor:
    parent_delta = self.reg_branches[layer_id](hidden_states)
    if not self.odq_enabled or self.odq_alpha == 0.0:
        return parent_delta
    logits = self.odq_branches[layer_id](hidden_states)
    residual_xywh = centered_expectation(logits, self.odq_radii)
    correction = torch.cat((self.odq_alpha * residual_xywh,
                            torch.zeros_like(parent_delta[..., 4:5])), dim=-1)
    return parent_delta + correction


def decoder_reg_branches(self):
    return tuple(partial(self.combined_regression_delta, layer_id=index)
                 for index in range(len(self.reg_branches)))
```

Replace only `OVCapFlow.forward_decoder` with the upstream DINO-equivalent body, passing `self.bbox_head.decoder_reg_branches()` as `reg_branches`. Preserve the upstream DN embedding graph connection and then call the existing `_append_calibration_state` on the returned dictionary. Do not change `pre_decoder`, query initialization, decoder layers, or reference-point math.

- [ ] **Step 5: Override head forward so prediction and decoder refinement share one delta function**

Copy the installed `RotatedGroundingDINOHead.forward` control flow into
`OVCapFlowHead.forward`. Replace the regression call with a private helper that
returns the parent delta, distribution logits, and combined delta in one pass:

```python
parent_delta, odq_logits, tmp_reg_preds = self._odq_regression_outputs(
    layer_id, hidden_state)
reference = inverse_sigmoid(references[layer_id])
parent_boxes = (parent_delta + reference).sigmoid()
outputs_coord = (tmp_reg_preds + reference).sigmoid()
if self.odq_enabled:
    self._cache_odq_forward(layer_id, odq_logits, parent_boxes)
```

Implement `combined_regression_delta` by delegating to the same helper and
returning its third value, so the parent and ODQ branches are evaluated once per
call. The method must keep the existing classification branches. Cache the
parent-only absolute boxes using the same reference points so the loss target is
defined against the parent state, not against the ODQ-corrected box. Only head
`forward`, not decoder refinement, populates the loss cache.

- [ ] **Step 6: Test iterative references, exact angle inheritance, and all-row mouth**

Add tests using deterministic dummy hidden states and references:

- decoder layer `i+1` receives the combined ODQ update from layer `i`;
- output angles equal the parent output angles bitwise for all layers;
- an inference tensor with Q600 produces 600 class rows and 600 rotated boxes;
- no score sorting, slicing, NMS, top-k, or candidate mask occurs in forward/predict;
- S0 returns outputs bitwise equal to the parent path under identical inputs;
- R1 at initialization also returns outputs bitwise equal to the parent path.

- [ ] **Step 7: Run focused tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_head.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Inspect the checkpoint compatibility boundary**

Create a parent and ODQ head from real configs, compare state dictionaries, and assert that loading the parent checkpoint with `strict=False` reports only `odq_branches.*` missing keys and no unexpected keys. Save this assertion in the test file, not as an ad-hoc-only check.

- [ ] **Step 9: Commit integration**

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/ov_capflow/ov_capflow.py tests/test_projects/ov_capflow/test_odq_head.py
rtk git diff --cached --check
rtk git commit -m "feat: integrate ODQ iterative refinement"
```

## Task 3: Reuse existing grouped rotated assignments for ODQ loss

**Files:**

- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `projects/OVCapFlow/ov_capflow/oriented_distribution.py`
- Modify: `tests/test_projects/ov_capflow/test_odq_head.py`

- [ ] **Step 1: Write failing tests for assignment reuse and group/layer ordering**

Instrument `get_targets` with a call counter and build two matching-query groups across three decoder layers. Assert:

- enabling ODQ does not increase the parent's `get_targets` call count;
- ODQ contexts are consumed group-major, then layer-major, matching `grouped_matching_losses`;
- each context carries distribution logits, parent-only absolute boxes, and the matching targets from the same group/layer;
- final loss is `loss_odq`; auxiliary losses are `d0.loss_odq`, `d1.loss_odq`, and so on;
- group losses are averaged rather than summed, so duplicating an identical group does not double the scalar;
- DN rows are removed before grouped context splitting;
- an empty-positive batch yields finite graph-connected zero.

- [ ] **Step 2: Run the tests and observe missing ODQ loss keys**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_head.py -q -k "assignment or context or loss_odq or empty_positive"
```

Expected: assertions fail because no ODQ loss context exists.

- [ ] **Step 3: Add a scoped target capture around the existing assignment**

Do not call the assigner a second time. Use a private, call-scoped capture field:

```python
def get_targets(self, *args, **kwargs):
    result = super().get_targets(*args, **kwargs)
    if self.odq_enabled and self._odq_capture_targets:
        self._odq_last_targets = result
    return result
```

Preserve any existing D13 capture logic and make both captures independently gated. Clear the field in `finally` blocks so an exception cannot leak targets into the next batch.

- [ ] **Step 4: Prepare a deterministic ODQ context deque**

At the beginning of matching loss computation:

1. Strip exactly `num_denoising_queries` from cached logits and parent boxes.
2. Validate the remaining query count equals `matching_query_groups * num_queries`.
3. Split tensors by group in the same order as `grouped_matching_losses`.
4. Append contexts group-major and layer-major to `collections.deque`.
5. Refuse to start if the deque is non-empty, and require it to be empty after the parent loss returns.

Represent each item with a typed dataclass:

```python
@dataclass
class ODQLossContext:
    layer_id: int
    group_id: int
    logits: Tensor
    parent_boxes: Tensor
```

- [ ] **Step 5: Compute ODQ loss inside the existing `loss_by_feat_single` call**

Wrap the parent call so it performs the only matching assignment, then read captured `bbox_targets` and `bbox_weights` from that same call. Compute `distribution_residual_loss`, multiply by `self.odq_loss_weight`, and return the scalar and detached telemetry through a private accumulator. Preserve the public parent tuple shape expected by MMDetection.

After grouped matching finishes, average each layer's ODQ loss across groups and add:

```python
losses['loss_odq'] = layer_losses[-1]
for layer_id, loss in enumerate(layer_losses[:-1]):
    losses[f'd{layer_id}.loss_odq'] = loss
```

Keep telemetry out of the optimizer-facing loss dictionary; expose it via the audit cache with detached tensors.

- [ ] **Step 6: Add contract rejection and numerical tests**

Test explicit `ValueError` for every incompatible first-cycle option. Test finite forward/backward with extreme but finite logits. Test that NaN or Inf raises `FloatingPointError` and is never silently converted by `nan_to_num`, clipping, or zero replacement.

- [ ] **Step 7: Run core and integration suites**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_oriented_distribution.py tests/test_projects/ov_capflow/test_odq_head.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit assignment reuse**

```bash
rtk git add projects/OVCapFlow/ov_capflow/oriented_distribution.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_odq_head.py
rtk git diff --cached --check
rtk git commit -m "feat: train ODQ from grouped rotated assignments"
```

## Task 4: Freeze paired S0 and R1 full recipes

**Files:**

- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_s0.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_r1.py`
- Create: `tests/test_projects/ov_capflow/test_odq_configs.py`

- [ ] **Step 1: Write a failing paired-config test**

Load both configs with `mmengine.Config.fromfile` and assert:

```python
def test_s0_r1_differ_only_in_frozen_odq_switches_and_work_dir():
    s0 = load_config(S0_PATH)
    r1 = load_config(R1_PATH)
    assert s0.model.bbox_head.odq_cfg.num_bins == 17
    assert tuple(s0.model.bbox_head.odq_cfg.radii) == (2.0, 2.0, 2.0, 2.0)
    assert s0.model.bbox_head.odq_cfg.alpha == 0.0
    assert s0.model.bbox_head.odq_cfg.loss_weight == 0.0
    assert r1.model.bbox_head.odq_cfg.alpha == 1.0
    assert r1.model.bbox_head.odq_cfg.loss_weight == 1.0
    assert normalized_pair_diff(s0, r1) == {
        'model.bbox_head.odq_cfg.alpha',
        'model.bbox_head.odq_cfg.loss_weight',
        'work_dir',
    }
```

Also assert:

- `num_queries == 600`;
- grouped matching remains exactly three groups of Q600;
- DN configuration is inherited unchanged;
- the rotated Hungarian assigner still includes the existing classification, rotated L1, and rotated IoU costs;
- the existing GDLoss is inherited unchanged;
- the data pipeline still converts quadrilateral boxes to rotated boxes;
- test configuration has no NMS, top-k, max-per-image truncation, or candidate filtering;
- no encoder-proposal selection is enabled;
- the parent config path and parent Git SHA are recorded as immutable metadata.

- [ ] **Step 2: Run the config test and confirm the files are absent**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_configs.py -q
```

Expected: file-not-found failure.

- [ ] **Step 3: Create thin child configs**

Each config must inherit the authoritative T7 full recipe and override only `odq_cfg`, immutable experiment metadata, and a unique work directory. The S0 core is:

```python
_base_ = './ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py'

model = dict(
    bbox_head=dict(
        odq_cfg=dict(
            num_bins=17,
            radii=(2.0, 2.0, 2.0, 2.0),
            num_fcs=2,
            alpha=0.0,
            loss_weight=0.0)))

experiment_contract = dict(
    method='ODQ-S0',
    physical_gpu_ids=(8, 9),
    query_mouth='rotated_e2e_q600_all_rows',
    parent_config='configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py',
    parent_git_sha='f5b1b8d15debcc6bf23f825929d3a05949f9dbe6')

work_dir = 'work_dirs/odq/s0_full24e_gpu89'
```

For R1, change only the two ODQ scalar switches, method name, and work directory.

- [ ] **Step 4: Run config parsing and the paired diff test**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_configs.py -q
```

Expected: all tests pass and both configs parse without importing a user-site package.

- [ ] **Step 5: Commit recipes**

```bash
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_s0.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_r1.py tests/test_projects/ov_capflow/test_odq_configs.py
rtk git diff --cached --check
rtk git commit -m "config: freeze ODQ S0 and R1 recipes"
```

## Task 5: Implement the five-check ODQ stage-zero audit

**Files:**

- Create: `projects/OVCapFlow/tools/audit_odq_stage0.py`
- Create: `tests/test_projects/ov_capflow/test_odq_stage0.py`
- Reuse: `projects/OVCapFlow/tools/audit_d13n_stage0.py`
- Reuse: `projects/OVCapFlow/tools/audit_d12_preflight.py`
- Reuse: `projects/OVCapFlow/tools/validate_dotav2_q600_dump.py`
- Reuse: `projects/OVCapFlow/ov_capflow/no_replace.py`

- [ ] **Step 1: Write failing tests for the report schema and hard gates**

The JSON report must contain these exact top-level fields:

```python
REQUIRED_FIELDS = {
    'schema_version', 'created_at', 'git_sha', 'working_tree',
    'environment', 'config_fingerprints', 'checkpoint_compatibility',
    'shape_and_mouth', 'zero_step_parent_equality',
    'finite_and_gradient', 'optimizer_and_provenance',
    'rotated_e2e_anti_shortcut', 'verdict', 'failures',
}
```

Test that every failed hard check forces `verdict == 'FAIL'`, a missing check is a failure, and an existing report path is never overwritten.

- [ ] **Step 2: Run the tests and confirm the audit module is absent**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_stage0.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement pure report evaluation and immutable output**

Reuse `no_replace.py` for exclusive creation. Define pure check evaluators so unit tests require no GPU. Record SHA256 hashes of configs, checkpoint, source files, and the complete argv/environment used for the real audit. Never record secrets or the full ambient environment.

- [ ] **Step 4: Implement check 1 — shape and rotated E2E mouth**

On one real validation batch, assert:

- decoder hidden states retain their expected layer/batch/query dimensions;
- ODQ logits are `[layers, batch, training_queries, 4, 17]`;
- parent and combined boxes are five-dimensional rotated boxes;
- the inference output has exactly 600 rows before serialization;
- serialized prediction counts equal the model's all-row counts;
- no row selector is applied between the head and evaluator.

- [ ] **Step 5: Implement check 2 — exact zero-step parent equality**

Load the same canonical parent checkpoint into parent, S0, and R1 models. Permit only `odq_branches.*` missing keys. With deterministic evaluation mode and identical input, require:

```python
torch.equal(parent_cls_scores, s0_cls_scores)
torch.equal(parent_boxes, s0_boxes)
torch.equal(parent_cls_scores, r1_cls_scores)
torch.equal(parent_boxes, r1_boxes)
torch.equal(parent_boxes[..., 4], r1_boxes[..., 4])
```

Any mismatch is a hard failure; do not downgrade to tolerance-based equality.

- [ ] **Step 6: Implement check 3 — finite values and ODQ gradients**

Run one real training batch on physical GPU 8 only, with optimizer stepping disabled. Require finite input, logits, parent boxes, combined boxes, all parent losses, all ODQ losses, and all gradients. Require non-zero finite gradient norm on at least one `odq_branches.*` parameter in R1. Require the parent-only boxes used for target construction to remain detached from the ODQ loss.

- [ ] **Step 7: Implement check 4 — optimizer and provenance**

Build the real optimizer wrapper and assert:

- every trainable ODQ parameter occurs exactly once;
- no frozen parameter is included;
- every parent parameter membership and hyperparameter matches T7;
- the only checkpoint incompatibility is missing ODQ keys;
- config hashes, checkpoint hash, Git SHA, interpreter path, package versions, CUDA visibility, physical GPU identity, and dirty-tree diff hash are recorded.

- [ ] **Step 8: Implement check 5 — rotated E2E anti-shortcut**

Combine effective-config assertions with a static/runtime source audit. Fail if reachable inference code contains or executes query-row truncation, `topk`, NMS, max-per-image selection, encoder-proposal selection, horizontal-box conversion, or an angle overwrite. Confirm the DOTA evaluator receives five-coordinate rotated boxes and all Q600 rows.

- [ ] **Step 9: Add CLI and isolated test hooks**

Provide:

```text
--parent-config
--s0-config
--r1-config
--checkpoint
--output
--device
--seed
```

The default device is `cuda:0`, which maps to physical GPU 8 only when the required environment command is used. Do not embed an absolute checkpoint guess in the tool.

- [ ] **Step 10: Run unit tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_odq_stage0.py tests/test_projects/ov_capflow/test_odq_configs.py tests/test_projects/ov_capflow/test_odq_head.py tests/test_projects/ov_capflow/test_oriented_distribution.py -q
```

Expected: all tests pass.

- [ ] **Step 11: Run the real five-check audit on physical GPU 8**

Use the canonical T7 Epoch 24 checkpoint frozen in the approved design and the
new immutable report path `work_dirs/odq/stage0/20260802_odq_r1_first/report.json`.
If that report path exists at execution time, choose a new explicit suffixed
directory before running; the audit must never overwrite it.

```bash
rtk env CUDA_VISIBLE_DEVICES=8 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_odq_stage0.py --parent-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py --s0-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_s0.py --r1-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2_odq_r1.py --checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --output work_dirs/odq/stage0/20260802_odq_r1_first/report.json --device cuda:0 --seed 20260802
```

- [ ] **Step 12: Inspect the report and refuse training on any failure**

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m json.tool work_dirs/odq/stage0/20260802_odq_r1_first/report.json
```

The report must show `verdict: PASS`, no failures, exact parent equality, non-zero ODQ gradient, only allowed missing checkpoint keys, physical GPU 8 identity, and all five hard checks passing.

- [ ] **Step 13: Commit the audit tooling and tests**

```bash
rtk git add projects/OVCapFlow/tools/audit_odq_stage0.py tests/test_projects/ov_capflow/test_odq_stage0.py
rtk git diff --cached --check
rtk git commit -m "audit: add ODQ five-check preflight"
```

Do not commit generated reports or work-directory outputs.

## Task 6: Run the complete implementation verification without training

**Files:**

- Verify all files changed by Tasks 1–5

- [ ] **Step 1: Run syntax compilation**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m py_compile projects/OVCapFlow/ov_capflow/oriented_distribution.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/tools/audit_odq_stage0.py
```

- [ ] **Step 2: Run the focused ODQ suite**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_oriented_distribution.py tests/test_projects/ov_capflow/test_odq_head.py tests/test_projects/ov_capflow/test_odq_configs.py tests/test_projects/ov_capflow/test_odq_stage0.py -q
```

- [ ] **Step 3: Run the existing OV-CapFlow regression tests**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow -q
```

If an unrelated pre-existing test fails, record the exact command, traceback, and proof that the failure exists at the pre-implementation commit. Do not mask or delete the test.

- [ ] **Step 4: Scan the diff for forbidden shortcuts and incomplete markers**

```bash
rtk rg -n "TODO|TBD|FIXME|NotImplementedError|nan_to_num|topk|nms|max_per_img" projects/OVCapFlow/ov_capflow/oriented_distribution.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/tools/audit_odq_stage0.py configs/ov_capflow/dotav2/*odq*.py tests/test_projects/ov_capflow/test_odq_*.py tests/test_projects/ov_capflow/test_oriented_distribution.py
```

Every match must be either an intentional anti-shortcut assertion or removed before completion.

- [ ] **Step 5: Review the final diff and history**

```bash
rtk git status --short
rtk git diff --check
rtk git log --oneline -6
```

Expected: only the intended commits and any clearly identified pre-existing user files; no training processes and no generated result artifacts are introduced by this plan.

## Completion boundary

This plan is complete only when the focused and regression tests pass, the real five-check stage-zero audit says `PASS`, checkpoint compatibility is proven, and no training has started. The next authorized unit is the separate overnight supervisor implementation plan; its own dry-run audit must pass before launching S0/R1 proxy training on physical GPUs 8 and 9.
