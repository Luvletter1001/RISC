# RISC-ER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and causally evaluate a single-view RISC-ER semantic quotient that minimizes rotation-induced object risk above an identity-repeat control, without changing OV-CapFlow geometry or output mouth.

**Architecture:** Preserve the historical POQ implementation for reproducibility and add a separate `RISCOrbitCenteredQuotient` plus one excess-risk loss. Three sequential training views have fixed roles—identity A, identity B, rotated—and align owners by GT identity rather than query index. A matched control consumes the same views, RNG, rotated detection loss, optimizer schedule, and evaluation protocol while disabling the quotient and ER gradient.

**Tech Stack:** Python 3.8, PyTorch 1.12.1+cu113, MMCV 2.1.0, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate/OV-CapFlow, pytest, DOTA2 rotated AP50, 8×A40 DDP.

---

## 0. Authority, base, and file map

### Frozen authority

- Approved design: `/data1/zcy/OV-CapFlow/docs/superpowers/specs/2026-08-11-risc-er-formal-design.md`.
- Implementation base: commit `4f3fd42` on `research/risc-soft-lowrank-map62-gpu89`.
- Clean worktree: `/data1/zcy/OV-CapFlow/.worktrees/risc-er`.
- New branch: `research/risc-er`.
- Current E12 checkpoint: `/data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/epoch_12.pth`.
- Current E12 resolved config: `/data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/20260810_062830/vis_data/config.py`.
- Current E12 exact filtered metric: `0.6340868473052979/0.6340`.
- Historical source worktree is dirty. It is read-only evidence and must not be cleaned, committed, or used as an implementation working tree.

### Files to create

- `projects/OVCapFlow/ov_capflow/risc_excess_risk.py` — centered quotient, ER state, contender assignment, risk and loss.
- `projects/OVCapFlow/ov_capflow/risc_er_hook.py` — optimizer-bound update counter and runtime audit.
- `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_base.py` — shared recipe.
- `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_control.py` — matched B0.
- `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_candidate.py` — B1.
- `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er_smoke.py` — two-iteration DDP smoke.
- `tests/test_projects/ov_capflow/test_risc_er_quotient.py`.
- `tests/test_projects/ov_capflow/test_risc_er_orbit.py`.
- `tests/test_projects/ov_capflow/test_risc_er_loss.py`.
- `tests/test_projects/ov_capflow/test_risc_er_detector.py`.
- `tests/test_projects/ov_capflow/test_risc_er_head.py`.
- `tests/test_projects/ov_capflow/test_risc_er_configs.py`.
- `docs/project_history/exp_20260811_risc_er/flog_risc_er_zh.md` — factual run ledger.

### Files to modify

- `projects/OVCapFlow/ov_capflow/paired_orbit.py` — add the identity-A/identity-B/rotated builder without changing old builders.
- `projects/OVCapFlow/ov_capflow/ov_capflow_head.py` — classification-only ER quotient and final primary-Q600 state capture.
- `projects/OVCapFlow/ov_capflow/ov_capflow.py` — strict config, matched control, three-view scheduling, single ER loss.
- `projects/OVCapFlow/ov_capflow/__init__.py` — register exports and hook.
- `projects/OVCapFlow/tools/audit_checkpoint_load.py` — recognize only the new zero-initialized quotient keys.

The old `risc_paired_quotient.py`, old configs, checkpoints, and tests remain behaviorally unchanged.

## Task 1: Create the clean implementation worktree and seal the source

**Files:**
- Read: approved spec and E12 archived config/checkpoint.
- Create: `/data1/zcy/OV-CapFlow/.worktrees/risc-er/` through Git.

- [ ] **Step 1: Verify exact base, branch absence, and target absence**

```bash
rtk git -C /data1/zcy/OV-CapFlow show --no-patch --oneline 4f3fd42
rtk git -C /data1/zcy/OV-CapFlow show-ref --verify refs/heads/research/risc-er
rtk git -C /data1/zcy/OV-CapFlow worktree list --porcelain
```

Expected: `4f3fd42` resolves; the branch verification exits nonzero; no worktree path equals `/data1/zcy/OV-CapFlow/.worktrees/risc-er`. If either target exists, inspect and stop instead of overwriting it.

- [ ] **Step 2: Create the worktree from the committed POQ source**

```bash
rtk git -C /data1/zcy/OV-CapFlow worktree add -b research/risc-er /data1/zcy/OV-CapFlow/.worktrees/risc-er 4f3fd42
rtk git -C /data1/zcy/OV-CapFlow/.worktrees/risc-er rev-parse HEAD
rtk git -C /data1/zcy/OV-CapFlow/.worktrees/risc-er status --short
```

Expected: exact hash `4f3fd429187d690b10b3bfb8e52d3d6774cf8529` and empty status.

- [ ] **Step 3: Bind the approved design and E12 evidence**

```bash
rtk sha256sum /data1/zcy/OV-CapFlow/docs/superpowers/specs/2026-08-11-risc-er-formal-design.md
rtk sha256sum /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/epoch_12.pth
rtk sha256sum /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/20260810_062830/vis_data/config.py
```

Record the three exact hashes in `docs/project_history/exp_20260811_risc_er/flog_risc_er_zh.md` with labels `[FACT]` and `status=SOURCE_SEALED`.

- [ ] **Step 4: Verify the runtime**

```bash
rtk env PYTHONPATH=/data1/zcy/OV-CapFlow/.worktrees/risc-er /data/zcy/anaconda3/envs/mmdet/bin/python -c "import torch, mmcv, mmengine, mmdet; print(torch.__version__, mmcv.__version__, mmengine.__version__, mmdet.__version__)"
```

Expected: `1.12.1+cu113 2.1.0 0.10.4 3.3.0`.

## Task 2: Implement the orbit-centered quotient

**Files:**
- Create: `projects/OVCapFlow/ov_capflow/risc_excess_risk.py`.
- Test: `tests/test_projects/ov_capflow/test_risc_er_quotient.py`.

- [ ] **Step 1: Write failing quotient tests**

The tests instantiate `RISCOrbitCenteredQuotient(embed_dims=4, rank=2, harmonic_orders=(2, 4), quadrature_points=12, max_gate=.2, max_delta_norm_ratio=.05, basis_seed=7)` and assert:

```python
def test_zero_init_is_exact_parent_identity():
    module = _module()
    query = torch.randn(2, 5, 4, requires_grad=True)
    theta = torch.rand(2, 5) * math.pi
    stable, debug = module(query, theta)
    torch.testing.assert_close(stable, query, rtol=0, atol=0)
    assert debug['orbit_gate_mean_abs'].item() == 0.0


def test_gate_has_zero_discrete_orbit_mean():
    module = _module(max_delta_norm_ratio=1.0)
    with torch.no_grad():
        module.query_to_rank.weight.normal_()
        module.angle_to_rank.weight.normal_()
        module.rank_bias.normal_()
    query = torch.randn(2, 3, 4)
    gate = module.orbit_centered_gates(query)
    torch.testing.assert_close(
        gate.mean(dim=-2), torch.zeros_like(gate.mean(dim=-2)),
        rtol=0, atol=2e-7)


def test_er_residual_is_pi_periodic_bounded_and_classification_only():
    module = _module(max_delta_norm_ratio=.01)
    query = torch.randn(2, 5, 4)
    theta = torch.rand(2, 5)
    first, first_debug = module(query, theta)
    second, _ = module(query, theta + math.pi)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    assert first_debug['delta_norm_ratio'].max() <= .010001
    assert first_debug['gate_abs_max'] <= .200001
```

Use the same theta tensor for the periodic equality assertion. Also test invalid rank, empty harmonics, nonpositive quadrature, invalid shapes/dtypes, and nonfinite input.

- [ ] **Step 2: Run the focused tests and observe import failure**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_quotient.py -q
```

Expected: FAIL because `risc_excess_risk.py` does not exist.

- [ ] **Step 3: Implement exact group centering**

The centered gate implementation must follow this shape contract:

```python
class RISCOrbitCenteredQuotient(nn.Module):
    def quadrature_angles(self, query: Tensor) -> Tensor:
        return torch.arange(
            self.quadrature_points, device=query.device,
            dtype=query.dtype) * (math.pi / self.quadrature_points)

    def _raw_gate(self, normalized: Tensor, theta: Tensor) -> Tensor:
        return (self.query_to_rank(normalized)
                + self.angle_to_rank(self.angle_features(theta))
                + self.rank_bias)

    def _orbit_squashed_gates(self, normalized: Tensor) -> Tensor:
        grid = self.quadrature_angles(normalized)
        query_term = self.query_to_rank(normalized).unsqueeze(-2)
        angle_term = self.angle_to_rank(
            self.angle_features(grid)).view(
                *((1,) * (normalized.ndim - 1)),
                self.quadrature_points, self.rank)
        return torch.tanh(query_term + angle_term + self.rank_bias)

    def centered_gate(self, query: Tensor, theta: Tensor) -> Tensor:
        normalized = F.layer_norm(query.float(), (self.embed_dims,))
        actual = torch.tanh(self._raw_gate(normalized, theta.float()))
        orbit = self._orbit_squashed_gates(normalized)
        return .5 * self.max_gate * (
            actual - orbit.mean(dim=-2))

    def orbit_centered_gates(self, query: Tensor) -> Tensor:
        normalized = F.layer_norm(query.float(), (self.embed_dims,))
        orbit = self._orbit_squashed_gates(normalized)
        return .5 * self.max_gate * (
            orbit - orbit.mean(dim=-2, keepdim=True))
```

`quadrature_angles` returns `[12]`; `orbit_centered_gates` broadcasts it and returns `[B,Q,12,rank]` without expanding query embeddings over the grid. Keep QR basis construction and norm clipping equivalent to legacy POQ. Initialize `query_to_rank`, `angle_to_rank`, and `rank_bias` to exact zero.

- [ ] **Step 4: Run quotient and legacy regression tests**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_quotient.py tests/test_projects/ov_capflow/test_risc_poq_module.py -q
rtk git diff --check
```

Expected: all pass; legacy POQ file has no diff.

- [ ] **Step 5: Commit the centered quotient**

```bash
rtk git add projects/OVCapFlow/ov_capflow/risc_excess_risk.py tests/test_projects/ov_capflow/test_risc_er_quotient.py
rtk git commit -m "feat: add orbit-centered RISC quotient"
```

## Task 3: Build identity-repeat-controlled orbit views

**Files:**
- Modify: `projects/OVCapFlow/ov_capflow/paired_orbit.py`.
- Test: `tests/test_projects/ov_capflow/test_risc_er_orbit.py`.

- [ ] **Step 1: Write failing view-role tests**

Test the new API:

```python
orbit = build_risc_er_orbit_batch(
    batch_inputs, samples, completed_updates=4,
    angles_deg=torch.tensor([90]))
assert orbit.roles == ('identity_a', 'identity_b', 'rotated')
torch.testing.assert_close(orbit.inputs[0], batch_inputs)
torch.testing.assert_close(orbit.inputs[1], batch_inputs)
assert orbit.angles_deg.tolist() == [[0, 0, 90]]
assert orbit.active_index == 2
assert orbit.samples[0] is not orbit.samples[1]
assert orbit.samples[0][0] is not orbit.samples[1][0]
assert orbit.samples[0][0].gt_instances.orbit_instance_ids.tolist() == [0]
assert orbit.samples[1][0].gt_instances.orbit_instance_ids.tolist() == [0]
```

Also assert that a rotated object leaving the valid canvas is removed only from the rotated view, and that the original sample is not mutated.

- [ ] **Step 2: Run the focused test**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_orbit.py -q
```

Expected: FAIL because the new builder is undefined.

- [ ] **Step 3: Add a separate builder without changing historical builders**

Add:

```python
@dataclass(frozen=True)
class RISCEROrbitBatch:
    inputs: Tuple[Tensor, Tensor, Tensor]
    samples: Tuple[List[DetDataSample], List[DetDataSample],
                   List[DetDataSample]]
    angles_deg: Tensor
    valid_masks: Tensor
    visibility_counts: Mapping[str, int]
    roles: Tuple[str, str, str] = (
        'identity_a', 'identity_b', 'rotated')
    active_index: int = 2


def build_risc_er_orbit_batch(
        batch_inputs, batch_data_samples, completed_updates,
        angles_deg=None) -> RISCEROrbitBatch:
    identity_a = _identity_orbit_view(batch_inputs, batch_data_samples)
    identity_b = _identity_orbit_view(batch_inputs, batch_data_samples)
    angles = deterministic_single_orbit_angle(
        completed_updates, tuple(s.img_id for s in batch_data_samples),
        angles_deg)
    rotated = _rotated_orbit_view(
        batch_inputs, batch_data_samples, angles, view_slot=2)
    return RISCEROrbitBatch(
        inputs=(identity_a[0], identity_b[0], rotated[0]),
        samples=(identity_a[1], identity_b[1], rotated[1]),
        angles_deg=torch.stack((
            torch.zeros_like(angles), torch.zeros_like(angles), angles), 1),
        valid_masks=torch.stack((identity_a[2], identity_b[2], rotated[2])),
        visibility_counts={
            key: sum(view[3][key] for view in
                     (identity_a, identity_b, rotated))
            for key in ('before', 'after_corners', 'after_center_mask')})
```

Implement the selector without process-local RNG:

```python
def deterministic_single_orbit_angle(
        completed_updates: int, img_ids: Sequence[object],
        supplied: Optional[Tensor] = None) -> Tensor:
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError('completed_updates must be a nonnegative int')
    if not img_ids:
        raise ValueError('img_ids must be nonempty')
    if supplied is not None:
        angles = torch.as_tensor(supplied, dtype=torch.long, device='cpu')
        if angles.shape != (len(img_ids),):
            raise ValueError('angles_deg must have shape [B]')
        allowed = torch.tensor(ORBIT_ANGLES, dtype=torch.long)
        if not (angles[..., None] == allowed).any(-1).all():
            raise ValueError('angles_deg values must be in ORBIT_ANGLES')
        return angles
    selected = []
    for img_id in img_ids:
        selected.append(min(
            ORBIT_ANGLES,
            key=lambda angle: hashlib.sha256(
                f'risc-er-angle-v1\0{completed_updates}\0{img_id}\0{angle}'
                .encode('utf-8')).digest()))
    return torch.tensor(selected, dtype=torch.long)
```

- [ ] **Step 4: Run new and old orbit tests**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_orbit.py tests/test_projects/ov_capflow/test_risc_obf_three_view.py -q
rtk git diff --check
```

Expected: all pass; old `build_three_view_orbit_batch` behavior is unchanged.

- [ ] **Step 5: Commit view construction**

```bash
rtk git add projects/OVCapFlow/ov_capflow/paired_orbit.py tests/test_projects/ov_capflow/test_risc_er_orbit.py
rtk git commit -m "feat: add identity-controlled RISC orbit views"
```

## Task 4: Capture geometry-owned primary-query state

**Files:**
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`.
- Modify: `projects/OVCapFlow/ov_capflow/risc_excess_risk.py`.
- Test: `tests/test_projects/ov_capflow/test_risc_er_head.py`.

- [ ] **Step 1: Write failing head/state tests**

Define and test this state:

```python
@dataclass(frozen=True)
class RISCERReadoutState:
    raw_query: Tensor              # [B,600,D]
    theta: Tensor                  # [B,600], radians
    pred_boxes: Tensor             # [B,600,5], normalized xywha
    raw_text: Tensor               # [B,T,D]
    text_token_mask: Tensor        # [B,T]
    owner_gt_ids: Tensor           # [B,600], -1 for unmatched
    gt_boxes_by_id: Tuple[Dict[int, Tensor], ...]
    labels_by_gt_id: Tuple[Dict[int, int], ...]
    class_to_token: Tuple[Dict[int, Tuple[int, ...]], ...]
    role: str
```

The head test must verify:

- primary group only: shape `[B,600,*]`, not all three training groups;
- `theta == pred_boxes[...,4] * angle_factor`;
- exact owner identities are recovered from final Hungarian targets;
- GT boxes and labels use the same `orbit_instance_ids` keys;
- quotient changes class logits but bbox tensors remain bitwise parent-equivalent at zero init and unchanged after nonzero quotient parameters;
- every geometry tensor stored for ER is detached.

- [ ] **Step 2: Run the test and observe failure**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_head.py -q
```

Expected: FAIL because the head does not accept `risc_er_quotient_cfg` or build `RISCERReadoutState`.

- [ ] **Step 3: Integrate a mutually exclusive classification-only path**

Add `risc_er_quotient_cfg=None` to `OVCapFlowHead.__init__` with exact keys:

```python
{
    'enabled', 'rank', 'harmonic_orders', 'quadrature_points',
    'max_gate', 'max_delta_norm_ratio', 'basis_seed'
}
```

Reject simultaneous enablement with legacy RISC, OBF, soft projection, POQ, or existence residual. Instantiate `RISCOrbitCenteredQuotient` only when enabled.

In `forward`, retain the existing ordering:

```python
tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
outputs_coord = (tmp_reg_preds + reference).sigmoid()
classification_hidden = hidden_state
if self.risc_er_quotient is not None:
    theta = outputs_coord[..., 4] * self.angle_factor
    classification_hidden, er_debug = self.risc_er_quotient(
        hidden_state, theta)
outputs_class = self.cls_branches[layer_id](
    classification_hidden, memory_text, text_token_mask)
```

Store raw query, detached theta, detached boxes, text and mask only for the final layer and first primary Q600 group. Build the state after final Hungarian target capture.

- [ ] **Step 4: Run head, POQ and grouped-query regressions**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_head.py tests/test_projects/ov_capflow/test_risc_poq_head.py tests/test_projects/ov_capflow/test_grouped_queries.py -q
rtk git diff --check
```

Expected: all pass; regression outputs are unchanged when ER is disabled.

- [ ] **Step 5: Commit state capture**

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/ov_capflow/risc_excess_risk.py tests/test_projects/ov_capflow/test_risc_er_head.py
rtk git commit -m "feat: capture RISC object risk state"
```

## Task 5: Implement contender assignment and the single ER loss

**Files:**
- Modify: `projects/OVCapFlow/ov_capflow/risc_excess_risk.py`.
- Test: `tests/test_projects/ov_capflow/test_risc_er_loss.py`.

- [ ] **Step 1: Write failing risk tests**

Use synthetic states with three classes, one shared GT and four queries. Test:

```python
def test_rotation_wrong_class_hub_exceeds_identity_repeat():
    loss, telemetry = compute_risc_er_loss(
        states=(identity_a, identity_b, rotated_hub),
        quotient=quotient, contender_iou=.5,
        max_text_len=8, log_scale='auto', bias=None)
    assert loss > 0
    assert telemetry['shared_object_count'] == 1
    assert telemetry['rotation_excess_mean'] > 0


def test_equal_rotation_and_identity_drift_has_zero_excess():
    loss, _ = compute_risc_er_loss(
        states=(identity_a, same_drift, same_drift),
        quotient=quotient, contender_iou=.5,
        max_text_len=8, log_scale='auto', bias=None)
    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_high_iou_nonowner_adds_contender_energy_once():
    ids = assign_contender_gt_ids(
        pred_boxes, gt_ids, gt_boxes, owner_ids,
        iou_threshold=.5, angle_factor=math.pi)
    assert ids.tolist() == [-1, 7, -1, -1]


def test_er_gradient_reaches_only_quotient_parameters():
    states = (identity_a, identity_b, rotated_hub)
    loss, _ = compute_risc_er_loss(
        states=states,
        quotient=quotient, contender_iou=.5,
        max_text_len=8, log_scale='auto', bias=None)
    loss.backward()
    assert quotient.query_to_rank.weight.grad.abs().sum() > 0
    assert all(state.raw_query.grad is None for state in states)
    assert all(state.raw_text.grad is None for state in states)
```

Also test empty shared objects, zero GT, zero contenders, repeated owner identity, label mismatch, nonfinite scores, and contender threshold fixed to exactly float `.5`.

- [ ] **Step 2: Run tests and observe missing implementation**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_loss.py -q
```

Expected: FAIL on missing contender/risk functions.

- [ ] **Step 3: Implement one-to-one contender ownership**

Use detached normalized boxes; multiply angle by `angle_factor` before rotated IoU. Each nonowner query is assigned to only its maximum-IoU GT:

```python
def _boxes_in_radians(boxes: Tensor, angle_factor: float) -> Tensor:
    decoded = boxes.detach().float().clone()
    decoded[..., 4] = decoded[..., 4] * float(angle_factor)
    return decoded


def assign_contender_gt_ids(pred_boxes, gt_ids, gt_boxes,
                            owner_gt_ids, iou_threshold=.5,
                            angle_factor=math.pi):
    overlaps = box_iou_rotated(
        _boxes_in_radians(pred_boxes, angle_factor),
        _boxes_in_radians(gt_boxes, angle_factor))
    best_iou, best_index = overlaps.max(dim=1)
    result = gt_ids.index_select(0, best_index)
    keep = (best_iou >= iou_threshold) & (owner_gt_ids < 0)
    return torch.where(keep, result, result.new_full((), -1))
```

Chunk the prediction axis in blocks of 600 when an image has more than 600 GT so temporary overlap storage remains bounded. Do not use semantic scores to assign contenders.

- [ ] **Step 4: Implement object risk and excess-over-control**

For each shared object:

```python
wrong = owner_scores[negative_mask] - owner_scores[label]
contender = (
    contender_scores.max(dim=-1).values - owner_scores[label])
energies = torch.cat((wrong, contender), dim=0)
risk = torch.logsumexp(energies.float(), dim=0)

rotation_increase = F.relu(risk_theta - risk_identity_a.detach())
repeat_increase = F.relu(
    risk_identity_b.detach() - risk_identity_a.detach())
term = F.relu(rotation_increase - repeat_increase)
```

Recompute the rotated stable query by calling the quotient on `raw_query.detach()` and `theta.detach()` so ER gradients cannot enter the detector. Compute both identity risks inside `torch.no_grad()`. Use `global_positive_mean` for DDP-correct object averaging. Return exactly one differentiable loss plus detached telemetry; do not return feature/margin/background losses.

- [ ] **Step 5: Run loss and legacy regression tests**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_loss.py tests/test_projects/ov_capflow/test_risc_poq_loss.py -q
rtk git diff --check
```

Expected: all pass and no legacy loss behavior changes.

- [ ] **Step 6: Commit the objective**

```bash
rtk git add projects/OVCapFlow/ov_capflow/risc_excess_risk.py tests/test_projects/ov_capflow/test_risc_er_loss.py
rtk git commit -m "feat: add rotation excess object risk"
```

## Task 6: Integrate matched control and optimizer-bound scheduling

**Files:**
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow.py`.
- Create: `projects/OVCapFlow/ov_capflow/risc_er_hook.py`.
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`.
- Test: `tests/test_projects/ov_capflow/test_risc_er_detector.py`.

- [ ] **Step 1: Write failing detector tests**

The strict config is:

```python
RISC_ER_CANDIDATE = dict(
    enabled=True, matched_control=False, orbit_interval=4,
    contender_iou=.5, freeze_language_backbone=True)
RISC_ER_CONTROL = dict(
    enabled=False, matched_control=True, orbit_interval=4,
    contender_iou=.5, freeze_language_backbone=True)
```

Tests must prove:

- exactly one of `enabled` and `matched_control` is true;
- interval is exactly integer 4 and IoU exactly float `.5`;
- old POQ/orbit/OBF modes are mutually exclusive;
- updates `0,4,8` are orbit updates;
- orbit call order is identity A `no_grad`, identity B `no_grad`, rotated `grad`;
- active index is always 2;
- control executes the same three forwards but returns only rotated parent losses;
- candidate returns parent losses plus exactly `loss_risc_er` and detached telemetry;
- auxiliary loss is multiplied by 4 on interval-4 updates;
- update counter advances only when `optim_wrapper.should_update()` is true and survives checkpoint/resume.

- [ ] **Step 2: Run the detector test**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_detector.py -q
```

Expected: FAIL on missing config and hook.

- [ ] **Step 3: Add the strict detector branch**

Add `risc_er_training_cfg=None` to `OVCapFlow.__init__`. Register `risc_er_optimizer_updates` only for enabled or matched-control runs. Route losses as follows:

```python
def loss(self, batch_inputs, batch_data_samples):
    if self.risc_er_training_cfg is not None:
        update = int(self.risc_er_optimizer_updates.item())
        if update % 4 == 0:
            return self._risc_er_or_matched_control_loss(
                batch_inputs, batch_data_samples)
        return self._single_view_loss(batch_inputs, batch_data_samples)
    if getattr(self, 'risc_poq_training_enabled', False):
        if self.risc_poq_training_mode == 'joint_sparse':
            update = int(self.risc_poq_optimizer_updates.item())
            if not self._risc_poq_is_orbit_update(update):
                return self._single_view_loss(
                    batch_inputs, batch_data_samples)
            return self._risc_poq_loss(
                batch_inputs, batch_data_samples,
                auxiliary_scale=float(
                    self.risc_poq_training_cfg['orbit_interval']))
        return self._risc_poq_loss(
            batch_inputs, batch_data_samples, auxiliary_scale=1.0)
    return self._legacy_or_obf_loss(batch_inputs, batch_data_samples)
```

In the orbit method, always execute both identity forwards under `no_grad`, then the rotated parent loss with gradients. Candidate calls:

```python
branch = self.bbox_head.cls_branches[-1]
er_loss, telemetry = compute_risc_er_loss(
    states=tuple(states),
    quotient=self.bbox_head.risc_er_quotient,
    contender_iou=self.risc_er_training_cfg['contender_iou'],
    max_text_len=branch.max_text_len,
    log_scale=branch.log_scale,
    bias=branch.bias)
losses['loss_risc_er'] = 4.0 * er_loss
```

Matched control does not require ER states; it discards the two no-grad identity outputs after validating view roles and finite parent loss.

- [ ] **Step 4: Implement runtime audit hook**

`RISCERTrainingHook` must:

- increment the persistent update counter only at optimizer boundaries;
- require finite loss and telemetry;
- assert centered gate bound `<=max_gate`, delta ratio bound, role tuple, active index 2 and nonnegative shared count;
- skip quotient-bound assertions for matched control while still checking roles, update count and finite parent loss;
- store `risc_er` metadata in every checkpoint;
- never stop or restart a run itself.

- [ ] **Step 5: Run detector, hook, resume and POQ regressions**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_detector.py tests/test_projects/ov_capflow/test_risc_poq_detector.py tests/test_projects/ov_capflow/test_risc_poq_joint.py -q
rtk git diff --check
```

Expected: all pass; old POQ checkpoint keys and scheduling remain loadable.

- [ ] **Step 6: Commit integration**

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/ov_capflow/risc_er_hook.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_risc_er_detector.py
rtk git commit -m "feat: integrate matched RISC excess training"
```

## Task 7: Add paired configs and checkpoint-load audit

**Files:**
- Create: the four configs listed in the file map.
- Modify: `projects/OVCapFlow/tools/audit_checkpoint_load.py`.
- Test: `tests/test_projects/ov_capflow/test_risc_er_configs.py`.

- [ ] **Step 1: Write failing config tests**

Load control and candidate with MMEngine and assert exact equality for:

- parent checkpoint and provenance;
- data roots, sampler, Q600, matching groups 3, batch 2, accumulation 2;
- effective global batch 32, LR `1.2e-4`, cosine schedule, 12 epochs;
- all non-RISC model fields, prompts, seed, evaluators and output mouth;
- same orbit interval/control view execution;
- no post-decoder top-k/NMS;
- candidate has centered quotient enabled and ER enabled;
- control has quotient disabled and matched control enabled;
- neither uses legacy POQ/OBF/orbit loss.

- [ ] **Step 2: Run tests and observe missing configs**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_configs.py -q
```

Expected: FAIL because the configs do not exist.

- [ ] **Step 3: Create the common recipe**

The common base extends the committed rare4x recipe and freezes:

```python
_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py']

model = dict(
    backbone=dict(init_cfg=None),
    orbit_training_cfg=None,
    risc_obf_training_cfg=None,
    risc_poq_training_cfg=None,
    freeze_except_patterns=None)

optim_wrapper = dict(
    _delete_=True, type='OptimWrapper', accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1.2e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'bbox_head.risc_er_quotient': dict(decay_mult=0.)}))

custom_hooks = [dict(type='RISCERTrainingHook')]
randomness = dict(seed=20260811, deterministic=False,
                  diff_rank_seed=False)
```

Use the same single official MM-Grounding-DINO-T compatibility parent and exact SHA from E12. Do not claim it is a remote-sensing parent. The two decay exemptions do not change learning rates and therefore do not create a second optimization time scale.

- [ ] **Step 4: Create control and candidate deltas**

Candidate:

```python
model = dict(
    risc_er_training_cfg=dict(
        enabled=True, matched_control=False, orbit_interval=4,
        contender_iou=.5, freeze_language_backbone=True),
    bbox_head=dict(risc_er_quotient_cfg=dict(
        enabled=True, rank=16, harmonic_orders=(2, 4),
        quadrature_points=12, max_gate=.20,
        max_delta_norm_ratio=.05, basis_seed=20260811)))
```

Control uses the same training cfg with `enabled=False, matched_control=True` and the same head cfg with `enabled=False`. The smoke config inherits candidate, switches to a two-iteration `IterBasedTrainLoop`, disables validation, uses batch size 1 and zero dataloader workers, replaces `custom_hooks` with only `RISCERTrainingHook`, and writes only one temporary checkpoint.

Each child defines its own unique `work_dir` and replaces the hook list so the loss ledger resolves against that directory:

```python
work_dir = 'work_dirs/risc_er/weak_20260811/candidate_seed20260811'
custom_hooks = [
    dict(type='RISCERTrainingHook'),
    dict(type='LossUpdateLedgerHook',
         path_template=work_dir + '/loss/loss_epoch_{epoch}.json')]
```

The control uses `work_dirs/risc_er/weak_20260811/control_seed20260811`; the smoke uses `work_dirs/risc_er/smoke_candidate_20260811`.

- [ ] **Step 5: Extend load audit narrowly**

Allow missing keys only matching:

```python
r'^bbox_head\.risc_er_quotient\.(?:raw_basis|rank_bias|query_to_rank\.weight|angle_to_rank\.weight)$'
```

Reject any other missing or unexpected key. Audit step-0 parent equivalence on one fixed batch: control and zero-init candidate boxes/classes must match before any optimizer update.

- [ ] **Step 6: Run config and load tests**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_configs.py tests/test_projects/ov_capflow/test_audit_checkpoint_load.py -q
rtk git diff --check
```

Expected: all pass.

- [ ] **Step 7: Commit configs**

```bash
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_base.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_control.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_candidate.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er_smoke.py projects/OVCapFlow/tools/audit_checkpoint_load.py tests/test_projects/ov_capflow/test_risc_er_configs.py
rtk git commit -m "exp: add matched RISC excess recipes"
```

## Task 8: Complete CPU verification and 8-GPU smoke

**Files:**
- Read all changed files.
- Append factual results to `docs/project_history/exp_20260811_risc_er/flog_risc_er_zh.md`.

- [ ] **Step 1: Run the focused suite**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_er_quotient.py tests/test_projects/ov_capflow/test_risc_er_orbit.py tests/test_projects/ov_capflow/test_risc_er_loss.py tests/test_projects/ov_capflow/test_risc_er_detector.py tests/test_projects/ov_capflow/test_risc_er_head.py tests/test_projects/ov_capflow/test_risc_er_configs.py -q
```

Expected: all pass.

- [ ] **Step 2: Run legacy regression suite**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_risc_poq_module.py tests/test_projects/ov_capflow/test_risc_poq_loss.py tests/test_projects/ov_capflow/test_risc_poq_head.py tests/test_projects/ov_capflow/test_risc_poq_detector.py tests/test_projects/ov_capflow/test_risc_poq_joint.py tests/test_projects/ov_capflow/test_risc_obf_three_view.py tests/test_projects/ov_capflow/test_grouped_queries.py tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
```

Expected: all pass.

- [ ] **Step 3: Check GPUs 0–7 before smoke**

```bash
rtk nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Expected: GPUs 0–7 are available. If another legitimate task occupies any of them, do not launch or interrupt it.

- [ ] **Step 4: Run candidate two-iteration DDP smoke**

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow/.worktrees/risc-er /data/zcy/anaconda3/envs/mmdet/bin/torchrun --nproc_per_node=8 --master_port=29911 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er_smoke.py --launcher pytorch
```

Expected: two iterations finish; `loss_risc_er` is finite; shared-object count is positive on at least one rank; max gate and delta ratio stay within bounds; one optimizer update is recorded; no OOM/NCCL/unused-parameter error.

- [ ] **Step 5: Run matched-control smoke**

Create the control smoke via config override, not a code change:

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow/.worktrees/risc-er /data/zcy/anaconda3/envs/mmdet/bin/torchrun --nproc_per_node=8 --master_port=29912 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er_smoke.py --launcher pytorch --cfg-options model.risc_er_training_cfg.enabled=False model.risc_er_training_cfg.matched_control=True model.bbox_head.risc_er_quotient_cfg.enabled=False work_dir=work_dirs/risc_er/smoke_control_20260811
```

Expected: same view schedule and number of forwards, no `loss_risc_er`, finite parent losses, one optimizer update.

- [ ] **Step 6: Record measured smoke cost**

Record wall time/update, peak allocated/reserved memory per rank, shared objects, contender count, ER loss and bounds as `[FACT]`. Do not infer AP from smoke losses.

## Task 9: Run the current-E12 diagnostic gate before formal training

**Files:**
- Reuse: `projects/OVCapFlow/tools/validate_dotav2_q600_dump.py`.
- Reuse: `projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py`.
- Create artifacts only under: `work_dirs/mmgd_risc_poq/e12_risc_er_preflight_20260811/`.

- [ ] **Step 1: Verify E12 identity and exact prior metric**

```bash
rtk sha256sum /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/epoch_12.pth
rtk rg -n '"dota/mAP": 0.6340868473052979' /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/20260810_062830/vis_data/scalars.json
```

Expected: one exact metric row and the same checkpoint hash sealed in Task 1.

- [ ] **Step 2: Produce an exact raw-13833 Q600 dump**

Use GPU 8 only after verifying it is free:

```bash
rtk env CUDA_VISIBLE_DEVICES=8 PYTHONPATH=/data1/zcy/OV-CapFlow/.worktrees/risc-er /data/zcy/anaconda3/envs/mmdet/bin/python tools/test.py /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/20260810_062830/vis_data/config.py /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/epoch_12.pth --work-dir work_dirs/mmgd_risc_poq/e12_risc_er_preflight_20260811/raw13833 --out work_dirs/mmgd_risc_poq/e12_risc_er_preflight_20260811/raw13833/predictions.pkl --cfg-options test_dataloader.dataset.filter_cfg.filter_empty_gt=False test_dataloader.dataset.test_mode=True
```

Expected: 13,833 records and exactly 600 predictions per record. If the archived dataset wrapper does not honor the override, stop and create a derived evaluation config rather than editing the archived config.

- [ ] **Step 3: Validate and analyze the dump**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/validate_dotav2_q600_dump.py work_dirs/mmgd_risc_poq/e12_risc_er_preflight_20260811/raw13833/predictions.pkl --expected-records 13833 --queries-per-image 600 --num-classes 18
```

Run the analyzer; its metric reader selects the sole DOTA metric row under the evaluation directory:

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py work_dirs/mmgd_risc_poq/e12_risc_er_preflight_20260811/raw13833/predictions.pkl --config /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/20260810_062830/vis_data/config.py --checkpoint /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/epoch_12.pth --official-metrics-json work_dirs/mmgd_risc_poq/e12_risc_er_preflight_20260811/raw13833 --training-metrics-json /data1/zcy/OV-CapFlow/work_dirs/mmgd_risc_poq/single_stage12e_world8_b2a2_lr120e6_20260810/20260810_062830/vis_data/scalars.json --training-step 12 --output-dir work_dirs/mmgd_risc_poq/e12_risc_er_preflight_20260811/raw13833/diagnostics --expected-records 13833 --queries-per-image 600 --num-classes 18
```

Expected artifacts include perfect-ranking AP, GT geometry/semantic/ownership decomposition, AP-support FP decomposition, per-class table, size/density strata, query table, report and manifest.

- [ ] **Step 4: Make a fail-closed substrate decision**

Record exactly one decision:

- `RISC_ER_ONLY_ELIGIBLE`: geometry/ranking substrate changes are not justified before the weak B0/B1 pair;
- `BUILD_BSTAR_GEOMETRY_FIRST`: geometry miss is the largest GT-miss state in at least three of the four main gap classes, tiny/dense strata reproduce the same direction, and perfect-ranking headroom is below 1.5 AP;
- `BUILD_BSTAR_RANKING_FIRST`: the existing fixed-box perfect-ranking oracle recovers at least 1.5 AP, regardless of raw candidate count;
- `INVALID_NO_DECISION`: dump, mouth, config, checkpoint or analyzer authority failed.

The `1.5` threshold is approximately 20% of the observed 7.10-point gap and is a preregistered engineering gate, not a statistical confidence bound. Do not design both geometry and ranking changes in the same B* iteration.

## Task 10: Run the matched weak-substrate causal experiment

**Files:**
- Control config and candidate config from Task 7.
- New run directories under `work_dirs/risc_er/weak_20260811/`.

- [ ] **Step 1: Audit config equivalence and step-0 parent equality**

Run config diff, checkpoint-load audit, one-batch zero-init output equality, Q600 all-row inference audit and strict prompt audit. Publish hashes before training.

Expected: the only candidate/control differences are quotient presence, ER flag and resulting parameter count; parent boxes/classes are identical at step 0.

- [ ] **Step 2: Launch control and candidate with equal resources**

Run the two jobs sequentially on the same physical GPUs 0–7. Do not compare an 8-GPU candidate with a 2-GPU control.

Control command:

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow/.worktrees/risc-er /data/zcy/anaconda3/envs/mmdet/bin/torchrun --nproc_per_node=8 --master_port=29921 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_control.py --launcher pytorch
```

Candidate command, launched only after the control process exits and its endpoint is sealed:

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow/.worktrees/risc-er /data/zcy/anaconda3/envs/mmdet/bin/torchrun --nproc_per_node=8 --master_port=29922 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_mmgd_risc_er12e_candidate.py --launcher pytorch
```

Use the exact 12-epoch configs, save every epoch, and validate at the same epochs. No automatic supervisor may modify LR, restart from another checkpoint, or promote based only on training loss.

- [ ] **Step 3: Apply the Stage-0 gate at the first registered validation endpoint**

Continue only if candidate versus matched control shows:

- finite and positive ER support;
- lower scene-macro flip/margin/hub excess;
- nondecreasing geometry retention;
- no material true-small-vehicle recall collapse;
- no AP degradation greater than 0.5 point.

Otherwise stop the candidate, preserve the endpoint, and classify the failure as objective, support, optimization, or substrate mismatch.

- [ ] **Step 4: Apply the full endpoint gate**

At the frozen endpoint, promote to three seeds only if exact same-mouth AP improves by at least 0.3 point and the mechanism metrics improve in the preregistered direction. A gain below 0.3 with no mechanism support closes the recipe; AP70 is not used as an early stopping rule.

## Task 11: Build the strong-substrate 2×2 evidence only after the gate

**Files:**
- Create one separate B* implementation plan chosen by Task 9's single decision.
- Reuse the same RISC-ER implementation unchanged.

- [ ] **Step 1: Select exactly one B* route**

`BUILD_BSTAR_GEOMETRY_FIRST` permits one standard high-resolution/remote-domain geometry change. `BUILD_BSTAR_RANKING_FIRST` permits one established ranking change. `RISC_ER_ONLY_ELIGIBLE` keeps the weak substrate. `INVALID_NO_DECISION` permits only authority repair.

- [ ] **Step 2: Freeze B0-strong before enabling RISC**

Train and validate the strong substrate without RISC first. Audit remote-sensing parent provenance for DOTA leakage and report canonical/text7 prompts separately.

- [ ] **Step 3: Run B1-strong with no RISC recipe changes**

Candidate and control must share every B* component. Do not retune rank, gate, LR, orbit interval or loss after seeing B0-strong results.

- [ ] **Step 4: Complete ICLR evidence**

Run three seeds, strict held-out-class folds, lossless-90/arbitrary-angle controls, raw consistency, quotient-without-ER, contender removal, orbit-centering removal, efficiency, P0148 qualitative analysis and labeled DOTA2 case studies. Use P0148 quantitatively only if an independent authoritative annotation protocol is completed.

## Task 12: Final verification and documentation

**Files:**
- Update: `docs/project_history/exp_20260811_risc_er/flog_risc_er_zh.md`.
- Create after real results: `docs/project_history/exp_20260811_risc_er/fres_risc_er_zh.md`.

- [ ] **Step 1: Run all changed and neighboring tests**

```bash
rtk env PYTHONPATH=. /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow -q
rtk git diff --check
rtk git status --short
```

Expected: all OV-CapFlow tests pass; no whitespace errors; only planned files differ.

- [ ] **Step 2: Verify claim/evidence alignment**

Every result statement must be tagged `[FACT]`, `[INFERENCE]`, `[DECISION]`, `[OPEN]`, or `[NON-CLAIM]`. Confirm that:

- no old E24 error proportion is presented as current E12 fact;
- no P0148 prediction is called wrong without labels;
- no all-18 result is called strict OV;
- no B* gain is attributed to RISC;
- no candidate is compared to OpenRSD across different mouths;
- every AP number links to exact config/checkpoint/evaluator artifact and SHA.

- [ ] **Step 3: Commit the factual record without pushing**

```bash
rtk git add docs/project_history/exp_20260811_risc_er/flog_risc_er_zh.md docs/project_history/exp_20260811_risc_er/fres_risc_er_zh.md
rtk git commit -m "docs: record RISC excess evidence"
```

No `git push`, deletion, checkpoint cleanup, or change to the historical dirty worktree is part of this plan.
