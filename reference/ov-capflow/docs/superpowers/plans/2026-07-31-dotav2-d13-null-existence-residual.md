# DOTA-v2 D13-N Existence Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Implement, audit, and—only after every Stage-0 gate passes—run the frozen D13-N parent-preserving existence-residual control/candidate pair on physical GPUs 5–9.

**Architecture:** Add one head-owned, zero-RNG, zero-initialized 256-to-1 existence adapter. Capture the actual final-layer Hungarian assignments used by each of the three independent matching groups, train only the detached adapter with a branchless matched/unmatched-balanced BCE, and add its centered log residual after class selection. Freeze all parent parameters before optimizer construction and keep stochastic parent feature modules in eval mode on every train iteration. Formal execution is a fail-closed, sequential control-then-candidate state machine with immutable provenance, exact topology checks, no-replace evidence, complete proxy identity checks, and conditional raw evaluation.

**Tech Stack:** Python 3.8.19; PyTorch 1.12.1+cu113; MMEngine 0.10.4; MMDetection 3.3.0; MMRotate 1.0.0rc1; pytest; five NVIDIA A40 GPUs exposed as physical 5,6,7,8,9; torch distributed with NCCL P2P and IB disabled.

---

## Frozen boundaries

- Scientific authority: docs/superpowers/specs/2026-07-31-dotav2-d13-null-existence-residual-design.md.
- Parent: work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth.
- Parent SHA256: a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8.
- Train manifest: work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/train/manifest.json.
- Train manifest SHA256: 1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290.
- Proxy manifest: work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json.
- Proxy manifest SHA256: a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e.
- Frozen analyzer SHA256: f4f10734948b714dcbae726142e859b3ddac793ee5ab5c78d9a3f3436d99fa23.
- Frozen diagnostics SHA256: bd29dc88650e0db545479c60adaa6a9ee64da1734d3a1756c7cd787c7269c92d.
- Frozen analyzer source commit: abd860d157d562aa9d30422c05e41e246effc8a7.
- Required validator SHA256: 50e63a33823a64cdf8df84b4bdacc32eb0a486f9ea79b9f92e4c7e99c61cc1b3.
- Canonical E24 raw dump: work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl.
- Canonical E24 raw dump SHA256: 39c1d6d3193cbc9e7f8d4e6daace9fc95df86c0d0fc7f1be7870cb13ee092c45.
- Physical GPU allowlist for every D13-N compute process: 5,6,7,8,9 only.
- Control port 29842; candidate port 29841; raw-control port 29843; raw-candidate port 29844.
- Stage-0 derived-parent/control/candidate dump ports: 29845,29846,29847.
- Every distributed launch exports NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1.
- Formal arms are sequential. Candidate release requires a valid control Epoch 12 and a fresh idle/provenance/topology recheck.
- Do not modify analyze_dotav2_q600_dump.py or dotav2_q600_diagnostics.py.
- Do not stop or repurpose the independent D12 monitor.
- Do not delete, overwrite, clean, reset, resume, auto-retry, or silently rescue any formal artifact.
- Every report_sha256, bundle_sha256, or fingerprint_sha256 field is the
  lowercase SHA256 of canonical sorted compact UTF-8 JSON after omitting only
  that self-hash field; tests use this one non-self-referential convention.

## Task 1: Freeze the fail-closed dump validator dependency

**Files:**

- Modify: projects/OVCapFlow/tools/validate_dotav2_q600_dump.py
- Modify: tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py
- Modify: docs/superpowers/plans/2026-07-31-dotav2-d13-null-existence-residual.md

**Audit revision:** The initially tracked validator identity
`b5c1d3cb6ec515a7ed706c231bbe9888f72a512f1ec43df47a6e32498d742a4a`
is rejected historical evidence only. It implemented scientific gates with
`assert`; `python -O` removed those gates and accepted duplicate image IDs,
non-finite boxes, out-of-range labels, and malformed GT. No Stage-0 or formal
artifact may accept that hash. The required validator is the explicit
fail-closed implementation whose checks remain active under `python -O`.
The next historical identity,
`08c97e4d21d8069a7fb2fa1737412eab22853023d3d25f7a1ca870a4698953e5`,
is optimization-safe, but its plain `ValueError` contract is incompatible
with the frozen analyzer's `AssertionError` validation boundary: a canonical
record-count failure is degraded from the precise `record`/`13833` diagnostic
to a generic dump-load error. No Stage-0 or formal artifact may accept that
compatibility-incomplete hash either. The required validator defines
`DumpValidationError(ValueError, AssertionError)` and raises it for every
validator contract error, preserving both public interfaces without weakening
any fail-closed gate or modifying the frozen analyzer.

- [x] **Step 1: Reproduce the optimization-mode fail-open bug with RED tests**

Add tests covering positive contract arguments, record count and structure,
required prediction/GT keys, unique image IDs, every prediction shape and CPU
device, prediction finiteness and integer label range, GT `[N,5]`/`[N]`
shape and length parity, GT finiteness and integer label range, empty GT,
malformed tensor containers, and non-CPU meta tensors. Exercise the CLI in
ordinary and `python -O` modes with duplicate IDs, prediction NaN, label 99,
and GT NaN.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py -q
~~~

Observed RED before the fix: `43 failed, 4 passed`; specifically, the
optimized label-99 CLI returned exit code zero.

- [x] **Step 2: Replace every assertion gate with explicit fail-closed validation**

`validate_records` must contain no `assert`. Every contract violation raises
an explicit `DumpValidationError`, which is simultaneously a `ValueError` and
an `AssertionError`: positive integer arguments; sequence/mapping/key
structure; hashable unique image IDs; tensor/container conversion; exact
prediction shapes; prediction CPU devices and finite boxes/scores; non-bool
integer labels in `[0,num_classes)`; exact GT shapes and equal lengths; GT CPU
devices, finite boxes, and non-bool integer labels in range. Empty GT remains
valid. Helper and chained validation failures use the same exception. No check
may disappear under `python -O`.

- [x] **Step 3: Prove GREEN, compilation, and the corrected frozen identity**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py -q
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py::test_analyzer_python_optimized_canonical_count_gate_is_explicit -q
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= /data/zcy/anaconda3/envs/mmdet/bin/python -m py_compile projects/OVCapFlow/tools/validate_dotav2_q600_dump.py tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py
rtk sha256sum projects/OVCapFlow/tools/validate_dotav2_q600_dump.py
~~~

Expected: validator suite `55 passed`; the frozen analyzer compatibility test
passes and retains the precise `record`/`13833` diagnostic; compilation
succeeds; implementation SHA256 is exactly
`50e63a33823a64cdf8df84b4bdacc32eb0a486f9ea79b9f92e4c7e99c61cc1b3`.

- [x] **Step 4: Commit only the validator, its regression tests, and this audit revision**

Run:

~~~bash
rtk git add projects/OVCapFlow/tools/validate_dotav2_q600_dump.py tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py docs/superpowers/plans/2026-07-31-dotav2-d13-null-existence-residual.md
rtk git diff --cached --check
rtk git commit -m "fix: make q600 validator fail closed"
~~~

## Task 2: Add the zero-RNG existence primitive and grouped loss

**Files:**

- Create: projects/OVCapFlow/ov_capflow/existence_residual.py
- Create: tests/test_projects/ov_capflow/test_existence_residual.py
- Modify: projects/OVCapFlow/ov_capflow/__init__.py

- [ ] **Step 1: Write failing primitive tests**

Tests must assert:

1. construction preserves torch.get_rng_state() bitwise;
2. parameter names are weight and bias, shapes are [1,256] and [1], both are exact zero, and the count is 257;
3. centered_existence_log_residual returns exact positive zero for zero fp16 and fp32 logits, including extreme companion values;
4. grouped_existence_bce_loss matches a hand calculation for mixed strata;
5. empty and all-matched groups are finite and backpropagate;
6. correctly classified `+inf/-inf` logits return exact zero with a non-NaN
   backward pass, while wrong-sign infinities may return infinity but never
   NaN;
7. each of `[0,3,600]`, `[1,0,600]`, and `[1,3,0]` is rejected explicitly;
8. no data-dependent Python branch or .item() exists in the numerical loss
   path; fixed shape/dtype metadata validation branches are permitted;
9. a zero-weight AdamW step leaves the control adapter bitwise zero.

Use this public API:

~~~python
class ExistenceResidual(nn.Linear):
    def __init__(self, embed_dims: int = 256):
        super().__init__(embed_dims, 1, bias=True)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()


def centered_existence_log_residual(logits: Tensor) -> Tensor:
    logits = logits.float()
    return F.logsigmoid(logits) - F.logsigmoid(torch.zeros_like(logits))


def grouped_existence_bce_loss(
        logits: Tensor, matched_mask: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError('logits must have shape [batch, groups, queries]')
    if matched_mask.shape != logits.shape:
        raise ValueError('matched_mask must match logits')
    if matched_mask.dtype != torch.bool:
        raise TypeError('matched_mask must be boolean')
    if 0 in logits.shape:
        raise ValueError('logits dimensions must be non-empty')
    logits = logits.float()
    matched = matched_mask.to(dtype=logits.dtype)
    unmatched = 1.0 - matched
    matched_count = matched.sum(dim=-1)
    unmatched_count = unmatched.sum(dim=-1)
    signed_logits = torch.where(matched_mask, -logits, logits)
    element_loss = F.softplus(signed_logits)
    zero_loss = torch.zeros_like(element_loss)
    matched_mean = torch.where(
        matched_mask, element_loss, zero_loss).sum(
            dim=-1) / matched_count.clamp_min(1)
    unmatched_mean = torch.where(
        matched_mask, zero_loss, element_loss).sum(
            dim=-1) / unmatched_count.clamp_min(1)
    has_matched = matched_count > 0
    has_unmatched = unmatched_count > 0
    active_count = (
        has_matched.to(dtype=logits.dtype) +
        has_unmatched.to(dtype=logits.dtype)).clamp_min(1)
    group_loss = (
        torch.where(has_matched, matched_mean, torch.zeros_like(matched_mean)) +
        torch.where(has_unmatched, unmatched_mean,
                    torch.zeros_like(unmatched_mean))
    ) / active_count
    return group_loss.mean()
~~~

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_existence_residual.py -q
~~~

Expected: FAIL because existence_residual.py does not exist.

- [ ] **Step 2: Implement the minimum branchless float32 reduction**

Require a non-empty `[B,G,Q]` tensor. Reject any zero dimension, shape
mismatch, non-3D logits, and non-bool masks before numerical reduction. Build
one per-element signed logit with `torch.where(matched_mask, -logits, logits)`
and evaluate softplus only for that selected target sign. Use `torch.where`,
not multiplication, to split and activate matched/unmatched strata so an
inactive `0 * inf` cannot create NaN. Divide per-[B,G] sums by clamped tensor
counts, divide by the clamped active-stratum count, and average over groups and
images. The numerical path remains float32 and has no data-dependent Python
branch.

- [ ] **Step 3: Export and rerun**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_existence_residual.py -q
rtk git diff --check
~~~

Expected: PASS; no ordinary random-initialized Linear path exists.

- [ ] **Step 4: Commit**

Run:

~~~bash
rtk git add projects/OVCapFlow/ov_capflow/existence_residual.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_existence_residual.py
rtk git commit -m "feat: add deterministic existence residual"
~~~

## Task 3: Extend the score readout without changing parent behavior

**Files:**

- Modify: projects/OVCapFlow/ov_capflow/calibration.py
- Modify: tests/test_projects/ov_capflow/test_calibration.py

- [ ] **Step 1: Add failing identity and ordering tests**

Add log_residual=None to calibrate_selected_log_scores. Tests must prove:

- omitted and explicit zero residual are torch.equal for extreme fp16/fp32 selected scores;
- a nonzero residual changes only the returned score tensor;
- residual addition occurs after power/temperature and before clamp/exp;
- null_logits and capacity legacy paths are bitwise unchanged when log_residual is omitted;
- shape mismatch raises ValueError.

Target calculation:

~~~python
calibrated = selected_log_scores.float() * (power / temperature)
if null_logits is not None:
    calibrated = calibrated + F.logsigmoid(-null_logits.float())
if capacity is not None:
    calibrated = calibrated + capacity.float().clamp(min=1e-8).log()
if log_residual is not None:
    if log_residual.shape != selected_log_scores.shape:
        raise ValueError('log_residual must match selected_log_scores')
    calibrated = calibrated + log_residual.float()
return calibrated.clamp(min=-80.0, max=0.0).exp()
~~~

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_calibration.py -q
~~~

Expected: FAIL because log_residual is not accepted.

- [ ] **Step 2: Implement the optional residual and rerun regression tests**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_calibration.py tests/test_projects/ov_capflow/test_null_reservoir.py -q
rtk git diff --check
~~~

Expected: PASS and legacy HRSC/null behavior unchanged.

- [ ] **Step 3: Commit**

Run:

~~~bash
rtk git add projects/OVCapFlow/ov_capflow/calibration.py tests/test_projects/ov_capflow/test_calibration.py
rtk git commit -m "feat: support centered existence score residual"
~~~

## Task 4: Capture actual group assignments and integrate the head

**Files:**

- Modify: projects/OVCapFlow/ov_capflow/ov_capflow_head.py
- Create: tests/test_projects/ov_capflow/test_d13n_head.py
- Modify if an existing assertion belongs there: tests/test_projects/ov_capflow/test_grouped_queries.py

- [ ] **Step 1: Write failing head-contract tests**

Cover these cases with lightweight stubs where possible and a real head fixture where assignment behavior matters:

- existence_loss_weight=None creates no module and preserves the legacy state dict;
- weights 0.0 and 1.0 both create exactly one ExistenceResidual;
- invalid coefficients fail closed;
- training takes only hidden_states[-1][:,-1800:,:], reshapes [B,3,600,256], and detaches before the adapter;
- get_targets masks captured from the actual parent loss path have [B,600] per group and are stacked to [B,3,600];
- intentionally distinct assignments for groups 0,1,2 remain distinct;
- DN prefix values cannot change logits or targets;
- replay used only for balanced classification cannot overwrite the actual capture;
- matched counts equal min(num_gt,600) for empty, ordinary, and 1,223-GT fixtures;
- control loss remains differentiable but produces zero head gradients;
- candidate gradients reach only weight and bias, never hidden states;
- inference uses all 600 rows and a nonzero adapter changes scores only, not labels, boxes, row order;
- a zero adapter is torch.equal to legacy scores, labels, and boxes.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_head.py tests/test_projects/ov_capflow/test_grouped_queries.py -q
~~~

Expected: FAIL because the head has no D13-N path.

- [ ] **Step 2: Add fail-closed construction**

Extend OVCapFlowHead.__init__ with existence_loss_weight=None. Construct the
adapter only when the value is not None. Store the numeric coefficient,
last_existence_logits, last_existence_residual,
last_matching_group_masks, and detached tensor-valued per-group counts,
logit/residual quantiles, and would-be readout clamp-hit counts needed by
telemetry. Compute the clamp statistics from unchanged final parent class
scores plus the centered residual; they are diagnostic scalars only and never
enter a loss, matcher, or inference filter.

- [ ] **Step 3: Capture actual assignment targets**

Override get_targets so capture is opt-in. Around each _matching_loss_by_feat invocation:

1. reset a local capture list;
2. enable capture;
3. call the unchanged parent loss path;
4. disable capture in finally;
5. retain the final actual decoder-layer mask for that group;
6. append it to the current loss-call group list.

Suppress capture only around the existing balanced-classification replay. At the beginning of loss_by_feat reset the group list; after grouped_matching_losses returns, require exactly matching_query_groups masks and stack them. Keep last_matching_mask as group 0 only for the disabled legacy null path.

- [ ] **Step 4: Add detached training loss**

After parent losses and actual assignments are available:

~~~python
matching_hidden = hidden_states[-1][:, -1800:, :]
matching_hidden = matching_hidden.reshape(batch_size, 3, 600, 256)
existence_logits = self.existence_residual(
    matching_hidden.detach()).squeeze(-1)
raw_loss = grouped_existence_bce_loss(
    existence_logits, self.last_matching_group_masks)
if self.existence_loss_weight == 0.0:
    losses['loss_existence'] = (
        self.existence_residual.weight.sum() * 0.0
        + self.existence_residual.bias.sum() * 0.0)
else:
    losses['loss_existence'] = raw_loss
~~~

The branch is over the frozen exact `0.0`/`1.0` configuration metadata, not
over tensor data. Always compute `raw_loss` for candidate execution and
diagnostics. The control graph must instead use both finite adapter parameter
sums so wrong-sign infinite logits cannot create `0 * inf = NaN`, while both
optimizer state slots can still initialize with exact finite zero gradients.
The candidate uses `raw_loss` directly, so an infinite or NaN anomaly remains
visible rather than being hidden. Derive 1800 from the frozen dn_meta geometry,
validate every dimension, and do not hard-code a permissive fallback.

- [ ] **Step 5: Add centered inference readout**

For inference, slice exactly hidden_states[-1][:,-600:,:], compute logits and centered log residual, pass one image residual into _predict_by_feat_single, and pass it to calibrate_selected_log_scores only after class argmax.

- [ ] **Step 6: Run focused and legacy tests**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_head.py tests/test_projects/ov_capflow/test_grouped_queries.py tests/test_projects/ov_capflow/test_calibration.py tests/test_projects/ov_capflow/test_null_reservoir.py -q
rtk git diff --check
~~~

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

~~~bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_d13n_head.py tests/test_projects/ov_capflow/test_grouped_queries.py
rtk git commit -m "feat: train existence residual from actual assignments"
~~~

## Task 5: Freeze before optimizer construction and enforce parent eval mode

**Files:**

- Modify: projects/OVCapFlow/ov_capflow/freeze_except_hook.py
- Modify: projects/OVCapFlow/ov_capflow/ov_capflow.py
- Create: projects/OVCapFlow/ov_capflow/d13n_mode_hook.py
- Modify: projects/OVCapFlow/ov_capflow/__init__.py
- Create: tests/test_projects/ov_capflow/test_d13n_freeze_mode.py
- Modify: tests/test_projects/ov_capflow/test_freeze_except_hook.py

- [ ] **Step 1: Write failing freeze and mode tests**

Require:

- apply_freeze_except(model, patterns) is a pure reusable function and fails on no match;
- OVCapFlow constructor applies freeze_except_patterns only after all modules exist;
- exact allowlist is ^bbox_head\.existence_residual\.(weight|bias)$;
- named trainable tensors are exactly the two adapter tensors and total 257;
- the registered D13NOptimWrapperConstructor is used through MMEngine's real
  build_optim_wrapper path, while a regression proves the default constructor
  incorrectly retains frozen parent parameters and is forbidden for D13-N;
- the D13-N constructor rejects caller paramwise_cfg overrides, forces an
  internal non-empty bypass_duplicate=True paramwise configuration so
  MMEngine add_params filters requires_grad=False tensors, and validates
  before and after construction that the optimizer contains exactly the two
  ordered adapter tensor objects/names and 257 scalars;
- Torch 1.12 parameter deduplication is never trusted for alias safety: a
  shared all-path audit uses named_modules(remove_duplicate=False) plus each
  module's direct parameter registrations, groups paths by Parameter identity,
  and requires each expected adapter identity to occur at only its one exact
  canonical path;
- apply_freeze_except completes that all-path audit before changing any
  requires_grad bit; an identity split across matched and unmatched paths, or
  any trainable identity with multiple/noncanonical paths, fails closed with
  zero partial mutation. Shared aliases of ordinary frozen parent parameters
  remain legal but are excluded from the optimizer;
- D13NParentEvalModeHook.before_save_checkpoint maps the two live optimizer
  tensor objects, in serialized param-group order, back only to the two unique
  canonical names from the shared all-registration-path audit and refuses
  aliases, unknown, duplicate, missing, or reordered entries;
- the checkpoint receives hash-bound meta.d13n_integrity with schema, role,
  epoch/iter, ordered optimizer names by group, optimizer class, live and
  serialized group sizes/parameter IDs, exactly two initialized optimizer
  state slots, parent-state hash, adapter-state hash, config hash, and an
  integrity_sha256 computed with that field omitted;
- checkpoint publication requires the exact torch.optim.AdamW class and exact
  live/serialized param-group topology [1,1] in canonical weight,bias order;
  empty or additional groups are forbidden;
- every raw-live, live-state_dict, and serialized param_groups outer container
  is an exact built-in list; every group is an exact built-in dict and its
  params is an exact built-in list. Every group has exactly the Torch-1.12
  AdamW fields params, lr, betas, eps, weight_decay, amsgrad, foreach,
  maximize, and capturable, with only scheduler-added initial_lr optional. The
  lr, eps, weight_decay, optional initial_lr, and both betas are exact built-in
  floats, never ints, bools, Real subclasses, or equality-overloading objects;
  they are finite and in their optimizer domains, and betas is an exact
  built-in tuple in [0,1). amsgrad/maximize/capturable are exactly false,
  foreach is exactly None, and both groups have identical non-param keys and
  values;
- live and serialized OptimWrapper state has exactly state, param_groups, and
  base_param_settings. Both state_dict top-level containers and their state
  mappings are exact built-in dicts; every raw-live/live-state_dict/serialized
  group, base mapping, and state slot is an exact built-in dict. The raw live
  optimizer.state outer mapping must have exact type collections.defaultdict
  with default_factory exactly dict; its keys must be the two canonical live
  parameters in exact group order and by object identity before any slot access,
  and its two slot values must be exact dicts. Plain dicts, defaultdict
  subclasses, altered factories, reordered keys, and equality-overloading
  containers are forbidden. The live wrapper base mapping is
  authoritative and its serialized copy has the same exact AdamW schema,
  built-in float types, and domains as the two real groups, including
  consistent presence or absence of optional initial_lr and identical
  non-param values across all three groups. Its params field is only
  MMEngine's exact type-is-Tensor CPU float32 shape-[1] all-zero dummy with
  requires_grad false, grad None, and is_inference false, never a
  Parameter/subclass or model tensor/storage. It must own canonical contiguous
  standard-strided storage with the exact row-major stride implied by its
  shape (including singleton dimensions): no view base or storage offset,
  storage size exactly numel, and neither conjugate nor negative view bits set;
- each live and serialized AdamW state slot has exactly step, exp_avg, and
  exp_avg_sq: the Torch-1.12 step is a finite positive integer-valued scalar
  in its frozen CPU-float32 representation; live moments match their live
  parameter's shape/dtype/device while serialized moments match shape/dtype
  but are on CPU because MMEngine moves checkpoint tensors before calling the
  hook. Every state value has exact type Tensor, requires_grad false, grad
  None, and is_inference false; both moments are finite and exp_avg_sq is
  elementwise nonnegative. Every step and moment also owns canonical contiguous
  standard-strided storage with the exact row-major stride implied by its
  shape, including singleton dimensions, no view base or offset, storage size
  exactly numel, and neither conjugate nor negative view bits set; expanded,
  sliced, offset, conjugate, negative, and fake-contiguous singleton-stride
  tensors are forbidden. After both sides validate, every raw-live step/moment
  must be type/shape/dtype/device/value-equal to its live OptimWrapper.state_dict
  counterpart through a device-aware comparison with CPU value comparison. A
  genuine MMEngine StepLR over the OptimWrapper must retain exact built-in float
  settings and pass save, scheduler-state resume, resave, and second resume;
- the six live state tensors are pairwise identity/storage-disjoint and the six
  serialized tensors independently are too; each base dummy is disjoint from
  its six state tensors. Live audited tensors are identity/storage-disjoint
  from every registered model parameter/buffer, and serialized audited tensors
  are disjoint from every supplied checkpoint state_dict tensor. Only valid
  nonempty canonical standard-strided Torch-1.12 storage/data_ptr evidence is
  accepted;
- serialized state mapping keys are exact ints in the exact flattened canonical
  param-group ID order; bool/int key equivalence and reordered state mappings
  are forbidden, and recursive live/serialized mapping equality includes key
  types as well as order and values. Every recursive mapping/list/tuple
  container must also have the exact same concrete type, so equal-content
  subclasses cannot substitute for built-in containers;
- any present meta.d13n_integrity field, including None, must be an exact dict
  with the exact expected key schema/order. Its lowercase hexadecimal
  integrity_sha256 is independently recomputed after omitting only that field,
  then the complete structure is recursively compared to fresh integrity with
  exact key/value and recursive container types, order, and values;
  equal-valued bool/int/float substitutions and equality-overloading objects
  are forbidden;
- checkpoint meta is bound to the live by-epoch runner: runner epoch/iter are
  exact nonnegative integers, meta.epoch equals runner.epoch+1, meta.iter
  equals runner.iter, and meta.cfg exactly equals the nonempty
  runner.cfg.pretty_text. The configured role, runner.cfg.pretty_text, and
  meta.cfg must each be exact built-in str, not subclasses or objects with
  overloaded equality. The hook's construction-time arm is stored only in a
  module-private WeakKeyDictionary keyed by the exact hook object, never in an
  instance field. The read-only public role property and every checkpoint save
  read and revalidate that external authority; arbitrary synchronized role-like
  instance fields cannot change the published arm, and a missing authority
  fails closed. HOOKS.build constructs normally, while deepcopy and pickle
  reconstruct through __init__ so each new exact object receives its own
  authority without trusting instance fields. Formal D13-N uses save_best=None,
  so no alternative save convention is admitted;
- deleting, renaming, swapping, or adding either live optimizer tensor, a
  serialized optimizer parameter ID/state slot, or an integrity field makes
  checkpoint publication fail;
- model.train() followed by D13NParentEvalModeHook.before_train_iter keeps detector.training True, all direct parent feature children eval, bbox_head parent eval, and existence_residual train;
- repeated enforcement is idempotent;
- DDP-style model.module unwrapping works;
- after_train_iter publishes the cached per-group matched counts, existence-logit/residual quantiles, and clamp-hit counts into runner.message_hub under fixed d13n/* keys without retaining graphs;
- identical repeated input produces bitwise-identical parent feature tensors;
- one candidate step changes a head tensor and no parent tensor/buffer;
- one control step changes nothing.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_freeze_except_hook.py tests/test_projects/ov_capflow/test_d13n_freeze_mode.py -q
~~~

Expected: FAIL because the pure helper, constructor freeze, and mode hook do not exist.

- [ ] **Step 2: Refactor the freeze helper and optimizer constructor**

FreezeExceptHook.apply delegates to apply_freeze_except. Add freeze_except_patterns=None to OVCapFlow.__init__, call super().__init__ first, then call the helper. Do not rely on before_train for D13-N freezing.

Register D13NOptimWrapperConstructor as a thin
DefaultOptimWrapperConstructor subclass. It rejects every caller-supplied
paramwise_cfg, installs only the internal non-empty
`dict(bypass_duplicate=True)` configuration, validates the exact ordered
trainable allowlist before delegating to MMEngine, and validates the flattened
live optimizer tensor identities/order after delegation. Do not copy
MMEngine's optimizer-building implementation.

The freeze helper, constructor pre/post checks, and checkpoint hook share the
same Torch-1.12-compatible all-registration-path audit. Direct Parameter
aliases, shared existence_residual module aliases, and aliases injected after
optimizer construction all fail. Checkpoint-time live ID-to-name binding is
derived only from the two uniquely registered canonical adapter paths, never
from deduplicated model.named_parameters().

- [ ] **Step 3: Implement per-iteration mode enforcement**

D13NParentEvalModeHook unwraps runner.model.module, requires
bbox_head.existence_residual, sets the detector root training flag true
without recursively retraining children, puts parent children in eval, and
finally puts only existence_residual in train. Apply in before_train_iter. In
after_train_iter, move only the already-detached scalar telemetry to CPU and
update fixed runner.message_hub keys so the ordinary logger and 60-second
monitor have an auditable source.

Implement before_save_checkpoint against the actual runner.model,
runner.optim_wrapper, and the checkpoint dict already containing state_dict
and optimizer. Bind every serialized param-group position to the corresponding
live optimizer tensor object and model parameter name. Require the exact two
allowlisted names once each, the exact torch.optim.AdamW class, exact [1,1]
live/serialized group topology, two unique serialized parameter IDs, and two
initialized state slots. Require the exact Torch-1.12 group schema and domains,
allow only optional scheduler-added initial_lr, and require identical
non-parameter settings across groups. Require exact built-in float types for
lr, eps, weight_decay, optional initial_lr, and both beta values before any
equality comparison. Require exact OptimWrapper top-level state, param_groups,
base_param_settings fields in both the live state_dict and serialized
checkpoint; their top/state containers are exact dicts, every group/base/slot
is an exact dict, and every param_groups/params container is an exact list. Raw
live groups, base, and slots obey the same exact-container contract. The raw
optimizer.state must be exact collections.defaultdict with default_factory
exactly dict; validate its canonical parameter key order and object identities
before length, iteration, equality, or slot indexing can reach attacker-defined
behavior. Validate the live base mapping as authority and its serialized copy
as an exact pseudo-group: same schema/domains, exact numeric types, and
non-param values as both real groups;
optional initial_lr is present in all three or none. Its params is only the
exact type-is-Tensor CPU float32 shape-[1] all-zero dummy with requires_grad
false, grad None, and is_inference false, and canonical owned contiguous
standard-strided storage with exact row-major shape stride, including singleton
dimensions: no view base or storage offset, storage size exactly numel, and
neither conjugate nor negative view bits set. Each live and serialized slot
must contain exactly step, exp_avg, and exp_avg_sq; every value is exact type
Tensor with requires_grad false, grad None, and is_inference false, owns the
same canonical storage and exact-stride form, and the observed Torch-1.12
CPU-float32 scalar step is finite, positive, and integer-valued. Live moments
match their parameter shape/dtype/device;
MMEngine has already moved serialized checkpoint moments to CPU, so those match
shape/dtype, must be CPU, and remain value-equal to live state through CPU
comparison. After validating both raw live slots and live state_dict slots,
explicitly bind every step, exp_avg, and exp_avg_sq by exact type, shape, dtype,
device, and CPU-compared value before comparing the checkpoint serialization.
All moments are finite and exp_avg_sq is elementwise nonnegative.
Require pairwise identity/storage disjointness across each six-tensor
live/serialized state, between each base dummy and its state, between live
audited tensors and all registered model parameters/buffers, and between
serialized audited tensors and every checkpoint state_dict tensor; reject
invalid/non-strided/zero-pointer state storage. Serialized state keys must be
exact ints in exact flattened canonical param-ID order, and recursive mapping
equality binds key types so bool never substitutes for int and binds every
recursive mapping/list/tuple container to its exact type, rejecting subclasses.
Bind
meta.epoch/meta.iter/meta.cfg exactly to nonnegative runner.epoch+1,
runner.iter, and nonempty runner.cfg.pretty_text under the formal by-epoch,
save_best=None convention. Require the hook role, runner.cfg.pretty_text, and
meta.cfg to be exact built-in str values, never subclasses or
equality-overloading objects. Store the validated configured role only in a
module-private WeakKeyDictionary keyed by the exact hook object. Make the
public property and checkpoint publication read/revalidate only that authority,
ignore arbitrary instance role-like fields, and fail closed if the authority is
missing. Implement deepcopy/pickle as safe __init__ reconstruction from the
external authority, never instance __dict__. Store that ordered binding plus
the role configured on the hook, epoch/iter, optimizer class, state/config
hashes, and a canonical self-hash under meta.d13n_integrity. If that field is
present at all,
including when its value is None, require an exact dict and expected key order,
recompute its exact lowercase self-hash by omitting only integrity_sha256, and
recursively compare exact nested scalar and container types, order, and values
to the fresh structure. This checkpoint-time binding is the sole authority for
names in a serialized optimizer; requires_grad alone and bare integer optimizer
IDs are never accepted as evidence.

- [ ] **Step 4: Rerun and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_freeze_except_hook.py tests/test_projects/ov_capflow/test_d13n_freeze_mode.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/ov_capflow/freeze_except_hook.py projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/ov_capflow/d13n_mode_hook.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_freeze_except_hook.py tests/test_projects/ov_capflow/test_d13n_freeze_mode.py
rtk git commit -m "feat: isolate D13-N optimizer and parent modes"
~~~

## Task 6: Freeze the sequential GPU5–9 configurations

**Files:**

- Create: configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_control.py
- Create: configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_candidate.py
- Create: configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_control_raw13833.py
- Create: configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_candidate_raw13833.py
- Create: tests/test_projects/ov_capflow/test_d13n_world5_configs.py

- [ ] **Step 1: Write failing resolved-config tests**

Both configs must resolve to:

- physical_gpus=(5,6,7,8,9), selected_world_size=5;
- Q600, three matching groups, DN100, six decoder layers;
- batch_size=2, accumulation=1, 160 updates per epoch, 1,920 updates total;
- seed 20260716, diff_rank_seed=False, resume=False;
- identical E24 load_from and unique never-used work/audit paths;
- audit_path contains {epoch:02d} and audit_noreplace=True in both arms;
- all 12 epoch checkpoints retained, no best replacement;
- decoder null, semantic fusion, density capacity, balanced classification, position supervision, adaptive DN, capacity readout, legacy null loss, and density loss disabled;
- freeze allowlist exactly the two existence tensors;
- optim_wrapper.constructor='D13NOptimWrapperConstructor', with
  paramwise_cfg absent or None in both arms so the registered fail-closed
  constructor is the only parameter-collection path;
- D13NParentEvalModeHook enabled in both arms with its corresponding frozen
  control/candidate role and otherwise identical settings, so checkpoint
  integrity metadata cannot inherit or guess the arm role;
- control coefficient 0.0, candidate 1.0;
- resolved configs differ only in role, coefficient, port, and path fields;
- explicit formal ports 29842 and 29841, raw ports 29843 and 29844.

The two raw wrappers must additionally resolve to:

- test-only metadata that the owner refuses to pass to tools/train.py;
- the exact E24 scale1024 DOTA-v2 ss_val mouth at /data1/zcy/datasets/DOTA2_1024_500/, ss_val/annfiles/, and ss_val/images/;
- filter_empty_gt=False, test_mode=True, the frozen ordered 18-class metainfo, return_classes=True, and the unchanged scale1024 rotated validation pipeline;
- Q600, one inference group, no NMS/top-k/dense mouth, and the same disabled mechanisms/readout as the corresponding training arm;
- exactly 13,833 dataset rows and unique image IDs when the real dataset is built;
- no optimizer or train-loop execution in raw mode;
- raw control coefficient 0.0 and raw candidate coefficient 1.0;
- owner-supplied fixed E12 checkpoints rather than E24 load_from being used for adjudication.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_world5_configs.py -q
~~~

Expected: FAIL because configs do not exist.

- [ ] **Step 2: Implement thin wrappers**

The control inherits the rare4x scale1024 recipe, sets E24 load, all disable
flags, constructor freeze, the registered D13NOptimWrapperConstructor with no
caller paramwise_cfg, mode hook, batch/retention/runtime metadata, and
coefficient 0.0. Candidate inherits control and changes only the
preregistered fields. The resolved optimizer constructor and lack of a
paramwise override must be identical across both arms.

Each raw wrapper inherits its corresponding arm and overrides only the
test_dataloader/val_dataloader mouth and test-only path/port metadata from the
canonical E24 scale1024 config. Mark train_cfg, train_dataloader,
optim_wrapper, and param_scheduler unavailable in the raw resolved config so a
raw wrapper cannot be used for an optimizer step. The owner passes the
explicit arm Epoch-12 checkpoint on the d13n_test.py argv.

- [ ] **Step 3: Prove config parity and exact coverage**

Reuse the D12 exact-cover fixture but require both arms on physical GPUs5–9 and exactly 1,600 row indices with no padding. Also assert 1,492 unique image IDs plus the preregistered 108 repeated exposures when the real manifest is available. Build both raw datasets and assert 13,833 unique IDs, 600-query model output, all 18 configured classes, filter_empty_gt=False, and byte-identical normalized raw mouths apart from role/path/port/coefficient.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_world5_configs.py tests/test_projects/ov_capflow/test_d12_world5_configs.py -q
rtk git diff --check
~~~

- [ ] **Step 4: Commit**

Run:

~~~bash
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_control.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_candidate.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_control_raw13833.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_candidate_raw13833.py tests/test_projects/ov_capflow/test_d13n_world5_configs.py
rtk git commit -m "config: freeze D13-N sequential world5 pair"
~~~

## Task 7: Make sampler evidence epoch-qualified and no-replace

**Files:**

- Create: projects/OVCapFlow/ov_capflow/no_replace.py
- Modify: projects/OVCapFlow/ov_capflow/dn_budget_batch_sampler.py
- Modify: projects/OVCapFlow/ov_capflow/__init__.py
- Modify: projects/OVCapFlow/tools/audit_sampler_coverage.py
- Create: tests/test_projects/ov_capflow/test_no_replace.py
- Modify: tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py
- Modify: tests/test_projects/ov_capflow/test_sampler_coverage_audit.py

- [ ] **Step 1: Add failing no-clobber tests**

Require:

- an audit template containing {epoch:02d} resolves to a distinct path per epoch;
- audit_noreplace=True uses an exclusive create and fsync, never Path.write_text replacement;
- publication stages to an exclusive .pending.PID.INDEX path, fsyncs, hard-links atomically to the final path, fsyncs the directory, and removes only the successfully published temporary;
- an occupied final or any matching pending path raises FileExistsError without changing bytes;
- independent audit materializes plans with audit_path=None and publishes its enriched report once;
- repeated ranks never race to publish; only rank 0 writes;
- audit_noreplace defaults to False so old fixed-path D11/D12 configs retain their existing replacement behavior and report schema.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_no_replace.py tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py tests/test_projects/ov_capflow/test_sampler_coverage_audit.py -q
~~~

Expected: FAIL on replacement/no-template assertions.

- [ ] **Step 2: Implement one exclusive publication helper**

Extract the tracked prepare_d12_checkpoint_pair.py staging/link algorithm into
no_replace.py as publish_stream_noreplace, publish_bytes_noreplace, and
publish_json_noreplace. The stream form accepts a writer callable so multi-GB
dumps are serialized directly into the pending file without constructing a
second in-memory byte copy. publish_json_noreplace writes sorted compact UTF-8
JSON on exactly one line plus a trailing newline so frozen JSONL readers can
consume it. It must
perform two collision scans, exclusive pending creation, file fsync, atomic
os.link to the final, directory fsync, and cleanup of only the successfully
published pending file. Add audit_noreplace: bool = False to
DNQueryBudgetBatchSampler. Resolve epoch templates from the report. When
audit_noreplace is true, call the shared atomic helper. When false, retain the
legacy writer exactly. The independent audit must call _build_plan directly
rather than asking the sampler to publish and then overwriting the same path;
its requested output is always a unique no-replace final.

- [ ] **Step 3: Rerun and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_no_replace.py tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py tests/test_projects/ov_capflow/test_sampler_coverage_audit.py tests/test_projects/ov_capflow/test_d13n_world5_configs.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/ov_capflow/no_replace.py projects/OVCapFlow/ov_capflow/dn_budget_batch_sampler.py projects/OVCapFlow/ov_capflow/__init__.py projects/OVCapFlow/tools/audit_sampler_coverage.py tests/test_projects/ov_capflow/test_no_replace.py tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py tests/test_projects/ov_capflow/test_sampler_coverage_audit.py
rtk git commit -m "fix: publish epoch sampler audits without replacement"
~~~

## Task 8: Add the atomic distributed dump metric

**Files:**

- Create: projects/OVCapFlow/ov_capflow/d13n_dump_results.py
- Modify: projects/OVCapFlow/ov_capflow/__init__.py
- Create: tests/test_projects/ov_capflow/test_d13n_dump_results.py

- [ ] **Step 1: Write failing metric tests**

Freeze these public symbols:

- cpu_prediction_tree(value) recursively maps tensors to CPU without changing dtype, shape, value, dictionary/list order, or non-tensor fields.
- dump_results_to_stream(results, stream) writes one pickle stream readable by mmengine.load.
- D13NNoReplaceDumpResults(out_file_path, collect_device='cpu', collect_dir=None) extends MMEngine BaseMetric, gathers the complete ordered distributed result list, and publishes through publish_stream_noreplace.

Tests prove occupied final/pending/concurrent publication preserves existing
bytes; a reader sees either no final or one complete pickle; serialization
writes directly to the pending stream without a second full byte copy; and
240,000 synthetic rows retain exact record/row order.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_dump_results.py -q
~~~

Expected: FAIL because d13n_dump_results.py does not exist.

- [ ] **Step 2: Implement, export, rerun, and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_dump_results.py tests/test_projects/ov_capflow/test_no_replace.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/ov_capflow/d13n_dump_results.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_d13n_dump_results.py
rtk git commit -m "feat: publish distributed dumps without replacement"
~~~

## Task 9: Add the deterministic distributed test runner

**Files:**

- Create: projects/OVCapFlow/tools/d13n_test.py
- Create: tests/test_projects/ov_capflow/test_d13n_test.py

- [ ] **Step 1: Write failing parser/config tests**

Freeze this CLI:

~~~text
d13n_test.py CONFIG CHECKPOINT
  --launcher pytorch
  --role stage0-parent|proxy-control|proxy-candidate|raw-control|raw-candidate
  --out PREDICTIONS.pkl
  --metrics-out OFFICIAL_METRICS.json
  --identity-out IDENTITY.json
  [--derive-adapter-free-parent]
~~~

Exit 0 means a complete test and three no-replace outputs; argparse usage is
2; output collision is 3; role/config/checkpoint identity failure is 4;
model/test runtime failure is 5. All three output paths are required, absolute
after resolution, pairwise distinct, and unoccupied before model construction.

Normal roles require matching d13n_role and d13n_mouth config metadata.
raw-control/raw-candidate additionally require train_cfg, train_dataloader,
optim_wrapper, and param_scheduler all None. stage0-parent requires the control
proxy config plus --derive-adapter-free-parent. Every rank deterministically
deletes exactly:

1. model.bbox_head.existence_loss_weight;
2. model.freeze_except_patterns;
3. the single custom_hooks entry whose type is D13NParentEvalModeHook.

No other resolved field may change. The derived parent config hash is gathered
across all five ranks and must be identical before Runner.test(). It must load
E24 with zero missing/unexpected keys. All other D13 roles must reject the
derive flag.

The identity JSON schema is:

~~~text
schema=d13n-test-identity-v1
role, mouth, config_path, config_sha256, resolved_config_sha256
derived_parent, derived_deleted_fields
checkpoint_path, checkpoint_sha256
predictions_path, predictions_sha256
official_metrics_path, official_metrics_sha256
records, unique_image_ids, queries_per_image, prediction_rows, finite
world_size, local_ranks, cuda_visible_devices, nccl_p2p_disable, nccl_ib_disable
~~~

OFFICIAL_METRICS.json contains exactly one JSON line and is directly readable
by the frozen read_metric_record function:

~~~text
schema=d13n-official-metrics-v1
role, mouth, step
dota/mAP, dota/AP50
config_path, config_sha256, checkpoint_path, checkpoint_sha256
records, queries_per_image, prediction_rows, report_sha256
~~~

The focused test passes the generated file to the unchanged
dotav2_q600_diagnostics.read_metric_record and requires exact dota/mAP,
dota/AP50, and step recovery.

- [ ] **Step 2: Implement the runner**

Load Config, validate role, optionally derive the adapter-free parent, set
cfg.launcher='pytorch', replace the dump metric with
D13NNoReplaceDumpResults, bind cfg.load_from to the positional checkpoint, run
Runner.from_cfg(cfg).test(), validate the dump, publish the returned exact
evaluator dota/mAP and dota/AP50 as the one-line official metrics JSON, and
then publish identity JSON no-replace on rank 0 after distributed barriers.
The identity is last so its presence proves both prior artifacts are complete.
Direct tools/test.py --out is forbidden.

- [ ] **Step 3: Test the real argv builder and commit**

Expose build_torchrun_argv(config, checkpoint, role, out, metrics_out,
identity_out, port, derive_adapter_free_parent=False). It returns this prefix
exactly:

~~~text
/data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run
--nproc_per_node=5 --master_port=PORT
projects/OVCapFlow/tools/d13n_test.py CONFIG CHECKPOINT
--launcher pytorch --role ROLE --out OUT
--metrics-out OFFICIAL_METRICS --identity-out IDENTITY
~~~

The caller supplies environment separately; tests require
CUDA_VISIBLE_DEVICES=5,6,7,8,9, both NCCL flags 1, ranks 0..4, and the frozen
role port. The derive flag is appended only for stage0-parent.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_test.py tests/test_projects/ov_capflow/test_d13n_dump_results.py tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/d13n_test.py tests/test_projects/ov_capflow/test_d13n_test.py
rtk git commit -m "feat: run deterministic D13-N test dumps"
~~~

## Task 10: Add shared provenance and topology runtime

**Files:**

- Create: projects/OVCapFlow/tools/d13n_runtime.py
- Create: tests/test_projects/ov_capflow/test_d13n_runtime.py

- [ ] **Step 1: Write failing pure-runtime tests**

Freeze these constants and public functions:

- PHYSICAL_GPUS=(5,6,7,8,9), the five UUIDs from the design; CONTROL_PORT=29842,
  CANDIDATE_PORT=29841, RAW_CONTROL_PORT=29843, RAW_CANDIDATE_PORT=29844,
  STAGE0_PARENT_PORT=29845, STAGE0_CONTROL_PORT=29846,
  STAGE0_CANDIDATE_PORT=29847; and exact Path constants for E24, the canonical
  E24 dump, train manifest, proxy manifest, four configs, analyzer,
  diagnostics, and validator.
- sha256_file(path), canonical_json_sha256(payload), and publish_json_noreplace(path,payload).
- normalized_pair_config(config_path) removing only preregistered role/coefficient/port/path fields.
- collect_environment_fingerprint(repo, control_config, candidate_config, raw_control_config, raw_candidate_config).
- EXPECTED_ENVIRONMENT and validate_frozen_environment(observed). The exact
  expected interpreter, package/runtime versions, driver, and physical GPU
  mapping are constants in the runtime, not values learned from the first
  observed machine.
- assert_fingerprint_equal(expected,observed), classify_process_snapshot(processes), assert_world5_topology(snapshot,master_port), assert_paths_unoccupied(paths), and assert_ports_free(ports).
- MONITOR_STARTUP_TIMEOUT_SECONDS=180,
  MONITOR_HEARTBEAT_MAX_AGE_SECONDS=150,
  publish_integrity_failure_lock, load_integrity_failure_lock,
  publish_monitor_ready, validate_monitor_ready, and
  validate_monitor_heartbeat. These are shared owner/monitor primitives, not
  duplicated implementations.
- CheckpointHashCache keyed by resolved path, size, and mtime_ns, plus
  audit_d13n_checkpoint(checkpoint, stage0_report, role, cache) returning
  parent parameter/buffer hashes, optimizer parameter names, adapter
  hash/zero status, endpoint epoch, and checkpoint SHA. Both owner and monitor
  import this one implementation.

audit_d13n_checkpoint must validate meta.d13n_integrity before returning any
name claim: recompute its self-hash, role/epoch/iter/config and parent/adapter
state hashes; require the exact two ordered allowlisted names captured by the
checkpoint hook; bind their recorded group positions to exactly two unique
integer IDs in checkpoint.optimizer.param_groups; require state keys to be
exact ints in that exact flattened order; and require exactly two corresponding
initialized slots with exact step/exp_avg/exp_avg_sq fields. It must also
require checkpoint.optimizer to have exactly state, param_groups, and
base_param_settings. Standalone auditing requires its top/state mappings,
base/group/slot mappings, param_groups outer sequence, every params sequence,
and betas sequence to be exact built-in dict/dict/list/list/tuple containers as
applicable, never subclasses. Revalidate exact AdamW group schemas/domains,
identical real-group settings, and three-way parity with the base pseudo-group
including all-three-or-none optional initial_lr; require lr, eps, weight_decay,
optional initial_lr, and both betas to be exact built-in floats before any
equality comparison; and revalidate the base as exact
type-is-Tensor CPU-float32 shape-[1] zero dummy with requires_grad false and
grad None and is_inference false, canonical contiguous standard-strided owned
storage with the exact row-major stride implied by shape, including singleton
dimensions, no base/offset, storage size exactly numel, and no
conjugate/negative view bits. Every slot value must satisfy the same exact
Tensor, autograd, inference, canonical-storage, and exact-stride contract; step
is exact CPU-float32 finite positive integer-valued scalar; moments have exact
serialized shape/dtype/CPU device, are finite, and exp_avg_sq is nonnegative.
Revalidate six-state pairwise storage disjointness, base/state disjointness,
optimizer/checkpoint-model disjointness, valid strided storage pointers, and
mapping key types/order. Revalidate meta.d13n_integrity as an exact dict with
the exact expected key schema/order, recompute its lowercase self-hash after
omitting only integrity_sha256, and recursively enforce exact nested key/value
and container types, order, and values; any present field including None must
pass this validation. Require role, live config text, and serialized meta.cfg
to be exact built-in str values and reject subclasses or equality-overloading
objects; role must also equal the expected frozen construction-time control or
candidate arm. The hook's save-time authority proof requires an external
per-object WeakKeyDictionary arm authority, exact raw defaultdict(dict) key
identity/order, and exact raw-slot to live-state_dict tensor equality before
serialization; the standalone auditor then proves the same ordered serialized
topology, slots, wrapper schema, base settings, exact
tensor/autograd/inference/canonical-storage contract, typed metadata, and
no-alias evidence. A checkpoint
with legacy/missing metadata or two anonymous IDs is invalid even if
requires_grad reconstruction would appear plausible.

The shared monitor-ready report is no-replace and hash-bound:

~~~text
schema=d13n-monitor-ready-v1
pid, proc_start_ticks, proc_cmdline_sha256, created_monotonic_ns
owner_root, owner_log, monitor_log, failure_lock
stage0_report_path, stage0_report_sha256, pre_run_commit
report_sha256
~~~

The shared terminal lock schema is exactly:

~~~text
schema=d13n-integrity-failure-v1
status=FAIL
sequence, timestamp, owner_state, failure_class
owner_event={present,sha256}
sample={present,sha256}
evidence_paths={path:{exists,sha256}}
failures, report_sha256
~~~

failure_class is one of identity, topology, nonfinite_fatal, collision,
checkpoint_parent, checkpoint_buffer, optimizer_binding, control_head,
sampler, gpu_contract, or input_runtime. Every evidence path is resolved and
hashed when it exists; a missing/unreadable path that caused the failure is
recorded with exists=False and sha256=None before report_sha256 is computed.
For a pre-sample input/runtime failure, owner_state=UNKNOWN, sequence=0, and
both optional evidence objects explicitly use present=False/sha256=None; this
is a valid terminal lock, not permission to continue.

validate_monitor_ready rehashes the report, verifies every bound path/commit,
reads /proc/PID/stat field 22 and /proc/PID/cmdline, and rejects a missing
process, PID reuse, wrong script/argv identity, wrong output root, or drifted
Stage-0 identity. validate_monitor_heartbeat reads only complete newline-
terminated records from the append-only monitor JSONL, validates the
previous_sample_sha256 chain and latest sample hash/owner identity, and
requires a nonnegative monotonic age no greater than 150 seconds. The first
valid sample is mandatory before formal control launch. The shared failure-
lock publisher uses O_EXCL, fsyncs the file and parent directory, and supports
an explicit absent sample so either process can latch a fail-silent monitor.

The fingerprint top-level schema is exactly:

~~~text
schema=d13n-environment-v1
interpreter={path,python_version}
software={torch,torch_cuda,mmengine,mmdet,mmrotate}
driver={version,reported_cuda}
gpus=[{physical_index,name,uuid}]
git={commit,tracked_dirty}
inputs={name:{path,sha256}}
analyzer_source_commit
fingerprint_sha256
~~~

inputs has exact keys parent, canonical_raw_dump, train_manifest,
proxy_manifest, control_config, candidate_config, raw_control_config,
raw_candidate_config, analyzer, diagnostics, validator. Missing paths or any
hash drift fail before fingerprint publication.

EXPECTED_ENVIRONMENT is exactly:

~~~text
interpreter.path=/data/zcy/anaconda3/envs/mmdet/bin/python
interpreter.python_version=3.8.19
software.torch=1.12.1+cu113
software.torch_cuda=11.3
software.mmengine=0.10.4
software.mmdet=3.3.0
software.mmrotate=1.0.0rc1
driver.version=580.173.02
driver.reported_cuda=13.0
gpus=[
  {physical_index:5,name:NVIDIA A40,uuid:GPU-6c98da7c-69bd-4c20-5094-ff3b60141515},
  {physical_index:6,name:NVIDIA A40,uuid:GPU-b30e9f80-7925-6816-9014-d228784c5a12},
  {physical_index:7,name:NVIDIA A40,uuid:GPU-d614452d-c145-3751-ce23-f9e21dd0b535},
  {physical_index:8,name:NVIDIA A40,uuid:GPU-f8f2c2c6-4c20-237b-f6df-ba5b5408573e},
  {physical_index:9,name:NVIDIA A40,uuid:GPU-ff760a2d-9a1b-38ee-06a0-8c284e002545}]
~~~

collect_environment_fingerprint first gathers a complete observed structure,
then calls validate_frozen_environment before adding input hashes or computing
fingerprint_sha256. Tests independently perturb every scalar field, GPU order,
index, name, UUID, cardinality, and interpreter resolution and require a
fail-closed exception with no output artifact. A successful fingerprint is
therefore validation against a predeclared environment, never self-freezing
of whatever environment happened to be present.

Topology fixtures include rtk env as wrapper ancestor, exactly one real python
-m torch.distributed.run launcher, and five direct workers. Wrong/missing
readable LOCAL_RANK, LOCAL_WORLD_SIZE, WORLD_SIZE, MASTER_PORT,
CUDA_VISIBLE_DEVICES, NCCL flags, duplicate/extra workers, listeners, or
artifacts fail closed without mutation.

Monitor fixtures cover a live exact PID/start-tick/cmdline match, never
started PID, dead PID with no failure lock, reused PID with different start
ticks, wrong argv/root/commit/Stage-0 hash, missing first heartbeat, truncated
last JSONL line, broken sample hash chain, future monotonic timestamp, and
heartbeat ages 150 seconds and 150 seconds plus one nanosecond. Each failure
is distinguishable and publishable as input_runtime through the same shared
failure-lock API.

- [ ] **Step 2: Implement and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_runtime.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/d13n_runtime.py tests/test_projects/ov_capflow/test_d13n_runtime.py
rtk git commit -m "feat: add D13-N provenance and topology runtime"
~~~

## Task 11: Build the complete Stage-0 auditor

> **Execution amendment (latest user instruction):** implement this task as a
> thin adapter over the repository's existing D13-N test/runtime, dump
> validator, strict-inference, open-vocabulary, mouth, freeze/mode, and
> no-replace utilities. Do not build a second audit infrastructure before the
> first zero-step run. Aggregate the requirements below into exactly five
> top-level checks: (1) identity/load authority, (2) static D13-N
> freeze/optimizer/positive-zero authority, (3) complete three-way zero-step
> equality, (4) strict protocol regression, and (5) real finite smoke. The
> detailed bullets below define evidence inside those five checks; they do not
> authorize separate analyzer-bundle, monitor, work-dir-snapshot, or report
> frameworks. Keep the implementation concentrated and run the existing
> `d13n_test.py` zero-step path as soon as GPUs5--9 are jointly idle.

**Files:**

- Create: projects/OVCapFlow/tools/audit_d13n_stage0.py
- Create: tests/test_projects/ov_capflow/test_d13n_stage0.py

- [ ] **Step 1: Write failing audit tests around explicit gates**

The report schema must contain conjunctive booleans and evidence for:

- frozen file/config/data/environment identity;
- exactly two expected missing checkpoint keys and no unexpected keys;
- exact parent tensor equality across E24, control, and candidate;
- zero adapter state, zero-RNG module construction, 257 trainable scalars;
- optimizer names exactly bbox_head.existence_residual.weight and bias;
- parent parameter and persistent-buffer hashes;
- mode enforcement and repeated-parent-feature determinism;
- all disabled mechanisms;
- full 400-image, 240,000-row three-way zero-step identity using torch.equal for scores, labels, boxes and original image/row order;
- strict Q600/open-vocabulary/rotated-mouth/forbidden-call audits;
- actual empty, ordinary, and 1,223-GT [B,3,600] target stress;
- candidate disposable one-step head-only change and control no-change;
- finite logits, residuals, gradients, and clamp-hit counts;
- no formal checkpoint or formal optimizer-step publication;
- exclusive report publication.

Unit tests may use synthetic dumps and fake models, but one integration marker must build the real configs and load E24.

Freeze these public APIs:

- compare_prediction_dumps(parent_path, control_path, candidate_path, expected_records=400, queries_per_image=600) returns per-field torch.equal evidence and the first mismatch location.
- audit_model_state(parent_model, control_model, candidate_model,
  control_optim_wrapper, candidate_optim_wrapper, checkpoint) returns exact
  load/freeze/optimizer/mode/RNG/parameter/buffer evidence. It requires the
  actual MMEngine-built optimizer wrappers, maps optimizer tensor object IDs
  back to named_parameters, and proves each optimizer contains exactly the
  two adapter tensor objects—no inference from requires_grad names alone is
  accepted.
- run_numerical_stress(control_model, candidate_model, samples) returns the empty/ordinary/1,223-GT forward/backward and disposable-step evidence.
- build_stage0_report(fingerprint, model_audit, dump_audit, protocol_audits, stress_audit, artifacts) returns the schema below and never publishes.
- publish_stage0_report(path, report) validates all gates and publishes no-replace.

The report top-level schema is exactly:

~~~text
schema=d13n-stage0-v1
status=PASS|FAIL
pre_run_commit, fingerprint, fingerprint_sha256
inputs, artifacts
gates={provenance,load,freeze,optimizer,modes,rng,parent_identity,
       zero_head,zero_update_identity,strict_q600,open_vocabulary,
       rotated_mouth,forbidden_calls,numerical_stress}
model_evidence, dump_evidence, protocol_evidence, stress_evidence
failures, report_sha256
~~~

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_stage0.py -q
~~~

Expected: FAIL because the Stage-0 auditor does not exist.

- [ ] **Step 2: Implement a composable auditor**

Expose pure comparison helpers plus a CLI. Use the atomic
D13NNoReplaceDumpResults path from d13n_test.py for each complete proxy dump in
unique Stage-0 paths, validate each dump, compare all 400 records in source
order, and then invoke frozen analyzers in noncanonical 400-image mode. Direct
tools/test.py --out is forbidden. Bind the report to every input hash and the
pre-run commit.

Invoke d13n_test.py three times through build_torchrun_argv:

1. stage0-parent on port 29845, using the control proxy config, E24, and --derive-adapter-free-parent;
2. proxy-control on port 29846, using the control proxy config and E24;
3. proxy-candidate on port 29847, using the candidate proxy config and E24.

Thus every rank reconstructs the third adapter-free parent through the exact
Task-9 deletion contract; no ephemeral in-memory config is passed across a
process boundary. Record all three d13n-test identity JSON files and resolved
config hashes. The parent must load E24 with no missing/unexpected keys;
control and candidate each load with exactly the two adapter keys missing.

The CLI must fail nonzero on any false gate and publish exactly once:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5,6,7,8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_d13n_stage0.py --control-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_control.py --candidate-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_candidate.py --parent work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --output-root .lab/workspace/exp-8-d13n-stage0 --report .lab/workspace/exp-8-d13n-stage0/stage0_report.json
~~~

--output-root and --report are required, distinct in role, and must not exist
along with any fixed child artifact. Exit 0 requires status PASS; argparse is
2; collision is 3; provenance/config/load failure is 4; any false scientific
gate is 5; child runtime failure is 6. Do not run this full command until Task
16; implement and unit-test it here.

- [ ] **Step 3: Run focused tests and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_stage0.py tests/test_projects/ov_capflow/test_d13n_runtime.py tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/audit_d13n_stage0.py tests/test_projects/ov_capflow/test_d13n_stage0.py
rtk git commit -m "feat: audit D13-N Stage-0 gates"
~~~

## Task 12: Produce endpoint assignment and existence-stratum evidence

**Files:**

- Modify: projects/OVCapFlow/ov_capflow/ov_capflow_head.py
- Create: projects/OVCapFlow/tools/d13n_endpoint_diagnostics.py
- Create: tests/test_projects/ov_capflow/test_d13n_endpoint_diagnostics.py

- [ ] **Step 1: Write failing diagnostic-assignment tests**

Require a head diagnostic method that:

- accepts the same final inference tensors and batch_data_samples as predict;
- requires the single inference group of exactly 600 rows;
- recomputes final parent cls/bbox tensors but never passes existence logits or residuals into get_targets or matcher costs;
- returns the diagnostic-only Hungarian mask, existence logits/probabilities, image ID, GT count, and category ordinary/empty/all_matched;
- produces the same assignment when the existence adapter is changed from zero to arbitrarily large finite values;
- does not alter normal prediction scores, labels, boxes, fields, or row order when diagnostics are disabled.

The aggregator must represent each category and stratum with explicit
present, count, sum, and mean fields. An absent matched stratum on empty
images and absent unmatched stratum on all-matched images is valid and must
use present=False, count=0, mean=None. It is forbidden to invent a zero or
finite mean for an absent stratum.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_endpoint_diagnostics.py -q
~~~

Expected: FAIL because neither the head method nor tool exists.

- [ ] **Step 2: Implement the read-only diagnostic path**

Add OVCapFlowHead.d13n_endpoint_assignment with explicit arguments. It calls
the unchanged head forward once, selects final one-group cls/bbox predictions,
runs get_targets under assignment-capture suppression, and separately computes
the adapter probability. It runs under torch.no_grad and has no training,
state-dict, optimizer, or ordinary-predict side effect.

d13n_endpoint_diagnostics.py builds the fixed arm config/checkpoint, iterates
the complete proxy400 dataset with annotations, gathers rank records in source
image order, requires exactly 400 unique image IDs and 240,000 rows, aggregates
global and category strata, binds config/checkpoint/prediction-dump hashes, and
publishes JSON no-replace. It runs separately for control and candidate after
their fixed endpoint dumps exist.

Freeze these pure APIs:

- classify_endpoint_category(num_gt, queries=600) returning empty for 0, all_matched for at least 600, and ordinary otherwise.
- aggregate_existence_strata(records) returning global/empty/ordinary/all_matched summaries with explicit present/count/sum/mean.
- validate_endpoint_diagnostics(report, expected_role, config, checkpoint, predictions) returning field-level failures.

Freeze this distributed CLI:

~~~text
d13n_endpoint_diagnostics.py CONFIG CHECKPOINT PREDICTIONS.pkl
  --launcher pytorch
  --role proxy-control|proxy-candidate
  --out STRATA.json
~~~

It is launched with five-process torchrun on the matching arm port and
GPU5–9/NCCL environment. Exit 0 means one complete no-replace report; argparse
is 2; collision is 3; identity/config/checkpoint/dump failure is 4; diagnostic
runtime or structural failure is 5.

The report top-level schema is exactly:

~~~text
schema=d13n-endpoint-strata-v1
role, mouth=proxy, config_path, config_sha256
checkpoint_path, checkpoint_sha256
predictions_path, predictions_sha256
records, unique_image_ids, prediction_rows, queries_per_image
matcher_uses_existence_residual=False
global={matched,unmatched}
categories={empty,ordinary,all_matched}
finite, failures, report_sha256
~~~

- [ ] **Step 3: Test fail-closed aggregation and publication**

Reject duplicate/missing image IDs, non-600 rows, nonfinite probabilities,
matcher-cost mutation, residual-dependent assignments, missing category
reports, invented absent-stratum means, checkpoint/dump hash drift, and
occupied report paths.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_endpoint_diagnostics.py tests/test_projects/ov_capflow/test_d13n_head.py -q
rtk git diff --check
~~~

- [ ] **Step 4: Commit**

Run:

~~~bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/tools/d13n_endpoint_diagnostics.py tests/test_projects/ov_capflow/test_d13n_endpoint_diagnostics.py
rtk git commit -m "feat: report D13-N endpoint existence strata"
~~~

## Task 13: Implement endpoint and conditional raw pair gates

**Files:**

- Create: projects/OVCapFlow/tools/d13n_pair_gate.py
- Modify: projects/OVCapFlow/tools/audit_strict_inference.py
- Create: tests/test_projects/ov_capflow/test_d13n_pair_gate.py
- Modify: tests/test_projects/ov_capflow/test_strict_audit.py

- [ ] **Step 1: Write failing proxy/raw gate tests**

Use exact integer gates and 1e-12 arithmetic tolerance. Proxy authorization requires:

- endpoint Epoch 12 only;
- control E12 bitwise equal to zero-step replay;
- candidate/control labels, boxes, row counts, and row order torch.equal;
- parent tensors equal, control adapter zero, candidate-only allowed delta;
- both endpoint bundles are finite, contain exactly 400 unique source-ordered image IDs and 240,000 rows at exactly 600 rows/image, and bind the expected checkpoint/config/dump hashes;
- all 12 sampler reports prove exact 1,600-row coverage, 160 synchronized updates, zero missing/padded/unregistered row indices, and the registered 108 rare-repeat exposures;
- both endpoints independently pass strict Q600, strict E2E,
  open-vocabulary, rotated-mouth, checkpoint-load, and forbidden-call audits;
- delta mAP and AP50 at least +0.0100;
- novel4 nonregression and base14 no worse than -0.0050;
- micro recall and fixed-recall oracle mAP each no worse than -0.0050;
- oracle headroom no greater than control plus 1e-12;
- 10*candidate <= 9*control for localization/background plus empty, with neither category increasing;
- 100*candidate_duplicate <= 105*control_duplicate;
- matched mean existence probability exceeds unmatched with finite ordinary/empty/all-matched reports.

Raw keep requires:

- 13,833 unique source-ordered image IDs, exactly 600 rows per image, 8,299,800 finite rows, all 18 configured classes, and filter_empty_gt=False;
- raw control scores, labels, boxes, image IDs, and row order bitwise equal work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl with SHA256 39c1d6d3193cbc9e7f8d4e6daace9fc95df86c0d0fc7f1be7870cb13ee092c45;
- control mAP within 1e-7 of 0.6064053488274416 and official AP50 exactly 0.6060;
- candidate/control labels and boxes equal;
- both raw endpoint bundles independently pass strict Q600, strict E2E, open-vocabulary, rotated-mouth, checkpoint-load, finite, and forbidden-call audits;
- delta mAP/AP50 at least +0.0050;
- base14 and novel4 each no worse than -0.0010;
- micro recall/oracle mAP no worse than -0.0050;
- oracle headroom strictly improves by more than 1e-12.

Reject rounded console-only metrics, best-epoch selection, missing category
reports, missing global matched/unmatched populations, missing hashes, or
partial single-arm results. Category-local absent strata are valid only when
encoded as present=False, count=0, mean=None for empty matched or all-matched
unmatched. Publish decisions no-replace.

The strict_e2e member is the no-replace JSON emitted by the existing
audit_strict_inference.py running the exact arm config/checkpoint. Extend that
report to schema=d13n-strict-e2e-v1 with separate strict_e2e_pass and
forbidden_calls_pass booleans, config/checkpoint paths and SHA256 values,
num_matching_queries=600, predictions_per_image all exactly 600, static and
runtime forbidden-call evidence, failures, and report_sha256. Its overall pass
is their conjunction. Change only its publication primitive to the shared
atomic O_EXCL no-replace helper. The endpoint bundle binds this report once by
path/hash, but validates strict_e2e_pass and forbidden_calls_pass as distinct
gates; tests flip either boolean independently and recompute a structurally
valid report hash so the other gate remains true.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_pair_gate.py -q
~~~

Expected: FAIL because d13n_pair_gate.py does not exist.

- [ ] **Step 2: Implement immutable bundle consumers**

Freeze the endpoint bundle schema and builder:

~~~text
schema=d13n-endpoint-bundle-v1
role=control|candidate
mouth=proxy|raw
epoch=12
config, checkpoint, predictions_identity, analyzer_manifest
protocol_audits={strict_q600,strict_e2e,open_vocabulary,rotated_mouth,
                 checkpoint_load,forbidden_calls,finite}
state_audit, sampler_audits, strata_report, official_metrics
member_sha256, bundle_sha256
~~~

build_endpoint_bundle(role, mouth, epoch, member_paths) hashes every member,
requires exactly 12 sampler audits for proxy and none for test-only raw,
requires strata_report for proxy and None for raw, and returns the bundle
without publishing. load_endpoint_bundle(path, expected_role, expected_mouth)
rechecks every member hash. strict_e2e is its own required, hash-bound audit
member for both mouths; it is not inferred from strict_q600 or any analyzer
field. evaluate_proxy_pair and evaluate_raw_pair are pure and return decision,
gates, metrics, failures. Tests independently flip strict_q600 and strict_e2e
for each raw arm and require RAW_FAIL_CLOSE, while leaving every other audit
true, so Q600 success can never mask an E2E contract failure.

Freeze these CLIs:

~~~text
d13n_pair_gate.py proxy --control-bundle CONTROL.json
  --candidate-bundle CANDIDATE.json --stage0-report STAGE0.json
  --output PROXY_GATE.json
d13n_pair_gate.py raw --control-bundle CONTROL_RAW.json
  --candidate-bundle CANDIDATE_RAW.json --stage0-report STAGE0.json
  --output RAW_GATE.json
~~~

The output schema is d13n-pair-gate-v1 with mouth, decision, fixed thresholds,
observed metrics, gates, failures, all input paths/hashes, and report_sha256.
Exit 0 means PROXY_PASS_RAW_AUTHORIZED or RAW_PASS_KEEP; argparse is 2;
collision is 3; malformed/drifted input is 4; a valid scientific fail decision
is 10.

Consume JSON bundles from the frozen analyzer, endpoint audits, sampler audits,
strict/OV/mouth/forbidden-call audits, and D13-specific identity/stratum
reports. Require analyzer source commit
abd860d157d562aa9d30422c05e41e246effc8a7 plus the three frozen tool hashes.
Do not import or edit threshold constants in the frozen analyzer. Return one
of PROXY_PASS_RAW_AUTHORIZED, PROXY_FAIL_DISCARD, RAW_PASS_KEEP, or
RAW_FAIL_CLOSE.

- [ ] **Step 3: Rerun and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_pair_gate.py tests/test_projects/ov_capflow/test_strict_audit.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/d13n_pair_gate.py projects/OVCapFlow/tools/audit_strict_inference.py tests/test_projects/ov_capflow/test_d13n_pair_gate.py tests/test_projects/ov_capflow/test_strict_audit.py
rtk git commit -m "feat: gate D13-N proxy and raw pairs"
~~~

## Task 14: Implement the sequential owner

**Files:**

- Create: projects/OVCapFlow/tools/d13n_run_owner.py
- Create: tests/test_projects/ov_capflow/test_d13n_run_owner.py

- [ ] **Step 1: Write failing state-machine tests**

Freeze this state machine:

~~~text
idle/recheck
  -> control train on GPU5-9 port29842
  -> control E12 atomic proxy dump on GPU5-9 port29842
  -> control validator/audits/full zero-step bitwise equality
  -> idle/recheck
  -> candidate train on GPU5-9 port29841
  -> paired proxy dumps/analyzers/gate
  -> if proxy passes: idle/recheck
  -> raw control replay port29843
  -> idle/recheck
  -> raw candidate replay port29844
  -> raw gate
  -> terminal preserved outcome
~~~

Tests require two consecutive idle polls, exact config/commit/fingerprint/hash rechecks before each transition, unique artifacts and tmux names, one rtk wrapper plus one real launcher plus five workers, rank environments 0..4, correct CUDA_VISIBLE_DEVICES=5,6,7,8,9 and NCCL flags, fixed ports, O_EXCL release barrier, and no candidate launch after any control failure/drift.

The monitor and owner share OwnerPaths.integrity_failure_lock. At the start of
every poll, before and after every child process, before constructing any next
arm argv, before every state transition, immediately before publishing the
candidate/raw release barrier, and immediately before subprocess launch, the
owner calls load_integrity_failure_lock. Any valid lock forces TERMINAL with
INTEGRITY_FAIL_CLOSE and exit 20; malformed or hash-drifted lock content forces
the infrastructure terminal with exit 21. No later-arm command may be built,
published, or launched. Parameterized tests prepublish each monitor failure
class—identity, topology, nonfinite/fatal, collision, checkpoint parent,
checkpoint buffer, optimizer binding, control head, sampler, and GPU
contract—at every release boundary and prove candidate, raw-control, and
raw-candidate argv builders and launch fakes are never called.

Before the first control argv construction and at all those same boundaries,
the owner also requires a valid shared monitor.ready report, the exact Python
monitor PID/start-tick/cmdline still live, and a valid hash-chained heartbeat
no older than 150 seconds. It may wait at most 180 seconds from owner startup
for ready plus the first sample, without constructing a formal command. A
missing/stale/dead/reused/wrong-identity monitor makes the owner call the
shared publish_integrity_failure_lock with failure_class=input_runtime, then
enter TERMINAL. Tests cover monitor never started, pre-ready crash, abrupt
exit without a lock, stale heartbeat, truncated heartbeat, and PID reuse at
control/candidate/raw boundaries; all later-arm builders and launch fakes must
remain uncalled.

Before the candidate release barrier, tests must require the complete control
Epoch-12 proxy dump, exactly 400 unique image IDs and 240,000 finite rows, and
torch.equal for every score, label, rotated box, image ID, and row position
against the Stage-0 zero-step control replay. They must also require all
control endpoint strict/OV/mouth/forbidden-call, checkpoint, parent/buffer,
sampler, optimizer-name, and zero-head gates. Flip each conjunct separately
and prove the candidate command is never constructed or launched.

Checkpoint/state evidence is produced by the shared Task-10
audit_d13n_checkpoint API. Endpoint bundle construction uses the Task-13
build_endpoint_bundle API; the owner does not invent a second schema or hash
implementation.

Also prove no kill, retry, resume, deletion, replacement, coefficient change, batch change, epoch extension, or candidate-only continuation path exists.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_run_owner.py -q
~~~

Expected: FAIL because owner does not exist.

- [ ] **Step 2: Implement commands as explicit argv**

Freeze these public interfaces:

- OwnerState enum values WAIT_CONTROL, TRAIN_CONTROL, AUDIT_CONTROL,
  WAIT_CANDIDATE, TRAIN_CANDIDATE, GATE_PROXY, WAIT_RAW_CONTROL,
  RAW_CONTROL, WAIT_RAW_CANDIDATE, RAW_CANDIDATE, GATE_RAW, TERMINAL.
- OwnerPaths.from_root(root) returning fixed unique paths for owner.jsonl,
  terminal.json, monitor.ready, monitor.jsonl, integrity_failure.lock, candidate.release,
  raw_control.release, raw_candidate.release, each console/workdir/checkpoint/dump
  official-metrics/identity/analyzer/protocol/state/strata/bundle/gate
  artifact.
- build_training_argv(role), build_test_argv(role, checkpoint, out,
  metrics_out, identity_out), build_diagnostics_argv(role, checkpoint,
  predictions, out), build_strict_e2e_argv(role, config, checkpoint, out),
  build_analyzer_argv(mouth, config, checkpoint, predictions,
  official_metrics, output_dir), and
  advance_owner_state(state,evidence) as pure functions.
- append_owner_event(path,event) creating the JSONL with O_EXCL on its first
  line and using append-only writes thereafter; publish_terminal(path,payload)
  is no-replace.
- the Task-10 shared monitor-ready, heartbeat, and integrity-lock functions;
  a present lock is monotonic and is never removed, replaced, or acknowledged
  away.

Freeze this CLI:

~~~text
d13n_run_owner.py --stage0-report STAGE0.json
  --pre-run-commit COMMIT --output-root ROOT
  --poll-seconds 60 --idle-polls 2
~~~

poll-seconds and idle-polls reject every value except 60 and 2. ROOT and every
OwnerPaths child must be unoccupied at startup. Exit 0 means RAW_PASS_KEEP; 10
means a valid PROXY_FAIL_DISCARD or RAW_FAIL_CLOSE; 20 means an integrity
closure at any formal stage; 21 means a fail-closed infrastructure/runtime
error.

Each JSONL event uses schema=d13n-owner-event-v1, sequence, timestamp, state,
event, evidence_paths_sha256, child_argv, child_pid, topology, and decision.
terminal.json uses schema=d13n-owner-terminal-v1, final_state, decision,
stage0/report/commit hashes, all event/artifact hashes, failures, and
report_sha256.

Training uses:

~~~text
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5,6,7,8,9
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1
PYTHONPATH=/data1/zcy/OV-CapFlow
/data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run
--nproc_per_node=5 --master_port=29842 tools/train.py
configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_d13n_control.py
--launcher pytorch
~~~

Candidate training substitutes port 29841 and the candidate training config.
Every proxy or raw dump uses five-process torchrun around d13n_test.py, never
tools/test.py --out. Control/candidate proxy replay use their training configs
and ports 29842/29841. Raw control/candidate use the explicit raw13833 configs,
their exact Epoch-12 checkpoints, and ports 29843/29844. Every evaluation argv
retains CUDA_VISIBLE_DEVICES=5,6,7,8,9, NCCL_P2P_DISABLE=1,
NCCL_IB_DISABLE=1, --nproc_per_node=5, and --launcher pytorch. Build subprocess
argv without shell interpolation. Record every transition in exclusive JSONL
and publish terminal state no-replace.

The integrity-failure lock is an asynchronous terminal input to the state
machine, not monitor-only telemetry. The owner validates it on all boundaries
listed in Step 1 and records its path, SHA256, monitor sequence, failure class,
and evidence hashes in terminal.json. The candidate and raw release files are
O_EXCL monotonic barriers and are never created after a failure lock exists.

Child execution uses Popen plus a fixed 60-second guard loop rather than a
single blocking run. While any train, dump, diagnostic, audit, or gate child
is alive, the owner revalidates the failure lock, monitor PID identity, and
heartbeat freshness every poll. On fail-silent monitor detection it publishes
the shared input_runtime lock and latches the terminal decision, but does not
kill or restart the already-running child; it waits for that child to exit and
then publishes terminal.json without constructing any later command.

After each successful d13n_test invocation, pass its exact
OFFICIAL_METRICS.json to analyze_dotav2_q600_dump.py through
--official-metrics-json. Proxy analyzer argv additionally freezes
--allow-noncanonical --expected-records 400 --queries-per-image 600
--num-classes 18. Raw candidate/control diagnostics also use
--allow-noncanonical with explicit --expected-records 13833
--queries-per-image 600 --num-classes 18, because the frozen analyzer's
canonical=True branch is intentionally tied to the T7 Epoch-24 training-metric
replay and would invalidly compare a D13-N Epoch-12 endpoint to the T7 E24
training metric. This analyzer-mode label does not weaken the raw mouth:
separate endpoint gates still require the exact raw dataset/config, 13,833
unique images, canonical-control bitwise equality, strict Q600/E2E/OV audits,
and official same-dump parity. The analyzer output directory is
unique/no-replace, and its manifest plus the official metrics path/hash are
mandatory members of build_endpoint_bundle.

For each proxy and raw arm, build_strict_e2e_argv runs
projects/OVCapFlow/tools/audit_strict_inference.py with the exact resolved
config, exact endpoint checkpoint, --device cuda, and that arm's unique
strict_e2e.json. It uses CUDA_VISIBLE_DEVICES=5 only for this one-batch audit;
physical GPUs0–4 remain hidden. The owner requires the report's independent
strict_e2e_pass and forbidden_calls_pass gates before bundle construction.

- [ ] **Step 3: Rerun and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_run_owner.py tests/test_projects/ov_capflow/test_d13n_runtime.py tests/test_projects/ov_capflow/test_d13n_pair_gate.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/d13n_run_owner.py tests/test_projects/ov_capflow/test_d13n_run_owner.py
rtk git commit -m "feat: own sequential D13-N experiment pair"
~~~

## Task 15: Implement the 60-second monitor and checkpoint integrity audit

**Files:**

- Create: projects/OVCapFlow/tools/monitor_d13n_run.py
- Create: tests/test_projects/ov_capflow/test_d13n_monitor.py

- [ ] **Step 1: Write failing telemetry tests**

Require one sample schema containing:

- timestamp and owner state;
- physical GPU5–9 memory, utilization, UUID, and compute PIDs;
- wrapper/launcher/five-worker topology, rank environments, and port;
- console size/growth plus epoch/iteration/LR/parent losses/existence loss/gradient norm/data time/iteration time;
- per-group matched counts and logit/residual/clamp quantiles;
- epoch-qualified sampler report/checksum;
- checkpoint discovery with no deletion;
- every parent parameter and persistent-buffer hash against Stage 0;
- optimizer parameter names exactly the two adapter names;
- control adapter all-zero hash;
- fatal-pattern status.

Unchanged state is not an error. Any identity, topology, nonfinite, OOM, NCCL, collision, or control-integrity failure is terminal evidence and must not trigger a restart.

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_monitor.py -q
~~~

Expected: FAIL because monitor does not exist.

- [ ] **Step 2: Implement one-shot sampling plus recurring CLI**

Freeze these public functions:

- parse_training_log_delta(previous_offset, console_path) returns the new
  offset, parsed epoch/iter/LR/loss/time/telemetry fields, growth bytes, and
  fatal matches.
- build_monitor_sample(owner_event, gpu_records, process_records, log_delta,
  sampler_report, checkpoint_audit) returns one schema-valid sample.
- the Task-10 shared publish_monitor_ready and
  publish_integrity_failure_lock functions. The monitor publishes ready only
  after validating all inputs and exclusively creating its JSONL, and can
  still close the owner before the first valid sample exists.

The sample schema is d13n-monitor-sample-v1 with sequence, timestamp,
monotonic_ns, monitor_pid, proc_start_ticks, owner_event_sha256,
previous_sample_sha256, sample_sha256,
owner_state, gpu, topology, console, training, d13n_telemetry, sampler,
checkpoint, fatal_patterns, status. A terminal summary uses
d13n-monitor-terminal-v1 with owner terminal identity, sample count, last
sequence, failures, and report_sha256.

The monitor uses the Task-10 terminal-lock schema and failure-class registry
unchanged; it does not define a private variant.

Freeze this CLI:

~~~text
monitor_d13n_run.py --owner-log ROOT/owner.jsonl
  --owner-root ROOT --stage0-report STAGE0.json
  --pre-run-commit COMMIT --output ROOT/monitor.jsonl
  --ready-output ROOT/monitor.ready
  --terminal-output ROOT/monitor_terminal.json
  --failure-lock ROOT/integrity_failure.lock --poll-seconds 60
~~~

poll-seconds rejects every value except 60. Ready, JSONL, terminal, and failure
lock paths are fixed OwnerPaths values; all four must be
unoccupied;
the JSONL is created O_EXCL and then append-only. After JSONL creation and
input validation, publish monitor.ready O_EXCL with the real Python PID,
/proc start ticks and exact cmdline hash, bound root/log/Stage-0/commit paths,
then append and fsync the first hash-chained heartbeat immediately rather than
waiting 60 seconds. Each later sample is one newline-terminated JSON record
and is fsynced before the interval starts. Exit 0 follows an owner
terminal event with no monitor failure; 20 is an integrity failure; 21 is an
input/runtime failure. For checkpoint evidence it calls the shared Task-10
audit_d13n_checkpoint and cache rather than reimplementing hashes. The CLI
exits only at owner terminal state or terminal integrity failure. Do not use a
single long blocking sleep in the agent; the detached monitor process owns the
interval.

For every integrity or post-start input/runtime failure, publish and fsync the
failure lock before publishing monitor_terminal.json and before exiting. The
owner treats the lock as authoritative even if monitor_terminal.json is
missing or the monitor process has already exited. Tests inject every failure
class, assert the lock is the first terminal artifact, and then drive the real
Task-14 state transition to prove no candidate or raw command is constructed.

- [ ] **Step 3: Rerun and commit**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d13n_monitor.py tests/test_projects/ov_capflow/test_d13n_run_owner.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/monitor_d13n_run.py tests/test_projects/ov_capflow/test_d13n_monitor.py
rtk git commit -m "feat: monitor D13-N integrity and progress"
~~~

## Task 16: Run the full verification and real Stage 0

**Files:**

- Update ignored evidence only after successful commands: .lab/log.md
- Update ignored evidence only after successful commands: .lab/results.tsv
- Publish no-replace runtime artifacts under: .lab/workspace/exp-8-d13n-stage0/

- [ ] **Step 1: Run focused D13-N tests**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_existence_residual.py tests/test_projects/ov_capflow/test_calibration.py tests/test_projects/ov_capflow/test_d13n_head.py tests/test_projects/ov_capflow/test_d13n_freeze_mode.py tests/test_projects/ov_capflow/test_d13n_world5_configs.py tests/test_projects/ov_capflow/test_no_replace.py tests/test_projects/ov_capflow/test_d13n_dump_results.py tests/test_projects/ov_capflow/test_d13n_test.py tests/test_projects/ov_capflow/test_d13n_runtime.py tests/test_projects/ov_capflow/test_d13n_stage0.py tests/test_projects/ov_capflow/test_d13n_endpoint_diagnostics.py tests/test_projects/ov_capflow/test_d13n_pair_gate.py tests/test_projects/ov_capflow/test_d13n_run_owner.py tests/test_projects/ov_capflow/test_d13n_monitor.py tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py -q
~~~

Expected: PASS.

- [ ] **Step 2: Run protected regressions**

Run:

~~~bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_d12_world5_configs.py tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py tests/test_projects/ov_capflow/test_sampler_coverage_audit.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_d12_checkpoint_pair.py tests/test_projects/ov_capflow/test_strict_audit.py tests/test_projects/ov_capflow/test_open_vocabulary_audit.py tests/test_projects/ov_capflow/test_null_reservoir.py tests/test_projects/ov_capflow/test_grouped_queries.py tests/test_projects/ov_capflow/test_freeze_except_hook.py -q
~~~

Expected: PASS.

- [ ] **Step 3: Run syntax, hash, diff, and resolved-config checks**

Run:

~~~bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m py_compile projects/OVCapFlow/ov_capflow/existence_residual.py projects/OVCapFlow/ov_capflow/d13n_mode_hook.py projects/OVCapFlow/ov_capflow/no_replace.py projects/OVCapFlow/ov_capflow/d13n_dump_results.py projects/OVCapFlow/tools/d13n_runtime.py projects/OVCapFlow/tools/d13n_test.py projects/OVCapFlow/tools/audit_d13n_stage0.py projects/OVCapFlow/tools/d13n_endpoint_diagnostics.py projects/OVCapFlow/tools/d13n_pair_gate.py projects/OVCapFlow/tools/d13n_run_owner.py projects/OVCapFlow/tools/monitor_d13n_run.py
rtk sha256sum projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py projects/OVCapFlow/tools/dotav2_q600_diagnostics.py projects/OVCapFlow/tools/validate_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl
rtk git diff --check
rtk git status --short --untracked-files=all
~~~

Expected: frozen hashes match; only known user-owned unrelated untracked files remain.

- [ ] **Step 4: Commit the verified pre-run source state**

Run:

~~~bash
rtk git status --short
rtk git commit --allow-empty -m "chore: freeze verified D13-N pre-run state"
rtk git rev-parse HEAD
~~~

Record this exact commit in Stage 0.

- [ ] **Step 5: Recheck resources without taking them**

Run:

~~~bash
rtk nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
rtk nvidia-smi pmon -c 1
rtk tmux list-sessions
rtk ps -eo pid,ppid,lstart,cmd
rtk ss -ltnp
rtk df -h /data1/zcy/OV-CapFlow
~~~

Expected: evidence captured; no process is killed and GPUs0–4 are not used by D13-N.

- [ ] **Step 6: Run real Stage 0 on GPU5–9**

Require physical GPUs5–9 to be idle for two consecutive polls before starting
Stage 0. If any of them is occupied, leave all jobs untouched and keep
monitoring; do not spill work onto GPUs0–4. Once idle, run the exact Task-11
Stage-0 CLI
with NCCL flags and a fresh no-replace output root. It must build/load real
models, execute actual empty/ordinary/1,223-GT stress, generate the complete
three-way 400-image zero-step dumps, and prove every conjunctive gate.

Expected: PASS report with a stable hash. If any gate fails, preserve evidence, stop before formal launch, and diagnose under systematic-debugging.

- [ ] **Step 7: Independent Stage-0 review**

Run a fresh specification reviewer and a fresh code-quality reviewer over the final code, test evidence, Stage-0 report, and frozen hashes. Any finding returns to RED→GREEN and requires a new pre-run commit plus a fresh Stage-0 output root.

## Task 17: Launch only the approved sequential formal pair

**Files:**

- Runtime evidence only: .lab/workspace/exp-8-d13n-formal/
- Update: .lab/log.md
- Update: .lab/results.tsv

- [ ] **Step 1: Start owner and monitor in unique persistent tmux sessions**

Launch d13n_run_owner.py and monitor_d13n_run.py with the Stage-0 report/hash, pre-run commit, unique output root, and 60-second telemetry. The owner waits for two idle polls and uses physical GPUs5–9 only.
Pass the monitor the exact OwnerPaths.integrity_failure_lock path and verify
the owner consumes that same path before any release or launch.
Start the owner in WAIT_CONTROL, then start the monitor within the frozen
180-second startup window with the exact owner root, pre-run commit,
monitor.ready path, and JSONL path. Control cannot be constructed until the
ready PID/start-tick/cmdline identity and first hash-chained heartbeat validate.

- [ ] **Step 2: Verify the live control topology**

Require one rtk wrapper ancestor, one real torch distributed launcher, five workers with LOCAL_RANK 0..4, WORLD_SIZE=5, LOCAL_WORLD_SIZE=5, CUDA_VISIBLE_DEVICES=5,6,7,8,9, NCCL flags equal 1, and MASTER_PORT=29842. Verify console growth and no fatal pattern.
Also verify the bound Python monitor PID is live with the same /proc start
ticks and that its latest monotonic heartbeat is at most 150 seconds old.

- [ ] **Step 3: Monitor control through fixed Epoch 12**

At every checkpoint verify parent parameters/buffers unchanged, optimizer contains exactly the two head tensors, control head remains bitwise zero, sampler exact cover holds, and all checkpoints remain present. Any failure closes the pair without candidate.
The monitor must publish the no-replace integrity lock before exiting; the
owner must record that lock in its terminal evidence and must not publish a
candidate/raw release artifact.

- [ ] **Step 4: Allow the owner to recheck and launch candidate**

After control training exits, create its complete proxy dump atomically with
five-process d13n_test.py, validate 400 unique image IDs and 240,000 finite
rows, run endpoint audits, and prove every score, label, rotated box, image ID,
and row position torch.equal to the frozen Stage-0 zero-step control replay.
Only after this and all strict/OV/mouth/forbidden-call, sampler, checkpoint,
parent/buffer, optimizer-name, and zero-head gates pass, followed by two fresh
GPU-idle polls, may the O_EXCL barrier release. Recheck the complete Stage-0
fingerprint, commit, configs, ports, paths, hashes, and topology before
candidate on port 29841.

- [ ] **Step 5: Adjudicate the fixed proxy endpoint**

Reuse the already frozen control endpoint dump; generate the complete
candidate Epoch-12 proxy dump atomically. Run
d13n_endpoint_diagnostics.py for both checkpoints so ordinary, empty, and
all-matched reports explicitly encode present and absent strata. Generate
frozen analyzer bundles, complete endpoint audits, identity/strata evidence,
and d13n_pair_gate.py output. If proxy fails, record PROXY_FAIL_DISCARD and
close D13-N without tuning.

- [ ] **Step 6: Conditionally run raw control then raw candidate**

Only PROXY_PASS_RAW_AUTHORIZED permits test-only raw replays. Run the
raw13833 control config and its exact control Epoch-12 checkpoint first, then
the raw13833 candidate config and its exact candidate Epoch-12 checkpoint,
sequentially through five-process d13n_test.py on physical GPUs5–9 with ports
29843 then 29844 and fresh preflight before each. Require exact CUDA/NCCL/rank
topology, 13,833 unique source-ordered image IDs, 600 rows per image,
8,299,800 finite rows, all 18 classes, and filter_empty_gt=False.

- [ ] **Step 7: Record the scientific decision**

Publish RAW_PASS_KEEP or RAW_FAIL_CLOSE no-replace. Update the claim-evidence ledger with exact mAP, AP50, base14, novel4, per-class, recall, oracle, FP-region, empty, duplicate, and existence-strata results. A pass supports only the bounded calibration claim; it does not establish the final ICLR innovation or the 0.7000 engineering goal.

## Final handoff checklist

- [ ] Every task was implemented by a fresh implementer, then reviewed by a fresh spec reviewer and a fresh quality reviewer before the next task.
- [ ] Every production change was preceded by an observed failing test.
- [ ] No formal compute used physical GPUs0–4.
- [ ] No D12 session/monitor/artifact was stopped, reused, or modified.
- [ ] No frozen analyzer was modified.
- [ ] No formal artifact was overwritten, deleted, resumed, or auto-retried.
- [ ] Every monitor terminal failure was latched into the owner through the
      immutable integrity-failure lock before any later-arm release.
- [ ] Monitor readiness, PID start identity, and heartbeat freshness were
      required before control and every later-arm launch; fail-silent monitor
      death was latched by the owner itself.
- [ ] Control preceded candidate and remained bitwise zero.
- [ ] Proxy evidence was not reported as raw promotion evidence.
- [ ] Raw promotion used the complete 13,833-image strict Q600 mouth.
- [ ] All claims are marked known, inferred, or unknown and stay within the frozen claim boundary.
