# OV-CapFlow Unified Query Allocation–Quality Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one shared, fixed-Q query-to-image coupling to OV-CapFlow so the same non-null mass drives bounded center transport, final-layer matching/IoU validity training, and suppressive all-row score calibration, while preserving exact T7 behavior at zero route and stopping after one registered proxy pair if the scientific gates fail.

**Architecture:** One new production file owns the parameter-shared float32 coupling, its stateless Hungarian validity cost, and the small frozen-parent mode hook. Existing detector preprocessing supplies one pre-registered encoder level and frozen text evidence; the existing decoder loop applies center-only transport to matching rows while leaving DN untouched; the existing head consumes the same final validity for final-layer matching, a natural-prevalence soft-IoU loss, and post-argmax log-score calibration. Two thin world-5 configs define a zero-route control and one candidate. Existing launch, evaluator, dump, diagnostics, and strict-mouth tools remain authoritative.

**Tech Stack:** Python 3.8.19, PyTorch 1.12.1+cu113, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate 1.0.0rc1, CUDA/NCCL with P2P and IB disabled, pytest, and ten A40 GPUs split into two simultaneous five-rank jobs.

---

## Non-negotiable prerequisite

Do not execute any task in this plan unless all three checks succeed:

```bash
rtk run "test -f .lab/workspace/exp-8-qaf-source-v1/source_report.json"
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_query_evidence_source.py --verify-only --output .lab/workspace/exp-8-qaf-source-v1/source_report.json
rtk sha256sum .lab/workspace/exp-8-qaf-source-v1/source_report.json
```

Expected: `--verify-only` reports `status: PASS`, all source gates true, and exit code `0`. The exact report SHA256 produced by the third command must be committed in the governance row; both experiment configs must resolve that same digest from governance and embed it in their resolved configs. A missing report, FAIL report, hash mismatch, or unverifiable report closes this plan without production edits.

## Frozen production recipe

| Field | Frozen value |
|---|---|
| Parent | T7 E24 SHA `a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8` |
| Frozen dump analyzer | `projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py`, SHA `f4f10734948b714dcbae726142e859b3ddac793ee5ab5c78d9a3f3436d99fa23` |
| Frozen diagnostics core | `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`, SHA `bd29dc88650e0db545479c60adaa6a9ee64da1734d3a1756c7cd787c7269c92d` |
| Query contract | Q600 inference; 3 x 600 matching suffix plus parent DN during training |
| Encoder source | level index 2, stride 32, valid tokens only |
| Shared latent dimension | 32 |
| Solver | float32 log-domain row-normalized soft coupling, 5 fixed column-reweight iterations |
| Entropy temperature | `0.1` |
| Soft-column KL strength | `1.0` |
| Spatial prior share | `0.75` |
| Null prior share | `0.25` |
| Content / spatial / semantic weights | `1.0 / 1.0 / 1.0` |
| Oriented spatial cost cap | `4.0` |
| Null-logit bound | `4.0`, applied as `4*tanh(raw_null/4)` |
| Minimum reference extent | `0.01` normalized |
| Angle factor | `3.141592653589793`; stored reference angle is multiplied by this before rotation |
| Maximum center step | L2 radius `0.05` normalized per layer |
| Validity epsilon | `1e-6` |
| Center and score gates | deterministic `target_strength * flow_progress`; the checkpointed progress buffer starts at `0.0` and reaches `1.0` at update 160 |
| Validity target | final-layer detached aligned rotated IoU for matched queries, zero for unmatched |
| Validity loss | binary cross entropy with soft targets, natural mean over all B x 3 x 600 matching rows |
| Matching validity cost | final layer only, `-lambda_match * log(v.clamp(min=1e-6, max=1.0))`, constant across GT columns |
| Matching warm-up | linear from `0.0` to `0.5` over optimizer updates 0 through 160; frozen at `0.5` thereafter |
| Control weights | center target `0.0`, score target `0.0`, validity loss `0.0`, match `0.0` |
| Candidate weights | center target `1.0`, score target `1.0`, validity loss `1.0`, match target `0.5` |
| Optimizer | AdamW, LR `1e-4`, weight decay `1e-4`, gradient norm clip `0.1` |
| Pair endpoint | 12 epochs x 160 updates = 1,920 optimizer updates per arm |
| Seed | `20260716`, same data order and no rank-dependent seed |
| Topology | control GPUs 0-4 / port 29910; candidate GPUs 5-9 / port 29911; 5 ranks, batch 2 per rank |

The coupling is one object. The initial implementation must not add a density/count head, existence classifier, extent/angle head, extra feature level, token top-k, dynamic Q, proposal list, dense detector, RPN, RoI head, threshold, sort, pruning, or NMS.

## Exact coupling definition

For matching query states `h` and registered encoder tokens `x`, use deterministic bias-free projections `Wq,Wk: 256 -> 32`. Initialize the first 32 diagonal entries to one and all remaining entries to zero. Initialize the learned 32-D null key and null bias to zero. Register `flow_progress` as a persistent scalar buffer initialized to zero; construction must not consume RNG. Fixed target strengths are config values, not fake trainable parameters.

For each query/token pair, compute in float32:

```python
content = cosine(normalize(Wq(h)), normalize(Wk(x)))
dxdy = token_center - reference_center
theta = reference_angle * 3.141592653589793
dxdy_local = rotate(dxdy, -theta)
spatial_cost = clamp(max(2*abs(dx_local)/clamp(width, min=0.01),
                         2*abs(dy_local)/clamp(height, min=0.01)), max=4.0)
token_logit = content - spatial_cost + semantic_evidence
raw_null = dot(normalize(Wq(h)), null_key) + null_bias
null_logit = 4.0 * tanh(raw_null / 4.0)
```

Construct a bounded column prior with 75% spatial mass proportional to `clamp(sigmoid(semantic_evidence), 0.01, 0.99)` over valid tokens and exactly 25% null mass. Invalid spatial columns receive zero prior and `-inf` logits.

Run exactly five deterministic soft-column iterations:

```python
column_bias = zeros_like(column_prior)
gamma = 1.0 / (1.0 + 0.1)
for _ in range(5):
    log_pi = log_softmax(concat(token_logit, null_logit) / 0.1
                         + column_bias[:, None, :], dim=-1)
    column_mass = exp(log_pi).sum(dim=1) / num_queries
    column_bias = column_bias + gamma * (
        log(column_prior + 1e-6) - log(column_mass + 1e-6))
log_pi = log_softmax(concat(token_logit, null_logit) / 0.1
                     + column_bias[:, None, :], dim=-1)
pi = exp(log_pi)
```

Every row must sum to one within `1e-5`. Then

```python
validity = 1.0 - pi[..., -1]
barycenter = (pi[..., :-1, None] * token_centers[:, None]).sum(-2) \
             / (validity[..., None] + 1e-6)
direction = barycenter - parent_reference[..., :2]
bounded_direction = direction * minimum(1, 0.05 / (norm(direction) + 1e-6))
center_delta = validity[..., None] * bounded_direction
```

All finite checks happen before values are returned. No `nan_to_num` repair is allowed.

## Scope and commit accounting

This plan uses concentrated commits 2 and 3. Together with the source-plan commit, that is the complete pre-proxy budget.

Files changed by this plan:

- Create: `projects/OVCapFlow/ov_capflow/query_evidence_coupling.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_layers.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py`
- Create: `tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py`
- Modify after observed decisions: `.lab/results.tsv`, `progress.md`, `task_plan.md`, `projects/OVCapFlow/README.md`

Across both plans there are exactly two new test files and two new configs. Do not add a third test/config or a new launcher/analyzer.

## Task 1: Implement and prove the shared coupling core

**Files:**

- Create: `projects/OVCapFlow/ov_capflow/query_evidence_coupling.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Create: `tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py`
- Modify: `.lab/results.tsv`, `progress.md`, `task_plan.md`, `projects/OVCapFlow/README.md`

- [ ] **Step 1: Add RED tests for the public production API**

The test imports these exact public names:

```python
from projects.OVCapFlow.ov_capflow.query_evidence_coupling import (
    CouplingOutput,
    QueryEvidenceCoupling,
    QueryFlowParentEvalHook,
    QueryValidityCost,
    aligned_rotated_iou_validity_targets,
    natural_validity_loss,
)
```

Test the core with fixed tensors and no random fixtures:

```python
def build_coupling():
    return QueryEvidenceCoupling(
        embed_dims=256, latent_dims=32, solver_iterations=5,
        entropy=0.1, column_relaxation=1.0,
        spatial_prior_share=0.75, null_prior_share=0.25,
        content_weight=1.0, spatial_weight=1.0,
        semantic_weight=1.0, spatial_cost_cap=4.0,
        min_reference_extent=0.01, max_center_delta=0.05,
        eps=1e-6)


def fixed_inputs(dtype=torch.float32):
    queries = torch.arange(2 * 3 * 256, dtype=torch.float32).reshape(
        2, 3, 256) / 1000
    memory = torch.arange(2 * 4 * 256, dtype=torch.float32).reshape(
        2, 4, 256).flip(-1) / 1000
    references = torch.tensor(
        [[[0.25, 0.25, 0.20, 0.10, 0.00],
          [0.50, 0.50, 0.30, 0.20, 0.25],
          [0.75, 0.75, 0.10, 0.30, 0.50]]] * 2)
    evidence = torch.tensor([[0.2, 0.4, -0.1, 0.3],
                             [0.1, -0.2, 0.5, 0.0]])
    centers = torch.tensor(
        [[[0.125, 0.125], [0.375, 0.375],
          [0.625, 0.625], [0.875, 0.875]]] * 2)
    valid_mask = torch.tensor([[True, True, False, True],
                               [True, True, True, True]])
    floating = (queries, references, memory, evidence, centers)
    return tuple(value.to(dtype=dtype) for value in floating) + (valid_mask,)


def test_constructor_is_rng_free_and_exactly_initialized():
    before = torch.random.get_rng_state().clone()
    module = build_coupling()
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert module.trainable_parameter_count == 16417
    assert module.center_strength.item() == 0.0
    assert module.score_strength.item() == 0.0
    module.set_update(160)
    assert module.center_strength.item() == 1.0
    assert module.score_strength.item() == 1.0


def test_rows_include_null_and_sum_to_one():
    queries, references, memory, evidence, centers, valid_mask = fixed_inputs()
    out = build_coupling()(queries, references, memory, evidence,
                           centers, valid_mask)
    assert isinstance(out, CouplingOutput)
    assert out.coupling.shape == (2, 3, 5)
    assert torch.allclose(out.coupling.sum(-1), torch.ones(2, 3),
                          atol=1e-5, rtol=0)
    assert torch.equal(out.validity, 1 - out.coupling[..., -1])


def test_invalid_tokens_receive_exact_zero_mass():
    queries, references, memory, evidence, centers, valid_mask = fixed_inputs()
    out = build_coupling()(queries, references, memory, evidence,
                           centers, valid_mask)
    assert torch.count_nonzero(out.coupling[..., 2]) == 0


def test_center_delta_is_validity_scaled_and_l2_bounded():
    queries, references, memory, evidence, centers, valid_mask = fixed_inputs()
    out = build_coupling()(queries, references, memory, evidence,
                           centers, valid_mask)
    assert torch.all(torch.linalg.vector_norm(out.center_delta, dim=-1)
                     <= 0.050001)
    assert torch.count_nonzero(out.center_delta[out.validity == 0]) == 0


def test_zero_strength_returns_parent_object_without_arithmetic():
    module = build_coupling()
    parent = torch.zeros(2, 3, 5)
    transported, applied = module.apply_center_transport(
        parent, torch.ones(2, 3, 2))
    assert transported is parent
    assert torch.count_nonzero(applied) == 0


def test_validity_cost_is_constant_over_gt_columns():
    pred = InstanceData(flow_match_cost=torch.tensor([0.1, 0.2]))
    gt = InstanceData(labels=torch.tensor([0, 1, 2]))
    cost = QueryValidityCost()(pred, gt)
    assert torch.equal(cost, torch.tensor([[0.1, 0.1, 0.1],
                                           [0.2, 0.2, 0.2]]))


def test_natural_validity_loss_uses_all_rows_without_rebalancing():
    validity = torch.tensor([0.8, 0.2, 0.1])
    target = torch.tensor([1.0, 0.5, 0.0])
    expected = F.binary_cross_entropy(validity, target, reduction='mean')
    assert torch.equal(natural_validity_loss(validity, target, 1e-6),
                       expected)
```

Append this exact block for the remaining core contracts:

```python
def test_empty_valid_mask_is_rejected():
    values = list(fixed_inputs())
    values[-1] = torch.zeros_like(values[-1])
    with pytest.raises(ValueError, match='valid spatial token'):
        build_coupling()(*values)


def test_all_null_case_is_finite():
    queries, references, memory, evidence, centers, valid = fixed_inputs()
    out = build_coupling()(queries, references, memory,
                           torch.full_like(evidence, -1000), centers, valid)
    assert torch.isfinite(out.coupling).all()
    assert torch.all(out.validity < 1e-4)


def test_null_logits_are_strictly_bounded():
    module = build_coupling()
    module.null_key.data.fill_(1e6)
    module.null_bias.data.fill_(1e6)
    queries = fixed_inputs()[0]
    bounded = module._bounded_null_logits(queries)
    assert torch.all(bounded <= 4.0)
    assert torch.all(bounded >= -4.0)


def test_zero_queries_are_rejected():
    queries, references, memory, evidence, centers, valid = fixed_inputs()
    with pytest.raises(ValueError, match='query count'):
        build_coupling()(queries[:, :0], references[:, :0], memory,
                         evidence, centers, valid)


def test_solver_is_bitwise_repeatable():
    module = build_coupling()
    inputs = fixed_inputs()
    first = module(*inputs)
    second = module(*inputs)
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_half_inputs_produce_float32_coupling():
    out = build_coupling()(*fixed_inputs(torch.float16))
    assert out.coupling.dtype == torch.float32
    assert out.validity.dtype == torch.float32
    assert out.center_delta.dtype == torch.float32


def test_nonfinite_inputs_fail_closed():
    values = list(fixed_inputs())
    values[0] = values[0].clone()
    values[0][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        build_coupling()(*values)


def test_all_16417_trainable_gradients_are_finite():
    module = build_coupling()
    module.set_update(160)
    out = module(*fixed_inputs())
    (out.validity.mean() + out.center_delta.square().mean()).backward()
    parameters = list(module.parameters())
    assert sum(parameter.numel() for parameter in parameters) == 16417
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_aligned_rotated_iou_targets_cover_matched_and_unmatched():
    predicted = torch.tensor([[[0., 0., 2., 2., 0.],
                               [4., 4., 2., 2., 0.],
                               [8., 8., 2., 2., 0.]]])
    targets = torch.tensor([[[0., 0., 2., 2., 0.],
                             [4., 4., 1., 1., 0.],
                             [0., 0., 0., 0., 0.]]])
    weights = torch.tensor([[True, True, False]])
    validity, masks = aligned_rotated_iou_validity_targets(
        predicted, targets, weights,
        [dict(img_shape=(16, 16))], torch.tensor([False]))
    assert torch.allclose(validity, torch.tensor([[1.0, 0.25, 0.0]]),
                          atol=1e-6, rtol=0)
    assert masks['positive'].tolist() == [[True, True, False]]
    assert masks['negative'].tolist() == [[False, False, True]]


def test_zero_weight_loss_retains_zero_gradient_connection():
    validity = torch.tensor([0.8, 0.2], requires_grad=True)
    loss = natural_validity_loss(validity, torch.tensor([1.0, 0.0]), 1e-6)
    (loss * 0.0).backward()
    assert torch.equal(validity.grad, torch.zeros_like(validity))


def test_progress_buffer_persists_across_checkpoint_resume():
    first = build_coupling()
    first.set_update(80)
    second = build_coupling()
    second.load_state_dict(first.state_dict())
    assert second.flow_progress.item() == 0.5


def test_reference_angle_is_scaled_by_pi():
    module = build_coupling()
    references = fixed_inputs()[1]
    centers = fixed_inputs()[4]
    cost = module._oriented_spatial_cost(references, centers)
    references_zero = references.clone()
    references_zero[..., 4] = 0
    cost_zero = module._oriented_spatial_cost(references_zero, centers)
    assert not torch.equal(cost, cost_zero)


def test_transport_clamps_centers_and_returns_actual_applied_delta():
    module = build_coupling()
    module.set_update(160)
    parent = torch.tensor([[[0.99, 0.01, 0.2, 0.2, 0.0]]])
    transported, applied = module.apply_center_transport(
        parent, torch.tensor([[[1.0, -1.0]]]))
    assert torch.equal(transported[..., :2], torch.tensor([[[1.0, 0.0]]]))
    assert torch.allclose(applied, torch.tensor([[[0.01, -0.01]]]),
                          atol=1e-7, rtol=0)


def test_parent_eval_hook_unwraps_mmengine_model_wrapper(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = nn.Linear(2, 2)
            self.decoder = nn.Module()
            self.decoder.query_evidence_coupling = build_coupling()

    root = Root()
    wrapper = SimpleNamespace(module=root, training=False)
    runner = SimpleNamespace(model=wrapper, iter=80)
    monkeypatch.setattr(query_evidence_coupling,
                        'is_model_wrapper', lambda model: model is wrapper)
    QueryFlowParentEvalHook().before_train_iter(runner, 0)
    assert wrapper.training is True and root.training is True
    assert root.parent.training is False
    assert root.decoder.query_evidence_coupling.training is True
    assert root.decoder.query_evidence_coupling.flow_progress.item() == 0.5
```

The test file's standard imports are exactly `from types import SimpleNamespace`, `import pytest`, `import torch`, `from torch import nn`, `from torch.nn import functional as F`, `from mmengine.structures import InstanceData`, plus `import projects.OVCapFlow.ov_capflow.query_evidence_coupling as query_evidence_coupling` for the wrapper monkeypatch.

- [ ] **Step 2: Run RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py -q
```

Expected: `ModuleNotFoundError` for `query_evidence_coupling` only.

- [ ] **Step 3: Implement the one production component file**

Use `apply_patch`. The public output is:

```python
class CouplingOutput(NamedTuple):
    coupling: Tensor       # [B,Q,S+1], float32
    validity: Tensor       # [B,Q], float32
    barycenter: Tensor     # [B,Q,2], float32
    center_delta: Tensor   # [B,Q,2], float32
```

The production file begins with the following imports and complete implementation. Keep helper names private so the exported surface remains the registered six names:

```python
import math
from typing import Dict, NamedTuple, Optional, Tuple

import torch
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.structures import InstanceData
from torch import Tensor, nn
from torch.nn import functional as F

from mmrotate.registry import HOOKS, MODELS, TASK_UTILS
from mmrotate.structures.bbox import rbbox_overlaps


class CouplingOutput(NamedTuple):
    coupling: Tensor
    validity: Tensor
    barycenter: Tensor
    center_delta: Tensor


class _DeterministicProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        weight = torch.zeros(out_features, in_features)
        diagonal = min(in_features, out_features)
        weight[torch.arange(diagonal), torch.arange(diagonal)] = 1
        self.weight = nn.Parameter(weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight)


@MODELS.register_module()
class QueryEvidenceCoupling(nn.Module):
    def __init__(self, embed_dims=256, latent_dims=32,
                 solver_iterations=5, entropy=0.1,
                 column_relaxation=1.0, spatial_prior_share=0.75,
                 null_prior_share=0.25, content_weight=1.0,
                 spatial_weight=1.0, semantic_weight=1.0,
                 spatial_cost_cap=4.0, min_reference_extent=0.01,
                 max_center_delta=0.05, angle_factor=math.pi,
                 null_logit_bound=4.0, target_center_strength=1.0,
                 target_score_strength=1.0, warmup_updates=160,
                 eps=1e-6):
        super().__init__()
        frozen = dict(
            embed_dims=embed_dims, latent_dims=latent_dims,
            solver_iterations=solver_iterations, entropy=entropy,
            column_relaxation=column_relaxation,
            spatial_prior_share=spatial_prior_share,
            null_prior_share=null_prior_share,
            content_weight=content_weight, spatial_weight=spatial_weight,
            semantic_weight=semantic_weight,
            spatial_cost_cap=spatial_cost_cap,
            min_reference_extent=min_reference_extent,
            max_center_delta=max_center_delta, angle_factor=angle_factor,
            null_logit_bound=null_logit_bound,
            target_center_strength=target_center_strength,
            target_score_strength=target_score_strength,
            warmup_updates=warmup_updates, eps=eps)
        expected = dict(
            embed_dims=256, latent_dims=32, solver_iterations=5,
            entropy=0.1, column_relaxation=1.0,
            spatial_prior_share=0.75, null_prior_share=0.25,
            content_weight=1.0, spatial_weight=1.0,
            semantic_weight=1.0, spatial_cost_cap=4.0,
            min_reference_extent=0.01, max_center_delta=0.05,
            angle_factor=math.pi, null_logit_bound=4.0,
            warmup_updates=160, eps=1e-6)
        for name, value in expected.items():
            if frozen[name] != value:
                raise ValueError(f'query flow freezes {name}={value}')
        if target_center_strength not in (0.0, 1.0):
            raise ValueError('center target strength must be 0 or 1')
        if target_score_strength not in (0.0, 1.0):
            raise ValueError('score target strength must be 0 or 1')
        for name, value in frozen.items():
            setattr(self, name, value)
        self.q_proj = _DeterministicProjection(embed_dims, latent_dims)
        self.k_proj = _DeterministicProjection(embed_dims, latent_dims)
        self.null_key = nn.Parameter(torch.zeros(latent_dims))
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer('flow_progress', torch.zeros(()),
                             persistent=True)
        self.reset_frozen_initialization()

    @torch.no_grad()
    def reset_frozen_initialization(self):
        self.q_proj.weight.zero_()
        self.k_proj.weight.zero_()
        diagonal = torch.arange(self.latent_dims,
                                device=self.q_proj.weight.device)
        self.q_proj.weight[diagonal, diagonal] = 1
        self.k_proj.weight[diagonal, diagonal] = 1
        self.null_key.zero_()
        self.null_bias.zero_()
        self.flow_progress.zero_()

    @property
    def center_strength(self):
        return self.flow_progress * self.target_center_strength

    @property
    def score_strength(self):
        return self.flow_progress * self.target_score_strength

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def set_update(self, update):
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise ValueError('update must be a nonnegative integer')
        progress = min(float(update) / self.warmup_updates, 1.0)
        self.flow_progress.copy_(self.flow_progress.new_tensor(progress))

    @staticmethod
    def _finite(tensor, name):
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f'{name} must be finite')

    def _validate_inputs(self, queries, references, memory,
                         semantic_evidence, token_centers, valid_mask):
        if (queries.ndim != 3 or queries.shape[-1] != self.embed_dims or
                queries.shape[1] == 0):
            raise ValueError('query count/shape must be [B,Q,256] with Q>0')
        if references.shape != queries.shape[:2] + (5,):
            raise ValueError('references must have shape [B,Q,5]')
        if (memory.ndim != 3 or memory.shape[0] != queries.shape[0] or
                memory.shape[2] != self.embed_dims):
            raise ValueError('memory must have shape [B,S,256]')
        if semantic_evidence.shape != memory.shape[:2]:
            raise ValueError('semantic evidence must have shape [B,S]')
        if token_centers.shape != memory.shape[:2] + (2,):
            raise ValueError('token centers must have shape [B,S,2]')
        if valid_mask.shape != memory.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError('valid mask must be bool [B,S]')
        if not valid_mask.any(dim=1).all().item():
            raise ValueError('every sample requires a valid spatial token')
        for name, tensor in (
                ('queries', queries), ('references', references),
                ('memory', memory), ('semantic evidence', semantic_evidence),
                ('token centers', token_centers)):
            self._finite(tensor, name)

    def _project_queries(self, queries):
        return F.normalize(self.q_proj(queries.float()), dim=-1, eps=self.eps)

    def _bounded_null_logits(self, queries):
        projected = self._project_queries(queries)
        raw = torch.einsum('bqd,d->bq', projected, self.null_key.float())
        raw = raw + self.null_bias.float()
        return self.null_logit_bound * torch.tanh(
            raw / self.null_logit_bound)

    def _oriented_spatial_cost(self, references, token_centers):
        refs = references.float()
        centers = token_centers.float()
        delta = centers[:, None, :, :] - refs[:, :, None, :2]
        theta = refs[..., 4] * self.angle_factor
        cosine = torch.cos(theta)[:, :, None]
        sine = torch.sin(theta)[:, :, None]
        local_x = cosine * delta[..., 0] + sine * delta[..., 1]
        local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
        width = refs[..., 2].clamp(min=self.min_reference_extent)[:, :, None]
        height = refs[..., 3].clamp(min=self.min_reference_extent)[:, :, None]
        return torch.maximum(2 * local_x.abs() / width,
                             2 * local_y.abs() / height).clamp(
                                 max=self.spatial_cost_cap)

    @staticmethod
    def level_token_geometry(spatial_shape, padding_mask):
        height, width = (int(value) for value in spatial_shape)
        mask = padding_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[1] != height * width:
            raise ValueError('level padding mask/token count mismatch')
        grid_mask = mask.reshape(mask.shape[0], height, width)
        valid = ~grid_mask
        valid_rows = valid.any(dim=2)
        valid_cols = valid.any(dim=1)
        valid_height = valid_rows.sum(dim=1)
        valid_width = valid_cols.sum(dim=1)
        if not torch.all((valid_height > 0) & (valid_width > 0)).item():
            raise ValueError('each level needs a valid spatial token')
        rectangular = (valid_rows[:, :, None] & valid_cols[:, None, :])
        if not torch.equal(valid, rectangular):
            raise ValueError('level padding must be a top-left rectangle')
        ys = torch.arange(height, device=mask.device, dtype=torch.float32)
        xs = torch.arange(width, device=mask.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        x = (grid_x.reshape(1, -1) + 0.5) / valid_width[:, None]
        y = (grid_y.reshape(1, -1) + 0.5) / valid_height[:, None]
        centers = torch.stack((x, y), dim=-1)
        return centers, valid.reshape(mask.shape[0], -1)

    def forward(self, queries, references, memory, semantic_evidence,
                token_centers, valid_mask):
        self._validate_inputs(queries, references, memory, semantic_evidence,
                              token_centers, valid_mask)
        query_latent = self._project_queries(queries)
        key_latent = F.normalize(self.k_proj(memory.float()), dim=-1,
                                 eps=self.eps)
        content = torch.einsum('bqd,bsd->bqs', query_latent, key_latent)
        spatial = self._oriented_spatial_cost(references, token_centers)
        token_logits = (self.content_weight * content -
                        self.spatial_weight * spatial +
                        self.semantic_weight *
                        semantic_evidence.float()[:, None, :])
        token_logits = token_logits.masked_fill(
            ~valid_mask[:, None, :], -torch.inf)
        null_logits = self._bounded_null_logits(queries)[..., None]
        logits = torch.cat((token_logits, null_logits), dim=-1)

        spatial_prior = torch.sigmoid(semantic_evidence.float()).clamp(.01, .99)
        spatial_prior = spatial_prior.masked_fill(~valid_mask, 0)
        spatial_prior = spatial_prior / spatial_prior.sum(dim=-1, keepdim=True)
        column_prior = torch.cat(
            (self.spatial_prior_share * spatial_prior,
             spatial_prior.new_full((spatial_prior.shape[0], 1),
                                    self.null_prior_share)), dim=-1)
        column_bias = torch.zeros_like(column_prior)
        gamma = self.column_relaxation / (1.0 + self.entropy)
        for _ in range(self.solver_iterations):
            log_pi = F.log_softmax(
                logits / self.entropy + column_bias[:, None, :], dim=-1)
            column_mass = log_pi.exp().sum(dim=1) / queries.shape[1]
            column_bias = column_bias + gamma * (
                torch.log(column_prior + self.eps) -
                torch.log(column_mass + self.eps))
        coupling = F.log_softmax(
            logits / self.entropy + column_bias[:, None, :], dim=-1).exp()
        validity = 1.0 - coupling[..., -1]
        barycenter = (
            coupling[..., :-1, None] * token_centers.float()[:, None]
        ).sum(dim=-2) / (validity[..., None] + self.eps)
        direction = barycenter - references.float()[..., :2]
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        bounded = direction * torch.minimum(
            torch.ones_like(norm), self.max_center_delta / (norm + self.eps))
        center_delta = validity[..., None] * bounded
        for name, tensor in (
                ('coupling', coupling), ('validity', validity),
                ('barycenter', barycenter), ('center delta', center_delta)):
            self._finite(tensor, name)
        row_error = (coupling.sum(dim=-1) - 1).abs().max()
        if row_error.item() > 1e-5:
            raise ValueError('coupling rows are not normalized')
        return CouplingOutput(coupling, validity, barycenter, center_delta)

    def apply_center_transport(self, parent_reference, center_delta):
        if self.center_strength.item() == 0.0:
            return parent_reference, torch.zeros_like(parent_reference[..., :2])
        transported = parent_reference.clone()
        parent_center = parent_reference[..., :2]
        transported[..., :2] = (
            parent_center + self.center_strength * center_delta).clamp(0, 1)
        return transported, transported[..., :2] - parent_center

    def score_log_residual(self, validity):
        return self.score_strength * torch.log(
            validity.float().clamp(min=self.eps, max=1.0))


@TASK_UTILS.register_module()
class QueryValidityCost(BaseMatchCost):
    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None) -> Tensor:
        if 'flow_match_cost' not in pred_instances:
            raise ValueError('pred_instances.flow_match_cost is required')
        vector = pred_instances.flow_match_cost
        if vector.ndim != 1 or not torch.isfinite(vector).all().item():
            raise ValueError('flow_match_cost must be finite [Q]')
        return vector[:, None].expand(vector.shape[0], len(gt_instances))


def aligned_rotated_iou_validity_targets(
        predicted_boxes: Tensor, target_boxes: Tensor, matched_mask: Tensor,
        batch_img_metas, empty_image_mask: Tensor,
        angle_factor: float = math.pi) -> Tuple[Tensor, Dict[str, Tensor]]:
    if (predicted_boxes.ndim != 3 or predicted_boxes.shape[-1] != 5 or
            target_boxes.shape != predicted_boxes.shape or
            matched_mask.shape != predicted_boxes.shape[:2] or
            matched_mask.dtype != torch.bool):
        raise ValueError('validity target tensors have inconsistent shapes')
    if len(batch_img_metas) != predicted_boxes.shape[0]:
        raise ValueError('one image meta is required per batch item')
    targets = predicted_boxes.new_zeros(predicted_boxes.shape[:2],
                                         dtype=torch.float32)
    for batch_index, meta in enumerate(batch_img_metas):
        selected = matched_mask[batch_index]
        if not selected.any().item():
            continue
        img_h, img_w = meta['img_shape'][:2]
        factor = predicted_boxes.new_tensor(
            [img_w, img_h, img_w, img_h, angle_factor], dtype=torch.float32)
        predicted = predicted_boxes[batch_index, selected].detach().float()
        assigned = target_boxes[batch_index, selected].detach().float()
        quality = rbbox_overlaps(
            predicted * factor, assigned * factor, is_aligned=True)
        if not torch.isfinite(quality).all().item():
            raise ValueError('aligned rotated IoU target is non-finite')
        targets[batch_index, selected] = quality.clamp(0, 1)
    empty = empty_image_mask.to(dtype=torch.bool)[:, None].expand_as(matched_mask)
    masks = {
        'positive': matched_mask.detach(),
        'negative': (~matched_mask).detach(),
        'empty_image': empty.detach(),
        'all_matched': matched_mask.detach(),
    }
    return targets.detach(), masks


def natural_validity_loss(validity: Tensor, targets: Tensor, eps: float):
    if validity.shape != targets.shape:
        raise ValueError('validity and targets must share shape')
    if not torch.isfinite(validity).all().item():
        raise ValueError('validity must be finite')
    return F.binary_cross_entropy(
        validity.clamp(min=eps, max=1.0 - eps), targets.float(),
        reduction='mean')


@HOOKS.register_module()
class QueryFlowParentEvalHook(Hook):
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        wrapper = runner.model
        root = wrapper.module if is_model_wrapper(wrapper) else wrapper
        root.eval()
        wrapper.training = True
        root.training = True
        coupling = root.decoder.query_evidence_coupling
        coupling.train(True)
        coupling.set_update(runner.iter)
```

Register `QueryEvidenceCoupling` with this exact constructor signature:

`QueryEvidenceCoupling(embed_dims: int = 256, latent_dims: int = 32, solver_iterations: int = 5, entropy: float = 0.1, column_relaxation: float = 1.0, spatial_prior_share: float = 0.75, null_prior_share: float = 0.25, content_weight: float = 1.0, spatial_weight: float = 1.0, semantic_weight: float = 1.0, spatial_cost_cap: float = 4.0, min_reference_extent: float = 0.01, max_center_delta: float = 0.05, angle_factor: float = 3.141592653589793, null_logit_bound: float = 4.0, target_center_strength: float = 1.0, target_score_strength: float = 1.0, warmup_updates: int = 160, eps: float = 1e-6)`

Its exact public methods/properties are:

- `center_strength -> Tensor`
- `score_strength -> Tensor`
- `trainable_parameter_count -> int`
- `set_update(update: int) -> None`
- `reset_frozen_initialization() -> None`
- `level_token_geometry(spatial_shape: Tuple[int, int], padding_mask: Tensor) -> Tuple[Tensor, Tensor]`
- `forward(queries: Tensor, references: Tensor, memory: Tensor, semantic_evidence: Tensor, token_centers: Tensor, valid_mask: Tensor) -> CouplingOutput`
- `apply_center_transport(parent_reference: Tensor, center_delta: Tensor) -> Tuple[Tensor, Tensor]`, returning transported references and the actual post-clamp applied center delta
- `score_log_residual(validity: Tensor) -> Tensor`

Register stateless `QueryValidityCost(BaseMatchCost)` with `__call__(pred_instances: InstanceData, gt_instances: InstanceData, img_meta: Optional[dict] = None) -> Tensor`.

Register `QueryFlowParentEvalHook(Hook)` with `before_train_iter(runner, batch_idx, data_batch=None)`.

The hook must handle both an unwrapped model and MMEngine's distributed wrapper exactly:

```python
from mmengine.model import is_model_wrapper


def before_train_iter(self, runner, batch_idx, data_batch=None):
    wrapper = runner.model
    root = wrapper.module if is_model_wrapper(wrapper) else wrapper
    root.eval()
    wrapper.training = True
    root.training = True
    coupling = root.decoder.query_evidence_coupling
    coupling.train(True)
    coupling.set_update(runner.iter)
```

Add `test_parent_eval_hook_unwraps_mmengine_model_wrapper`, using a minimal wrapper fixture whose `.module` is the detector, and assert wrapper/root `training` flags are true, every parent submodule remains eval, only the coupling is train, and its persistent progress equals the registered update. The hook must not contain D13 names, existence-head assumptions, checkpoint rewriting, or telemetry infrastructure.

Implement `aligned_rotated_iou_validity_targets` by reusing `rbbox_overlaps` with `is_aligned=True` and the existing image/angle scaling convention. It returns detached targets and four detached diagnostic masks: positive, negative, empty-image, and all-matched. `natural_validity_loss` optimizes the unweighted all-query mean; diagnostic strata never alter the objective.

- [ ] **Step 4: Export only the registered names and run GREEN**

Add the public names to `projects/OVCapFlow/ov_capflow/__init__.py`. Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py -q
rtk git diff --check
```

Expected: every core test passes and `git diff --check` is silent.

- [ ] **Step 5: Record the source PASS and create concentrated commit 2**

Use `apply_patch` to insert the exact `8-D160-QAF-SOURCE` row from the verified report and its observed values into the four governance files. Then run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/query_evidence_coupling.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py .lab/results.tsv progress.md task_plan.md projects/OVCapFlow/README.md
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "feat: add unified query evidence coupling"
```

Expected: concentrated commit 2 only. Do not stage the generated `.lab/workspace` report or unrelated untracked files.

## Task 2: Integrate the same flow into decoder, matching, loss, and readout

**Files:**

- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_layers.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py`
- Create: both registered world-5 configs

- [ ] **Step 1: Add RED integration and protocol tests before editing model code**

Create named tests for this frozen matrix; the deterministic unit bodies immediately below are copied verbatim, while the three opt-in real-batch/gate bodies use the exact paths and arithmetic frozen later in Tasks 3-5:

1. `OVCapFlow.pre_decoder` uses the frozen final encoder classification branch to return `flow_semantic_evidence [B,S_all]`, masks invalid prompt tokens before `amax`, and does not call encoder proposal regression.
2. Inference matching suffix is `[B,600,256]`; training matching suffix is `[B,1800,256]`; DN prefix never enters the coupling.
3. Six decoder layers reuse the same `id(decoder.query_evidence_coupling)` and emit `flow_validities [6,B,M]` plus already strength-scaled `flow_center_deltas [6,B,M,2]`.
4. `test_real_batch_zero_route_is_parent_exact`: at raw gate zero, labels/scores/boxes/references/order and every parent loss/assignment are `torch.equal` to the parent; `loss_query_validity` is exactly zero and separate.
5. Nonzero `target_center_strength * flow_progress` changes matching-row centers in the same layer and in the next-layer reference; the sixth-layer delta changes the final boxes.
6. Width, height, angle, hidden content, DN references, and query row order do not change at the flow injection point.
7. The final-layer assignment cost is exactly `-warmup*0.5*log(v.clamp(min=1e-6,max=1.0))` repeated over GT columns; auxiliary assignments are unchanged by the first implementation.
8. Validity targets use actual three-group final assignments and detached aligned rotated IoU; unmatched targets are zero.
9. Readout performs unchanged class argmax first, then adds `target_score_strength*flow_progress*log(v.clamp(min=1e-6,max=1.0))` to the selected log score; it cannot change labels or boxes.
10. Static AST inspection rejects `topk`, score-threshold comparisons, query sorting/reordering, NMS/rotated NMS, dense/RPN/RoI head attributes, proposal-selection calls, and literal truncation of prediction/output tensors below Q600.
11. `test_real_batch_q600_finite_contract`: exactly 600 rows emerge for empty `P0021__1024__1048___2096`, normal `P0000__1024__1572___1048`, and dense `P4076__1024__2620___0`; forward, assignment, loss, backward, and optimizer state are finite.
12. `test_real_optimizer_step_freezes_parent`: only 16,417 parameters under `decoder.query_evidence_coupling.*` are trainable; a real optimizer step leaves the SHA256 of every parent tensor unchanged.
13. `test_control_stays_zero_after_real_optimizer_step` proves control target strengths remain zero, its output stays parent-exact, and learned coupling changes cannot open center/score routes.
14. `test_model_init_and_e24_load_preserve_frozen_coupling_initialization` builds the candidate detector, calls the existing parent `init_weights`, loads E24, and after both operations asserts exact identity-truncation projection weights plus zero null key/bias/progress; this prevents the parent DINO Xavier pass from silently changing the registered start.
15. `test_checkpoint_key_contract` asserts the exact four new parameter-tensor keys plus `flow_progress` buffer and no other load mismatch.
16. `test_proxy_promotion_gate_from_frozen_reports` is opt-in, validates finite analyzer fields and exact registered arithmetic, writes one no-replace decision JSON, and fails pytest on any proxy gate.
17. `test_proxy_endpoint_checkpoint_contract` is opt-in and CPU-loads both E12 checkpoints, asserts epoch 12/iter 1,920, finite tensor state, `flow_progress==1`, exact byte equality of every non-flow parent tensor against E24, and at least one changed candidate coupling parameter.
18. `test_full_recipe_world10_sampler_contract` is opt-in, merges only the registered full-training overrides, asserts manifest SHA `152d1244778ecfab454cca459f8cfec03a9927b8894a9eb4d48ec1a9c5cc4ce6`, 47,294 images, world 10 x batch 2, exactly 2,365 updates/epoch, no duplicate/missing sample, and the 12-epoch endpoint.
19. `test_raw_promotion_gate_from_frozen_reports` is opt-in, validates AP50/AP75 namespaces, analyzer fields, strata thresholds, and strict artifacts, writes one no-replace decision JSON, and fails pytest on any raw gate.
20. `test_full_checkpoint_parent_hash_contract` is opt-in and CPU-loads the full E12 candidate, asserts epoch 12/iter 28,380, finite tensors, `flow_progress==1`, at least one coupling parameter changed from the frozen start, and every non-flow tensor byte-identical to E24.

```python
class _FixedQueryInitializer(nn.Module):
    def forward(self, batch_size):
        return (torch.zeros(batch_size, 600, 256),
                torch.full((batch_size, 600, 5), 0.5))


class _PaddedTokenBranch(nn.Module):
    def forward(self, visual, memory_text, text_token_mask):
        logits = visual.new_zeros(visual.shape[0], visual.shape[1], 4)
        logits[..., 0] = 1.0
        logits[..., 1] = 2.0
        logits[..., 2:] = 999.0
        return logits


def test_pre_decoder_uses_runtime_text_slice_without_proposals():
    detector = OVCapFlow.__new__(OVCapFlow)
    nn.Module.__init__(detector)
    detector.num_queries = 600
    detector.train_query_groups = 3
    detector.query_initializer = _FixedQueryInitializer()
    detector.decoder = SimpleNamespace(num_layers=6)
    detector.bbox_head = SimpleNamespace(
        cls_branches=[_PaddedTokenBranch() for _ in range(7)])
    detector.eval()
    memory = torch.zeros(1, 6, 256)
    decoder_inputs, _ = detector.pre_decoder(
        memory=memory,
        memory_mask=torch.zeros(1, 6, dtype=torch.bool),
        spatial_shapes=torch.tensor([[1, 1], [1, 1], [1, 2], [1, 2]]),
        memory_text=torch.zeros(1, 2, 256),
        text_token_mask=torch.tensor([[True, True]]))
    assert torch.equal(decoder_inputs['flow_semantic_evidence'],
                       torch.full((1, 6), 2.0))
    assert decoder_inputs['matching_query_count'] == 600
    assert decoder_inputs['query'].shape == (1, 600, 256)


def test_level2_slice_and_padding_geometry_are_exact():
    decoder = SimpleNamespace(
        query_flow_level_index=2, embed_dims=256,
        query_evidence_coupling=build_coupling())
    value = torch.arange(12 * 256, dtype=torch.float32).reshape(1, 12, 256)
    evidence = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    memory, semantic, centers, valid = \
        OVCapFlowDecoder._prepare_query_flow_level(
            decoder, value, None,
            torch.tensor([[1, 2], [1, 2], [2, 3], [1, 2]]),
            torch.tensor([0, 2, 4, 10]), evidence)
    assert torch.equal(memory, value[:, 4:10])
    assert torch.equal(semantic, evidence[:, 4:10])
    assert centers.shape == (1, 6, 2)
    assert valid.shape == (1, 6) and valid.all()


def test_final_match_cost_uses_suppressive_clamped_log():
    validity = torch.tensor([1.0, 0.5, 0.0])
    current_match_weight = 0.25
    vector = -current_match_weight * torch.log(
        validity.clamp(min=1e-6, max=1.0))
    pred = InstanceData(flow_match_cost=vector)
    gt = InstanceData(labels=torch.tensor([0, 1]))
    cost = QueryValidityCost()(pred, gt)
    assert torch.equal(cost[:, 0], vector)
    assert torch.equal(cost[:, 0], cost[:, 1])
    assert cost[0, 0].item() == 0.0


def test_score_residual_is_neutral_or_suppressive_only():
    coupling = build_coupling()
    coupling.set_update(160)
    validity = torch.tensor([[1.0, 0.5, 0.0]])
    residual = coupling.score_log_residual(validity)
    assert residual[0, 0].item() == 0.0
    assert torch.all(residual <= 0)
    assert torch.isfinite(residual).all()


class _IdentityDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_references = []

    def forward(self, query, **kwargs):
        self.seen_references.append(
            kwargs['reference_points'].detach().clone())
        return query


class _ZeroRegBranch(nn.Module):
    def forward(self, query):
        return query.new_zeros(query.shape[:2] + (5,))


def _run_decoder_suffix_contract(matching_count, dn_prefix):
    decoder = OVCapFlowDecoder.__new__(OVCapFlowDecoder)
    nn.Module.__init__(decoder)
    decoder.embed_dims = 256
    decoder.angle_factor = math.pi
    decoder.return_intermediate = True
    decoder.norm = nn.Identity()
    decoder.ref_point_head = nn.Identity()
    decoder.query_flow_level_index = 2
    decoder.query_evidence_coupling = build_coupling()
    decoder.query_evidence_coupling.set_update(160)
    decoder.layers = nn.ModuleList(
        [_IdentityDecoderLayer() for _ in range(6)])
    decoder.last_flow_validities = None
    decoder.last_flow_center_deltas = None
    decoder.last_null_logits = None
    decoder.null_reservoir = None
    total_queries = matching_count + dn_prefix
    row_ids = torch.arange(total_queries, dtype=torch.float32)
    query = (row_ids[:, None].expand(total_queries, 256) /
             max(total_queries, 1)).unsqueeze(0)
    original_query = query.clone()
    references = torch.full((1, total_queries, 5), 0.5)
    value = torch.arange(12 * 256, dtype=torch.float32).reshape(
        1, 12, 256) / 3072
    evidence = torch.linspace(-0.5, 0.5, 12).unsqueeze(0)
    module_ids = []

    def capture_module(module, _args, _output):
        module_ids.append(id(module))

    handle = decoder.query_evidence_coupling.register_forward_hook(
        capture_module)
    try:
        states, output_references = decoder.forward(
            query=query,
            value=value,
            key_padding_mask=torch.zeros(1, 12, dtype=torch.bool),
            self_attn_mask=None,
            reference_points=references,
            spatial_shapes=torch.tensor(
                [[1, 2], [1, 2], [2, 3], [1, 2]]),
            level_start_index=torch.tensor([0, 2, 4, 10]),
            valid_ratios=torch.ones(1, 4, 2),
            reg_branches=nn.ModuleList(
                [_ZeroRegBranch() for _ in range(6)]),
            matching_query_count=matching_count,
            flow_semantic_evidence=evidence)
    finally:
        handle.remove()
    assert module_ids == [id(decoder.query_evidence_coupling)] * 6
    assert decoder.last_flow_validities.shape == (
        6, 1, matching_count)
    assert decoder.last_flow_center_deltas.shape == (
        6, 1, matching_count, 2)
    assert torch.count_nonzero(decoder.last_flow_center_deltas[-1]) > 0
    assert states.shape == (6, 1, total_queries, 256)
    assert torch.equal(
        states, original_query.unsqueeze(0).expand_as(states))
    assert output_references.shape == (7, 1, total_queries, 5)
    if dn_prefix:
        assert torch.equal(
            output_references[:, :, :dn_prefix],
            references[:, :dn_prefix].unsqueeze(0).expand(
                7, -1, -1, -1))
    assert torch.equal(
        output_references[..., 2:],
        torch.full_like(output_references[..., 2:], 0.5))
    assert not torch.equal(
        output_references[-1, :, -matching_count:, :2],
        references[:, -matching_count:, :2])
    second_layer_input = decoder.layers[1].seen_references[0]
    assert torch.equal(
        second_layer_input[:, :, 0, :2], output_references[1, ..., :2])


def test_decoder_uses_exact_inference_and_training_matching_suffixes():
    _run_decoder_suffix_contract(matching_count=600, dn_prefix=0)
    _run_decoder_suffix_contract(matching_count=1800, dn_prefix=100)


class _FixedHeadClassBranch(nn.Module):
    def forward(self, hidden, memory_text, text_token_mask):
        del memory_text, text_token_mask
        return torch.stack(
            [hidden[..., 0], hidden[..., 1], hidden[..., 2]], dim=-1)


def test_same_layer_and_sixth_layer_transport_changes_only_suffix_xy():
    head = OVCapFlowHead.__new__(OVCapFlowHead)
    nn.Module.__init__(head)
    head.cls_branches = nn.ModuleList(
        [_FixedHeadClassBranch() for _ in range(6)])
    head.reg_branches = nn.ModuleList(
        [_ZeroRegBranch() for _ in range(6)])
    hidden = torch.arange(6 * 7 * 256, dtype=torch.float32).reshape(
        6, 1, 7, 256) / 10000
    hidden_before = hidden.clone()
    references = torch.full((6, 1, 7, 5), 0.5)
    deltas = torch.zeros(6, 1, 4, 2)
    deltas[..., 0] = torch.tensor([0.001, 0.002, 0.003, 0.004])
    deltas[..., 1] = torch.tensor([0.004, 0.003, 0.002, 0.001])
    parent_cls, parent_boxes = OVCapFlowHead.forward(
        head, hidden, references, torch.zeros(1, 2, 256),
        torch.ones(1, 2, dtype=torch.bool))
    flow_cls, flow_boxes = OVCapFlowHead.forward(
        head, hidden, references, torch.zeros(1, 2, 256),
        torch.ones(1, 2, dtype=torch.bool),
        flow_center_deltas=deltas, flow_center_strength=torch.tensor(1.0))
    assert torch.equal(hidden, hidden_before)
    assert torch.equal(parent_cls, flow_cls)
    assert torch.equal(parent_boxes[:, :, :3], flow_boxes[:, :, :3])
    assert torch.equal(parent_boxes[..., 2:], flow_boxes[..., 2:])
    assert torch.equal(
        flow_boxes[:, :, -4:, :2],
        (parent_boxes[:, :, -4:, :2] + deltas).clamp(0, 1))
    assert not torch.equal(flow_boxes[-1], parent_boxes[-1])


class _FlowLossSpy:
    def __init__(self):
        self.query_flow_loss_cfg = {'match_weight': 0.5}
        self.auxiliary_inputs = []
        self.seen_costs = None
        self.last_flow_validity_targets = None
        self.last_flow_target_masks = None

    def loss_by_feat_single(self, cls_scores, bbox_preds, *_args):
        self.auxiliary_inputs.append((cls_scores.clone(), bbox_preds.clone()))
        scalar = cls_scores.sum() * 0
        return scalar, scalar, scalar

    def get_targets(self, cls_rows, bbox_rows, *_args, **kwargs):
        self.seen_costs = kwargs['flow_match_costs']
        batch_size, queries = len(cls_rows), cls_rows[0].shape[0]
        self.last_flow_validity_targets = torch.zeros(
            batch_size, queries, 5)
        self.last_flow_target_masks = torch.zeros(
            batch_size, queries, dtype=torch.bool)
        return object()

    def _loss_from_targets(self, cls_scores, *_args):
        scalar = cls_scores.sum() * 0
        return scalar, scalar, scalar


def test_nonzero_matching_changes_final_assignment_only():
    spy = _FlowLossSpy()
    cls_scores = torch.arange(
        6 * 1 * 4 * 3, dtype=torch.float32).reshape(6, 1, 4, 3)
    bbox_preds = torch.full((6, 1, 4, 5), 0.5)
    validity = torch.tensor([[1.0, 0.5, 0.25, 0.0]])
    losses, _, _ = OVCapFlowHead._flow_matching_loss_by_feat(
        spy, cls_scores, bbox_preds, validity, torch.tensor(1.0),
        [InstanceData()], [{'img_shape': (1024, 1024)}])
    assert len(spy.auxiliary_inputs) == 5
    assert all(torch.equal(seen[0], cls_scores[layer_id])
               for layer_id, seen in enumerate(spy.auxiliary_inputs))
    expected = -0.5 * torch.log(validity[0].clamp(1e-6, 1.0))
    assert len(spy.seen_costs) == 1
    assert torch.equal(spy.seen_costs[0], expected)
    assert set(losses) == {
        'loss_cls', 'loss_bbox', 'loss_iou',
        'd0.loss_cls', 'd0.loss_bbox', 'd0.loss_iou',
        'd1.loss_cls', 'd1.loss_bbox', 'd1.loss_iou',
        'd2.loss_cls', 'd2.loss_bbox', 'd2.loss_iou',
        'd3.loss_cls', 'd3.loss_bbox', 'd3.loss_iou',
        'd4.loss_cls', 'd4.loss_bbox', 'd4.loss_iou'}


def test_readout_residual_preserves_labels_boxes_and_row_order(monkeypatch):
    head = OVCapFlowHead.__new__(OVCapFlowHead)
    nn.Module.__init__(head)
    head.readout_cfg = {
        'temperature': 1.0, 'power': 1.0, 'use_capacity': False}
    head.angle_factor = math.pi
    class_log_scores = torch.tensor([
        [-0.2, -0.5, -0.8],
        [-1.0, -0.1, -0.4],
        [-0.6, -0.7, -0.3],
        [-0.9, -0.2, -0.5],
    ])
    monkeypatch.setattr(
        ov_capflow_head, 'grounding_logits_to_class_log_scores',
        lambda _scores, _positive_map: class_log_scores)
    cls_score = torch.zeros(4, 2)
    boxes = torch.tensor([
        [0.1, 0.2, 0.2, 0.3, 0.1],
        [0.2, 0.3, 0.3, 0.4, 0.2],
        [0.3, 0.4, 0.4, 0.5, 0.3],
        [0.4, 0.5, 0.5, 0.6, 0.4],
    ])
    meta = {'img_shape': (100, 200), 'scale_factor': (1.0, 1.0)}
    parent = head._predict_by_feat_single(
        cls_score, boxes, {}, meta, rescale=False)
    flow = head._predict_by_feat_single(
        cls_score, boxes, {}, meta, rescale=False,
        flow_score_log_residual=torch.tensor([0.0, -0.2, -0.4, -0.6]))
    assert torch.equal(parent.labels, torch.tensor([0, 1, 2, 1]))
    assert torch.equal(flow.labels, parent.labels)
    assert torch.equal(flow.bboxes, parent.bboxes)
    assert torch.equal(boxes, torch.tensor([
        [0.1, 0.2, 0.2, 0.3, 0.1],
        [0.2, 0.3, 0.3, 0.4, 0.2],
        [0.3, 0.4, 0.4, 0.5, 0.3],
        [0.4, 0.5, 0.5, 0.6, 0.4],
    ]))
    assert torch.all(flow.scores <= parent.scores)


def test_production_ast_contains_no_forbidden_mouth_calls():
    forbidden = {'topk', 'sort', 'argsort', 'nms', 'nms_rotated',
                 'multiclass_nms', 'gen_encoder_output_proposals',
                 'select_proposals', 'select_encoder_proposals'}
    forbidden_heads = {'dense_head', 'rpn_head', 'roi_head'}
    threshold_names = {
        'score_thr', 'score_threshold', 'confidence_thr',
        'confidence_threshold', 'filter_thr', 'filter_threshold'}
    output_names = {
        'pred_instances', 'predictions', 'results', 'outputs',
        'scores', 'labels', 'bboxes', 'boxes'}
    paths = [
        Path('projects/OVCapFlow/ov_capflow/query_evidence_coupling.py'),
        Path('projects/OVCapFlow/ov_capflow/ov_capflow.py'),
        Path('projects/OVCapFlow/ov_capflow/ov_capflow_layers.py'),
        Path('projects/OVCapFlow/ov_capflow/ov_capflow_head.py'),
    ]
    violations = []

    def names_below(node):
        return {
            child.id for child in ast.walk(node)
            if isinstance(child, ast.Name)
        } | {
            child.attr for child in ast.walk(node)
            if isinstance(child, ast.Attribute)
        }

    def slice_bound(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (isinstance(node, ast.UnaryOp) and
                isinstance(node.op, ast.USub) and
                isinstance(node.operand, ast.Constant) and
                isinstance(node.operand.value, int)):
            return -node.operand.value
        return None

    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (node.func.attr if isinstance(node.func, ast.Attribute)
                        else node.func.id if isinstance(node.func, ast.Name)
                        else None)
                if name in forbidden or (name and 'proposal' in name.lower()):
                    violations.append((str(path), node.lineno, name))
            if isinstance(node, ast.Attribute) and node.attr in forbidden_heads:
                violations.append((str(path), node.lineno, node.attr))
            if (isinstance(node, ast.Compare) and
                    names_below(node) & threshold_names):
                violations.append((str(path), node.lineno,
                                   'score-threshold-comparison'))
            if isinstance(node, ast.Subscript):
                slice_node = node.slice
                if isinstance(slice_node, ast.Tuple):
                    slices = slice_node.elts
                else:
                    slices = [slice_node]
                bounds = []
                for item in slices:
                    if isinstance(item, ast.Slice):
                        bounds.extend((slice_bound(item.lower),
                                       slice_bound(item.upper)))
                literal_sub_q = any(
                    bound is not None and 0 < abs(bound) < 600
                    for bound in bounds)
                if literal_sub_q and names_below(node.value) & output_names:
                    violations.append((str(path), node.lineno,
                                       'literal-sub-q600-output-slice'))
    assert violations == []


_FLOW_PREFIX = 'decoder.query_evidence_coupling.'


def _load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location='cpu')
    assert isinstance(checkpoint, dict) and isinstance(checkpoint['meta'], dict)
    state = checkpoint['state_dict']
    assert isinstance(state, dict) and state
    assert all(torch.is_tensor(value) and torch.isfinite(value).all()
               for value in state.values())
    return checkpoint, state


def _assert_nonflow_parent_exact(candidate_state, parent_state):
    nonflow = {name: value for name, value in candidate_state.items()
               if not name.startswith(_FLOW_PREFIX)}
    assert set(nonflow) == set(parent_state)
    for name, parent_value in parent_state.items():
        assert torch.equal(nonflow[name], parent_value), name


def _coupling_changed_from_frozen_start(state):
    q_key = _FLOW_PREFIX + 'q_proj.weight'
    k_key = _FLOW_PREFIX + 'k_proj.weight'
    null_key = _FLOW_PREFIX + 'null_key'
    null_bias = _FLOW_PREFIX + 'null_bias'
    identity = torch.zeros_like(state[q_key])
    diagonal = torch.arange(32)
    identity[diagonal, diagonal] = 1
    expected = {
        q_key: identity,
        k_key: identity.clone(),
        null_key: torch.zeros_like(state[null_key]),
        null_bias: torch.zeros_like(state[null_bias]),
    }
    return any(not torch.equal(state[name], value)
               for name, value in expected.items())


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_PROXY_CHECKPOINTS') != '1',
    reason='proxy endpoint checkpoint gate is opt-in')
def test_proxy_endpoint_checkpoint_contract():
    parent_checkpoint, parent_state = _load_checkpoint_state(
        os.environ['OVCAPFLOW_QUERY_FLOW_PARENT_CHECKPOINT'])
    del parent_checkpoint
    for variable, require_changed in (
            ('OVCAPFLOW_QUERY_FLOW_CONTROL_CHECKPOINT', False),
            ('OVCAPFLOW_QUERY_FLOW_CANDIDATE_CHECKPOINT', True)):
        checkpoint, state = _load_checkpoint_state(os.environ[variable])
        assert checkpoint['meta']['epoch'] == 12
        assert checkpoint['meta']['iter'] == 1920
        assert state[_FLOW_PREFIX + 'flow_progress'].item() == 1.0
        _assert_nonflow_parent_exact(state, parent_state)
        if require_changed:
            assert _coupling_changed_from_frozen_start(state)


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_FULL_CHECKPOINT') != '1',
    reason='full endpoint checkpoint gate is opt-in')
def test_full_checkpoint_parent_hash_contract():
    checkpoint, state = _load_checkpoint_state(
        os.environ['OVCAPFLOW_QUERY_FLOW_FULL_CHECKPOINT'])
    _, parent_state = _load_checkpoint_state(
        os.environ['OVCAPFLOW_QUERY_FLOW_PARENT_CHECKPOINT'])
    assert checkpoint['meta']['epoch'] == 12
    assert checkpoint['meta']['iter'] == 28380
    assert state[_FLOW_PREFIX + 'flow_progress'].item() == 1.0
    assert _coupling_changed_from_frozen_start(state)
    _assert_nonflow_parent_exact(state, parent_state)
```

Add `import ast`, `import csv`, `import json`, `import math`, `import os`,
`from pathlib import Path`, `from mmengine import Config`,
`from mmengine.utils import import_modules_from_strings`,
`from mmrotate.registry import DATASETS, MODELS`,
`from mmrotate.utils import register_all_modules`,
`from projects.OVCapFlow.tools.audit_sampler_coverage import
_build_sampler_coverage_report`, and imports for `OVCapFlow` and
`OVCapFlowDecoder`, `OVCapFlowHead`, plus
`import projects.OVCapFlow.ov_capflow.ov_capflow_head as ov_capflow_head` to
the standard imports already frozen in Task 1. The
opt-in real-batch test reuses the repository's existing
`Runner.build_dataloader` plus `model.data_preprocessor` path, selects the
three exact image ids listed in item 11, and asserts the itemized tensor
shapes/equalities directly; it must not create a third harness or test file.

Run the newly added integration tests and confirm they fail on missing arguments/behavior, not on fixture or environment errors.

- [ ] **Step 2: Thread frozen semantic evidence through the detector**

In `OVCapFlow.pre_decoder`, after the unchanged parent query/reference construction, compute:

```python
full_token_logits = self.bbox_head.cls_branches[self.decoder.num_layers](
    memory, memory_text, text_token_mask)
text_length = text_token_mask.shape[1]
token_logits = full_token_logits[..., :text_length]
masked = token_logits.float().masked_fill(
    ~text_token_mask[:, None, :], -torch.inf)
flow_semantic_evidence = masked.amax(dim=-1)
```

Override detector initialization so the parent DINO Xavier sweep runs first and the new frozen start is restored last:

```python
def init_weights(self):
    super().init_weights()
    coupling = getattr(self.decoder, 'query_evidence_coupling', None)
    if coupling is not None:
        coupling.reset_frozen_initialization()
```

Replace the detector's `init_weights`, `pre_decoder`, and calibration-state
append methods with these complete bodies. The only change inside the existing
fixed-query construction is the evidence reduction and one decoder argument:

```python
def init_weights(self):
    super().init_weights()
    coupling = getattr(self.decoder, 'query_evidence_coupling', None)
    if coupling is not None:
        coupling.reset_frozen_initialization()


def pre_decoder(self, memory, memory_mask, spatial_shapes, memory_text,
                text_token_mask, batch_data_samples=None):
    batch_size = memory.shape[0]
    query, reference_points = self.query_initializer(batch_size)
    matching_query_count = self.num_queries
    if self.training:
        query, reference_points = repeat_matching_queries(
            query, reference_points, self.train_query_groups)
        matching_query_count *= self.train_query_groups
        dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
            self.dn_query_generator(batch_data_samples)
        num_dn = int(dn_meta['num_denoising_queries'])
        dn_mask = expand_dn_attention_mask(
            dn_mask, num_dn=num_dn, queries_per_group=self.num_queries,
            groups=self.train_query_groups)
        dn_meta = dict(dn_meta)
        dn_meta.update(
            num_matching_query_groups=self.train_query_groups,
            num_matching_queries_per_group=self.num_queries)
        query = torch.cat([dn_label_query, query], dim=1)
        matching_reference_logits = torch.logit(
            reference_points.clamp(min=1e-4, max=1 - 1e-4))
        reference_points = torch.cat(
            [dn_bbox_query, matching_reference_logits], dim=1).sigmoid()
    else:
        dn_mask, dn_meta = None, None

    full_token_logits = self.bbox_head.cls_branches[
        self.decoder.num_layers](memory, memory_text, text_token_mask)
    text_length = int(text_token_mask.shape[1])
    if (full_token_logits.shape[:2] != memory.shape[:2] or
            full_token_logits.shape[2] < text_length or
            not text_token_mask.any(dim=1).all().item()):
        raise ValueError('query-flow encoder classification shape is invalid')
    token_logits = full_token_logits[..., :text_length]
    masked = token_logits.float().masked_fill(
        ~text_token_mask[:, None, :], -torch.inf)
    flow_semantic_evidence = masked.amax(dim=-1)
    if not torch.isfinite(flow_semantic_evidence).all().item():
        raise ValueError('query-flow semantic evidence must be finite')

    decoder_inputs_dict = dict(
        query=query, native_query=query,
        matching_query_count=matching_query_count,
        memory=memory, reference_points=reference_points,
        dn_mask=dn_mask, memory_text=memory_text,
        text_attention_mask=~text_token_mask,
        flow_semantic_evidence=flow_semantic_evidence)
    head_inputs_dict = dict(
        enc_outputs_class=None, enc_outputs_coord=None,
        dn_meta=dn_meta) if self.training else {}
    head_inputs_dict['memory_text'] = memory_text
    head_inputs_dict['text_token_mask'] = text_token_mask
    return decoder_inputs_dict, head_inputs_dict


def _append_calibration_state(self, outputs):
    outputs = dict(outputs)
    last_layer = self.decoder.layers[-1]
    if self.decoder.last_null_logits is not None:
        outputs['null_logits'] = self.decoder.last_null_logits
    if last_layer.last_capacity is not None:
        outputs['capacity'] = last_layer.last_capacity
    if last_layer.last_semantic_gate is not None:
        outputs['semantic_gate'] = last_layer.last_semantic_gate
    coupling = getattr(self.decoder, 'query_evidence_coupling', None)
    if coupling is not None:
        validities = self.decoder.last_flow_validities
        deltas = self.decoder.last_flow_center_deltas
        if validities is None or deltas is None:
            raise RuntimeError('query-flow decoder cache is incomplete')
        outputs['flow_validities'] = validities
        outputs['flow_center_deltas'] = deltas
        outputs['flow_progress'] = coupling.flow_progress.detach().clone()
        outputs['flow_center_strength'] = \
            coupling.center_strength.detach().clone()
        outputs['flow_score_log_residual'] = \
            coupling.score_log_residual(validities[-1])
    return outputs
```

The exact integration test is:

```python
def _assert_frozen_coupling_start(coupling):
    expected = torch.zeros(32, 256)
    expected[torch.arange(32), torch.arange(32)] = 1
    assert torch.equal(coupling.q_proj.weight.detach().cpu(), expected)
    assert torch.equal(coupling.k_proj.weight.detach().cpu(), expected)
    assert torch.count_nonzero(coupling.null_key) == 0
    assert coupling.null_bias.item() == 0.0
    assert coupling.flow_progress.item() == 0.0


def test_model_init_and_e24_load_preserve_frozen_coupling_initialization():
    cfg = Config.fromfile(CANDIDATE_CONFIG)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    model.init_weights()
    _assert_frozen_coupling_start(model.decoder.query_evidence_coupling)
    checkpoint = torch.load(PARENT_CHECKPOINT, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    _assert_frozen_coupling_start(model.decoder.query_evidence_coupling)
```

Define `BASE_CONFIG` and `CONTROL_CONFIG` beside the already registered
`CANDIDATE_CONFIG` and `PARENT_CHECKPOINT`. Add `copy`, `hashlib`,
`pseudo_collate`, and `Runner` to the test imports, then add these complete
Stage-0 fixtures and named tests. The tests are opt-in only because each builds
the real Swin-T detector and the three registered batches:

```python
CANDIDATE_CONFIG = Path(
    'configs/ov_capflow/dotav2/'
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_query_flow_candidate.py')
PARENT_CHECKPOINT = Path(
    'work_dirs/dotav2_cleanstart/'
    'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/'
    'epoch_24.pth')
BASE_CONFIG = Path(
    'configs/ov_capflow/dotav2/'
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py')
CONTROL_CONFIG = Path(
    'configs/ov_capflow/dotav2/'
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_query_flow_control.py')
REAL_BATCH_IDS = (
    'P0021__1024__1048___2096',
    'P0000__1024__1572___1048',
    'P4076__1024__2620___0',
)
EXPECTED_FLOW_KEYS = {
    'decoder.query_evidence_coupling.q_proj.weight',
    'decoder.query_evidence_coupling.k_proj.weight',
    'decoder.query_evidence_coupling.null_key',
    'decoder.query_evidence_coupling.null_bias',
    'decoder.query_evidence_coupling.flow_progress',
}


def _build_real_model(config_path, device='cuda:0'):
    cfg = Config.fromfile(str(config_path))
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = torch.load(PARENT_CHECKPOINT, map_location='cpu')
    incompatible = model.load_state_dict(
        checkpoint['state_dict'], strict=False)
    expected_missing = (EXPECTED_FLOW_KEYS if
                        'query_flow' in str(config_path) else set())
    assert set(incompatible.missing_keys) == expected_missing
    assert incompatible.unexpected_keys == []
    return model.to(device)


def _build_real_dataset():
    cfg = Config.fromfile(str(BASE_CONFIG))
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    return Runner.build_dataloader(cfg.train_dataloader).dataset


def _dataset_indices_by_image_id(dataset):
    result = {}
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        image_id = str(info.get('img_id') or Path(info['img_path']).stem)
        assert image_id not in result
        result[image_id] = index
    assert all(image_id in result for image_id in REAL_BATCH_IDS)
    return result


def _processed_real_batch(model, dataset, index, training):
    raw = pseudo_collate([dataset[int(index)]])
    return model.data_preprocessor(raw, training=training)


def _frozen_parent_training_mode(model):
    model.eval()
    model.training = True
    coupling = getattr(model.decoder, 'query_evidence_coupling', None)
    if coupling is not None:
        coupling.train(True)
    return model


def _loss_total(losses):
    selected = [value for name, value in losses.items()
                if 'loss' in name and torch.is_tensor(value)]
    assert selected and all(torch.isfinite(value).all() for value in selected)
    return torch.stack([value.float() for value in selected]).sum()


def _all_tensors_finite(value):
    if torch.is_tensor(value):
        return torch.isfinite(value).all().item()
    if isinstance(value, dict):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    return True


def _capture_references_during(call, model):
    captured = []

    def capture(_module, arguments):
        captured.append(arguments[1].detach().clone())

    handle = model.bbox_head.register_forward_pre_hook(capture)
    try:
        output = call()
    finally:
        handle.remove()
    assert len(captured) == 1
    return output, captured[0]


def _state_without_flow(model):
    return {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in model.state_dict().items()
        if not name.startswith('decoder.query_evidence_coupling.')
    }


def test_identity_contract():
    coupling = build_coupling()
    query, reference, memory, evidence, centers, valid = fixed_inputs()
    flow = coupling(query, reference, memory, evidence, centers, valid)
    transported, delta = coupling.apply_center_transport(
        reference, flow.center_delta)
    assert torch.equal(transported, reference)
    assert torch.count_nonzero(delta) == 0
    assert torch.count_nonzero(coupling.score_log_residual(flow.validity)) == 0


def test_checkpoint_key_contract():
    model = _build_real_model(CANDIDATE_CONFIG, device='cpu')
    state = model.state_dict()
    assert EXPECTED_FLOW_KEYS <= set(state)
    assert set(name for name in state if name.startswith(_FLOW_PREFIX)) == \
        EXPECTED_FLOW_KEYS


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH') != '1',
    reason='real Q600 Stage-0 gate is opt-in')
def test_real_batch_q600_finite_contract():
    model = _build_real_model(CANDIDATE_CONFIG)
    model.decoder.query_evidence_coupling.set_update(160)
    dataset = _build_real_dataset()
    indices = _dataset_indices_by_image_id(dataset)
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    assert sum(parameter.numel() for parameter in parameters) == 16417
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    for image_id in REAL_BATCH_IDS:
        batch = _processed_real_batch(
            model, dataset, indices[image_id], training=True)
        _frozen_parent_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(batch['inputs'], batch['data_samples'])
        total = _loss_total(losses)
        total.backward()
        assert all(parameter.grad is None or
                   torch.isfinite(parameter.grad).all().item()
                   for parameter in model.parameters())
        optimizer.step()
        assert _all_tensors_finite(optimizer.state)
        eval_batch = _processed_real_batch(
            model, dataset, indices[image_id], training=False)
        model.eval()
        with torch.no_grad():
            predictions = model.predict(
                eval_batch['inputs'], eval_batch['data_samples'],
                rescale=True)
        instances = predictions[0].pred_instances
        assert len(instances) == 600
        assert all(torch.isfinite(value).all().item() for value in
                   (instances.bboxes, instances.scores))


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH') != '1',
    reason='real parent-freeze Stage-0 gate is opt-in')
def test_real_optimizer_step_freezes_parent():
    model = _build_real_model(CANDIDATE_CONFIG)
    model.decoder.query_evidence_coupling.set_update(160)
    dataset = _build_real_dataset()
    index = _dataset_indices_by_image_id(dataset)[REAL_BATCH_IDS[1]]
    before = _state_without_flow(model)
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    assert sum(parameter.numel() for parameter in parameters) == 16417
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    batch = _processed_real_batch(model, dataset, index, training=True)
    _frozen_parent_training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    _loss_total(model.loss(
        batch['inputs'], batch['data_samples'])).backward()
    optimizer.step()
    after = _state_without_flow(model)
    assert set(before) == set(after)
    assert before == after


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH') != '1',
    reason='real zero-route Stage-0 gate is opt-in')
def test_real_batch_zero_route_is_parent_exact():
    parent = _build_real_model(BASE_CONFIG)
    control = _build_real_model(CONTROL_CONFIG)
    control.decoder.query_evidence_coupling.set_update(160)
    dataset = _build_real_dataset()
    index = _dataset_indices_by_image_id(dataset)[REAL_BATCH_IDS[1]]
    base_batch = _processed_real_batch(
        parent, dataset, index, training=False)
    parent.eval()
    control.eval()
    with torch.no_grad():
        parent_predictions, parent_references = _capture_references_during(
            lambda: parent.predict(
                base_batch['inputs'], copy.deepcopy(base_batch['data_samples']),
                rescale=True), parent)
        control_predictions, control_references = _capture_references_during(
            lambda: control.predict(
                base_batch['inputs'], copy.deepcopy(base_batch['data_samples']),
                rescale=True), control)
    assert torch.equal(parent_references, control_references)
    parent_instances = parent_predictions[0].pred_instances
    control_instances = control_predictions[0].pred_instances
    for name in ('bboxes', 'scores', 'labels'):
        assert torch.equal(parent_instances[name], control_instances[name])

    train_batch = _processed_real_batch(
        parent, dataset, index, training=True)
    _frozen_parent_training_mode(parent)
    _frozen_parent_training_mode(control)
    torch.manual_seed(20260716)
    parent_losses, parent_train_references = _capture_references_during(
        lambda: parent.loss(
            train_batch['inputs'],
            copy.deepcopy(train_batch['data_samples'])), parent)
    parent_assignments = parent.bbox_head.last_matching_mask.clone()
    torch.manual_seed(20260716)
    control_losses, control_train_references = _capture_references_during(
        lambda: control.loss(
            train_batch['inputs'],
            copy.deepcopy(train_batch['data_samples'])), control)
    control_assignments = control.bbox_head.last_matching_mask.clone()
    assert torch.equal(parent_train_references, control_train_references)
    assert set(parent_losses) == set(control_losses) - {'loss_query_validity'}
    assert all(torch.equal(parent_losses[name], control_losses[name])
               for name in parent_losses)
    assert control_losses['loss_query_validity'].item() == 0.0
    assert torch.equal(parent_assignments, control_assignments)


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH') != '1',
    reason='real nonzero-flow Stage-0 gate is opt-in')
def test_real_candidate_nonzero_flow_contract():
    model = _build_real_model(CANDIDATE_CONFIG)
    model.decoder.query_evidence_coupling.set_update(160)
    dataset = _build_real_dataset()
    index = _dataset_indices_by_image_id(dataset)[REAL_BATCH_IDS[1]]
    decoder_calls = []
    coupling_calls = []

    def capture_decoder(_module, args, kwargs):
        query = kwargs.get('query', args[0] if args else None)
        decoder_calls.append(
            (tuple(query.shape), int(kwargs['matching_query_count'])))

    def capture_coupling(module, args, _output):
        coupling_calls.append((id(module), tuple(args[0].shape)))

    decoder_handle = model.decoder.register_forward_pre_hook(
        capture_decoder, with_kwargs=True)
    coupling_handle = \
        model.decoder.query_evidence_coupling.register_forward_hook(
            capture_coupling)
    try:
        batch = _processed_real_batch(
            model, dataset, index, training=True)
        _frozen_parent_training_mode(model)
        losses = model.loss(
            batch['inputs'], copy.deepcopy(batch['data_samples']))
        assert torch.isfinite(_loss_total(losses))
    finally:
        decoder_handle.remove()
        coupling_handle.remove()
    assert len(decoder_calls) == 1
    training_shape, training_matching_count = decoder_calls[0]
    assert training_matching_count == 1800
    assert training_shape[0] == 1 and training_shape[2] == 256
    assert training_shape[1] > training_matching_count
    dn_prefix = training_shape[1] - training_matching_count
    assert dn_prefix > 0 and dn_prefix % 2 == 0
    coupling_id = id(model.decoder.query_evidence_coupling)
    assert coupling_calls == [(coupling_id, (1, 1800, 256))] * 6
    assert model.decoder.last_flow_validities.shape == (6, 1, 1800)
    assert model.decoder.last_flow_center_deltas.shape == (6, 1, 1800, 2)
    assert torch.count_nonzero(
        model.decoder.last_flow_center_deltas[-1]) > 0
    targets = model.bbox_head.last_flow_validity_targets
    group_masks = model.bbox_head.last_matching_group_masks
    assert targets.shape == (1, 1800) and not targets.requires_grad
    assert group_masks.shape == (1, 3, 600) and not group_masks.requires_grad
    flat_matched = group_masks.flatten(1, 2)
    assert torch.count_nonzero(targets[~flat_matched]) == 0
    assert set(model.bbox_head.last_flow_target_masks) == {
        'positive', 'negative', 'empty_image', 'all_matched'}
    assert all(not mask.requires_grad for mask in
               model.bbox_head.last_flow_target_masks.values())

    decoder_calls.clear()
    coupling_calls.clear()
    decoder_handle = model.decoder.register_forward_pre_hook(
        capture_decoder, with_kwargs=True)
    coupling_handle = \
        model.decoder.query_evidence_coupling.register_forward_hook(
            capture_coupling)
    try:
        eval_batch = _processed_real_batch(
            model, dataset, index, training=False)
        model.eval()
        with torch.no_grad():
            predictions = model.predict(
                eval_batch['inputs'], eval_batch['data_samples'],
                rescale=True)
    finally:
        decoder_handle.remove()
        coupling_handle.remove()
    assert decoder_calls == [((1, 600, 256), 600)]
    assert coupling_calls == [(coupling_id, (1, 600, 256))] * 6
    assert model.decoder.last_flow_validities.shape == (6, 1, 600)
    assert model.decoder.last_flow_center_deltas.shape == (6, 1, 600, 2)
    assert len(predictions[0].pred_instances) == 600


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH') != '1',
    reason='real control optimizer Stage-0 gate is opt-in')
def test_control_stays_zero_after_real_optimizer_step():
    parent = _build_real_model(BASE_CONFIG)
    control = _build_real_model(CONTROL_CONFIG)
    candidate = _build_real_model(CANDIDATE_CONFIG)
    control.decoder.query_evidence_coupling.set_update(160)
    candidate.decoder.query_evidence_coupling.set_update(160)
    dataset = _build_real_dataset()
    index = _dataset_indices_by_image_id(dataset)[REAL_BATCH_IDS[1]]
    coupling = candidate.decoder.query_evidence_coupling
    before = {name: value.detach().clone()
              for name, value in coupling.state_dict().items()}
    parameters = [parameter for parameter in candidate.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    train_batch = _processed_real_batch(
        candidate, dataset, index, training=True)
    _frozen_parent_training_mode(candidate)
    optimizer.zero_grad(set_to_none=True)
    _loss_total(candidate.loss(
        train_batch['inputs'],
        copy.deepcopy(train_batch['data_samples']))).backward()
    optimizer.step()
    after = coupling.state_dict()
    assert any(not torch.equal(before[name], after[name])
               for name in before if name != 'flow_progress')
    control.decoder.query_evidence_coupling.load_state_dict(after, strict=True)
    control_coupling = control.decoder.query_evidence_coupling
    assert control_coupling.center_strength.item() == 0.0
    assert control_coupling.score_strength.item() == 0.0

    eval_batch = _processed_real_batch(
        parent, dataset, index, training=False)
    parent.eval()
    control.eval()
    with torch.no_grad():
        parent_predictions, parent_references = _capture_references_during(
            lambda: parent.predict(
                eval_batch['inputs'],
                copy.deepcopy(eval_batch['data_samples']), rescale=True),
            parent)
        control_predictions, control_references = _capture_references_during(
            lambda: control.predict(
                eval_batch['inputs'],
                copy.deepcopy(eval_batch['data_samples']), rescale=True),
            control)
    assert torch.equal(parent_references, control_references)
    for name in ('bboxes', 'scores', 'labels'):
        assert torch.equal(
            parent_predictions[0].pred_instances[name],
            control_predictions[0].pred_instances[name])
```

Add the following report readers and the three complete opt-in promotion
tests to the same file. They deliberately consume the existing analyzer and
sampler outputs; they are test-side gate arithmetic, not a new analyzer or
launcher:

```python
def _required_env_path(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f'missing required environment variable {name}')
    path = Path(value)
    if not path.exists():
        raise ValueError(f'{name} does not exist: {path}')
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact(path):
    path = Path(path)
    return {'path': str(path), 'sha256': _sha256(path)}


def _json_object(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'JSON artifact must contain an object: {path}')
    return payload


def _finite_number(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f'missing numeric field {".".join(keys)}')
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'field {".".join(keys)} is not numeric')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'field {".".join(keys)} is not finite')
    return value


def _unique_metric_row(directory, required_keys):
    matches = []
    for path in sorted(Path(directory).rglob('*.json')):
        text = path.read_text(encoding='utf-8')
        try:
            payloads = [json.loads(text)]
        except json.JSONDecodeError:
            payloads = [json.loads(line) for line in text.splitlines()
                        if line.strip()]
        for payload in payloads:
            if (isinstance(payload, dict) and
                    all(key in payload for key in required_keys)):
                matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(
            'metric directory must contain exactly one row with {}: {}'
            .format(tuple(required_keys), directory))
    path, row = matches[0]
    for key in required_keys:
        _finite_number(row, key)
    return path, row


def _diagnostic_contract(payload, records, rows):
    if payload.get('integrity', {}).get('all_cpu_finite') is not True:
        raise ValueError('diagnostics all_cpu_finite is not true')
    if payload.get('parity', {}).get('replay_delta_warning') is not False:
        raise ValueError('diagnostics metric parity did not pass')
    expected = {
        'records': records,
        'unique_image_ids': records,
        'prediction_rows': rows,
        'queries': 600,
        'classes': 18,
    }
    for key, value in expected.items():
        if payload['integrity'].get(key) != value:
            raise ValueError(f'diagnostics integrity mismatch for {key}')
    required = (
        ('metrics', 'map'), ('groups', 'base', 'map'),
        ('groups', 'novel', 'map'),
        ('gt_decomposition', 'geometry_miss'),
        ('gt_decomposition', 'total_gt'),
        ('calibration', 'tp_score_iou_spearman'),
        ('fp_regions', 'ap_support', 'row_count'),
    )
    for keys in required:
        _finite_number(payload, *keys)
    if _finite_number(payload, 'gt_decomposition', 'total_gt') <= 0:
        raise ValueError('diagnostics total_gt must be positive')
    if _finite_number(payload, 'fp_regions', 'ap_support',
                      'row_count') <= 0:
        raise ValueError('diagnostics AP-support row_count must be positive')


def _audit_pass(path, field='pass'):
    payload = _json_object(path)
    if payload.get(field) is not True:
        raise ValueError(f'audit did not pass: {path}')
    return payload


def _publish_json_noreplace(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + '\n')
    with path.open('x', encoding='utf-8') as handle:
        handle.write(encoded)


def _gate_test(output_env, builder):
    output_value = os.getenv(output_env)
    if not output_value:
        raise ValueError(f'missing required environment variable {output_env}')
    output = Path(output_value)
    if output.exists():
        raise FileExistsError(f'decision already exists: {output}')
    try:
        decision = builder()
    except Exception as error:
        decision = {
            'schema_version': 1,
            'status': 'FAIL',
            'gates': {'artifact_validation': False},
            'failed_gates': ['artifact_validation'],
            'validation_error': f'{type(error).__name__}: {error}',
        }
    _publish_json_noreplace(output, decision)
    assert decision['status'] == 'PASS', json.dumps(
        decision, sort_keys=True, allow_nan=False)


def _build_proxy_decision():
    control_path = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_CONTROL_DIAGNOSTICS')
    candidate_path = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_CANDIDATE_DIAGNOSTICS')
    control = _json_object(control_path)
    candidate = _json_object(candidate_path)
    _diagnostic_contract(control, records=400, rows=240000)
    _diagnostic_contract(candidate, records=400, rows=240000)
    control_metric_path, control_metric = _unique_metric_row(
        _required_env_path('OVCAPFLOW_QUERY_FLOW_CONTROL_METRICS_DIR'),
        ('dota/mAP', 'dota/AP50'))
    candidate_metric_path, candidate_metric = _unique_metric_row(
        _required_env_path('OVCAPFLOW_QUERY_FLOW_CANDIDATE_METRICS_DIR'),
        ('dota/mAP', 'dota/AP50'))
    control_strict = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_CONTROL_STRICT')
    candidate_strict = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_CANDIDATE_STRICT')
    open_vocabulary = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_OPEN_VOCAB')
    _audit_pass(control_strict)
    _audit_pass(candidate_strict)
    _audit_pass(open_vocabulary)

    values = {
        'delta_map': (_finite_number(candidate, 'metrics', 'map') -
                      _finite_number(control, 'metrics', 'map')),
        'delta_ap50': (_finite_number(candidate_metric, 'dota/AP50') -
                       _finite_number(control_metric, 'dota/AP50')),
        'geometry_drop': (
            _finite_number(control, 'gt_decomposition', 'geometry_miss') /
            _finite_number(control, 'gt_decomposition', 'total_gt') -
            _finite_number(candidate, 'gt_decomposition', 'geometry_miss') /
            _finite_number(candidate, 'gt_decomposition', 'total_gt')),
        'spearman_gain': (
            _finite_number(candidate, 'calibration',
                           'tp_score_iou_spearman') -
            _finite_number(control, 'calibration',
                           'tp_score_iou_spearman')),
        'base_delta': (_finite_number(candidate, 'groups', 'base', 'map') -
                       _finite_number(control, 'groups', 'base', 'map')),
        'novel_delta': (
            _finite_number(candidate, 'groups', 'novel', 'map') -
            _finite_number(control, 'groups', 'novel', 'map')),
    }
    gates = {
        'delta_map_ge_0.010': values['delta_map'] >= 0.010,
        'delta_ap50_ge_0.010': values['delta_ap50'] >= 0.010,
        'geometry_drop_ge_0.020': values['geometry_drop'] >= 0.020,
        'spearman_gain_ge_0.05': values['spearman_gain'] >= 0.05,
        'base_delta_ge_-0.005': values['base_delta'] >= -0.005,
        'novel_delta_ge_0': values['novel_delta'] >= 0.0,
        'strict_and_open_vocabulary': True,
        'diagnostic_integrity': True,
    }
    values['fp_relative_worsening'] = {}
    for region in ('empty_tile_fp', 'duplicate_fp',
                   'localization_background_fp'):
        control_count = _finite_number(
            control, 'fp_regions', 'ap_support', region)
        candidate_count = _finite_number(
            candidate, 'fp_regions', 'ap_support', region)
        control_rows = _finite_number(
            control, 'fp_regions', 'ap_support', 'row_count')
        candidate_rows = _finite_number(
            candidate, 'fp_regions', 'ap_support', 'row_count')
        if control_count == 0:
            worsening = None
            passed = candidate_count == 0
        else:
            worsening = ((candidate_count / candidate_rows) /
                          (control_count / control_rows) - 1.0)
            passed = worsening <= 0.05
        values['fp_relative_worsening'][region] = worsening
        gates[f'{region}_relative_worsening_le_0.05'] = passed
    failed = [name for name, passed in gates.items() if not passed]
    return {
        'schema_version': 1,
        'gate': 'query-flow-proxy-e12',
        'status': 'PASS' if not failed else 'FAIL',
        'values': values,
        'gates': gates,
        'failed_gates': failed,
        'artifacts': {
            'control_diagnostics': _artifact(control_path),
            'candidate_diagnostics': _artifact(candidate_path),
            'control_metrics': _artifact(control_metric_path),
            'candidate_metrics': _artifact(candidate_metric_path),
            'control_strict': _artifact(control_strict),
            'candidate_strict': _artifact(candidate_strict),
            'open_vocabulary': _artifact(open_vocabulary),
        },
    }


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_PROXY_GATE') != '1',
    reason='query-flow proxy promotion gate is opt-in')
def test_proxy_promotion_gate_from_frozen_reports():
    _gate_test('OVCAPFLOW_QUERY_FLOW_PROXY_DECISION',
               _build_proxy_decision)


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_FULL_RECIPE') != '1',
    reason='query-flow full sampler gate is opt-in')
def test_full_recipe_world10_sampler_contract():
    manifest = Path(
        'work_dirs/dotav2_cleanstart/full_rare4x_seed20260718/'
        'train/manifest.json')
    assert _sha256(manifest) == (
        '152d1244778ecfab454cca459f8cfec03a9927b8894a9eb4d48ec1a9c5cc4ce6')
    cfg = Config.fromfile(str(CANDIDATE_CONFIG))
    cfg.merge_from_dict({
        'selected_world_size': 10,
        'physical_gpus': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        'query_flow_master_port': 29920,
        'query_flow_updates_per_epoch': 2365,
        'query_flow_total_updates': 28380,
        'train_dataloader.dataset.data_root': (
            '/data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/'
            'full_rare4x_seed20260718/train/'),
        'train_dataloader.dataset.ann_file': 'annfiles/',
        'train_dataloader.dataset.data_prefix.img_path': 'images/',
        'train_dataloader.batch_size': 2,
        'train_dataloader.batch_sampler.update_count_multiple': 1,
        'train_dataloader.batch_sampler.audit_path': (
            'work_dirs/dotav2_cleanstart/audits/'
            'query_flow_full_candidate_world10_epoch_{epoch:02d}.json'),
        'train_cfg.max_epochs': 12,
        'train_cfg.val_interval': 12,
        'val_dataloader.dataset.data_root': (
            '/data1/zcy/datasets/DOTA2_1024_500/'),
        'val_dataloader.dataset.ann_file': 'ss_val/annfiles/',
        'val_dataloader.dataset.data_prefix.img_path': 'ss_val/images/',
        'val_dataloader.dataset.filter_cfg.filter_empty_gt': False,
        'val_dataloader.dataset.test_mode': True,
    })
    assert cfg.selected_world_size == 10
    assert cfg.train_dataloader.batch_size == 2
    assert cfg.query_flow_updates_per_epoch == 2365
    assert cfg.query_flow_total_updates == 28380
    assert cfg.train_cfg.max_epochs == 12
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    assert len(dataset) == 47294
    report = _build_sampler_coverage_report(
        dataset=dataset,
        batch_size=cfg.train_dataloader.batch_size,
        batch_sampler_cfg=cfg.train_dataloader.batch_sampler,
        seed=cfg.randomness.seed,
        epoch=0,
        world_size=cfg.selected_world_size)
    assert report['dataset_size'] == 47294
    assert report['update_count'] == 2365
    assert report['duplicate_count'] == 0
    assert report['missing_count'] == 0
    assert report['global_batch_size_min'] == 14
    assert report['global_batch_size_max'] == 20
    report.update({
        'manifest': str(manifest),
        'manifest_sha256': _sha256(manifest),
        'world_size': 10,
        'batch_size_per_rank': 2,
        'max_epochs': 12,
        'total_updates': 28380,
    })
    output_value = os.getenv('OVCAPFLOW_QUERY_FLOW_FULL_SAMPLER_REPORT')
    if not output_value:
        raise ValueError(
            'missing OVCAPFLOW_QUERY_FLOW_FULL_SAMPLER_REPORT')
    output = Path(output_value).absolute()
    if output.exists():
        raise FileExistsError(f'sampler report already exists: {output}')
    _publish_json_noreplace(output, report)


def _strata_rows(path):
    selected = {}
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {
            'stratum_type', 'stratum', 'class_id', 'gt_count',
            'geometry_miss_rate', 'evaluator_reachable_rate'}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f'strata schema mismatch: {path}')
        for row in reader:
            if row['stratum_type'] not in {
                    'density', 'size', 'angle', 'aspect'}:
                continue
            key = (row['stratum_type'], row['stratum'], row['class_id'])
            if key in selected:
                raise ValueError(f'duplicate stratum key: {key}')
            try:
                gt_count = int(row['gt_count'])
                geometry_rate = float(row['geometry_miss_rate'])
                reachable_rate = float(row['evaluator_reachable_rate'])
            except (TypeError, ValueError) as error:
                raise ValueError(f'invalid stratum numerics: {key}') from error
            if (gt_count <= 0 or not math.isfinite(geometry_rate) or
                    not math.isfinite(reachable_rate)):
                raise ValueError(f'non-finite/zero stratum denominator: {key}')
            selected[key] = {
                'gt_count': gt_count,
                'geometry_miss_rate': geometry_rate,
                'evaluator_reachable_rate': reachable_rate,
            }
    if not selected:
        raise ValueError(f'no registered strata in {path}')
    return selected


def _build_raw_decision():
    parent_path = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_PARENT_DIAGNOSTICS')
    candidate_path = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_CANDIDATE_DIAGNOSTICS')
    parent = _json_object(parent_path)
    candidate = _json_object(candidate_path)
    _diagnostic_contract(parent, records=13833, rows=8299800)
    _diagnostic_contract(candidate, records=13833, rows=8299800)
    metric_keys = ('dota/mAP', 'dota/AP50',
                   'dota75/mAP', 'dota75/AP75')
    parent_metric_path, parent_metric = _unique_metric_row(
        _required_env_path('OVCAPFLOW_QUERY_FLOW_PARENT_METRICS_DIR'),
        metric_keys)
    candidate_metric_path, candidate_metric = _unique_metric_row(
        _required_env_path('OVCAPFLOW_QUERY_FLOW_CANDIDATE_METRICS_DIR'),
        metric_keys)
    parent_strata_path = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_PARENT_STRATA')
    candidate_strata_path = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_CANDIDATE_STRATA')
    parent_strata = _strata_rows(parent_strata_path)
    candidate_strata = _strata_rows(candidate_strata_path)
    if set(parent_strata) != set(candidate_strata):
        raise ValueError('parent/candidate stratum keys differ')
    underpowered = []
    stratum_decisions = []
    strata_pass = True
    for key in sorted(parent_strata):
        parent_row = parent_strata[key]
        candidate_row = candidate_strata[key]
        if candidate_row['gt_count'] != parent_row['gt_count']:
            raise ValueError(f'parent/candidate stratum counts differ: {key}')
        if parent_row['gt_count'] < 100:
            underpowered.append({'key': key, **parent_row})
            continue
        reachable_delta = (
            candidate_row['evaluator_reachable_rate'] -
            parent_row['evaluator_reachable_rate'])
        geometry_delta = (
            candidate_row['geometry_miss_rate'] -
            parent_row['geometry_miss_rate'])
        passed = reachable_delta >= -0.010 and geometry_delta <= 0.010
        strata_pass = strata_pass and passed
        stratum_decisions.append({
            'key': key,
            'gt_count': parent_row['gt_count'],
            'reachable_delta': reachable_delta,
            'geometry_miss_delta': geometry_delta,
            'pass': passed,
        })

    mouth_path = _required_env_path('OVCAPFLOW_QUERY_FLOW_RAW_MOUTH')
    strict_path = _required_env_path('OVCAPFLOW_QUERY_FLOW_RAW_STRICT')
    open_vocabulary_path = _required_env_path(
        'OVCAPFLOW_QUERY_FLOW_RAW_OPEN_VOCAB')
    mouth = _audit_pass(mouth_path, field='valid')
    _audit_pass(strict_path)
    _audit_pass(open_vocabulary_path)
    if (mouth.get('train_image_count') != 47294 or
            mouth.get('train_annotation_count') != 47294 or
            mouth.get('val_image_count') != 13833 or
            mouth.get('val_annotation_count') != 13833):
        raise ValueError('raw mouth counts do not match 47294/13833')

    parent_geometry = (
        _finite_number(parent, 'gt_decomposition', 'geometry_miss') /
        _finite_number(parent, 'gt_decomposition', 'total_gt'))
    candidate_geometry = (
        _finite_number(candidate, 'gt_decomposition', 'geometry_miss') /
        _finite_number(candidate, 'gt_decomposition', 'total_gt'))
    values = {
        'parent_map': _finite_number(parent, 'metrics', 'map'),
        'candidate_map': _finite_number(candidate, 'metrics', 'map'),
        'parent_ap50': _finite_number(parent_metric, 'dota/AP50'),
        'candidate_ap50': _finite_number(candidate_metric, 'dota/AP50'),
        'parent_ap75': _finite_number(parent_metric, 'dota75/AP75'),
        'candidate_ap75': _finite_number(candidate_metric, 'dota75/AP75'),
        'base_delta': (
            _finite_number(candidate, 'groups', 'base', 'map') -
            _finite_number(parent, 'groups', 'base', 'map')),
        'novel_delta': (
            _finite_number(candidate, 'groups', 'novel', 'map') -
            _finite_number(parent, 'groups', 'novel', 'map')),
        'parent_geometry_miss_rate': parent_geometry,
        'candidate_geometry_miss_rate': candidate_geometry,
        'parent_tp_score_iou_spearman': _finite_number(
            parent, 'calibration', 'tp_score_iou_spearman'),
        'candidate_tp_score_iou_spearman': _finite_number(
            candidate, 'calibration', 'tp_score_iou_spearman'),
        'strata': stratum_decisions,
        'underpowered_strata': underpowered,
    }
    gates = {
        'parent_map_anchor': abs(
            values['parent_map'] - 0.6064053488274416) <= 1e-12,
        'parent_ap50_anchor': values['parent_ap50'] == 0.606,
        'candidate_map_ge_0.6114053488': (
            values['candidate_map'] >= 0.6114053488),
        'candidate_ap50_ge_0.6110': values['candidate_ap50'] >= 0.6110,
        'base_delta_ge_-0.001': values['base_delta'] >= -0.001,
        'novel_delta_ge_-0.001': values['novel_delta'] >= -0.001,
        'ap75_strictly_improves': (
            values['candidate_ap75'] > values['parent_ap75']),
        'geometry_miss_rate_strictly_improves': (
            values['candidate_geometry_miss_rate'] <
            values['parent_geometry_miss_rate']),
        'spearman_strictly_improves': (
            values['candidate_tp_score_iou_spearman'] >
            values['parent_tp_score_iou_spearman']),
        'powered_strata_within_tolerance': strata_pass,
        'mouth_strict_open_vocabulary_integrity': True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        'schema_version': 1,
        'gate': 'query-flow-raw-e12',
        'status': 'PASS' if not failed else 'FAIL',
        'values': values,
        'gates': gates,
        'failed_gates': failed,
        'artifacts': {
            'parent_diagnostics': _artifact(parent_path),
            'candidate_diagnostics': _artifact(candidate_path),
            'parent_strata': _artifact(parent_strata_path),
            'candidate_strata': _artifact(candidate_strata_path),
            'parent_metrics': _artifact(parent_metric_path),
            'candidate_metrics': _artifact(candidate_metric_path),
            'mouth': _artifact(mouth_path),
            'strict': _artifact(strict_path),
            'open_vocabulary': _artifact(open_vocabulary_path),
        },
    }


@pytest.mark.skipif(
    os.getenv('OVCAPFLOW_RUN_QUERY_FLOW_RAW_GATE') != '1',
    reason='query-flow raw promotion gate is opt-in')
def test_raw_promotion_gate_from_frozen_reports():
    _gate_test('OVCAPFLOW_QUERY_FLOW_RAW_DECISION', _build_raw_decision)
```

The constants above are the only model/config/checkpoint paths used by these
tests; no path is supplied implicitly by the shell environment.

The classification branch pads its last dimension to `max_text_len`, so slicing to the runtime mask length before masking is mandatory. Validate `full_token_logits [B,S_all,max_text_len]`, `token_logits [B,S_all,T]`, `flow_semantic_evidence [B,S_all]`, finiteness on unmasked rows, and at least one valid text token. Padded columns `T:max_text_len` never enter the reduction. Add `flow_semantic_evidence` to `decoder_inputs_dict`; do not put it in `head_inputs_dict` as an independent prediction.

Extend `_append_calibration_state` to pass through the decoder's cached `flow_validities`, already strength-scaled `flow_center_deltas`, and detached scalar `flow_progress`. It also calls `self.decoder.query_evidence_coupling.score_log_residual(flow_validities[-1])` once and passes the resulting `[B,M]` `flow_score_log_residual` to the head. The head never reaches back into the decoder module. Existing null/capacity/semantic paths stay behaviorally unchanged when query flow is disabled. Configs in this plan explicitly disable old null, density, semantic-fusion, balanced, position-supervised, adaptive-DN, and D13 existence branches.

- [ ] **Step 3: Replace only the decoder loop necessary for in-layer transport**

In `OVCapFlowDecoder.__init__`, copy `query_flow_cfg`, pop and validate `enabled` and `level_index`, store `level_index == 2` on the decoder, and pass only the constructor fields listed in Task 1 to one `QueryEvidenceCoupling`. Do not forward control keys the constructor does not accept and do not build one coupling per layer.

Override the existing parent loop in `OVCapFlowDecoder.forward` using the upstream RHINO loop as the source of truth. Preserve its operation order, detach behavior, angle convention, valid ratios, intermediate-state stack, and parent reference update. Add `_prepare_query_flow_level(self, value, key_padding_mask, spatial_shapes, level_start_index, flow_semantic_evidence)` containing this exact body, and call it exactly once before the loop:

```python
def _prepare_query_flow_level(self, value, key_padding_mask, spatial_shapes,
                              level_start_index, flow_semantic_evidence):
    level_id = self.query_flow_level_index
    height = int(spatial_shapes[level_id, 0].item())
    width = int(spatial_shapes[level_id, 1].item())
    level_start = int(level_start_index[level_id].item())
    level_stop = level_start + height * width
    if level_id + 1 < spatial_shapes.shape[0]:
        if level_stop != int(level_start_index[level_id + 1].item()):
            raise ValueError('query-flow level offsets disagree')
    if (value.ndim != 3 or value.shape[2] != self.embed_dims or
            level_stop > value.shape[1]):
        raise ValueError('query-flow value/level shape mismatch')
    if key_padding_mask is None:
        key_padding_mask = torch.zeros(
            value.shape[:2], dtype=torch.bool, device=value.device)
    elif (key_padding_mask.ndim != 2 or
            key_padding_mask.shape != value.shape[:2]):
        raise ValueError('query-flow padding mask shape mismatch')
    if flow_semantic_evidence.shape != value.shape[:2]:
        raise ValueError('query-flow evidence must cover all encoder tokens')
    level_memory = value[:, level_start:level_stop, :]
    level_semantic_evidence = flow_semantic_evidence[:, level_start:level_stop]
    level_padding_mask = key_padding_mask[:, level_start:level_stop]
    level_centers, level_valid_mask = \
        self.query_evidence_coupling.level_token_geometry(
            (height, width), level_padding_mask)
    if (level_memory.shape[1] != height * width or
            level_semantic_evidence.shape != level_valid_mask.shape or
            level_centers.shape != level_valid_mask.shape + (2,)):
        raise ValueError('query-flow level tensors have inconsistent shapes')
    return (level_memory, level_semantic_evidence,
            level_centers, level_valid_mask)


level_memory, level_semantic_evidence, level_centers, level_valid_mask = \
    self._prepare_query_flow_level(
        value, key_padding_mask, spatial_shapes, level_start_index,
        flow_semantic_evidence)
```

Here `value`, `key_padding_mask`, `spatial_shapes`, and `level_start_index` are the unchanged arguments already received by the upstream decoder. `flow_semantic_evidence` is the detector value added in Step 2. The raster centers are normalized by each sample's valid width/height, invalid padded cells remain masked, and no GT enters this construction.

The only inserted sequence after each parent reference update is:

```python
matching_query = query[:, -matching_query_count:, :]
matching_ref = new_reference[:, -matching_query_count:, :]
flow = self.query_evidence_coupling(
    matching_query, matching_ref, level_memory,
    level_semantic_evidence, level_centers, level_valid_mask)
transported_matching_ref, applied_center_delta = \
    self.query_evidence_coupling.apply_center_transport(
    matching_ref, flow.center_delta)
new_reference = torch.cat(
    [new_reference[:, :-matching_query_count, :],
     transported_matching_ref], dim=1)
reference_points = new_reference.detach()
if self.return_intermediate:
    intermediate.append(self.norm(query))
    intermediate_reference_points.append(new_reference)
```

This sequence replaces upstream lines that detach and append the untransported `new_reference_points`; do not execute both versions. `reference_points = new_reference.detach()` is what makes layer `l+1` consume the transported center, while `intermediate_reference_points.append(new_reference)` is what exposes the same transported reference to the head stack. At inference `matching_query_count=600`; at training it is exactly 1,800 and the DN prefix is copied without arithmetic. `apply_center_transport` applies the deterministic strength, clamps transported centers to `[0,1]`, and returns `transported_center - parent_center` as the actual applied delta. Cache that exact applied delta for the same-layer head. If the center strength is exactly zero, it returns the original parent tensor plus an exact-zero delta and the final assembled reference must use an explicit zero-route branch, not add-zero arithmetic.

Add `inverse_sigmoid` and `coordinate_to_encoding` to the existing layer
imports and replace the current decoder constructor/forward wrapper with the
following complete implementation. The disabled branch is the existing
parent call byte-for-byte; the enabled branch is the RHINO loop with only the
registered flow insertion:

```python
from mmdet.models.layers.transformer import inverse_sigmoid
from mmrotate.models.layers.transformer.utils import coordinate_to_encoding

from .query_evidence_coupling import QueryEvidenceCoupling


class OVCapFlowDecoder(GroundingDinoTransformerDecoder):
    def __init__(self, enable_null_reservoir=False,
                 null_reservoir_cfg=None, query_flow_cfg=None, **kwargs):
        flow_cfg = dict(query_flow_cfg or {})
        enabled = flow_cfg.pop('enabled', False)
        level_index = flow_cfg.pop('level_index', 2)
        if type(enabled) is not bool:
            raise TypeError('query_flow_cfg.enabled must be bool')
        if type(level_index) is not int or level_index != 2:
            raise ValueError('query flow freezes level_index=2')
        super().__init__(**kwargs)
        self.enable_null_reservoir = bool(enable_null_reservoir)
        null_reservoir_cfg = dict(null_reservoir_cfg or {})
        self.null_reservoir = (
            ExplicitNullReservoir(
                embed_dims=self.embed_dims, **null_reservoir_cfg)
            if self.enable_null_reservoir else None)
        self.query_flow_level_index = level_index
        self.query_evidence_coupling = (
            QueryEvidenceCoupling(**flow_cfg) if enabled else None)
        self.last_null_logits = None
        self.last_flow_validities = None
        self.last_flow_center_deltas = None

    def _prepare_query_flow_level(self, value, key_padding_mask,
                                  spatial_shapes, level_start_index,
                                  flow_semantic_evidence):
        level_id = self.query_flow_level_index
        height = int(spatial_shapes[level_id, 0].item())
        width = int(spatial_shapes[level_id, 1].item())
        level_start = int(level_start_index[level_id].item())
        level_stop = level_start + height * width
        if level_id + 1 < spatial_shapes.shape[0]:
            if level_stop != int(level_start_index[level_id + 1].item()):
                raise ValueError('query-flow level offsets disagree')
        if (value.ndim != 3 or value.shape[2] != self.embed_dims or
                level_stop > value.shape[1]):
            raise ValueError('query-flow value/level shape mismatch')
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(
                value.shape[:2], dtype=torch.bool, device=value.device)
        elif (key_padding_mask.ndim != 2 or
              key_padding_mask.shape != value.shape[:2]):
            raise ValueError('query-flow padding mask shape mismatch')
        if flow_semantic_evidence.shape != value.shape[:2]:
            raise ValueError('query-flow evidence must cover all tokens')
        level_memory = value[:, level_start:level_stop, :]
        level_semantic_evidence = \
            flow_semantic_evidence[:, level_start:level_stop]
        level_padding_mask = key_padding_mask[:, level_start:level_stop]
        level_centers, level_valid_mask = \
            self.query_evidence_coupling.level_token_geometry(
                (height, width), level_padding_mask)
        if (level_memory.shape[1] != height * width or
                level_semantic_evidence.shape != level_valid_mask.shape or
                level_centers.shape != level_valid_mask.shape + (2,)):
            raise ValueError('query-flow level tensors disagree')
        return (level_memory, level_semantic_evidence,
                level_centers, level_valid_mask)

    def forward(self, query, value, key_padding_mask, self_attn_mask,
                reference_points, spatial_shapes, level_start_index,
                valid_ratios, reg_branches, return_sampling_results=False,
                matching_query_count=None, flow_semantic_evidence=None,
                **kwargs):
        if self.query_evidence_coupling is None:
            outputs = super().forward(
                query=query, value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reg_branches=reg_branches,
                return_sampling_results=return_sampling_results,
                matching_query_count=matching_query_count, **kwargs)
            inter_states = outputs[0] if return_sampling_results else outputs[0]
            self.last_flow_validities = None
            self.last_flow_center_deltas = None
            self.last_null_logits = None
            if self.null_reservoir is not None:
                if matching_query_count is None:
                    raise ValueError(
                        'null reservoir requires matching query count')
                self.last_null_logits = self.null_reservoir(
                    inter_states[-1][:, -matching_query_count:])
            return outputs

        if (matching_query_count is None or
                matching_query_count <= 0 or
                matching_query_count > query.shape[1]):
            raise ValueError('query flow requires a valid matching suffix')
        if flow_semantic_evidence is None:
            raise ValueError('query flow requires semantic evidence')
        (level_memory, level_semantic_evidence,
         level_centers, level_valid_mask) = self._prepare_query_flow_level(
             value, key_padding_mask, spatial_shapes, level_start_index,
             flow_semantic_evidence)

        intermediate = []
        intermediate_reference_points = [reference_points]
        flow_validities = []
        flow_center_deltas = []
        if return_sampling_results:
            sampling_locations = []
            sampling_offsets = []

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 5:
                dummy_angle = torch.ones_like(
                    valid_ratios[..., :1]) * self.angle_factor
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios, dummy_angle], -1)[:, None])
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError('decoder reference must have 2 or 5 dims')
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None])
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :4])
            query_pos = self.ref_point_head(query_sine_embed)
            query = layer(
                query, query_pos=query_pos, value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                return_sampling_results=return_sampling_results,
                matching_query_count=matching_query_count, **kwargs)
            if return_sampling_results:
                sampling_locations.append(layer.cross_attn.sampling_locs)
                sampling_offsets.append(layer.cross_attn.sampling_offs)
            if reg_branches is None:
                raise ValueError('query flow requires refinement branches')
            parent_new_reference = (
                reg_branches[layer_id](query) +
                inverse_sigmoid(reference_points, eps=1e-3)).sigmoid()
            matching_query = query[:, -matching_query_count:, :]
            matching_reference = \
                parent_new_reference[:, -matching_query_count:, :]
            flow = self.query_evidence_coupling(
                matching_query, matching_reference, level_memory,
                level_semantic_evidence, level_centers, level_valid_mask)
            transported, applied_delta = \
                self.query_evidence_coupling.apply_center_transport(
                    matching_reference, flow.center_delta)
            if self.query_evidence_coupling.center_strength.item() == 0.0:
                new_reference = parent_new_reference
            else:
                prefix = parent_new_reference[:, :-matching_query_count, :]
                new_reference = (transported if prefix.shape[1] == 0 else
                                 torch.cat([prefix, transported], dim=1))
            reference_points = new_reference.detach()
            flow_validities.append(flow.validity)
            flow_center_deltas.append(applied_delta)
            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference)

        self.last_flow_validities = torch.stack(flow_validities)
        self.last_flow_center_deltas = torch.stack(flow_center_deltas)
        self.last_null_logits = None
        if self.null_reservoir is not None:
            self.last_null_logits = self.null_reservoir(
                intermediate[-1][:, -matching_query_count:])
        if self.return_intermediate:
            states = torch.stack(intermediate)
            references = torch.stack(intermediate_reference_points)
            if return_sampling_results:
                return (states, references,
                        sampling_locations, sampling_offsets)
            return states, references
        return query, reference_points
```

- [ ] **Step 4: Make the sixth-layer center delta observable in the existing head**

The parent head layer `l` consumes `references[l]`, while decoder transport creates the reference used by layer `l+1`. Add `flow_center_deltas` to the head `forward`, `loss`, and `predict` signatures. After the unchanged parent bbox branch output for each layer, add the same layer's gated matching delta to only the last `M` box centers. This makes layer 6 affect final boxes and keeps width/height/angle untouched.

Use an explicit zero-strength branch returning the parent bbox tensor unchanged. Never add the delta twice to the next layer: transported references affect refinement state; same-layer head addition affects that layer's emitted box only.

Add `inverse_sigmoid` to the head imports and replace `forward` with this
complete method. It retains the existing text classifier and regression order:

```python
def forward(self, hidden_states, references, memory_text, text_token_mask,
            flow_center_deltas=None, flow_center_strength=None):
    all_layers_outputs_classes = []
    all_layers_outputs_coords = []
    if flow_center_deltas is not None:
        if (flow_center_strength is None or
                flow_center_deltas.ndim != 4 or
                flow_center_deltas.shape[0] != hidden_states.shape[0] or
                flow_center_deltas.shape[1] != hidden_states.shape[1] or
                flow_center_deltas.shape[-1] != 2):
            raise ValueError('flow center-delta stack has invalid shape')
        if not torch.isfinite(flow_center_deltas).all().item():
            raise ValueError('flow center deltas must be finite')
    for layer_id in range(hidden_states.shape[0]):
        reference = inverse_sigmoid(references[layer_id])
        hidden_state = hidden_states[layer_id]
        outputs_class = self.cls_branches[layer_id](
            hidden_state, memory_text, text_token_mask)
        tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
        if reference.shape[-1] not in (4, 5):
            raise NotImplementedError()
        tmp_reg_preds += reference
        parent_coord = tmp_reg_preds.sigmoid()
        if (flow_center_deltas is None or
                flow_center_strength.detach().item() == 0.0):
            outputs_coord = parent_coord
        else:
            matching_count = flow_center_deltas.shape[2]
            if matching_count > parent_coord.shape[1]:
                raise ValueError('flow delta exceeds the matching suffix')
            outputs_coord = parent_coord.clone()
            outputs_coord[:, -matching_count:, :2] = (
                parent_coord[:, -matching_count:, :2] +
                flow_center_deltas[layer_id]).clamp(0, 1)
        all_layers_outputs_classes.append(outputs_class)
        all_layers_outputs_coords.append(outputs_coord)
    return (torch.stack(all_layers_outputs_classes),
            torch.stack(all_layers_outputs_coords))
```

- [ ] **Step 5: Thread final validity through final-layer matching and natural loss**

Add `query_flow_loss_cfg` to `OVCapFlowHead.__init__`, accepting only:

```python
{
    'enabled': bool,
    'loss_weight': 0.0 or 1.0,
    'match_weight': float in [0, 0.5],
    'match_warmup_updates': 160,
    'eps': 1e-6,
}
```

Never modify `self.assigner`: even appending an all-zero cost changes stack/reduction order and can break tie-level exactness. When query flow is enabled, build `self.flow_assigner = copy.deepcopy(self.assigner)` and append one stateless `QueryValidityCost` only to that private copy. `QueryValidityCost.__call__` requires the attached `[Q]` `flow_match_cost` and repeats it over GT columns; a missing field is an error. The control, every auxiliary layer, and candidate update 0 call the original parent assigner. Only a candidate final-layer call with `current_match_weight > 0` calls `self.flow_assigner`.

Implement explicit, argument-driven helpers—never a mutable queue or implicit call-order context:

- `get_targets(cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas, flow_match_costs: Optional[List[Tensor]] = None, capture_flow_targets: bool = False)`;
- `_get_targets_single(cls_score, bbox_pred, gt_instances, img_meta, flow_match_cost: Optional[Tensor] = None)`;
- `_flow_matching_loss_by_feat(cls_scores, bbox_preds, final_validity, batch_gt_instances, batch_img_metas)`.

`_flow_matching_loss_by_feat` reproduces the current parent loss-key/order contract. At `current_match_weight == 0`, it calls the original final-layer parent path and separately reuses those parent targets for the candidate validity objective; at positive weight it calls the private flow assigner for the final layer only. Every auxiliary layer calls the original `loss_by_feat_single`. In the existing three-group branch, split `flow_validities[-1]` with the same `split_matching_groups` operation as class/box tensors and pass each group's `[B,600]` final validity explicitly to that group's final loss. No cost or target is inferred from tensor pointers, global state, or iteration order.

For the final layer only, attach

```python
pred_instances.flow_match_cost = (
    -current_match_weight * torch.log(
        validity.detach().clamp(min=1e-6, max=1.0)))
```

before assignment. `current_match_weight` is `flow_progress*0.5` for the candidate and exactly zero for the control. It is detached and constant across GT columns. All auxiliary decoder assignments use the unchanged parent cost.

Capture final bbox targets/weights from each of the three actual group assignments. Build soft IoU targets from final predicted boxes and those targets, reshape to `[B,3,600]`, and compute one natural all-row BCE from the same final validity. Add detached telemetry for positive, negative, empty-image, and all-matched strata without reweighting. Parent losses retain their original keys and values.

Add `copy`, `inverse_sigmoid`, `reduce_mean`,
`average_matching_loss_dicts`, and the three query-flow helpers to the head
imports:

```python
import copy
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.losses import QualityFocalLoss
from mmdet.utils import reduce_mean

from .grouped_queries import average_matching_loss_dicts
from .query_evidence_coupling import (
    QueryValidityCost,
    aligned_rotated_iou_validity_targets,
    natural_validity_loss,
)
```

Accept `query_flow_loss_cfg=None` in `OVCapFlowHead.__init__`. Immediately
before its existing `super().__init__(**kwargs)` call, run the first block;
immediately after that call, run the second. These are the only constructor
insertions:

```python
flow_cfg = dict(query_flow_loss_cfg or {})
allowed = {'enabled', 'loss_weight', 'match_weight',
           'match_warmup_updates', 'eps'}
unknown = set(flow_cfg) - allowed
if unknown:
    raise ValueError(f'unknown query-flow loss key: {sorted(unknown)[0]}')
self.query_flow_loss_cfg = {
    'enabled': flow_cfg.get('enabled', False),
    'loss_weight': flow_cfg.get('loss_weight', 0.0),
    'match_weight': flow_cfg.get('match_weight', 0.0),
    'match_warmup_updates': flow_cfg.get('match_warmup_updates', 160),
    'eps': flow_cfg.get('eps', 1e-6),
}
if type(self.query_flow_loss_cfg['enabled']) is not bool:
    raise TypeError('query-flow loss enabled must be bool')
if self.query_flow_loss_cfg['loss_weight'] not in (0.0, 1.0):
    raise ValueError('query-flow loss weight must be 0 or 1')
if not 0.0 <= self.query_flow_loss_cfg['match_weight'] <= 0.5:
    raise ValueError('query-flow match weight must lie in [0,0.5]')
if self.query_flow_loss_cfg['match_warmup_updates'] != 160:
    raise ValueError('query-flow match warmup must be 160 updates')
if self.query_flow_loss_cfg['eps'] != 1e-6:
    raise ValueError('query-flow loss eps must be 1e-6')
```

```python
self.flow_assigner = None
self.last_flow_target_masks = None
self.last_flow_validity_targets = None
if self.query_flow_loss_cfg['enabled']:
    if (self.balanced_cfg.get('enabled', False) or
            self.position_supervised_enabled or
            self.adaptive_dn_enabled or self._existence_enabled):
        raise ValueError('query flow excludes prior adapter branches')
    self.flow_assigner = copy.deepcopy(self.assigner)
    self.flow_assigner.match_costs.append(
        TASK_UTILS.build(dict(type='QueryValidityCost')))
```

Replace `get_targets` and `_get_targets_single` with the following complete
argument-driven versions. The `flow_match_cost=None` branch delegates to the
existing parent and therefore keeps every auxiliary/control assignment
unchanged:

```python
def get_targets(self, cls_scores_list, bbox_preds_list,
                batch_gt_instances, batch_img_metas,
                flow_match_costs=None, capture_flow_targets=False):
    batch_size = len(cls_scores_list)
    if not (len(bbox_preds_list) == len(batch_gt_instances) ==
            len(batch_img_metas) == batch_size):
        raise ValueError('target batch inputs have different lengths')
    if flow_match_costs is None:
        flow_match_costs = [None] * batch_size
    if len(flow_match_costs) != batch_size:
        raise ValueError('one flow cost vector is required per image')
    rows = [
        self._get_targets_single(cls_score, bbox_pred, gt_instances,
                                 img_meta, flow_match_cost)
        for cls_score, bbox_pred, gt_instances, img_meta, flow_match_cost in
        zip(cls_scores_list, bbox_preds_list, batch_gt_instances,
            batch_img_metas, flow_match_costs)
    ]
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
     pos_inds_list, neg_inds_list) = map(list, zip(*rows))
    num_total_pos = sum(indices.numel() for indices in pos_inds_list)
    num_total_neg = sum(indices.numel() for indices in neg_inds_list)
    if (self._capture_matching_targets and
            not self._suppress_matching_target_capture):
        self._captured_matching_mask = torch.stack([
            weights.any(dim=-1) for weights in bbox_weights_list
        ]).detach()
    if capture_flow_targets:
        if any(cost is not None for cost in flow_match_costs) and \
                self.flow_assigner is None:
            raise RuntimeError('flow target capture requires flow assigner')
        self.last_flow_validity_targets = torch.stack(
            bbox_targets_list).detach()
        self.last_flow_target_masks = torch.stack([
            weights.any(dim=-1) for weights in bbox_weights_list
        ]).detach()
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def _get_targets_single(self, cls_score, bbox_pred, gt_instances, img_meta,
                        flow_match_cost=None):
    if flow_match_cost is None:
        parent_targets = super()._get_targets_single(
            cls_score, bbox_pred, gt_instances, img_meta)
    else:
        if (flow_match_cost.ndim != 1 or
                flow_match_cost.shape[0] != bbox_pred.shape[0] or
                not torch.isfinite(flow_match_cost).all().item()):
            raise ValueError('flow match cost must be finite [Q]')
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor(
            [img_w, img_h, img_w, img_h,
             self.angle_factor]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        absolute_bbox_pred = bbox_pred * factor
        gt_instances.bboxes.regularize_boxes(**self.angle_cfg)
        pred_instances = InstanceData(
            scores=cls_score, bboxes=absolute_bbox_pred,
            flow_match_cost=flow_match_cost)
        assign_result = self.flow_assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances, img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes.tensor
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        assigned = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[assigned.long(), :]
        labels = gt_bboxes.new_full(
            (num_bboxes, self.max_text_len), 0, dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[assigned]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(
            absolute_bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(
            absolute_bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = pos_gt_bboxes / factor
        parent_targets = (labels, label_weights, bbox_targets, bbox_weights,
                          pos_inds, neg_inds)
    if not self.position_supervised_enabled:
        return parent_targets
    (labels, label_weights, bbox_targets, bbox_weights,
     pos_inds, neg_inds) = parent_targets
    labels = scale_positive_maps_by_rotated_iou(
        labels, bbox_pred, bbox_targets, pos_inds, img_meta,
        self.angle_factor)
    return (labels, label_weights, bbox_targets, bbox_weights,
            pos_inds, neg_inds)
```

Add these complete loss helpers. `_loss_from_targets` is the existing rotated
head arithmetic with its operation order retained; the flow helper changes
only the final assignment call:

```python
def _loss_from_targets(self, cls_scores, bbox_preds, batch_img_metas,
                       target_data):
    (labels_list, label_weights_list, bbox_targets_list,
     bbox_weights_list, num_total_pos, num_total_neg) = target_data
    labels = torch.stack(labels_list, 0)
    label_weights = torch.stack(label_weights_list, 0)
    bbox_targets = torch.cat(bbox_targets_list, 0)
    bbox_weights = torch.cat(bbox_weights_list, 0)
    if self.text_masks.dim() != 2:
        raise ValueError('text mask must be [B,T]')
    text_masks = self.text_masks.new_zeros(
        (self.text_masks.size(0), self.max_text_len))
    text_masks[:, :self.text_masks.size(1)] = self.text_masks
    text_mask = (text_masks > 0).unsqueeze(1)
    text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
    cls_scores = torch.masked_select(
        cls_scores, text_mask).contiguous()
    labels = torch.masked_select(labels, text_mask)
    label_weights = label_weights[..., None].repeat(
        1, 1, text_mask.size(-1))
    label_weights = torch.masked_select(label_weights, text_mask)
    cls_avg_factor = (
        num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight)
    if self.sync_cls_avg_factor:
        cls_avg_factor = reduce_mean(
            cls_scores.new_tensor([cls_avg_factor]))
    cls_avg_factor = max(cls_avg_factor, 1)
    if isinstance(self.loss_cls, QualityFocalLoss):
        raise NotImplementedError(
            'QualityFocalLoss for GroundingDINO is unsupported')
    loss_cls = self.loss_cls(
        cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
    normalized_pos = loss_cls.new_tensor([num_total_pos])
    normalized_pos = torch.clamp(
        reduce_mean(normalized_pos), min=1).item()
    factors = []
    for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
        img_h, img_w = img_meta['img_shape']
        factors.append(bbox_pred.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]
        ).unsqueeze(0).repeat(bbox_pred.size(0), 1))
    factors = torch.cat(factors, 0)
    flattened_bbox = bbox_preds.reshape(-1, 5)
    absolute_bbox = flattened_bbox * factors
    absolute_target = bbox_targets * factors
    loss_iou = self.loss_iou(
        absolute_bbox, absolute_target, bbox_weights,
        avg_factor=normalized_pos)
    loss_bbox = self.loss_bbox(
        flattened_bbox, bbox_targets, bbox_weights,
        avg_factor=normalized_pos)
    return loss_cls, loss_bbox, loss_iou


def _flow_matching_loss_by_feat(
        self, cls_scores, bbox_preds, final_validity, flow_progress,
        batch_gt_instances, batch_img_metas):
    if (cls_scores.ndim != 4 or bbox_preds.shape[:3] != cls_scores.shape[:3]
            or bbox_preds.shape[-1] != 5 or
            final_validity.shape != cls_scores.shape[1:3]):
        raise ValueError('flow matching tensors have inconsistent shapes')
    layer_losses = [
        self.loss_by_feat_single(
            cls_scores[layer_id], bbox_preds[layer_id],
            batch_gt_instances, batch_img_metas)
        for layer_id in range(cls_scores.shape[0] - 1)
    ]
    current_match_weight = float(
        flow_progress.detach().item()) * float(
            self.query_flow_loss_cfg['match_weight'])
    if current_match_weight == 0.0:
        costs = None
    else:
        costs = [
            -current_match_weight * torch.log(
                row.detach().clamp(min=1e-6, max=1.0))
            for row in final_validity
        ]
    with torch.no_grad():
        target_data = self.get_targets(
            [row for row in cls_scores[-1]],
            [row for row in bbox_preds[-1]],
            batch_gt_instances, batch_img_metas,
            flow_match_costs=costs, capture_flow_targets=True)
    final_loss = self._loss_from_targets(
        cls_scores[-1], bbox_preds[-1], batch_img_metas, target_data)
    layer_losses.append(final_loss)
    losses = {
        'loss_cls': layer_losses[-1][0],
        'loss_bbox': layer_losses[-1][1],
        'loss_iou': layer_losses[-1][2],
    }
    for layer_id, values in enumerate(layer_losses[:-1]):
        losses[f'd{layer_id}.loss_cls'] = values[0]
        losses[f'd{layer_id}.loss_bbox'] = values[1]
        losses[f'd{layer_id}.loss_iou'] = values[2]
    return (losses, self.last_flow_validity_targets,
            self.last_flow_target_masks)


def _query_flow_loss_by_feat(
        self, all_layers_cls_scores, all_layers_bbox_preds,
        enc_cls_scores, enc_bbox_preds, batch_gt_instances,
        batch_img_metas, dn_meta, flow_validities, flow_progress):
    if enc_cls_scores is not None or enc_bbox_preds is not None:
        raise ValueError('fixed-query flow forbids encoder supervision')
    if not isinstance(dn_meta, dict):
        raise ValueError('query-flow training requires dn_meta')
    groups = int(dn_meta['num_matching_query_groups'])
    queries_per_group = int(dn_meta['num_matching_queries_per_group'])
    if groups != 3 or queries_per_group != 600:
        raise ValueError('query flow freezes 3x600 matching geometry')
    (matching_cls, matching_bbox, denoising_cls,
     denoising_bbox) = self.split_outputs(
         all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
    if flow_validities.shape != (
            matching_cls.shape[0], matching_cls.shape[1],
            groups * queries_per_group):
        raise ValueError('flow validity stack does not match 3x600 queries')
    cls_groups = split_matching_groups(
        matching_cls, queries_per_group, groups)
    bbox_groups = split_matching_groups(
        matching_bbox, queries_per_group, groups)
    validity_groups = flow_validities[-1].split(
        queries_per_group, dim=1)
    group_results = [
        self._flow_matching_loss_by_feat(
            group_cls, group_bbox, group_validity, flow_progress,
            batch_gt_instances, batch_img_metas)
        for group_cls, group_bbox, group_validity in zip(
            cls_groups, bbox_groups, validity_groups)
    ]
    losses = average_matching_loss_dicts(
        [result[0] for result in group_results])
    target_boxes = torch.stack([result[1] for result in group_results], 1)
    matched_masks = torch.stack([result[2] for result in group_results], 1)
    final_boxes = torch.stack(
        [group_bbox[-1] for group_bbox in bbox_groups], dim=1)
    flat_targets, diagnostic_masks = \
        aligned_rotated_iou_validity_targets(
            final_boxes.flatten(1, 2), target_boxes.flatten(1, 2),
            matched_masks.flatten(1, 2), batch_img_metas,
            torch.tensor([len(gt) == 0 for gt in batch_gt_instances],
                         device=final_boxes.device, dtype=torch.bool),
            angle_factor=self.angle_factor)
    validity = flow_validities[-1]
    raw_validity_loss = natural_validity_loss(
        validity, flat_targets, self.query_flow_loss_cfg['eps'])
    losses['loss_query_validity'] = (
        float(self.query_flow_loss_cfg['loss_weight']) * raw_validity_loss)
    self.last_flow_validity_targets = flat_targets.detach()
    self.last_flow_target_masks = {
        name: mask.detach() for name, mask in diagnostic_masks.items()
    }
    self.last_matching_group_masks = matched_masks.detach()
    self.last_matching_mask = matched_masks[:, 0].detach()
    if denoising_cls is not None:
        losses.update(self._denoising_loss_dict(
            denoising_cls, denoising_bbox, batch_gt_instances,
            batch_img_metas, dn_meta))
    return losses
```

At the first line of the existing `loss_by_feat`, add this exact dispatch;
the remainder of the legacy method stays unchanged:

```text
if self.query_flow_loss_cfg['enabled']:
    return self._query_flow_loss_by_feat(
        all_layers_cls_scores, all_layers_bbox_preds,
        enc_cls_scores, enc_bbox_preds, batch_gt_instances,
        batch_img_metas, dn_meta, flow_validities, flow_progress)
```

Add keyword parameters `flow_validities=None, flow_progress=None` to that
method's signature. In `loss`, add `flow_center_deltas=None`,
`flow_center_strength=None`, `flow_validities=None`, `flow_progress=None`, and
`flow_score_log_residual=None` to the signature, then dispatch before the
legacy body:

```text
if self.query_flow_loss_cfg['enabled']:
    if any(value is None for value in (
            flow_center_deltas, flow_center_strength,
            flow_validities, flow_progress)):
        raise ValueError('query-flow loss inputs are incomplete')
    batch_gt_instances = [
        sample.gt_instances for sample in batch_data_samples]
    batch_img_metas = [sample.metainfo for sample in batch_data_samples]
    outs = self(
        hidden_states, references, memory_text, text_token_mask,
        flow_center_deltas=flow_center_deltas,
        flow_center_strength=flow_center_strength)
    self.text_masks = text_token_mask
    return self.loss_by_feat(
        *outs, enc_outputs_class, enc_outputs_coord,
        batch_gt_instances, batch_img_metas, dn_meta,
        flow_validities=flow_validities, flow_progress=flow_progress)
```

- [ ] **Step 6: Apply suppressive all-row readout after unchanged argmax**

In `predict`, accept the detector-supplied final `flow_score_log_residual [B,600]`. In `_predict_by_feat_single`, keep `select_from_class_log_scores` unchanged, then pass the per-image residual to the existing calibrator:

```python
scores = calibrate_selected_log_scores(
    selected_log_scores,
    null_logits=None,
    capacity=None,
    temperature=1.0,
    power=1.0,
    log_residual=flow_score_log_residual)
```

Because the detector computed this residual as `target_score_strength * flow_progress * log(validity.clamp(min=1e-6,max=1.0))`, it is nonpositive and therefore strictly suppressive or neutral. It follows class argmax and cannot alter the 600 labels, boxes, or row order.

Add the following complete query-flow prediction path. Add the five flow
keywords shown here to the existing `predict` signature and execute this
dispatch before its legacy body:

```python
def _query_flow_predict(
        self, hidden_states, references, memory_text, text_token_mask,
        batch_data_samples, rescale, flow_center_deltas,
        flow_center_strength, flow_score_log_residual):
    if any(value is None for value in (
            flow_center_deltas, flow_center_strength,
            flow_score_log_residual)):
        raise ValueError('query-flow prediction inputs are incomplete')
    cls_scores, bbox_preds = self(
        hidden_states, references, memory_text, text_token_mask,
        flow_center_deltas=flow_center_deltas,
        flow_center_strength=flow_center_strength)
    cls_scores = cls_scores[-1]
    bbox_preds = bbox_preds[-1]
    if (cls_scores.shape[1] != 600 or bbox_preds.shape[1] != 600 or
            flow_score_log_residual.shape != cls_scores.shape[:2]):
        raise ValueError('query-flow inference must contain exactly Q600')
    if (not torch.isfinite(flow_score_log_residual).all().item() or
            not torch.all(flow_score_log_residual <= 0).item()):
        raise ValueError('flow score residual must be finite/nonpositive')
    positive_maps = [sample.token_positive_map
                     for sample in batch_data_samples]
    image_metas = [sample.metainfo for sample in batch_data_samples]
    return [
        self._predict_by_feat_single(
            cls_score, bbox_pred, positive_map, image_meta,
            rescale=rescale,
            flow_score_log_residual=flow_score_log_residual[index])
        for index, (cls_score, bbox_pred, positive_map, image_meta) in
        enumerate(zip(cls_scores, bbox_preds, positive_maps, image_metas))
    ]
```

```diff
 def predict(self, hidden_states, references, memory_text, text_token_mask,
             batch_data_samples, rescale=True, null_logits=None,
             capacity=None, semantic_gate=None,
+            flow_center_deltas=None, flow_center_strength=None,
+            flow_validities=None, flow_progress=None,
+            flow_score_log_residual=None):
+    if self.query_flow_loss_cfg['enabled']:
+        return self._query_flow_predict(
+            hidden_states, references, memory_text, text_token_mask,
+            batch_data_samples, rescale, flow_center_deltas,
+            flow_center_strength, flow_score_log_residual)
     # existing legacy body follows unchanged
```

Extend `_predict_by_feat_single` with
`flow_score_log_residual: Tensor = None`. Immediately before its existing
calibrator call, insert the validation and select one residual; then replace
the calibrator's final argument exactly as shown:

```python
if (flow_score_log_residual is not None and
        flow_score_log_residual.shape != (num_queries,)):
    raise ValueError('flow score residual must have one value per query')
if (flow_score_log_residual is not None and
        existence_log_residual is not None):
    raise ValueError('query flow and legacy existence cannot be combined')
if flow_score_log_residual is not None:
    if (not torch.isfinite(flow_score_log_residual).all().item() or
            not torch.all(flow_score_log_residual <= 0).item()):
        raise ValueError('flow score residual must be finite/nonpositive')
    selected_residual = flow_score_log_residual
else:
    selected_residual = existence_log_residual

scores = calibrate_selected_log_scores(
    selected_log_scores,
    null_logits=null_logits,
    capacity=(capacity if self.readout_cfg.get('use_capacity', False)
              else None),
    temperature=float(self.readout_cfg.get('temperature', 1.0)),
    power=float(self.readout_cfg.get('power', 1.0)),
    log_residual=selected_residual)
```

- [ ] **Step 7: Create exactly two thin configs**

The control inherits the registered rare4x base directly:

```python
_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py'
]

physical_gpus = (0, 1, 2, 3, 4)
selected_world_size = 5
query_flow_pair_id = 'QAF-v1-world5-seed20260716'
query_flow_role = 'control'
query_flow_master_port = 29910
query_flow_updates_per_epoch = 160
query_flow_total_updates = 1920
query_flow_source_report = '.lab/workspace/exp-8-qaf-source-v1/source_report.json'


def _validated_source_digest():
    import hashlib
    import json
    from pathlib import Path
    import re
    root = Path('/data1/zcy/OV-CapFlow')
    report_bytes = (root / query_flow_source_report).read_bytes()
    rows = [
        line for line in (root / '.lab/results.tsv').read_text().splitlines()
        if line.startswith('8-D160-QAF-SOURCE\t')]
    if len(rows) != 1:
        raise RuntimeError('query-flow config requires one frozen source row')
    match = re.search(r'report_sha256=([0-9a-f]{64})', rows[0])
    if match is None:
        raise RuntimeError('frozen source row has no report digest')
    expected = match.group(1)
    actual = hashlib.sha256(report_bytes).hexdigest()
    payload = json.loads(report_bytes)
    gates = payload.get('decision', {}).get('gates', {})
    if (payload.get('schema') != 'ov-capflow-query-evidence-source-v1' or
            payload.get('status') != 'PASS' or len(gates) != 5 or
            not all(value is True for value in gates.values())):
        raise RuntimeError('query-flow config requires a PASS source report')
    if actual != expected:
        raise RuntimeError('query-flow source report differs from governance')
    return expected


query_flow_source_report_sha256 = _validated_source_digest()
del _validated_source_digest

model = dict(
    decoder=dict(
        enable_null_reservoir=False,
        query_flow_cfg=dict(
            enabled=True, level_index=2, embed_dims=256, latent_dims=32,
            solver_iterations=5, entropy=0.1, column_relaxation=1.0,
            spatial_prior_share=0.75, null_prior_share=0.25,
            content_weight=1.0, spatial_weight=1.0,
            semantic_weight=1.0, spatial_cost_cap=4.0,
            min_reference_extent=0.01, max_center_delta=0.05,
            angle_factor=3.141592653589793,
            null_logit_bound=4.0,
            target_center_strength=0.0, target_score_strength=0.0,
            warmup_updates=160,
            eps=1e-6),
        layer_cfg=dict(
            enable_semantic_fusion=False,
            enable_density_capacity=False)),
    bbox_head=dict(
        matching_query_groups=3,
        balanced_cfg=dict(enabled=False),
        position_supervised_cfg=dict(enabled=False),
        adaptive_dn_cfg=dict(enabled=False),
        existence_loss_weight=None,
        readout_cfg=dict(temperature=1.0, power=1.0,
                         use_capacity=False),
        query_flow_loss_cfg=dict(
            enabled=True, loss_weight=0.0, match_weight=0.0,
            match_warmup_updates=160, eps=1e-6)),
    density_loss_cfg=dict(weight=0.0),
    null_loss_cfg=dict(),
    freeze_except_patterns=[r'^decoder\.query_evidence_coupling\.'])

train_dataloader = dict(
    batch_size=2,
    batch_sampler=dict(
        update_count_multiple=1,
        audit_path='work_dirs/dotav2_cleanstart/audits/'
                   'query_flow_control_world5_epoch_{epoch:02d}.json',
        audit_noreplace=True))
optim_wrapper = dict(
    _delete_=True, type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2), accumulative_counts=1)
val_evaluator = [
    dict(type='DOTAMetric', metric='mAP', iou_thrs=0.5,
         prefix='dota', _scope_='mmrotate'),
    dict(type='DOTAMetric', metric='mAP', iou_thrs=0.75,
         prefix='dota75', _scope_='mmrotate'),
]
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(
    by_epoch=True, interval=1, max_keep_ckpts=-1,
    save_best=None, rule=None, save_last=True))
custom_hooks = [dict(type='QueryFlowParentEvalHook')]
randomness = dict(seed=20260716, deterministic=False, diff_rank_seed=False)
load_from = 'work_dirs/dotav2_cleanstart/' \
            'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/' \
            'epoch_24.pth'
work_dir = 'work_dirs/dotav2_cleanstart/' \
           'query_flow_world5_control_seed20260716_gpu01234_batch2'
resume = False
```

The config therefore fails closed when the report is absent/negative, the committed governance row is absent/duplicated, or the report bytes differ from the exact digest recorded after observation. The resolved run config embeds that frozen digest. This is a measured prerequisite, not a tunable value.

The candidate inherits the control and changes only:

```python
physical_gpus = (5, 6, 7, 8, 9)
query_flow_role = 'candidate'
query_flow_master_port = 29911
model = dict(
    decoder=dict(query_flow_cfg=dict(
        target_center_strength=1.0, target_score_strength=1.0)),
    bbox_head=dict(query_flow_loss_cfg=dict(
        loss_weight=1.0, match_weight=0.5)))
train_dataloader = dict(batch_sampler=dict(
    audit_path='work_dirs/dotav2_cleanstart/audits/'
               'query_flow_candidate_world5_epoch_{epoch:02d}.json'))
work_dir = 'work_dirs/dotav2_cleanstart/' \
           'query_flow_world5_candidate_seed20260716_gpu56789_batch2'
```

Do not inherit either D13 config and do not use `D13NOptimWrapperConstructor` or D13 hooks.

- [ ] **Step 8: Run focused tests and exact config comparisons**

Run:

```bash
rtk run "test ! -e .lab/workspace/exp-8-qaf-stage0-v1"
rtk mkdir .lab/workspace/exp-8-qaf-stage0-v1
rtk bash -o pipefail -c 'rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_evidence_source.py tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py -q 2>&1 | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/focused_pytest.log'
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_sampler_coverage.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py --world-size 5 --epoch 0 --seed 20260716 --output .lab/workspace/exp-8-qaf-stage0-v1/control_sampler_epoch0.json
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_sampler_coverage.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --world-size 5 --epoch 0 --seed 20260716 --output .lab/workspace/exp-8-qaf-stage0-v1/candidate_sampler_epoch0.json
rtk git diff --check
```

Expected: both configs resolve to 1,600 training images, 400 validation
images, batch 2 x world 5, 160 updates/epoch, 12 epochs, Q600, three matching
groups, a DN budget of 100 with the unchanged parent dynamic prefix rule,
scale1024, rare4x, identical sampler orders, all epoch checkpoints retained,
and separate `dota` AP50 plus `dota75` AP75 evaluators. Only registered
role/target-strength/loss/match/work-dir/audit-path/port/GPU metadata differ.

All commands shown use immutable attempt `v1`. If and only if the single
allowed in-recipe root-cause repair is approved, repeat Step 8 and every Task
3 command with the literal directory
`.lab/workspace/exp-8-qaf-stage0-v2`; first assert that directory does not
exist. Never alter `v1`, never reuse an output name, and never create `v3`.
Failure of `v2` closes the family.

```bash
rtk run "test ! -e .lab/workspace/exp-8-qaf-stage0-v2"
```

- [ ] **Step 9: Freeze the exact integration diff for Stage-0**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/ov_capflow/ov_capflow_layers.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py
rtk git diff --cached --check
rtk git diff --cached --stat
rtk bash -o pipefail -c 'rtk git diff --cached --binary | rtk sha256sum'
```

Expected: only the six listed paths are staged and their exact staged-diff SHA is copied into every Stage-0 artifact. Do not commit until Stage-0 PASS, so the Stage-0 governance row can be included without a fourth pre-proxy commit.

## Task 3: Pass all five Stage-0 gates

**Files generated:** `.lab/workspace/exp-8-qaf-stage0-v1/*` and, only after
the one allowed repair, `.lab/workspace/exp-8-qaf-stage0-v2/*`

Require the Step-8 no-replace directory and its two sampler reports before the remaining checks:

```bash
rtk run "test -d .lab/workspace/exp-8-qaf-stage0-v1"
rtk run "test -f .lab/workspace/exp-8-qaf-stage0-v1/control_sampler_epoch0.json"
rtk run "test -f .lab/workspace/exp-8-qaf-stage0-v1/candidate_sampler_epoch0.json"
```

- [ ] **Step 1: Identity and checkpoint-load gate**

Run:

```bash
rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py -q -k "checkpoint_key_contract or identity_contract" 2>&1 | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/identity_pytest.log'
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_parent_equivalence.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --output .lab/workspace/exp-8-qaf-stage0-v1/parent_equivalence.json
```

Expected: the focused test uses `load_state_dict(strict=False)` and asserts the missing set is exactly the two projection weights, null key, null bias, and persistent `flow_progress` buffer under `decoder.query_evidence_coupling.*`; there are no unexpected or shape-mismatched keys. Parent labels/scores/boxes/decoder states/references are exact. The focused real-batch test separately proves parent losses and assignments exact. The legacy `audit_checkpoint_load.py` is not used because its allowlist predates query flow.

- [ ] **Step 2: Protocol and open-vocabulary gates**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_strict_inference.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py --checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --output .lab/workspace/exp-8-qaf-stage0-v1/strict_inference.json --device cuda:0
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_open_vocabulary.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py --checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --output .lab/workspace/exp-8-qaf-stage0-v1/open_vocabulary.json --device cuda:0
```

Expected: Q600 all-row order, same prompt/class order, no forbidden mouth operation, PASS.

- [ ] **Step 3: Finite, parent-freeze, and zero-step equality gates**

Run:

```bash
rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_real_batch_q600_finite_contract -q -s 2>&1 | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/finite_pytest.log'
rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_real_optimizer_step_freezes_parent -q -s 2>&1 | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/parent_freeze_pytest.log'
rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_real_batch_zero_route_is_parent_exact -q -s 2>&1 | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/zero_step_pytest.log'
rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_real_candidate_nonzero_flow_contract -q -s 2>&1 | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/nonzero_flow_pytest.log'
rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_REAL_BATCH=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_control_stays_zero_after_real_optimizer_step -q -s 2>&1 | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/control_optimizer_pytest.log'
rtk bash -o pipefail -c 'rtk sha256sum .lab/workspace/exp-8-qaf-stage0-v1/identity_pytest.log .lab/workspace/exp-8-qaf-stage0-v1/parent_equivalence.json .lab/workspace/exp-8-qaf-stage0-v1/strict_inference.json .lab/workspace/exp-8-qaf-stage0-v1/open_vocabulary.json .lab/workspace/exp-8-qaf-stage0-v1/finite_pytest.log .lab/workspace/exp-8-qaf-stage0-v1/parent_freeze_pytest.log .lab/workspace/exp-8-qaf-stage0-v1/zero_step_pytest.log .lab/workspace/exp-8-qaf-stage0-v1/nonzero_flow_pytest.log .lab/workspace/exp-8-qaf-stage0-v1/control_optimizer_pytest.log .lab/workspace/exp-8-qaf-stage0-v1/control_sampler_epoch0.json .lab/workspace/exp-8-qaf-stage0-v1/candidate_sampler_epoch0.json | rtk tee .lab/workspace/exp-8-qaf-stage0-v1/artifact_sha256.txt'
```

Expected: all tests pass for empty, normal, and 1,223-GT stress samples; parent tensors have identical hashes before/after optimizer step; zero route is exact for scores, labels, boxes, order, references, parent losses, and assignments; no non-finite coupling, Hungarian input, loss, gradient, or optimizer state.

`artifact_sha256.txt` is the mandatory Stage-0 evidence index. The governance
row maps identity to `identity_pytest.log` plus `parent_equivalence.json`,
protocol to `strict_inference.json` plus `open_vocabulary.json`, finiteness to
`finite_pytest.log` plus `nonzero_flow_pytest.log`, parent freeze to
`parent_freeze_pytest.log`, and zero-step equality to `zero_step_pytest.log`
plus `control_optimizer_pytest.log`; it also carries both sampler-report hashes.
No PASS/FAIL row may cite terminal output that is absent from this index. For
an approved repair, the identical file set must exist under `v2`, and the row
must cite only `v2` as the terminal attempt while retaining `v1` unchanged.

- [ ] **Step 4: Archive Stage-0 and create concentrated commit 3**

If any check fails, do not launch. One root-cause repair may be proposed only if it preserves the frozen mathematical recipe; a second scientific recipe is forbidden. If the failure remains after that one in-recipe repair, use `apply_patch` to record row `8-D161-QAF-STAGE0` as negative with the failed gate, staged-diff SHA, and artifact hashes, mark the family closed in the three Markdown governance files, then run:

```bash
rtk git restore --staged projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/ov_capflow/ov_capflow_layers.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py
rtk git add .lab/results.tsv progress.md task_plan.md projects/OVCapFlow/README.md
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "results: close query flow at stage zero"
```

Expected: only the observed negative governance record is committed; the uncommitted implementation diff remains quarantined and Plan Tasks 4-5 are forbidden.

On PASS, use `apply_patch` to add row `8-D161-QAF-STAGE0` to `.lab/results.tsv`, listing the staged-diff SHA and existing report paths/hashes, and update the three Markdown surfaces. Then run:

```bash
rtk git add .lab/results.tsv progress.md task_plan.md projects/OVCapFlow/README.md projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/ov_capflow/ov_capflow_layers.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "feat: integrate verified unified query flow"
```

Expected: this is concentrated commit 3. The relevant worktree is clean before proxy launch and the commit contains integration, two configs, focused tests, and the observed Stage-0 governance record.

## Task 4: Launch the one registered ten-GPU proxy pair

**Resources:** control GPUs 0-4; candidate GPUs 5-9.

- [ ] **Step 1: Verify all ten GPUs and frozen repository state**

Run:

```bash
rtk run "test ! -e .lab/workspace/exp-8-qaf-proxy-v1"
rtk mkdir .lab/workspace/exp-8-qaf-proxy-v1
rtk bash -o pipefail -c 'rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-proxy-v1/gpu_snapshot_0.csv'
rtk bash -o pipefail -c 'rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-proxy-v1/compute_apps.csv'
rtk run "sleep 30"
rtk bash -o pipefail -c 'rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-proxy-v1/gpu_snapshot_1.csv'
rtk run "sleep 30"
rtk bash -o pipefail -c 'rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-proxy-v1/gpu_snapshot_2.csv'
rtk ss -ltnp
rtk tmux list-sessions
rtk run "test ! -e work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2"
rtk run "test ! -e work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2"
rtk git status --short
rtk git diff --exit-code -- projects/OVCapFlow/ov_capflow projects/OVCapFlow/tools tests/test_projects/ov_capflow configs/ov_capflow
rtk git rev-parse HEAD
rtk sha256sum .lab/workspace/exp-8-qaf-source-v1/source_report.json work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/train/manifest.json work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py projects/OVCapFlow/tools/dotav2_q600_diagnostics.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py
```

Proceed only if all three saved samples show each GPU at most 1,024 MiB used and at most 5% utilization, the compute-app query is empty, ports 29910-29915 are absent from `ss`, the four planned tmux names are absent, work directories do not exist, Stage-0 is PASS, source/analyzer hashes match the frozen values above, config hashes match `8-D161-QAF-STAGE0`, and no post-commit production/config diff exists. The three snapshots are the frozen GPU/UUID launch evidence.

- [ ] **Step 2: Launch both arms simultaneously**

Run:

```bash
rtk mkdir -p work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2 work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2
rtk tmux new-session -d -s qaf_control_world5 -c /data1/zcy/OV-CapFlow "rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29910 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py --launcher pytorch 2>&1 | rtk tee work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/launcher.log'"
rtk tmux new-session -d -s qaf_candidate_world5 -c /data1/zcy/OV-CapFlow "rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=5,6,7,8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29911 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --launcher pytorch 2>&1 | rtk tee work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/launcher.log'"
rtk tmux list-sessions
```

Expected: both sessions remain alive, five processes per arm, disjoint physical GPUs, P2P/IB disabled, epoch-0 sampler hashes equal, and no NCCL/non-finite/OOM error.

- [ ] **Step 3: Monitor to the fixed E12 endpoint**

Launch one existing monitor per arm rather than writing a new daemon:

```bash
rtk tmux new-session -d -s qaf_control_monitor -c /data1/zcy/OV-CapFlow "rtk bash -o pipefail -c 'pane_pid=\$(rtk tmux display-message -p -t qaf_control_world5:0 \"#{pane_pid}\"); exec rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/monitor_gpu_run.py --pid \"\$pane_pid\" --gpu-indices 0 1 2 3 4 --interval 30 --train-log work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/launcher.log --output .lab/workspace/exp-8-qaf-proxy-v1/control_monitor.jsonl'"
rtk tmux new-session -d -s qaf_candidate_monitor -c /data1/zcy/OV-CapFlow "rtk bash -o pipefail -c 'pane_pid=\$(rtk tmux display-message -p -t qaf_candidate_world5:0 \"#{pane_pid}\"); exec rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/monitor_gpu_run.py --pid \"\$pane_pid\" --gpu-indices 5 6 7 8 9 --interval 30 --train-log work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/launcher.log --output .lab/workspace/exp-8-qaf-proxy-v1/candidate_monitor.jsonl'"
rtk tmux capture-pane -pt qaf_control_world5:0 -S -120
rtk tmux capture-pane -pt qaf_candidate_world5:0 -S -120
rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
rtk tail -n 2 .lab/workspace/exp-8-qaf-proxy-v1/control_monitor.jsonl .lab/workspace/exp-8-qaf-proxy-v1/candidate_monitor.jsonl
```

Archive every epoch checkpoint but select only `epoch_12.pth`. An intermediate peak cannot promote. A monitor fatal label or missing process before E12 stops both arms and records an infrastructure failure; scientific non-finiteness closes the family. OOM permits only a reviewed, mathematically identical effective-batch restart of both arms. Before checkpoint validation, both monitor JSONL files must end with a summary, contain no fatal label, and the training logs must contain the E12 completion markers.

- [ ] **Step 4: Produce endpoint all-row dumps on the same two GPU groups**

After both `epoch_12.pth` files appear, run the registered CPU checkpoint gate and freeze their digests:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_PROXY_CHECKPOINTS=1 OVCAPFLOW_QUERY_FLOW_CONTROL_CHECKPOINT=work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/epoch_12.pth OVCAPFLOW_QUERY_FLOW_CANDIDATE_CHECKPOINT=work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/epoch_12.pth OVCAPFLOW_QUERY_FLOW_PARENT_CHECKPOINT=work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_proxy_endpoint_checkpoint_contract -q -s
rtk sha256sum work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/epoch_12.pth work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/epoch_12.pth
```

Only after that gate passes, run:

```bash
rtk run "test ! -e work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12"
rtk run "test ! -e work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12"
rtk ss -ltnp
rtk tmux list-sessions
rtk tmux new-session -d -s qaf_control_eval -c /data1/zcy/OV-CapFlow "rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29912 tools/test.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/epoch_12.pth --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12 --out work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/predictions.pkl"
rtk tmux new-session -d -s qaf_candidate_eval -c /data1/zcy/OV-CapFlow "rtk env CUDA_VISIBLE_DEVICES=5,6,7,8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29913 tools/test.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/epoch_12.pth --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12 --out work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/predictions.pkl"
```

Proceed only if ports 29912/29913, both eval tmux names, and both output directories are absent. Expected: each dump has exactly 400 records and 240,000 ordered rows.

- [ ] **Step 5: Reuse the frozen analyzer for both endpoint dumps**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/predictions.pkl --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/epoch_12.pth --official-metrics-json work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12 --output-dir work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/diagnostics --allow-noncanonical --expected-records 400 --queries-per-image 600
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/predictions.pkl --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/epoch_12.pth --official-metrics-json work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12 --output-dir work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/diagnostics --allow-noncanonical --expected-records 400 --queries-per-image 600
```

- [ ] **Step 6: Re-run endpoint mouth, vocabulary, and dump-integrity checks**

Run:

```bash
rtk bash -o pipefail -c 'rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/validate_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/predictions.pkl --expected-records 400 --queries-per-image 600 --num-classes 18 | rtk tee work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/dump_validation.json'
rtk bash -o pipefail -c 'rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/validate_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/predictions.pkl --expected-records 400 --queries-per-image 600 --num-classes 18 | rtk tee work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/dump_validation.json'
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_strict_inference.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/epoch_12.pth --output work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/strict_inference.json --device cuda:0
rtk env CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_strict_inference.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/epoch_12.pth --output work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/strict_inference.json --device cuda:0
rtk env CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_open_vocabulary.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/epoch_12.pth --output work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/open_vocabulary.json --device cuda:0
```

Expected: both validators report 400 records, 240,000 finite CPU rows, 600 per image, unique image ids, and labels in `[0,17]`; strict and open-vocabulary audits PASS.

- [ ] **Step 7: Apply the proxy promotion arithmetic exactly and publish one decision**

Read each `diagnostics/diagnostics.json` and compute:

```python
delta_map = C['metrics']['map'] - K['metrics']['map']
delta_ap50 = (C_official['dota/AP50'] -
              K_official['dota/AP50'])
geometry_drop = (K['gt_decomposition']['geometry_miss'] /
                 K['gt_decomposition']['total_gt'] -
                 C['gt_decomposition']['geometry_miss'] /
                 C['gt_decomposition']['total_gt'])
spearman_gain = (C['calibration']['tp_score_iou_spearman'] -
                 K['calibration']['tp_score_iou_spearman'])
base_delta = C['groups']['base']['map'] - K['groups']['base']['map']
novel_delta = C['groups']['novel']['map'] - K['groups']['novel']['map']
for region in ('empty_tile_fp', 'duplicate_fp',
               'localization_background_fp'):
    control_rate = (K['fp_regions']['ap_support'][region] /
                    K['fp_regions']['ap_support']['row_count'])
    candidate_rate = (C['fp_regions']['ap_support'][region] /
                      C['fp_regions']['ap_support']['row_count'])
    relative_worsening = candidate_rate / control_rate - 1
```

Here `K` and `C` are the analyzer `diagnostics.json` payloads and their `metrics.map` values are the full-precision reconstructed same-dump mAP. `K_official` and `C_official` are the unique official evaluator rows loaded from the two `eval_e12` metric directories; their `dota/AP50` values are the evaluator's registered three-decimal AP50. Never substitute diagnostics `metrics.ap50`, which is rounded and cannot represent the full-precision mAP gate.

PASS requires `delta_map >= 0.010`, `delta_ap50 >= 0.010`, `geometry_drop >= 0.020`, `spearman_gain >= 0.05`, `novel_delta >= 0`, `base_delta >= -0.005`, and every relative worsening `<=0.05`, plus strict mouth/integrity PASS. Zero control rate permits no positive candidate count.

If any condition fails, record row `8-D162-QAF-PROXY` as negative, archive both hashes and diagnostics, close the family, and stop. No coefficient, epoch, seed, feature level, or partial-interface rescue is authorized.

Run the opt-in fail-closed test:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_PROXY_GATE=1 OVCAPFLOW_QUERY_FLOW_CONTROL_DIAGNOSTICS=work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/diagnostics/diagnostics.json OVCAPFLOW_QUERY_FLOW_CANDIDATE_DIAGNOSTICS=work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/diagnostics/diagnostics.json OVCAPFLOW_QUERY_FLOW_CONTROL_METRICS_DIR=work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12 OVCAPFLOW_QUERY_FLOW_CANDIDATE_METRICS_DIR=work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12 OVCAPFLOW_QUERY_FLOW_CONTROL_STRICT=work_dirs/dotav2_cleanstart/query_flow_world5_control_seed20260716_gpu01234_batch2/eval_e12/strict_inference.json OVCAPFLOW_QUERY_FLOW_CANDIDATE_STRICT=work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/strict_inference.json OVCAPFLOW_QUERY_FLOW_OPEN_VOCAB=work_dirs/dotav2_cleanstart/query_flow_world5_candidate_seed20260716_gpu56789_batch2/eval_e12/open_vocabulary.json OVCAPFLOW_QUERY_FLOW_PROXY_DECISION=.lab/workspace/exp-8-qaf-proxy-v1/proxy_decision.json /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_proxy_promotion_gate_from_frozen_reports -q -s
```

The test creates the decision directory, rejects an existing output, validates every required finite field and artifact hash, serializes full-precision arithmetic, and exits nonzero on FAIL after writing the negative decision. On either PASS or FAIL, use `apply_patch` to record row `8-D162-QAF-PROXY` in `.lab/results.tsv` and copy its exact status, decision/report paths, full-precision metrics, failed gates, and artifact hashes into the three Markdown governance files. Then run:

```bash
rtk git add .lab/results.tsv progress.md task_plan.md projects/OVCapFlow/README.md
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "results: record query flow proxy gate"
```

Expected: one results-only commit. A FAIL record closes the family immediately; only the committed PASS record continues.

## Task 5: Train the canonical full candidate and run the raw gate only after proxy PASS

This task trains one full 47,294-image frozen-parent candidate from E24, then compares it with the exact zero-route E24 control on all 13,833 raw validation images. It reuses the candidate/control configs with CLI data/topology overrides and does not create a third tracked config.

The full recipe is frozen as: manifest `work_dirs/dotav2_cleanstart/full_rare4x_seed20260718/train/manifest.json`, SHA `152d1244778ecfab454cca459f8cfec03a9927b8894a9eb4d48ec1a9c5cc4ce6`, 47,294 images, 20,103 empty tiles, rare4x exposure 3,296, scale1024, seed 20260716, world 10 x batch 2, 2,365 exact no-replace updates/epoch, 12 epochs, 28,380 optimizer updates, E12-only raw validation, AdamW `1e-4/1e-4`, clip `0.1`, and every E24 parent tensor frozen/eval.

- [ ] **Step 1: Verify the full sampler, manifest, GPUs, port, and output collision**

Run:

```bash
rtk run "test ! -e .lab/workspace/exp-8-qaf-raw-v1"
rtk mkdir .lab/workspace/exp-8-qaf-raw-v1
rtk sha256sum work_dirs/dotav2_cleanstart/full_rare4x_seed20260718/train/manifest.json projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py projects/OVCapFlow/tools/dotav2_q600_diagnostics.py
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_FULL_RECIPE=1 OVCAPFLOW_QUERY_FLOW_FULL_SAMPLER_REPORT=.lab/workspace/exp-8-qaf-raw-v1/full_sampler_epoch0.json /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_full_recipe_world10_sampler_contract -q -s
rtk bash -o pipefail -c 'rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-raw-v1/gpu_snapshot_0.csv'
rtk bash -o pipefail -c 'rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-raw-v1/compute_apps.csv'
rtk run "sleep 30"
rtk bash -o pipefail -c 'rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-raw-v1/gpu_snapshot_1.csv'
rtk run "sleep 30"
rtk bash -o pipefail -c 'rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | rtk tee .lab/workspace/exp-8-qaf-raw-v1/gpu_snapshot_2.csv'
rtk ss -ltnp
rtk tmux list-sessions
rtk run "test ! -e work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2"
```

Expected: hashes equal the frozen values, sampler report has `dataset_size=47294`, `world_size=10`, `update_count=2365`, `duplicate_count=0`, `missing_count=0`, `global_batch_size_min=14`, `global_batch_size_max=20`; all three samples meet the proxy idle threshold and the compute-app query is empty; port 29920, tmux `qaf_full_candidate_world10`, and the output directory are free.

- [ ] **Step 2: Launch the one full frozen-parent candidate on GPUs 0-9**

Run:

```bash
rtk mkdir -p work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2
rtk tmux new-session -d -s qaf_full_candidate_world10 -c /data1/zcy/OV-CapFlow "rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=10 --master_port=29920 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2 --cfg-options selected_world_size=10 physical_gpus=[0,1,2,3,4,5,6,7,8,9] query_flow_master_port=29920 query_flow_updates_per_epoch=2365 query_flow_total_updates=28380 train_dataloader.dataset.data_root=/data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/full_rare4x_seed20260718/train/ train_dataloader.dataset.ann_file=annfiles/ train_dataloader.dataset.data_prefix.img_path=images/ train_dataloader.batch_size=2 train_dataloader.batch_sampler.update_count_multiple=1 train_dataloader.batch_sampler.audit_path=work_dirs/dotav2_cleanstart/audits/query_flow_full_candidate_world10_epoch_{epoch:02d}.json train_cfg.max_epochs=12 train_cfg.val_interval=12 val_dataloader.dataset.data_root=/data1/zcy/datasets/DOTA2_1024_500/ val_dataloader.dataset.ann_file=ss_val/annfiles/ val_dataloader.dataset.data_prefix.img_path=ss_val/images/ val_dataloader.dataset.filter_cfg.filter_empty_gt=False val_dataloader.dataset.test_mode=True 2>&1 | rtk tee work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/launcher.log'"
rtk tmux new-session -d -s qaf_full_candidate_monitor -c /data1/zcy/OV-CapFlow "rtk bash -o pipefail -c 'pane_pid=\$(rtk tmux display-message -p -t qaf_full_candidate_world10:0 \"#{pane_pid}\"); exec rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/monitor_gpu_run.py --pid \"\$pane_pid\" --gpu-indices 0 1 2 3 4 5 6 7 8 9 --interval 30 --train-log work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/launcher.log --output .lab/workspace/exp-8-qaf-raw-v1/full_candidate_monitor.jsonl'"
```

Expected: ten ranks, 2,365 updates each epoch, 28,380 at E12, exact full-manifest coverage per epoch, parent modules eval/frozen, flow progress reaches 1.0 at update 160 and remains 1.0, all checkpoints retained, no non-finite/OOM/NCCL event. Poll the existing monitor with `rtk tail -n 2 .lab/workspace/exp-8-qaf-raw-v1/full_candidate_monitor.jsonl`; before checkpoint validation it must end with a summary, contain no fatal label, and the training log must contain the E12 completion marker.

- [ ] **Step 3: Verify the full E12 checkpoint before raw inference**

Run:

```bash
rtk sha256sum work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/epoch_12.pth
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_FULL_CHECKPOINT=1 OVCAPFLOW_QUERY_FLOW_FULL_CHECKPOINT=work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/epoch_12.pth OVCAPFLOW_QUERY_FLOW_PARENT_CHECKPOINT=work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_full_checkpoint_parent_hash_contract -q -s
```

Expected: checkpoint metadata epoch 12/iter 28,380, `flow_progress==1`, at least one coupling parameter differs from deterministic initialization, and every non-flow tensor is byte-identical to E24.

- [ ] **Step 4: Run parent and full candidate raw inference with separate AP50/AP75 evaluators**

Use the QAF control config plus E24 on GPUs 0-4; Stage-0 proves its zero route exact. Use the full candidate E12 on GPUs 5-9. Both configs already contain separate `dota` IoU-0.5 and `dota75` IoU-0.75 evaluators, so `dota/mAP` remains the canonical AP50-compatible quantity.

```bash
rtk run "test ! -e work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24"
rtk run "test ! -e work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12"
rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
rtk ss -ltnp
rtk tmux list-sessions
rtk tmux new-session -d -s qaf_parent_raw -c /data1/zcy/OV-CapFlow "rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29914 tools/test.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24 --out work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24/predictions.pkl --cfg-options test_dataloader.dataset.data_root=/data1/zcy/datasets/DOTA2_1024_500/ test_dataloader.dataset.ann_file=ss_val/annfiles/ test_dataloader.dataset.data_prefix.img_path=ss_val/images/ test_dataloader.dataset.filter_cfg.filter_empty_gt=False test_dataloader.dataset.test_mode=True"
rtk tmux new-session -d -s qaf_candidate_raw -c /data1/zcy/OV-CapFlow "rtk env CUDA_VISIBLE_DEVICES=5,6,7,8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=5 --master_port=29915 tools/test.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/epoch_12.pth --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12 --out work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12/predictions.pkl --cfg-options test_dataloader.dataset.data_root=/data1/zcy/datasets/DOTA2_1024_500/ test_dataloader.dataset.ann_file=ss_val/annfiles/ test_dataloader.dataset.data_prefix.img_path=ss_val/images/ test_dataloader.dataset.filter_cfg.filter_empty_gt=False test_dataloader.dataset.test_mode=True"
```

Proceed only if no compute process remains, all GPUs meet the idle threshold, ports 29914/29915 and both raw tmux names are absent, and both output directories are absent. Expected: each dump has 13,833 records and exactly 8,299,800 ordered rows; each metrics log contains `dota/AP50`, `dota/mAP`, `dota75/AP75`, and `dota75/mAP`. At single IoU 0.5, `dota/mAP` retains evaluator precision while `dota/AP50` is its registered three-decimal rounded companion; they must satisfy the analyzer's existing parity rule, not exact numeric equality.

- [ ] **Step 5: Run exact raw diagnostics, dump validation, mouth, and endpoint strict checks**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24/predictions.pkl --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_control.py --checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --official-metrics-json work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24 --output-dir work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24/diagnostics --allow-noncanonical --expected-records 13833 --queries-per-image 600
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12/predictions.pkl --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/epoch_12.pth --official-metrics-json work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12 --output-dir work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12/diagnostics --allow-noncanonical --expected-records 13833 --queries-per-image 600
rtk bash -o pipefail -c 'rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/validate_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24/predictions.pkl --expected-records 13833 --queries-per-image 600 --num-classes 18 | rtk tee work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24/dump_validation.json'
rtk bash -o pipefail -c 'rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/validate_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12/predictions.pkl --expected-records 13833 --queries-per-image 600 --num-classes 18 | rtk tee work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12/dump_validation.json'
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_dotav2_mouth.py --train-root work_dirs/dotav2_cleanstart/full_rare4x_seed20260718/train --val-root /data1/zcy/datasets/DOTA2_1024_500/ss_val --expected-train 47294 --expected-val 13833 --expected-classes airport baseball-diamond basketball-court bridge container-crane ground-track-field harbor helicopter helipad large-vehicle plane roundabout ship small-vehicle soccer-ball-field storage-tank swimming-pool tennis-court --output .lab/workspace/exp-8-qaf-raw-v1/raw_mouth.json
rtk env CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_strict_inference.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/epoch_12.pth --output .lab/workspace/exp-8-qaf-raw-v1/candidate_strict_inference.json --device cuda:0
rtk env CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_open_vocabulary.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch2_rare4x_world5_query_flow_candidate.py --checkpoint work_dirs/dotav2_cleanstart/query_flow_fullrare4x_candidate_seed20260716_gpu0123456789_batch2/epoch_12.pth --output .lab/workspace/exp-8-qaf-raw-v1/candidate_open_vocabulary.json --device cuda:0
```

Expected: analyzer parity/integrity PASS, both validators report 8,299,800 finite rows, mouth reports 47,294/13,833 paired files and exact 18-class vocabulary, strict/open-vocabulary audits PASS.

- [ ] **Step 6: Enforce exact raw promotion and stratum tolerances**

PASS requires candidate `dota/mAP >= 0.6114053488`, `dota/AP50 >= 0.6110`, base14 delta `>= -0.001`, novel4 delta `>= -0.001`, `dota75/AP75` strictly above parent, lower geometry-miss rate, higher TP score-IoU Spearman, and every strict/integrity artifact PASS.

For every `strata.csv` row whose `stratum_type` is `density`, `size`, `angle`, or the frozen analyzer's exact name `aspect` and whose parent `gt_count >= 100`, candidate `evaluator_reachable_rate` may decline by at most `0.010` absolute and candidate `geometry_miss_rate` may worsen by at most `0.010` absolute. Rows below 100 GT are reported in `underpowered_strata` and do not determine PASS. Missing/mismatched rows, zero parent denominators, or non-finite fields fail closed.

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OVCAPFLOW_RUN_QUERY_FLOW_RAW_GATE=1 OVCAPFLOW_QUERY_FLOW_PARENT_DIAGNOSTICS=work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24/diagnostics/diagnostics.json OVCAPFLOW_QUERY_FLOW_CANDIDATE_DIAGNOSTICS=work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12/diagnostics/diagnostics.json OVCAPFLOW_QUERY_FLOW_PARENT_STRATA=work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24/diagnostics/strata.csv OVCAPFLOW_QUERY_FLOW_CANDIDATE_STRATA=work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12/diagnostics/strata.csv OVCAPFLOW_QUERY_FLOW_PARENT_METRICS_DIR=work_dirs/dotav2_cleanstart/query_flow_raw_parent_e24 OVCAPFLOW_QUERY_FLOW_CANDIDATE_METRICS_DIR=work_dirs/dotav2_cleanstart/query_flow_raw_candidate_e12 OVCAPFLOW_QUERY_FLOW_RAW_MOUTH=.lab/workspace/exp-8-qaf-raw-v1/raw_mouth.json OVCAPFLOW_QUERY_FLOW_RAW_STRICT=.lab/workspace/exp-8-qaf-raw-v1/candidate_strict_inference.json OVCAPFLOW_QUERY_FLOW_RAW_OPEN_VOCAB=.lab/workspace/exp-8-qaf-raw-v1/candidate_open_vocabulary.json OVCAPFLOW_QUERY_FLOW_RAW_DECISION=.lab/workspace/exp-8-qaf-raw-v1/raw_decision.json /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_allocation_quality_flow.py::test_raw_promotion_gate_from_frozen_reports -q -s
```

The test writes the complete decision before raising on FAIL. On either outcome, use `apply_patch` to record row `8-D163-QAF-RAW` in `.lab/results.tsv` and copy the exact status, full-precision AP50/AP75/group/geometry/calibration/stratum decisions, failed gates, and every checkpoint/dump/metric/diagnostic hash into the three Markdown governance files. Then run:

```bash
rtk git add .lab/results.tsv progress.md task_plan.md projects/OVCapFlow/README.md
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "results: record query flow raw gate"
```

Expected: one results-only commit. A raw FAIL closes the family and stops. A raw PASS freezes the evidence and unlocks a separate paper-evidence plan for three seeds, a true class holdout, a second oriented dataset, unity ablations, efficiency, figures, and manuscript claims. Those paper-scale actions are intentionally outside this plan because the approved raw gate must precede them.

## Final verification checklist

- [ ] Source report is PASS and both configs fail closed against the exact digest in committed governance.
- [ ] Exactly one new production file, two configs, and two total focused test files exist across both plans.
- [ ] Shared coupling parameters total exactly 16,417 and are shared across all six layers.
- [ ] DN rows never enter coupling, matching validity, or validity loss.
- [ ] Level index 2 is the only evidence source.
- [ ] Coupling is float32, deterministic, finite, row-normalized, null-aware, and softly column-competitive.
- [ ] The same final validity drives center transport, final matching cost, soft-IoU loss, and all-row score residual.
- [ ] Final-layer transport affects final boxes; width/height/angle and row order remain unchanged.
- [ ] Zero route is exact for parent outputs, parent losses, and assignments.
- [ ] No top-k, filter, threshold, sort, NMS, proposal, dense, RPN, or RoI mouth exists.
- [ ] Five Stage-0 checks pass before any distributed launch.
- [ ] Only the fixed E12 proxy endpoint is eligible for promotion.
- [ ] Proxy failure closes the family; only proxy PASS unlocks raw.
- [ ] Proxy PASS launches one 47,294-image, world-10, E12 full candidate before raw evaluation.
- [ ] Raw failure closes the family; only raw PASS unlocks paper-scale planning.
- [ ] The three concentrated commits are source audit, coupling core, and integration/configs—no infrastructure expansion.
