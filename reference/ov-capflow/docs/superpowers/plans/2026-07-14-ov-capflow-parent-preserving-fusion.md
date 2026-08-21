# OV-CapFlow Parent-Preserving Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a freshly initialized HRSC C1 semantic-fusion arm exactly reproduce the C0 parent before training, then admit it to a five-epoch HRSC experiment only after unit, real-batch, and full-replay equivalence gates pass.

**Architecture:** Change semantic fusion from a native-query base to a zero-gated residual on the transported C0 layer output. Keep denoising queries and disabled-fusion behavior unchanged, add a non-overwriting C1 config, and audit raw decoder/head tensors from the same C0 checkpoint before measuring full-dataset AP.

**Tech Stack:** Python 3.8, PyTorch 1.12.1, MMEngine 0.10.4, MMDetection 3.3.0, local MMRotate, pytest, HRSC2016, A40 GPU.

---

## File map

- `projects/OVCapFlow/ov_capflow/semantic_capacity.py`: implement the parent-base semantic residual.
- `tests/test_projects/ov_capflow/test_semantic_capacity.py`: lock zero-gate, signed-gate, capacity, and matching-suffix behavior.
- `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py`: define the repaired arm without overwriting the original C1 work directory.
- `tests/test_projects/ov_capflow/test_hrsc_config.py`: enforce the fixed HRSC protocol and repaired-arm switches.
- `projects/OVCapFlow/tools/audit_parent_equivalence.py`: compare parent-visible tensors from C0 and repaired C1 on one real HRSC batch.
- `tests/test_projects/ov_capflow/test_parent_equivalence.py`: unit-test exact and non-exact tensor reports.
- `.lab/log.md`, `.lab/results.tsv`, `.lab/branches.md`: keep untracked research state and measured decisions.
- `docs/project_history/exp_20260714_hrsc_parent_preserving_fusion/fres_hrsc_parent_preserving_fusion_zh.md`: record the final Chinese result without altering the completed matrix report.

### Task 1: Lock and implement the parent-preserving fusion contract

**Files:**
- Modify: `tests/test_projects/ov_capflow/test_semantic_capacity.py:13-40`
- Modify: `projects/OVCapFlow/ov_capflow/semantic_capacity.py:8-53`

- [ ] **Step 1: Replace the incorrect zero-gate test and add signed residual tests**

Replace `test_identity_fusion_starts_as_exact_native_query` and
`test_zero_capacity_blocks_evidence_even_with_open_gate` with the following,
and add the signed-gate test immediately after them:

```python
def test_zero_gate_starts_as_exact_parent_query():
    fusion = SemanticEvidenceFusion(embed_dims=8, adapter_init='identity')
    native = torch.randn(2, 4, 8)
    parent = torch.randn(2, 4, 8)

    fused, gate = fusion(native, parent)

    assert torch.equal(fused, parent)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_signed_gate_moves_parent_along_native_residual():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    native = torch.tensor([[[3.0, 1.0, -1.0, 2.0]]])
    parent = torch.tensor([[[1.0, 2.0, 1.0, -2.0]]])

    with torch.no_grad():
        fusion.gate.bias.fill_(1.0)
    positive, positive_gate = fusion(native, parent)
    expected_positive = parent + positive_gate * (native - parent)
    torch.testing.assert_close(positive, expected_positive)

    with torch.no_grad():
        fusion.gate.bias.fill_(-1.0)
    negative, negative_gate = fusion(native, parent)
    expected_negative = parent + negative_gate * (native - parent)
    torch.testing.assert_close(negative, expected_negative)


def test_zero_capacity_preserves_parent_even_with_open_gate():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    with torch.no_grad():
        fusion.gate.bias.fill_(1.0)
    native = torch.randn(1, 3, 4)
    parent = torch.randn(1, 3, 4)

    fused, gate = fusion(
        native, parent, capacity=torch.zeros(1, 3))

    assert torch.equal(fused, parent)
    assert torch.equal(gate, torch.zeros_like(gate))
```

- [ ] **Step 2: Add a zero-initialized suffix-equivalence test**

Add this test after `test_matching_suffix_fusion_leaves_denoising_prefix_unchanged`:

```python
def test_zero_initialized_matching_suffix_preserves_complete_parent():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    native = torch.randn(1, 5, 4)
    parent = torch.randn(1, 5, 4)

    output, gate = apply_matching_query_interventions(
        native_query=native,
        transported_query=parent,
        matching_query_count=3,
        fusion=fusion,
        capacity=None)

    assert torch.equal(output, parent)
    assert torch.equal(gate, torch.zeros_like(gate))
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_projects/ov_capflow/test_semantic_capacity.py::test_zero_gate_starts_as_exact_parent_query \
  tests/test_projects/ov_capflow/test_semantic_capacity.py::test_zero_initialized_matching_suffix_preserves_complete_parent \
  -q
```

Expected: both tests fail because the current zero gate returns `native` for
the matching suffix instead of the transported parent tensor.

- [ ] **Step 4: Implement the minimal parent-base residual**

Change only the class docstring and the final projection/base computation in
`SemanticEvidenceFusion`:

```python
class SemanticEvidenceFusion(nn.Module):
    """Apply a gated native-semantic residual to the parent query."""

    def forward(self,
                native_query: Tensor,
                transported_evidence: Tensor,
                capacity: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if native_query.shape != transported_evidence.shape:
            raise ValueError(
                'native_query and transported_evidence must match')
        if native_query.shape[-1] != self.embed_dims:
            raise ValueError(
                'query channel dimension does not match embed_dims')

        native_residual = native_query - transported_evidence
        projected_residual = self.evidence_adapter(native_residual)
        gate = torch.tanh(
            self.gate(torch.cat([native_query, transported_evidence], dim=-1)))
        if capacity is not None:
            if capacity.shape != native_query.shape[:-1]:
                raise ValueError('capacity must have one value per query')
            gate = gate * capacity.unsqueeze(-1)
        fused_query = transported_evidence + gate * projected_residual
        return fused_query, gate
```

- [ ] **Step 5: Run the focused file and verify GREEN**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_semantic_capacity.py -q
```

Expected: all tests in the file pass, including exact `torch.equal` checks.

### Task 2: Add a non-overwriting repaired HRSC arm

**Files:**
- Create: `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py`
- Modify: `tests/test_projects/ov_capflow/test_hrsc_config.py:10-50`
- Modify: `tests/test_projects/ov_capflow/test_hrsc_real_batch.py:17-23`

- [ ] **Step 1: Add the missing config to protocol and switch tests**

Add the repaired config to `test_hrsc_configs_share_fixed_protocol`:

```python
    ('ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py', 'OVCapFlow'),
```

Extend `test_causal_switches_are_single_direction` with:

```python
    repaired = Config.fromfile(
        CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py')
    assert repaired.model.decoder.layer_cfg.enable_semantic_fusion
    assert not repaired.model.decoder.layer_cfg.enable_density_capacity
    assert not repaired.model.bbox_head.balanced_cfg.enabled
    assert not repaired.model.decoder.enable_null_reservoir
    assert repaired.load_from.endswith(
        'hrsc_c0_native_10e/epoch_10.pth')
    assert repaired.work_dir.endswith(
        'hrsc_c1_parent_preserving_5e')
```

Add the repaired filename to the opt-in real-batch parameter list:

```python
    'ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py',
```

Extend the registry import and apply the real freeze hook before the forward
pass for the repaired arm:

```python
from mmrotate.registry import HOOKS, MODELS

    if 'parent_preserving' in config_name:
        freeze_hook_cfg = next(
            item for item in cfg.custom_hooks
            if item.get('type') == 'FreezeExceptHook')
        trainable = HOOKS.build(freeze_hook_cfg).apply(model)
        assert trainable
        assert all('.semantic_fusion.' in name for name in trainable)
```

- [ ] **Step 2: Run the config test and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_projects/ov_capflow/test_hrsc_config.py::test_hrsc_configs_share_fixed_protocol \
  tests/test_projects/ov_capflow/test_hrsc_config.py::test_causal_switches_are_single_direction \
  -q
```

Expected: failure because
`ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py` does not exist.

- [ ] **Step 3: Create the repaired C1 config**

Create the file with exactly:

```python
_base_ = './ov_capflow_swin-t_hrsc_c1_fusion_5e.py'

work_dir = (
    'work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e')
```

The inherited config already fixes the parent checkpoint, seed, five-epoch
budget, disabled balanced/null/density switches, and semantic-only freeze hook.

- [ ] **Step 4: Run the config tests and verify GREEN**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_hrsc_config.py -q
```

Expected: all HRSC config tests pass.

### Task 3: Build an exact real-checkpoint equivalence audit

**Files:**
- Create: `projects/OVCapFlow/tools/audit_parent_equivalence.py`
- Create: `tests/test_projects/ov_capflow/test_parent_equivalence.py`

- [ ] **Step 1: Write tests for exact, changed, and shape-mismatched tensors**

Create the test file:

```python
import torch

from projects.OVCapFlow.tools.audit_parent_equivalence import tensor_report


def test_tensor_report_marks_identical_values_exact():
    value = torch.tensor([[1.0, 2.0]])

    report = tensor_report(value, value.clone())

    assert report == {
        'parent_shape': [1, 2],
        'candidate_shape': [1, 2],
        'exact': True,
        'max_abs_diff': 0.0,
    }


def test_tensor_report_measures_changed_values():
    parent = torch.tensor([[1.0, 2.0]])
    candidate = torch.tensor([[1.0, 2.25]])

    report = tensor_report(parent, candidate)

    assert not report['exact']
    assert report['max_abs_diff'] == 0.25


def test_tensor_report_rejects_shape_mismatch_without_subtraction():
    report = tensor_report(torch.zeros(1, 2), torch.zeros(2, 1))

    assert not report['exact']
    assert report['max_abs_diff'] is None
```

- [ ] **Step 2: Run the audit tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_parent_equivalence.py -q
```

Expected: collection fails because `audit_parent_equivalence.py` does not
exist.

- [ ] **Step 3: Implement the audit tool**

Create `projects/OVCapFlow/tools/audit_parent_equivalence.py`:

```python
import argparse
import copy
import json
from pathlib import Path

import torch
from mmengine import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_config')
    parser.add_argument('candidate_config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def tensor_report(parent, candidate):
    report = {
        'parent_shape': list(parent.shape),
        'candidate_shape': list(candidate.shape),
        'exact': False,
        'max_abs_diff': None,
    }
    if parent.shape != candidate.shape:
        return report
    report['exact'] = bool(torch.equal(parent, candidate))
    report['max_abs_diff'] = float(
        (parent.float() - candidate.float()).abs().max().item())
    return report


def _capture(config_path, checkpoint_path, raw_batch):
    cfg = Config.fromfile(config_path)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False)
    model = model.cuda().eval()
    captured = {}

    def capture_decoder(module, inputs, output):
        captured['decoder_states'] = output[0].detach().cpu()
        captured['references'] = output[1].detach().cpu()

    def capture_head(module, inputs, output):
        captured['classification_scores'] = output[0].detach().cpu()
        captured['rotated_boxes'] = output[1].detach().cpu()

    handles = [
        model.decoder.register_forward_hook(capture_decoder),
        model.bbox_head.register_forward_hook(capture_head),
    ]
    batch = model.data_preprocessor(
        copy.deepcopy(raw_batch), training=False)
    with torch.no_grad():
        predictions = model.predict(
            batch['inputs'], batch['data_samples'], rescale=True)
    for handle in handles:
        handle.remove()

    captured['prediction_counts'] = [
        len(sample.pred_instances) for sample in predictions]
    captured['prediction_scores'] = torch.cat([
        sample.pred_instances.scores.detach().cpu()
        for sample in predictions])
    captured['prediction_labels'] = torch.cat([
        sample.pred_instances.labels.detach().cpu()
        for sample in predictions])
    captured['prediction_boxes'] = torch.cat([
        sample.pred_instances.bboxes.tensor.detach().cpu()
        for sample in predictions])
    del model
    torch.cuda.empty_cache()
    return captured


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    parent_cfg = Config.fromfile(args.parent_config)
    import_modules_from_strings(**parent_cfg.custom_imports)
    parent_cfg.val_dataloader.num_workers = 0
    parent_cfg.val_dataloader.persistent_workers = False
    dataloader = Runner.build_dataloader(parent_cfg.val_dataloader)
    raw_batch = next(iter(dataloader))

    parent = _capture(args.parent_config, args.checkpoint, raw_batch)
    candidate = _capture(args.candidate_config, args.checkpoint, raw_batch)
    tensor_names = [
        'decoder_states',
        'references',
        'classification_scores',
        'rotated_boxes',
        'prediction_scores',
        'prediction_labels',
        'prediction_boxes',
    ]
    comparisons = {
        name: tensor_report(parent[name], candidate[name])
        for name in tensor_names
    }
    parent_counts = parent['prediction_counts']
    candidate_counts = candidate['prediction_counts']
    passed = (
        all(item['exact'] for item in comparisons.values())
        and parent_counts == candidate_counts
        and all(count == parent_cfg.model.num_queries
                for count in parent_counts))
    report = {
        'pass': passed,
        'parent_config': str(Path(args.parent_config).resolve()),
        'candidate_config': str(Path(args.candidate_config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'parent_prediction_counts': parent_counts,
        'candidate_prediction_counts': candidate_counts,
        'comparisons': comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the audit unit tests and verify GREEN**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_parent_equivalence.py -q
```

Expected: `3 passed`.

### Task 4: Verify engineering gates and commit before metric measurement

**Files:**
- Modify: `.lab/log.md`
- Modify: `.lab/branches.md`

- [ ] **Step 1: Run the complete portable OV-CapFlow suite**

Run:

```bash
rtk env MPLCONFIGDIR=/tmp/ovcapflow-mpl \
  /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Expected: all tests pass; only opt-in CUDA tests are skipped.

- [ ] **Step 2: Run the opt-in repaired-C1 real-batch forward/backward test**

Run on an available A40:

```bash
rtk env CUDA_VISIBLE_DEVICES=4 RUN_OVCAPFLOW_INTEGRATION=1 \
  MPLCONFIGDIR=/tmp/ovcapflow-mpl HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_projects/ov_capflow/test_hrsc_real_batch.py \
  -k parent_preserving -q
```

Expected: one repaired-C1 real-batch test passes with finite loss and active
semantic-fusion gradients.

- [ ] **Step 3: Inspect the change set and log the pre-experiment THINK entry**

Run:

```bash
rtk git diff --check
rtk git diff -- projects/OVCapFlow configs/ov_capflow/hrsc tests/test_projects/ov_capflow
```

First append a mandatory `## 3-Discard Guardrail — after Experiment 3` entry
to `.lab/log.md`: C1/C2/C3 were three consecutive valid model discards; this
run continues on the already-created strategy fork because it inverts the
shared native-reset assumption and requires exact C0 equivalence before
training. Then append a `## THINK — before Experiment 4` entry stating that
the confirmed hypothesis is: a transported-query residual with a zero gate
will exactly reproduce C0 at zero update; E1 and E2 invalidate the experiment
if any parent-visible tensor or AP differs. Update the new branch row in
`.lab/branches.md` from `active-design` to `active`.

- [ ] **Step 4: Commit the repository-changing experiment before E1/E2**

Run:

```bash
rtk git add \
  projects/OVCapFlow/ov_capflow/semantic_capacity.py \
  projects/OVCapFlow/tools/audit_parent_equivalence.py \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py \
  tests/test_projects/ov_capflow/test_semantic_capacity.py \
  tests/test_projects/ov_capflow/test_hrsc_config.py \
  tests/test_projects/ov_capflow/test_hrsc_real_batch.py \
  tests/test_projects/ov_capflow/test_parent_equivalence.py
rtk git commit \
  -m "experiment #4: preserve C0 fusion function at initialization" \
  -m "Branch: research/hrsc-parent-preserving-fusion
Parent: #3-C0R
Hypothesis: a zero-gated native residual exactly preserves C0 before training"
```

Expected: one commit containing only the repaired C1 implementation, config,
audit, and tests. The user-owned untracked two-week review directory remains
untouched.

### Task 5: Run E1/E2, then conditionally run the five-epoch HRSC arm

**Files:**
- Modify: `.lab/log.md`
- Modify: `.lab/results.tsv`
- Modify: `.lab/branches.md`
- Create: `docs/project_history/exp_20260714_hrsc_parent_preserving_fusion/fres_hrsc_parent_preserving_fusion_zh.md`

- [ ] **Step 1: Run the real-checkpoint E1 tensor audit**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/ovcapflow-mpl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /data/zcy/anaconda3/envs/mmdet/bin/python \
  projects/OVCapFlow/tools/audit_parent_equivalence.py \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c0_native_10e.py \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py \
  work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e/epoch_10.pth \
  --output work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e/audits/parent_equivalence.json
```

Expected: exit code 0, `pass=true`, all seven tensor comparisons exact with
`max_abs_diff=0.0`, and both prediction-count lists contain 600.

- [ ] **Step 2: Run the E2 full HRSC zero-update replay**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/ovcapflow-mpl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /data/zcy/anaconda3/envs/mmdet/bin/python tools/test.py \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py \
  work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e/epoch_10.pth \
  --work-dir work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e/zero_update_replay
```

Expected: `dota/AP50=0.5870`, recall `0.941`, and no load keys outside the new
semantic-fusion parameters.

- [ ] **Step 3: Run the strict 600-row audit**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/ovcapflow-mpl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /data/zcy/anaconda3/envs/mmdet/bin/python \
  projects/OVCapFlow/tools/audit_strict_inference.py \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py \
  --output work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e/audits/strict_zero_update.json
```

Expected: `pass=true`, 600 rows per image, and zero forbidden calls.

- [ ] **Step 4: Apply the E1/E2 decision gate**

If any E1 tensor differs, AP50 is not exactly `0.5870`, recall is not `0.941`
at report precision, or strict inference fails, log Experiment 4 as
`invalid-engineering`, stop before training, and return to root-cause analysis.

If all gates pass, append the exact audit and replay values to `.lab/log.md`
before launching training.

- [ ] **Step 5: Train the repaired C1 for five epochs**

Run on an available A40:

```bash
rtk .lab/bin/run 5 \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py \
  work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e
```

Expected: completion within 7200 seconds with finite losses and an epoch-5
checkpoint/evaluation.

- [ ] **Step 6: Measure mediator and checkpoint diagnostics**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=5 MPLCONFIGDIR=/tmp/ovcapflow-mpl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /data/zcy/anaconda3/envs/mmdet/bin/python \
  projects/OVCapFlow/tools/hrsc_mediator_metrics.py \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py \
  work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e/epoch_5.pth \
  --output work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e/mediators.json

rtk /data/zcy/anaconda3/envs/mmdet/bin/python \
  projects/OVCapFlow/tools/audit_checkpoint_load.py \
  configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py \
  work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e/epoch_10.pth \
  --output work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e/audits/checkpoint_load.json

rtk sha256sum \
  work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_5e/epoch_5.pth
```

Expected: mediator JSON covers all 600 rows, checkpoint audit has no invalid
missing or unexpected keys, and the checkpoint hash is recorded.

- [ ] **Step 7: Log the decision before any reset**

Copy the measured AP50, recall, coverage, duplicate extras per GT, gate means,
runtime, peak memory, final loss, strict counts, and SHA256 verbatim into
`.lab/log.md` and one Experiment 4 row in `.lab/results.tsv`.

Keep/promote the model only if AP50 is at least `0.5890`, or if it lies in
`[0.5860, 0.5880]`, recall is at least `0.936`, and two preregistered mediator
improvements pass. Otherwise mark the model candidate discarded but record the
repository experiment as `interesting`, retaining the verified
parent-equivalence engineering change because it is simpler and fixes the
proven initialization bug. This avoids a `discard` ledger status without the
mandatory reset.

- [ ] **Step 8: Write the Chinese result report and verify the final state**

Create the report at the file-map path. State the E0-E3 outcomes, exact
metrics, keep/discard decision, and whether any mechanism is eligible for a
second seed or DOTA2. Then run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow -q
rtk git diff --check
rtk git status --short --branch
```

Expected: the portable suite passes, diff checks are clean, and only intended
report/code changes plus the preserved user-owned untracked review directory
appear.

- [ ] **Step 9: Commit the completed research record**

Run:

```bash
rtk git add \
  docs/project_history/exp_20260714_hrsc_parent_preserving_fusion/fres_hrsc_parent_preserving_fusion_zh.md
rtk git commit -m "research complete: validate parent-preserving HRSC fusion"
```

Expected: the branch ends at a complete, auditable HRSC result. No merge,
push, PR, DOTA2 run, second seed, or ten-epoch continuation occurs without a
separate gate decision.
