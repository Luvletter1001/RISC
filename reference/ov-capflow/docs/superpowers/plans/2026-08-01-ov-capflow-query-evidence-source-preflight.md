# OV-CapFlow Query Evidence Source Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one compact, frozen-E24 audit that decides whether the pre-registered stride-32 encoder evidence can reach enough canonical Dense400 geometry misses to justify any production implementation of OV-CapFlow.

**Architecture:** Keep the detector, checkpoint, prompts, and predictions frozen. A single tracked audit tool hooks the existing encoder output and existing text-conditioned classification branch, extracts only feature level index 2, assigns at most 600 valid spatial tokens per image by a deterministic no-GT evidence rule, and evaluates those locations with the already established oriented center-inside geometry definition. The tool publishes a no-replace JSON report containing provenance, strata, placebos, and five fail-closed gates. It does not add a detector component, change a config, train a model, or launch distributed work.

**Tech Stack:** Python 3.8.19, PyTorch 1.12.1+cu113, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate 1.0.0rc1, pytest, the existing OV-CapFlow E24 model/analyzer assets, and one A40 GPU for read-only inference.

---

## Frozen contract

This plan implements only Milestones M0 and M1 of the approved design. Plan 2 is forbidden unless this plan emits `status: PASS` and the report SHA256 is archived.

The following values are fixed before execution:

| Field | Frozen value |
|---|---|
| Design commit | `9dee60d80c562a998348ce822e19dbd42ba99cb9` |
| Parent checkpoint | `work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth` |
| Parent SHA256 | `a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8` |
| Canonical dump | `work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl` |
| Dump SHA256 | `39c1d6d3193cbc9e7f8d4e6daace9fc95df86c0d0fc7f1be7870cb13ee092c45` |
| Dense400 manifest | `docs/project_history/exp_20260723_e24_route_audit/evidence/dense400_manifest.json` |
| Dense400 manifest SHA256 | `cf187eedba4f19703a77475639887147e4e1285b28f7db4c91686c0296030594` |
| Proxy400 manifest | `work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json` |
| Proxy400 manifest SHA256 | `a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e` |
| Rare4x train manifest | `work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/train/manifest.json` |
| Rare4x train manifest SHA256 | `1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290` |
| Encoder level | zero-based index `2`, stride `32`, normally `32 x 32 = 1024` tokens at scale 1024 |
| Evidence budget | at most `600` valid spatial tokens per image |
| Main evaluation set | Dense400; 400 images and 123,888 GT objects |
| Empty-image diagnostic | the fixed 120 empty tiles in proxy400 |
| Prior any-label ceilings | T `0.145148646`, E `0.210711449` |
| Prior same-label E reference | `0.203709` |
| Spatial shuffle seed | `2026071601` |
| Semantic shuffle seed | `2026071602` |
| Output directory | `.lab/workspace/exp-8-qaf-source-v1` |
| Report | `.lab/workspace/exp-8-qaf-source-v1/source_report.json` |

The primary reachability metric is any-label geometry reachability. For a canonical geometry-miss GT and a selected evidence center, rotate the normalized center displacement into the GT frame and count the GT reached iff

```text
max(2 * abs(dx_local) / gt_width,
    2 * abs(dy_local) / gt_height) <= 1.
```

Same-label/text-compatible reachability is reported separately against `0.203709`; it is not silently substituted for the approved any-label gate.

The five source gates are evaluated exactly as:

```python
gates = {
    'reachability_at_least_25pct': overall >= 0.25,
    'beats_prior_e_by_3pp': overall - 0.210711449 >= 0.03,
    'placebo_sensitivity': (
        overall - spatial_shuffle >= 0.05 and
        overall - semantic_shuffle >= 0.05),
    'novel_base_gap_at_most_10pp': base14 - novel4 <= 0.10,
    'finite_reproducible_provenance': finite and hashes_match,
}
status = 'PASS' if all(gates.values()) else 'FAIL'
```

The report keeps `spatial_shuffle_loses_5pp` and `semantic_shuffle_loses_5pp` as two named subchecks under `placebo_sensitivity`, but there are exactly five top-level source gates.

The first two gates are intentionally both retained even though the 25% threshold is the stricter numerical bound. Scientific failure exits with code `3` after publishing the report. Infrastructure/provenance failure exits with code `2`. PASS exits with code `0`.

## Scope and commit accounting

This plan consumes concentrated commit 1 of the approved three-commit pre-proxy budget.

Files changed by this plan:

- Modify: `.lab/results.tsv`
- Modify: `progress.md`
- Modify: `task_plan.md`
- Modify: `projects/OVCapFlow/README.md`
- Create: `projects/OVCapFlow/tools/audit_query_evidence_source.py`
- Create: `tests/test_projects/ov_capflow/test_query_evidence_source.py`

It must not create a model class, experiment config, launcher, queue manager, generalized audit platform, result database, or a third test file. Existing untracked files are user-owned and must not be staged.

## Task 1: Freeze governance and write the failing audit tests

**Files:**

- Modify: `.lab/results.tsv`
- Modify: `progress.md`
- Modify: `task_plan.md`
- Modify: `projects/OVCapFlow/README.md`
- Create: `tests/test_projects/ov_capflow/test_query_evidence_source.py`

- [ ] **Step 1: Verify the selected environment and immutable inputs**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -c "import sys,torch,mmengine,mmdet,mmrotate; print(sys.version.split()[0], torch.__version__, mmengine.__version__, mmdet.__version__, mmrotate.__version__)"
rtk sha256sum work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl docs/project_history/exp_20260723_e24_route_audit/evidence/dense400_manifest.json work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/train/manifest.json
```

Expected first line:

```text
3.8.19 1.12.1+cu113 0.10.4 3.3.0 1.0.0rc1
```

Expected hashes are exactly the values in the frozen-contract table. Any mismatch is an infrastructure/provenance stop; do not continue by changing the registered inputs.

- [ ] **Step 2: Correct the four governance surfaces with `apply_patch`**

Append these two tab-separated records to `.lab/results.tsv` without altering prior rows:

```text
8-D158-D13N-FINAL	research/dotav2-cleanstart-ov-e2e-ap70	8-D157-D13N-SPEC	2b326af802d1561d324aed90b1db7c09689b0636	negative	proxy1600x400;endpoint_updates=1920;control_mAP=0.847486973;candidate_mAP=0.838305831;delta=-0.009181142;intermediate_E2_delta=0.0004875;endpoint_fixed=true	close-no-scale-no-rescue-no-stack	0	D13-N is formally negative; the E2 excursion is tiny and cannot be selected.
8-D159-QAF-SPEC	research/dotav2-cleanstart-ov-e2e-ap70	8-D158-D13N-FINAL	9dee60d80c562a998348ce822e19dbd42ba99cb9	design-approved	approach=unified-query-allocation-quality-flow;source_gate_first=true;production_forbidden_before_pass=true;fixed_q=600;strict_all_row=true	implement-frozen-source-preflight-only	0	Approach A is the only active branch; all prior adapter branches remain closed.
```

Add one `2026-08-01` section to both `progress.md` and `task_plan.md` stating exactly:

```text
- D13-N closed at the registered E12 endpoint: 0.838305831 candidate versus 0.847486973 control, delta -0.009181142.
- Active hypothesis: one query-to-image evidence coupling jointly governs center transport, null validity, matching/training quality, and all-row score calibration.
- Current authorized action: frozen E24 stride-32 source preflight only.
- Hard stop: no production model/config code and no distributed training before all five source gates pass.
```

Add the same active-branch statement under a `Research status (2026-08-01)` heading in `projects/OVCapFlow/README.md`. Preserve all existing history.

- [ ] **Step 3: Add unit tests for the exact public audit API**

The test file begins with `import json`, `import numpy as np`, `import pytest`, and `import torch`, followed by only these project-level public imports:

```python
from projects.OVCapFlow.tools.audit_query_evidence_source import (
    SOURCE_SCHEMA,
    SourceGateThresholds,
    build_level_centers,
    choose_evidence_indices,
    format_results_tsv_row,
    geometry_reachability,
    make_placebo_logits,
    normalized_points_to_original_pixels,
    require_file_sha256,
    require_unique_image_ids,
    semantic_token_evidence,
    source_gate_decision,
    validate_level_token_count,
    write_report_no_replace,
)
```

Write tests covering all of the following concrete behaviors:

```python
def test_level_centers_are_raster_ordered_and_normalized():
    centers = build_level_centers((2, 2), torch.tensor([[1.0, 1.0]]))
    expected = torch.tensor([[[0.25, 0.25], [0.75, 0.25],
                              [0.25, 0.75], [0.75, 0.75]]])
    assert torch.equal(centers, expected)


def test_evidence_selection_is_stable_masked_and_capped_at_600():
    evidence = torch.arange(0, 605, dtype=torch.float32)[None]
    valid = torch.ones_like(evidence, dtype=torch.bool)
    valid[:, 604] = False
    selected = choose_evidence_indices(evidence, valid, budget=600)
    assert selected.shape == (1, 600)
    assert 604 not in selected[0].tolist()
    assert selected[0, 0].item() == 603


def test_semantic_evidence_masks_padding_before_max():
    logits = torch.tensor([[[1.0, -torch.inf], [2.0, -torch.inf]]])
    text_mask = torch.tensor([[True, False]])
    assert torch.equal(semantic_token_evidence(logits, text_mask),
                       torch.tensor([[1.0, 2.0]]))


def test_oriented_center_inside_definition_matches_dense400():
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
    centers = torch.tensor([[0.59, 0.50], [0.61, 0.50]])
    reached = geometry_reachability(gt, centers)
    assert reached.tolist() == [True]


def test_normalized_points_use_resized_shape_and_scale_factor():
    points = np.array([[0.5, 0.25]], dtype=np.float32)
    pixels = normalized_points_to_original_pixels(
        points, img_shape=(800, 1000), scale_factor=(0.5, 0.5))
    np.testing.assert_array_equal(
        pixels, np.array([[1000.0, 400.0]], dtype=np.float32))


def test_placebos_are_deterministic_and_break_distinct_associations():
    logits = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    centers = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    a = make_placebo_logits(logits, centers, 'spatial', 2026071601)
    b = make_placebo_logits(logits, centers, 'spatial', 2026071601)
    semantic = make_placebo_logits(logits, centers, 'semantic', 2026071602)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[1], centers)
    assert torch.equal(semantic[1], centers)
    assert not torch.equal(semantic[0], logits)


def test_source_gate_boundary_is_inclusive_and_fail_closed():
    t = SourceGateThresholds()
    passing = dict(overall=0.260711449,
                   spatial_shuffle=0.210711449,
                   semantic_shuffle=0.210711449,
                   base14=0.30, novel4=0.20,
                   finite=True, hashes_match=True)
    decision = source_gate_decision(passing, t)
    assert decision['status'] == 'PASS'
    failing = {**passing, 'semantic_shuffle': 0.210711449002}
    assert source_gate_decision(failing, t)['status'] == 'FAIL'


def test_report_is_canonical_no_replace_and_schema_locked(tmp_path):
    path = tmp_path / 'source_report.json'
    payload = {
        'schema': SOURCE_SCHEMA,
        'status': 'PASS',
        'decision': {
            'gates': {
                'reachability_at_least_25pct': True,
                'beats_prior_e_by_3pp': True,
                'placebo_sensitivity': True,
                'novel_base_gap_at_most_10pp': True,
                'finite_reproducible_provenance': True,
            },
            'failed_gates': [],
        },
        'frozen_contract': {},
        'provenance': {},
        'counts': {},
        'reachability': {},
        'placebos': {},
        'strata': {},
        'empty_images': {},
        'finite_checks': {},
    }
    first_sha = write_report_no_replace(path, payload)
    assert len(first_sha) == 64
    with pytest.raises(FileExistsError):
        write_report_no_replace(path, payload)


def test_results_row_uses_external_report_digest():
    payload = {'schema': SOURCE_SCHEMA, 'status': 'PASS',
               'decision': {'gates': {'finite': True}},
               'reachability': {'overall': 0.31}}
    digest = 'a' * 64
    row = format_results_tsv_row(payload, digest)
    assert row.startswith('8-D160-QAF-SOURCE\t')
    assert digest in row
```

Append these exact fail-closed tests; they deliberately exercise only APIs defined in Task 2:

```python
def test_nonfinite_logits_fail_closed():
    logits = torch.tensor([[[float('nan'), 0.0]]])
    with pytest.raises(ValueError, match='finite'):
        semantic_token_evidence(logits, torch.tensor([[True, True]]))


def test_no_valid_prompt_token_fails_closed():
    with pytest.raises(ValueError, match='valid prompt token'):
        semantic_token_evidence(
            torch.zeros(1, 2, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_wrong_level_token_count_fails_closed():
    with pytest.raises(ValueError, match='level token count'):
        validate_level_token_count((32, 32), observed_token_count=1023)


def test_manifest_hash_mismatch_fails_closed(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    with pytest.raises(ValueError, match='SHA256'):
        require_file_sha256(manifest, '0' * 64)


def test_missing_gate_field_fails_closed():
    metrics = dict(overall=0.31, spatial_shuffle=0.20,
                   semantic_shuffle=0.20, base14=0.30,
                   finite=True, hashes_match=True)
    with pytest.raises(ValueError, match='novel4'):
        source_gate_decision(metrics, SourceGateThresholds())


def test_duplicate_image_id_fails_closed():
    with pytest.raises(ValueError, match='duplicate image id'):
        require_unique_image_ids(['P0001', 'P0001'])


def test_padded_text_columns_never_enter_evidence():
    full = torch.tensor([[[2.0, 1.0, 999.0, 999.0]]])
    runtime_mask = torch.tensor([[True, True]])
    reduced = semantic_token_evidence(full[..., :2], runtime_mask)
    assert torch.equal(reduced, torch.tensor([[2.0]]))


def test_padded_spatial_tokens_never_convert_to_pixels():
    evidence = torch.tensor([[1.0, 2.0, 999.0, 3.0]])
    valid = torch.tensor([[True, True, False, True]])
    selected = choose_evidence_indices(evidence, valid, budget=2)
    assert selected.tolist() == [[3, 1]]
    centers = np.array([[0.75, 0.75], [0.25, 0.25]], dtype=np.float32)
    pixels = normalized_points_to_original_pixels(
        centers, img_shape=(1024, 1024), scale_factor=(1.0, 1.0))
    assert pixels.shape == (2, 2)


def test_scientific_fail_still_writes_complete_report(tmp_path):
    metrics = dict(overall=0.20, spatial_shuffle=0.19,
                   semantic_shuffle=0.19, base14=0.20, novel4=0.20,
                   finite=True, hashes_match=True)
    decision = source_gate_decision(metrics, SourceGateThresholds())
    payload = {
        'schema': SOURCE_SCHEMA, 'status': decision['status'],
        'decision': decision, 'frozen_contract': {}, 'provenance': {},
        'counts': {}, 'reachability': metrics, 'placebos': {},
        'strata': {}, 'empty_images': {}, 'finite_checks': {},
    }
    digest = write_report_no_replace(tmp_path / 'source_report.json', payload)
    assert decision['status'] == 'FAIL'
    assert len(digest) == 64
    assert json.loads((tmp_path / 'source_report.json').read_text()) == payload
```

- [ ] **Step 4: Run the tests and verify RED for missing implementation only**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_evidence_source.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'projects.OVCapFlow.tools.audit_query_evidence_source'`. Any unrelated failure must be diagnosed before implementation.

## Task 2: Implement the compact frozen-source audit

**Files:**

- Create: `projects/OVCapFlow/tools/audit_query_evidence_source.py`
- Test: `tests/test_projects/ov_capflow/test_query_evidence_source.py`

- [ ] **Step 1: Implement deterministic pure functions first**

Use `apply_patch` to create the tool with these exact imports and constants:

```python
import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


SOURCE_SCHEMA = 'ov-capflow-query-evidence-source-v1'
LEVEL_INDEX = 2
TOKEN_BUDGET = 600
PRIOR_T = 0.145148646
PRIOR_E = 0.210711449
PRIOR_E_SAME_LABEL = 0.203709
SPATIAL_SHUFFLE_SEED = 2026071601
SEMANTIC_SHUFFLE_SEED = 2026071602


@dataclass(frozen=True)
class SourceGateThresholds:
    min_reachability: float = 0.25
    min_prior_margin: float = 0.03
    min_placebo_drop: float = 0.05
    max_base_novel_gap: float = 0.10
```

Insert the following exact pure-function implementation below those constants. It is complete for the unit-test surface; live model loading stays in Step 2:

```python
def _require_finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f'{name} must be finite')


def build_level_centers(spatial_shape, valid_ratios):
    height, width = (int(value) for value in spatial_shape)
    if height <= 0 or width <= 0:
        raise ValueError('spatial shape must be positive')
    ratios = torch.as_tensor(valid_ratios, dtype=torch.float32)
    if ratios.ndim != 2 or ratios.shape[1] != 2:
        raise ValueError('valid_ratios must have shape [B,2]')
    _require_finite(ratios, 'valid_ratios')
    if not torch.all((ratios > 0) & (ratios <= 1)).item():
        raise ValueError('valid_ratios must lie in (0,1]')
    ys = (torch.arange(height, device=ratios.device, dtype=torch.float32)
          + 0.5) / height
    xs = (torch.arange(width, device=ratios.device, dtype=torch.float32)
          + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    raster = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
    return raster / ratios[:, None, :]


def semantic_token_evidence(token_logits, text_token_mask):
    logits = torch.as_tensor(token_logits).float()
    mask = torch.as_tensor(text_token_mask, dtype=torch.bool,
                           device=logits.device)
    if logits.ndim != 3 or mask.ndim != 2:
        raise ValueError('token logits/mask must have shapes [B,S,T]/[B,T]')
    if logits.shape[0] != mask.shape[0] or logits.shape[2] != mask.shape[1]:
        raise ValueError('token logits and text mask shapes disagree')
    if not mask.any(dim=1).all().item():
        raise ValueError('every sample needs a valid prompt token')
    valid_values = logits.masked_select(mask[:, None, :].expand_as(logits))
    _require_finite(valid_values, 'valid token logits')
    return logits.masked_fill(~mask[:, None, :], -torch.inf).amax(dim=-1)


def choose_evidence_indices(evidence, valid_mask, budget=TOKEN_BUDGET):
    values = torch.as_tensor(evidence).float()
    valid = torch.as_tensor(valid_mask, dtype=torch.bool,
                            device=values.device)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError('evidence and valid mask must share shape [B,S]')
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError('budget must be a positive integer')
    _require_finite(values, 'evidence')
    valid_counts = valid.sum(dim=1)
    if not torch.all(valid_counts > 0).item():
        raise ValueError('each sample needs a valid spatial token')
    count = min(budget, values.shape[1], int(valid_counts.min().item()))
    selected = []
    for batch_index in range(values.shape[0]):
        candidates = torch.nonzero(valid[batch_index], as_tuple=False)[:, 0]
        order = sorted(
            candidates.tolist(),
            key=lambda index: (-float(values[batch_index, index]), index))
        selected.append(torch.tensor(
            order[:count], dtype=torch.long, device=values.device))
    return torch.stack(selected, dim=0)


def make_placebo_logits(token_logits, centers, mode, seed):
    logits = torch.as_tensor(token_logits).clone()
    points = torch.as_tensor(centers).clone()
    if logits.ndim != 3 or points.ndim != 3:
        raise ValueError('placebo inputs must have shapes [B,S,T]/[B,S,2]')
    if logits.shape[:2] != points.shape[:2] or points.shape[2] != 2:
        raise ValueError('placebo input shapes disagree')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    spatial_count, token_count = logits.shape[1], logits.shape[2]
    if mode == 'spatial':
        for batch_index in range(logits.shape[0]):
            permutation = torch.randperm(
                spatial_count, generator=generator).to(points.device)
            points[batch_index] = points[batch_index, permutation]
    elif mode == 'semantic':
        for batch_index in range(logits.shape[0]):
            for token_index in range(token_count):
                permutation = torch.randperm(
                    spatial_count, generator=generator).to(logits.device)
                logits[batch_index, :, token_index] = \
                    logits[batch_index, permutation, token_index]
    elif mode == 'uniform':
        logits.zero_()
    else:
        raise ValueError(f'unknown placebo mode: {mode}')
    return logits, points


def geometry_reachability(gt_rboxes, evidence_centers):
    boxes = torch.as_tensor(gt_rboxes).float()
    centers = torch.as_tensor(evidence_centers).float()
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError('gt_rboxes must have shape [G,5]')
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('evidence_centers must have shape [S,2]')
    _require_finite(boxes, 'gt_rboxes')
    _require_finite(centers, 'evidence centers')
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)
    if centers.numel() == 0:
        return torch.zeros(boxes.shape[0], dtype=torch.bool,
                           device=boxes.device)
    delta = centers[None, :, :] - boxes[:, None, :2]
    cosine = torch.cos(boxes[:, 4])[:, None]
    sine = torch.sin(boxes[:, 4])[:, None]
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    normalized = torch.maximum(
        2 * local_x.abs() / boxes[:, None, 2],
        2 * local_y.abs() / boxes[:, None, 3])
    return (normalized <= 1).any(dim=1)


def normalized_points_to_original_pixels(points, img_shape, scale_factor):
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError('points must be a finite [N,2] array')
    img_height, img_width = int(img_shape[0]), int(img_shape[1])
    scales = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
    if scales.size not in (2, 4) or not np.isfinite(scales).all():
        raise ValueError('scale_factor must contain two or four finite values')
    scale_x, scale_y = float(scales[0]), float(scales[1])
    if img_height <= 0 or img_width <= 0 or scale_x <= 0 or scale_y <= 0:
        raise ValueError('image shape and scale factors must be positive')
    result = array.copy()
    result[:, 0] = result[:, 0] * img_width / scale_x
    result[:, 1] = result[:, 1] * img_height / scale_y
    return result


def source_gate_decision(metrics, thresholds):
    required = ('overall', 'spatial_shuffle', 'semantic_shuffle',
                'base14', 'novel4', 'finite', 'hashes_match')
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f'missing source gate field: {missing[0]}')
    numeric = {name: float(metrics[name]) for name in required[:5]}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError('source gate metrics must be finite')
    epsilon = 1e-12
    gates = {
        'reachability_at_least_25pct': (
            numeric['overall'] + epsilon >= thresholds.min_reachability),
        'beats_prior_e_by_3pp': (
            numeric['overall'] - PRIOR_E + epsilon >=
            thresholds.min_prior_margin),
        'placebo_sensitivity': (
            numeric['overall'] - numeric['spatial_shuffle'] + epsilon >=
            thresholds.min_placebo_drop and
            numeric['overall'] - numeric['semantic_shuffle'] + epsilon >=
            thresholds.min_placebo_drop),
        'novel_base_gap_at_most_10pp': (
            numeric['base14'] - numeric['novel4'] <=
            thresholds.max_base_novel_gap + epsilon),
        'finite_reproducible_provenance': (
            metrics['finite'] is True and metrics['hashes_match'] is True),
    }
    return {
        'status': 'PASS' if all(gates.values()) else 'FAIL',
        'gates': gates,
        'placebo_subchecks': {
            'spatial_shuffle_loses_5pp': (
                numeric['overall'] - numeric['spatial_shuffle'] + epsilon >=
                thresholds.min_placebo_drop),
            'semantic_shuffle_loses_5pp': (
                numeric['overall'] - numeric['semantic_shuffle'] + epsilon >=
                thresholds.min_placebo_drop),
        },
        'failed_gates': [name for name, passed in gates.items() if not passed],
    }


def write_report_no_replace(path, payload):
    required = {'schema', 'status', 'decision', 'frozen_contract',
                'provenance', 'counts', 'reachability', 'placebos', 'strata',
                'empty_images', 'finite_checks'}
    if set(payload) != required or payload['schema'] != SOURCE_SCHEMA:
        raise ValueError('source report schema is incomplete or changed')
    encoded = (json.dumps(payload, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def format_results_tsv_row(payload, report_sha256):
    gates = payload['decision']['gates']
    gate_text = ','.join(
        f'{name}={str(value).lower()}' for name, value in sorted(gates.items()))
    report_path = payload.get('provenance', {}).get(
        'report_path',
        '.lab/workspace/exp-8-qaf-source-v1/source_report.json')
    return ('8-D160-QAF-SOURCE\tresearch/dotav2-cleanstart-ov-e2e-ap70\t'
            '8-D159-QAF-SPEC\t9dee60d80c562a998348ce822e19dbd42ba99cb9\t'
            f"{payload['status'].lower()}\t{gate_text};"
            f"overall={payload['reachability']['overall']};"
            f'report={report_path};'
            f'report_sha256={report_sha256}\t'
            'source-gate-decision\t0\tfrozen E24 source preflight\n')
```

The public function signatures are exact:

- `build_level_centers(spatial_shape: Tuple[int, int], valid_ratios: Tensor) -> Tensor`
- `semantic_token_evidence(token_logits: Tensor, text_token_mask: Tensor) -> Tensor`
- `choose_evidence_indices(evidence: Tensor, valid_mask: Tensor, budget: int = TOKEN_BUDGET) -> Tensor`
- `make_placebo_logits(token_logits: Tensor, centers: Tensor, mode: str, seed: int) -> Tuple[Tensor, Tensor]`
- `geometry_reachability(gt_rboxes: Tensor, evidence_centers: Tensor) -> Tensor`
- `normalized_points_to_original_pixels(points: np.ndarray, img_shape, scale_factor) -> np.ndarray`
- `source_gate_decision(metrics: Mapping[str, object], thresholds: SourceGateThresholds) -> Dict[str, object]`
- `write_report_no_replace(path: Path, payload: Mapping[str, object]) -> str`
- `format_results_tsv_row(payload: Mapping[str, object], report_sha256: str) -> str`
- `validate_level_token_count(spatial_shape: Tuple[int, int], observed_token_count: int) -> None`
- `require_file_sha256(path: Path, expected_sha256: str) -> str`
- `require_unique_image_ids(image_ids: Sequence[str]) -> None`

Implementation rules:

- Perform evidence reductions in float32.
- Evaluate inclusive decimal gate boundaries with a fixed `1e-12` comparison epsilon and the module constant `PRIOR_E`; never trust a prior-ceiling value supplied in runtime metrics.
- Mask invalid text tokens with `-torch.inf` before `amax`; reject a sample with no valid prompt token.
- Stable selection is descending evidence, then ascending raster index for exact ties.
- `spatial` placebo applies one seeded permutation to centers while leaving the full `[S,T]` evidence matrix fixed.
- `semantic` placebo independently permutes the `S` rows of each valid prompt-token column before the `amax`, while leaving centers fixed. A global class-column permutation is forbidden because `amax` would make it a no-op.
- `uniform` placebo assigns zero evidence to all valid tokens and therefore selects ascending raster indices deterministically.
- Use the same angle convention and center-inside computation as `projects/OVCapFlow/tools/analyze_dense400_center_availability.py`; do not invent an axis-aligned approximation.
- Serialize JSON with `sort_keys=True`, `indent=2`, `allow_nan=False`, a final newline, exclusive-create mode `x`, and SHA256 of the exact bytes.

Use these complete validation helpers; the live path calls them before any metric accumulation:

```python
def validate_level_token_count(spatial_shape, observed_token_count):
    height, width = (int(value) for value in spatial_shape)
    expected = height * width
    if height <= 0 or width <= 0 or observed_token_count != expected:
        raise ValueError(
            f'level token count mismatch: expected {expected}, '
            f'got {observed_token_count}')


def require_file_sha256(path, expected_sha256):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected SHA256 must be lowercase hex')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f'SHA256 mismatch for {path}: {digest}')
    return digest


def require_unique_image_ids(image_ids):
    seen = set()
    for image_id in image_ids:
        if not isinstance(image_id, str) or not image_id:
            raise ValueError('image id must be a non-empty string')
        if image_id in seen:
            raise ValueError(f'duplicate image id: {image_id}')
        seen.add(image_id)
```

- [ ] **Step 2: Implement live E24 extraction by reusing the current model path**

The CLI is one end-to-end command with these required arguments and defaults:

```text
--config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py
--proxy-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x.py
--checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth
--canonical-dump work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl
--dense-manifest docs/project_history/exp_20260723_e24_route_audit/evidence/dense400_manifest.json
--proxy-manifest work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json
--train-manifest work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/train/manifest.json
--level-index 2
--token-budget 600
--spatial-shuffle-seed 2026071601
--semantic-shuffle-seed 2026071602
--output .lab/workspace/exp-8-qaf-source-v1/source_report.json
--device cuda:0
```

Use the tracked model/data initialization path below and call `pre_transformer` plus `forward_encoder` directly; the production tool does not import or depend on the historical `.lab` capture. The direct encoder call exposes:

```python
memory: Tensor                 # [B, sum(H_l*W_l), 256]
memory_mask: Tensor            # [B, sum(H_l*W_l)]
spatial_shapes: Tensor         # [4, 2]
level_start_index: Tensor      # [4]
valid_ratios: Tensor           # [B,4,2]
memory_text: Tensor            # existing encoded prompt tokens
text_token_mask: Tensor        # valid prompt tokens
full_token_logits: Tensor      # existing frozen cls branch, [B,S_level,256]
token_logits: Tensor           # exact slice [..., :text_token_mask.shape[1]]
```

Slice memory only at `[level_start_index[2]:level_start_index[3]]`, and build centers with `build_level_centers(spatial_shapes[2], valid_ratios[:, 2])`. Slice the classification output's padded token dimension to the runtime text-mask length before any mask or maximum, and assert padded columns never enter evidence or class diagnostics. Do not call encoder proposal regression, generate boxes, or emit detections. The existing classification branch is evaluated frozen to derive token evidence; all model parameters must have `requires_grad=False` and no optimizer may be constructed.

Add the following live implementation below the validation helpers. This is
the complete model, dataset, join, aggregation, report, and CLI path; there is
no runtime import from `.lab`:

```python
EXPECTED_SHA256 = {
    'config': 'f47af897b26c85a6f30fe6ecbc0a59144622d2e4e64cdd0e9346f037a0e31691',
    'proxy_config': 'f5cbb5d8e216d5a3f9d34e74596275ca03fd638e8b85c55486deb819e7d19c5d',
    'checkpoint': 'a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8',
    'canonical_dump': '39c1d6d3193cbc9e7f8d4e6daace9fc95df86c0d0fc7f1be7870cb13ee092c45',
    'dense_manifest': 'cf187eedba4f19703a77475639887147e4e1285b28f7db4c91686c0296030594',
    'proxy_manifest': 'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e',
    'train_manifest': '1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290',
    'center_analyzer': '07ecd6cdce83c012987883f5ce6f7b9ea4c9d60de30253bc8fdd1139454866b6',
    'diagnostics_core': 'bd29dc88650e0db545479c60adaa6a9ee64da1734d3a1756c7cd787c7269c92d',
}


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain one JSON object')
    return payload


def _load_config(path):
    from mmengine import Config
    from mmengine.utils import import_modules_from_strings
    from mmrotate.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(path))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    return cfg


def _build_dataset(cfg):
    from mmengine.runner import Runner

    return Runner.build_dataloader(cfg.val_dataloader).dataset


def _load_frozen_model(cfg, checkpoint_path, device):
    from mmrotate.registry import MODELS

    model = MODELS.build(cfg.model).to(device).eval()
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            'checkpoint mismatch: missing={} unexpected={}'.format(
                incompatible.missing_keys, incompatible.unexpected_keys))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('frozen source audit found a trainable parameter')
    return model


def _dataset_image_id(dataset, index):
    info = dataset.get_data_info(int(index))
    image_id = info.get('img_id')
    if image_id is None:
        image_id = Path(info['img_path']).stem
    return str(image_id)


def _make_replay_batch(model, dataset, dataset_index, batch_size):
    from mmengine.dataset import pseudo_collate

    if batch_size < 1:
        raise ValueError('replay batch size must be positive')
    raw = pseudo_collate(
        [dataset[int(dataset_index)] for _ in range(batch_size)])
    return model.data_preprocessor(raw, training=False)


def _prepare_prompt(model, batch_inputs, batch_data_samples):
    text_prompts = [sample.text for sample in batch_data_samples]
    enhanced = [sample.get('caption_prompt', None)
                for sample in batch_data_samples]
    tokens_positive = [sample.get('tokens_positive', None)
                       for sample in batch_data_samples]
    custom_entities = batch_data_samples[0].get('custom_entities', False)
    prepared = [
        model.get_tokens_positive_and_prompts(
            prompt, custom_entities, enhancement, positive)
        for prompt, enhancement, positive in zip(
            text_prompts, enhanced, tokens_positive)
    ]
    positive_maps, captions, _, entities = zip(*prepared)
    if any(isinstance(caption, list) for caption in captions):
        raise ValueError('source preflight forbids chunked prompts')
    text_dict = model.language_model(list(captions))
    if model.text_feat_map is not None:
        text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])
    for sample, positive_map in zip(batch_data_samples, positive_maps):
        if positive_map is None:
            raise ValueError('source preflight requires a class positive map')
        sample.token_positive_map = positive_map
    if len(batch_inputs) != len(positive_maps):
        raise ValueError('prompt batch and image batch disagree')
    return text_dict, positive_maps, entities


def _canonical_positive_signature(positive_map, entities):
    canonical_map = {
        str(int(label)): [int(value) for value in torch.as_tensor(
            token_ids).reshape(-1).tolist()]
        for label, token_ids in sorted(
            positive_map.items(), key=lambda item: int(item[0]))
    }
    encoded = json.dumps(
        {'positive_map': canonical_map,
         'class_order': [str(value) for value in entities]},
        sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _gather_rows(values, indices):
    suffix = values.shape[2:]
    gather_index = indices.reshape(
        indices.shape + (1,) * len(suffix)).expand(indices.shape + suffix)
    return torch.gather(values, 1, gather_index)


def _extract_registered_level(model, batch, level_index, budget,
                              spatial_seed, semantic_seed):
    from projects.OVCapFlow.ov_capflow.calibration import (
        grounding_logits_to_class_log_scores,
    )

    batch_inputs = batch['inputs']
    samples = batch['data_samples']
    with torch.no_grad():
        text_dict, positive_maps, entities = _prepare_prompt(
            model, batch_inputs, samples)
        visual_features = model.extract_feat(batch_inputs)
        encoder_inputs, _ = model.pre_transformer(
            visual_features, samples)
        encoder_outputs = model.forward_encoder(
            **encoder_inputs, text_dict=text_dict)

        memory = encoder_outputs['memory']
        memory_mask = encoder_outputs['memory_mask']
        if memory_mask is None:
            memory_mask = torch.zeros(
                memory.shape[:2], dtype=torch.bool, device=memory.device)
        spatial_shapes = encoder_inputs['spatial_shapes']
        level_start_index = encoder_inputs['level_start_index']
        valid_ratios = encoder_inputs['valid_ratios']
        memory_text = encoder_outputs['memory_text']
        text_token_mask = encoder_outputs['text_token_mask']
        if (spatial_shapes.shape != (4, 2) or
                level_start_index.shape != (4,) or
                valid_ratios.shape != (memory.shape[0], 4, 2)):
            raise ValueError('encoder geometry does not match four levels')

        height = int(spatial_shapes[level_index, 0].item())
        width = int(spatial_shapes[level_index, 1].item())
        start = int(level_start_index[level_index].item())
        stop = start + height * width
        if (level_index + 1 < len(level_start_index) and
                stop != int(level_start_index[level_index + 1].item())):
            raise ValueError('registered level offsets disagree')
        level_memory = memory[:, start:stop]
        level_mask = memory_mask[:, start:stop]
        validate_level_token_count((height, width), level_memory.shape[1])
        centers = build_level_centers(
            (height, width), valid_ratios[:, level_index])
        valid = (~level_mask & (centers >= 0).all(dim=-1) &
                 (centers <= 1).all(dim=-1))

        full_token_logits = model.bbox_head.cls_branches[
            model.decoder.num_layers](
                level_memory, memory_text, text_token_mask)
        text_length = int(text_token_mask.shape[1])
        if (full_token_logits.shape[:2] != level_memory.shape[:2] or
                full_token_logits.shape[2] < text_length):
            raise ValueError('classification branch shape mismatch')
        token_logits = full_token_logits[..., :text_length]

        variants = {
            'primary': (token_logits, centers),
            'spatial_shuffle': make_placebo_logits(
                token_logits, centers, 'spatial', spatial_seed),
            'semantic_shuffle': make_placebo_logits(
                token_logits, centers, 'semantic', semantic_seed),
            'uniform': make_placebo_logits(
                token_logits, centers, 'uniform', semantic_seed),
        }
        results = {}
        for name, (variant_logits, variant_centers) in variants.items():
            evidence = semantic_token_evidence(
                variant_logits, text_token_mask)
            selected = choose_evidence_indices(evidence, valid, budget)
            selected_logits = _gather_rows(variant_logits, selected)
            selected_centers = _gather_rows(variant_centers, selected)
            normalized_mass = torch.softmax(
                evidence.masked_fill(~valid, -torch.inf), dim=-1)
            selected_mass = torch.gather(
                normalized_mass, 1, selected).sum(dim=-1)
            labels = []
            for batch_index, positive_map in enumerate(positive_maps):
                class_logs = grounding_logits_to_class_log_scores(
                    selected_logits[batch_index], positive_map)
                labels.append(class_logs.argmax(dim=-1))
            results[name] = {
                'centers': selected_centers,
                'labels': torch.stack(labels),
                'evidence': torch.gather(evidence, 1, selected),
                'selected_mass': selected_mass,
                'selected_count': int(selected.shape[1]),
            }
        signature = _canonical_positive_signature(
            positive_maps[0], entities[0])
        if any(_canonical_positive_signature(mapping, entity) != signature
               for mapping, entity in zip(positive_maps, entities)):
            raise ValueError('prompt mapping changed inside a replay batch')
    return results, signature


def _new_counter(modes, strata):
    return {
        mode: {stratum: {'hit': 0, 'same_hit': 0, 'total': 0}
               for stratum in strata}
        for mode in modes
    }


def _rate(hit, total):
    if total <= 0:
        raise ValueError('registered reachability denominator is empty')
    return float(hit / total)


def _stratum_masks(record, geometry_ids, classes, base_classes,
                   novel_classes):
    labels = record.gt_labels[geometry_ids].astype(np.int64, copy=False)
    names = np.asarray([classes[int(label)] for label in labels])
    boxes = record.gt_boxes[geometry_ids]
    sqrt_area = np.sqrt(boxes[:, 2] * boxes[:, 3])
    return {
        'overall': np.ones(len(geometry_ids), dtype=np.bool_),
        'le8px': sqrt_area <= 8.0,
        'small_vehicle': names == 'small-vehicle',
        'gt_count_over_600': np.full(
            len(geometry_ids), len(record.gt_boxes) > 600, dtype=np.bool_),
        'base14': np.isin(names, tuple(base_classes)),
        'novel4': np.isin(names, tuple(novel_classes)),
    }


def _accumulate_dense(args, model, dataset, records, dense_manifest,
                      class_manifest):
    from projects.OVCapFlow.tools.analyze_dense400_center_availability import (
        normalized_center_distances,
    )
    from projects.OVCapFlow.tools.analyze_dense400_center_controls import (
        _record_edges,
    )

    modes = ('primary', 'spatial_shuffle', 'semantic_shuffle', 'uniform')
    strata = ('overall', 'le8px', 'small_vehicle',
              'gt_count_over_600', 'base14', 'novel4')
    counters = _new_counter(modes, strata)
    selected_counts = []
    selected_mass = []
    already_reachable_mass = []
    signature = None
    replay_batch_size = int(args.replay_batch_size)

    entries = dense_manifest['records']
    if len(entries) != 400 or len(records) != 400:
        raise ValueError('Dense400 must contain exactly 400 records')
    require_unique_image_ids([str(entry['img_id']) for entry in entries])
    for entry, record in zip(entries, records):
        image_id = str(entry['img_id'])
        dataset_index = int(entry['dataset_index'])
        if (_dataset_image_id(dataset, dataset_index) != image_id or
                record.img_id != image_id or
                len(record.gt_boxes) != int(entry['gt_count'])):
            raise ValueError(f'Dense400 join mismatch for {image_id}')
        batch = _make_replay_batch(
            model, dataset, dataset_index, replay_batch_size)
        observed = [str(sample.img_id) for sample in batch['data_samples']]
        if any(value != image_id for value in observed):
            raise ValueError(f'preprocessed image id mismatch for {image_id}')
        extracted, observed_signature = _extract_registered_level(
            model, batch, args.level_index, args.token_budget,
            args.spatial_shuffle_seed, args.semantic_shuffle_seed)
        if signature is None:
            signature = observed_signature
        elif signature != observed_signature:
            raise ValueError('prompt mapping changed across Dense400')

        meta = batch['data_samples'][0].metainfo
        geometry = _record_edges(record, 0.5)['geometry_miss']
        geometry_ids = np.flatnonzero(geometry)
        masks = _stratum_masks(
            record, geometry_ids, class_manifest['classes'],
            class_manifest['base_classes'],
            class_manifest['novel_classes'])
        gt_boxes = record.gt_boxes[geometry_ids]
        gt_labels = record.gt_labels[geometry_ids]
        for mode in modes:
            result = extracted[mode]
            points = normalized_points_to_original_pixels(
                result['centers'][0].float().cpu().numpy(),
                meta['img_shape'], meta['scale_factor'])
            labels = result['labels'][0].cpu().numpy()
            distances = normalized_center_distances(points, gt_boxes)
            inside = distances <= 1.0
            any_reached = inside.any(axis=1)
            same_reached = (
                inside & (gt_labels[:, None] == labels[None, :])).any(axis=1)
            for stratum, mask in masks.items():
                counters[mode][stratum]['hit'] += int(
                    np.count_nonzero(any_reached & mask))
                counters[mode][stratum]['same_hit'] += int(
                    np.count_nonzero(same_reached & mask))
                counters[mode][stratum]['total'] += int(np.count_nonzero(mask))

            if mode == 'primary':
                selected_counts.append(result['selected_count'])
                selected_mass.append(float(result['selected_mass'][0].item()))
                reachable_gt = record.gt_boxes[~geometry]
                if len(reachable_gt):
                    occupied = (normalized_center_distances(
                        points, reachable_gt) <= 1.0).any(axis=0)
                    mass = result['selected_mass'].new_tensor(
                        result['evidence'][0]).softmax(dim=0)
                    already_reachable_mass.append(float(
                        mass[torch.as_tensor(occupied, device=mass.device)]
                        .sum().item()))
    return {
        'counters': counters,
        'selected_counts': selected_counts,
        'selected_mass': selected_mass,
        'already_reachable_mass': already_reachable_mass,
        'positive_map_sha256': signature,
    }


def _proxy_empty_diagnostic(args, model, proxy_dataset, proxy_manifest,
                            expected_signature):
    entries = [entry for entry in proxy_manifest['records']
               if entry.get('s1_split') == 'val']
    registered_ids = [str(value) for value in proxy_manifest['s1']['val_stems']]
    entry_ids = [str(entry['stem']) for entry in entries]
    dataset_ids = [
        _dataset_image_id(proxy_dataset, index)
        for index in range(len(proxy_dataset))
    ]
    require_unique_image_ids(registered_ids)
    require_unique_image_ids(dataset_ids)
    if entry_ids != registered_ids or set(dataset_ids) != set(registered_ids):
        raise ValueError('proxy validation identities differ from manifest')
    dataset_index_by_id = {
        image_id: index for index, image_id in enumerate(dataset_ids)
    }
    empty_entries = [entry for entry in entries
                     if int(entry['gt_count']) == 0]
    if len(empty_entries) != 120:
        raise ValueError('proxy manifest must register exactly 120 empties')
    masses = []
    counts = []
    for entry in empty_entries:
        image_id = str(entry['stem'])
        dataset_index = dataset_index_by_id[image_id]
        batch = _make_replay_batch(
            model, proxy_dataset, dataset_index, 1)
        if str(batch['data_samples'][0].img_id) != image_id:
            raise ValueError(f'proxy preprocessing join failed for {image_id}')
        extracted, signature = _extract_registered_level(
            model, batch, args.level_index, args.token_budget,
            args.spatial_shuffle_seed, args.semantic_shuffle_seed)
        if signature != expected_signature:
            raise ValueError('proxy prompt mapping differs from Dense400')
        primary = extracted['primary']
        masses.append(float(primary['selected_mass'][0].item()))
        counts.append(primary['selected_count'])
    return {
        'registered_empty_images': 120,
        'evaluated_empty_images': len(masses),
        'selected_softmax_mass_mean': float(np.mean(masses)),
        'selected_softmax_mass_min': float(np.min(masses)),
        'selected_softmax_mass_max': float(np.max(masses)),
        'selected_tokens_min': int(min(counts)),
        'selected_tokens_max': int(max(counts)),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(
        'configs/ov_capflow/dotav2/'
        'ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_'
        'scale1024_rare4x_gpu89_batch2.py'))
    parser.add_argument('--proxy-config', type=Path, default=Path(
        'configs/ov_capflow/dotav2/'
        'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
        'scale1024_batch1_rare4x.py'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/'
        'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/'
        'epoch_24.pth'))
    parser.add_argument('--canonical-dump', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/'
        'eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl'))
    parser.add_argument('--dense-manifest', type=Path, default=Path(
        'docs/project_history/exp_20260723_e24_route_audit/evidence/'
        'dense400_manifest.json'))
    parser.add_argument('--proxy-manifest', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json'))
    parser.add_argument('--train-manifest', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/'
        'train/manifest.json'))
    parser.add_argument('--level-index', type=int, default=LEVEL_INDEX)
    parser.add_argument('--token-budget', type=int, default=TOKEN_BUDGET)
    parser.add_argument('--spatial-shuffle-seed', type=int,
                        default=SPATIAL_SHUFFLE_SEED)
    parser.add_argument('--semantic-shuffle-seed', type=int,
                        default=SEMANTIC_SHUFFLE_SEED)
    parser.add_argument('--output', type=Path, default=Path(
        '.lab/workspace/exp-8-qaf-source-v1/source_report.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replay-batch-size', type=int, default=2)
    parser.add_argument('--verify-only', action='store_true')
    return parser


def run_source_preflight(args):
    from projects.OVCapFlow.tools.analyze_dense400_center_controls import (
        load_dense400,
    )

    if (args.level_index != LEVEL_INDEX or
            args.token_budget != TOKEN_BUDGET or
            args.spatial_shuffle_seed != SPATIAL_SHUFFLE_SEED or
            args.semantic_shuffle_seed != SEMANTIC_SHUFFLE_SEED):
        raise ValueError('CLI differs from the frozen source contract')
    sidecar = args.output.with_name('source_results.tsv')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('source report or sidecar already exists')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('preflight output directory must pre-exist')

    paths = {
        'config': args.config,
        'proxy_config': args.proxy_config,
        'checkpoint': args.checkpoint,
        'canonical_dump': args.canonical_dump,
        'dense_manifest': args.dense_manifest,
        'proxy_manifest': args.proxy_manifest,
        'train_manifest': args.train_manifest,
        'center_analyzer': Path(
            'projects/OVCapFlow/tools/analyze_dense400_center_controls.py'),
        'diagnostics_core': Path(
            'projects/OVCapFlow/tools/dotav2_q600_diagnostics.py'),
    }
    provenance_files = {}
    for name, path in paths.items():
        digest = require_file_sha256(path, EXPECTED_SHA256[name])
        provenance_files[name] = {'path': str(path), 'sha256': digest}

    dense_manifest = _read_json(args.dense_manifest)
    proxy_manifest = _read_json(args.proxy_manifest)
    records, loaded_manifest = load_dense400(
        args.canonical_dump, args.dense_manifest)
    if loaded_manifest != dense_manifest:
        raise ValueError('Dense400 manifest changed during load')
    if (sum(len(record.gt_boxes) for record in records) != 123888 or
            sum(int(_record['gt_count'])
                for _record in dense_manifest['records']) != 123888):
        raise ValueError('Dense400 GT anchor must be exactly 123888')

    cfg = _load_config(args.config)
    proxy_cfg = _load_config(args.proxy_config)
    model = _load_frozen_model(cfg, args.checkpoint, args.device)
    dense_dataset = _build_dataset(cfg)
    proxy_dataset = _build_dataset(proxy_cfg)
    dense = _accumulate_dense(
        args, model, dense_dataset, records, dense_manifest, proxy_manifest)
    primary_total = dense['counters']['primary']['overall']['total']
    if primary_total != 61068:
        raise ValueError('canonical geometry-miss anchor must be 61068')
    empty = _proxy_empty_diagnostic(
        args, model, proxy_dataset, proxy_manifest,
        dense['positive_map_sha256'])

    primary = dense['counters']['primary']
    metrics = {
        'overall': _rate(primary['overall']['hit'],
                         primary['overall']['total']),
        'spatial_shuffle': _rate(
            dense['counters']['spatial_shuffle']['overall']['hit'],
            primary['overall']['total']),
        'semantic_shuffle': _rate(
            dense['counters']['semantic_shuffle']['overall']['hit'],
            primary['overall']['total']),
        'base14': _rate(primary['base14']['hit'], primary['base14']['total']),
        'novel4': _rate(primary['novel4']['hit'], primary['novel4']['total']),
        'finite': True,
        'hashes_match': True,
    }
    decision = source_gate_decision(metrics, SourceGateThresholds())
    strata = {
        name: {
            'reached': values['hit'],
            'same_label_reached': values['same_hit'],
            'total': values['total'],
            'reachability': _rate(values['hit'], values['total']),
            'same_label_reachability': _rate(
                values['same_hit'], values['total']),
        }
        for name, values in primary.items()
    }
    payload = {
        'schema': SOURCE_SCHEMA,
        'status': decision['status'],
        'decision': decision,
        'frozen_contract': {
            'design_commit':
            '9dee60d80c562a998348ce822e19dbd42ba99cb9',
            'encoder_level_index': LEVEL_INDEX,
            'encoder_stride': 32,
            'token_budget': TOKEN_BUDGET,
            'prior_t': PRIOR_T,
            'prior_e': PRIOR_E,
            'prior_e_same_label': PRIOR_E_SAME_LABEL,
            'spatial_shuffle_seed': SPATIAL_SHUFFLE_SEED,
            'semantic_shuffle_seed': SEMANTIC_SHUFFLE_SEED,
        },
        'provenance': {
            'command': [sys.executable] + sys.argv,
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'cuda_device': str(args.device),
            'report_path': str(args.output),
            'files': provenance_files,
            'positive_map_and_class_order_sha256':
            dense['positive_map_sha256'],
        },
        'counts': {
            'dense_images': len(records),
            'dense_gt': 123888,
            'geometry_miss_gt': primary_total,
            'selected_tokens_min': int(min(dense['selected_counts'])),
            'selected_tokens_max': int(max(dense['selected_counts'])),
            'empty_proxy_images': empty['evaluated_empty_images'],
        },
        'reachability': {
            'overall': metrics['overall'],
            'same_label': _rate(
                primary['overall']['same_hit'], primary_total),
            'prior_t': PRIOR_T,
            'prior_e': PRIOR_E,
            'prior_e_same_label': PRIOR_E_SAME_LABEL,
        },
        'placebos': {
            mode: {
                'reachability': _rate(
                    dense['counters'][mode]['overall']['hit'], primary_total),
                'same_label_reachability': _rate(
                    dense['counters'][mode]['overall']['same_hit'],
                    primary_total),
            }
            for mode in ('spatial_shuffle', 'semantic_shuffle', 'uniform')
        },
        'strata': strata,
        'empty_images': empty,
        'finite_checks': {
            'all_metrics_finite': all(math.isfinite(float(value))
                                      for value in metrics.values()),
            'selected_evidence_mass_mean': float(np.mean(
                dense['selected_mass'])),
            'already_reachable_selected_mass_mean': float(np.mean(
                dense['already_reachable_mass'])),
            'all_parameters_frozen': True,
            'optimizer_constructed': False,
            'detections_emitted': False,
        },
    }
    report_sha256 = write_report_no_replace(args.output, payload)
    with sidecar.open('x', encoding='utf-8') as stream:
        stream.write(format_results_tsv_row(payload, report_sha256))
    return payload, report_sha256


def verify_existing_report(path):
    report_bytes = Path(path).read_bytes()
    payload = json.loads(report_bytes)
    required = {'schema', 'status', 'decision', 'frozen_contract',
                'provenance', 'counts', 'reachability', 'placebos', 'strata',
                'empty_images', 'finite_checks'}
    gates = payload.get('decision', {}).get('gates', {})
    if (set(payload) != required or payload.get('schema') != SOURCE_SCHEMA or
            len(gates) != 5 or
            any(type(value) is not bool for value in gates.values())):
        raise ValueError('stored source report violates the locked schema')
    expected_status = 'PASS' if all(gates.values()) else 'FAIL'
    if payload.get('status') != expected_status:
        raise ValueError('stored source status disagrees with its gates')
    digest = hashlib.sha256(report_bytes).hexdigest()
    sidecar = Path(path).with_name('source_results.tsv').read_text(
        encoding='utf-8')
    if f'report_sha256={digest}' not in sidecar:
        raise ValueError('stored sidecar does not bind the report digest')
    print(json.dumps({
        'status': expected_status,
        'gates': gates,
        'placebo_subchecks': payload['decision']['placebo_subchecks'],
        'report_sha256': digest,
    }, sort_keys=True))
    return 0 if expected_status == 'PASS' else 3


def main():
    args = build_parser().parse_args()
    try:
        if args.verify_only:
            return verify_existing_report(args.output)
        payload, digest = run_source_preflight(args)
    except Exception as error:
        print(f'infrastructure/provenance failure: {error}', file=sys.stderr)
        return 2
    print(json.dumps(
        {'status': payload['status'], 'report_sha256': digest},
        sort_keys=True))
    return 0 if payload['status'] == 'PASS' else 3


if __name__ == '__main__':
    raise SystemExit(main())
```

Primary Dense400 processing must:

1. Load canonical predictions and reproduce the established geometry-miss set.
2. Assert the Dense400 image-id order and GT counts against the manifest.
3. Extract level-2 evidence on each image without GT entering selection.
4. Select at most 600 valid token centers and convert their normalized resized-image coordinates to original pixels with the tracked `normalized_points_to_original_pixels` function defined above: multiply x/y by resized `img_w/img_h`, then divide by positive x/y `scale_factor`; padded/masked tokens never enter conversion.
5. Compute any-label reachability for overall, `<=8 px`, small-vehicle, `>600 GT/image`, base14, and novel4 strata.
6. Compute same-label/text-compatible reachability as a separately named diagnostic. Treat each selected spatial token as one query, call the existing `grounding_logits_to_class_log_scores(selected_token_logits, token_positive_map)`, and take its unchanged class argmax; this freezes prompt-token aggregation, tie handling, and class order to the production mouth. The positive map and class order are hashed in report provenance.
7. Repeat selection for spatial, semantic, and uniform placebos.
8. Report evidence mass on already-reachable GT.

Proxy400 processing builds the existing rare4x config's validation dataloader, proves a unique image-id bijection against `--proxy-manifest`, then reorders through that bijection into the manifest's registered sequence and filters exactly its 120 empty tiles for `empty_evidence_concentration`. It must not assume the dataset's internal order equals manifest order, enter the primary reachability denominator, or tune any threshold.

- [ ] **Step 3: Lock provenance and fail-closed report semantics**

The report contains no NaN values and has exactly these top-level keys: `schema`, `status`, `decision`, `frozen_contract`, `provenance`, `counts`, `reachability`, `placebos`, `strata`, `empty_images`, and `finite_checks`. `decision` contains `gates` and `failed_gates`. `frozen_contract` contains the literal design commit, level, stride, budget, prior ceilings, and seeds listed above. `provenance` contains the exact command/environment plus path and SHA256 records for the checkpoint, canonical dump, and three manifests.

After `write_report_no_replace` returns the SHA256 of the final report bytes, call `format_results_tsv_row(payload, report_sha256)` and write its output with exclusive-create mode to `.lab/workspace/exp-8-qaf-source-v1/source_results.tsv`. The row begins with experiment id `8-D160-QAF-SOURCE`, records PASS or FAIL, every gate boolean, the report path, and the external report SHA256. The report never attempts to contain its own digest.

- [ ] **Step 4: Run focused tests to GREEN**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_query_evidence_source.py -q
```

Expected: all tests pass. The exact test count is allowed to grow while satisfying the listed cases; no test may be skipped.

- [ ] **Step 5: Run static scope checks**

Run:

```bash
rtk rg -n "topk|nms|roi_head|rpn_head|dense_head|gen_encoder_output_proposals" projects/OVCapFlow/tools/audit_query_evidence_source.py
rtk git diff --check
rtk git status --short
```

Expected:

- No inference threshold/filter, NMS, RoI, RPN, dense detector, or encoder-proposal path appears; numeric source-gate thresholds remain confined to `source_gate_decision`.
- The focused AST test, not this text search, proves selection/sorting occurs only inside the preflight audit's registered 600-token selector.
- `git diff --check` is silent.
- Only the six planned files are staged later; unrelated untracked files remain untouched.

- [ ] **Step 6: Create concentrated commit 1**

Run:

```bash
rtk git add .lab/results.tsv progress.md task_plan.md projects/OVCapFlow/README.md projects/OVCapFlow/tools/audit_query_evidence_source.py tests/test_projects/ov_capflow/test_query_evidence_source.py
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "audit: add frozen query evidence source gate"
```

Expected: one commit containing only the six listed files. Do not use `git add -A`.

## Task 3: Run the one-GPU source audit

**Files:**

- Generate, no-replace: `.lab/workspace/exp-8-qaf-source-v1/source_report.json`
- Generate, no-replace: `.lab/workspace/exp-8-qaf-source-v1/source_results.tsv`
- Generate: `.lab/workspace/exp-8-qaf-source-v1/source_run.log`

- [ ] **Step 1: Verify GPU 0 is free enough for read-only extraction**

Run:

```bash
rtk nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Proceed on GPU 0 only if it has at least 20 GiB free. If it does not, wait; do not silently switch the pre-registered source scale or occupy ten GPUs.

- [ ] **Step 2: Run the audit once**

Run:

```bash
rtk run "test ! -e .lab/workspace/exp-8-qaf-source-v1"
rtk mkdir .lab/workspace/exp-8-qaf-source-v1
rtk bash -o pipefail -c 'rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow OMP_NUM_THREADS=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_query_evidence_source.py --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py --proxy-config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1_rare4x.py --checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --canonical-dump work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl --dense-manifest docs/project_history/exp_20260723_e24_route_audit/evidence/dense400_manifest.json --proxy-manifest work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json --train-manifest work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/train/manifest.json --level-index 2 --token-budget 600 --spatial-shuffle-seed 2026071601 --semantic-shuffle-seed 2026071602 --output .lab/workspace/exp-8-qaf-source-v1/source_report.json --device cuda:0 2>&1 | rtk tee .lab/workspace/exp-8-qaf-source-v1/source_run.log'
```

Expected terminal outcome is exactly one of:

- exit `0`, complete report, `status: PASS`;
- exit `3`, complete report, `status: FAIL`;
- exit `2`, provenance/infrastructure failure and no scientific decision.

Do not rerun into the same output path. A scientific FAIL cannot be converted into PASS by changing level, budget, seed, subset, or threshold.

- [ ] **Step 3: Verify the produced report independently**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m json.tool .lab/workspace/exp-8-qaf-source-v1/source_report.json
rtk sha256sum .lab/workspace/exp-8-qaf-source-v1/source_report.json .lab/workspace/exp-8-qaf-source-v1/source_results.tsv .lab/workspace/exp-8-qaf-source-v1/source_run.log
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_query_evidence_source.py --verify-only --output .lab/workspace/exp-8-qaf-source-v1/source_report.json
```

Expected: JSON parses, `--verify-only` prints the same status, five top-level gates, and both placebo subchecks, and exits `0` only for PASS or `3` only for scientific FAIL.

## Task 4: Record the branch decision and enforce the handoff gate

**Files:**

- Modify after observation: `.lab/results.tsv`
- Modify after observation: `progress.md`
- Modify after observation: `task_plan.md`
- Modify after observation: `projects/OVCapFlow/README.md`

- [ ] **Step 1: Apply the canonical observed record**

Copy the exact line from `.lab/workspace/exp-8-qaf-source-v1/source_results.tsv` into `.lab/results.tsv` using `apply_patch`. In the three Markdown governance files, paste the report status, external report SHA256, overall/base14/novel4/spatial-shuffle/semantic-shuffle reachabilities, and failed-gate list exactly as serialized. Do not round gate arithmetic before making the decision.

- [ ] **Step 2: Branch only on the immutable decision**

If `status` is `FAIL`:

- mark Approach A closed in all three Markdown governance files;
- state that no production module, config, proxy run, raw run, scale rescue, or alternate evidence level is authorized;
- stage only the four observed governance surfaces with `rtk git add .lab/results.tsv progress.md task_plan.md projects/OVCapFlow/README.md`, verify them with `rtk git diff --cached --check` and `rtk git diff --cached --stat`, then commit with `rtk git commit -m "results: close query flow at source gate"`;
- stop. Do not execute Plan 2.

If `status` is `PASS`:

- record the report path and SHA256 as the sole Plan-2 prerequisite;
- commit the observed PASS record together with Task 1 of Plan 2, so the total remains within the three concentrated commits;
- proceed to Plan 2 without changing any frozen source parameter.

If exit code was `2`, fix only the demonstrated infrastructure/provenance cause and rerun to a new versioned output directory after review. Do not treat infrastructure failure as a scientific result.

## Final verification checklist

- [ ] One tracked audit tool and one focused test file exist; no production model/config exists.
- [ ] Governance names D13-N negative and Approach A as the sole active branch.
- [ ] E24, dump, Dense400, proxy400, and rare4x manifest hashes match the frozen contract.
- [ ] Evidence uses level index 2 only and no GT enters evidence selection.
- [ ] Exactly 600 or fewer valid spatial centers are considered per image.
- [ ] Any-label, same-label, all requested strata, three placebos, empty images, and already-reachable GT are reported.
- [ ] Spatial and semantic placebos break different associations.
- [ ] The report is finite, canonical, no-replace, and independently verified.
- [ ] A FAIL closes the family; only a PASS report SHA unlocks Plan 2.
- [ ] Commit 1 contains only the six registered source/governance files.
