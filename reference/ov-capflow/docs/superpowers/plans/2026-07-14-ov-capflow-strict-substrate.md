# OV-CapFlow Strict Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a testable strict-E2E Oriented GroundingDINO variant whose matching queries and rotated reference boxes are learned parameters and whose inference returns one class decision per query without global top-k.

**Architecture:** Add OV-CapFlow as an isolated project extension rather than editing vendored RHINO or GroundingDINO classes. A detector subclass overrides pre-decoder query initialization; a head subclass overrides only prediction selection. Training-time denoising remains available, while encoder proposal losses are disabled by passing `None` encoder outputs.

**Tech Stack:** Python 3.8, PyTorch 1.12.1+cu113, MMCV 2.1.0, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate 1.0.0rc1, pytest.

---

### Task 1: Lock strict query initialization behavior

**Files:**
- Create: `tests/test_projects/ov_capflow/test_strict_query_initializer.py`
- Create: `projects/OVCapFlow/ov_capflow/query_initializer.py`

- [ ] **Step 1: Write the failing tests**

```python
import torch

from projects.OVCapFlow.ov_capflow.query_initializer import (
    FixedRotatedQueryInitializer,
)


def test_fixed_initializer_returns_all_queries_and_five_dimensional_refs():
    module = FixedRotatedQueryInitializer(num_queries=9, embed_dims=16)
    query, refs = module(batch_size=2)
    assert query.shape == (2, 9, 16)
    assert refs.shape == (2, 9, 5)
    assert torch.all((refs > 0) & (refs < 1))


def test_fixed_initializer_is_independent_of_encoder_memory():
    module = FixedRotatedQueryInitializer(num_queries=4, embed_dims=8)
    query_a, refs_a = module(batch_size=1)
    query_b, refs_b = module(batch_size=1)
    torch.testing.assert_close(query_a, query_b)
    torch.testing.assert_close(refs_a, refs_b)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_strict_query_initializer.py -q`

Expected: collection fails because `projects.OVCapFlow.ov_capflow` does not yet exist.

- [ ] **Step 3: Implement the minimal initializer**

Create an `nn.Module` with learned content and five-dimensional reference
embeddings. Initialize reference logits from a square center grid, width and
height `1 / ceil(sqrt(Q))`, and normalized angle `0.5`; clamp probabilities
before applying `torch.logit`.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_strict_query_initializer.py -q`

Expected: `2 passed`.

### Task 2: Remove encoder proposal selection from decoder initialization

**Files:**
- Create: `tests/test_projects/ov_capflow/test_strict_detector.py`
- Create: `projects/OVCapFlow/ov_capflow/ov_capflow.py`
- Create: `projects/OVCapFlow/ov_capflow/__init__.py`
- Create: `projects/OVCapFlow/__init__.py`

- [ ] **Step 1: Write a failing pre-decoder test**

Construct the detector with `object.__new__`, initialize its `nn.Module`
state, attach a real `FixedRotatedQueryInitializer`, set evaluation mode, and
call `pre_decoder` with synthetic image and text memories. Assert matching
query/reference shapes and assert that `enc_outputs_class` and
`enc_outputs_coord` are absent at inference.

- [ ] **Step 2: Run the test and verify RED**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_strict_detector.py -q`

Expected: import fails because `OVCapFlow` is undefined.

- [ ] **Step 3: Implement `OVCapFlow`**

Subclass `RotatedGroundingDINO`. In `_init_layers`, call the parent and replace
the inherited content embedding with `FixedRotatedQueryInitializer`. In
`pre_decoder`, never call `gen_encoder_output_proposals`, encoder classifier,
encoder regressor, `torch.topk`, or `torch.gather`. Concatenate denoising state
only during training and pass `None` encoder outputs to the head.

- [ ] **Step 4: Run the detector and initializer tests**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_strict_query_initializer.py tests/test_projects/ov_capflow/test_strict_detector.py -q`

Expected: all tests pass.

### Task 3: Remove global prediction top-k

**Files:**
- Create: `tests/test_projects/ov_capflow/test_strict_head.py`
- Create: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`

- [ ] **Step 1: Write failing per-query selection tests**

Use three queries and four class logits. Assert that the selection helper
returns exactly three scores, three labels, and the unchanged three box rows.
Include a case where two queries select the same class to prove that no NMS or
deduplication is performed.

- [ ] **Step 2: Run the test and verify RED**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_strict_head.py -q`

Expected: import fails because `OVCapFlowHead` is undefined.

- [ ] **Step 3: Implement `OVCapFlowHead`**

Subclass `RotatedGroundingDINOHead`. Convert grounding logits to class logits
when positive maps exist, choose `max` only along the class dimension, retain
every query row, denormalize the direct 5-D rotated boxes, and return one
record per query. Do not read `max_per_img`.

- [ ] **Step 4: Run all OV-CapFlow tests**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow -q`

Expected: all tests pass.

### Task 4: Register a minimal strict configuration

**Files:**
- Create: `configs/ov_capflow/ov_capflow_swin-t_strict_visdrone.py`
- Create: `projects/OVCapFlow/README.md`

- [ ] **Step 1: Write a failing config-build test**

Load the config with `mmengine.Config.fromfile`, import the custom project,
replace pretrained paths with `None`, reduce encoder/decoder layers for the
unit build, and call `MODELS.build`. Assert `model.__class__.__name__ ==
'OVCapFlow'` and `model.bbox_head.__class__.__name__ == 'OVCapFlowHead'`.

- [ ] **Step 2: Run the build test and verify RED**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_config_build.py -q`

Expected: config file not found.

- [ ] **Step 3: Add the inherited config and usage documentation**

Inherit the upstream Oriented GroundingDINO config, replace custom imports,
set detector/head types, retain 900 fixed queries, and document the exact
environment and raw-prediction invariants.

- [ ] **Step 4: Run the build test and full project tests**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow -q`

Expected: all tests pass.

### Task 5: Verify the strict substrate before semantic-capacity work

**Files:**
- Modify: `docs/superpowers/specs/2026-07-14-ov-capflow-design.md`

- [ ] **Step 1: Scan production inference paths for forbidden selection**

Run: `rtk rg -n "torch\.topk|\.topk\(|nms|multiclass_nms|minAreaRect" projects/OVCapFlow configs/ov_capflow`

Expected: no matches in executable project code; documentation mentions are
allowed and reviewed manually.

- [ ] **Step 2: Run import and project tests**

Run: `rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow -q`

Expected: all tests pass with no collection errors.

- [ ] **Step 3: Record the environment and upstream commit**

Append the verified interpreter/package versions, upstream commit, test count,
and remaining unimplemented phases to the architecture specification. Do not
claim semantic transport or density capacity is active until their separate
test-driven implementation plans have passed.

