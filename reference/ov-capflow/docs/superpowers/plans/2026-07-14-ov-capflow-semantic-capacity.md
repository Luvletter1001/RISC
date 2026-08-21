# OV-CapFlow Semantic Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve each matching query's native semantic state while fusing image/text evidence under a continuous scene-density capacity signal.

**Architecture:** Implement two small state modules before touching the decoder: a zero-gated semantic residual with identity/orthogonal adapter initialization, and a differentiable capacity estimator with global count prediction. After isolated tests pass, a GroundingDINO decoder-layer subclass applies them only to the final `Q` matching queries, leaving training-only denoising queries on the upstream path.

**Tech Stack:** PyTorch, MMDetection transformer layers, CastDet Oriented GroundingDINO, pytest.

---

### Task 1: Semantic-preserving fusion

**Files:**
- Create: `tests/test_projects/ov_capflow/test_semantic_capacity.py`
- Create: `projects/OVCapFlow/ov_capflow/semantic_capacity.py`

- [ ] Write tests proving identity initialization returns native queries exactly, orthogonal initialization is orthogonal, and zero capacity blocks transported evidence.
- [ ] Run the focused test and observe import failure.
- [ ] Implement `SemanticEvidenceFusion` with `tanh` gate initialized to zero and an identity/orthogonal evidence adapter.
- [ ] Run the focused test and obtain a pass.

### Task 2: Continuous density capacity

**Files:**
- Modify: `tests/test_projects/ov_capflow/test_semantic_capacity.py`
- Modify: `projects/OVCapFlow/ov_capflow/semantic_capacity.py`

- [ ] Write tests proving capacities stay strictly between zero and one, padded memory does not affect global count, and mass/count losses backpropagate.
- [ ] Run the focused test and observe the expected missing-class failure.
- [ ] Implement `ContinuousDensityCapacity` and `density_capacity_losses` without thresholding, sorting, or top-k.
- [ ] Run the focused test and obtain a pass.

### Task 3: Matching-query decoder integration

**Files:**
- Create: `projects/OVCapFlow/ov_capflow/ov_capflow_layers.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Modify: `tests/test_projects/ov_capflow/test_semantic_capacity.py`

- [ ] Write a test around the fusion helper used by the layer proving a denoising prefix is unchanged and only the final matching queries are fused.
- [ ] Run the test and observe the expected missing-helper failure.
- [ ] Implement `OVCapFlowDecoderLayer`/`OVCapFlowDecoder`; pass immutable `native_query` and matching-query count through decoder kwargs.
- [ ] Add density and capacity mass losses in `OVCapFlow.loss` using stored last-layer continuous states.
- [ ] Run all OV-CapFlow tests.

### Task 4: Verify invariants

**Files:**
- Modify: `docs/superpowers/specs/2026-07-14-ov-capflow-design.md`

- [ ] Run YAPF diff, focused pytest, and forbidden-selection scan.
- [ ] Record exactly which semantic/capacity paths are active and any model-build limitation; do not claim a training run before a real batch succeeds.
