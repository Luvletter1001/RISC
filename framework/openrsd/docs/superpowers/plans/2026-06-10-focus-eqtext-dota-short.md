# FOCUS-EQText DOTA Short Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run the minimal detector-level FOCUS-EQText experiment for the five requested DOTA variants.

**Architecture:** Reuse the existing P1A spatial pseudo-region realbatch path. Add a zero-init Fourier-conditioned text residual adapter, a capped visual/text dual-fusion gate, EQText auxiliary losses, and a negative text bank, then expose them through variant configs and scripts 93-97.

**Tech Stack:** PyTorch, OpenRSD/mmrotate, existing FOCUS modules, P1A realbatch scripts, native DINOv2/A10 support pkl.

---

### Task 1: Module Tests

**Files:**
- Create: `tests/test_focus_eqtext_modules.py`

- [ ] Add tests for text adapter zero-init normalized identity, alpha cap, class masking, and diagnostics.
- [ ] Add tests for dual fusion disabled/zero-alpha exact visual fallback and text-weight cap.
- [ ] Add tests for text anchor, consistency, negative margin, prototype separation, and negative-bank guardrails.
- [ ] Run: `rtk env PYTHONNOUSERSITE=1 python3 -m pytest -o addopts='' tests/test_focus_eqtext_modules.py -q`
- [ ] Expected before implementation: import/file-not-found failures.

### Task 2: EQText Modules

**Files:**
- Create: `M_AD/models/utils/focus_eqtext_adapter.py`
- Create: `M_AD/models/utils/focus_dual_support_fusion.py`
- Create: `M_AD/models/losses/focus_eqtext_losses.py`
- Create: `M_AD/models/utils/focus_negative_text_bank.py`

- [ ] Implement `FourierEquivariantTextAdapter` with `alpha_t_init=0`, `alpha_t_max<=0.05`, additive residual, normalization, and debug metrics.
- [ ] Implement `FocusDualSupportFusion` with init weights 0.9/0.1, max text weight 0.2, disabled fallback, and zero-alpha fallback.
- [ ] Implement the four requested auxiliary losses.
- [ ] Implement positive/negative prompt bank as auxiliary-only data, not support replacement.
- [ ] Run focused tests until green.

### Task 3: Detector Integration

**Files:**
- Modify: `M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py`
- Modify: `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`
- Modify: `M_AD/models/utils/focus_contrastive_embed.py`

- [ ] Pass native visual and mapped text support tensors into the dense head without changing support bank selection.
- [ ] Initialize optional EQText adapter and dual fusion from `bbox_head.focus_ovd.eqtext` / `dual_fusion`.
- [ ] Keep all new paths default-off and preserve existing baseline behavior when disabled or alpha-zero.
- [ ] Extend debug output with `text_delta_norm`, `text_support_cos_max`, and `dual_text_weight`.

### Task 4: Scripts 93-97

**Files:**
- Create: `experiments/rotation_semantic_attractor/scripts/focus_eqtext_common.py`
- Create: `experiments/rotation_semantic_attractor/scripts/93_focus_eqtext_preflight.py`
- Create: `experiments/rotation_semantic_attractor/scripts/94_focus_eqtext_generate_configs.py`
- Create: `experiments/rotation_semantic_attractor/scripts/95_focus_eqtext_train_short.py`
- Create: `experiments/rotation_semantic_attractor/scripts/96_focus_eqtext_eval_short.py`
- Create: `experiments/rotation_semantic_attractor/scripts/97_focus_eqtext_safety_and_report.py`

- [ ] Preflight required assets, P1A readiness, DeCLIP-disabled policy, and baseline-equivalence prerequisites.
- [ ] Generate only the five core variant configs.
- [ ] Train only EQ_V10, EQ_V20, and EQ_V30 on `CUDA_VISIBLE_DEVICES=6,9`.
- [ ] Eval all five core variants and record AP as `AP_BLOCKED` if full AP is unavailable.
- [ ] Build the final safety/report artifacts under `resultmd/exp_focus_eqtext_dota_short_20260609/`.

### Task 5: Verification And Run

- [ ] Run py_compile for all new modules/scripts plus touched integration files.
- [ ] Run focused unit tests.
- [ ] Run preflight, config generation, train short, eval short, and report.
- [ ] Update `task_plan.md`, `findings.md`, `progress.md`, and `CODEX_WORKLOG.md` with results.
