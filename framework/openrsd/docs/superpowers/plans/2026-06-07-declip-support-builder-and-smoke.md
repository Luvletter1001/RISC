# DeCLIP Support Builder and Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a real DeCLIP support PKL for DOTA2 and run an initial OpenRSD smoke with `use_declip=True`.

**Architecture:** Keep the current OpenRSD detector and `use_declip_support` switch unchanged. Add a DOTA2 support builder that emits the existing support schema (`texts`, `text_embeds`, `visual_embeds`, `confidence_scores`) at `data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DeCLIP_support.pkl`.

**Tech Stack:** Python, PyTorch, OpenRSD support PKL schema, DeCLIP EVA-B checkpoint, MMEngine training smoke on GPU 6/9.

---

### Task 1: Test Support Builder Utilities

**Files:**
- Create: `tests/openrsd/test_declip_support_builder.py`
- Create/modify: `M_Tools/Data1_DOTA2/Step5_3_Prepare_Visual_Text_DeCLIP_support.py`

- [ ] Write tests for crop extraction, L2 normalization, support schema validation, and per-class top-k selection.
- [ ] Run the test file and confirm it fails because the builder module does not exist.
- [ ] Implement the minimal utility functions needed by the tests.
- [ ] Re-run the test file and confirm it passes.

### Task 2: Generate DeCLIP Support PKL

**Files:**
- Modify: `M_Tools/Data1_DOTA2/Step5_3_Prepare_Visual_Text_DeCLIP_support.py`
- Generate: `data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DeCLIP_support.pkl`

- [ ] Load the local DeCLIP B checkpoint from `pretrained/declip/DeCLIP_EVA-B_DINOv2-B_csa_0.05_2.0/epoch_6.pt`.
- [ ] Encode DOTA2 class prompts and GT object crops into DeCLIP feature arrays.
- [ ] Keep output compatible with the current OpenRSD support loader.
- [ ] Validate the generated PKL shape and keys.

### Task 3: Run Initial DeCLIP Smoke

**Files:**
- Read: `M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py`
- Write: `resultmd/exp_declip_ccl_bstage_20260607/`

- [ ] Run a short training smoke with `use_declip=True` and `use_ccl=False`.
- [ ] If feasible, run `use_declip=True` and `use_ccl=True`.
- [ ] Scan logs for Traceback, ERROR, RuntimeError, OOM, and NaN.
- [ ] Append a concise result summary.
