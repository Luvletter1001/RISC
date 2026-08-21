# DeCLIP and CCL Config Switches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config-controlled DeCLIP support loading and CCL loss toggles to OpenRSD without changing baseline behavior when disabled.

**Architecture:** `OpenRTMDet` selects the active support feature dictionary during initialization. `OpenRotatedRTMDetSepBNHead` owns the additive CCL branch because it already has dense embeddings, assigned labels, support features, and support labels.

**Tech Stack:** Python, PyTorch, MMEngine/MMRotate-style configs, pytest-compatible smoke tests.

---

### Task 1: Add Tests for Switch Semantics

**Files:**
- Create: `tests/openrsd/test_declip_ccl_switches.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest
import torch

from M_AD.models.detectors.Flex_Rtmdet_v3_1_formal import select_active_support_feat_dict
from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import compute_ccl_loss


def test_select_active_support_feat_dict_uses_baseline_when_declip_disabled(tmp_path: Path):
    baseline = {'Data1_DOTA2': str(tmp_path / 'baseline.pkl')}
    declip = {'Data1_DOTA2': str(tmp_path / 'declip.pkl')}

    selected = select_active_support_feat_dict(
        support_feat_dict=baseline,
        use_declip_support=False,
        declip_support_feat_dict=declip,
    )

    assert selected is baseline


def test_select_active_support_feat_dict_requires_declip_dict_when_enabled():
    with pytest.raises(ValueError, match='declip_support_feat_dict'):
        select_active_support_feat_dict(
            support_feat_dict={'Data1_DOTA2': 'baseline.pkl'},
            use_declip_support=True,
            declip_support_feat_dict=None,
        )


def test_compute_ccl_loss_returns_finite_scalar_for_positive_assignments():
    dense_embeds = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]])
    assigned_labels = torch.tensor([[0, 0, 1]])
    support_feats = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    support_labels = torch.tensor([[0, 1]])

    loss = compute_ccl_loss(
        dense_embeds=dense_embeds,
        assigned_labels=assigned_labels,
        support_feats=support_feats,
        support_labels=support_labels,
        num_classes=2,
        temperature=0.1,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/openrsd/test_declip_ccl_switches.py -q`

Expected: import failures for `select_active_support_feat_dict` and `compute_ccl_loss`.

### Task 2: Implement DeCLIP Support Selection

**Files:**
- Modify: `M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py`

- [ ] **Step 1: Add helper and constructor arguments**

Add `select_active_support_feat_dict(...)`, `use_declip_support=False`, and `declip_support_feat_dict=None`.

- [ ] **Step 2: Use helper before support loading**

Replace direct `support_feat_dict` loading with the selected active dictionary.

- [ ] **Step 3: Run targeted tests**

Run: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/openrsd/test_declip_ccl_switches.py -q`

Expected: CCL import still fails until Task 3; DeCLIP helper tests pass.

### Task 3: Implement CCL Loss Switch

**Files:**
- Modify: `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`

- [ ] **Step 1: Add `compute_ccl_loss` helper**

The helper normalizes dense embeddings and per-class support prototypes, then computes cross-entropy over cosine logits for positive assigned locations only.

- [ ] **Step 2: Add head config knobs**

Add `use_ccl_loss=False`, `ccl_loss_weight=0.05`, and `ccl_temperature=0.1`.

- [ ] **Step 3: Add optional `loss_ccl` in `loss_by_feat`**

Use flattened `embed_preds`, `labels_list`, `support_feats`, and `support_labels`. Skip the branch when disabled.

- [ ] **Step 4: Run targeted tests**

Run: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/openrsd/test_declip_ccl_switches.py -q`

Expected: all tests pass.

### Task 4: Add Config Switches

**Files:**
- Modify: `M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py`

- [ ] **Step 1: Add top-level switches**

Add `use_declip = False` and `use_ccl = False` near the top.

- [ ] **Step 2: Wire model config**

Set `use_declip_support=use_declip`, `declip_support_feat_dict=...`, and the head CCL fields.

- [ ] **Step 3: Run config import smoke**

Run: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py`

Expected: exit code 0.

### Task 5: Verification

**Files:**
- Review changed files only.

- [ ] **Step 1: Run unit tests**

Run: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/openrsd/test_declip_ccl_switches.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run config compile smoke**

Run: `rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py tests/openrsd/test_declip_ccl_switches.py`

Expected: exit code 0.
