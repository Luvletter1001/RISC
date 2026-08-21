# P8A FPN-Level Gaussian Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative FPN-level Gaussian semantic routing bias for HRRSD RTMDet experiments.

**Architecture:** Use class-conditional Gaussian log-area priors to compute a fixed class-wise bias for each FPN level. The bias is non-positive and normalized so each class keeps zero penalty at its best-supported level; unsupported classes receive zero bias.

**Tech Stack:** PyTorch, MMEngine/MMRotate configs, existing `GSRRotatedRTMDetSepBNHead`, pytest.

---

### Task 1: Head Utility And Tests

**Files:**
- Modify: `tests/test_gs3c_closed_set_rtmdet_head.py`
- Modify: `M_AD/models/dense_heads/gs3c_rtmdet_head.py`

- [ ] **Step 1: Write failing tests**

Add tests for `_apply_gaussian_semantic_level_routing`:

```python
def test_closed_set_level_routing_bias_penalizes_wrong_fpn_level():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_level_routing_enable = True
    head.gaussian_semantic_level_routing_weight = 0.5
    head.gaussian_semantic_level_routing_max_bias_abs = 2.0
    head.gaussian_semantic_level_routing_bias = torch.tensor([
        [0.0, -2.0, 0.0],
        [-2.0, 0.0, 0.0],
    ])

    logits = torch.zeros((1, 3))

    routed_small = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_level_routing(
        head, logits, level_idx=0)
    routed_large = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_level_routing(
        head, logits, level_idx=1)

    assert routed_small[0, 0].item() == pytest.approx(0.0)
    assert routed_small[0, 1].item() == pytest.approx(-2.0)
    assert routed_small[0, 2].item() == pytest.approx(0.0)
    assert routed_large[0, 0].item() == pytest.approx(-2.0)
    assert routed_large[0, 1].item() == pytest.approx(0.0)
    assert routed_large[0, 2].item() == pytest.approx(0.0)
```

- [ ] **Step 2: Run red test**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_gs3c_closed_set_rtmdet_head.py::test_closed_set_level_routing_bias_penalizes_wrong_fpn_level -q
```

Expected: fail because `_apply_gaussian_semantic_level_routing` is not defined.

- [ ] **Step 3: Implement minimal utility**

Add config parsing for `gaussian_semantic_scale.level_routing`, build a non-positive `(num_levels, num_classes)` bias tensor from class log-area priors and level reference sizes, and apply it to 2D or 4D class logits.

- [ ] **Step 4: Run green tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_gs3c_closed_set_rtmdet_head.py tests/test_gaussian_semantic_scale_density_head.py -q
```

Expected: pass.

### Task 2: P8A Config And Launcher

**Files:**
- Create: `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8a_levelroute_train_gpu67.py`
- Create: `M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8a_levelroute_eval_gpu67.py`
- Create: `M_Tools/experiments/run_hrrsd_bass_gsf_p8a_levelroute_train_20260621.sh`

- [ ] **Step 1: Add train/eval configs**

Inherit from P7 so P8A tests the additive effect of level routing over the strongest current Gaussian component.

- [ ] **Step 2: Add launcher**

Follow the P7 launcher shape, run train, eval, and risk audits against baseline, no-G3, density, and P7.

- [ ] **Step 3: Config safety checks**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8a_levelroute_train_gpu67.py M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8a_levelroute_eval_gpu67.py
```

Expected: pass.

### Task 3: Review Ledger

**Files:**
- Create: `resultmd/exp_p4_scale_semantic_validation/freview_20260621_p8a_levelroute_preflight.md`

- [ ] **Step 1: Record strict ICLR pre-review**

Score novelty, expected effectiveness, AP safety, implementation risk, and evidence gate before training.

- [ ] **Step 2: Start one GPU run**

Launch only one P8A run first. Parse mAP/RISK before deciding whether to expand.
