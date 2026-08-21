# HRSC Parent-Preserving Seed-13 Repeat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repeat the promoted parent-preserving HRSC C1 arm with seed `20260713` while changing no other effective model, data, optimizer, or evaluation setting.

**Architecture:** A new config inherits the verified seed-12 parent-preserving config and overrides only `randomness.seed` and `work_dir`. A config-differential test proves the effective configs are otherwise identical. Exact parent equivalence, strict all-query inference, a full zero-update replay, five-epoch training, mediator measurement, and an independent final evaluation gate the result.

**Tech Stack:** Python 3.8, MMEngine Config, PyTorch 1.12, MMDetection/MMRotate, pytest, NVIDIA A40, untracked `.lab` researcher ledger.

---

## File Map

- Create `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_seed13_5e.py`: seed-13 repeat config.
- Modify `tests/test_projects/ov_capflow/test_hrsc_config.py`: assert seed-13 differs from seed-12 only in seed and work directory.
- Modify `docs/project_history/exp_20260714_two_week_review/fplan_ov_capflow_next_stage_zh.md`: replace the disproven native-reset fusion formula with the verified transported-parent residual formula; preserve this user-owned untracked document.
- Modify `.lab/log.md`, `.lab/results.tsv`, `.lab/branches.md`, `.lab/summary.md`: experiment genealogy, raw results, and decision.
- Modify `docs/project_history/exp_20260714_hrsc_parent_preserving_fusion/fres_hrsc_parent_preserving_fusion_zh.md`: add the seed-13 result and reproducibility decision.
- Write ignored artifacts under `work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_seed13_5e/`.

### Task 1: Create and commit the execution plan

- [ ] **Step 1: Save this plan and self-review it**

Run:

```bash
rtk rg -n "[T]BD|[T]ODO|implement [l]ater|fill [i]n" docs/superpowers/plans/2026-07-14-ov-capflow-hrsc-parent-preserving-seed13.md
rtk git diff --check
```

Expected: the placeholder scan has no matches and the diff check exits zero.

- [ ] **Step 2: Create the isolated research branch**

Run:

```bash
rtk git checkout -b research/hrsc-parent-preserving-seed13
```

Expected: the new branch starts from merged Experiment 4 commit `4e4ecc3`.

- [ ] **Step 3: Commit the plan**

Run:

```bash
rtk git add docs/superpowers/plans/2026-07-14-ov-capflow-hrsc-parent-preserving-seed13.md
rtk git commit -m "docs: plan HRSC parent-preserving seed repeat"
```

### Task 2: Add the seed-only config using TDD

- [ ] **Step 1: Write the failing differential test**

Add to `tests/test_projects/ov_capflow/test_hrsc_config.py`:

```python
def test_parent_preserving_seed13_only_changes_seed_and_work_dir():
    seed12 = Config.fromfile(
        CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py')
    seed13 = Config.fromfile(
        CONFIG_DIR /
        'ov_capflow_swin-t_hrsc_c1_parent_preserving_seed13_5e.py')
    assert seed12.randomness.seed == 20260712
    assert seed13.randomness.seed == 20260713
    assert seed13.work_dir.endswith(
        'hrsc_c1_parent_preserving_seed13_5e')

    seed12_effective = seed12.to_dict()
    seed13_effective = seed13.to_dict()
    seed12_effective.pop('randomness')
    seed13_effective.pop('randomness')
    seed12_effective.pop('work_dir')
    seed13_effective.pop('work_dir')
    assert seed13_effective == seed12_effective
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_hrsc_config.py::test_parent_preserving_seed13_only_changes_seed_and_work_dir -q
```

Expected: FAIL because the seed-13 config file does not exist.

- [ ] **Step 3: Add the minimal seed-13 config**

Create `configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_seed13_5e.py`:

```python
_base_ = './ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py'

randomness = dict(seed=20260713)
work_dir = (
    'work_dirs/ov_capflow_hrsc/'
    'hrsc_c1_parent_preserving_seed13_5e')
```

- [ ] **Step 4: Run the focused and portable tests and verify GREEN**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_hrsc_config.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Expected: both commands exit zero.

- [ ] **Step 5: Correct the stale future-plan formula**

Replace the old native-reset formula with:

```text
native_residual = q_native - q_parent
q_fused = q_parent + tanh(g(q_native, q_parent)) * P(native_residual)
```

State that zero gate exactly recovers `q_parent` and that capacity remains a future single-variable multiplier rather than part of the verified Experiment 4/5 arm.

- [ ] **Step 6: Commit Experiment 5 before any real run**

Commit only the tracked config and test with:

```text
experiment #5: repeat parent-preserving HRSC fusion with seed 20260713

Branch: research/hrsc-parent-preserving-seed13
Parent: #4
Hypothesis: the parent-preserving C1 gain survives a seed-only five-epoch repeat
```

### Task 3: Run pre-training engineering gates

- [ ] **Step 1: Verify the selected Python environment**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -V
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -c "import torch, mmengine, mmdet, mmrotate; print(torch.__version__, mmengine.__version__, mmdet.__version__, mmrotate.__version__)"
```

Expected: Python 3.8 and the previously audited MMDetection/MMRotate stack.

- [ ] **Step 2: Run real-checkpoint parent equivalence**

Run `audit_parent_equivalence.py` with the C0 config, seed-13 config, and C0 epoch-10 checkpoint. Write `audits/parent_equivalence.json` in the seed-13 work directory.

Expected: `pass=true`; all seven tensor groups are exact with maximum absolute difference `0.0`; both sides emit 600 rows.

- [ ] **Step 3: Run the full zero-update replay**

Run `tools/test.py` with the seed-13 config and C0 epoch-10 checkpoint into `zero_update_replay/`.

Expected: AP50 `0.5870`, recall `0.941`, 453 images, 1228 GT, and 271800 detections.

- [ ] **Step 4: Run strict inference and checkpoint-load audits**

Expected: exactly 600 rows per image, no forbidden operation, 18 expected semantic-fusion missing keys, zero invalid missing keys, and zero unexpected keys.

- [ ] **Step 5: Apply the engineering gate**

If any exact-equivalence, zero-update replay, strict inference, or checkpoint mapping condition fails, log Experiment 5 as `invalid-engineering` and stop before training. Otherwise record the raw gate values in `.lab/log.md` and continue.

### Task 4: Train and measure the seed-13 repeat

- [ ] **Step 1: Launch five-epoch training**

Run on an available A40:

```bash
rtk .lab/bin/run <gpu> configs/ov_capflow/hrsc/ov_capflow_swin-t_hrsc_c1_parent_preserving_seed13_5e.py work_dirs/ov_capflow_hrsc/hrsc_c1_parent_preserving_seed13_5e
```

Expected: completion within 7200 seconds, finite loss/gradients, and epoch-5 validation.

- [ ] **Step 2: Independently re-evaluate `epoch_5.pth`**

Run `tools/test.py` into `final_revalidation/` and use this fresh result for the final primary metric.

- [ ] **Step 3: Measure mediators and artifact identity**

Run `hrsc_mediator_metrics.py`, audit the trained checkpoint load, and compute SHA256. Record recall, GT coverage, duplicate extras/GT, matched/unmatched gate means, gate gap, duration, peak memory, and final loss.

- [ ] **Step 4: Apply the pre-registered H3 decision**

- `AP50 >= 0.5890` with strict inference intact: both seeds retain the gate; keep Experiment 5.
- AP50 within `0.001` of C0 is eligible only with two material mediator improvements and recall decrease no worse than `0.005`.
- Otherwise log the result before resetting the Experiment 5 commit and classify Experiment 4 as single-seed/parked.

Also report the two-seed mean and sample standard deviation. Do not silently replace the pre-registered gate with a post-hoc stability threshold.

### Task 5: Record and verify the result

- [ ] **Step 1: Update the experiment ledger before any reset**

Append the structured Experiment 5 entry and TSV row; update branch status and summary.

- [ ] **Step 2: Update the Chinese result report**

Add seed-13 primary/mediator values, two-seed mean/std, reproducibility decision, and DOTA2 eligibility. Preserve the report's distinction between HRSC detection evidence and open-vocabulary evidence.

- [ ] **Step 3: Run final verification**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
rtk git diff --check
rtk git status --short --branch
```

Expected: the portable suite passes; tracked changes are intentional; the user-owned two-week review directory remains preserved.
