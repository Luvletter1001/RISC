# RISC-on-OpenRSD Final-Readout S0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a zero-training, default-off RISC final-readout adapter and reusable full-tensor capture boundary to the frozen OpenRSD A10 head.

**Architecture:** A focused utility module owns the bounded low-rank residual. The existing OpenRSD head calls it only between `pred_embed` and the existing semantic Align Head, while the existing experiment-side hook registry captures semantic and geometry tensors without changing the head output contract.

**Tech Stack:** Python 3.10, PyTorch 1.12, MMEngine/MMRotate, pytest; `/data/zcy/anaconda3/envs/openrsd/bin/python`.

---

### Task 1: Align the RISC authority

**Files:**
- Create: `docs/superpowers/specs/2026-08-22-risc-openrsd-final-readout-s0-design.md`
- Modify: `README.md`
- Modify: `RISC_GOAL.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [x] **Step 1: Record the strong-parent mainline**

Make OpenRSD A10 E24 the primary parent, `filtered6605/scale1024/text7` the
main mouth, and `S0 -> N0-O -> M1-P -> M1-C` the active route. Preserve the
former E12/Q600 route as historical weak-substrate evidence rather than the
next gate.

- [x] **Step 2: Verify authority consistency**

Run:

```bash
rtk rg -n "S0|N0-O|OpenRSD A10|0.7049593925476074" README.md RISC_GOAL.md task_plan.md docs/superpowers/specs/2026-08-22-risc-openrsd-final-readout-s0-design.md
rtk run 'git diff --check -- README.md RISC_GOAL.md task_plan.md findings.md progress.md docs/superpowers'
```

Expected: the active route and parent appear in all authority files; the
scoped whitespace check exits zero.

- [x] **Step 3: Commit the authority update**

```bash
rtk git add README.md RISC_GOAL.md task_plan.md findings.md progress.md docs/superpowers
rtk git commit -m "docs: align RISC with OpenRSD final readout"
```

### Task 2: Implement the bounded final-readout adapter with TDD

**Files:**
- Create: `framework/openrsd/tests/test_risc_final_readout.py`
- Create: `framework/openrsd/M_AD/models/utils/risc_final_readout.py`

- [x] **Step 1: Write failing adapter tests**

Add tests that require this public API:

```python
from M_AD.models.utils.risc_final_readout import RISCFinalReadoutAdapter


def build_adapter(**overrides):
    config = dict(
        embed_dims=4,
        rank=2,
        enabled=True,
        init_alpha=0.0,
        max_alpha=0.1,
        max_delta_norm_ratio=0.05,
        init_seed=20260822,
    )
    config.update(overrides)
    return RISCFinalReadoutAdapter(**config)


def test_zero_alpha_is_bitwise_identity():
    module = build_adapter()
    value = torch.randn(2, 4, 3, 5)
    output = module(value)
    assert torch.equal(output, value)


def test_disabled_mode_returns_original_tensor():
    module = build_adapter(enabled=False)
    value = torch.randn(1, 4, 2, 2)
    assert module(value) is value


def test_construction_preserves_global_cpu_rng():
    torch.manual_seed(9)
    before = torch.random.get_rng_state().clone()
    build_adapter()
    assert torch.equal(torch.random.get_rng_state(), before)
```

Also cover a nonzero finite gradient on `raw_alpha`, per-location residual
ratio at or below `max_delta_norm_ratio`, and explicit invalid-input errors.

- [x] **Step 2: Run the test and verify RED**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_risc_final_readout.py -q
```

Expected: collection fails because `risc_final_readout` does not exist.

- [x] **Step 3: Add the minimal adapter**

Implement `RISCFinalReadoutAdapter(nn.Module)` with constructor arguments
`embed_dims`, `rank`, `enabled`, `init_alpha`, `max_alpha`,
`max_delta_norm_ratio`, and `init_seed`. Use a non-affine `LayerNorm`, two
bias-free linear layers, GELU, `max_alpha * tanh(raw_alpha)`, and per-location
norm clipping. Initialize the two projections inside
`torch.random.fork_rng(devices=[])` with `torch.manual_seed(init_seed)`.

- [x] **Step 4: Run the adapter test and verify GREEN**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_risc_final_readout.py -q
```

Expected: all adapter tests pass.

- [x] **Step 5: Commit the adapter**

```bash
rtk git add framework/openrsd/M_AD/models/utils/risc_final_readout.py framework/openrsd/tests/test_risc_final_readout.py
rtk git commit -m "feat: add bounded RISC final readout"
```

### Task 3: Integrate the adapter into the A10 head with TDD

**Files:**
- Modify: `framework/openrsd/tests/test_risc_final_readout.py`
- Modify: `framework/openrsd/M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`
- Create: `framework/openrsd/M_configs/Diagnostics/risc_openrsd_a10_final_readout_s0.py`

- [x] **Step 1: Write failing head/config tests**

Require the legacy A10 resolved config to omit `risc_final_readout`, the S0
config to declare rank 8 and zero alpha, and the head helper to return its
input unchanged when no adapter exists. Require an instantiated adapter to
operate on `pred_embed` without accepting regression tensors.

- [x] **Step 2: Run the focused test and verify RED**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_risc_final_readout.py -q
```

Expected: failure because the head option, helper and S0 config do not exist.

- [x] **Step 3: Wire the adapter into the semantic boundary**

Add optional `risc_final_readout: Optional[dict] = None` to the head. When it
is absent, set `self.risc_final_readout = None`; otherwise instantiate
`RISCFinalReadoutAdapter(embed_dims=self.embed_dims, **config)`. Add
`_apply_risc_final_readout(pred_embed)` and call it immediately after
`self.rtm_cls[idx](cls_feat)` and before either semantic classification path.
Do not change `reg_feat`, return tuple shapes, or prediction post-processing.

- [x] **Step 4: Add the interface-only S0 config**

Create a config inheriting the existing A10 formal config and overriding only:

```python
_base_ = [
    '../Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
]

model = dict(
    bbox_head=dict(
        risc_final_readout=dict(
            enabled=True,
            rank=8,
            init_alpha=0.0,
            max_alpha=0.1,
            max_delta_norm_ratio=0.05,
            init_seed=20260822,
        )))
```

Document in the config that it is an interface build target, not an authorized
N0-O evaluation mouth.

- [x] **Step 5: Run the focused test and verify GREEN**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_risc_final_readout.py -q
```

Expected: all adapter/head/config tests pass.

- [x] **Step 6: Commit the head integration**

```bash
rtk git add framework/openrsd/M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py framework/openrsd/M_configs/Diagnostics/risc_openrsd_a10_final_readout_s0.py framework/openrsd/tests/test_risc_final_readout.py
rtk git commit -m "feat: wire RISC into OpenRSD semantic readout"
```

### Task 4: Extend the existing OpenRSD hook recorder with TDD

**Files:**
- Modify: `framework/openrsd/experiments/rotation_semantic_attractor/tests/test_openrsd_hooks.py`
- Modify: `framework/openrsd/experiments/rotation_semantic_attractor/src/model_adapters/openrsd_hook_registry.py`

- [ ] **Step 1: Write failing full-tensor recorder tests**

Build a tiny model whose named modules match `bbox_head.rtm_cls.0`,
`bbox_head.risc_final_readout`, `bbox_head.rtm_cls_heads.0`,
`bbox_head.rtm_reg.0`, `bbox_head.rtm_ang.0`, and `bbox_head.rtm_obj.0`.
Require multiple calls to be stored in order, every tensor to be detached CPU
clone, classifier inputs to retain adapted embedding/support/labels, and the
original module outputs to remain bitwise unchanged.

- [ ] **Step 2: Run the recorder test and verify RED**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest experiments/rotation_semantic_attractor/tests/test_openrsd_hooks.py -q
```

Expected: failure because the RISC full-tensor recorder does not exist.

- [ ] **Step 3: Add `RISCReadoutHookRecorder`**

Reuse `classify_openrsd_module` and the existing hook selection conventions.
Register forward hooks on the six precise module families, record both inputs
and outputs where required, append rather than overwrite repeated calls, and
provide `snapshot()` plus `validate_complete()` methods. Do not write files or
filter tensors in this class.

- [ ] **Step 4: Run the recorder test and verify GREEN**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest experiments/rotation_semantic_attractor/tests/test_openrsd_hooks.py -q
```

Expected: all hook tests pass.

- [ ] **Step 5: Commit the capture extension**

```bash
rtk git add framework/openrsd/experiments/rotation_semantic_attractor/src/model_adapters/openrsd_hook_registry.py framework/openrsd/experiments/rotation_semantic_attractor/tests/test_openrsd_hooks.py
rtk git commit -m "feat: capture RISC readout evidence"
```

### Task 5: Verify S0 and close the records

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`
- External append-only log: `/data1/zcy/OpenRSD/CODEX_WORKLOG.md`

- [ ] **Step 1: Run focused and adjacent tests**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_risc_final_readout.py tests/test_risc_orbit_projection.py experiments/rotation_semantic_attractor/tests/test_openrsd_hooks.py -q
```

Expected: zero failures.

- [ ] **Step 2: Compile changed Python files**

```bash
rtk /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_AD/models/utils/risc_final_readout.py M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py experiments/rotation_semantic_attractor/src/model_adapters/openrsd_hook_registry.py
```

Expected: exit zero with no output.

- [ ] **Step 3: Run repository checks**

```bash
rtk run 'git diff --check'
rtk git status --short --branch
```

Expected: no whitespace errors; only planned files are modified before the
final records commit.

- [ ] **Step 4: Update records and append the OpenRSD finish log**

Record exact test counts and commands in `progress.md`, technical findings in
`findings.md`, and mark S0 complete in `task_plan.md`. Append a concise finish
entry through the OpenRSD worklog helper.

- [ ] **Step 5: Commit the verified records**

```bash
rtk git add task_plan.md findings.md progress.md docs/superpowers/plans/2026-08-22-risc-openrsd-final-readout-s0.md
rtk git commit -m "docs: record verified OpenRSD S0"
```
