# RISC OpenRSD N0-O Input Seal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a CPU-only, fail-closed input manifest that seals the A10 parent, paper mouth, 160-scene C4/C8 plan and exact per-scene mapped text7 support tensors before N0-O inference.

**Architecture:** A standalone deterministic builder validates immutable external authorities, copies the sealed scene plan, derives prompt indices by SHA-256 rather than runtime RNG, applies only the checkpoint's text mapping on CPU, and emits canonical JSON/JSONL provenance. No model forward or GPU API is used.

**Tech Stack:** Python 3.10, PyTorch 1.12 CPU, NumPy, pickle, SHA-256, pytest, canonical JSON/JSONL.

---

### Task 1: Record the approved input-seal authority

**Files:**
- Create: `docs/superpowers/specs/2026-08-22-risc-openrsd-n0o-input-seal-design.md`
- Create: `docs/superpowers/plans/2026-08-22-risc-openrsd-n0o-input-seal.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [x] **Step 1: Record the frozen assets and support rule**

Write the exact paths, sizes, hashes, dataset counts, P77E mouth, S0 code
identity, scene-plan identity and SHA-ranked per-scene text7 rule from the
approved design. State that GPU work remains unauthorized.

- [x] **Step 2: Self-review the authority**

Run:

```bash
rtk rg -n "09758508|0b2c190b|6605|13833|text7|GPU_NOT_AUTHORIZED|SHA256" docs/superpowers/specs/2026-08-22-risc-openrsd-n0o-input-seal-design.md
rtk run 'p="T""BD|T""ODO|choose ""later|best ""checkpoint|fill ""in|implement ""later"; if rg -n "$p" docs/superpowers/specs/2026-08-22-risc-openrsd-n0o-input-seal-design.md docs/superpowers/plans/2026-08-22-risc-openrsd-n0o-input-seal.md; then exit 1; else test "$?" -eq 1; fi'
rtk run 'git diff --check -- docs/superpowers task_plan.md findings.md progress.md'
```

Expected: all authority markers are present, placeholder search returns no
matches, and the scoped whitespace check exits zero.

- [x] **Step 3: Commit the authority documents**

```bash
rtk git add docs/superpowers task_plan.md findings.md progress.md
rtk git commit -m "docs: freeze OpenRSD N0-O input contract"
```

### Task 2: Implement the deterministic builder with TDD

**Files:**
- Create: `framework/openrsd/tools/risc_n0o/prepare_openrsd_n0o_input_seal.py`
- Create: `framework/openrsd/tests/test_prepare_openrsd_n0o_input_seal.py`

- [x] **Step 1: Write failing canonical and scene-plan tests**

Tests must import the builder and require:

```python
def test_canonical_json_is_sorted_compact_utf8_with_one_lf():
    value = {'z': 1, 'a': '场景'}
    assert canonical_json_bytes(value) == (
        b'{"a":"\xe5\x9c\xba\xe6\x99\xaf","z":1}\n')


def test_scene_plan_requires_four_disjoint_folds_and_excludes_p0148():
    plan = synthetic_scene_plan(folds=4, scenes_per_fold=2)
    summary = validate_scene_plan(plan, scenes_per_fold=2)
    assert summary['scene_count'] == 8
    plan['records'][1]['scene_id'] = plan['records'][0]['scene_id']
    with pytest.raises(SealError, match='unique scene'):
        validate_scene_plan(plan, scenes_per_fold=2)
```

Also reject wrong angles, wrong fold sizes, P0148, duplicate paths, missing
hashes, non-lowercase SHA-256 and a source scene-plan hash mismatch.

- [x] **Step 2: Run the tests and verify RED**

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_prepare_openrsd_n0o_input_seal.py -q
```

Expected: collection fails because the builder module does not exist.

- [x] **Step 3: Implement canonical serialization and scene validation**

Add `SealError`, `canonical_json_bytes`, `canonical_jsonl_bytes`,
`sha256_file`, `require_file_hash`, and `validate_scene_plan`. The validator
must return counts only after checking every contract in the design.

- [x] **Step 4: Write failing support-selection and mapping tests**

Use a synthetic two-class pickle and a tiny checkpoint whose text mapping is
known exactly. Require:

```python
def test_support_indices_are_scene_specific_and_repeatable():
    first = select_support_indices('P0001', source, shot=2)
    second = select_support_indices('P0001', source, shot=2)
    other = select_support_indices('P0002', source, shot=2)
    assert first == second
    assert first != other


def test_mapping_uses_raw_state_dict_and_emits_bounded_schema():
    row, mapped = build_support_row(scene_record, source, checkpoint, shot=2)
    assert mapped.shape == (2, 2, 3)
    assert mapped.dtype == np.float32
    assert np.isfinite(mapped).all()
    assert row['mapped_tensor_sha256'] == sha256(mapped.tobytes(order='C'))
```

Also require seven unique indices per class, canonical class order, little-
endian float32 bytes, exact checkpoint key allowlist, rejection of EMA-only or
non-finite tensors, and no CUDA calls.

- [x] **Step 5: Run the focused tests and verify RED**

Run the same pytest command. Expected: failures identify missing selection,
mapping and ledger functions.

- [x] **Step 6: Implement support derivation and artifact publication**

Implement:

```python
def select_support_indices(scene_id, support_data, *, classes, shot=7):
    """Return SHA-ranked prompt indices in canonical class order."""


def load_text_mapping(checkpoint_path):
    """Load exactly four raw state_dict tensors on CPU."""


def build_support_row(scene_record, support_data, mapping, *, classes, shot=7):
    """Return one canonical ledger row plus mapped float32 tensor."""


def build_input_seal(output_dir):
    """Validate all frozen assets and publish three files without overwrite."""
```

The CLI accepts only `--output-dir`; all authorities are constants. It writes
to a temporary sibling directory, fsyncs files, and publishes with no replace.

- [x] **Step 7: Run tests and verify GREEN**

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_prepare_openrsd_n0o_input_seal.py -q
```

Expected: all builder tests pass without GPU initialization.

- [x] **Step 8: Commit the builder**

```bash
rtk git add framework/openrsd/tools/risc_n0o/prepare_openrsd_n0o_input_seal.py framework/openrsd/tests/test_prepare_openrsd_n0o_input_seal.py
rtk git commit -m "feat: build OpenRSD N0-O input seal"
```

### Task 3: Generate and verify the real input seal

**Files:**
- Create: `docs/provenance/risc_openrsd_n0o/scene_plan_40.json`
- Create: `docs/provenance/risc_openrsd_n0o/support_ledger.jsonl`
- Create: `docs/provenance/risc_openrsd_n0o/input_manifest.json`

- [x] **Step 1: Run two independent temporary builds**

```bash
rtk run 'd1=$(mktemp -d /tmp/risc-n0o-seal-a.XXXXXX); d2=$(mktemp -d /tmp/risc-n0o-seal-b.XXXXXX); printf "%s\n%s\n" "$d1" "$d2"'
```

Run the builder once for each path with the OpenRSD interpreter. Expected:
both exit zero and report 160 scenes, 2,880 class selections and finite
`[18,7,256]` mapped tensors.

- [x] **Step 2: Prove deterministic output**

For each of the three relative filenames, run `cmp -s` between the two builds
and compare SHA-256. Expected: byte-identical pairs.

- [x] **Step 3: Generate the tracked seal**

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python tools/risc_n0o/prepare_openrsd_n0o_input_seal.py --output-dir /data1/zcy/RISC/docs/provenance/risc_openrsd_n0o_v3
```

Expected: atomic publication succeeds only because the target does not yet
exist. The builder must refuse a second invocation at the same path.

- [x] **Step 4: Validate the real outputs independently**

Run a separate Python read-only audit that checks canonical LF bytes, artifact
hash chaining, source hashes, 160 ledger rows, 18 classes per row, seven unique
indices per class, all selected source indices in range, and scene-plan byte
identity. Do not call the builder's validation functions for this audit.

- [x] **Step 5: Commit the generated provenance**

```bash
rtk git add docs/provenance/risc_openrsd_n0o_invalid_v1 docs/provenance/risc_openrsd_n0o_invalid_v2 docs/provenance/risc_openrsd_n0o_v3
rtk git commit -m "chore: seal OpenRSD N0-O inputs"
```

### Task 4: Final review and stop-boundary handoff

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`
- External append-only log: `/data1/zcy/OpenRSD/CODEX_WORKLOG.md`

- [x] **Step 1: Run focused and adjacent tests**

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_prepare_openrsd_n0o_input_seal.py tests/test_risc_final_readout.py experiments/rotation_semantic_attractor/tests/test_openrsd_hooks.py -q
```

Expected: zero failures.

- [x] **Step 2: Compile and inspect**

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile tools/risc_n0o/prepare_openrsd_n0o_input_seal.py
rtk run 'git diff main...HEAD --check'
rtk run 'git diff --check'
rtk git status --short --branch
```

Expected: compilation and both whitespace checks pass; only planned record
files remain before the final commit.

- [x] **Step 3: Obtain independent read-only review**

Review must check scientific mouth separation, source/hash correctness,
support determinism, raw-versus-EMA mapping, canonical serialization,
no-overwrite behavior and absence of GPU/model-forward paths. Fix every
Critical/Important finding before completion.

- [x] **Step 4: Update records and worklog**

Record exact commands, hashes, test counts, review verdict and the status
`SEALED_INPUTS_GPU_NOT_AUTHORIZED`. Append an OpenRSD finish entry explicitly
stating that no GPU inference or training ran.

- [x] **Step 5: Commit verified records**

```bash
rtk git add task_plan.md findings.md progress.md docs/superpowers/plans/2026-08-22-risc-openrsd-n0o-input-seal.md
rtk git commit -m "docs: record verified OpenRSD N0-O seal"
```

Stop after this commit. Do not implement or launch the N0-O GPU runner without
new explicit user authorization.
