# OpenRSD P0-A Geometry and Prompt Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seal one CPU-only P0-A full-vocabulary diagnostic authority from the audited inventory, text7 support bytes, and live-hook source hashes without model execution.

**Architecture:** A pure standard-library module validates receipt-bound inventory snapshots, recomputes the fixed geometry policy, seals the single text7 condition and oracle code identities, then emits deterministic authority artifacts. A thin CLI atomically publishes a no-overwrite package and produces only a minimal failure package on invalid inputs.

**Tech Stack:** Python 3.10 standard library, JSON/JSONL, SHA-256, `ctypes`; no torch/MMEngine/MMDetection/MMRotate import.

---

## File structure

- Create `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py` — snapshot validation, primary policy, authority artifacts, and no-replace publisher.
- Create `framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_authority.py` — explicit-path CLI.
- Create `framework/openrsd/tests/test_ovd_orbit_p0a_authority.py` — synthetic inventory, opaque support/code fixtures, and no-runtime tests.
- Modify `docs/research/ovd_orbit_p0/README.md` and `p0_protocol.md` — P0-A authority status and strict nonclaim.
- Modify `docs/research/ovd_orbit_p0/progress.md` only after the real authority receipt is verified.

## Fixed policy and interface

The builder seals this exact policy, not caller-provided thresholds:

```json
{
  "area_min": 43.5,
  "area_max": 3265.0,
  "max_overlap_iou": 0.05,
  "min_objects_per_class": 5,
  "selection_basis": "p0a_inventory_gt_quantiles_v1"
}
```

Primary eligibility is inclusive on area and overlap. Supported classes have at
least five eligible objects. Expected real counts are 2094 primary objects and
10 supported classes:

```text
bridge, harbor, large-vehicle, plane, roundabout,
ship, small-vehicle, storage-tank, swimming-pool, tennis-court
```

The three zero-observed vocabulary classes are exactly `container-crane`,
`helicopter`, and `helipad`. The authority contains all 18 diagnostic vocabulary
classes, no base/novel fields, and
`prompt_stability_status=NOT_TESTED_SINGLE_CONDITION`.

CLI arguments are exactly:

```text
--inventory-receipt PATH
--inventory-diagnostics PATH
--object-inventory PATH
--source-input-manifest PATH
--support-asset PATH
--oracle-code-file PATH          # repeat exactly 3 times
--output-dir NEW_DIRECTORY
```

The three code paths must be supplied in sorted path order and correspond to
`M_Tools/analysis/ovd_orbit_p0.py`, `M_Tools/analysis/ovd_orbit_p0_export.py`,
and `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`. The source input
manifest binds the support asset path, expected SHA-256, 18-class order,
`support_type=text`, and `shot=7`.

## Task 1: Implement primary-policy validation with TDD

**Files:**
- Create: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py`
- Create: `framework/openrsd/tests/test_ovd_orbit_p0a_authority.py`

- [ ] **Step 1: Write failing policy tests**

Create synthetic canonical inventory rows that test inclusive area/overlap
boundaries, two supported classes, and one class with four objects. Require:

```python
selection = authority.select_primary_objects(rows)

assert selection.policy["area_min"] == 43.5
assert selection.primary_object_count == 10
assert selection.supported_classes == ("bridge", "ship")
assert all(row["class_name"] != "rare-class" for row in selection.primary_rows)
```

Add tests rejecting duplicate object IDs, nonfinite size/overlap, unrecognized
row keys, missing zero-observed class declaration, and any caller attempt to
override one fixed threshold.

- [ ] **Step 2: Run RED**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_authority_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0a_authority.py -q
```

Expected: missing authority-module import.

- [ ] **Step 3: Implement pure primary selection**

Implement `P0AAuthorityError(ValueError)`, `canonical_json_bytes`,
`sha256_bytes`, `sha256_file`, a frozen `PrimarySelection` dataclass, and
`select_primary_objects(rows)`. Validate exact seven-field inventory rows,
freeze sorted rows by `(scene_id,object_id)`, compute eligible rows with the
fixed policy, count classes, remove classes below five objects, and require
the authority zero-observed class declaration to match exactly. Do not import
NumPy or a model package.

- [ ] **Step 4: Run GREEN**

Run the Task 1 command. Expected: all selection tests pass.

- [ ] **Step 5: Commit the policy slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py framework/openrsd/tests/test_ovd_orbit_p0a_authority.py
rtk git commit -m "feat: select P0-A diagnostic primary objects"
```

## Task 2: Seal inventory, support, and code identities with TDD

**Files:**
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0a_authority.py`

- [ ] **Step 1: Write failing snapshot and artifact tests**

Use temporary canonical inventory receipt/diagnostics/object JSONL, a canonical
source input manifest, one opaque 18-class text7 support file, and three opaque
code files. Require `build_authority_artifacts` to emit exactly:

```text
p0a_diagnostic_authority.json
primary_object_ids.jsonl
authority_diagnostics.json
receipt.json
result.md
```

Assert `prompt_stability_status` is `NOT_TESTED_SINGLE_CONDITION`, support type
is `text`, shot is `7`, all three code SHA-256s occur in the authority, no
`base`/`novel` key exists, and receipt hashes every non-receipt artifact. Add
negative tests for receipt hash mismatch, diagnostics hash mismatch, support
byte mismatch, source manifest class order mismatch, code hash mismatch, and
wrong real-count assertion.

- [ ] **Step 2: Run RED**

Run the Task 1 command. Expected: missing snapshot/artifact APIs.

- [ ] **Step 3: Implement snapshot-bound artifacts**

Implement `load_canonical_json`, `load_inventory_rows`,
`validate_inventory_receipt`, `validate_source_support_manifest`, and
`build_authority_artifacts`. Read/hash each input from one byte snapshot before
decoding. Require inventory receipt status
`P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD` and all four receipt artifact
digests to equal snapshot bytes. Require the source manifest support metadata
and the supplied support path/hash to agree. Require exactly three sorted code
paths. The authority JSON includes the fixed policy, inventory hashes,
full/observed/supported/zero-observed class lists, single text7 condition,
oracle adapter type, source schema, score field, and code identity hashes.

- [ ] **Step 4: Run GREEN**

Run the Task 1 command. Expected: all snapshot and artifact tests pass.

- [ ] **Step 5: Commit the authority artifact slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py framework/openrsd/tests/test_ovd_orbit_p0a_authority.py
rtk git commit -m "feat: seal P0-A diagnostic authority artifacts"
```

## Task 3: Add no-overwrite CLI and documentation with TDD

**Files:**
- Create: `framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_authority.py`
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0a_authority.py`
- Modify: `docs/research/ovd_orbit_p0/README.md`
- Modify: `docs/research/ovd_orbit_p0/p0_protocol.md`

- [ ] **Step 1: Write failing CLI and AST tests**

Call `cli.main(argv)` with valid temporary snapshots. Require return `0`, exactly
five success artifacts, and `P0A_DIAGNOSTIC_AUTHORITY_READY_NO_FORWARD`.
Rerun at the same output path and require `FileExistsError` without overwrite.
Mutate the support file after manifest construction and require return `2` plus
only `receipt.json`, `authority_diagnostics.json`, and `result.md`.

Use `ast.parse` on both new modules to reject imports rooted at `torch`,
`mmengine`, `mmdet`, `mmrotate`, and `M_AD`, plus `torch.load`. Add document
anchors for the authority readiness state, `NOT_TESTED_SINGLE_CONDITION`, and
the strict-OVD nonclaim.

- [ ] **Step 2: Run RED**

Run the Task 1 command. Expected: CLI import and documentation anchor failures.

- [ ] **Step 3: Implement atomic CLI and docs**

Implement the exact arguments above, source snapshot rehash checks, and a
same-parent temporary publisher with flat safe names, fsync, and
`renameat2(RENAME_NOREPLACE)` via `ctypes`. If no-replace publication is
unavailable, fail closed. On `P0AAuthorityError`, publish exactly the minimal
failure package and return `2`; preserve `FileExistsError`. Update README and
protocol: diagnostic authority success is not model inference, strict G0,
prompt stability, or a paper result.

- [ ] **Step 4: Run GREEN and compatibility verification**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_authority_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0a_authority.py tests/test_ovd_orbit_p0a_inventory.py tests/test_ovd_orbit_p0_g0.py tests/test_ovd_orbit_p0.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_authority_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_Tools/analysis/ovd_orbit_p0a_authority.py M_Tools/analysis/prepare_ovd_orbit_p0a_authority.py
rtk git diff --check
```

Expected: all authority, inventory, G0, and live-hook tests pass; authority
modules compile; diff check is silent.

- [ ] **Step 5: Commit the CLI/documentation slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_authority.py framework/openrsd/tests/test_ovd_orbit_p0a_authority.py docs/research/ovd_orbit_p0/README.md docs/research/ovd_orbit_p0/p0_protocol.md
rtk git commit -m "feat: add P0-A diagnostic authority CLI"
```

## Task 4: Run the authorized real read-only authority builder

**Files:**
- Create through CLI: `docs/provenance/ovd_orbit_p0a_authority_20260831/`
- Modify: `docs/research/ovd_orbit_p0/progress.md`

- [ ] **Step 1: Verify all real evidence hashes**

```bash
rtk sha256sum docs/provenance/ovd_orbit_p0a_inventory_20260831/inventory_diagnostics.json docs/provenance/ovd_orbit_p0a_inventory_20260831/object_inventory.jsonl
rtk sha256sum /data1/zcy/OpenRSD/work_dirs/dotav2_p4_lowtext_lser_sise_gpu67_20260617_153546/eval_bundle/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl
```

Expected hashes are `89e96a1b9f471c719c6a94cd26e46f43e8375d40468d0ea6e54db62adcabc03a`, `a618010efbd8e4c413f2eea9fcc9018309d610ff776ef8178b22543502b46d44`, and `4ea3572d8184bfbfa556d051efaaea06d575bcb9a4e9f05efe3cc3564e5d737c`.

- [ ] **Step 2: Run the authority builder once**

From `framework/openrsd` run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_authority_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m M_Tools.analysis.prepare_ovd_orbit_p0a_authority --inventory-receipt ../../docs/provenance/ovd_orbit_p0a_inventory_20260831/receipt.json --inventory-diagnostics ../../docs/provenance/ovd_orbit_p0a_inventory_20260831/inventory_diagnostics.json --object-inventory ../../docs/provenance/ovd_orbit_p0a_inventory_20260831/object_inventory.jsonl --source-input-manifest ../../docs/provenance/risc_openrsd_n0o_v3/input_manifest.json --support-asset /data1/zcy/OpenRSD/work_dirs/dotav2_p4_lowtext_lser_sise_gpu67_20260617_153546/eval_bundle/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl --oracle-code-file M_Tools/analysis/ovd_orbit_p0.py --oracle-code-file M_Tools/analysis/ovd_orbit_p0_export.py --oracle-code-file M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py --output-dir ../../docs/provenance/ovd_orbit_p0a_authority_20260831
```

Expected return is `0` only for
`P0A_DIAGNOSTIC_AUTHORITY_READY_NO_FORWARD`; it is `2` for
`P0_INPUT_FAIL_STOP`. Never retry at the same output path.

- [ ] **Step 3: Audit receipt and update preparation record**

```bash
rtk jq . ../../docs/provenance/ovd_orbit_p0a_authority_20260831/receipt.json
rtk jq . ../../docs/provenance/ovd_orbit_p0a_authority_20260831/authority_diagnostics.json
rtk wc -l ../../docs/provenance/ovd_orbit_p0a_authority_20260831/primary_object_ids.jsonl
```

If failed, record exact error and stop. If ready, record the receipt status,
2094 IDs, ten supported classes, single text7 condition, and the exact Chinese
sentence “未执行模型前向、checkpoint 加载、GPU、AP 或 P0 指标计算。” Do not record a
semantic result or strict OVD claim.

- [ ] **Step 4: Commit verified code and authority evidence**

```bash
rtk git add docs/provenance/ovd_orbit_p0a_authority_20260831 docs/research/ovd_orbit_p0/progress.md framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_authority.py framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_authority.py framework/openrsd/tests/test_ovd_orbit_p0a_authority.py
rtk git commit -m "docs: record P0-A diagnostic authority"
```

## Plan self-review

- Task 1 seals the fixed geometry policy and supported-class rule; Task 2 binds inventory/support/code snapshots; Task 3 adds CPU-only publication and documents its nonclaim; Task 4 is the separately authorized real read-only authority build.
- No task creates a base/novel split, unpickles support data, loads a checkpoint, opens an image, imports a model framework, uses a GPU, or computes a semantic metric.
- The only real operation hashes already-authorized input bytes and publishes one no-overwrite authority package.
