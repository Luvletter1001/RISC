# OpenRSD P0-A Diagnostic Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the sealed 80-scene OpenRSD C4 candidate pool and official DOTA annotation text into a deterministic, CPU-only P0-A diagnostic object inventory.

**Architecture:** A standard-library geometry module validates the historical scene plan, streams selected annotation hashes, parses DOTA quadrilaterals into canonical five-value boxes, and calculates GT-only area and convex IoU. A thin CLI atomically publishes a diagnostic inventory package. It never calls strict G0, a model, a checkpoint, or a GPU.

**Tech Stack:** Python 3.10 standard library, OpenRSD pytest environment, JSON/JSONL, SHA-256, `math`, `ctypes`; no torch/MMEngine/MMDetection/MMRotate import.

---

## File structure

- Create `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py` — pure contract, DOTA parser, geometry, artifact bytes, and no-replace publisher.
- Create `framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_inventory.py` — CLI for source-plan snapshot, pure build, and publish.
- Create `framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py` — only temporary annotations and synthetic source-plan fixtures.
- Modify `docs/research/ovd_orbit_p0/README.md` and `p0_protocol.md` — record diagnostic-only inventory status, never strict OVD.
- Modify `docs/research/ovd_orbit_p0/progress.md` only after Task 4 has a verified real receipt.

## Fixed interfaces

The CLI has exactly four required arguments:

```text
--source-scene-plan PATH
--source-plan-sha256 LOWERCASE_64_HEX
--source-input-manifest PATH
--output-dir NEW_DIRECTORY
```

The source plan must be canonical `risc-n0-set-orbit-scene-plan-v1` with
`unique_scene_count == 160`. Retain only `c4_a` / `c4_b`; reject P0148,
duplicates, or any retained count other than 80. Output candidates use exactly
the later-G0 keys `scene_id`, `split`, `image_path`, `image_sha256`,
`annotation_path`, and `annotation_sha256`, with `split="diagnostic"`.

The source manifest is canonical JSON and must expose 18 unique ordered names
at `support.class_order`. This is `diagnostic_full_vocabulary`, never a
base/novel split.

Every nonblank annotation line is exactly:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

All coordinates must be finite; class name must be in the 18-name vocabulary;
difficulty is exactly `0`, `1`, or `2`. Difficulty `2` is recorded then
excluded; `0` and `1` are retained. The retained output row keys are exactly
`annotation_sha256`, `box`, `class_name`, `object_id`, `overlap`, `scene_id`,
and `size`.

`object_id` is `${scene_id}:${source_line_number}`. The output box is
`[cx,cy,w,h,theta]`: center is the four-point mean; width is edge 1→2; height
is edge 2→3; swap width/height and add pi/2 if width < height; normalize theta
into `[-pi/2,pi/2)`. `size` is absolute shoelace area. `overlap` is maximum
same-scene IoU over retained convex quadrilaterals from Sutherland–Hodgman
clipping. No normal-size or isolated-object threshold is selected here.

## Task 1: Add DOTA parsing and geometry with TDD

**Files:**
- Create: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py`
- Create: `framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py`

- [x] **Step 1: Write failing geometry tests**

Add tests using these lines:

```python
first = inventory.parse_dota_line(
    "0 0 4 0 4 2 0 2 bridge 1", line_number=1,
    scene_id="P0001", annotation_sha256="a" * 64,
    vocabulary=("bridge",))
second = inventory.parse_dota_line(
    "2 0 6 0 6 2 2 2 bridge 0", line_number=2,
    scene_id="P0001", annotation_sha256="a" * 64,
    vocabulary=("bridge",))

assert first["object_id"] == "P0001:1"
assert first["box"] == [2.0, 1.0, 4.0, 2.0, 0.0]
assert first["size"] == 8.0
assert inventory.convex_iou(first["polygon"], second["polygon"]) == pytest.approx(1 / 3)
```

Also require malformed token count, `nan`, unknown class, difficulty `3`, and
zero-width quadrilateral to raise `P0AInventoryError` mentioning its line.

- [x] **Step 2: Run RED**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0a_inventory.py -q
```

Expected: module import failure.

- [x] **Step 3: Implement pure geometry**

Implement `P0AInventoryError`, `canonical_json_bytes`, `sha256_bytes`,
`sha256_file`, `parse_dota_line`, `normalize_rbox`, `polygon_area`, and
`convex_iou`. Implement Sutherland–Hodgman with a positive-area clip polygon,
and return `0.0` for disjoint polygons. Do not import NumPy or an ML package.

- [x] **Step 4: Run GREEN**

Run the Task 1 command. Expected: all geometry tests pass.

- [x] **Step 5: Commit the geometry slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py
rtk git commit -m "feat: parse P0-A DOTA diagnostic inventory"
```

## Task 2: Build 80-scene artifacts with TDD

**Files:**
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py`

- [x] **Step 1: Write failing source-plan tests**

Build a canonical temporary source plan with 40 `c4_a`, 40 `c4_b`, and one
`c8_a` record. Create every selected annotation text file and write its actual
SHA-256 into the record. Build a canonical source manifest with 18 unique class
names including `bridge`. Require:

```python
artifacts = inventory.build_inventory_artifacts(
    source_plan=source_plan,
    source_plan_sha256=inventory.sha256_bytes(
        inventory.canonical_json_bytes(source_plan)),
    source_input_manifest=source_manifest)
candidates = json.loads(artifacts["candidate_scene_plan.json"])
rows = [json.loads(line) for line in artifacts["object_inventory.jsonl"].splitlines()]
assert len(candidates["records"]) == 80
assert {row["split"] for row in candidates["records"]} == {"diagnostic"}
assert all(row["scene_id"] != "P0148" for row in candidates["records"])
assert all("difficulty" not in row and "polygon" not in row for row in rows)
assert json.loads(artifacts["receipt.json"])["status"] == "P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD"
```

Also require P0148, incorrect plan SHA, duplicate C4 scene, not-exactly-80 C4
scenes, annotation hash mismatch, unexpected class, and source manifest with
not-exactly-18 classes to raise `P0AInventoryError`. Verify difficulty-2 count
is reported but its line creates no object row, and reverse input record order
produces byte-identical artifacts.

- [x] **Step 2: Run RED**

Run the Task 1 command. Expected: missing source-plan and artifact APIs.

- [x] **Step 3: Implement source selection and artifacts**

Implement `load_canonical_json`, `validate_source_plan`,
`validate_source_manifest`, `build_inventory_artifacts`, and
`build_inventory_failure_artifacts`. Hash every selected annotation before
parsing; never open or hash an image. Sort selected records by
`(fold_id,scene_rank,scene_id)`, retained object rows by `(scene_id,object_id)`,
and all JSON mapping keys canonically.

Successful artifacts are exactly `candidate_scene_plan.json`,
`object_inventory.jsonl`, `inventory_diagnostics.json`, `receipt.json`, and
`result.md`. Diagnostics include selected/retained/difficulty-2 counts, sorted
class counts, annotation verification count, and deterministic size/overlap
quantiles at `0.0,0.25,0.5,0.75,1.0`. Receipt hashes every non-receipt artifact.
Result markdown is Chinese and count-only.

- [x] **Step 4: Run GREEN**

Run the Task 1 command. Expected: all source, geometry, and artifact tests pass.

- [x] **Step 5: Commit the artifact slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py
rtk git commit -m "feat: build P0-A diagnostic inventory artifacts"
```

## Task 3: Add CLI, atomic publication, and documentation with TDD

**Files:**
- Create: `framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_inventory.py`
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py`
- Modify: `docs/research/ovd_orbit_p0/README.md`
- Modify: `docs/research/ovd_orbit_p0/p0_protocol.md`

- [x] **Step 1: Write failing CLI and AST tests**

Call `cli.main(argv)` with temporary valid inputs. Require return `0`, exactly
five success artifacts, and the diagnostic readiness receipt. Call it again at
the same output path and require `FileExistsError` with unchanged receipt.
Mutate one selected annotation after source-plan construction and require return
`2` plus exactly `receipt.json`, `inventory_diagnostics.json`, and `result.md`.

For both new modules use `ast.parse` to reject import roots `torch`, `mmengine`,
`mmdet`, `mmrotate`, and `M_AD`, plus a `torch.load` call. Add document anchors
for `P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD`, `P0_INPUT_FAIL_STOP`,
`diagnostic_full_vocabulary`, and a strict-OVD non-claim.

- [x] **Step 2: Run RED**

Run the Task 1 command. Expected: CLI import and document-anchor failures.

- [x] **Step 3: Implement CLI and no-replace publication**

Implement the four CLI arguments. Reuse only the pure canonical JSON/hash
helpers and same-parent atomic `renameat2` with `RENAME_NOREPLACE` pattern from
the strict G0 module; never call its strict authority validator. Publishing
accepts safe flat names and nonempty bytes, fsyncs each file and parent
directories, cleans temporary output on failure, and fails closed if no-replace
rename is unavailable. On `P0AInventoryError`, publish exactly three failure
artifacts and return `2`; preserve `FileExistsError`.

Update README/protocol to label this a full-vocabulary diagnostic inventory
preparation, not strict G0 or a model result.

- [x] **Step 4: Run GREEN and compatibility verification**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0a_inventory.py tests/test_ovd_orbit_p0_g0.py tests/test_ovd_orbit_p0.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_Tools/analysis/ovd_orbit_p0a_inventory.py M_Tools/analysis/prepare_ovd_orbit_p0a_inventory.py
rtk git diff --check
```

Expected: all inventory, G0, and live-hook tests pass; modules compile; diff
check is silent.

- [x] **Step 5: Commit the CLI/documentation slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_inventory.py framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py docs/research/ovd_orbit_p0/README.md docs/research/ovd_orbit_p0/p0_protocol.md
rtk git commit -m "feat: add P0-A diagnostic inventory CLI"
```

## Task 4: Run the authorized real read-only inventory conversion

**Files:**
- Create through CLI: `docs/provenance/ovd_orbit_p0a_inventory_20260831/`
- Modify: `docs/research/ovd_orbit_p0/progress.md`

- [x] **Step 1: Verify sources before execution**

```bash
rtk sha256sum docs/provenance/risc_openrsd_n0o_v3/scene_plan_40.json
rtk jq '[.records[] | select(.fold_id == "c4_a" or .fold_id == "c4_b")] | length' docs/provenance/risc_openrsd_n0o_v3/scene_plan_40.json
```

Expected SHA-256 is `0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35`; expected count is `80`.

- [x] **Step 2: Run converter once**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0a_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m M_Tools.analysis.prepare_ovd_orbit_p0a_inventory --source-scene-plan ../../docs/provenance/risc_openrsd_n0o_v3/scene_plan_40.json --source-plan-sha256 0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35 --source-input-manifest ../../docs/provenance/risc_openrsd_n0o_v3/input_manifest.json --output-dir ../../docs/provenance/ovd_orbit_p0a_inventory_20260831
```

Expected return is `0` only for `P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD`; it
is `2` for `P0_INPUT_FAIL_STOP`. Never retry against the same output directory.

- [x] **Step 3: Verify receipt and record evidence**

```bash
rtk jq . ../../docs/provenance/ovd_orbit_p0a_inventory_20260831/receipt.json
rtk jq . ../../docs/provenance/ovd_orbit_p0a_inventory_20260831/inventory_diagnostics.json
rtk wc -l ../../docs/provenance/ovd_orbit_p0a_inventory_20260831/object_inventory.jsonl
```

On a failure receipt, record its exact error and stop. On readiness, verify 80
candidates, no P0148, and no model/GPU result, then append Chinese progress
text with path, status, source hash, scene count, row count, difficulty-2 count,
and “未执行模型前向、checkpoint 加载、GPU、AP 或 P0 指标计算。”

- [x] **Step 4: Commit verified code and evidence**

```bash
rtk git add docs/provenance/ovd_orbit_p0a_inventory_20260831 docs/research/ovd_orbit_p0/progress.md framework/openrsd/M_Tools/analysis/ovd_orbit_p0a_inventory.py framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0a_inventory.py framework/openrsd/tests/test_ovd_orbit_p0a_inventory.py
rtk git commit -m "docs: record P0-A diagnostic inventory"
```

## Plan self-review

- Task 1 covers exact DOTA parsing, rbox normalization, area, and convex IoU; Task 2 covers immutable C4 selection, annotation hashes, full vocabulary, and artifacts; Task 3 covers CLI/no-runtime/doc boundaries; Task 4 is the separately authorized real read-only conversion.
- No task defines base/novel classes, loads a model or checkpoint, imports a model framework, uses a GPU, or computes a semantic metric.
- The sole real run reads the user-authorized 80 C4 annotations and publishes one no-overwrite output directory.
