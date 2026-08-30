# OVD Object-Orbit P0 G0 Input Seal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, CPU-only G0 input-seal command that turns a predeclared P0 authority, a candidate scene plan, and a canonical object inventory into deterministic P0 inputs and a no-overwrite receipt without loading a model or a checkpoint.

**Architecture:** A pure-Python contract module validates canonical JSON, streams SHA-256 for opaque assets, seals combined base/novel vocabulary, selects annotation-only eligible objects, expands explicit C4/identity-repeat views, and emits deterministic bytes. A small CLI only reads supplied paths and publishes the returned artifact bytes atomically. A malformed or incomplete input publishes a `P0_INPUT_FAIL_STOP` receipt; only a complete package receives `G0_INPUTS_SEALED_NO_FORWARD`.

**Tech Stack:** Python 3.10 standard library, existing OpenRSD `pytest` environment, JSON/JSONL, SHA-256, `pathlib`; no PyTorch/MMEngine/MMRotate import.

---

## File structure

- Create `framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py` — pure contract, validation, eligibility, view expansion, deterministic JSON and atomic no-overwrite writer.
- Create `framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py` — CLI adapter; parses four paths and maps a `G0SealError` to exit code 2 only after publishing the fail-closed receipt.
- Create `framework/openrsd/tests/test_ovd_orbit_p0_g0.py` — fixture builders plus all contract and CLI tests; it imports neither live head code nor a model framework.
- Modify `docs/research/ovd_orbit_p0/README.md` — document the new G0 builder as unexecuted infrastructure and distinguish it from a sealed run.
- Modify `docs/research/ovd_orbit_p0/p0_protocol.md` — add the two G0 receipt states and make clear that `G0_INPUTS_SEALED_NO_FORWARD` is not G1/G2 evidence.
- Modify `docs/research/ovd_orbit_p0/progress.md` — record implementation verification only after tests pass; never record a scientific result.

## Shared contract

The CLI uses four explicit paths:

```text
--authority-json AUTHORITY.json
--candidate-scene-plan CANDIDATES.json
--object-inventory OBJECTS.jsonl
--output-dir NEW_DIRECTORY
```

`AUTHORITY.json` must be canonical JSON with this exact top-level schema:

```json
{
  "schema":"ovd-orbit-p0-g0-authority-v1",
  "forbidden_scene_id":"P0148",
  "candidate_scene_plan_sha256":"<64 lowercase hex>",
  "object_inventory_sha256":"<64 lowercase hex>",
  "checkpoint":{"path":"/sealed/checkpoint.pth","sha256":"<64 lowercase hex>"},
  "resolved_config":{"path":"/sealed/config.py","sha256":"<64 lowercase hex>"},
  "code":{"commit":"<nonempty string>","files":[{"path":"/sealed/readout.py","sha256":"<64 lowercase hex>"}]},
  "vocabulary":{
    "base":[{"name":"base-a","provenance":"training-ledger"}],
    "novel":[{"name":"novel-a","provenance":"holdout-ledger"}]
  },
  "prompt_families":[
    {"name":"primary","sha256":"<64 lowercase hex>"},
    {"name":"template-b","sha256":"<64 lowercase hex>"},
    {"name":"template-c","sha256":"<64 lowercase hex>"}
  ],
  "primary_prompt_family":"primary",
  "native_temperature":{"rule_id":"native","sha256":"<64 lowercase hex>"},
  "render_contract":{"kind":"lossless-square-c4","sha256":"<64 lowercase hex>"},
  "eligibility_policy":{
    "schema":"canonical-object-inventory-v1",
    "min_size":10.0,
    "max_size":1000.0,
    "max_overlap":0.10,
    "sha256":"<64 lowercase hex>"
  },
  "bootstrap":{"seed":17,"repetitions":2000}
}
```

The values above are shape examples only. Each asset path is an explicit,
nonempty regular-file path supplied by the authority; the executable performs
no path interpolation and hashes its bytes without deserializing it. The
executable accepts no implicit default for any science-facing field. `base` and `novel` are both nonempty,
their names are unique within and disjoint across groups, every provenance is
nonempty, prompt-family names are exactly three and unique, and the primary
name is one of them.

`CANDIDATES.json` is canonical JSON with `schema` equal to
`ovd-orbit-p0-candidate-plan-v1` and a nonempty `records` list. Every record
has `scene_id`, `split`, `image_path`, `image_sha256`, `annotation_path`, and
`annotation_sha256`. The validator allows exactly one record per `scene_id`,
requires each split to be a nonempty string, and therefore rejects all
cross-split leakage. `P0148` is rejected before reading object rows.

`OBJECTS.jsonl` has one canonical JSON object per nonblank line:

```json
{"annotation_sha256":"<64 lowercase hex>","box":[1.0,2.0,3.0,4.0,0.0],"class_name":"base-a","object_id":"scene-001:0","overlap":0.02,"scene_id":"scene-001","size":100.0}
```

Object rows are annotation-only. `box` is five finite numbers, `size` and
`overlap` are finite nonnegative numbers, `class_name` belongs to the ordered
combined vocabulary, the row annotation hash agrees with its candidate scene,
and `(scene_id, object_id)` is unique. A row is eligible iff
`min_size <= size <= max_size` and `overlap <= max_overlap`; otherwise it is
persisted with exactly one reason: `below_min_size`, `above_max_size`, or
`overlap_exceeds_max`.

Every eligible object expands in this exact order:

```python
VIEW_IDS = ('rot000_a', 'rot000_b', 'rot090', 'rot180', 'rot270')
```

`rot000_a` and `rot000_b` share a deterministic `render_digest`, but retain
different `view_id` values. Their carrier identities must never be collapsed.

## Task 1: Establish the pure module and validation primitives with TDD

**Files:**
- Create: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py`
- Create: `framework/openrsd/tests/test_ovd_orbit_p0_g0.py`

- [x] **Step 1: Write the first failing authority tests**

Add tiny fixture builders that return canonical in-memory authority and
candidate dictionaries. Add these tests before production code:

```python
def test_validate_authority_seals_disjoint_provenanced_combined_vocabulary():
    authority = authority_dict()

    sealed = module.validate_authority(authority)

    assert sealed.vocabulary == ('base-a', 'novel-a')
    assert sealed.primary_prompt_family == 'primary'


def test_validate_authority_rejects_overlapping_or_unprovenanced_classes():
    overlapping = authority_dict()
    overlapping['vocabulary']['novel'][0]['name'] = 'base-a'
    with pytest.raises(module.G0SealError, match='disjoint'):
        module.validate_authority(overlapping)

    missing_provenance = authority_dict()
    missing_provenance['vocabulary']['novel'][0]['provenance'] = ''
    with pytest.raises(module.G0SealError, match='provenance'):
        module.validate_authority(missing_provenance)
```

- [x] **Step 2: Run RED**

Run from `framework/openrsd`:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0_g0.py -q
```

Expected: import failure because `ovd_orbit_p0_g0.py` does not exist.

- [x] **Step 3: Implement only canonical JSON, hashes, and authority parsing**

Implement these names in the new module:

```python
class G0SealError(ValueError):
    pass


@dataclass(frozen=True)
class SealedAuthority:
    forbidden_scene_id: str
    vocabulary: Sequence[str]  # immutable ordered class names
    base_classes: Sequence[str]  # derived from raw `vocabulary.base`; not a JSON key
    novel_classes: Sequence[str]  # derived from raw `vocabulary.novel`; not a JSON key
    primary_prompt_family: str
    raw: Mapping[str, Any]


# canonical_json_bytes(value: Mapping[str, Any]) -> bytes
# sha256_bytes(value: bytes) -> str
# sha256_file(path: Path | str) -> str
# load_canonical_json(path: Path | str, *, label: str) -> Mapping[str, Any]
# validate_authority(value: Mapping[str, Any]) -> SealedAuthority
```

`canonical_json_bytes` uses UTF-8, `ensure_ascii=False`, sorted keys, compact
separators, `allow_nan=False`, and exactly one trailing newline. Validate every
declared SHA-256 as lowercase 64-character hex. `validate_authority` must
require all shared-contract keys, exactly three prompt families, a primary
prompt member, a nonempty code commit/file list, finite ordered policy values
with `0 <= min_size <= max_size` and `max_overlap >= 0`, and positive integer
bootstrap seed/repetitions. Do not import NumPy, torch, MMEngine, or MMRotate.

- [x] **Step 4: Run GREEN**

Run the Task 1 command. Expected: both authority tests pass.

- [x] **Step 5: Commit the pure authority slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py framework/openrsd/tests/test_ovd_orbit_p0_g0.py
rtk git commit -m "feat: add P0 G0 authority validation"
```

## Task 2: Validate candidate and object rows, then select eligibility with TDD

**Files:**
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0_g0.py`

- [x] **Step 1: Write failing scene and eligibility tests**

Use two candidate rows and four object rows. Require that the selector retains
one eligible row and reports all three sealed exclusion reasons:

```python
def test_select_eligible_objects_accounts_for_each_policy_exclusion():
    authority = module.validate_authority(authority_dict())
    candidates = module.validate_candidate_plan(candidate_plan())
    rows = module.validate_object_rows(object_rows(), candidates, authority)

    decisions, eligible = module.select_eligible_objects(rows, authority)

    assert [row['object_id'] for row in eligible] == ['scene-001:keep']
    assert {row['exclusion_reason'] for row in decisions if not row['eligible']} == {
        'below_min_size', 'above_max_size', 'overlap_exceeds_max'}


def test_candidate_plan_rejects_p0148_and_cross_split_scene_identity():
    forbidden = candidate_plan()
    forbidden['records'][0]['scene_id'] = 'P0148'
    with pytest.raises(module.G0SealError, match='P0148'):
        module.validate_candidate_plan(forbidden)

    leaked = candidate_plan_with_same_scene_in_two_splits()
    with pytest.raises(module.G0SealError, match='leakage'):
        module.validate_candidate_plan(leaked)
```

Also add negative tests for unknown class, duplicate `(scene_id, object_id)`,
object annotation-hash mismatch, malformed five-value box, and nonfinite size.

- [x] **Step 2: Run RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0_g0.py -q
```

Expected: failure because candidate/object validation and selection functions do
not exist.

- [x] **Step 3: Implement deterministic candidate/object validation**

Add exactly these public functions:

```python
# validate_candidate_plan(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]
# load_object_rows(path: Path | str) -> immutable ordered object-row tuple
def validate_object_rows(
        rows: Iterable[Mapping[str, Any]],
        candidates: Mapping[str, Mapping[str, Any]],
        authority: SealedAuthority) -> immutable ordered validated-row tuple
def select_eligible_objects(
        rows: Iterable[Mapping[str, Any]],
        authority: SealedAuthority) -> (immutable decision-row tuple, immutable eligible-row tuple)
```

Return candidates keyed by `scene_id`, preserving canonical row dictionaries.
`load_object_rows` reads UTF-8 JSONL, rejects blank-only files and noncanonical
line JSON, and reports line numbers. Build decision rows with original identity,
class, box, size, overlap, `eligible`, and `exclusion_reason`; use `None` for
eligible rows. Sort all returned rows by `(scene_id, object_id)` so input line
order cannot alter the package.

- [x] **Step 4: Run GREEN**

Run the Task 2 command. Expected: all authority, candidate, and eligibility
tests pass.

- [x] **Step 5: Commit the selection slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py framework/openrsd/tests/test_ovd_orbit_p0_g0.py
rtk git commit -m "feat: add P0 G0 candidate eligibility seal"
```

## Task 3: Build C4 identities and deterministic success/failure artifacts with TDD

**Files:**
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0_g0.py`

- [x] **Step 1: Write failing view-plan and receipt tests**

Add a synthetic input with 80 distinct scenes, 800 eligible objects, and eight
classes. Require a success package and exact five-view expansion:

```python
def test_build_g0_artifacts_emits_distinct_identity_repeats_and_success_receipt(tmp_path):
    package = module.build_g0_artifacts(
        authority=module.validate_authority(authority_dict()),
        candidate_plan=candidate_plan_for_scope(),
        object_rows=object_rows_for_scope(),
        candidate_plan_sha256='a' * 64,
        object_inventory_sha256='b' * 64,
        asset_hashes={'checkpoint': 'c' * 64, 'resolved_config': 'd' * 64},
    )

    views = [json.loads(line) for line in package['object_view_plan.jsonl'].splitlines()]
    first = [row for row in views if row['object_id'] == 'scene-001:0']
    assert [row['view_id'] for row in first] == list(module.VIEW_IDS)
    assert first[0]['render_digest'] == first[1]['render_digest']
    assert first[0]['view_id'] != first[1]['view_id']
    assert json.loads(package['receipt.json'])['status'] == 'G0_INPUTS_SEALED_NO_FORWARD'
```

Also assert a 79-scene or 799-object candidate produces a package whose receipt
is `P0_INPUT_FAIL_STOP`, includes `g0_scope_ready: false`, and does not claim
strict OVD readiness. Add an assertion that output bytes are identical when
object rows are supplied in reverse order.

- [x] **Step 2: Run RED**

Run the Task 2 pytest command. Expected: failure because `VIEW_IDS` and
`build_g0_artifacts` do not exist.

- [x] **Step 3: Implement view expansion and artifact construction**

Add:

```python
VIEW_IDS = ('rot000_a', 'rot000_b', 'rot090', 'rot180', 'rot270')

# build_view_plan(rows: Iterable[Mapping[str, Any]]) -> immutable ordered view-row tuple
# build_g0_artifacts(
#     *, authority: SealedAuthority,
#     candidate_plan: Mapping[str, Mapping[str, Any]],
#     object_rows: Iterable[Mapping[str, Any]],
#     candidate_plan_sha256: str,
#     object_inventory_sha256: str,
#     asset_hashes: Mapping[str, str]) -> dict[str, bytes]
```

For each eligible decision row, calculate a render digest from canonical JSON
containing only `scene_id`, `object_id`, `box`, and the authority render
contract hash. Use it unchanged for both zero-degree identity rows; for other
views include `view_id` in the digest payload. Construct a diagnostics mapping
with scene/object/class/split/exclusion counts, `g0_scope_ready`, `strict_ovd`
counts, and `strict_ovd_ready`. Strict readiness is true only when novel-object
count is at least 300 and novel-class count at least five; it never changes the
success status of an otherwise valid diagnostic G0 package.

Create `input_manifest.json`, `object_eligibility.jsonl`,
`object_view_plan.jsonl`, `seal_diagnostics.json`, `receipt.json`, and
`result.md` with canonical UTF-8 bytes. `result.md` must state only input counts
and the status; it must contain no AP, rotation metric, P0 effect, or paper
novelty language. The receipt hashes every other generated artifact.

- [x] **Step 4: Run GREEN**

Run the Task 2 command. Expected: complete deterministic package tests pass.

- [x] **Step 5: Commit the artifact slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py framework/openrsd/tests/test_ovd_orbit_p0_g0.py
rtk git commit -m "feat: build P0 G0 deterministic seal artifacts"
```

## Task 4: Add the no-overwrite CLI with TDD

**Files:**
- Create: `framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py`
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0_g0.py`

- [x] **Step 1: Write failing CLI and publication tests**

Use `tmp_path` to write canonical authority/candidate/object inputs and execute
the CLI through its `main(argv)` function:

```python
def test_cli_publishes_success_once_and_refuses_overwrite(tmp_path):
    authority_path, candidate_path, objects_path = write_success_inputs(tmp_path)
    output_dir = tmp_path / 'g0-seal'

    assert cli.main([
        '--authority-json', str(authority_path),
        '--candidate-scene-plan', str(candidate_path),
        '--object-inventory', str(objects_path),
        '--output-dir', str(output_dir),
    ]) == 0
    assert json.loads((output_dir / 'receipt.json').read_text())['status'] == 'G0_INPUTS_SEALED_NO_FORWARD'
    args = [
        '--authority-json', str(authority_path),
        '--candidate-scene-plan', str(candidate_path),
        '--object-inventory', str(objects_path),
        '--output-dir', str(output_dir),
    ]
    with pytest.raises(FileExistsError):
        cli.main(args)
```

Add an asset-hash-mismatch fixture where a tiny checkpoint byte file differs
from the declared authority hash. Require exit code 2, a newly published
`receipt.json` with `P0_INPUT_FAIL_STOP`, and no `object_view_plan.jsonl`.

- [x] **Step 2: Run RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0_g0.py -q
```

Expected: CLI import failure because `prepare_ovd_orbit_p0_g0_seal.py` does not
exist.

- [x] **Step 3: Implement atomic no-overwrite publication and CLI**

In the pure module add:

```python
# publish_artifacts(output_dir: Path | str, artifacts: Mapping[str, bytes]) -> None
# build_failure_artifacts(error: G0SealError) -> dict[str, bytes]
# verify_authority_assets(authority: SealedAuthority) -> dict[str, str]
```

`publish_artifacts` creates a same-parent temporary directory, writes and
flushes every named bytes payload, atomically renames it into a previously
nonexistent output path, and raises `FileExistsError` when the target already
exists. It must clean its temporary directory when writing fails. Failure
artifacts contain only canonical `receipt.json`, `seal_diagnostics.json`, and
Chinese `result.md`; they include the safe validation message but no artifact
claim or view plan.

The CLI uses `argparse`, resolves no defaults beyond supplied paths, loads
canonical authority/candidate JSON and object JSONL, then calls
`verify_authority_assets(authority)`. That function streams the declared
checkpoint, resolved-config, and code-file paths exactly as written in the
authority and returns a stable mapping of asset keys to observed SHA-256 values;
it raises `G0SealError('<asset> hash mismatch')` on a missing or different file.
Pass that mapping to `build_g0_artifacts`. The CLI must not deserialize the
checkpoint or inspect its container.
It must not import torch, call `torch.load`, import the dense head, allocate a
GPU, or start a dataset iterator. On a valid success return 0; after publishing
a contract failure return 2; preserve `FileExistsError` for an existing output
directory.

- [x] **Step 4: Run GREEN and baseline regression**

Run from `framework/openrsd`:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0_g0.py tests/test_ovd_orbit_p0.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_Tools/analysis/ovd_orbit_p0_g0.py M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py
```

Expected: all G0 and existing live-hook tests pass; both modules compile.

- [x] **Step 5: Commit the CLI slice**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py framework/openrsd/tests/test_ovd_orbit_p0_g0.py
rtk git commit -m "feat: add P0 G0 input seal CLI"
```

## Task 5: Document only the implemented, no-forward boundary

**Files:**
- Modify: `docs/research/ovd_orbit_p0/README.md`
- Modify: `docs/research/ovd_orbit_p0/p0_protocol.md`
- Modify: `docs/research/ovd_orbit_p0/progress.md`

- [x] **Step 1: Write failing documentation assertions**

Add a small text-based test in `tests/test_ovd_orbit_p0_g0.py` that reads the
three documents and requires all of: `G0_INPUTS_SEALED_NO_FORWARD`,
`P0_INPUT_FAIL_STOP`, `rot000_a`, `rot000_b`, and a statement that G0 does not
run a model or compute a P0 metric.

- [x] **Step 2: Run RED**

Run the Task 4 pytest command. Expected: at least one required G0 string is
absent from the current documentation.

- [x] **Step 3: Update the documents**

In `README.md`, add the G0 builder as **unexecuted infrastructure** and state
that only an actual sealed package can hold `G0_INPUTS_SEALED_NO_FORWARD`.
In `p0_protocol.md`, define the two receipt states and repeat that G0 neither
passes G1 nor provides a P0 effect. In `progress.md`, record code/test
verification with the exact sentence: “No authority package was generated from
real project assets in this implementation task.”

- [x] **Step 4: Run GREEN and final verification**

Run from `framework/openrsd`:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0_g0.py tests/test_ovd_orbit_p0.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_Tools/analysis/ovd_orbit_p0_g0.py M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py
rtk git diff --check
rtk git status --short
```

Expected: all focused tests pass, both modules compile, `git diff --check` is
silent, and changed files are limited to this plan's listed module, CLI, test,
and documentation paths.

- [x] **Step 5: Commit the documented G0 implementation**

```bash
rtk git add docs/research/ovd_orbit_p0/README.md docs/research/ovd_orbit_p0/p0_protocol.md docs/research/ovd_orbit_p0/progress.md framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py framework/openrsd/tests/test_ovd_orbit_p0_g0.py
rtk git commit -m "docs: record P0 G0 input seal boundary"
```

## Plan self-review

- Spec coverage: Task 1 seals authority, Task 2 seals candidates/object
  eligibility and P0148/leakage, Task 3 seals C4 identity views and artifacts,
  Task 4 seals opaque asset hashes plus no-overwrite publication, and Task 5
  prevents an implementation claim from being mistaken for a scientific run.
- No model code, checkpoint deserialization, dataloader, GPU, AP, bootstrap,
  G1/G2/G3 calculation, CastDet change, or training appears in any task.
- All public names introduced by later tasks are defined in the shared contract
  or an earlier task; tests specify their expected failure and success states.

## Protocol-authority completion addendum

This addendum is required by the controlling P0 protocol before the branch can
be considered G0-complete. Prompt-definition hashes and the three currently
implemented sampling counts alone are not sufficient.

### Task 6: Seal remaining protocol authority and harden boundary tests

**Files:**
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py`
- Modify: `framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py`
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0_g0.py`
- Modify: `docs/superpowers/specs/2026-08-31-ovd-orbit-p0-g0-input-seal-design.md`
- Modify: `docs/research/ovd_orbit_p0/README.md`
- Modify: `docs/research/ovd_orbit_p0/p0_protocol.md`
- Modify: `docs/research/ovd_orbit_p0/progress.md`

- [x] **Step 1: Write failing protocol-authority and AST tests**

Extend the synthetic authority fixture with these exact new fields:

```python
"text_embedding_hashes": [
    {"prompt_family": "primary", "sha256": "1" * 64},
    {"prompt_family": "template-b", "sha256": "2" * 64},
    {"prompt_family": "template-c", "sha256": "3" * 64},
],
"oracle_mouth": {
    "adapter_type": "dense-carrier-fallback",
    "carrier_source_identity_schema": "level-row-v1",
    "definition_sha256": "4" * 64,
},
"protocol_threshold_bundle": {
    "g0": {"min_scenes": 80, "min_objects": 800, "min_supported_classes": 8,
           "min_novel_objects": 300, "min_novel_classes": 5},
    "g1": {"identity_p99": 0.0001, "noise_multiplier": 1.25,
           "aggregation_relative_difference": 0.25},
    "g2": {"er_acc": 0.03, "smd_margin": 0.20, "er_js": 0.01,
           "min_passing_conditions": 2, "min_prompt_intervals": 2},
    "g3": {"min_model_families": 2, "did_closed_smd": 0.20,
           "did_shift_smd": 0.20, "replication_ratio_min": 0.33,
           "replication_ratio_max": 3.0},
    "g4": {"min_optimizer_steps": 100, "max_optimizer_steps": 250},
},
```

Set `protocol_threshold_bundle_sha256` to the SHA-256 of canonical bundle
bytes. Add tests rejecting missing/reordered/extra text-embedding family,
missing oracle adapter or carrier schema, threshold-bundle digest mismatch,
missing `g0`–`g4` section, and changed P0-v1 G0/strict-count values. Assert the
manifest contains the sealed oracle/text/threshold identities, and a changed
sealed G0 count is rejected before scope selection.

Replace brittle source-string assertions with `ast.parse`: reject imports whose
root is `torch`, `M_AD`, `mmcv`, `mmdet`, or `mmengine`, and reject a call whose
attribute chain is `torch.load`. Replace full-sentence documentation matching
with semantic anchors for receipt states, no-forward boundary, text embeddings,
oracle mouth, carrier identity, and threshold bundle.

- [x] **Step 2: Run RED**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0_g0.py -q
```

Expected: failures for missing protocol-authority fields and obsolete static
test assumptions.

- [x] **Step 3: Implement sealed protocol authority**

Require exactly three ordered text-embedding records whose `prompt_family`
sequence equals the prompt-family sequence; each hash is lowercase 64-hex.
Require the exact three-key `oracle_mouth` mapping shown above. Require a
threshold bundle with exactly `{g0,g1,g2,g3,g4}`, a matching canonical digest,
finite non-bool numeric values, and the P0-v1 G0/strict count values from Step
1. Derive scope and strict-readiness thresholds from the bundle, removing
duplicated literals. Include all new identities and the bundle digest in
`input_manifest.json`; do not import an ML runtime.

- [x] **Step 4: Update docs and run GREEN**

Update README/protocol/progress to say the implementation seals text-embedding
hashes, oracle adapter/carrier identity, and the protocol threshold bundle,
while still recording no real-asset seal or scientific result. Then run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0_g0.py tests/test_ovd_orbit_p0.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_g0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_Tools/analysis/ovd_orbit_p0_g0.py M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py
rtk git diff --check
```

Expected: all focused tests pass, modules compile, and diff check is silent.

- [x] **Step 5: Commit the protocol-authority completion**

```bash
rtk git add framework/openrsd/M_Tools/analysis/ovd_orbit_p0_g0.py framework/openrsd/M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py framework/openrsd/tests/test_ovd_orbit_p0_g0.py docs/superpowers/specs/2026-08-31-ovd-orbit-p0-g0-input-seal-design.md docs/research/ovd_orbit_p0/README.md docs/research/ovd_orbit_p0/p0_protocol.md docs/research/ovd_orbit_p0/progress.md
rtk git commit -m "feat: seal P0 G0 protocol authority"
```
