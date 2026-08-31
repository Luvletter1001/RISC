# OpenRSD P0-A Diagnostic Inventory Design

## Status

**Status:** implementation code is complete and synthetic CPU-only tests have
passed. Task 4 real conversion has not been executed; no real P0-A readiness
receipt is claimed.

This is a pre-G0, diagnostic-only preparation step for the official OpenRSD A10
parent. It creates the missing canonical object inventory from official DOTA
annotation text for a fixed C4 candidate pool. It is not a strict OVD protocol,
does not define base or novel classes, and cannot emit `OV_gap`, `DID_closed`,
`DID_shift`, a G1/G2/G3 decision, or a paper claim.

## Authority and scope

- Candidate source: `docs/provenance/risc_openrsd_n0o_v3/scene_plan_40.json`,
  SHA-256 `0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35`.
- Include only historical folds `c4_a` and `c4_b`: exactly 80 unique scenes.
- Exclude `P0148` at source-scene level before annotation parsing.
- Diagnostic vocabulary: the ordered 18-class DOTA2 support order recorded in
  `docs/provenance/risc_openrsd_n0o_v3/input_manifest.json`. It is one full
  diagnostic vocabulary, not a base/novel split.
- Annotation source: each selected record's official DOTA `.txt` path and
  SHA-256 from the candidate source.

The adapter performs only annotation-driven selection. It never opens selected
images, loads a checkpoint, reads support embeddings, imports a model framework,
constructs a dataset iterator, allocates a GPU, or evaluates a detector.

## Inputs and exact parsing contract

The new CLI receives `--source-scene-plan`, `--source-plan-sha256`,
`--source-input-manifest`, and `--output-dir`. It requires the canonical
historical source-scene-plan schema, verifies its whole-file SHA-256 against
the required `--source-plan-sha256`, then keeps only `c4_a` and `c4_b` records.
It rewrites those records into the G0 candidate schema with `split="diagnostic"`
while preserving image/annotation paths and hashes.

Each nonblank DOTA annotation line must contain exactly:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

Coordinates are finite decimal numbers; `class_name` belongs to the 18-class
diagnostic vocabulary; difficulty is integer `0`, `1`, or `2`. Difficulty `2`
is excluded under the standard ignored-DOTA annotation convention and recorded
in diagnostics. Difficulties `0` and `1` remain candidate objects. Malformed
lines, unexpected classes, duplicate object identities, missing annotations, or
annotation SHA-256 mismatches are fail-closed errors.

For each retained line, `object_id` is `${scene_id}:${source_line_number}`.
The adapter emits the canonical G0 five-value box `[cx, cy, w, h, theta]`:

1. `cx` and `cy` are the arithmetic means of the four vertices.
2. `w` is the distance from vertex 1 to 2; `h` is the distance from vertex 2
   to 3.
3. If `w < h`, swap them and add `pi/2` to the edge angle.
4. Normalize `theta` to the half-open interval `[-pi/2, pi/2)`.

`size` is the absolute shoelace area of the original quadrilateral. `overlap`
is the maximum convex-polygon IoU with every other retained difficulty-0/1
quadrilateral in the same scene, calculated with deterministic
Sutherland–Hodgman clipping. These quantities are descriptive inventory fields;
normal-size and isolation thresholds are deliberately not selected in this
step.

## Outputs and statuses

Publication is same-parent, no-overwrite and atomic. A successful inventory
package contains:

- `candidate_scene_plan.json`: normalized 80-scene G0-shaped candidate plan;
- `object_inventory.jsonl`: canonical rows compatible with the later G0 input
  schema, with no difficulty field or raw annotation text;
- `inventory_diagnostics.json`: source counts, retained/difficulty-2 counts,
  class counts, finite size and overlap quantiles, and asset verification counts;
- `receipt.json`: SHA-256 records for every non-receipt artifact;
- Chinese `result.md`: an input-preparation summary with no model metric.

Allowed receipt states:

- `P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD`: source plan, all 80 selected
  annotations, and canonical geometry inventory are valid;
- `P0_INPUT_FAIL_STOP`: any source hash, parsing, vocabulary, or identity
  violation. It writes only a minimal failure receipt, diagnostics, and result
  summary.

Neither receipt is a strict G0 seal. The readiness receipt merely authorizes a
later decision on geometry thresholds and diagnostic prompt authority.

## Components and tests

`M_Tools/analysis/ovd_orbit_p0a_inventory.py` is standard-library geometry and
contract code. `M_Tools/analysis/prepare_ovd_orbit_p0a_inventory.py` is a thin
CLI adapter. They must not import torch, mmengine, mmdet, mmrotate, or the
OpenRSD head.

`tests/test_ovd_orbit_p0a_inventory.py` uses tiny temporary annotation files
and a synthetic source plan. It verifies source-plan selection/P0148 rejection,
annotation hash checks, exact box normalization, area, convex IoU, difficulty-2
accounting, deterministic line-order output, atomic no-overwrite publication,
minimal failure artifacts, and source-plan snapshot-to-rehash TOCTOU rejection.
A source-level AST test prevents ML-runtime imports and `torch.load` calls in
both new modules.

## Handoff gate

After a real successful P0-A inventory package exists, we inspect only its GT
geometry distributions to propose sealed normal-size/isolation thresholds and
choose diagnostic prompt families. That requires a separate explicit authority
design. It does not authorize model inference or strict OVD claims.
