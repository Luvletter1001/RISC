# OVD Object-Orbit P0 G0 Input Seal Design

## Status and decision

**Status:** implemented and CPU-tested with synthetic fixtures; no real project
assets were executed and no G0 success status is claimed.

This is the next stage after `E0_LIVE_HOOK_READY_NO_FORWARD`. It prepares a
fail-closed, CPU-only G0 authority package for the P0-A diagnostic route. It
does not run OpenRSD, deserialize a checkpoint, construct a dataloader, write
a receipt from live predictions, allocate a GPU, compute AP, or make a
scientific claim.

The historical N0-O 160-scene plan is a **candidate pool only**. Its hashes and
P0148 exclusion are useful evidence, but it is not automatically a valid P0
sample because it was sealed for a different RISC question. P0 eligibility,
vocabulary, prompts, and transformation identity are rebuilt under this design.

## Research authority

- P0 scientific protocol: `docs/research/ovd_orbit_p0/p0_protocol.md` and
  `/data1/zcy/Large-Discovery-Models/research_sessions/risc_ovd_rotation_p0_20260830/p0_protocol_final.md`.
- Current live-hook authority:
  `docs/superpowers/specs/2026-08-30-ovd-orbit-p0-live-hook-design.md`.
- Historical candidate-pool authority:
  `docs/superpowers/specs/2026-08-22-risc-openrsd-n0o-input-seal-design.md`.

The P0 protocol controls whenever these sources differ. In particular, G0 must
seal a combined `B union N` vocabulary and an object-level eligibility policy;
the N0-O plan cannot supply either by implication.

## Goal

Build one deterministic command, `prepare_ovd_orbit_p0_g0_seal.py`, that reads
a caller-supplied canonical authority JSON plus candidate scene and annotation
inputs and writes a no-overwrite G0 package:

```text
authority JSON + candidate scene plan + annotations
  -> canonical input/hash verification
  -> scene leakage and P0148 exclusion checks
  -> object-level eligibility selection
  -> C4 / identity-repeat view plan
  -> G0 manifest + receipt
```

The output is an input seal, not model evidence. It contains no images, raw
annotations, weights, text embeddings, logits, detections, or personal data.

## Inputs

The command receives paths explicitly; it contains no hard-coded dataset,
checkpoint, or user-home path.

### Canonical authority JSON

The authority JSON is canonical JSON and supplies all scientific choices before
model output is inspected:

- `checkpoint`: path and expected SHA-256; it is hashed as opaque bytes and is
  never deserialized;
- resolved config path and expected SHA-256;
- expected Git commit and the list of code paths to hash;
- `base_classes` and `novel_classes`, each nonempty, disjoint, ordered, and
  accompanied by a provenance string; their ordered union is the only allowed
  P0 vocabulary;
- exactly three named prompt families, one declared primary, and a hash for
  each prompt definition;
- one immutable native temperature/logit-scale rule identifier and hash;
- a C4 render contract describing lossless square-image rotation;
- an eligibility policy containing explicit, predeclared normal-size and
  isolation thresholds, annotation format/version, and a policy hash;
- a sealed bootstrap seed and repetition count;
- the forbidden P0148 source-scene ID.

No default threshold, base/novel split, prompt, or temperature may be inferred
from dataset labels or model scores. Missing or noncanonical authority JSON is
an input failure.

### Candidate scene plan and annotations

The candidate plan must be canonical JSON. Each candidate references one scene,
one image identifier, and one annotation identifier with expected SHA-256. The
historical N0-O 160-scene plan may be passed here only after its own source hash
is supplied in the authority JSON.

Annotations are consumed through a small versioned adapter that emits only
canonical object rows: `scene_id`, `object_id`, `class_name`, rotated box,
size statistic, and overlap statistic. The adapter must reject unknown classes,
duplicate object identities, missing hashes, and unsupported formats. It must
not call a model or a dataset iterator.

## Selection and identity contract

1. Verify every configured file hash and selected candidate image/annotation
   hash. A checkpoint is streamed for SHA-256 only; it is never loaded by
   PyTorch.
2. Reject any scene appearing in more than one declared split.
3. Reject the P0148 source scene before object filtering.
4. Apply the sealed normal-size and isolation inequalities to annotation-only
   object rows. Persist a row for every included and excluded object, with one
   machine-readable exclusion reason.
5. Require globally unique pre-transform `(scene_id, object_id)` identities.
6. Produce five view rows for every eligible object:
   `rot000_a`, `rot000_b`, `rot090`, `rot180`, and `rot270`. The two zero-degree
   rows are separate forward identities with the same render digest; they never
   share a carrier key.
7. Require at least 80 scene-disjoint scenes, 800 eligible objects, and eight
   supported classes for `g0_scope_ready`. The stricter OVD confirmation counts
   (300 novel objects and five novel classes) are reported separately; they do
   not become true merely because an OpenRSD diagnostic seal exists.

## Output package

The command writes only to a new output directory and refuses overwrite:

- `input_manifest.json`: authority hashes, candidate-pool hash, code/config/
  checkpoint identities, split/leakage counts, vocabulary/prompt/temperature
  hashes, and view-plan digest;
- `object_eligibility.jsonl`: one canonical row per candidate object, including
  decision and exclusion reason, but no image bytes;
- `object_view_plan.jsonl`: one row per eligible object/view identity;
- `seal_diagnostics.json`: deterministic counts by split, class, and exclusion
  reason plus strict-OVD readiness counts;
- `receipt.json`: one status and hashes of every generated artifact;
- Chinese `result.md`: a human-readable table with no metric values or paper
  claim.

The receipt has only two outcomes:

- `P0_INPUT_FAIL_STOP`: one or more authority, hash, leakage, vocabulary,
  identity, or eligibility requirements failed;
- `G0_INPUTS_SEALED_NO_FORWARD`: all G0 requirements passed, but no model
  result was generated. This is a progress state, not a P0 phenomenon decision.

## Components and boundaries

`M_Tools/analysis/ovd_orbit_p0_g0.py` is a pure-Python contract module. It owns
canonical JSON, SHA-256, authority validation, candidate/annotation row
validation, selection, C4 identity expansion, and deterministic receipts. It
does not import PyTorch, MMEngine, MMRotate, or the live-hook head.

`M_Tools/analysis/prepare_ovd_orbit_p0_g0_seal.py` is a narrow CLI adapter. It
reads files, streams hashes, delegates decisions to the pure module, and writes
the no-overwrite package. It does not deserialize checkpoints or invoke a
model.

`tests/test_ovd_orbit_p0_g0.py` uses only tiny JSON/annotation fixtures. Tests
must prove deterministic output, P0148 rejection, split-leakage rejection,
hash mismatch rejection, missing `B union N` provenance rejection, identity
repeat distinctness, policy-based exclusion accounting, no-overwrite behavior,
and correct receipt status.

## Non-goals and handoff gate

- No live P0 receipt, OpenRSD model forward, checkpoint deserialization,
  dataloader, GPU use, AP, bootstrap, `G1`, `G2`, `G3`, or training.
- No claim that OpenRSD is a strict independent OVD parent.
- No CastDet integration; that is a later independent-parent task.

Only `G0_INPUTS_SEALED_NO_FORWARD` authorizes a separate, explicitly approved
P0-A evaluator-smoke plan. That later plan must attach the already-merged live
sink, load one sealed checkpoint, and verify export behavior before measuring
any P0 estimand.
