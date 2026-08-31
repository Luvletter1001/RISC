# OpenRSD P0 Protocol

## P0-A full-vocabulary diagnostic inventory receipt values

The P0-A preparer operates on the historical C4 scene selection and the
`diagnostic_full_vocabulary`: the 18 ordered DOTA class names. It is a
GT-only, CPU-only inventory construction step. It validates and hashes only
the declared annotation text files; it does not read image bytes.

- `P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD` means the 80 selected diagnostic
  scenes and their annotation-derived object inventory were prepared. It
  records difficulty-2 counts but excludes those objects from the retained
  inventory rows. No model forward was performed.
- `P0_INPUT_FAIL_STOP` means a declared P0-A input failed canonical JSON,
  source-plan, annotation, vocabulary, or hash validation. The CLI publishes
  only its minimal fail-stop receipt, diagnostics, and result files, exits 2,
  and authorizes no forward.

P0-A is not a strict OVD authority, not strict G0 sealing, and not a model
result or semantic metric. Its full vocabulary does not assert base/novel
splits, OVD performance, AP, or a scientific effect.

## P0-A diagnostic authority receipt values

`P0A_DIAGNOSTIC_AUTHORITY_READY_NO_FORWARD` means that the CPU-only authority
builder sealed the inventory receipt, fixed GT geometry policy, opaque text7
support-byte identity, and the three oracle-code identities. Its single
condition must state `prompt_stability_status = NOT_TESTED_SINGLE_CONDITION`.
The builder does not deserialize the support file or execute a model.

This readiness state is not model inference, strict G0 sealing, a strict-OVD
claim, prompt stability, AP, a semantic metric, or a paper result. It grants no
forward authorization and establishes no scientific effect.

`P0_INPUT_FAIL_STOP` for this builder publishes only a receipt,
`authority_diagnostics.json`, and `result.md`, exits 2, and authorizes no
forward.

## G0 input-seal receipt values

The G0 builder has exactly two terminal input-seal receipt values:

- `P0_INPUT_FAIL_STOP` has two fail-closed forms: a malformed/hash/asset
  mismatch in a declared authority, candidate-scene plan, canonical object
  inventory, or declared input-file byte/hash check produces a minimal failure
  package; a valid-but-below-G0-scope input produces a full diagnostic package.
  The CLI exits 2 in both forms, and neither form authorizes a forward.
- `G0_INPUTS_SEALED_NO_FORWARD` means the declared inputs passed G0 sealing and
  no model forward was performed. It is a receipt value that an actual
  successful G0 builder run may produce; its presence is not claimed for this
  implementation task.

The G0 view plan records the five canonical C4 views. `rot000_a` and
`rot000_b` are distinct identity-repeat records, alongside the non-identity C4
rotations.

Before either receipt state, the no-forward builder seals the ordered text
embedding hashes for all prompt families, the oracle mouth adapter and carrier
source identity schema, and the canonical digest of the exact P0-v1 threshold
bundle. This authority step uses no GPU and computes no P0 metric; synthetic
tests do not publish a receipt based on real project assets.

`G0_INPUTS_SEALED_NO_FORWARD` does not pass G1, G2, or G3; does not authorize a
method; and does not establish an effect, AP, or paper result. No actual model
run, no checkpoint deserialization, no dataset iterator, no GPU use, no P0
metric computation occurred. No receipt based on real project assets was
published, and no live-P0 receipt was published in this implementation task.

## Export boundary

The default-off live OpenRSD hook retains native classifier logits immediately
after HWC reshaping, then calls its in-memory sink after the normal calibration
path and before filtering. The canonical E0 `scores` field is the raw native
foreground-logit `[N,C]` tensor; optional `calibrated_scores` preserves the
post-calibration `[N,C]` diagnostic tensor. For softmax heads, only the final
background column is removed from raw logits so both fields share the same
foreground vocabulary. Aligned rotated boxes have shape `[N,5]`.

Every record uses the exact pre-selection source identity `(level,row)`. Duplicate class-emitted rows must be quotientable only through that exact source identity; score order, class label, text similarity, and IoU tie-breaking are prohibited as identity rules.

## E0 limits

- The live hook and `OvdOrbitFullLogitSink` perform sink-only, in-memory
  native-logit and optional calibration-diagnostic validation; the head and
  sink do not write a receipt.
- `run_ovd_orbit_p0.py` is the separate CPU-only record-validation and receipt
  serialization CLI.
- This status was established without an actual model forward, checkpoint load,
  dataset iteration, GPU allocation, or AP computation. No receipt based on
  real project assets was published, and no live-P0 receipt was published in
  this implementation task.
- Legacy object-array NPZ queue samples are not deserialized in E0.
- Existing top-1 JSON can support a later full-detector mouth, but never the oracle semantic mouth.

## Sink activation metadata

Attaching a sink requires each image metadata dictionary to provide the exact
nonempty strings `ovd_orbit_p0_scene_id` and `ovd_orbit_p0_view_id`. Missing or
invalid keys fail closed only while a sink is attached; a `None` sink does not
request or validate these P0 fields.

## Terminal state

`E0_CONTRACT_READY` means that a full-score JSONL fixture passed schema validation and can be written to a no-replace receipt. It is not evidence that a live model hook or a scientific phenomenon has been verified.

`E0_LIVE_HOOK_READY_NO_FORWARD` means the default-off in-memory native-logit
and calibrated-diagnostic lifecycle passed fake-head/sink contract tests. It
is not a real model run and cannot establish a scientific phenomenon, AP value,
G2 decision, or evidence of a live-P0 receipt based on real project assets.
