# OpenRSD E0 Export Contract

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
  dataset iteration, GPU allocation, AP computation, or receipt publication.
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
G2 decision, or published receipt.
