# OVD Object-Orbit P0 Live-Hook Design

## Goal

Expose both the OpenRSD head's complete native class-logit fields and its
complete calibrated pre-selection score fields, with aligned decoded boxes, to
the E0 exporter through an opt-in in-memory sink. Native logits are the P0
primary analysis field; calibrated scores are diagnostic only. Default detector
behavior, prediction outputs, model weights, and runtime path remain unchanged
when no sink is attached.

## Boundary

The hook lives at the only authoritative boundary in `OpenRotatedRTMDetSepBNHead._predict_by_feat_single`:

```text
raw cls logits
  -> sigmoid/softmax
  -> existing semantic calibration
  -> P0 dual-field sink       # new, opt-in; raw logits plus calibrated scores
  -> filter_scores_and_topk
  -> existing decode / NMS
```

The sink is called after semantic calibration and before `filter_scores_and_topk`.
It receives all rows of one FPN level, preserving (a) the raw native `[N,C]`
classifier logits captured before sigmoid/softmax and semantic calibration and
(b) the post-calibration `[N,C]` pre-filter scores. For softmax heads, the raw
background column is removed so both fields index the same foreground
vocabulary `C`. The head decodes aligned `[N,5]` boxes using the same
`bbox_coder.decode` and `img_shape` contract used by normal prediction. Source
identity is exactly `(level_idx,row)`.

## Components

### `OvdOrbitFullLogitSink`

Defined in `M_Tools/analysis/ovd_orbit_p0_export.py`. It owns only in-memory normalized `ScoreCarrier` records.

- `begin_image(scene_id, view_id)`: establishes required, nonempty metadata and a
  rollback boundary.
- `record_level(level, boxes, scores)`: required minimal sink API. `scores`
  remains the legacy E0 field name but contains native logits.
- `record_level_with_calibrated_scores(level, boxes, scores,
  calibrated_scores)`: optional capability. `OvdOrbitFullLogitSink` exposes it
  to preserve the explicit calibration diagnostic without requiring custom
  minimal sinks to accept a fourth argument.
- `end_image()`: commits one complete image context while retaining its record
  start solely for immediate exception recovery.
- `abort_image()`: removes every record created after either the active context
  began or the immediately prior committed context began, then clears the
  recovery state. `begin_image()` and a successful `snapshot()` clear that
  obsolete committed rollback boundary.
- `snapshot()`: returns a tuple sorted by scene, view, and exact source identity.
- Duplicate matching records collapse through the existing pure contract; conflicting records fail closed.

### Head attachment

`OpenRotatedRTMDetSepBNHead` receives an optional `ovd_orbit_p0_sink` attribute set only by the explicit `set_ovd_orbit_p0_sink(sink)` method. The default is `None`; no config creates it. A sink must expose the minimal `begin_image(scene_id=..., view_id=...)`, `record_level(level=..., boxes=..., scores=...)`, `end_image()`, and `abort_image()` methods. If present, `record_level_with_calibrated_scores(...)` must be callable and receives the optional diagnostic; the head never infers support by catching `TypeError` or inspecting a method signature.

### Metadata contract

The per-image `img_meta` must include two strings:

- `ovd_orbit_p0_scene_id`
- `ovd_orbit_p0_view_id`

If a sink is attached and either field is missing/empty, prediction fails before score filtering. This prevents accidental P0 artifacts with inferred or unstable identities. The sink never derives identity from emitted labels, scores, filenames, text similarity, or IoU.

`C4_VIEW_IDS = {rot000, rot090, rot180, rot270}` remains the E0 public,
physical-rotation contract. `OVD_ORBIT_P0_VIEW_IDS` accepts those legacy C4
records plus `rot000_a` and `rot000_b` for the two independently forwarded,
byte-identical zero-degree identity repeats required by G1. A scientific P0
identity-repeat run must use `rot000_a` and `rot000_b`; it must never collapse
both repeats into a generic `id` carrier.

## Invariants

1. Without a sink, `predict_by_feat` must produce bitwise-identical outputs for the same inputs.
2. With a sink, records are detached CPU copies; sink writes do not mutate input tensors or normal prediction tensors.
3. Every captured level preserves all class columns. The sink never transforms
   either field: the head supplies native logits and, separately, the existing
   calibrated score field.
4. The sink receives decoded boxes before filtering but does not influence filtering, `nms_pre`, score threshold, labels, or NMS.
5. No file is written by the head or sink. Receipt serialization remains the explicit CPU CLI's responsibility.
6. E0 validates fake/head-level tensors only. No `model.test_step`, dataset iterator, checkpoint load, GPU allocation, AP calculation, or training is permitted.
7. A failed image transaction, including an exception raised just after
   `end_image()` commits, leaves no records from that image serializable.

## Tests

- A fake sink proves that raw logits, calibrated `[N,C]` scores, and exact
  source rows are captured without conflating the two fields.
- Missing P0 metadata fails only when a sink is attached.
- A `None` sink preserves ordinary prediction results exactly.
- Sink mutation attempts cannot mutate head input tensors.
- A prediction exception after one captured level, or immediately after commit,
  aborts its image transaction; `snapshot()` cannot serialize a partial image.
- The existing E0 pure-contract tests remain green.

## Non-goals

- No live OpenRSD model run.
- No CastDet change.
- No P0 receipt publication from real predictions.
- No method loss, prompt adaptation, rotation evaluation, or LDM task registration.
