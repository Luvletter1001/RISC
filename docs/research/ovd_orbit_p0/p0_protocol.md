# OpenRSD E0 Export Contract

## Export boundary

The future live OpenRSD hook must call `normalize_level_outputs` immediately after an aligned full semantic score tensor with shape `[N,C]` and its matching rotated boxes with shape `[N,5]` are available. This occurs before per-class thresholding, top-k selection, class-label emission, or NMS.

Every record uses the exact pre-selection source identity `(level,row)`. Duplicate class-emitted rows must be quotientable only through that exact source identity; score order, class label, text similarity, and IoU tie-breaking are prohibited as identity rules.

## E0 limits

- `run_ovd_orbit_p0.py` is CPU-only record validation and receipt serialization.
- It must not invoke a detector, `model.test_step`, a GPU, AP evaluation, or an optimizer.
- Legacy object-array NPZ queue samples are not deserialized in E0.
- Existing top-1 JSON can support a later full-detector mouth, but never the oracle semantic mouth.

## Terminal state

`P0_BLOCKED_NO_FULL_LOGIT_EXPORT` means that top-1 JSON or legacy object-array NPZ artifacts cannot support the oracle mouth and cannot decide G2.

`E0_CONTRACT_READY` means that a full-score JSONL fixture passed schema validation and can be written to a no-replace receipt. It is not evidence that a live model hook or a scientific phenomenon has been verified.
