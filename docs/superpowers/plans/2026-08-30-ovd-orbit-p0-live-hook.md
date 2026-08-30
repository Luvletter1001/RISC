# OVD Object-Orbit P0 Live-Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off, in-memory OpenRSD head sink that captures complete calibrated pre-filter score fields and aligned decoded boxes with exact source identity, without any model run or output mutation.

**Architecture:** `OvdOrbitFullLogitSink` owns active `(scene_id,view_id)` metadata and immutable `ScoreCarrier` records. The existing dense head owns decoding and calls the sink after score calibration but before filtering. The head never writes files; existing E0 CLI owns receipt serialization.

**Tech Stack:** Python 3.10, PyTorch 1.12, existing MMRotate `OpenRotatedRTMDetSepBNHead`, pytest, E0 `ovd_orbit_p0` modules.

---

### Task 1: Add the in-memory full-logit sink with TDD

**Files:**
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0.py`
- Modify: `framework/openrsd/M_Tools/analysis/ovd_orbit_p0_export.py`

- [x] **Step 1: Write failing sink tests**

Add the following test before implementation:

```python
from M_Tools.analysis.ovd_orbit_p0_export import OvdOrbitFullLogitSink


def test_sink_requires_active_metadata_and_preserves_full_scores():
    sink = OvdOrbitFullLogitSink()
    with pytest.raises(OrbitP0Error, match='active'):
        sink.record_level(
            level=0,
            boxes=torch.tensor([[1., 2., 3., 4., .1]]),
            scores=torch.tensor([[.2, .3, .5]]),
        )

    sink.begin_image(scene_id='scene_a', view_id='rot090')
    sink.record_level(
        level=2,
        boxes=torch.tensor([[1., 2., 3., 4., .1]]),
        scores=torch.tensor([[.2, .3, .5]]),
    )
    sink.end_image()

    records = sink.snapshot()
    assert [record.source for record in records] == [(2, 0)]
    np.testing.assert_allclose(records[0].scores, [.2, .3, .5])
```

Also add a test that `begin_image` rejects missing metadata, nesting, and a second `end_image` call.

- [x] **Step 2: Run RED**

From `framework/openrsd` run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0.py -q
```

Expected: import failure because `OvdOrbitFullLogitSink` does not exist.

- [x] **Step 3: Implement the minimal sink**

Add this public API:

```python
class OvdOrbitFullLogitSink:
    def __init__(self) -> None:
        self._active: tuple[str, str] | None = None
        self._records: list[ScoreCarrier] = []

    def begin_image(self, *, scene_id: str, view_id: str) -> None: ...
    def record_level(self, *, level: int, boxes: Any, scores: Any) -> None: ...
    def end_image(self) -> None: ...
    def snapshot(self) -> tuple[ScoreCarrier, ...]: ...
```

`record_level` must delegate to `normalize_level_outputs`; `snapshot` must call `collapse_carriers`. It must not serialize, apply a score transform, mutate tensors, or create output paths.

- [x] **Step 4: Run GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 2: Add opt-in head attachment and metadata validation with TDD

**Files:**
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0.py`
- Modify: `framework/openrsd/M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`

- [x] **Step 1: Write failing attachment tests**

Use `object.__new__(OpenRotatedRTMDetSepBNHead)` plus `torch.nn.Module.__init__(head)` to require:

```python
head.set_ovd_orbit_p0_sink(None)
assert head.ovd_orbit_p0_sink is None

with pytest.raises(TypeError, match='record_level'):
    head.set_ovd_orbit_p0_sink(object())

sink = OvdOrbitFullLogitSink()
head.set_ovd_orbit_p0_sink(sink)
assert head.ovd_orbit_p0_sink is sink
```

Also require `_ovd_orbit_p0_metadata({})` to fail with `OrbitP0Error`, while a metadata dict with nonempty `ovd_orbit_p0_scene_id` and `ovd_orbit_p0_view_id='rot090'` returns the two strings.

- [x] **Step 2: Run RED**

Run the Task 1 pytest command. Expected: missing head method failure.

- [x] **Step 3: Implement default-off attachment**

In the head constructor set `self.ovd_orbit_p0_sink = None`. Add:

```python
def set_ovd_orbit_p0_sink(self, sink) -> None:
    if sink is not None and not callable(getattr(sink, 'record_level', None)):
        raise TypeError('ovd_orbit_p0 sink must expose record_level')
    self.ovd_orbit_p0_sink = sink

def _ovd_orbit_p0_metadata(self, img_meta):
    ...
```

The metadata helper validates strings only when a sink is attached. Do not add any config field or instantiate a sink from config.

- [x] **Step 4: Run GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 3: Capture decoded pre-filter rows with TDD

**Files:**
- Modify: `framework/openrsd/tests/test_ovd_orbit_p0.py`
- Modify: `framework/openrsd/M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`

- [x] **Step 1: Write the failing capture-helper test**

Create fake `angle_coder.decode` and `bbox_coder.decode` objects. Attach a sink to a minimal head and call:

```python
head._record_ovd_orbit_p0_level(
    level_idx=1,
    scores=torch.tensor([[.2, .8]], dtype=torch.float32),
    bbox_pred=torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32),
    angle_pred=torch.tensor([[.1]], dtype=torch.float32),
    priors=torch.tensor([[10., 20.]], dtype=torch.float32),
    img_shape=(32, 32),
)
```

Require the sink record to have source `(1,0)`, the full two-score vector, and the fake decoded five-value box. Clone `scores` before the call and assert the source tensor remains equal after capture.

- [x] **Step 2: Run RED**

Run the Task 1 pytest command. Expected: missing capture-helper failure.

- [x] **Step 3: Implement the capture helper and wire the prediction loop**

Implement `_record_ovd_orbit_p0_level` to return immediately for a `None` sink. With a sink, decode all rows before filtering:

```python
decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
full_bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)
decoded_boxes = self.bbox_coder.decode(priors, full_bbox_pred, max_shape=img_shape)
self.ovd_orbit_p0_sink.record_level(
    level=level_idx, boxes=decoded_boxes, scores=scores)
```

In `predict_by_feat`, call `sink.begin_image` before `_predict_by_feat_single`,
commit with `sink.end_image()` only after successful prediction, and abort on
every exception. In `_predict_by_feat_single`, call the helper immediately
after the existing semantic calibration and before `filter_scores_and_topk`.

- [x] **Step 4: Run GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Safety-correction addendum: preserve C4 identity and fail closed on partial images

This addendum supersedes the accidentally introduced `{id, rot090, rot180,
rot270}` view set. It is required before Task 3 can be accepted.

- [x] Restore the public `C4_VIEW_IDS = {rot000, rot090, rot180, rot270}`
  contract and accept `rot000_a` / `rot000_b` as explicit P0 identity-repeat
  carriers without collapsing them. Add legacy-import and legacy-`rot000`
  regression tests plus repeat-distinctness tests.
- [x] Add `abort_image()` to the sink as a rollback transaction. Attachment
  validation must require it. In the head, commit with `end_image()` only on a
  successful per-image prediction; on every exception call `abort_image()` and
  re-raise. Add a test that records a level then raises and proves `snapshot()`
  has no partial records.
- [x] Add a CPU fake-head test which calls the real `_predict_by_feat_single`
  path with no sink and verifies the capture helper neither decodes nor changes
  its return tensors. Run the focused suite, `py_compile`, and `git diff --check`.

### Scientific-semantics correction addendum: export native logits without losing calibrated diagnostics

This addendum supersedes the prior statement that a calibrated score field by
itself is a P0 logit export. It is required before `E0_LIVE_HOOK_READY_NO_FORWARD`
may be claimed.

- [x] Preserve a full raw native foreground-logit vector for every source row,
  captured before sigmoid/softmax and before all semantic calibration. Keep the
  existing post-calibration pre-filter score vector as a separately named
  optional diagnostic field. The legacy `ScoreCarrier.scores`/JSONL `scores`
  field is the native-logit field for E0 schema compatibility; add explicit
  `calibrated_scores` support and compare it in duplicate-conflict checks.
- [x] In the softmax branch remove only the background column from the raw
  logits so it aligns exactly with foreground calibrated-score columns. Add
  fake-head tests proving raw logits and calibrated values differ as intended,
  complete class dimensions are retained, and neither field is transformed by
  the sink.
- [x] Put `begin_image`, prediction, and `end_image` within one guarded image
  transaction. If `begin_image` mutates state then raises, attempt `abort_image`
  and preserve the original error. Add the corresponding regression test.
- [x] Update README/protocol/progress to call the hook ready only after this
  corrected dual-field contract passes; otherwise retain an honest blocked
  status. Run the focused suite, compile, and `git diff --check`.

### Task 4: Document and verify the no-GPU hook seam

**Files:**
- Modify: `docs/research/ovd_orbit_p0/README.md`
- Modify: `docs/research/ovd_orbit_p0/p0_protocol.md`

- [x] **Step 1: Record the hook status**

Add `E0_LIVE_HOOK_READY_NO_FORWARD` to the status list. Define it as: fake-head contract tests pass; no real model forward, checkpoint load, dataset iteration, GPU allocation, AP computation, or receipt publication was performed.

- [x] **Step 2: Record sink activation metadata**

Document the exact metadata keys `ovd_orbit_p0_scene_id` and `ovd_orbit_p0_view_id`, and state that omitted keys are errors only when a sink is attached.

- [x] **Step 3: Run final verification**

From `framework/openrsd` run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest -p no:cacheprovider tests/test_ovd_orbit_p0.py -q
rtk env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_pyc /data/zcy/anaconda3/envs/openrsd/bin/python -m py_compile M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py M_Tools/analysis/ovd_orbit_p0.py M_Tools/analysis/ovd_orbit_p0_export.py M_Tools/analysis/run_ovd_orbit_p0.py
```

From the worktree root run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: all focused tests pass, compilation exits 0, and only live-hook P0 files are modified.

## Scope exclusions

- Do not call `model.test_step`, load a checkpoint, or create a dataloader.
- Do not add an experiment config or enable the sink from YAML.
- Do not write a live receipt or evaluate the P0 metrics.
- Do not modify CastDet or register an LDM task.
