# DOTA-v2 Q600 Canonical Dump Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, CPU-only analyzer that exactly replays the DOTA-v2 Q600 evaluator, decomposes GT and prediction failures, and atomically publishes the eight-file E24 diagnostic bundle.

**Architecture:** Put pure typed calculations and deterministic serializers in `dotav2_q600_diagnostics.py`; keep argument parsing, read-only loading, canonical gates, and atomic publication in `analyze_dotav2_q600_dump.py`. Reuse the existing CPU pickle validator, MMRotate rotated-IoU implementation, and MMDetection VOC07 AP implementation so evaluator parity is tested rather than approximated.

**Tech Stack:** Python 3.8, dataclasses, NumPy, PyTorch CPU, MMEngine `Config`, MMCV `box_iou_rotated`, MMDetection `average_precision`, pytest, JSON/CSV/Markdown, SHA256.

---

## Frozen implementation contract

Implement against the approved design at
`docs/superpowers/specs/2026-07-22-dotav2-q600-canonical-dump-analyzer-design.md`.
The implementation must preserve these invariants throughout all tasks:

- canonical input is 13,833 raw DOTA-v2 records, 600 immutable query rows per
  image, 18 config-ordered classes, and 8,299,800 total rows;
- evaluation uses rotated IoU 0.5, evaluator `np.argsort(-scores)` ordering,
  greedy same-class matching, and VOC07 11-point AP;
- no row filtering, NMS, top-k, max-per-image truncation, model construction,
  model forward, CUDA use, or training mutation;
- P0148/P0682 are optional cross-dataset case evidence and never enter the
  canonical record count, DOTA-v2 mAP, or base/novel aggregates;
- JSON contains finite numbers or `null`, never NaN/Infinity; CSV ordering is
  fixed; publication is atomic and refuses overwrite;
- expected data/protocol failures use `DiagnosticError` and exit code 2.

Use this selected interpreter for every test and analyzer command:

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python
```

## Shared core types and public functions

Create these exact public records in Task 1 and keep their field names stable
for all subsequent tasks:

```python
@dataclass(frozen=True)
class MetricRecord:
    mAP: float
    ap50: float
    source: str
    step: Optional[int]


@dataclass(frozen=True)
class ClassProtocol:
    classes: Tuple[str, ...]
    base_classes: Tuple[str, ...]
    novel_classes: Tuple[str, ...]
    canonical: bool


@dataclass
class PreparedRecord:
    img_id: str
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    gt_boxes: np.ndarray
    gt_labels: np.ndarray
    ignored_boxes: np.ndarray
    ignored_labels: np.ndarray


@dataclass
class ClassEvaluation:
    class_id: int
    num_gts: int
    image_ids: np.ndarray
    query_ids: np.ndarray
    scores: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    assigned_iou: np.ndarray
    outcomes: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    ap: float
    last_tp_end: int
    ap_support_end: int
    oracle_ap: float


@dataclass
class EvaluationBundle:
    records: Tuple[PreparedRecord, ...]
    class_results: Tuple[ClassEvaluation, ...]
    row_outcomes: Tuple[np.ndarray, ...]
    row_tp: Tuple[np.ndarray, ...]
    row_fp: Tuple[np.ndarray, ...]
    row_same_label_iou: Tuple[np.ndarray, ...]
    row_assigned_iou: Tuple[np.ndarray, ...]
    row_matched_gt: Tuple[np.ndarray, ...]
    mean_ap: float


@dataclass(frozen=True)
class GtEvidence:
    img_id: str
    gt_index: int
    class_id: int
    state: str
    image_gt_count: int
    gt_box: Tuple[float, float, float, float, float]
    best_any_query: Optional[int]
    best_any_iou: float
    best_same_query: Optional[int]
    best_same_iou: float
    candidate_count: int
    witness_query: Optional[int]
    witness_score: Optional[float]
    witness_iou: Optional[float]
    prior_center_distance: Optional[float] = None
    prior_scale_change: Optional[float] = None
    prior_aspect_change: Optional[float] = None
    prior_angle_change: Optional[float] = None
```

The core module's stable entry points are:

```python
load_class_protocol
read_metric_record
extract_reference_points
validate_metric_parity
prepare_records
evaluate_records
voc07_support_ranks
perfect_ranking_ap
decompose_ground_truth
summarize_capacity
stable_sample_keys
safe_spearman
summarize_calibration
summarize_queries
attach_refinement
summarize_strata
load_case_manifest
summarize_cases
build_diagnostics
render_payloads
```

The CLI module exposes `build_parser()` and `main(argv=None)`.

## Task 1: Lock protocol, metric, checkpoint, and record preparation gates

**Files:**

- Create: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`
- Create: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`
- Reuse: `projects/OVCapFlow/tools/validate_dotav2_q600_dump.py`
- Reference: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_base.py`

- [ ] **Step 1: Add a path-based module loader and failing protocol tests**

```python
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[3]
CORE_PATH = (ROOT / 'projects' / 'OVCapFlow' / 'tools' /
             'dotav2_q600_diagnostics.py')


def _load_core():
    spec = importlib.util.spec_from_file_location(
        'dotav2_q600_diagnostics', CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_and_metric_selection_are_exact(tmp_path):
    core = _load_core()
    config = tmp_path / 'config.py'
    config.write_text(
        "classes=('base-a','novel-b')\n"
        "base_classes=('base-a',)\n"
        "novel_classes=('novel-b',)\n")
    metrics = tmp_path / 'metrics.json'
    metrics.write_text(
        json.dumps({'step': 23, 'dota/mAP': .4, 'dota/AP50': .4}) + '\n' +
        json.dumps({'step': 24, 'dota/mAP': .5123456789,
                    'dota/AP50': .512}) + '\n')

    protocol = core.load_class_protocol(config, canonical=False)
    selected = core.read_metric_record(metrics, step=24)

    assert protocol.classes == ('base-a', 'novel-b')
    assert protocol.base_classes == ('base-a',)
    assert protocol.novel_classes == ('novel-b',)
    assert selected.mAP == .5123456789
    assert selected.ap50 == .512
    assert selected.step == 24


def test_metric_directory_requires_one_applicable_row(tmp_path):
    core = _load_core()
    first = tmp_path / 'a.json'
    second = tmp_path / 'b.json'
    row = {'dota/mAP': .5, 'dota/AP50': .5}
    first.write_text(json.dumps(row) + '\n')
    second.write_text(json.dumps(row) + '\n')

    with pytest.raises(core.DiagnosticError, match='exactly one'):
        core.read_metric_record(tmp_path)
```

- [ ] **Step 2: Run the focused tests and confirm the import fails**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: `FileNotFoundError` for
`projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`.

- [ ] **Step 3: Implement the exception, dataclasses, config reader, and exact JSON-line selector**

Start the core module with these imports, the records in “Shared core types,”
and the exception:

```python
import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.evaluation.functional import average_precision
from mmengine import Config


class DiagnosticError(RuntimeError):
    pass


def load_class_protocol(config_path: Path, canonical: bool) -> ClassProtocol:
    cfg = Config.fromfile(str(config_path))
    classes = tuple(cfg.get('classes', ()))
    base = tuple(cfg.get('base_classes', ()))
    novel = tuple(cfg.get('novel_classes', ()))
    if not classes or len(set(classes)) != len(classes):
        raise DiagnosticError('classes must be present and unique')
    if set(base) & set(novel):
        raise DiagnosticError('base and novel classes overlap')
    if set(base) | set(novel) != set(classes):
        raise DiagnosticError('base and novel classes must partition classes')
    if canonical and (len(classes), len(base), len(novel)) != (18, 14, 4):
        raise DiagnosticError('canonical protocol must be 18/14/4')
    return ClassProtocol(classes, base, novel, canonical)


def read_metric_record(path: Path,
                       step: Optional[int] = None) -> MetricRecord:
    files = sorted(path.rglob('*.json')) if path.is_dir() else [path]
    matches = []
    for file_path in files:
        for line in file_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if 'dota/mAP' not in row or 'dota/AP50' not in row:
                continue
            if step is not None and row.get('step') != step:
                continue
            matches.append((file_path, row))
    if len(matches) != 1:
        raise DiagnosticError('metric source must contain exactly one applicable row')
    source, row = matches[0]
    return MetricRecord(float(row['dota/mAP']), float(row['dota/AP50']),
                        str(source), row.get('step'))
```

The production module must also import `Config`, `box_iou_rotated`, and
`average_precision` at module scope so the selected environment is exercised
immediately by tests.

- [ ] **Step 4: Add failing checkpoint extraction and record-integrity tests**

```python
def test_reference_suffix_extraction_and_ambiguity():
    core = _load_core()
    weight = torch.zeros((2, 5))
    checkpoint = {
        'state_dict': {
            'model.query_initializer.reference_embedding.weight': weight,
        }
    }
    refs = core.extract_reference_points(
        checkpoint, expected_queries=2, canonical=False)
    assert refs.shape == (2, 5)
    assert np.allclose(refs, .5)

    checkpoint['state_dict'][
        'ema.query_initializer.reference_embedding.weight'] = weight
    with pytest.raises(core.DiagnosticError, match='exactly one'):
        core.extract_reference_points(
            checkpoint, expected_queries=2, canonical=False)
    with pytest.raises(core.DiagnosticError, match='exactly one'):
        core.extract_reference_points(
            {'state_dict': {}}, expected_queries=2, canonical=False)


def test_metric_parity_rejects_exact_rounded_and_training_mismatches():
    core = _load_core()
    official = core.MetricRecord(.5, .5, 'official.json', None)
    with pytest.raises(core.DiagnosticError, match='same-dump'):
        core.validate_metric_parity(.500001, official, None)
    rounded_wrong = core.MetricRecord(.5, .499, 'official.json', None)
    with pytest.raises(core.DiagnosticError, match='same-dump'):
        core.validate_metric_parity(.5, rounded_wrong, None)
    training = core.MetricRecord(.501, .501, 'training.json', 24)
    with pytest.raises(core.DiagnosticError, match='training replay'):
        core.validate_metric_parity(.5, official, training)


def _record(img_id, pred_boxes, scores, labels, gt_boxes=(), gt_labels=(),
            ignored_boxes=(), ignored_labels=()):
    return {
        'img_id': img_id,
        'pred_instances': {
            'bboxes': torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 5),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
        },
        'gt_instances': {
            'bboxes': torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 5),
            'labels': torch.tensor(gt_labels, dtype=torch.long),
        },
        'ignored_instances': {
            'bboxes': torch.tensor(ignored_boxes, dtype=torch.float32).reshape(-1, 5),
            'labels': torch.tensor(ignored_labels, dtype=torch.long),
        },
    }


def test_prepare_records_preserves_query_rows_and_rejects_bad_gt():
    core = _load_core()
    records = [_record('x', [[10, 10, 4, 4, 0], [20, 20, 4, 4, 0]],
                       [.9, .1], [0, 1], [[10, 10, 4, 4, 0]], [0])]
    prepared = core.prepare_records(records, queries_per_image=2,
                                    num_classes=2)
    assert prepared[0].img_id == 'x'
    assert prepared[0].boxes.shape == (2, 5)

    records[0]['gt_instances']['bboxes'][0, 2] = 0
    with pytest.raises(core.DiagnosticError, match='positive'):
        core.prepare_records(records, queries_per_image=2, num_classes=2)
```

- [ ] **Step 5: Implement checkpoint extraction, parity checks, and NumPy preparation**

```python
REFERENCE_SUFFIX = 'query_initializer.reference_embedding.weight'


def extract_reference_points(checkpoint: Mapping[str, Any],
                             expected_queries: int,
                             canonical: bool) -> np.ndarray:
    state = checkpoint.get('state_dict', checkpoint)
    matches = [value for key, value in state.items()
               if key.endswith(REFERENCE_SUFFIX)]
    if len(matches) != 1:
        raise DiagnosticError('checkpoint must contain exactly one reference key')
    weight = torch.as_tensor(matches[0]).detach().cpu()
    expected = (600, 5) if canonical else (expected_queries, 5)
    if tuple(weight.shape) != expected:
        raise DiagnosticError(f'reference shape must be {expected}')
    return weight.sigmoid().numpy().astype(np.float64, copy=False)


def validate_metric_parity(reconstructed: float,
                           official: MetricRecord,
                           training: Optional[MetricRecord]) -> Dict[str, Any]:
    exact_error = abs(reconstructed - official.mAP)
    if exact_error > 1e-7 or round(reconstructed, 3) != official.ap50:
        raise DiagnosticError('same-dump evaluator parity failed')
    result = {'exact_error': exact_error, 'replay_delta_warning': False}
    if training is not None:
        delta = abs(official.mAP - training.mAP)
        if delta > 5e-4 or official.ap50 != training.ap50:
            raise DiagnosticError('training replay parity failed')
        result.update(training_delta=delta,
                      replay_delta_warning=delta > 1e-7)
    return result
```

`prepare_records` must call the existing validator's `as_tensor` semantics,
convert every box/score/label tensor to contiguous CPU NumPy arrays, normalize
missing `ignored_instances` to empty arrays, and reject duplicate IDs,
nonfinite data, noninteger/out-of-range labels, nonpositive widths/heights,
or a query count different from `queries_per_image`.

- [ ] **Step 6: Run Task 1 tests**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 7: Commit the protocol foundation**

```bash
rtk git add projects/OVCapFlow/tools/dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "feat: add Q600 diagnostic protocol gates"
```

## Task 2: Reconstruct the exact evaluator and preserve immutable query IDs

**Files:**

- Modify: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`
- Reference: `mmrotate/evaluation/functional/mean_ap.py`
- Reference: `mmrotate/evaluation/metrics/dota_metric.py`

- [ ] **Step 1: Add the failing evaluator fixture covering all six outcomes**

```python
def test_exact_evaluator_preserves_query_ids_and_classifies_all_outcomes():
    core = _load_core()
    square = [10, 10, 4, 4, 0]
    far = [40, 40, 4, 4, 0]
    records = [
        _record('filled',
                [square, square, far, square],
                [.9, .8, .7, .6], [0, 0, 0, 1],
                [square], [0]),
        _record('empty', [square], [.95], [0]),
        _record('ignored', [square], [.99], [0],
                ignored_boxes=[square], ignored_labels=[0]),
    ]
    prepared = core.prepare_records_variable_queries(records, num_classes=2)
    result = core.evaluate_records(prepared, num_classes=2, iou_threshold=.5)

    assert result.mean_ap == pytest.approx(.5)
    assert result.class_results[0].query_ids.tolist() == [0, 0, 0, 1, 2]
    assert [outcome for rows in result.row_outcomes for outcome in rows] == [
        'tp', 'duplicate_fp', 'localization_background_fp', 'semantic_fp',
        'empty_tile_fp', 'ignored_prediction']
    assert sum(row.sum() for row in result.row_tp) == 1
    assert sum(row.sum() for row in result.row_fp) == 4
```

For this noncanonical unit fixture only,
`prepare_records_variable_queries(records, num_classes)` is a test helper in
the core module that applies the same integrity checks without asserting an
equal query count. It must not be called by the CLI.

- [ ] **Step 2: Run the test and confirm evaluator symbols are missing**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py::test_exact_evaluator_preserves_query_ids_and_classifies_all_outcomes -q
```

Expected: failure because `evaluate_records` is not defined.

- [ ] **Step 3: Implement per-image official assignment**

Add a private helper that returns rows in original query order:

```python
def _evaluate_image_class(record: PreparedRecord, class_id: int,
                          iou_threshold: float) -> Dict[str, Any]:
    query_ids = np.flatnonzero(record.labels == class_id)
    boxes = record.boxes[query_ids]
    scores = record.scores[query_ids]
    same_gt_ids = np.flatnonzero(record.gt_labels == class_id)
    ignored_ids = np.flatnonzero(record.ignored_labels == class_id)
    combined = np.concatenate((record.gt_boxes[same_gt_ids],
                               record.ignored_boxes[ignored_ids]), axis=0)
    ious = _rotated_iou(boxes, combined)
    order = np.argsort(-scores)
    covered = np.zeros(len(same_gt_ids), dtype=bool)
    tp = np.zeros(len(query_ids), dtype=np.float64)
    fp = np.zeros(len(query_ids), dtype=np.float64)
    outcomes = np.full(len(query_ids), '', dtype=object)
    assigned_iou = np.zeros(len(query_ids), dtype=np.float64)
    matched_gt = np.full(len(query_ids), -1, dtype=np.int64)
    same_label_iou = (ious[:, :len(same_gt_ids)].max(axis=1)
                      if len(same_gt_ids)
                      else np.zeros(len(query_ids), dtype=np.float64))
    all_gt_ious = _rotated_iou(boxes, record.gt_boxes)
    for local_id in order:
        if combined.shape[0]:
            assignment = int(np.argmax(ious[local_id]))
            overlap = float(ious[local_id, assignment])
        else:
            assignment = -1
            overlap = 0.0
        assigned_iou[local_id] = overlap
        if overlap >= iou_threshold and assignment >= len(same_gt_ids):
            outcomes[local_id] = 'ignored_prediction'
            continue
        if overlap >= iou_threshold:
            original_gt_id = int(same_gt_ids[assignment])
            matched_gt[local_id] = original_gt_id
            if covered[assignment]:
                fp[local_id] = 1.0
                outcomes[local_id] = 'duplicate_fp'
            else:
                covered[assignment] = True
                tp[local_id] = 1.0
                outcomes[local_id] = 'tp'
            continue
        fp[local_id] = 1.0
        if len(record.gt_boxes) == 0:
            outcomes[local_id] = 'empty_tile_fp'
            continue
        qualifying = np.flatnonzero(all_gt_ious[local_id] >= iou_threshold)
        other_class = any(record.gt_labels[gt_id] != class_id
                          for gt_id in qualifying)
        outcomes[local_id] = ('semantic_fp' if other_class else
                              'localization_background_fp')
    return dict(query_ids=query_ids, scores=scores, tp=tp, fp=fp,
                outcomes=outcomes, assigned_iou=assigned_iou,
                same_label_iou=same_label_iou, matched_gt=matched_gt,
                num_gts=len(same_gt_ids))
```

Use:

```python
def _rotated_iou(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if len(lhs) == 0 or len(rhs) == 0:
        return np.zeros((len(lhs), len(rhs)), dtype=np.float64)
    value = box_iou_rotated(torch.from_numpy(lhs).float(),
                            torch.from_numpy(rhs).float())
    return value.detach().cpu().numpy().astype(np.float64, copy=False)
```

Do not infer `query_id` after class sorting: it is always the row position
created by `np.flatnonzero(record.labels == class_id)`.

- [ ] **Step 4: Implement global class sorting and VOC07 AP**

```python
def evaluate_records(records: Sequence[PreparedRecord], num_classes: int,
                     iou_threshold: float) -> EvaluationBundle:
    row_outcomes = tuple(np.full(len(record.scores), '', dtype=object)
                         for record in records)
    row_tp = tuple(np.zeros(len(record.scores), dtype=np.float64)
                   for record in records)
    row_fp = tuple(np.zeros(len(record.scores), dtype=np.float64)
                   for record in records)
    row_same_iou = tuple(np.zeros(len(record.scores), dtype=np.float64)
                         for record in records)
    row_assigned_iou = tuple(np.zeros(len(record.scores), dtype=np.float64)
                             for record in records)
    row_matched_gt = tuple(np.full(len(record.scores), -1, dtype=np.int64)
                           for record in records)
    class_results = []
    for class_id in range(num_classes):
        pieces = []
        for record_id, record in enumerate(records):
            piece = _evaluate_image_class(record, class_id, iou_threshold)
            piece['record_id'] = np.full(len(piece['query_ids']), record_id,
                                         dtype=np.int64)
            piece['image_ids'] = np.full(len(piece['query_ids']), record.img_id,
                                         dtype=object)
            pieces.append(piece)
            query_ids = piece['query_ids']
            row_outcomes[record_id][query_ids] = piece['outcomes']
            row_tp[record_id][query_ids] = piece['tp']
            row_fp[record_id][query_ids] = piece['fp']
            row_same_iou[record_id][query_ids] = piece['same_label_iou']
            row_assigned_iou[record_id][query_ids] = piece['assigned_iou']
            row_matched_gt[record_id][query_ids] = piece['matched_gt']
        num_gts = sum(piece['num_gts'] for piece in pieces)
        scores = np.concatenate([piece['scores'] for piece in pieces])
        order = np.argsort(-scores)
        tp = np.concatenate([piece['tp'] for piece in pieces])[order]
        fp = np.concatenate([piece['fp'] for piece in pieces])[order]
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recall = cumulative_tp / max(num_gts, np.finfo(np.float64).eps)
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)
        ap = _voc07_ap(recall, precision)
        class_results.append(ClassEvaluation(
            class_id=class_id,
            num_gts=num_gts,
            image_ids=np.concatenate(
                [piece['image_ids'] for piece in pieces])[order],
            query_ids=np.concatenate(
                [piece['query_ids'] for piece in pieces])[order],
            scores=scores[order],
            tp=tp,
            fp=fp,
            assigned_iou=np.concatenate(
                [piece['assigned_iou'] for piece in pieces])[order],
            outcomes=np.concatenate(
                [piece['outcomes'] for piece in pieces])[order],
            precision=precision,
            recall=recall,
            ap=ap,
            last_tp_end=(int(np.flatnonzero(tp)[-1]) + 1 if tp.any() else 0),
            ap_support_end=0,
            oracle_ap=ap))
    valid_aps = [result.ap for result in class_results if result.num_gts > 0]
    if not valid_aps:
        raise DiagnosticError('evaluation contains no non-ignored GT')
    for outcomes, tp, fp in zip(row_outcomes, row_tp, row_fp):
        if np.any(outcomes == ''):
            raise DiagnosticError('every prediction row needs one outcome')
        ignored = outcomes == 'ignored_prediction'
        if not np.all(tp + fp + ignored.astype(np.float64) == 1):
            raise DiagnosticError('prediction assignment accounting failed')
    return EvaluationBundle(
        records=tuple(records), class_results=tuple(class_results),
        row_outcomes=row_outcomes, row_tp=row_tp, row_fp=row_fp,
        row_same_label_iou=row_same_iou,
        row_assigned_iou=row_assigned_iou,
        row_matched_gt=row_matched_gt,
        mean_ap=float(np.mean(valid_aps)))


def _voc07_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    return float(average_precision(recall, precision, mode='11points'))
```

For every record, allocate fixed-length row arrays before class splitting and
scatter official assignments back using immutable query IDs. Assert exactly
one nonempty outcome per row, `sum(tp) + sum(fp) + ignored_count == row_count`,
and consistent per-class/global GT counts before returning the bundle.

- [ ] **Step 5: Add a direct parity test against `average_precision`**

```python
def test_voc07_ap_matches_mmdetection_implementation():
    core = _load_core()
    recall = np.array([0., .5, .5, 1.], dtype=np.float64)
    precision = np.array([0., .5, 1 / 3, .5], dtype=np.float64)
    expected = core.average_precision(recall, precision, mode='11points')
    assert core._voc07_ap(recall, precision) == pytest.approx(expected)
```

- [ ] **Step 6: Run the complete focused test file**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: all tests pass, including exact class-0 AP `0.5` and all six row
outcomes.

- [ ] **Step 7: Commit evaluator reconstruction**

```bash
rtk git add projects/OVCapFlow/tools/dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "feat: reconstruct Q600 VOC07 evaluation"
```

## Task 3: Add AP-sensitive regions, ranking oracle, GT states, and Q-capacity

**Files:**

- Modify: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`

- [ ] **Step 1: Add failing AP-support and fixed-recall oracle tests**

```python
def test_voc07_support_ranks_choose_earliest_precision_tie():
    core = _load_core()
    recall = np.array([0., .5, .5, 1., 1.])
    precision = np.array([0., .6, .6, .4, .4])
    ranks = core.voc07_support_ranks(recall, precision)
    assert ranks[:6] == [1, 1, 1, 1, 1, 1]
    assert ranks[6:] == [3, 3, 3, 3, 3]
    assert max(ranks) + 1 == 4


def test_perfect_ranking_oracle_keeps_maximum_recall():
    core = _load_core()
    tp = np.array([0., 1., 0., 0.])
    fp = 1. - tp
    oracle = core.perfect_ranking_ap(tp, fp, num_gts=2)
    assert oracle == pytest.approx(6 / 11)
```

- [ ] **Step 2: Run the two tests and confirm the helpers are absent**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -k "support_ranks or perfect_ranking" -q
```

Expected: two `AttributeError` failures.

- [ ] **Step 3: Implement support ranks, endpoints, and oracle AP**

```python
def voc07_support_ranks(recall: np.ndarray,
                       precision: np.ndarray) -> List[Optional[int]]:
    ranks = []
    for threshold in np.arange(0., 1.0001, .1):
        eligible = np.flatnonzero(recall >= threshold)
        if len(eligible) == 0:
            ranks.append(None)
            continue
        best = precision[eligible].max()
        ranks.append(int(eligible[np.flatnonzero(
            precision[eligible] == best)[0]]))
    return ranks


def perfect_ranking_ap(tp: np.ndarray, fp: np.ndarray, num_gts: int) -> float:
    reordered_tp = np.concatenate((tp[tp == 1], tp[tp == 0]))
    reordered_fp = 1. - reordered_tp
    cumulative_tp = np.cumsum(reordered_tp)
    cumulative_fp = np.cumsum(reordered_fp)
    recall = cumulative_tp / max(num_gts, np.finfo(np.float64).eps)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp,
                                            np.finfo(np.float64).eps)
    return float(average_precision(recall, precision, mode='11points'))
```

Set `last_tp_end` to one past the last globally sorted TP, or zero when absent.
Set `ap_support_end` to one past the maximum non-null support rank, or zero.
Store `oracle_ap`; downstream code reports `oracle_ap - ap` and `0.70 - ap`.

- [ ] **Step 4: Add a failing four-state GT fixture and capacity test**

```python
def test_gt_decomposition_has_geometry_semantic_ownership_and_reachable():
    core = _load_core()
    gt = [[10, 10, 4, 4, 0], [30, 30, 4, 4, 0],
          [50, 50, 4, 4, 0], [52, 50, 4, 4, 0]]
    record = _record(
        'states',
        [[90, 90, 4, 4, 0], [30, 30, 4, 4, 0], [51, 50, 6, 4, 0]],
        [.2, .8, .9], [0, 1, 0], gt, [0, 0, 0, 0])
    prepared = core.prepare_records_variable_queries([record], num_classes=2)
    evidence = core.decompose_ground_truth(prepared, iou_threshold=.5)
    assert [row.state for row in evidence] == [
        'geometry_miss', 'semantic_miss',
        'evaluator_reachable', 'ownership_miss']
    assert evidence[2].witness_query == 2
    assert evidence[3].candidate_count == 1


def test_capacity_separates_lower_bound_from_observed_misses():
    core = _load_core()
    summary = core.summarize_capacity(
        image_gt_counts=[1, 5], class_image_gt_counts={0: [1, 4], 1: [0, 1]},
        query_count=2)
    assert summary['total_gt'] == 6
    assert summary['capacity_excess'] == 3
    assert summary['optimistic_recall_ceiling'] == pytest.approx(.5)
```

- [ ] **Step 5: Run the new tests and confirm GT decomposition is absent**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -k "gt_decomposition or capacity" -q
```

Expected: failures for missing `decompose_ground_truth` and
`summarize_capacity`.

- [ ] **Step 6: Implement compact all-label/same-label ownership evidence**

For each image, compute the all-row/all-GT rotated-IoU matrix once. For every
nonignored GT:

```python
best_any_query = int(np.argmax(ious[:, gt_index]))
best_any_iou = float(ious[best_any_query, gt_index])
same_query_ids = np.flatnonzero(record.labels == record.gt_labels[gt_index])
qualifying = same_query_ids[ious[same_query_ids, gt_index] >= iou_threshold]
owners = [query_id for query_id in qualifying
          if int(np.argmax(ious[query_id])) == gt_index]
```

Assign exactly one state in this order:

1. `geometry_miss` when `best_any_iou < threshold`;
2. `semantic_miss` when no same-label row reaches threshold;
3. `ownership_miss` when qualifying rows exist but `owners` is empty;
4. `evaluator_reachable` otherwise, with the highest-score owner as witness.

Store `candidate_count=len(qualifying)` and compute candidate excess as
`sum(max(candidate_count - 1, 0))`. Do not reuse score-greedy TP assignment as
the reachability witness.

Implement capacity as:

```python
capacity_excess = sum(max(count - query_count, 0) for count in image_gt_counts)
optimistic_recall_ceiling = (
    (total_gt - capacity_excess) / total_gt if total_gt else None)
```

Emit the same fields per class using the class-specific per-image GT counts.

- [ ] **Step 7: Run the focused suite**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: all tests pass, including the four mutually exclusive GT states.

- [ ] **Step 8: Commit causal decomposition**

```bash
rtk git add projects/OVCapFlow/tools/dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "feat: decompose Q600 reachability and ranking"
```

## Task 4: Add deterministic calibration, query, refinement, and strata summaries

**Files:**

- Modify: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`

- [ ] **Step 1: Add failing stable-sample, correlation, and bin-boundary tests**

```python
def test_stable_sample_is_order_independent_and_spearman_handles_constants():
    core = _load_core()
    keys = [('b', 1), ('a', 0), ('c', 2), ('d', 3)]
    first = core.stable_sample_keys(keys, sample_size=2, seed=20260722)
    second = core.stable_sample_keys(list(reversed(keys)), sample_size=2,
                                     seed=20260722)
    assert first == second
    assert core.safe_spearman([1, 1], [2, 3]) is None
    assert core.safe_spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.)


@pytest.mark.parametrize('value,expected', [
    (8, '<=8'), (8.0001, '(8,16]'), (16, '(8,16]'),
    (32, '(16,32]'), (64, '(32,64]'), (64.1, '>64')])
def test_size_bin_boundaries(value, expected):
    core = _load_core()
    assert core.size_bin(value) == expected


@pytest.mark.parametrize('count,expected', [
    (1, '1-10'), (10, '1-10'), (11, '11-50'), (50, '11-50'),
    (51, '51-100'), (101, '101-200'), (201, '201-400'),
    (401, '401-600'), (601, '>600')])
def test_density_bin_boundaries(count, expected):
    core = _load_core()
    assert core.density_bin(count) == expected


@pytest.mark.parametrize('ratio,expected', [
    (1.5, '<=1.5'), (1.5001, '(1.5,3]'), (3, '(1.5,3]'),
    (5, '(3,5]'), (5.1, '>5')])
def test_aspect_bin_boundaries(ratio, expected):
    core = _load_core()
    assert core.aspect_bin(ratio) == expected


@pytest.mark.parametrize('degrees,expected', [
    (0, '[0,15)'), (14.999, '[0,15)'), (15, '[15,30)'),
    (45, '[45,60)'), (75, '[75,90]'), (90, '[75,90]')])
def test_angle_bin_boundaries(degrees, expected):
    core = _load_core()
    assert core.angle_bin(np.deg2rad(degrees)) == expected


@pytest.mark.parametrize('score,expected', [
    (0, '[0,.05)'), (.0499, '[0,.05)'), (.05, '[.05,.10)'),
    (.10, '[.10,.25)'), (.25, '[.25,.50)'), (.50, '[.50,.75)'),
    (.75, '[.75,1]'), (1., '[.75,1]')])
def test_score_bin_boundaries(score, expected):
    core = _load_core()
    assert core.score_bin(score) == expected


def test_strata_emit_base_novel_and_small_vehicle_cross_rows():
    core = _load_core()
    protocol = core.ClassProtocol(
        classes=('small-vehicle', 'airport'),
        base_classes=('small-vehicle',),
        novel_classes=('airport',),
        canonical=False)
    evidence = [core.GtEvidence(
        img_id='dense', gt_index=0, class_id=0,
        state='geometry_miss', image_gt_count=11,
        gt_box=(10., 10., 8., 8., 0.),
        best_any_query=0, best_any_iou=.1,
        best_same_query=0, best_same_iou=.1,
        candidate_count=0, witness_query=None,
        witness_score=None, witness_iou=None)]
    rows = core.summarize_strata(
        evidence, protocol=protocol, small_vehicle_class_id=0)
    keyed = {(row['stratum_type'], row['stratum']): row for row in rows}
    assert keyed[('class_group', 'base')]['gt_count'] == 1
    assert keyed[('class_group', 'novel')]['gt_count'] == 0
    assert keyed[('small_vehicle_size_density',
                  '<=8|11-50')]['gt_count'] == 1
    assert keyed[('small_vehicle_size_density',
                  '>64|>600')]['rate'] is None
```

- [ ] **Step 2: Add failing query entropy and refinement-equation tests**

```python
def test_query_effective_count_and_reference_refinement():
    core = _load_core()
    counts = np.array([1., 1., 0.])
    assert core.effective_query_count(counts) == pytest.approx(2.)

    prior = np.array([[.5, .5, .25, .25, .0]])
    final = np.array([[520., 512., 512., 256., np.pi - .1]])
    values = core.refinement_values(prior[0], final[0], patch_size=1024)
    assert values['center_distance'] == pytest.approx(8.)
    assert values['scale_change'] == pytest.approx(np.sqrt(2.))
    assert values['aspect_change'] == pytest.approx(2.)
    assert values['angle_change'] == pytest.approx(.1)
```

- [ ] **Step 3: Run and confirm these diagnostics are absent**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -k "stable_sample or spearman or bin_boundaries or effective_count or refinement" -q
```

Expected: failures for missing sampling/binning/refinement helpers.

- [ ] **Step 4: Implement stable hashing, safe Spearman, and fixed score bins**

```python
SCORE_BINS = ((0., .05), (.05, .10), (.10, .25),
              (.25, .50), (.50, .75), (.75, 1.0000000000000002))


def stable_sample_keys(keys, sample_size, seed):
    ranked = []
    for img_id, query_id in keys:
        payload = f'{seed}\0{img_id}\0{query_id}'.encode('utf-8')
        ranked.append((hashlib.sha256(payload).hexdigest(), img_id, query_id))
    return [(img_id, query_id) for _, img_id, query_id
            in sorted(ranked)[:sample_size]]


def safe_spearman(lhs, rhs):
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if len(lhs) < 2 or np.ptp(lhs) == 0 or np.ptp(rhs) == 0:
        return None
    lhs_rank = _average_ranks(lhs)
    rhs_rank = _average_ranks(rhs)
    return float(np.corrcoef(lhs_rank, rhs_rank)[0, 1])
```

Implement `_average_ranks` with stable mergesort and average tied ranks; do
not add SciPy as a dependency. `summarize_calibration` reports all six score
bins with row count, TP precision, same-label hit@0.5, mean same-label IoU,
and TP assigned-IoU distribution. Empty bins contain count zero and null rates.

- [ ] **Step 5: Implement query allocation and effective-query summaries**

`summarize_queries` must emit exactly one row for every query ID from zero to
`query_count - 1`, including unused queries. Include label counts, TP, each FP
outcome, ignored rows, reachable witnesses, AP-support appearances, score
quantiles, all-row label allocation, and AP-support label allocation.

```python
def effective_query_count(counts: np.ndarray) -> Optional[float]:
    total = float(counts.sum())
    if total == 0:
        return None
    probabilities = counts[counts > 0] / total
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))
```

- [ ] **Step 6: Implement reference conversion and witness refinement**

```python
def refinement_values(prior, final, patch_size=1024):
    prior_box = np.array([prior[0] * patch_size, prior[1] * patch_size,
                          prior[2] * patch_size, prior[3] * patch_size,
                          prior[4] * np.pi], dtype=np.float64)
    center = float(np.linalg.norm(final[:2] - prior_box[:2]))
    prior_area = prior_box[2] * prior_box[3]
    final_area = final[2] * final[3]
    scale = float(np.exp(abs(np.log(np.sqrt(final_area / prior_area)))))
    prior_ar = max(prior_box[2:4]) / min(prior_box[2:4])
    final_ar = max(final[2:4]) / min(final[2:4])
    aspect = float(np.exp(abs(np.log(final_ar / prior_ar))))
    raw_angle = abs(float(final[4] - prior_box[4])) % np.pi
    angle = min(raw_angle, np.pi - raw_angle)
    return {'center_distance': center, 'scale_change': scale,
            'aspect_change': aspect, 'angle_change': angle}
```

`attach_refinement` applies this only to GT rows with a witness query and
returns new frozen `GtEvidence` values via `dataclasses.replace`.

- [ ] **Step 7: Implement every fixed stratum and assert null-preserving rows**

Use these exact ordered labels:

```python
SIZE_BINS = ('<=8', '(8,16]', '(16,32]', '(32,64]', '>64')
DENSITY_BINS = ('1-10', '11-50', '51-100', '101-200',
                '201-400', '401-600', '>600')
ASPECT_BINS = ('<=1.5', '(1.5,3]', '(3,5]', '>5')
ANGLE_BINS = ('[0,15)', '[15,30)', '[30,45)',
              '[45,60)', '[60,75)', '[75,90]')
```

`summarize_strata(evidence, protocol, small_vehicle_class_id)` emits dictionaries
with at least `stratum_type`, `stratum`, `gt_count`, the four GT-state counts,
and `rate`. It covers class, base/novel, size, density, aspect, periodic angle,
and the full Cartesian product of small-vehicle size-by-density labels joined
by `|`. Every predefined bin appears even when its GT count is zero; its rates
are `None`.

- [ ] **Step 8: Run the full focused suite**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: all calibration, query, refinement, and boundary tests pass.

- [ ] **Step 9: Commit quality and stratum diagnostics**

```bash
rtk git add projects/OVCapFlow/tools/dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "feat: add Q600 calibration and strata diagnostics"
```

## Task 5: Isolate optional P0148/P0682 case evidence

**Files:**

- Modify: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`

- [ ] **Step 1: Add failing manifest and not-run tests**

```python
EXPECTED_CASE_MANIFEST = {
    'context_false_sv': ['P0148__1024__651___0'],
    'true_sv_safety': ['P0682__1024__553___0'],
}


def test_case_manifest_is_exact_and_absence_is_not_run(tmp_path):
    core = _load_core()
    manifest_path = tmp_path / 'cases.json'
    manifest_path.write_text(json.dumps(EXPECTED_CASE_MANIFEST))
    assert core.load_case_manifest(manifest_path) == EXPECTED_CASE_MANIFEST
    assert core.summarize_cases(None, None, small_vehicle_class_id=0) == [{
        'case_gate': 'not_run', 'image_id': None, 'case_group': None}]

    manifest_path.write_text(json.dumps({
        'context_false_sv': ['P0148__1024__651___0']}))
    with pytest.raises(core.DiagnosticError, match='exact'):
        core.load_case_manifest(manifest_path)
```

- [ ] **Step 2: Add a failing supplied-dump separation test**

```python
def test_case_dump_requires_each_historical_id_once(tmp_path):
    core = _load_core()
    square = [10, 10, 4, 4, 0]
    case_records = [
        _record('P0148__1024__651___0', [square], [.8], [0]),
        _record('P0682__1024__553___0', [square], [.9], [0], [square], [0]),
    ]
    rows = core.summarize_cases(case_records, EXPECTED_CASE_MANIFEST,
                                small_vehicle_class_id=0)
    assert [row['image_id'] for row in rows] == [
        'P0148__1024__651___0', 'P0682__1024__553___0']
    assert rows[0]['small_vehicle_gt_count'] == 0
    assert rows[1]['small_vehicle_gt_count'] == 1

    with pytest.raises(core.DiagnosticError, match='exactly once'):
        core.summarize_cases(case_records[:1], EXPECTED_CASE_MANIFEST,
                             small_vehicle_class_id=0)
```

- [ ] **Step 3: Run and confirm optional-case functions are absent**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -k "case_manifest or case_dump" -q
```

Expected: failures for missing `load_case_manifest`/`summarize_cases`.

- [ ] **Step 4: Implement exact manifest validation and two-row case schema**

`load_case_manifest` must require exact key/value equality with
`EXPECTED_CASE_MANIFEST`; no extra ID, missing ID, threshold, or expected
outcome is allowed. `summarize_cases` must:

- return the single `case_gate=not_run` row when both optional inputs are
  absent;
- reject supplying only one of dump/manifest;
- require each fixed ID exactly once and reject extra records;
- evaluate only these records with the same evaluator/GT decomposition;
- report GT/SV counts, predicted-SV rows, top-1/top-5/top-20 SV scores, SV
  TP/FP/AP-support counts, every true-SV witness, four GT-state counts, and
  the highest-score false-SV witness with nearest GT label/IoU;
- never return an mAP contribution or mutate the canonical bundle.

- [ ] **Step 5: Run the complete test file**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: all tests pass and the no-case output is explicit.

- [ ] **Step 6: Commit case isolation**

```bash
rtk git add projects/OVCapFlow/tools/dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "feat: add isolated Q600 case diagnostics"
```

## Task 6: Build deterministic diagnostics and the seven hashed payloads

**Files:**

- Modify: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`

- [ ] **Step 1: Add a failing deterministic serialization test**

```python
def test_render_payloads_is_deterministic_finite_and_complete():
    core = _load_core()
    summary = {
        'schema_version': 1,
        'canonical': False,
        'integrity': {'records': 1, 'prediction_rows': 2},
        'parity': {'exact_error': 0.0},
        'metrics': {'map': .5, 'ap50': .5, 'ap70_gap': .2},
        'interpretation': {'tp_score_iou_spearman': None},
    }
    tables = {
        'per_class.csv': [{'class_id': 0, 'ap': .5}],
        'strata.csv': [{'stratum': '<=8', 'gt_count': 0, 'rate': None}],
        'per_query.csv': [{'query_id': 0, 'rows': 1}],
        'case_studies.csv': [{'case_gate': 'not_run'}],
        'error_examples.csv': [],
    }
    first = core.render_payloads(summary, tables)
    second = core.render_payloads(summary, tables)

    assert first == second
    assert set(first) == {
        'diagnostics.json', 'per_class.csv', 'strata.csv', 'per_query.csv',
        'case_studies.csv', 'error_examples.csv', 'report.md'}
    assert b'NaN' not in b''.join(first.values())
    assert json.loads(first['diagnostics.json'])['schema_version'] == 1
```

- [ ] **Step 2: Add a failing full summary and capped-example ordering test**

```python
def test_build_diagnostics_caps_errors_by_type_and_stable_order():
    core = _load_core()
    examples = [
        {'outcome': 'semantic_fp', 'img_id': 'z', 'query_id': 4, 'score': .9},
        {'outcome': 'semantic_fp', 'img_id': 'a', 'query_id': 1, 'score': .9},
        {'outcome': 'semantic_fp', 'img_id': 'b', 'query_id': 2, 'score': .8},
    ]
    kept = core.cap_error_examples(examples, max_per_type=2)
    assert [(row['img_id'], row['query_id']) for row in kept] == [
        ('a', 1), ('z', 4)]
```

- [ ] **Step 3: Run and confirm serializers are absent**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -k "render_payloads or caps_errors" -q
```

Expected: failures for missing report functions.

- [ ] **Step 4: Implement one machine-readable source of truth**

`build_diagnostics` assembles, without file I/O:

```python
def build_diagnostics(canonical, integrity, parity, reconstructed_map,
                      oracle_map, gt_decomposition, capacity, fp_all,
                      fp_last_tp, fp_ap_support, calibration, query_summary,
                      base_summary, novel_summary, case_gate, warnings):
    ap70_gap = .70 - reconstructed_map
    oracle_headroom = oracle_map - reconstructed_map
    fraction = (ap70_gap / oracle_headroom
                if oracle_headroom > 0 else None)
    return {
        'schema_version': 1,
        'canonical': bool(canonical),
        'integrity': integrity,
        'parity': parity,
        'metrics': {
            'map': float(reconstructed_map),
            'ap50': round(float(reconstructed_map), 3),
            'ap70_gap': float(ap70_gap),
            'oracle_map': float(oracle_map),
            'oracle_headroom': float(oracle_headroom),
            'headroom_fraction_to_ap70': (None if fraction is None
                                           else float(fraction)),
        },
        'gt_decomposition': gt_decomposition,
        'capacity': capacity,
        'fp_regions': {
            'all': fp_all,
            'through_last_tp': fp_last_tp,
            'ap_support': fp_ap_support,
        },
        'calibration': calibration,
        'queries': query_summary,
        'groups': {'base': base_summary, 'novel': novel_summary},
        'case_gate': case_gate,
        'warnings': list(warnings),
    }
```

The report's factual decision matrix must be derived from these fields and
must not introduce causal labels, authorize a new run, or state that AP70 was
achieved.

- [ ] **Step 5: Implement canonical JSON, fixed-schema CSV, and Markdown bytes**

```python
def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False,
                       ensure_ascii=False) + '\n').encode('utf-8')


def render_payloads(summary, tables):
    payloads = {'diagnostics.json': _json_bytes(summary)}
    for name, fieldnames in CSV_SCHEMAS.items():
        payloads[name] = _csv_bytes(tables[name], fieldnames)
    payloads['report.md'] = _render_report(summary).encode('utf-8')
    return payloads
```

Define explicit `CSV_SCHEMAS` constants; `csv.DictWriter` must receive those
field orders and `lineterminator='\n'`. Sort class rows by class order,
strata by fixed definition order, query rows by query ID, cases by fixed
manifest order, and errors by outcome, descending score, image ID, query ID.
Reject nonfinite floats recursively before encoding; convert undefined rates
to `None`/empty CSV cells.

- [ ] **Step 6: Implement the manifest builder without recursive self-hash**

```python
def build_manifest(payloads, provenance, command, environment):
    files = {
        name: {'sha256': hashlib.sha256(content).hexdigest(),
               'bytes': len(content)}
        for name, content in sorted(payloads.items())
    }
    return _json_bytes({
        'schema_version': 1,
        'files': files,
        'provenance': provenance,
        'command': command,
        'environment': environment,
    })
```

Assert that the input payload map has exactly the seven non-manifest names.
The caller adds `manifest.json` afterward; the manifest never hashes itself.

- [ ] **Step 7: Run the focused test file twice**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: identical passing counts both times; serializer equality is checked
inside the suite.

- [ ] **Step 8: Commit deterministic reporting**

```bash
rtk git add projects/OVCapFlow/tools/dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "feat: render deterministic Q600 diagnostics"
```

## Task 7: Add the thin CLI, canonical gates, and atomic publication

**Files:**

- Create: `projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`
- Reuse: `projects/OVCapFlow/tools/validate_dotav2_q600_dump.py`

- [ ] **Step 1: Add a CLI loader and failing parser-contract test**

```python
CLI_PATH = (ROOT / 'projects' / 'OVCapFlow' / 'tools' /
            'analyze_dotav2_q600_dump.py')


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        'analyze_dotav2_q600_dump', CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_requires_noncanonical_override_for_small_counts():
    cli = _load_cli()
    parser = cli.build_parser()
    args = parser.parse_args([
        'dump.pkl', '--config', 'config.py', '--checkpoint', 'epoch.pth',
        '--official-metrics-json', 'metrics.json', '--output-dir', 'out',
        '--expected-records', '2', '--queries-per-image', '3',
        '--num-classes', '2'])
    with pytest.raises(cli.DiagnosticError, match='allow-noncanonical'):
        cli.validate_mode(args)


def test_canonical_mode_rejects_visible_cuda(monkeypatch):
    cli = _load_cli()
    parser = cli.build_parser()
    args = parser.parse_args([
        'dump.pkl', '--config', 'config.py', '--checkpoint', 'epoch.pth',
        '--official-metrics-json', 'metrics.json',
        '--training-metrics-json', 'training.json', '--training-step', '24',
        '--output-dir', 'out'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(cli.DiagnosticError, match='CUDA_VISIBLE_DEVICES'):
        cli.validate_mode(args)
```

- [ ] **Step 2: Run and confirm the CLI file is missing**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py::test_cli_requires_noncanonical_override_for_small_counts -q
```

Expected: `FileNotFoundError` for `analyze_dotav2_q600_dump.py`.

- [ ] **Step 3: Implement the exact parser and mode gate**

```python
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--official-metrics-json', type=Path, required=True)
    parser.add_argument('--training-metrics-json', type=Path)
    parser.add_argument('--training-step', type=int, default=24)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iou-threshold', type=float, default=.5)
    parser.add_argument('--sample-size', type=int, default=500000)
    parser.add_argument('--sample-seed', type=int, default=20260722)
    parser.add_argument('--max-error-examples', type=int, default=100)
    parser.add_argument('--case-dump', type=Path)
    parser.add_argument('--case-manifest', type=Path)
    parser.add_argument('--allow-noncanonical', action='store_true')
    parser.add_argument('--expected-records', type=int, default=13833)
    parser.add_argument('--queries-per-image', type=int, default=600)
    parser.add_argument('--num-classes', type=int, default=18)
    return parser
```

`validate_mode` requires canonical defaults, `training_metrics_json`, exactly
step 24, and an unset/empty `CUDA_VISIBLE_DEVICES` unless
`--allow-noncanonical` is present. Any nondefault count or IoU threshold sets
`canonical=false`; it must never be silently called canonical.

- [ ] **Step 4: Add an end-to-end synthetic CLI success test**

```python
def test_cli_publishes_exact_eight_file_bundle(tmp_path, monkeypatch):
    cli = _load_cli()
    square = [10, 10, 4, 4, 0]
    dump = tmp_path / 'dump.pkl'
    checkpoint = tmp_path / 'epoch.pth'
    config = tmp_path / 'config.py'
    metrics = tmp_path / 'metrics.json'
    output = tmp_path / 'diagnostics'
    with dump.open('wb') as stream:
        pickle.dump([_record('x', [square], [.9], [0], [square], [0])],
                    stream)
    torch.save({'state_dict': {
        'query_initializer.reference_embedding.weight': torch.zeros((1, 5))}},
        checkpoint)
    config.write_text(
        "classes=('small-vehicle',)\n"
        "base_classes=('small-vehicle',)\n"
        "novel_classes=()\n")
    metrics.write_text(json.dumps({'dota/mAP': 1., 'dota/AP50': 1.}) + '\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    code = cli.main([
        str(dump), '--config', str(config), '--checkpoint', str(checkpoint),
        '--official-metrics-json', str(metrics), '--output-dir', str(output),
        '--allow-noncanonical', '--expected-records', '1',
        '--queries-per-image', '1', '--num-classes', '1'])

    assert code == 0
    assert sorted(path.name for path in output.iterdir()) == [
        'case_studies.csv', 'diagnostics.json', 'error_examples.csv',
        'manifest.json', 'per_class.csv', 'per_query.csv', 'report.md',
        'strata.csv']
    assert json.loads((output / 'diagnostics.json').read_text())['canonical'] is False
```

- [ ] **Step 5: Add failure/no-overwrite atomic-publication tests**

```python
def test_cli_refuses_overwrite_and_never_publishes_canonical_on_failure(
        tmp_path, monkeypatch):
    cli = _load_cli()
    output = tmp_path / 'diagnostics'
    output.mkdir()
    args = cli.build_parser().parse_args([
        'missing.pkl', '--config', 'missing.py', '--checkpoint', 'missing.pth',
        '--official-metrics-json', 'missing.json', '--output-dir', str(output),
        '--allow-noncanonical'])
    with pytest.raises(cli.DiagnosticError, match='already exists'):
        cli.run(args)
    assert not (output / 'diagnostics.json').exists()


def test_atomic_writer_preserves_failed_directory(tmp_path):
    cli = _load_cli()
    output = tmp_path / 'diagnostics'
    with pytest.raises(RuntimeError, match='boom'):
        with cli.atomic_output_directory(output) as temporary:
            (temporary / 'partial.txt').write_text('evidence')
            raise RuntimeError('boom')
    failed = list(tmp_path.glob('diagnostics.*.failed'))
    assert len(failed) == 1
    assert (failed[0] / 'partial.txt').read_text() == 'evidence'
    assert not output.exists()
```

- [ ] **Step 6: Implement read-only orchestration in the required order**

`run(args)` performs these operations in order:

1. validate mode and reject an existing output path;
2. set PyTorch intra-op/inter-op threads to one;
3. load config and exact metric rows, resolving a metric directory to the one
   selected JSON file;
4. hash and size config, checkpoint, dump, the resolved official metric file,
   optional resolved training metric file, and optional case inputs;
5. load checkpoint on CPU, extract sigmoid references, delete checkpoint;
6. load dump with the existing `CpuUnpickler`/`load_cpu`, call the existing
   `validate_records`, and prepare arrays;
7. evaluate, validate exact/rounded parity, decompose, summarize, and render;
8. write seven payloads and the manifest inside a temporary sibling;
9. fsync files/directories and rename the sibling to the requested output.

Import the two sibling tools without requiring package installation:

```python
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from dotav2_q600_diagnostics import DiagnosticError
from validate_dotav2_q600_dump import load_cpu, validate_records
```

Release raw checkpoint/dump objects as soon as their compact representations
exist. Do not create a model, registry, runner, dataloader, or CUDA tensor.
Resolve the small-vehicle ID with
`protocol.classes.index('small-vehicle')`; absence is a diagnostic error.
When optional case inputs are present, load the case pickle with the same CPU
unpickler, require exactly two 600-row records in canonical mode, and keep its
evaluation entirely outside the canonical `EvaluationBundle`.

- [ ] **Step 7: Implement atomic publication and exit behavior**

```python
@contextmanager
def atomic_output_directory(output_dir):
    if output_dir.exists():
        raise DiagnosticError(f'output directory already exists: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f'{output_dir.name}.', dir=str(output_dir.parent)))
    try:
        yield temporary
        os.replace(temporary, output_dir)
    except BaseException:
        failed = temporary.with_name(temporary.name + '.failed')
        if temporary.exists():
            os.replace(temporary, failed)
        raise
```

Enter this context immediately after the output-path check. Catch a
`DiagnosticError` inside it, write `failure.json` containing
`schema_version=1`, `canonical=false`, the exception message, and the command,
then re-raise; the context preserves that directory with `.failed`. Write no
`diagnostics.json` until parity and all causal summaries have passed. On the
success path, write/fsync all eight files before leaving the context so the
rename is the publication point.

`main(argv=None)` catches only `DiagnosticError`, prints one line to stderr,
and returns 2. Unexpected exceptions retain their traceback. The executable
module ends with `raise SystemExit(main())`.

- [ ] **Step 8: Run the CLI component tests**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py -q
```

Expected: the synthetic run publishes exactly eight files, reports
`canonical=false`, refuses overwrite, and retains a `.failed` directory.

- [ ] **Step 9: Check command help without importing CUDA**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py --help
```

Expected: exit 0 and all required/optional arguments appear.

- [ ] **Step 10: Commit the CLI**

```bash
rtk git add projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "feat: publish atomic Q600 diagnostic bundles"
```

## Task 8: Run regression suites, D143 full replay, and the eventual E24 gate

**Files:**

- Verify: `projects/OVCapFlow/tools/dotav2_q600_diagnostics.py`
- Verify: `projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py`
- Verify: `tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py`
- Verify: `tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py`
- Verify: `tests/test_projects/ov_capflow/test_run_train_queue.py`
- Verify: `tests/test_projects/ov_capflow/test_audit_dotav2_runtime.py`

- [ ] **Step 1: Run the focused analyzer and existing validator tests**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py tests/test_projects/ov_capflow/test_validate_dotav2_q600_dump.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run queue/runtime regression tests**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_run_train_queue.py tests/test_projects/ov_capflow/test_audit_dotav2_runtime.py -q
```

Expected: all tests pass; the analyzer introduced no training/queue mutation.

- [ ] **Step 3: Run the D143 13,833×600 noncanonical regression replay**

Ensure the target directory does not already exist, then run:

```bash
rtk mkdir -p .lab/workspace/exp-8-d144
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/eval_t7_epoch18_raw13833_scale1280_gpu45_dump/predictions.pkl --config work_dirs/dotav2_cleanstart/eval_t7_epoch18_raw13833_scale1280_gpu45_dump/20260721_124708/vis_data/config.py --checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_18.pth --official-metrics-json work_dirs/dotav2_cleanstart/eval_t7_epoch18_raw13833_scale1280_gpu45_dump/20260721_124708/20260721_124708.json --output-dir .lab/workspace/exp-8-d144/d143_analyzer_regression --allow-noncanonical
```

Expected: exit 0, eight published files, `records=13833`,
`prediction_rows=8299800`, and same-dump exact mAP error at most `1e-7`.

- [ ] **Step 4: Verify D143 preregistered numerical anchors**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -c "import json,pathlib; p=pathlib.Path('.lab/workspace/exp-8-d144/d143_analyzer_regression/diagnostics.json'); d=json.loads(p.read_text()); assert d['integrity']['records']==13833; assert d['integrity']['prediction_rows']==8299800; assert d['parity']['exact_error']<=1e-7; g=d['gt_decomposition']; assert g['evaluator_reachable']==159129; assert abs(100*g['geometry_miss']/g['total_gt']-33.787)<=.01; assert abs(100*g['semantic_miss']/g['total_gt']-.893)<=.01; assert abs(d['calibration']['tp_score_iou_spearman']-.2778)<=1e-4; f=d['fp_regions']['ap_support']['shares']; assert abs(100*f['duplicate_fp']-14.89)<=.01; assert abs(100*f['semantic_fp']-3.23)<=.01; assert abs(100*f['localization_background_fp']-65.36)<=.01; assert abs(100*f['empty_tile_fp']-16.52)<=.01; print('D143 regression anchors passed')"
```

Expected: `D143 regression anchors passed`.

- [ ] **Step 5: Audit the manifest hashes against all seven payloads**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -c "import hashlib,json,pathlib; root=pathlib.Path('.lab/workspace/exp-8-d144/d143_analyzer_regression'); m=json.loads((root/'manifest.json').read_text()); assert set(m['files'])=={'diagnostics.json','per_class.csv','strata.csv','per_query.csv','case_studies.csv','error_examples.csv','report.md'}; assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==meta['sha256'] and (root/name).stat().st_size==meta['bytes'] for name,meta in m['files'].items()); print(hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())"
```

Expected: exit 0 and one 64-character manifest SHA256 for the experiment
ledger.

- [ ] **Step 6: Fix only evidence-backed mismatches, then rerun Steps 1–5**

If a D143 anchor fails, add the smallest synthetic regression test that
reproduces that exact failure before changing production logic. Repeat all
five verification steps after the fix. Do not relax any tolerance or alter a
recorded D143 anchor to make a failure pass.

- [ ] **Step 7: Commit any regression-only fix after all reruns pass**

Skip this commit when Steps 1–5 pass without code changes. Otherwise:

```bash
rtk git add projects/OVCapFlow/tools/dotav2_q600_diagnostics.py projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py tests/test_projects/ov_capflow/test_dotav2_q600_diagnostics.py
rtk git commit -m "fix: align Q600 diagnostics with D143 replay"
```

- [ ] **Step 8: Run the canonical E24 analyzer only after all exact inputs exist**

The official metrics argument deliberately accepts the evaluation directory
and requires exactly one applicable metric row, avoiding a timestamp
placeholder. Ensure the output directory is new, then run:

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl --config configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_scale1024_rare4x_gpu89_batch2.py --checkpoint work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth --official-metrics-json work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump --training-metrics-json work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/20260720_011705/vis_data/scalars.json --training-step 24 --output-dir work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/diagnostics
```

Expected: exit 0; `canonical=true`; exact same-dump mAP error at most `1e-7`;
training replay delta at most `5e-4`; rounded AP50 agreement; 13,833 records;
8,299,800 rows; eight deterministic output files. With no auxiliary case
dump, `case_gate=not_run` and the complete three-question routing package
remains explicitly incomplete.

- [ ] **Step 9: Record the E24 manifest hash in the experiment ledger**

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/data1/zcy/OV-CapFlow /data/zcy/anaconda3/envs/mmdet/bin/python -c "import hashlib,pathlib; p=pathlib.Path('work_dirs/dotav2_cleanstart/eval_t7_epoch24_raw13833_gpu2389_dump/diagnostics/manifest.json'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Expected: one 64-character SHA256. Add that exact value to the existing
experiment ledger together with the analyzer command, checkpoint/config/dump
hashes already captured in the manifest, and `case_gate` status. This ledger
edit is evidence bookkeeping, not authorization for E25+.

## Final definition of done

- [ ] All 15 approved unit/component categories pass.
- [ ] Existing validator, queue, and runtime-audit tests pass.
- [ ] D143 reproduces every numerical anchor within its frozen tolerance.
- [ ] The E24 canonical run passes integrity and both metric-parity gates once
  its dump exists.
- [ ] All eight E24 files are finite and deterministic; the manifest hashes
  the other seven, and the ledger records the manifest hash.
- [ ] No filtering, model/training change, GPU work, auxiliary case inference,
  or implicit next-route decision was introduced.
