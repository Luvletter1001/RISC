"""Pure calculation and serialization helpers for DOTA-v2 Q600 diagnostics."""

import csv
import hashlib
import heapq
import io
import json
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.evaluation.functional import average_precision
from mmengine import Config


class DiagnosticError(RuntimeError):
    """Report an expected diagnostic input or protocol failure."""


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


REFERENCE_SUFFIX = 'query_initializer.reference_embedding.weight'
SCORE_BINS = (
    '[0,.05)',
    '[.05,.10)',
    '[.10,.25)',
    '[.25,.50)',
    '[.50,.75)',
    '[.75,1]',
)
SIZE_BINS = ('<=8', '(8,16]', '(16,32]', '(32,64]', '>64')
DENSITY_BINS = (
    '1-10',
    '11-50',
    '51-100',
    '101-200',
    '201-400',
    '401-600',
    '>600',
)
ASPECT_BINS = ('<=1.5', '(1.5,3]', '(3,5]', '>5')
ANGLE_BINS = (
    '[0,15)',
    '[15,30)',
    '[30,45)',
    '[45,60)',
    '[60,75)',
    '[75,90]',
)
_CASE_MANIFEST_ORACLE = (
    ('context_false_sv', ('P0148__1024__651___0',)),
    ('true_sv_safety', ('P0682__1024__553___0',)),
)
EXPECTED_CASE_MANIFEST = {
    group: list(image_ids) for group, image_ids in _CASE_MANIFEST_ORACLE
}


def _validate_case_manifest(manifest: Any) -> Dict[str, List[str]]:
    """Return a detached copy of the fixed auxiliary-case contract."""
    if not isinstance(manifest, Mapping):
        raise DiagnosticError('case manifest must match the exact contract')
    try:
        keys = tuple(manifest)
        expected_keys = tuple(group for group, _ in _CASE_MANIFEST_ORACLE)
        keys_are_exact = (
            len(keys) == len(expected_keys) and
            all(type(key) is str for key in keys) and
            set(keys) == set(expected_keys)
        )
    except Exception as error:
        raise DiagnosticError(
            'case manifest must match the exact contract') from error
    if not keys_are_exact:
        raise DiagnosticError('case manifest must match the exact contract')

    validated = {}
    for group, expected_ids in _CASE_MANIFEST_ORACLE:
        try:
            actual_ids = manifest[group]
        except Exception as error:
            raise DiagnosticError(
                'case manifest must match the exact contract') from error
        if (type(actual_ids) is not list or
                len(actual_ids) != len(expected_ids) or
                any(type(image_id) is not str for image_id in actual_ids) or
                tuple(actual_ids) != expected_ids):
            raise DiagnosticError(
                'case manifest must match the exact contract')
        validated[group] = list(expected_ids)
    return validated


def _case_manifest_json_object(pairs: Sequence[Tuple[str, Any]]) -> dict:
    """Reject duplicate JSON object keys instead of accepting last-key-wins."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticError(
                'case manifest JSON has duplicate keys and is not exact')
        result[key] = value
    return result


def load_case_manifest(path: Any) -> Dict[str, List[str]]:
    """Load and validate the exact optional Q600 auxiliary-case manifest."""
    try:
        manifest = json.loads(
            Path(path).read_text(encoding='utf-8'),
            object_pairs_hook=_case_manifest_json_object,
        )
    except (OSError, TypeError, ValueError, UnicodeError) as error:
        raise DiagnosticError(
            'case manifest must be readable JSON matching the exact contract'
        ) from error
    return _validate_case_manifest(manifest)


def summarize_cases(case_records: Any,
                    manifest: Any,
                    small_vehicle_class_id: Any) -> List[Dict[str, Any]]:
    """Summarize optional cases without touching the canonical evaluation."""
    if case_records is None and manifest is None:
        return [{
            'case_gate': 'not_run',
            'image_id': None,
            'case_group': None,
        }]
    if case_records is None or manifest is None:
        raise DiagnosticError(
            'case records and case manifest must be provided together')
    validated_manifest = _validate_case_manifest(manifest)
    if (isinstance(small_vehicle_class_id, (bool, np.bool_)) or
            not isinstance(
                small_vehicle_class_id, (int, np.integer)) or
            small_vehicle_class_id < 0):
        raise DiagnosticError(
            'small_vehicle_class_id must be a nonnegative integer')
    small_vehicle_class_id = int(small_vehicle_class_id)
    try:
        records = tuple(case_records)
    except TypeError as error:
        raise DiagnosticError(
            'each fixed case image ID must appear exactly once') from error
    image_ids = []
    for record in records:
        if not isinstance(record, Mapping) or 'img_id' not in record:
            raise DiagnosticError(
                'each fixed case image ID must appear exactly once')
        image_id = record['img_id']
        if not isinstance(image_id, str):
            raise DiagnosticError(
                'each fixed case image ID must be a stable string and appear '
                'exactly once')
        image_ids.append(image_id)
    expected_ids = [
        image_id
        for group_ids in validated_manifest.values()
        for image_id in group_ids
    ]
    if (len(records) != len(expected_ids) or
            any(image_ids.count(image_id) != 1 for image_id in expected_ids) or
            any(image_id not in expected_ids for image_id in image_ids)):
        raise DiagnosticError(
            'each fixed case image ID must appear exactly once')
    records_by_id = {
        record['img_id']: record for record in records
    }
    records = tuple(records_by_id[image_id] for image_id in expected_ids)

    label_values = [small_vehicle_class_id]
    for record_index, record in enumerate(records):
        for section in ('pred_instances', 'gt_instances',
                        'ignored_instances'):
            instances = record.get(section)
            if instances is None and section == 'ignored_instances':
                continue
            if not isinstance(instances, Mapping):
                raise DiagnosticError(
                    '{} labels at case record {} require an instance mapping'.
                    format(section, record_index))
            try:
                labels = _to_numpy(
                    instances['labels'],
                    '{} labels at case record {}'.format(
                        section, record_index))
            except KeyError as error:
                raise DiagnosticError(
                    '{} labels missing at case record {}'.format(
                        section, record_index)) from error
            if (labels.ndim != 1 or
                    not np.issubdtype(labels.dtype, np.number) or
                    np.issubdtype(labels.dtype, np.bool_) or
                    not np.isrealobj(labels)):
                raise DiagnosticError(
                    '{} labels at case record {} must be integer labels'.
                    format(section, record_index))
            if (not np.isfinite(labels).all() or
                    not np.equal(labels, np.floor(labels)).all() or
                    (labels < 0).any()):
                raise DiagnosticError(
                    '{} labels at case record {} must be nonnegative integer '
                    'labels'.format(section, record_index))
            if len(labels):
                label_values.append(int(labels.max()))
    num_classes = max(label_values) + 1
    prepared = prepare_records_variable_queries(records, num_classes)
    evaluation = evaluate_records(
        prepared, num_classes=num_classes, iou_threshold=0.5)
    evidence = decompose_ground_truth(prepared, iou_threshold=0.5)
    small_vehicle_result = evaluation.class_results[
        int(small_vehicle_class_id)]

    record_indexes = {
        record.img_id: index for index, record in enumerate(prepared)
    }
    rows = []
    for group, group_ids in validated_manifest.items():
        for image_id in group_ids:
            record_index = record_indexes[image_id]
            record = prepared[record_index]
            small_vehicle_rows = record.labels == small_vehicle_class_id
            scores = record.scores[small_vehicle_rows]
            score_order = np.argsort(-scores, kind='mergesort')
            sorted_scores = [
                float(value) for value in scores[score_order]
            ]
            outcomes = evaluation.row_outcomes[
                record_index][small_vehicle_rows]
            witnesses = sorted(
                (row for row in evidence
                 if row.img_id == image_id and
                 row.class_id == small_vehicle_class_id),
                key=lambda row: row.gt_index,
            )
            witness_rows = [{
                'gt_index': row.gt_index,
                'state': row.state,
                'best_same_query': row.best_same_query,
                'best_same_iou': row.best_same_iou,
                'best_any_query': row.best_any_query,
                'best_any_iou': row.best_any_iou,
                'candidate_count': row.candidate_count,
                'witness_query': row.witness_query,
                'witness_score': row.witness_score,
                'witness_iou': row.witness_iou,
            } for row in witnesses]

            false_witness = None
            false_offsets = np.flatnonzero(
                (outcomes != 'tp') &
                (outcomes != 'ignored_prediction'))
            if len(false_offsets):
                selected_offset = int(false_offsets[
                    np.argmax(scores[false_offsets])])
                query_ids = np.flatnonzero(small_vehicle_rows)
                query_id = int(query_ids[selected_offset])
                nearest_gt_index = None
                nearest_gt_label = None
                nearest_gt_iou = 0.0
                if len(record.gt_boxes):
                    ious = _rotated_iou(
                        record.boxes[query_id:query_id + 1],
                        record.gt_boxes)[0]
                    nearest_gt_index = int(np.argmax(ious))
                    nearest_gt_label = int(record.gt_labels[
                        nearest_gt_index])
                    nearest_gt_iou = float(ious[nearest_gt_index])
                false_witness = {
                    'query_id': query_id,
                    'score': float(record.scores[query_id]),
                    'outcome': str(outcomes[selected_offset]),
                    'nearest_gt_index': nearest_gt_index,
                    'nearest_gt_label': nearest_gt_label,
                    'nearest_gt_iou': nearest_gt_iou,
                }

            row = {
                'case_gate': 'complete',
                'image_id': image_id,
                'case_group': group,
                'gt_count': len(record.gt_boxes),
                'small_vehicle_gt_count': int(np.sum(
                    record.gt_labels == small_vehicle_class_id)),
                'predicted_small_vehicle_rows': int(np.sum(
                    small_vehicle_rows)),
                'top1_small_vehicle_scores': sorted_scores[:1],
                'top5_small_vehicle_scores': sorted_scores[:5],
                'top20_small_vehicle_scores': sorted_scores[:20],
                'small_vehicle_ap_support_count': sum(
                    value == image_id
                    for value in small_vehicle_result.image_ids[
                        :small_vehicle_result.ap_support_end]),
                'small_vehicle_gt_witnesses': witness_rows,
                'highest_score_false_small_vehicle_witness': false_witness,
            }
            for outcome in _ROW_OUTCOMES:
                row['small_vehicle_{}_count'.format(outcome)] = int(np.sum(
                    outcomes == outcome))
            for state in (
                    'geometry_miss', 'semantic_miss', 'ownership_miss',
                    'evaluator_reachable'):
                row['small_vehicle_{}_count'.format(state)] = sum(
                    witness.state == state for witness in witnesses)
            rows.append(row)
    return rows


def _finite_real_metric(value: Any, description: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise DiagnosticError(
            '{} must be a real numeric finite value'.format(description))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DiagnosticError(
            '{} must be a real numeric finite value'.format(
                description)) from error
    if not np.isfinite(result):
        raise DiagnosticError(
            '{} must be a real numeric finite value'.format(description))
    return result


def _canonical_image_id_bytes(img_id: Any) -> bytes:
    """Encode supported image IDs with stable recursive type boundaries."""
    if isinstance(img_id, str):
        tag = b's'
        payload = img_id.encode('utf-8')
    elif (not isinstance(img_id, (bool, np.bool_)) and
          isinstance(img_id, (int, np.integer))):
        tag = b'i'
        payload = str(int(img_id)).encode('ascii')
    elif isinstance(img_id, tuple):
        tag = b't'
        payload = b''.join(
            _canonical_image_id_bytes(value) for value in img_id)
    else:
        raise DiagnosticError(
            'image ID must use supported stable str, integer, or tuple values')
    return tag + len(payload).to_bytes(8, byteorder='big') + payload


def stable_sample_keys(keys: Any,
                       sample_size: int,
                       seed: int) -> List[Tuple[Any, int]]:
    """Select keys by their deterministic SHA256 rank in bounded memory."""
    if (isinstance(sample_size, (bool, np.bool_)) or
            not isinstance(sample_size, (int, np.integer)) or
            sample_size < 0):
        raise DiagnosticError('sample_size must be a nonnegative integer')
    if (isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer))):
        raise DiagnosticError('seed must be an integer')
    if sample_size == 0:
        return []

    def ranked_keys():
        try:
            iterator = iter(keys)
        except TypeError as error:
            raise DiagnosticError('sample keys must be iterable') from error
        for key in iterator:
            if (not isinstance(key, (tuple, list)) or len(key) != 2):
                raise DiagnosticError(
                    'sample keys must be (img_id, query_id) pairs')
            img_id, query_id = key
            if (isinstance(query_id, (bool, np.bool_)) or
                    not isinstance(query_id, (int, np.integer)) or
                    query_id < 0):
                raise DiagnosticError(
                    'sample query IDs must be nonnegative integers')
            query_id = int(query_id)
            canonical_id = _canonical_image_id_bytes(img_id)
            image_text = str(img_id)
            payload = f'{int(seed)}\0{image_text}\0{query_id}'.encode('utf-8')
            rank = (
                hashlib.sha256(payload).hexdigest(),
                canonical_id,
                query_id,
            )
            yield rank, img_id, query_id

    try:
        selected = heapq.nsmallest(
            int(sample_size), ranked_keys(), key=lambda item: item[0])
    except DiagnosticError:
        raise
    except (TypeError, ValueError) as error:
        raise DiagnosticError(
            'failed to rank deterministic sample keys') from error
    return [(img_id, query_id) for _, img_id, query_id in selected]


def _numeric_vector(values: Any, description: str) -> np.ndarray:
    """Return one finite real floating vector or raise DiagnosticError."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(
            '{} must be a numeric 1D array'.format(description)) from error
    if array.ndim != 1:
        raise DiagnosticError('{} must be a 1D array'.format(description))
    if (not np.issubdtype(array.dtype, np.number) or
            np.issubdtype(array.dtype, np.bool_) or
            not np.isrealobj(array)):
        raise DiagnosticError(
            '{} must be a real numeric array'.format(description))
    try:
        result = array.astype(np.float64, copy=False)
        finite = np.isfinite(result).all()
    except (TypeError, ValueError, OverflowError) as error:
        raise DiagnosticError(
            '{} must be a finite numeric array'.format(description)) from error
    if not finite:
        raise DiagnosticError(
            '{} must be a finite numeric array'.format(description))
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute one-based average tied ranks with a stable merge sort."""
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def safe_spearman(lhs: Any, rhs: Any) -> Optional[float]:
    """Return Spearman's rank association, or None without rank variation."""
    lhs_array = _numeric_vector(lhs, 'left Spearman values')
    rhs_array = _numeric_vector(rhs, 'right Spearman values')
    if len(lhs_array) != len(rhs_array):
        raise DiagnosticError('Spearman values must be aligned')
    if (len(lhs_array) < 2 or
            np.all(lhs_array == lhs_array[0]) or
            np.all(rhs_array == rhs_array[0])):
        return None
    lhs_rank = _average_ranks(lhs_array)
    rhs_rank = _average_ranks(rhs_array)
    lhs_centered = lhs_rank - lhs_rank.mean()
    rhs_centered = rhs_rank - rhs_rank.mean()
    denominator = np.sqrt(
        np.dot(lhs_centered, lhs_centered) *
        np.dot(rhs_centered, rhs_centered))
    return float(np.dot(lhs_centered, rhs_centered) / denominator)


def size_bin(value: Any) -> str:
    """Assign a positive square-root area to the fixed size bins."""
    size = _finite_real_metric(value, 'size')
    if size <= 0:
        raise DiagnosticError('size must be positive')
    if size <= 8:
        return SIZE_BINS[0]
    if size <= 16:
        return SIZE_BINS[1]
    if size <= 32:
        return SIZE_BINS[2]
    if size <= 64:
        return SIZE_BINS[3]
    return SIZE_BINS[4]


def density_bin(count: Any) -> str:
    """Assign a positive integer image GT count to fixed density bins."""
    if (isinstance(count, (bool, np.bool_)) or
            not isinstance(count, (int, np.integer)) or count < 1):
        raise DiagnosticError('density must be a positive integer')
    if count <= 10:
        return DENSITY_BINS[0]
    if count <= 50:
        return DENSITY_BINS[1]
    if count <= 100:
        return DENSITY_BINS[2]
    if count <= 200:
        return DENSITY_BINS[3]
    if count <= 400:
        return DENSITY_BINS[4]
    if count <= 600:
        return DENSITY_BINS[5]
    return DENSITY_BINS[6]


def aspect_bin(value: Any) -> str:
    """Assign a symmetric aspect ratio to the fixed aspect bins."""
    ratio = _finite_real_metric(value, 'aspect ratio')
    if ratio < 1:
        raise DiagnosticError('aspect ratio must be at least one')
    if ratio <= 1.5:
        return ASPECT_BINS[0]
    if ratio <= 3:
        return ASPECT_BINS[1]
    if ratio <= 5:
        return ASPECT_BINS[2]
    return ASPECT_BINS[3]


def angle_bin(value: Any) -> str:
    """Assign a periodic rotated-box angle to one bin over [0, pi/2]."""
    angle = abs(_finite_real_metric(value, 'angle')) % np.pi
    angle = min(angle, np.pi - angle)
    degrees = float(np.rad2deg(angle))
    nearest_boundary = round(degrees / 15.0) * 15.0
    if np.isclose(degrees, nearest_boundary, rtol=0.0, atol=1e-10):
        degrees = nearest_boundary
    degrees = min(max(degrees, 0.0), 90.0)
    if degrees < 15:
        return ANGLE_BINS[0]
    if degrees < 30:
        return ANGLE_BINS[1]
    if degrees < 45:
        return ANGLE_BINS[2]
    if degrees < 60:
        return ANGLE_BINS[3]
    if degrees < 75:
        return ANGLE_BINS[4]
    return ANGLE_BINS[5]


def score_bin(value: Any) -> str:
    """Assign a bounded confidence score to the fixed score bins."""
    score = _finite_real_metric(value, 'score')
    if not 0 <= score <= 1:
        raise DiagnosticError('score must be in [0, 1]')
    if score < .05:
        return SCORE_BINS[0]
    if score < .10:
        return SCORE_BINS[1]
    if score < .25:
        return SCORE_BINS[2]
    if score < .50:
        return SCORE_BINS[3]
    if score < .75:
        return SCORE_BINS[4]
    return SCORE_BINS[5]


def effective_query_count(counts: Any) -> Optional[float]:
    """Return exp(Shannon entropy) for nonnegative query contributions."""
    values = _numeric_vector(counts, 'query contributions')
    if np.any(values < 0):
        raise DiagnosticError('query contributions must be nonnegative')
    total = float(values.sum())
    if total == 0:
        return None
    probabilities = values[values > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def refinement_values(prior: Any,
                      final: Any,
                      patch_size: Any = 1024) -> Dict[str, float]:
    """Measure symmetric refinement from one normalized prior to final box."""
    prior_array = _numeric_vector(prior, 'prior box')
    final_array = _numeric_vector(final, 'final box')
    if prior_array.shape != (5,) or final_array.shape != (5,):
        raise DiagnosticError('prior and final boxes must have shape (5,)')
    patch_size = _finite_real_metric(patch_size, 'patch size')
    if patch_size <= 0:
        raise DiagnosticError('patch size must be positive')
    if np.any(prior_array[2:4] <= 0) or np.any(final_array[2:4] <= 0):
        raise DiagnosticError('prior and final box dimensions must be positive')
    if np.any(prior_array < 0) or np.any(prior_array > 1):
        raise DiagnosticError(
            'normalized prior box components must be in [0, 1]')

    prior_box = np.array([
        prior_array[0] * patch_size,
        prior_array[1] * patch_size,
        prior_array[2] * patch_size,
        prior_array[3] * patch_size,
        prior_array[4] * np.pi,
    ], dtype=np.float64)
    center = float(np.linalg.norm(final_array[:2] - prior_box[:2]))
    prior_area = prior_box[2] * prior_box[3]
    final_area = final_array[2] * final_array[3]
    scale = float(np.exp(abs(np.log(np.sqrt(final_area / prior_area)))))
    prior_aspect = max(prior_box[2:4]) / min(prior_box[2:4])
    final_aspect = max(final_array[2:4]) / min(final_array[2:4])
    aspect = float(np.exp(abs(np.log(final_aspect / prior_aspect))))
    raw_angle = abs(float(final_array[4] - prior_box[4])) % np.pi
    angle = min(raw_angle, np.pi - raw_angle)
    values = (center, scale, aspect, angle)
    if not np.isfinite(values).all():
        raise DiagnosticError('refinement values must be finite')
    return {
        'center_distance': center,
        'scale_change': scale,
        'aspect_change': aspect,
        'angle_change': angle,
    }


def _reference_matrix(reference_points: Any) -> np.ndarray:
    """Validate shared normalized query reference boxes."""
    try:
        references = np.asarray(reference_points)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(
            'reference points must be a numeric Qx5 array') from error
    if references.ndim != 2 or references.shape[1:] != (5,):
        raise DiagnosticError('reference points must have shape (Q, 5)')
    if (not np.issubdtype(references.dtype, np.number) or
            np.issubdtype(references.dtype, np.bool_) or
            not np.isrealobj(references)):
        raise DiagnosticError('reference points must be real numeric values')
    try:
        references = references.astype(np.float64, copy=False)
        finite = np.isfinite(references).all()
    except (TypeError, ValueError, OverflowError) as error:
        raise DiagnosticError(
            'reference points must be finite numeric values') from error
    if not finite:
        raise DiagnosticError('reference points must be finite numeric values')
    if len(references) and np.any(references[:, 2:4] <= 0):
        raise DiagnosticError('reference box dimensions must be positive')
    if np.any(references < 0) or np.any(references > 1):
        raise DiagnosticError(
            'normalized reference point components must be in [0, 1]')
    return references


def attach_refinement(
        evidence: Sequence[GtEvidence],
        records: Sequence[PreparedRecord],
        reference_points: Any,
        patch_size: Any = 1024) -> Tuple[GtEvidence, ...]:
    """Attach prior-to-final deltas to evidence rows with witnesses."""
    patch_size = _finite_real_metric(patch_size, 'patch size')
    if patch_size <= 0:
        raise DiagnosticError('patch size must be positive')
    references = _reference_matrix(reference_points)
    try:
        evidence_rows = tuple(evidence)
        prepared_records = tuple(records)
    except TypeError as error:
        raise DiagnosticError(
            'evidence and records must be sequences') from error

    record_by_id = {}
    for record in prepared_records:
        if not isinstance(record, PreparedRecord):
            raise DiagnosticError('refinement records must be prepared records')
        _validate_decomposition_record(record)
        if np.any(record.boxes[:, 2:4] <= 0):
            raise DiagnosticError('final box dimensions must be positive')
        try:
            duplicate = record.img_id in record_by_id
        except TypeError as error:
            raise DiagnosticError('record image IDs must be hashable') from error
        if duplicate:
            raise DiagnosticError(
                'duplicate refinement record image ID: {}'.format(
                    record.img_id))
        record_by_id[record.img_id] = record

    attached = []
    for row in evidence_rows:
        if not isinstance(row, GtEvidence):
            raise DiagnosticError('refinement evidence rows must be GtEvidence')
        if row.state not in (
                'geometry_miss', 'semantic_miss', 'ownership_miss',
                'evaluator_reachable'):
            raise DiagnosticError(
                'unknown GT evidence state: {}'.format(row.state))
        _validate_gt_evidence_state(row)
        try:
            record = record_by_id[row.img_id]
        except TypeError as error:
            raise DiagnosticError('evidence image IDs must be hashable') from error
        except KeyError as error:
            raise DiagnosticError(
                'missing refinement record for image {}'.format(
                    row.img_id)) from error
        if (isinstance(row.gt_index, (bool, np.bool_)) or
                not isinstance(row.gt_index, (int, np.integer)) or
                not 0 <= row.gt_index < len(record.gt_boxes)):
            raise DiagnosticError(
                'evidence GT index is missing from refinement record')
        if (isinstance(row.image_gt_count, (bool, np.bool_)) or
                not isinstance(row.image_gt_count, (int, np.integer)) or
                row.image_gt_count != len(record.gt_boxes)):
            raise DiagnosticError(
                'evidence image GT count disagrees with refinement record')
        if (isinstance(row.class_id, (bool, np.bool_)) or
                not isinstance(row.class_id, (int, np.integer)) or
                int(record.gt_labels[row.gt_index]) != int(row.class_id)):
            raise DiagnosticError(
                'evidence class disagrees with refinement record GT class')
        evidence_box = _numeric_vector(row.gt_box, 'evidence GT box')
        if (evidence_box.shape != (5,) or
                not np.allclose(
                    evidence_box, record.gt_boxes[row.gt_index],
                    rtol=0.0, atol=1e-7)):
            raise DiagnosticError(
                'evidence GT box disagrees with refinement record GT box')
        if row.witness_query is None:
            attached.append(row)
            continue

        query_id = int(row.witness_query)
        if query_id >= len(record.boxes):
            raise DiagnosticError(
                'witness query {} is missing from record {}'.format(
                    query_id, row.img_id))
        if query_id >= len(references):
            raise DiagnosticError(
                'witness query {} is missing from reference points'.format(
                    query_id))
        if int(record.labels[query_id]) != int(row.class_id):
            raise DiagnosticError(
                'witness query class label disagrees with evidence class')
        if not np.isclose(
                float(row.witness_score), float(record.scores[query_id]),
                rtol=0.0, atol=1e-7):
            raise DiagnosticError(
                'witness score disagrees with refinement record score')
        values = refinement_values(
            references[query_id], record.boxes[query_id], patch_size)
        attached.append(replace(
            row,
            prior_center_distance=values['center_distance'],
            prior_scale_change=values['scale_change'],
            prior_aspect_change=values['aspect_change'],
            prior_angle_change=values['angle_change'],
        ))
    return tuple(attached)


def load_class_protocol(config_path: Path, canonical: bool) -> ClassProtocol:
    """Load and validate the config-ordered base/novel class partition."""
    try:
        cfg = Config.fromfile(str(config_path))
        classes = tuple(cfg.get('classes', ()))
        base = tuple(cfg.get('base_classes', ()))
        novel = tuple(cfg.get('novel_classes', ()))
        unique_classes = set(classes)
        base_set = set(base)
        novel_set = set(novel)
    except (OSError, SyntaxError, TypeError, ValueError) as error:
        raise DiagnosticError(
            'failed to load class protocol from {}: {}'.format(
                config_path, error)) from error

    if not classes or len(unique_classes) != len(classes):
        raise DiagnosticError('classes must be present and unique')
    if len(base_set) != len(base) or len(novel_set) != len(novel):
        raise DiagnosticError('base and novel classes must be unique')
    if base_set & novel_set:
        raise DiagnosticError('base and novel classes overlap')
    if base_set | novel_set != unique_classes:
        raise DiagnosticError('base and novel classes must partition classes')
    if canonical and (len(classes), len(base), len(novel)) != (18, 14, 4):
        raise DiagnosticError('canonical protocol must be 18/14/4')
    return ClassProtocol(classes, base, novel, canonical)


def read_metric_record(path: Path,
                       step: Optional[int] = None) -> MetricRecord:
    """Select exactly one applicable DOTA metric row from JSONL files."""
    path = Path(path)
    files = sorted(path.rglob('*.json')) if path.is_dir() else [path]
    matches = []
    for file_path in files:
        try:
            lines = file_path.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeError) as error:
            raise DiagnosticError(
                'failed to read metric source {}: {}'.format(
                    file_path, error)) from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DiagnosticError(
                    'invalid JSON in {} at line {}: {}'.format(
                        file_path, line_number, error.msg)) from error
            if not isinstance(row, Mapping):
                raise DiagnosticError(
                    'metric row in {} at line {} must be an object'.format(
                        file_path, line_number))
            if 'dota/mAP' not in row or 'dota/AP50' not in row:
                continue
            if step is not None and row.get('step') != step:
                continue
            matches.append((file_path, row))

    if len(matches) != 1:
        raise DiagnosticError(
            'metric source must contain exactly one applicable row')

    source, row = matches[0]
    selected_step = row.get('step')
    if (selected_step is not None and
            (isinstance(selected_step, bool) or
             not isinstance(selected_step, int))):
        raise DiagnosticError(
            'metric step in {} must be an integer or null'.format(source))
    mean_ap = _finite_real_metric(
        row['dota/mAP'], 'dota/mAP in {}'.format(source))
    ap50 = _finite_real_metric(
        row['dota/AP50'], 'dota/AP50 in {}'.format(source))
    return MetricRecord(mean_ap, ap50, str(source), selected_step)


def extract_reference_points(checkpoint: Mapping[str, Any],
                             expected_queries: int,
                             canonical: bool) -> np.ndarray:
    """Extract the sole query reference tensor and map logits to points."""
    try:
        state = checkpoint.get('state_dict', checkpoint)
        matches = [value for key, value in state.items()
                   if key.endswith(REFERENCE_SUFFIX)]
    except (AttributeError, TypeError) as error:
        raise DiagnosticError(
            'checkpoint state must be a string-keyed mapping') from error
    if len(matches) != 1:
        raise DiagnosticError(
            'checkpoint must contain exactly one reference key')
    try:
        weight = torch.as_tensor(matches[0]).detach().cpu()
    except (RuntimeError, TypeError, ValueError) as error:
        raise DiagnosticError(
            'checkpoint reference value must be tensor-like') from error
    expected = (600, 5) if canonical else (expected_queries, 5)
    if tuple(weight.shape) != expected:
        raise DiagnosticError(
            'reference shape must be {}'.format(expected))
    if not weight.is_floating_point() or weight.is_complex():
        raise DiagnosticError(
            'checkpoint reference value must use a real floating dtype')
    if not torch.isfinite(weight).all().item():
        raise DiagnosticError('checkpoint reference values must be finite')
    try:
        references = weight.sigmoid().to(dtype=torch.float64).numpy()
    except (RuntimeError, TypeError, ValueError) as error:
        raise DiagnosticError(
            'failed to convert checkpoint reference values: {}'.format(
                error)) from error
    if not np.isfinite(references).all():
        raise DiagnosticError('checkpoint reference values must be finite')
    return references


def validate_metric_parity(
        reconstructed: float,
        official: MetricRecord,
        training: Optional[MetricRecord]) -> Dict[str, Any]:
    """Require exact same-dump parity and bounded training replay parity."""
    reconstructed_value = _finite_real_metric(
        reconstructed, 'reconstructed metric')
    official_map = _finite_real_metric(official.mAP, 'official mAP')
    official_ap50 = _finite_real_metric(official.ap50, 'official AP50')
    exact_error = abs(reconstructed_value - official_map)
    if (exact_error > 1e-7 or
            round(reconstructed_value, 3) != official_ap50):
        raise DiagnosticError('same-dump evaluator parity failed')
    result = {
        'exact_error': exact_error,
        'replay_delta_warning': False,
    }
    if training is not None:
        training_map = _finite_real_metric(training.mAP, 'training mAP')
        training_ap50 = _finite_real_metric(training.ap50, 'training AP50')
        delta = abs(official_map - training_map)
        if delta > 5e-4 or official_ap50 != training_ap50:
            raise DiagnosticError('training replay parity failed')
        result.update(
            training_delta=delta,
            replay_delta_warning=delta > 1e-7,
        )
    return result


def _as_tensor(value: Any) -> Any:
    """Unwrap MMRotate box containers while accepting plain tensors."""
    return value.tensor if hasattr(value, 'tensor') else value


def _to_numpy(value: Any, description: str) -> np.ndarray:
    try:
        tensor = torch.as_tensor(_as_tensor(value)).detach().cpu()
        return np.ascontiguousarray(tensor.numpy())
    except (RuntimeError, TypeError, ValueError) as error:
        raise DiagnosticError(
            '{} must be tensor-like: {}'.format(description, error)) from error


def _validate_instances(instances: Mapping[str, Any],
                        description: str,
                        num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        boxes = _to_numpy(instances['bboxes'], '{} boxes'.format(description))
        labels = _to_numpy(
            instances['labels'], '{} labels'.format(description))
    except KeyError as error:
        raise DiagnosticError(
            '{} missing {}'.format(description, error.args[0])) from error

    if boxes.ndim != 2 or boxes.shape[1:] != (5,):
        raise DiagnosticError(
            '{} box shape must be (N, 5), got {}'.format(
                description, boxes.shape))
    if labels.ndim != 1 or len(labels) != len(boxes):
        raise DiagnosticError(
            '{} label shape must be ({},), got {}'.format(
                description, len(boxes), labels.shape))
    try:
        boxes_finite = np.isfinite(boxes).all()
        labels_finite = np.isfinite(labels).all()
    except TypeError as error:
        raise DiagnosticError(
            '{} values must be numeric'.format(description)) from error
    if not boxes_finite or not labels_finite:
        raise DiagnosticError('{} values must be finite'.format(description))
    try:
        labels_are_integer = np.equal(labels, np.floor(labels)).all()
    except TypeError as error:
        raise DiagnosticError(
            '{} labels must have integer semantics'.format(
                description)) from error
    if not labels_are_integer:
        raise DiagnosticError(
            '{} labels must have integer semantics'.format(description))
    if ((labels < 0) | (labels >= num_classes)).any():
        raise DiagnosticError(
            '{} labels must be in [0, {})'.format(
                description, num_classes))
    if (boxes[:, 2:4] <= 0).any():
        raise DiagnosticError(
            '{} widths and heights must be positive'.format(description))
    return boxes, labels.astype(np.int64, copy=False)


def _prepare_records(
        records: Sequence[dict],
        num_classes: int,
        queries_per_image: Optional[int]) -> Tuple[PreparedRecord, ...]:
    """Validate records, optionally enforcing one prediction count."""
    prepared = []
    image_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DiagnosticError(
                'record at index {} must be a mapping'.format(index))
        try:
            img_id = record['img_id']
            pred = record['pred_instances']
            gt = record['gt_instances']
        except KeyError as error:
            raise DiagnosticError(
                'record at index {} missing {}'.format(
                    index, error.args[0])) from error
        try:
            duplicate = img_id in image_ids
        except TypeError as error:
            raise DiagnosticError(
                'image id at index {} must be hashable'.format(index)) from error
        if duplicate:
            raise DiagnosticError('duplicate image id: {}'.format(img_id))
        image_ids.add(img_id)

        if not isinstance(pred, Mapping) or not isinstance(gt, Mapping):
            raise DiagnosticError(
                'prediction and GT instances at index {} must be mappings'.format(
                    index))
        boxes, labels = _validate_instances(
            pred, 'prediction at index {}'.format(index), num_classes)
        try:
            scores = _to_numpy(
                pred['scores'], 'prediction scores at index {}'.format(index))
        except KeyError as error:
            raise DiagnosticError(
                'prediction at index {} missing scores'.format(index)) from error
        if (queries_per_image is not None and
                len(boxes) != queries_per_image):
            raise DiagnosticError(
                'prediction count at index {} must be {}, got {}'.format(
                    index, queries_per_image, len(boxes)))
        if scores.ndim != 1 or len(scores) != len(boxes):
            raise DiagnosticError(
                'prediction score shape at index {} must be ({},), got {}'.format(
                    index, len(boxes), scores.shape))
        if (not np.issubdtype(scores.dtype, np.number) or
                np.issubdtype(scores.dtype, np.bool_) or
                not np.isrealobj(scores)):
            raise DiagnosticError(
                'prediction scores at index {} must be real numeric values'.
                format(index))
        try:
            scores_finite = np.isfinite(scores).all()
        except TypeError as error:
            raise DiagnosticError(
                'prediction scores at index {} must be numeric'.format(
                    index)) from error
        if not scores_finite:
            raise DiagnosticError(
                'prediction values at index {} must be finite'.format(index))

        gt_boxes, gt_labels = _validate_instances(
            gt, 'GT at index {}'.format(index), num_classes)
        ignored = record.get('ignored_instances')
        if ignored is None:
            ignored_boxes = np.empty((0, 5), dtype=gt_boxes.dtype)
            ignored_labels = np.empty((0,), dtype=np.int64)
        else:
            if not isinstance(ignored, Mapping):
                raise DiagnosticError(
                    'ignored instances at index {} must be a mapping'.format(
                        index))
            ignored_boxes, ignored_labels = _validate_instances(
                ignored, 'ignored at index {}'.format(index), num_classes)

        prepared.append(PreparedRecord(
            img_id=img_id,
            boxes=boxes,
            scores=scores,
            labels=labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            ignored_boxes=ignored_boxes,
            ignored_labels=ignored_labels,
        ))
    return tuple(prepared)


def prepare_records(records: Sequence[dict],
                    queries_per_image: int,
                    num_classes: int) -> Tuple[PreparedRecord, ...]:
    """Validate canonical records without filtering or reordering rows."""
    return _prepare_records(records, num_classes, queries_per_image)


def prepare_records_variable_queries(
        records: Sequence[dict],
        num_classes: int) -> Tuple[PreparedRecord, ...]:
    """Prepare noncanonical fixtures while preserving per-image row counts."""
    return _prepare_records(records, num_classes, queries_per_image=None)


def _rotated_iou(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Calculate real CPU MMCV rotated IoU and expose float64 NumPy output."""
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.ndim != 2 or lhs.shape[1:] != (5,):
        raise DiagnosticError(
            'left IoU boxes must have shape (N, 5), got {}'.format(
                lhs.shape))
    if rhs.ndim != 2 or rhs.shape[1:] != (5,):
        raise DiagnosticError(
            'right IoU boxes must have shape (M, 5), got {}'.format(
                rhs.shape))
    if len(lhs) == 0 or len(rhs) == 0:
        return np.zeros((len(lhs), len(rhs)), dtype=np.float64)
    try:
        overlaps = box_iou_rotated(
            torch.from_numpy(np.ascontiguousarray(lhs)).float(),
            torch.from_numpy(np.ascontiguousarray(rhs)).float(),
        )
        return overlaps.detach().cpu().numpy().astype(
            np.float64, copy=False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise DiagnosticError(
            'failed to calculate rotated IoU: {}'.format(error)) from error


def _validate_decomposition_record(record: PreparedRecord) -> None:
    """Reject malformed prepared records at the decomposition boundary."""
    arrays = (
        ('prediction boxes', record.boxes, 2),
        ('prediction scores', record.scores, 1),
        ('prediction labels', record.labels, 1),
        ('GT boxes', record.gt_boxes, 2),
        ('GT labels', record.gt_labels, 1),
    )
    for description, array, dimensions in arrays:
        if not isinstance(array, np.ndarray) or array.ndim != dimensions:
            raise DiagnosticError(
                '{} in prepared record {} has invalid shape'.format(
                    description, record.img_id))
        try:
            finite = np.isfinite(array).all()
        except TypeError as error:
            raise DiagnosticError(
                '{} in prepared record {} must be numeric'.format(
                    description, record.img_id)) from error
        if not finite:
            raise DiagnosticError(
                '{} in prepared record {} must be finite'.format(
                    description, record.img_id))
    if record.boxes.shape[1:] != (5,) or record.gt_boxes.shape[1:] != (5,):
        raise DiagnosticError(
            'boxes in prepared record {} must have five columns'.format(
                record.img_id))
    if len(record.scores) != len(record.boxes) or len(record.labels) != len(
            record.boxes):
        raise DiagnosticError(
            'prediction rows in prepared record {} are inconsistent'.format(
                record.img_id))
    if len(record.gt_labels) != len(record.gt_boxes):
        raise DiagnosticError(
            'GT rows in prepared record {} are inconsistent'.format(
                record.img_id))


def decompose_ground_truth(
        records: Sequence[PreparedRecord],
        iou_threshold: float) -> Tuple[GtEvidence, ...]:
    """Decompose each ordinary GT into one query-reachability state."""
    iou_threshold = _finite_real_metric(iou_threshold, 'IoU threshold')
    if not 0.0 <= iou_threshold <= 1.0:
        raise DiagnosticError('IoU threshold must be in [0, 1]')
    try:
        records = tuple(records)
    except TypeError as error:
        raise DiagnosticError(
            'decomposition records must be prepared records') from error
    if any(not isinstance(record, PreparedRecord) for record in records):
        raise DiagnosticError(
            'decomposition records must be prepared records')

    evidence = []
    for record in records:
        _validate_decomposition_record(record)
        ious = _rotated_iou(record.boxes, record.gt_boxes)
        image_gt_count = len(record.gt_boxes)
        for gt_index in range(image_gt_count):
            class_id = int(record.gt_labels[gt_index])
            if len(record.boxes):
                best_any_query = int(np.argmax(ious[:, gt_index]))
                best_any_iou = float(ious[best_any_query, gt_index])
            else:
                best_any_query = None
                best_any_iou = 0.0

            same_query_ids = np.flatnonzero(record.labels == class_id)
            if len(same_query_ids):
                same_ious = ious[same_query_ids, gt_index]
                best_same_offset = int(np.argmax(same_ious))
                best_same_query = int(same_query_ids[best_same_offset])
                best_same_iou = float(same_ious[best_same_offset])
                qualifying = same_query_ids[same_ious >= iou_threshold]
            else:
                best_same_query = None
                best_same_iou = 0.0
                qualifying = np.empty(0, dtype=np.int64)

            same_class_gt_ids = np.flatnonzero(record.gt_labels == class_id)
            owners = []
            for query_id in qualifying:
                owned_offset = int(np.argmax(
                    ious[query_id, same_class_gt_ids]))
                owned_gt_id = int(same_class_gt_ids[owned_offset])
                if owned_gt_id == gt_index:
                    owners.append(int(query_id))

            witness_query = None
            witness_score = None
            witness_iou = None
            if len(record.boxes) == 0 or best_any_iou < iou_threshold:
                state = 'geometry_miss'
            elif len(qualifying) == 0:
                state = 'semantic_miss'
            elif len(owners) == 0:
                state = 'ownership_miss'
            else:
                state = 'evaluator_reachable'
                owner_ids = np.asarray(owners, dtype=np.int64)
                witness_offset = int(np.argmax(record.scores[owner_ids]))
                witness_query = int(owner_ids[witness_offset])
                witness_score = float(record.scores[witness_query])
                witness_iou = float(ious[witness_query, gt_index])

            evidence.append(GtEvidence(
                img_id=record.img_id,
                gt_index=gt_index,
                class_id=class_id,
                state=state,
                image_gt_count=image_gt_count,
                gt_box=tuple(float(value) for value in record.gt_boxes[
                    gt_index]),
                best_any_query=best_any_query,
                best_any_iou=best_any_iou,
                best_same_query=best_same_query,
                best_same_iou=best_same_iou,
                candidate_count=len(qualifying),
                witness_query=witness_query,
                witness_score=witness_score,
                witness_iou=witness_iou,
            ))
    return tuple(evidence)


def _validate_gt_evidence_state(row: GtEvidence) -> None:
    """Reject state/field combinations decomposition cannot produce."""
    witness_fields = (
        row.witness_query,
        row.witness_score,
        row.witness_iou,
    )
    has_witness_field = any(value is not None for value in witness_fields)
    if row.state in ('geometry_miss', 'semantic_miss'):
        if row.candidate_count != 0 or has_witness_field:
            raise DiagnosticError(
                '{} evidence cannot have candidates or witness fields'.format(
                    row.state))
        return
    if row.state == 'ownership_miss':
        if row.candidate_count <= 0 or has_witness_field:
            raise DiagnosticError(
                'ownership miss evidence requires candidates and no witness')
        return
    if row.state == 'evaluator_reachable':
        if row.candidate_count <= 0:
            raise DiagnosticError(
                'reachable evidence requires at least one candidate')
        if (isinstance(row.witness_query, (bool, np.bool_)) or
                not isinstance(row.witness_query, (int, np.integer)) or
                row.witness_query < 0):
            raise DiagnosticError(
                'reachable witness query must be a nonnegative integer')
        _finite_real_metric(row.witness_score, 'reachable witness score')
        witness_iou = _finite_real_metric(
            row.witness_iou, 'reachable witness IoU')
        if not 0.0 <= witness_iou <= 1.0:
            raise DiagnosticError('reachable witness IoU must be in [0, 1]')


def summarize_gt_evidence(evidence: Sequence[GtEvidence]) -> Dict[str, Any]:
    """Summarize mutually exclusive GT states and compact witnesses."""
    try:
        evidence = tuple(evidence)
    except TypeError as error:
        raise DiagnosticError('GT evidence must be a sequence') from error
    states = (
        'geometry_miss',
        'semantic_miss',
        'evaluator_reachable',
        'ownership_miss',
    )
    counts = {state: 0 for state in states}
    candidate_excess = 0
    witness_query_ids = set()
    for row in evidence:
        if not isinstance(row, GtEvidence):
            raise DiagnosticError('GT evidence rows must be GtEvidence')
        if row.state not in counts:
            raise DiagnosticError('unknown GT evidence state: {}'.format(
                row.state))
        if (isinstance(row.candidate_count, (bool, np.bool_)) or
                not isinstance(row.candidate_count, (int, np.integer)) or
                row.candidate_count < 0):
            raise DiagnosticError(
                'GT evidence candidate counts must be nonnegative integers')
        _validate_gt_evidence_state(row)
        counts[row.state] += 1
        candidate_excess += max(int(row.candidate_count) - 1, 0)
        if row.witness_query is not None:
            witness_query_ids.add(int(row.witness_query))

    sorted_witnesses = sorted(witness_query_ids)
    return {
        'total_gt': len(evidence),
        **counts,
        'candidate_excess': candidate_excess,
        'witness_query_ids': sorted_witnesses,
        'witness_query_count': len(sorted_witnesses),
    }


def _validated_count_sequence(counts: Any, description: str) -> Tuple[int, ...]:
    """Return a validated nonnegative integer count sequence."""
    if isinstance(counts, (str, bytes)):
        raise DiagnosticError('{} must be a count sequence'.format(
            description))
    try:
        values = tuple(counts)
    except TypeError as error:
        raise DiagnosticError('{} must be a count sequence'.format(
            description)) from error
    for value in values:
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or value < 0):
            raise DiagnosticError(
                '{} must contain nonnegative integers'.format(description))
    return tuple(int(value) for value in values)


def _capacity_fields(counts: Sequence[int], query_count: int) -> Dict[str, Any]:
    total_gt = sum(counts)
    capacity_excess = sum(
        max(count - query_count, 0) for count in counts)
    ceiling = ((total_gt - capacity_excess) / total_gt
               if total_gt else None)
    return {
        'total_gt': total_gt,
        'capacity_excess': capacity_excess,
        'optimistic_recall_ceiling': ceiling,
    }


def summarize_capacity(
        image_gt_counts: Sequence[int],
        class_image_gt_counts: Mapping[int, Sequence[int]],
        query_count: int) -> Dict[str, Any]:
    """Report the fixed-query capacity lower bound globally and by class."""
    if (isinstance(query_count, (bool, np.bool_)) or
            not isinstance(query_count, (int, np.integer)) or
            query_count <= 0):
        raise DiagnosticError('query_count must be a positive integer')
    query_count = int(query_count)
    image_counts = _validated_count_sequence(
        image_gt_counts, 'image GT counts')
    if not isinstance(class_image_gt_counts, Mapping):
        raise DiagnosticError('class image GT counts must be a mapping')

    class_counts = {}
    for class_id, counts in class_image_gt_counts.items():
        if (isinstance(class_id, (bool, np.bool_)) or
                not isinstance(class_id, (int, np.integer)) or
                class_id < 0):
            raise DiagnosticError(
                'capacity class IDs must be nonnegative integers')
        values = _validated_count_sequence(
            counts, 'class {} image GT counts'.format(class_id))
        if len(values) != len(image_counts):
            raise DiagnosticError(
                'class {} image GT counts must match image count'.format(
                    class_id))
        class_counts[int(class_id)] = values

    for image_index, image_total in enumerate(image_counts):
        class_total = sum(
            counts[image_index] for counts in class_counts.values())
        if class_total != image_total:
            raise DiagnosticError(
                'class image GT counts must partition image totals')

    per_class = {
        class_id: _capacity_fields(class_counts[class_id], query_count)
        for class_id in sorted(class_counts)
    }
    return {
        'query_count': query_count,
        **_capacity_fields(image_counts, query_count),
        'per_class': per_class,
    }


def _evaluate_image_class(
        record: PreparedRecord,
        class_id: int,
        iou_threshold: float) -> Dict[str, Any]:
    """Mirror MMRotate ``tpfp_default`` for one image and class."""
    query_ids = np.flatnonzero(record.labels == class_id)
    scores = record.scores[query_ids]
    boxes = record.boxes[query_ids]
    ordinary_gt_ids = np.flatnonzero(record.gt_labels == class_id)
    ignored_gt_ids = np.flatnonzero(record.ignored_labels == class_id)
    ordinary_boxes = record.gt_boxes[ordinary_gt_ids]
    ignored_boxes = record.ignored_boxes[ignored_gt_ids]
    all_same_class_gt = np.vstack((ordinary_boxes, ignored_boxes))
    ignored_assignment = np.concatenate((
        np.zeros(len(ordinary_boxes), dtype=bool),
        np.ones(len(ignored_boxes), dtype=bool),
    ))

    num_dets = len(query_ids)
    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)
    outcomes = np.full(num_dets, '', dtype=object)
    assigned_iou = np.zeros(num_dets, dtype=np.float64)
    same_label_iou = np.zeros(num_dets, dtype=np.float64)
    matched_gt = np.full(num_dets, -1, dtype=np.int64)

    if num_dets and len(ordinary_boxes):
        ordinary_ious = _rotated_iou(boxes, ordinary_boxes)
        same_label_iou = ordinary_ious.max(axis=1)

    other_gt_ids = np.flatnonzero(record.gt_labels != class_id)
    semantic_overlap = np.zeros(num_dets, dtype=bool)
    if num_dets and len(other_gt_ids):
        other_ious = _rotated_iou(boxes, record.gt_boxes[other_gt_ids])
        semantic_overlap = (other_ious >= iou_threshold).any(axis=1)

    ious_argmax = np.zeros(num_dets, dtype=np.int64)
    if num_dets and len(all_same_class_gt):
        overlaps = _rotated_iou(boxes, all_same_class_gt)
        assigned_iou = overlaps.max(axis=1)
        ious_argmax = overlaps.argmax(axis=1)

    local_order = np.argsort(-scores)
    gt_covered = np.zeros(len(all_same_class_gt), dtype=bool)
    for detection_index in local_order:
        if (len(all_same_class_gt) and
                assigned_iou[detection_index] >= iou_threshold):
            assignment = int(ious_argmax[detection_index])
            if ignored_assignment[assignment]:
                outcomes[detection_index] = 'ignored_prediction'
            else:
                matched_gt[detection_index] = ordinary_gt_ids[assignment]
                if not gt_covered[assignment]:
                    gt_covered[assignment] = True
                    tp[detection_index] = 1
                    outcomes[detection_index] = 'tp'
                else:
                    fp[detection_index] = 1
                    outcomes[detection_index] = 'duplicate_fp'
        else:
            fp[detection_index] = 1
            if len(record.gt_boxes) == 0:
                outcomes[detection_index] = 'empty_tile_fp'
            elif semantic_overlap[detection_index]:
                outcomes[detection_index] = 'semantic_fp'
            else:
                outcomes[detection_index] = 'localization_background_fp'

    return {
        'query_ids': query_ids,
        'scores': scores,
        'tp': tp,
        'fp': fp,
        'outcomes': outcomes,
        'assigned_iou': assigned_iou,
        'same_label_iou': same_label_iou,
        'matched_gt': matched_gt,
        'num_gts': len(ordinary_gt_ids),
    }


def _voc07_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Delegate VOC07 AP exactly to MMDetection's implementation."""
    return float(average_precision(recall, precision, mode='11points'))


def _validate_aligned_binary_rows(
        tp: Any, fp: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate official evaluator TP/FP row flags."""
    try:
        tp_array = np.asarray(tp)
        fp_array = np.asarray(fp)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(
            'TP/FP rows must be numeric arrays') from error
    if (tp_array.ndim != 1 or fp_array.ndim != 1 or
            len(tp_array) != len(fp_array)):
        raise DiagnosticError('TP/FP rows must be aligned 1D arrays')
    if not np.isrealobj(tp_array) or not np.isrealobj(fp_array):
        raise DiagnosticError('TP/FP rows must be real numeric arrays')
    try:
        finite = np.isfinite(tp_array).all() and np.isfinite(fp_array).all()
    except TypeError as error:
        raise DiagnosticError(
            'TP/FP rows must be finite numeric arrays') from error
    if not finite:
        raise DiagnosticError('TP/FP rows must be finite numeric arrays')
    if (not np.isin(tp_array, (0, 1)).all() or
            not np.isin(fp_array, (0, 1)).all()):
        raise DiagnosticError('TP/FP rows must contain only 0 or 1')
    if np.any((tp_array == 1) & (fp_array == 1)):
        raise DiagnosticError('a row cannot be both TP and FP')
    return tp_array, fp_array


def _perfect_ranking_rows(
        tp: Any, fp: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Put TPs first and FPs last while preserving neutral ignored rows."""
    tp_array, fp_array = _validate_aligned_binary_rows(tp, fp)
    true_positive = tp_array == 1
    false_positive = fp_array == 1
    ignored = ~(true_positive | false_positive)
    order = np.concatenate((
        np.flatnonzero(true_positive),
        np.flatnonzero(ignored),
        np.flatnonzero(false_positive),
    ))
    return tp_array[order], fp_array[order]


def _validate_pr_curve(
        recall: Any, precision: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate an aligned, bounded precision-recall curve."""
    try:
        recall_array = np.asarray(recall)
        precision_array = np.asarray(precision)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(
            'recall and precision must be numeric arrays') from error
    if (recall_array.ndim != 1 or precision_array.ndim != 1 or
            len(recall_array) != len(precision_array)):
        raise DiagnosticError(
            'recall and precision must be aligned 1D arrays')
    if not np.isrealobj(recall_array) or not np.isrealobj(precision_array):
        raise DiagnosticError(
            'recall and precision must be real numeric arrays')
    try:
        finite = (np.isfinite(recall_array).all() and
                  np.isfinite(precision_array).all())
        bounded = (
            ((recall_array >= 0.0) & (recall_array <= 1.0)).all() and
            ((precision_array >= 0.0) & (precision_array <= 1.0)).all())
        nondecreasing = np.all(recall_array[1:] >= recall_array[:-1])
    except TypeError as error:
        raise DiagnosticError(
            'recall and precision must be finite numeric arrays') from error
    if not finite:
        raise DiagnosticError(
            'recall and precision must be finite numeric arrays')
    if not bounded:
        raise DiagnosticError('recall and precision must be in [0, 1]')
    if not nondecreasing:
        raise DiagnosticError('recall must be nondecreasing')
    return recall_array, precision_array


def voc07_support_ranks(
        recall: Any, precision: Any) -> List[Optional[int]]:
    """Return the earliest rank supporting each VOC07 recall threshold."""
    recall_array, precision_array = _validate_pr_curve(recall, precision)

    support_ranks = []
    for threshold in np.arange(0., 1.0001, 0.1):
        eligible = np.flatnonzero(recall_array >= threshold)
        if len(eligible) == 0:
            support_ranks.append(None)
            continue
        precision_at_eligible = precision_array[eligible]
        support_ranks.append(int(eligible[np.argmax(precision_at_eligible)]))
    return support_ranks


def perfect_ranking_ap(tp: Any, fp: Any, num_gts: int) -> float:
    """Calculate VOC07 AP after moving every achieved TP before every FP."""
    if (isinstance(num_gts, (bool, np.bool_)) or
            not isinstance(num_gts, (int, np.integer)) or num_gts < 0):
        raise DiagnosticError('num_gts must be a nonnegative integer')
    ranked_tp, ranked_fp = _perfect_ranking_rows(tp, fp)
    if int(np.sum(ranked_tp)) > int(num_gts):
        raise DiagnosticError('TP count cannot exceed num_gts')

    cumulative_tp = np.cumsum(ranked_tp)
    cumulative_fp = np.cumsum(ranked_fp)
    eps = np.finfo(np.float32).eps
    recall = cumulative_tp / max(int(num_gts), eps)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, eps)
    return _voc07_ap(recall, precision)


def _repeat_object(value: Any, count: int) -> np.ndarray:
    """Repeat a possibly sequence-valued identifier as scalar objects."""
    values = np.empty(count, dtype=object)
    values.fill(value)
    return values


def evaluate_records(
        records: Sequence[PreparedRecord],
        num_classes: int,
        iou_threshold: float) -> EvaluationBundle:
    """Reconstruct the official DOTA rotated VOC07 evaluator exactly."""
    if (isinstance(num_classes, (bool, np.bool_)) or
            not isinstance(num_classes, int) or num_classes <= 0):
        raise DiagnosticError('num_classes must be a positive integer')
    iou_threshold = _finite_real_metric(iou_threshold, 'IoU threshold')
    records = tuple(records)
    if any(not isinstance(record, PreparedRecord) for record in records):
        raise DiagnosticError('evaluation records must be prepared records')

    row_outcomes = tuple(
        np.full(len(record.scores), '', dtype=object) for record in records)
    row_tp = tuple(
        np.zeros(len(record.scores), dtype=np.float32)
        for record in records)
    row_fp = tuple(
        np.zeros(len(record.scores), dtype=np.float32)
        for record in records)
    row_same_label_iou = tuple(
        np.zeros(len(record.scores), dtype=np.float64)
        for record in records)
    row_assigned_iou = tuple(
        np.zeros(len(record.scores), dtype=np.float64)
        for record in records)
    row_matched_gt = tuple(
        np.full(len(record.scores), -1, dtype=np.int64)
        for record in records)

    class_results = []
    for class_id in range(num_classes):
        pieces = []
        for record_index, record in enumerate(records):
            piece = _evaluate_image_class(
                record, class_id, iou_threshold)
            query_ids = piece['query_ids']
            row_outcomes[record_index][query_ids] = piece['outcomes']
            row_tp[record_index][query_ids] = piece['tp']
            row_fp[record_index][query_ids] = piece['fp']
            row_same_label_iou[record_index][query_ids] = (
                piece['same_label_iou'])
            row_assigned_iou[record_index][query_ids] = piece['assigned_iou']
            row_matched_gt[record_index][query_ids] = piece['matched_gt']
            pieces.append((record, piece))

        if pieces:
            image_ids = np.concatenate([
                _repeat_object(record.img_id, len(piece['query_ids']))
                for record, piece in pieces
            ])
            query_ids = np.concatenate([
                piece['query_ids'] for _, piece in pieces
            ]).astype(np.int64, copy=False)
            scores = np.concatenate([
                piece['scores'] for _, piece in pieces
            ])
            tp = np.concatenate([
                piece['tp'] for _, piece in pieces
            ]).astype(np.float32, copy=False)
            fp = np.concatenate([
                piece['fp'] for _, piece in pieces
            ]).astype(np.float32, copy=False)
            assigned_iou = np.concatenate([
                piece['assigned_iou'] for _, piece in pieces
            ]).astype(np.float64, copy=False)
            outcomes = np.concatenate([
                piece['outcomes'] for _, piece in pieces
            ])
            num_gts = sum(piece['num_gts'] for _, piece in pieces)
        else:
            image_ids = np.empty(0, dtype=object)
            query_ids = np.empty(0, dtype=np.int64)
            scores = np.empty(0, dtype=np.float32)
            tp = np.empty(0, dtype=np.float32)
            fp = np.empty(0, dtype=np.float32)
            assigned_iou = np.empty(0, dtype=np.float64)
            outcomes = np.empty(0, dtype=object)
            num_gts = 0

        global_order = np.argsort(-scores)
        image_ids = image_ids[global_order]
        query_ids = query_ids[global_order]
        scores = scores[global_order]
        tp = tp[global_order]
        fp = fp[global_order]
        assigned_iou = assigned_iou[global_order]
        outcomes = outcomes[global_order]

        cumulative_tp = np.cumsum(tp[np.newaxis, :], axis=1)
        cumulative_fp = np.cumsum(fp[np.newaxis, :], axis=1)
        eps = np.finfo(np.float32).eps
        gt_count = np.array([num_gts], dtype=int)
        recall = cumulative_tp / np.maximum(gt_count[:, np.newaxis], eps)
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, eps)
        recall = recall[0]
        precision = precision[0]
        ap = _voc07_ap(recall, precision)
        tp_indices = np.flatnonzero(tp)
        last_tp_end = int(tp_indices[-1] + 1) if len(tp_indices) else 0
        support_ranks = voc07_support_ranks(recall, precision)
        supported = [rank for rank in support_ranks if rank is not None]
        ap_support_end = max(supported) + 1 if supported else 0
        oracle_ap = perfect_ranking_ap(tp, fp, num_gts)
        class_results.append(ClassEvaluation(
            class_id=class_id,
            num_gts=num_gts,
            image_ids=image_ids,
            query_ids=query_ids,
            scores=scores,
            tp=tp,
            fp=fp,
            assigned_iou=assigned_iou,
            outcomes=outcomes,
            precision=precision,
            recall=recall,
            ap=ap,
            last_tp_end=last_tp_end,
            ap_support_end=ap_support_end,
            oracle_ap=oracle_ap,
        ))

    total_rows = sum(len(record.scores) for record in records)
    if sum(len(result.scores) for result in class_results) != total_rows:
        raise DiagnosticError('class detection accounting is inconsistent')
    total_gts = sum(len(record.gt_boxes) for record in records)
    if sum(result.num_gts for result in class_results) != total_gts:
        raise DiagnosticError('class GT accounting is inconsistent')
    for record_index, record in enumerate(records):
        outcomes = row_outcomes[record_index]
        ignored = (outcomes == 'ignored_prediction').astype(np.float32)
        accounting = row_tp[record_index] + row_fp[record_index] + ignored
        if (np.any(outcomes == '') or
                not np.array_equal(
                    accounting, np.ones(len(record.scores), dtype=np.float32))):
            raise DiagnosticError(
                'row outcome accounting is inconsistent for image {}'.format(
                    record.img_id))

    aps = [result.ap for result in class_results if result.num_gts > 0]
    mean_ap = np.array(aps).mean().item() if aps else 0.0
    return EvaluationBundle(
        records=records,
        class_results=tuple(class_results),
        row_outcomes=row_outcomes,
        row_tp=row_tp,
        row_fp=row_fp,
        row_same_label_iou=row_same_label_iou,
        row_assigned_iou=row_assigned_iou,
        row_matched_gt=row_matched_gt,
        mean_ap=mean_ap,
    )


_ROW_OUTCOMES = (
    'tp',
    'duplicate_fp',
    'semantic_fp',
    'localization_background_fp',
    'empty_tile_fp',
    'ignored_prediction',
)


def _integer_vector(values: Any, description: str) -> np.ndarray:
    """Return one integer-valued 1D array without changing its values."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(
            '{} must be an integer 1D array'.format(description)) from error
    if (array.ndim != 1 or
            not np.issubdtype(array.dtype, np.integer) or
            np.issubdtype(array.dtype, np.bool_)):
        raise DiagnosticError(
            '{} must be an integer 1D array'.format(description))
    return array.astype(np.int64, copy=False)


def _validate_evaluation_bundle(bundle: Any) -> None:
    """Validate row alignment needed by deterministic bundle summaries."""
    if not isinstance(bundle, EvaluationBundle):
        raise DiagnosticError('evaluation bundle must be an EvaluationBundle')
    row_names = (
        'row_outcomes',
        'row_tp',
        'row_fp',
        'row_same_label_iou',
        'row_assigned_iou',
        'row_matched_gt',
    )
    for name in row_names:
        rows = getattr(bundle, name)
        try:
            row_count = len(rows)
        except TypeError as error:
            raise DiagnosticError(
                '{} must align with evaluation records'.format(name)) from error
        if row_count != len(bundle.records):
            raise DiagnosticError(
                '{} must align with evaluation records'.format(name))

    image_ids = set()
    total_rows = 0
    for record_index, record in enumerate(bundle.records):
        if not isinstance(record, PreparedRecord):
            raise DiagnosticError('bundle records must be prepared records')
        _validate_decomposition_record(record)
        try:
            duplicate = record.img_id in image_ids
        except TypeError as error:
            raise DiagnosticError('bundle image IDs must be hashable') from error
        if duplicate:
            raise DiagnosticError(
                'bundle image IDs must be unique: {}'.format(record.img_id))
        image_ids.add(record.img_id)
        row_count = len(record.scores)
        total_rows += row_count
        arrays = {}
        for name in row_names:
            try:
                array = np.asarray(getattr(bundle, name)[record_index])
            except (TypeError, ValueError) as error:
                raise DiagnosticError(
                    '{} rows must be aligned 1D arrays'.format(name)) from error
            if array.ndim != 1 or len(array) != row_count:
                raise DiagnosticError(
                    '{} rows must align with record rows'.format(name))
            arrays[name] = array

        outcomes = arrays['row_outcomes']
        if not np.isin(outcomes, _ROW_OUTCOMES).all():
            raise DiagnosticError('bundle row outcomes contain unknown values')
        tp = _numeric_vector(arrays['row_tp'], 'bundle TP rows')
        fp = _numeric_vector(arrays['row_fp'], 'bundle FP rows')
        same_iou = _numeric_vector(
            arrays['row_same_label_iou'], 'bundle same-label IoU rows')
        assigned_iou = _numeric_vector(
            arrays['row_assigned_iou'], 'bundle assigned-IoU rows')
        matched_gt = _integer_vector(
            arrays['row_matched_gt'], 'bundle matched-GT rows')
        if (not np.isin(tp, (0, 1)).all() or
                not np.isin(fp, (0, 1)).all() or
                np.any((tp == 1) & (fp == 1))):
            raise DiagnosticError('bundle TP/FP row flags are invalid')
        if (np.any(same_iou < 0) or np.any(same_iou > 1) or
                np.any(assigned_iou < 0) or np.any(assigned_iou > 1)):
            raise DiagnosticError('bundle IoU rows must be in [0, 1]')
        if np.any(matched_gt < -1):
            raise DiagnosticError('bundle matched-GT rows are invalid')
        ignored = (outcomes == 'ignored_prediction').astype(np.float64)
        if not np.array_equal(tp + fp + ignored, np.ones(row_count)):
            raise DiagnosticError('bundle row outcome accounting is invalid')
        if not np.array_equal(tp == 1, outcomes == 'tp'):
            raise DiagnosticError('bundle TP rows and outcomes disagree')

    class_ids = []
    class_row_count = 0
    for result in bundle.class_results:
        if not isinstance(result, ClassEvaluation):
            raise DiagnosticError(
                'bundle class results must be ClassEvaluation values')
        if (isinstance(result.class_id, (bool, np.bool_)) or
                not isinstance(result.class_id, (int, np.integer)) or
                result.class_id < 0):
            raise DiagnosticError('bundle class IDs must be nonnegative integers')
        class_ids.append(int(result.class_id))
        result_rows = len(result.scores)
        class_row_count += result_rows
        aligned = (
            result.image_ids,
            result.query_ids,
            result.tp,
            result.fp,
            result.assigned_iou,
            result.outcomes,
        )
        if any(np.asarray(values).ndim != 1 or len(values) != result_rows
               for values in aligned):
            raise DiagnosticError('class result rows must be aligned')
        query_ids = _integer_vector(
            result.query_ids, 'class result query IDs')
        if np.any(query_ids < 0):
            raise DiagnosticError(
                'class result query IDs must be nonnegative')
        scores = _numeric_vector(result.scores, 'class result scores')
        tp = _numeric_vector(result.tp, 'class result TP rows')
        fp = _numeric_vector(result.fp, 'class result FP rows')
        assigned_iou = _numeric_vector(
            result.assigned_iou, 'class result assigned IoUs')
        if (not np.isin(tp, (0, 1)).all() or
                not np.isin(fp, (0, 1)).all() or
                np.any((tp == 1) & (fp == 1))):
            raise DiagnosticError('class result TP/FP flags are invalid')
        if np.any(assigned_iou < 0) or np.any(assigned_iou > 1):
            raise DiagnosticError('class result assigned IoUs must be in [0, 1]')
        if not np.isin(result.outcomes, _ROW_OUTCOMES).all():
            raise DiagnosticError('class result outcomes contain unknown values')
        if (isinstance(result.ap_support_end, (bool, np.bool_)) or
                not isinstance(result.ap_support_end, (int, np.integer)) or
                not 0 <= result.ap_support_end <= result_rows):
            raise DiagnosticError('class AP-support endpoint is invalid')
        del scores

    if len(set(class_ids)) != len(class_ids):
        raise DiagnosticError('bundle class IDs must be unique')
    if sorted(class_ids) != list(range(len(class_ids))):
        raise DiagnosticError('bundle class IDs must be contiguous from zero')
    if class_row_count != total_rows:
        raise DiagnosticError('class result rows must partition bundle rows')


def summarize_calibration(
        bundle: EvaluationBundle,
        sample_size: int,
        seed: int,
        iou_threshold: float = .5) -> Dict[str, Any]:
    """Summarize sampled score calibration and all-TP associations."""
    _validate_evaluation_bundle(bundle)
    iou_threshold = _finite_real_metric(iou_threshold, 'IoU threshold')
    if not 0 <= iou_threshold <= 1:
        raise DiagnosticError('IoU threshold must be in [0, 1]')
    for record in bundle.records:
        if np.any(record.scores < 0) or np.any(record.scores > 1):
            raise DiagnosticError('calibration scores must be in [0, 1]')

    sample_keys = stable_sample_keys(
        ((record.img_id, query_id)
         for record in bundle.records
         for query_id in range(len(record.scores))),
        sample_size=sample_size,
        seed=seed,
    )
    sample_positions = {
        key: index for index, key in enumerate(sample_keys)
    }
    sampled_rows = [None] * len(sample_keys)
    for record_index, record in enumerate(bundle.records):
        for query_id in range(len(record.scores)):
            position = sample_positions.get((record.img_id, query_id))
            if position is None:
                continue
            sampled_rows[position] = (
                float(record.scores[query_id]),
                str(bundle.row_outcomes[record_index][query_id]),
                float(bundle.row_tp[record_index][query_id]),
                float(bundle.row_fp[record_index][query_id]),
                float(bundle.row_same_label_iou[record_index][query_id]),
                float(bundle.row_assigned_iou[record_index][query_id]),
            )
    if any(row is None for row in sampled_rows):
        raise DiagnosticError('deterministic sample rows could not be resolved')

    accumulators = {
        label: {
            'row_count': 0,
            'evaluator_row_count': 0,
            'ignored_count': 0,
            'tp_count': 0,
            'same_label_hit_count': 0,
            'same_label_iou_sum': 0.0,
            'tp_assigned_iou_sum': 0.0,
        }
        for label in SCORE_BINS
    }
    for score, outcome, tp, fp, same_iou, assigned_iou in sampled_rows:
        label = score_bin(score)
        accumulator = accumulators[label]
        accumulator['row_count'] += 1
        accumulator['evaluator_row_count'] += int(tp + fp)
        accumulator['ignored_count'] += int(outcome == 'ignored_prediction')
        accumulator['tp_count'] += int(tp)
        hit = same_iou >= iou_threshold
        accumulator['same_label_hit_count'] += int(hit)
        accumulator['same_label_iou_sum'] += same_iou
        if tp:
            accumulator['tp_assigned_iou_sum'] += assigned_iou

    score_rows = []
    for label in SCORE_BINS:
        accumulator = accumulators[label]
        row_count = accumulator['row_count']
        evaluator_count = accumulator['evaluator_row_count']
        tp_count = accumulator['tp_count']
        score_rows.append({
            'score_bin': label,
            'row_count': row_count,
            'evaluator_row_count': evaluator_count,
            'ignored_count': accumulator['ignored_count'],
            'tp_count': tp_count,
            'tp_precision': (
                tp_count / evaluator_count if evaluator_count else None),
            'same_label_hit_count': accumulator['same_label_hit_count'],
            'same_label_hit_rate': (
                accumulator['same_label_hit_count'] / row_count
                if row_count else None),
            'mean_same_label_iou': (
                accumulator['same_label_iou_sum'] / row_count
                if row_count else None),
            'mean_tp_assigned_iou': (
                accumulator['tp_assigned_iou_sum'] / tp_count
                if tp_count else None),
        })

    per_class = []
    global_score_parts = []
    global_iou_parts = []
    for result in sorted(
            bundle.class_results, key=lambda value: value.class_id):
        tp_mask = result.tp == 1
        tp_scores = result.scores[tp_mask]
        tp_ious = result.assigned_iou[tp_mask]
        if len(tp_scores):
            global_score_parts.append(tp_scores)
            global_iou_parts.append(tp_ious)
        per_class.append({
            'class_id': int(result.class_id),
            'tp_count': int(tp_mask.sum()),
            'tp_score_iou_spearman': safe_spearman(tp_scores, tp_ious),
        })
    if global_score_parts:
        global_scores = np.concatenate(global_score_parts)
        global_ious = np.concatenate(global_iou_parts)
    else:
        global_scores = np.empty(0, dtype=np.float64)
        global_ious = np.empty(0, dtype=np.float64)

    return {
        'sample_size_requested': int(sample_size),
        'sample_size_used': len(sample_keys),
        'sample_keys': sample_keys,
        'tp_score_iou_spearman': safe_spearman(
            global_scores, global_ious),
        'per_class': per_class,
        'score_bins': score_rows,
    }


def _positive_integer(value: Any, description: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value <= 0):
        raise DiagnosticError('{} must be a positive integer'.format(
            description))
    return int(value)


def summarize_queries(
        bundle: EvaluationBundle,
        evidence: Sequence[GtEvidence],
        query_count: int,
        num_classes: int) -> Dict[str, Any]:
    """Summarize fixed original query IDs without mutating bundle rows."""
    query_count = _positive_integer(query_count, 'query_count')
    num_classes = _positive_integer(num_classes, 'num_classes')
    _validate_evaluation_bundle(bundle)
    if len(bundle.class_results) != num_classes:
        raise DiagnosticError(
            'num_classes must match evaluation class results')

    total_rows = np.zeros(query_count, dtype=np.int64)
    label_counts = np.zeros((query_count, num_classes), dtype=np.int64)
    outcome_counts = {
        outcome: np.zeros(query_count, dtype=np.int64)
        for outcome in _ROW_OUTCOMES
    }
    record_by_id = {}
    for record_index, record in enumerate(bundle.records):
        row_count = len(record.scores)
        if row_count > query_count:
            raise DiagnosticError(
                'original query ID exceeds query_count in image {}'.format(
                    record.img_id))
        labels = _integer_vector(record.labels, 'record class IDs')
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise DiagnosticError('record class IDs are out of range')
        query_ids = np.arange(row_count, dtype=np.int64)
        total_rows[:row_count] += 1
        np.add.at(label_counts, (query_ids, labels), 1)
        outcomes = np.asarray(bundle.row_outcomes[record_index])
        for outcome in _ROW_OUTCOMES:
            outcome_counts[outcome][:row_count] += outcomes == outcome
        record_by_id[record.img_id] = record

    witness_counts = np.zeros(query_count, dtype=np.int64)
    try:
        evidence_rows = tuple(evidence)
    except TypeError as error:
        raise DiagnosticError('GT evidence must be a sequence') from error
    for row in evidence_rows:
        if not isinstance(row, GtEvidence):
            raise DiagnosticError('GT evidence rows must be GtEvidence')
        if (isinstance(row.class_id, (bool, np.bool_)) or
                not isinstance(row.class_id, (int, np.integer)) or
                not 0 <= row.class_id < num_classes):
            raise DiagnosticError('GT evidence class IDs are out of range')
        if row.state not in (
                'geometry_miss', 'semantic_miss', 'ownership_miss',
                'evaluator_reachable'):
            raise DiagnosticError(
                'unknown GT evidence state: {}'.format(row.state))
        _validate_gt_evidence_state(row)
        try:
            record = record_by_id[row.img_id]
        except (KeyError, TypeError) as error:
            raise DiagnosticError(
                'GT evidence image is missing from evaluation records') from error
        if row.witness_query is not None:
            query_id = int(row.witness_query)
            if query_id >= query_count:
                raise DiagnosticError(
                    'GT evidence witness query is out of range')
            if query_id >= len(record.scores):
                raise DiagnosticError(
                    'GT evidence witness has no original record query')
            if int(record.labels[query_id]) != int(row.class_id):
                raise DiagnosticError(
                    'GT evidence witness class disagrees with record class')
            witness_counts[query_id] += 1

    support_counts = np.zeros(query_count, dtype=np.int64)
    support_label_counts = np.zeros(num_classes, dtype=np.int64)
    for result in bundle.class_results:
        class_id = int(result.class_id)
        if not 0 <= class_id < num_classes:
            raise DiagnosticError('evaluation class IDs are out of range')
        query_ids = _integer_vector(
            result.query_ids, 'class result query IDs')
        if np.any(query_ids < 0) or np.any(query_ids >= query_count):
            raise DiagnosticError('class result query IDs are out of range')
        support_ids = query_ids[:int(result.ap_support_end)]
        np.add.at(support_counts, support_ids, 1)
        support_label_counts[class_id] += len(support_ids)

    rows = []
    for query_id in range(query_count):
        count = int(total_rows[query_id])
        if count:
            scores = np.fromiter(
                (float(record.scores[query_id])
                 for record in bundle.records
                 if query_id < len(record.scores)),
                dtype=np.float64,
                count=count,
            )
            quantiles = np.quantile(scores, [.25, .5, .75])
            score_fields = {
                'score_min': float(scores.min()),
                'score_q25': float(quantiles[0]),
                'score_median': float(quantiles[1]),
                'score_q75': float(quantiles[2]),
                'score_max': float(scores.max()),
            }
        else:
            score_fields = {
                'score_min': None,
                'score_q25': None,
                'score_median': None,
                'score_q75': None,
                'score_max': None,
            }
        rows.append({
            'query_id': query_id,
            'total_rows': count,
            'per_label_counts': [
                int(value) for value in label_counts[query_id]
            ],
            'tp': int(outcome_counts['tp'][query_id]),
            'duplicate_fp': int(
                outcome_counts['duplicate_fp'][query_id]),
            'semantic_fp': int(outcome_counts['semantic_fp'][query_id]),
            'localization_background_fp': int(
                outcome_counts['localization_background_fp'][query_id]),
            'empty_tile_fp': int(
                outcome_counts['empty_tile_fp'][query_id]),
            'ignored_prediction': int(
                outcome_counts['ignored_prediction'][query_id]),
            'reachable_witnesses': int(witness_counts[query_id]),
            'ap_support_appearances': int(support_counts[query_id]),
            **score_fields,
        })

    return {
        'rows': rows,
        'effective_query_count': {
            'tp': effective_query_count(outcome_counts['tp']),
            'reachable_witness': effective_query_count(witness_counts),
            'ap_support': effective_query_count(support_counts),
        },
        'all_row_label_allocation': [
            int(value) for value in label_counts.sum(axis=0)
        ],
        'ap_support_label_allocation': [
            int(value) for value in support_label_counts
        ],
    }


def _validate_strata_protocol(
        protocol: Any, small_vehicle_class_id: Any) -> int:
    """Validate the complete class partition used by fixed strata."""
    if not isinstance(protocol, ClassProtocol):
        raise DiagnosticError('strata protocol must be a ClassProtocol')
    classes = tuple(protocol.classes)
    base = tuple(protocol.base_classes)
    novel = tuple(protocol.novel_classes)
    if (not classes or any(not isinstance(name, str) for name in classes) or
            len(set(classes)) != len(classes)):
        raise DiagnosticError('protocol classes must be present and unique')
    if (any(not isinstance(name, str) for name in base + novel) or
            len(set(base)) != len(base) or
            len(set(novel)) != len(novel) or
            set(base) & set(novel) or
            set(base) | set(novel) != set(classes)):
        raise DiagnosticError(
            'base and novel classes must partition protocol classes')
    if (isinstance(small_vehicle_class_id, (bool, np.bool_)) or
            not isinstance(small_vehicle_class_id, (int, np.integer)) or
            not 0 <= small_vehicle_class_id < len(classes)):
        raise DiagnosticError(
            'small-vehicle class ID must identify protocol small-vehicle')
    small_vehicle_class_id = int(small_vehicle_class_id)
    if classes[small_vehicle_class_id] != 'small-vehicle':
        raise DiagnosticError(
            'small-vehicle class ID must identify protocol small-vehicle')
    return small_vehicle_class_id


def summarize_strata(
        evidence: Sequence[GtEvidence],
        protocol: ClassProtocol,
        small_vehicle_class_id: int) -> List[Dict[str, Any]]:
    """Count four GT states over every predefined fixed stratum."""
    small_vehicle_class_id = _validate_strata_protocol(
        protocol, small_vehicle_class_id)
    states = (
        'geometry_miss',
        'semantic_miss',
        'ownership_miss',
        'evaluator_reachable',
    )

    rows = []

    def add_row(stratum_type: str,
                stratum: str,
                class_id: Optional[int] = None) -> None:
        result = {
            'stratum_type': stratum_type,
            'stratum': stratum,
            'gt_count': 0,
        }
        if class_id is not None:
            result['class_id'] = class_id
        for state in states:
            result[state] = 0
        rows.append(result)

    for class_id, class_name in enumerate(protocol.classes):
        add_row('class', class_name, class_id=class_id)
    for group in ('base', 'novel'):
        add_row('class_group', group)
    for label in SIZE_BINS:
        add_row('size', label)
    for label in DENSITY_BINS:
        add_row('density', label)
    for label in ASPECT_BINS:
        add_row('aspect', label)
    for label in ANGLE_BINS:
        add_row('angle', label)
    for size_label in SIZE_BINS:
        for density_label in DENSITY_BINS:
            add_row(
                'small_vehicle_size_density',
                '{}|{}'.format(size_label, density_label),
            )

    row_by_key = {
        (row['stratum_type'], row['stratum']): row for row in rows
    }
    base_classes = set(protocol.base_classes)
    try:
        evidence_rows = tuple(evidence)
    except TypeError as error:
        raise DiagnosticError('strata evidence must be a sequence') from error

    for evidence_row in evidence_rows:
        if not isinstance(evidence_row, GtEvidence):
            raise DiagnosticError('strata evidence rows must be GtEvidence')
        if (isinstance(evidence_row.class_id, (bool, np.bool_)) or
                not isinstance(evidence_row.class_id, (int, np.integer)) or
                not 0 <= evidence_row.class_id < len(protocol.classes)):
            raise DiagnosticError('strata evidence class ID is out of range')
        if evidence_row.state not in states:
            raise DiagnosticError(
                'unknown GT evidence state: {}'.format(evidence_row.state))
        if (isinstance(evidence_row.image_gt_count, (bool, np.bool_)) or
                not isinstance(evidence_row.image_gt_count,
                               (int, np.integer)) or
                evidence_row.image_gt_count < 1):
            raise DiagnosticError(
                'strata evidence image GT count must be a positive density')
        _validate_gt_evidence_state(evidence_row)
        box = _numeric_vector(evidence_row.gt_box, 'strata GT box')
        if box.shape != (5,):
            raise DiagnosticError('strata GT boxes must have shape (5,)')
        if np.any(box[2:4] <= 0):
            raise DiagnosticError(
                'strata GT box dimensions must be positive')

        class_id = int(evidence_row.class_id)
        class_name = protocol.classes[class_id]
        size_label = size_bin(np.sqrt(box[2] * box[3]))
        density_label = density_bin(evidence_row.image_gt_count)
        aspect_label = aspect_bin(max(box[2:4]) / min(box[2:4]))
        angle_label = angle_bin(box[4])
        group = 'base' if class_name in base_classes else 'novel'
        keys = [
            ('class', class_name),
            ('class_group', group),
            ('size', size_label),
            ('density', density_label),
            ('aspect', aspect_label),
            ('angle', angle_label),
        ]
        if class_id == small_vehicle_class_id:
            keys.append((
                'small_vehicle_size_density',
                '{}|{}'.format(size_label, density_label),
            ))
        for key in keys:
            result = row_by_key[key]
            result['gt_count'] += 1
            result[evidence_row.state] += 1

    for row in rows:
        gt_count = row['gt_count']
        for state in states:
            row['{}_rate'.format(state)] = (
                row[state] / gt_count if gt_count else None)
        row['rate'] = row['evaluator_reachable_rate']
    return rows


def _normalized_mapping_key(value: Any, description: str) -> str:
    """Return a stable JSON object key while detecting lossy collisions."""
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is str:
        return value
    if value is None or type(value) is bool or type(value) is int:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if type(value) is float:
        if not np.isfinite(value):
            raise DiagnosticError('{} contains a non-finite key'.format(
                description))
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise DiagnosticError(
        '{} contains a non-JSON-safe mapping key'.format(description))


_MAX_JSON_DEPTH = 64


def _normalize_json_value(
        value: Any,
        description: str = 'value',
        _active: Optional[set] = None,
        _depth: int = 0) -> Any:
    """Detach recursively into finite JSON-safe Python containers."""
    if _depth > _MAX_JSON_DEPTH:
        raise DiagnosticError(
            '{} exceeds the maximum JSON depth'.format(description))
    if _active is None:
        _active = set()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) is bool or type(value) is str:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise DiagnosticError('{} contains a non-finite value'.format(
                description))
        return value
    is_mapping = isinstance(value, Mapping)
    is_sequence = isinstance(value, (list, tuple))
    if is_mapping or is_sequence:
        identity = id(value)
        if identity in _active:
            raise DiagnosticError(
                '{} contains a cyclic container'.format(description))
        _active.add(identity)
        try:
            if is_sequence:
                return [
                    _normalize_json_value(
                        item, description, _active, _depth + 1)
                    for item in value
                ]
            result = {}
            items = value.items()
            for key, item in items:
                normalized_key = _normalized_mapping_key(key, description)
                if normalized_key in result:
                    raise DiagnosticError(
                        '{} contains colliding mapping keys'.format(
                            description))
                result[normalized_key] = _normalize_json_value(
                    item, description, _active, _depth + 1)
            return result
        except DiagnosticError:
            raise
        except RecursionError as error:
            raise DiagnosticError(
                '{} exceeds the maximum JSON depth'.format(
                    description)) from error
        except Exception as error:
            raise DiagnosticError(
                '{} must be JSON-safe'.format(description)) from error
        finally:
            _active.remove(identity)
    raise DiagnosticError('{} must be JSON-safe'.format(description))


def _mapping_keys(value: Mapping, description: str) -> set:
    try:
        return set(value)
    except DiagnosticError:
        raise
    except Exception as error:
        raise DiagnosticError(
            '{} keys could not be read'.format(description)) from error


def _mapping_value(value: Mapping, key: Any, description: str) -> Any:
    try:
        return value[key]
    except DiagnosticError:
        raise
    except Exception as error:
        raise DiagnosticError(
            '{} field could not be read'.format(description)) from error


def _materialize_iterable(value: Any, description: str) -> List[Any]:
    try:
        return list(value)
    except DiagnosticError:
        raise
    except Exception as error:
        raise DiagnosticError(description) from error


def _normalize_case_gate(case_gate: Any) -> Any:
    """Validate and deterministically order the closed case-gate states.

    The legacy mapping is exactly ``{'status': 'not_run'}``.  Complete Task5
    rows may retain their other JSON-safe evidence fields after the three gate
    identity fields have been validated.
    """
    if isinstance(case_gate, Mapping):
        normalized = _normalize_json_value(
            case_gate, 'diagnostics case_gate')
        if (set(normalized) != {'status'} or
                normalized['status'] != 'not_run'):
            raise DiagnosticError(
                'diagnostics case_gate mapping must equal '
                "{'status': 'not_run'}")
        return normalized
    if type(case_gate) is not list:
        raise DiagnosticError(
            'diagnostics case_gate must be a not_run mapping or Task5 rows')

    normalized = _normalize_json_value(case_gate, 'diagnostics case_gate')
    if len(normalized) == 1:
        row = normalized[0]
        if (type(row) is not dict or
                set(row) != {'case_gate', 'image_id', 'case_group'} or
                row['case_gate'] != 'not_run' or
                row['image_id'] is not None or
                row['case_group'] is not None):
            raise DiagnosticError(
                'diagnostics case_gate not_run rows must match the fixed '
                'Task5 shape')
        return normalized
    if len(normalized) != 2:
        raise DiagnosticError(
            'diagnostics case_gate Task5 rows must contain one not_run row '
            'or both complete rows')

    expected_pairs = tuple(
        (image_id, group)
        for group, image_ids in _CASE_MANIFEST_ORACLE
        for image_id in image_ids
    )
    rows_by_pair = {}
    for row in normalized:
        if (type(row) is not dict or
                row.get('case_gate') != 'complete'):
            raise DiagnosticError(
                'diagnostics case_gate complete rows require exact status')
        pair = (row.get('image_id'), row.get('case_group'))
        if pair in rows_by_pair:
            raise DiagnosticError(
                'diagnostics case_gate contains a duplicate fixed case')
        rows_by_pair[pair] = row
    if set(rows_by_pair) != set(expected_pairs):
        raise DiagnosticError(
            'diagnostics case_gate complete rows must match both fixed cases')
    return [rows_by_pair[pair] for pair in expected_pairs]


def _unit_interval_metric(value: Any, description: str) -> float:
    metric = _finite_real_metric(value, description)
    if not 0.0 <= metric <= 1.0:
        raise DiagnosticError('{} must be in [0, 1]'.format(description))
    return metric


_SUMMARY_ABS_TOLERANCE = 1e-12

_SUMMARY_MAPPING_FIELDS = (
    'integrity',
    'parity',
    'gt_decomposition',
    'capacity',
    'calibration',
    'queries',
)
_SUMMARY_METRIC_FIELDS = (
    'map',
    'ap50',
    'ap70_gap',
    'oracle_map',
    'oracle_headroom',
    'headroom_fraction_to_ap70',
)


def _summary_metric_number(
        metrics: Mapping[str, Any], field: str) -> float:
    value = metrics[field]
    if type(value) not in (int, float) or not np.isfinite(value):
        raise DiagnosticError(
            'diagnostics metric {} must be a finite real'.format(field))
    return float(value)


def _validate_diagnostics_summary(summary: Any) -> Dict[str, Any]:
    """Return a detached summary after validating the report schema."""
    if not isinstance(summary, Mapping):
        raise DiagnosticError('diagnostics summary must be a mapping')
    required_fields = {
        'schema_version', 'canonical', 'metrics', 'fp_regions', 'groups',
        'case_gate', 'warnings', *_SUMMARY_MAPPING_FIELDS,
    }
    if not required_fields <= _mapping_keys(
            summary, 'diagnostics summary'):
        raise DiagnosticError(
            'diagnostics summary is missing required report facts')
    raw = {
        field: _mapping_value(
            summary, field, 'diagnostics summary {}'.format(field))
        for field in required_fields
    }
    if (type(raw['schema_version']) is not int or
            raw['schema_version'] != 1):
        raise DiagnosticError('diagnostics schema_version must equal 1')
    if type(raw['canonical']) is not bool:
        raise DiagnosticError('diagnostics canonical must be a bool')
    for field in _SUMMARY_MAPPING_FIELDS:
        if not isinstance(raw[field], Mapping):
            raise DiagnosticError(
                'diagnostics {} must be a mapping'.format(field))

    metrics = raw['metrics']
    if not isinstance(metrics, Mapping):
        raise DiagnosticError('diagnostics metrics must be a mapping')
    if not set(_SUMMARY_METRIC_FIELDS) <= _mapping_keys(
            metrics, 'diagnostics metrics'):
        raise DiagnosticError(
            'diagnostics metrics are missing required fields')

    fp_regions = raw['fp_regions']
    fp_region_fields = ('all', 'through_last_tp', 'ap_support')
    if not isinstance(fp_regions, Mapping):
        raise DiagnosticError('diagnostics fp_regions must be a mapping')
    if not set(fp_region_fields) <= _mapping_keys(
            fp_regions, 'diagnostics fp_regions'):
        raise DiagnosticError(
            'diagnostics fp_regions are missing required fields')
    if any(not isinstance(_mapping_value(
            fp_regions,
            field,
            'diagnostics fp_regions {}'.format(field)), Mapping)
           for field in fp_region_fields):
        raise DiagnosticError(
            'diagnostics fp_regions entries must be mappings')

    groups = raw['groups']
    group_fields = ('base', 'novel')
    if not isinstance(groups, Mapping):
        raise DiagnosticError('diagnostics groups must be a mapping')
    if not set(group_fields) <= _mapping_keys(groups, 'diagnostics groups'):
        raise DiagnosticError(
            'diagnostics groups are missing required fields')
    if any(not isinstance(_mapping_value(
            groups, field, 'diagnostics group {}'.format(field)), Mapping)
           for field in group_fields):
        raise DiagnosticError('diagnostics group entries must be mappings')

    normalized_case_gate = _normalize_case_gate(raw['case_gate'])
    if type(raw['warnings']) is not list:
        raise DiagnosticError('diagnostics warnings must be a list')

    normalized = _normalize_json_value(summary, 'diagnostics summary')
    normalized['case_gate'] = normalized_case_gate
    normalized_metrics = normalized['metrics']
    reconstructed_map = _summary_metric_number(normalized_metrics, 'map')
    ap50 = _summary_metric_number(normalized_metrics, 'ap50')
    ap70_gap = _summary_metric_number(normalized_metrics, 'ap70_gap')
    oracle_map = _summary_metric_number(normalized_metrics, 'oracle_map')
    oracle_headroom = _summary_metric_number(
        normalized_metrics, 'oracle_headroom')
    if not 0.0 <= reconstructed_map <= 1.0:
        raise DiagnosticError('diagnostics metric map must be in [0, 1]')
    if not 0.0 <= oracle_map <= 1.0:
        raise DiagnosticError(
            'diagnostics metric oracle_map must be in [0, 1]')
    if oracle_map < reconstructed_map:
        if reconstructed_map - oracle_map <= _SUMMARY_ABS_TOLERANCE:
            oracle_map = reconstructed_map
            normalized_metrics['oracle_map'] = normalized_metrics['map']
            normalized_metrics['oracle_headroom'] = 0.0
            normalized_metrics['headroom_fraction_to_ap70'] = None
            oracle_headroom = 0.0
        else:
            raise DiagnosticError(
                'diagnostics oracle metric must not be below map')

    expected = {
        'ap50': round(reconstructed_map, 3),
        'ap70_gap': .70 - reconstructed_map,
        'oracle_headroom': oracle_map - reconstructed_map,
    }
    actual = {
        'ap50': ap50,
        'ap70_gap': ap70_gap,
        'oracle_headroom': oracle_headroom,
    }
    for field, expected_value in expected.items():
        if abs(actual[field] - expected_value) > _SUMMARY_ABS_TOLERANCE:
            raise DiagnosticError(
                'diagnostics metric {} is inconsistent'.format(field))
    if oracle_headroom < 0:
        if abs(oracle_headroom) <= _SUMMARY_ABS_TOLERANCE:
            oracle_headroom = 0.0
            normalized_metrics['oracle_headroom'] = 0.0
        else:
            raise DiagnosticError(
                'diagnostics metric oracle_headroom must not be negative')

    fraction = normalized_metrics['headroom_fraction_to_ap70']
    expected_fraction = (
        expected['ap70_gap'] / expected['oracle_headroom']
        if expected['oracle_headroom'] > 0 else None)
    if expected_fraction is None:
        if fraction is not None:
            raise DiagnosticError(
                'diagnostics metric headroom fraction is inconsistent')
    else:
        fraction_value = _summary_metric_number(
            normalized_metrics, 'headroom_fraction_to_ap70')
        if abs(fraction_value - expected_fraction) > _SUMMARY_ABS_TOLERANCE:
            raise DiagnosticError(
                'diagnostics metric headroom fraction is inconsistent')
    return normalized


def build_diagnostics(
        canonical,
        integrity,
        parity,
        reconstructed_map,
        oracle_map,
        gt_decomposition,
        capacity,
        fp_all,
        fp_last_tp,
        fp_ap_support,
        calibration,
        query_summary,
        base_summary,
        novel_summary,
        case_gate,
        warnings):
    """Build a detached, finite-only Q600 diagnostic summary."""
    if type(canonical) is not bool:
        raise DiagnosticError('canonical must be a bool')
    reconstructed_map = _unit_interval_metric(
        reconstructed_map, 'reconstructed mAP')
    oracle_map = _unit_interval_metric(oracle_map, 'oracle mAP')
    if oracle_map < reconstructed_map:
        if reconstructed_map - oracle_map <= _SUMMARY_ABS_TOLERANCE:
            oracle_map = reconstructed_map
        else:
            raise DiagnosticError(
                'oracle mAP must not be below reconstructed mAP')
    if isinstance(warnings, (str, bytes, bytearray)):
        raise DiagnosticError('warnings must be a non-string iterable')
    warning_rows = _materialize_iterable(
        warnings, 'warnings must be a non-string iterable')

    ap70_gap = .70 - reconstructed_map
    oracle_headroom = oracle_map - reconstructed_map
    result = {
        'schema_version': 1,
        'canonical': canonical,
        'integrity': integrity,
        'parity': parity,
        'metrics': {
            'map': reconstructed_map,
            'ap50': round(reconstructed_map, 3),
            'ap70_gap': ap70_gap,
            'oracle_map': oracle_map,
            'oracle_headroom': oracle_headroom,
            'headroom_fraction_to_ap70': (
                ap70_gap / oracle_headroom
                if oracle_headroom > 0 else None),
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
        'warnings': warning_rows,
    }
    return _validate_diagnostics_summary(result)


_ERROR_OUTCOMES = (
    'duplicate_fp',
    'semantic_fp',
    'localization_background_fp',
    'empty_tile_fp',
)


class _WorstFirstError:
    """Heap item whose root is the worst retained deterministic row."""

    __slots__ = ('key', 'row')

    def __init__(self, key: Tuple[Any, ...], row: Dict[str, Any]):
        self.key = key
        self.row = row

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, _WorstFirstError):
            return NotImplemented
        return self.key > other.key


def cap_error_examples(examples, max_per_type):
    """Return detached top-scoring false outcomes under a per-type cap.

    Streaming duplicate detection is intentionally limited to fixed sort keys
    that can affect the retained witness set, keeping memory at O(K).  Callers
    with a final bounded table can pass ``K=len(rows)`` for an exact gate.
    """
    if (isinstance(max_per_type, (bool, np.bool_)) or
            not isinstance(max_per_type, (int, np.integer)) or
            max_per_type < 0):
        raise DiagnosticError(
            'max_per_type must be a nonnegative integer')
    max_per_type = int(max_per_type)
    if isinstance(examples, (str, bytes, bytearray)):
        raise DiagnosticError('error examples must be an iterable of rows')
    try:
        iterator = iter(examples)
    except DiagnosticError:
        raise
    except Exception as error:
        raise DiagnosticError(
            'error examples must be an iterable of rows') from error

    heaps = {outcome: [] for outcome in _ERROR_OUTCOMES}
    selected_keys = {outcome: set() for outcome in _ERROR_OUTCOMES}
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except DiagnosticError:
            raise
        except Exception as error:
            raise DiagnosticError(
                'error example iteration failed') from error
        if not isinstance(row, Mapping):
            raise DiagnosticError('each error example must be a mapping')
        try:
            outcome = row['outcome']
            img_id = row['img_id']
            query_id = row['query_id']
            score = row['score']
        except DiagnosticError:
            raise
        except Exception as error:
            raise DiagnosticError(
                'each error example must contain outcome, img_id, '
                'query_id, and score') from error
        if type(outcome) is not str or outcome not in _ERROR_OUTCOMES:
            raise DiagnosticError(
                'error example outcome must be a known false outcome')
        if type(img_id) is not str:
            raise DiagnosticError(
                'error example img_id must be a stable string')
        if (isinstance(query_id, (bool, np.bool_)) or
                not isinstance(query_id, (int, np.integer)) or
                query_id < 0):
            raise DiagnosticError(
                'error example query_id must be a nonnegative integer')
        score = _finite_real_metric(score, 'error example score')
        detached = _normalize_json_value(row, 'error example')
        detached['outcome'] = outcome
        detached['img_id'] = img_id
        detached['query_id'] = int(query_id)
        detached['score'] = score
        sort_key = (-score, img_id, int(query_id))
        outcome_keys = selected_keys[outcome]
        if sort_key in outcome_keys:
            raise DiagnosticError(
                'error examples contain a duplicate fixed sort key')
        if max_per_type == 0:
            continue
        heap = heaps[outcome]
        item = _WorstFirstError(sort_key, detached)
        if len(heap) < max_per_type:
            heapq.heappush(heap, item)
            outcome_keys.add(sort_key)
        elif sort_key < heap[0].key:
            evicted = heapq.heapreplace(heap, item)
            outcome_keys.remove(evicted.key)
            outcome_keys.add(sort_key)

    result = []
    for outcome in _ERROR_OUTCOMES:
        retained = sorted(heaps[outcome], key=lambda item: item.key)
        result.extend(item.row for item in retained)
    return result


CSV_SCHEMAS = {
    'per_class.csv': (
        'class_id', 'class_name', 'group', 'num_gts', 'prediction_rows',
        'tp_count', 'fp_count', 'recall', 'ap', 'oracle_ap',
        'oracle_headroom', 'last_tp_end', 'ap_support_end',
        'geometry_miss', 'semantic_miss', 'ownership_miss',
        'evaluator_reachable', 'all_tp', 'all_duplicate_fp',
        'all_semantic_fp', 'all_localization_background_fp',
        'all_empty_tile_fp', 'all_ignored_prediction',
        'through_last_tp_tp', 'through_last_tp_duplicate_fp',
        'through_last_tp_semantic_fp',
        'through_last_tp_localization_background_fp',
        'through_last_tp_empty_tile_fp',
        'through_last_tp_ignored_prediction', 'ap_support_tp',
        'ap_support_duplicate_fp', 'ap_support_semantic_fp',
        'ap_support_localization_background_fp',
        'ap_support_empty_tile_fp', 'ap_support_ignored_prediction',
        'tp_score_iou_spearman',
    ),
    'strata.csv': (
        'stratum_type', 'stratum', 'class_id', 'gt_count',
        'geometry_miss', 'semantic_miss', 'ownership_miss',
        'evaluator_reachable', 'geometry_miss_rate',
        'semantic_miss_rate', 'ownership_miss_rate',
        'evaluator_reachable_rate', 'rate',
    ),
    'per_query.csv': (
        'query_id', 'total_rows', 'per_label_counts', 'tp',
        'duplicate_fp', 'semantic_fp', 'localization_background_fp',
        'empty_tile_fp', 'ignored_prediction', 'reachable_witnesses',
        'ap_support_appearances', 'score_min', 'score_q25',
        'score_median', 'score_q75', 'score_max',
    ),
    'case_studies.csv': (
        'case_gate', 'image_id', 'case_group', 'gt_count',
        'small_vehicle_gt_count', 'predicted_small_vehicle_rows',
        'top1_small_vehicle_scores', 'top5_small_vehicle_scores',
        'top20_small_vehicle_scores', 'small_vehicle_ap_support_count',
        'small_vehicle_tp_count', 'small_vehicle_duplicate_fp_count',
        'small_vehicle_semantic_fp_count',
        'small_vehicle_localization_background_fp_count',
        'small_vehicle_empty_tile_fp_count',
        'small_vehicle_ignored_prediction_count',
        'small_vehicle_geometry_miss_count',
        'small_vehicle_semantic_miss_count',
        'small_vehicle_ownership_miss_count',
        'small_vehicle_evaluator_reachable_count',
        'small_vehicle_gt_witnesses',
        'highest_score_false_small_vehicle_witness',
    ),
    'error_examples.csv': (
        'outcome', 'img_id', 'query_id', 'score', 'class_id',
        'matched_gt', 'assigned_iou', 'same_label_iou',
        'nearest_gt_index', 'nearest_gt_label', 'nearest_gt_iou',
    ),
}
NON_MANIFEST_PAYLOADS = (
    'diagnostics.json',
    'per_class.csv',
    'strata.csv',
    'per_query.csv',
    'case_studies.csv',
    'error_examples.csv',
    'report.md',
)


def _json_bytes(value: Any) -> bytes:
    normalized = _normalize_json_value(value, 'JSON payload')
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DiagnosticError('JSON payload must be serializable') from error
    return (text + '\n').encode('utf-8')


def _nonnegative_row_id(value: Any, description: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < 0):
        raise DiagnosticError(
            '{} must be a nonnegative integer'.format(description))
    return int(value)


def _normalize_table_rows(table_name: str, value: Any) -> List[dict]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise DiagnosticError('{} must be an iterable of rows'.format(
            table_name))
    input_rows = _materialize_iterable(
        value, '{} must be an iterable of rows'.format(table_name))
    allowed = set(CSV_SCHEMAS[table_name])
    rows = []
    for row in input_rows:
        if not isinstance(row, Mapping):
            raise DiagnosticError(
                '{} rows must be mappings'.format(table_name))
        keys = _mapping_keys(row, '{} row'.format(table_name))
        if any(type(key) is not str for key in keys) or not keys <= allowed:
            raise DiagnosticError(
                '{} row contains fields outside its fixed schema'.format(
                    table_name))
        normalized = _normalize_json_value(row, '{} row'.format(table_name))
        rows.append(normalized)
    return rows


def _reject_duplicate_keys(keys: Sequence[Any], table_name: str) -> None:
    if len(set(keys)) != len(keys):
        raise DiagnosticError(
            '{} contains a duplicate sort key'.format(table_name))


def _sort_id_table(
        rows: List[dict], field: str, table_name: str) -> List[dict]:
    keys = []
    for row in rows:
        if field not in row:
            raise DiagnosticError(
                '{} rows require {}'.format(table_name, field))
        key = _nonnegative_row_id(row[field], '{} {}'.format(
            table_name, field))
        row[field] = key
        keys.append(key)
    _reject_duplicate_keys(keys, table_name)
    return sorted(rows, key=lambda row: row[field])


def _strata_sort_key(row: dict) -> Tuple[int, int]:
    try:
        stratum_type = row['stratum_type']
        stratum = row['stratum']
    except KeyError as error:
        raise DiagnosticError(
            'strata.csv rows require stratum_type and stratum') from error
    if type(stratum_type) is not str or type(stratum) is not str:
        raise DiagnosticError('strata.csv strata must be stable strings')
    if stratum_type == 'class':
        if 'class_id' not in row:
            raise DiagnosticError('class strata require class_id')
        class_id = _nonnegative_row_id(
            row['class_id'], 'strata.csv class_id')
        row['class_id'] = class_id
        return (0, class_id)
    fixed = {
        'class_group': ('base', 'novel'),
        'size': SIZE_BINS,
        'density': DENSITY_BINS,
        'aspect': ASPECT_BINS,
        'angle': ANGLE_BINS,
        'small_vehicle_size_density': tuple(
            '{}|{}'.format(size, density)
            for size in SIZE_BINS for density in DENSITY_BINS),
    }
    type_order = {
        'class_group': 1,
        'size': 2,
        'density': 3,
        'aspect': 4,
        'angle': 5,
        'small_vehicle_size_density': 6,
    }
    try:
        stratum_index = fixed[stratum_type].index(stratum)
        return (type_order[stratum_type], stratum_index)
    except (KeyError, ValueError) as error:
        raise DiagnosticError(
            'strata.csv contains an unknown stratum') from error


def _sort_strata(rows: List[dict]) -> List[dict]:
    keys = [_strata_sort_key(row) for row in rows]
    _reject_duplicate_keys(keys, 'strata.csv')
    return [row for _, row in sorted(zip(keys, rows), key=lambda item: item[0])]


def _sort_cases(rows: List[dict]) -> List[dict]:
    return _normalize_case_gate(rows)


def _sort_error_rows(rows: List[dict]) -> List[dict]:
    return cap_error_examples(rows, max_per_type=len(rows))


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DiagnosticError('CSV cell must be JSON-safe') from error


def _csv_bytes(table_name: str, rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_SCHEMAS[table_name],
        lineterminator='\n',
    )
    writer.writeheader()
    for row in rows:
        rendered = {}
        for field in CSV_SCHEMAS[table_name]:
            value = row.get(field)
            rendered[field] = (
                _compact_json(value)
                if isinstance(value, (list, dict, tuple)) else value)
        try:
            writer.writerow(rendered)
        except (csv.Error, TypeError, ValueError) as error:
            raise DiagnosticError(
                '{} row could not be serialized'.format(table_name)) from error
    return stream.getvalue().encode('utf-8')


def _render_report(summary: Mapping[str, Any]) -> bytes:
    def snippet(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )

    calibration = {
        field: summary['calibration'][field]
        for field in (
            'sample_size_requested',
            'sample_size_used',
            'tp_score_iou_spearman',
            'per_class',
            'score_bins',
        )
        if field in summary['calibration']
    }
    queries = {
        field: summary['queries'][field]
        for field in (
            'effective_query_count',
            'all_row_label_allocation',
            'ap_support_label_allocation',
        )
        if field in summary['queries']
    }

    sections = [
        (
            '## Integrity and parity',
            {
                'canonical': summary['canonical'],
                'integrity': summary['integrity'],
                'parity': summary['parity'],
            },
        ),
        ('## Raw metrics/AP70 gap', summary['metrics']),
        (
            '## Macro AP-support/ranking',
            {
                'fp_regions': summary['fp_regions'],
                'calibration': calibration,
            },
        ),
        (
            '## Micro reachability/capacity',
            {
                'gt_decomposition': summary['gt_decomposition'],
                'capacity': summary['capacity'],
                'queries': queries,
            },
        ),
        (
            '## Base/novel and optional cases',
            {
                'groups': summary['groups'],
                'case_gate': summary['case_gate'],
            },
        ),
        (
            '## Decision inputs',
            {
                'schema_version': summary['schema_version'],
                'canonical': summary['canonical'],
                'warnings': summary['warnings'],
            },
        ),
    ]
    blocks = ['# Q600 Diagnostic Report']
    for heading, facts in sections:
        blocks.append('{}\n\n```json\n{}\n```'.format(
            heading, snippet(facts)))
    return ('\n\n'.join(blocks) + '\n').encode('utf-8')


def render_payloads(summary, tables):
    """Render the exact seven deterministic non-manifest payloads."""
    normalized_summary = _validate_diagnostics_summary(summary)
    if not isinstance(tables, Mapping):
        raise DiagnosticError('tables must be a mapping')
    table_names = _mapping_keys(tables, 'tables')
    if (any(type(name) is not str for name in table_names) or
            table_names != set(CSV_SCHEMAS)):
        raise DiagnosticError('tables must contain exactly the five CSV names')

    normalized_tables = {}
    for name in CSV_SCHEMAS:
        table_rows = _mapping_value(
            tables, name, 'tables {}'.format(name))
        normalized_tables[name] = _normalize_table_rows(name, table_rows)
    normalized_tables['per_class.csv'] = _sort_id_table(
        normalized_tables['per_class.csv'], 'class_id', 'per_class.csv')
    normalized_tables['strata.csv'] = _sort_strata(
        normalized_tables['strata.csv'])
    normalized_tables['per_query.csv'] = _sort_id_table(
        normalized_tables['per_query.csv'], 'query_id', 'per_query.csv')
    normalized_tables['case_studies.csv'] = _sort_cases(
        normalized_tables['case_studies.csv'])
    normalized_tables['error_examples.csv'] = _sort_error_rows(
        normalized_tables['error_examples.csv'])

    payloads = {'diagnostics.json': _json_bytes(normalized_summary)}
    for name in CSV_SCHEMAS:
        payloads[name] = _csv_bytes(name, normalized_tables[name])
    payloads['report.md'] = _render_report(normalized_summary)
    if tuple(payloads) != NON_MANIFEST_PAYLOADS:
        raise DiagnosticError('non-manifest payload set is inconsistent')
    return payloads


def build_manifest(payloads, provenance, command, environment):
    """Hash exactly the seven payloads and return a non-self-hashing manifest."""
    if not isinstance(payloads, Mapping):
        raise DiagnosticError('manifest payloads must be a mapping')
    names = _mapping_keys(payloads, 'manifest payloads')
    if (any(type(name) is not str for name in names) or
            names != set(NON_MANIFEST_PAYLOADS)):
        raise DiagnosticError(
            'manifest requires exactly the seven non-manifest payloads')

    files = {}
    for name in sorted(names):
        content = _mapping_value(
            payloads, name, 'manifest payload {}'.format(name))
        if type(content) is not bytes:
            raise DiagnosticError(
                'manifest payload contents must be bytes')
        files[name] = {
            'sha256': hashlib.sha256(content).hexdigest(),
            'bytes': len(content),
        }
    manifest = {
        'schema_version': 1,
        'files': files,
        'provenance': _normalize_json_value(provenance, 'provenance'),
        'command': _normalize_json_value(command, 'command'),
        'environment': _normalize_json_value(environment, 'environment'),
    }
    return _json_bytes(manifest)
