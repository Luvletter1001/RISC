#!/usr/bin/env python3
"""Run CPU-only candidate-count and shuffled-center controls on Dense400."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from dotav2_q600_diagnostics import (  # noqa: E402
    PreparedRecord,
    _rotated_iou,
    prepare_records,
)
from validate_dotav2_q600_dump import as_tensor, load_cpu  # noqa: E402


CANONICAL_ANCHORS = {
    'gt': 123888,
    'geometry_miss': 61068,
    'local_center_rescue': 58652,
    'baseline_matching': 62102,
    'center_matching': 99636,
}


class ControlError(RuntimeError):
    """Raised when a control input or canonical invariant is invalid."""


def nested_query_subsets(query_count: int,
                         counts: Sequence[int],
                         seed: int):
    """Return deterministic nested random query-ID subsets."""
    if type(query_count) is not int or query_count <= 0:
        raise ControlError('query count must be a positive integer')
    counts = tuple(counts)
    if (not counts or any(type(value) is not int or
                          not 0 < value <= query_count for value in counts)):
        raise ControlError('candidate query counts must be in [1, Q]')
    if len(set(counts)) != len(counts):
        raise ControlError('candidate query counts must be unique')
    permutation = np.random.default_rng(seed).permutation(query_count)
    return {
        count: permutation[:count].copy()
        for count in sorted(counts)
    }


def cyclic_same_class_donors(labels: np.ndarray, seed: int) -> np.ndarray:
    """Assign a different same-class GT as donor using seeded cycles."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ControlError('GT labels must be one-dimensional')
    donors = np.full(len(labels), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            continue
        cycle = rng.permutation(indices)
        donors[cycle] = np.roll(cycle, 1)
    return donors


def _matching_count(edges: np.ndarray) -> int:
    edges = np.asarray(edges, dtype=np.bool_)
    if edges.ndim != 2:
        raise ControlError('matching edges must be a matrix')
    if not edges.size:
        return 0
    matching = maximum_bipartite_matching(
        csr_matrix(edges), perm_type='column')
    return int(np.count_nonzero(matching >= 0))


def _record_edges(record: PreparedRecord, iou_threshold: float):
    """Build baseline and target-center edge matrices for one image."""
    if not isinstance(record, PreparedRecord):
        raise ControlError('records must be prepared records')
    query_count = len(record.boxes)
    gt_count = len(record.gt_boxes)
    baseline_iou = _rotated_iou(record.boxes, record.gt_boxes)
    if gt_count:
        geometry_miss = baseline_iou.max(axis=0) < iou_threshold
    else:
        geometry_miss = np.zeros(0, dtype=np.bool_)
    same_label = record.labels[:, None] == record.gt_labels[None, :]
    baseline = (same_label & (baseline_iou >= iou_threshold)).T
    center = baseline.copy()

    for gt_index in np.flatnonzero(geometry_miss):
        query_ids = np.flatnonzero(
            record.labels == record.gt_labels[gt_index])
        if not len(query_ids):
            continue
        boxes = record.boxes[query_ids].copy()
        boxes[:, :2] = record.gt_boxes[gt_index, :2]
        overlaps = _rotated_iou(
            boxes, record.gt_boxes[gt_index:gt_index + 1])[:, 0]
        center[gt_index, query_ids] = overlaps >= iou_threshold

    if baseline.shape != (gt_count, query_count):
        raise ControlError('internal edge shape mismatch')
    return {
        'record': record,
        'geometry_miss': geometry_miss,
        'baseline': baseline,
        'center': center,
    }


def _summary(values):
    values = np.asarray(tuple(values), dtype=np.float64)
    if not len(values):
        raise ControlError('cannot summarize empty runs')
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': int(values.min()),
        'max': int(values.max()),
    }


def _local_candidate_curve(images, counts, seeds):
    output = {}
    for count in counts:
        runs = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rescued = 0
            for image in images:
                record = image['record']
                for gt_index in np.flatnonzero(image['geometry_miss']):
                    candidates = np.flatnonzero(
                        record.labels == record.gt_labels[gt_index])
                    if not len(candidates):
                        continue
                    if count < len(candidates):
                        candidates = rng.choice(
                            candidates, size=count, replace=False)
                    if image['center'][gt_index, candidates].any():
                        rescued += 1
            runs.append({'seed': int(seed), 'rescued': int(rescued)})
        output[str(count)] = {
            'runs': runs,
            'summary': _summary(row['rescued'] for row in runs),
        }
    return output


def _matching_curve(images, counts, seeds, query_count):
    output = {str(count): {'runs': []} for count in counts}
    for seed in seeds:
        subsets = nested_query_subsets(query_count, counts, seed)
        for count, query_ids in subsets.items():
            baseline = 0
            center = 0
            for image in images:
                baseline += _matching_count(
                    image['baseline'][:, query_ids])
                center += _matching_count(image['center'][:, query_ids])
            output[str(count)]['runs'].append({
                'seed': int(seed),
                'baseline_matching': int(baseline),
                'center_matching': int(center),
            })
    for payload in output.values():
        payload['baseline_summary'] = _summary(
            row['baseline_matching'] for row in payload['runs'])
        payload['center_summary'] = _summary(
            row['center_matching'] for row in payload['runs'])
    return output


def _placebo_edges(image, donors, iou_threshold):
    record = image['record']
    placebo = image['baseline'].copy()
    eligible = image['geometry_miss'] & (donors >= 0)
    for gt_index in np.flatnonzero(eligible):
        query_ids = np.flatnonzero(
            record.labels == record.gt_labels[gt_index])
        if not len(query_ids):
            continue
        boxes = record.boxes[query_ids].copy()
        boxes[:, :2] = record.gt_boxes[donors[gt_index], :2]
        overlaps = _rotated_iou(
            boxes, record.gt_boxes[gt_index:gt_index + 1])[:, 0]
        placebo[gt_index, query_ids] = overlaps >= iou_threshold
    return eligible, placebo


def _placebo_control(images, seeds, iou_threshold):
    runs = []
    eligible_total = None
    true_local_total = None
    true_matching_total = None
    for seed in seeds:
        eligible_count = 0
        true_local = 0
        true_matching = 0
        placebo_local = 0
        placebo_matching = 0
        for image_index, image in enumerate(images):
            donors = cyclic_same_class_donors(
                image['record'].gt_labels,
                seed=int(seed) + image_index)
            eligible, placebo = _placebo_edges(
                image, donors, iou_threshold)
            eligible_count += int(eligible.sum())
            true_local += int(np.count_nonzero(
                image['center'][eligible].any(axis=1)))
            placebo_local += int(np.count_nonzero(
                placebo[eligible].any(axis=1)))

            eligible_center = image['baseline'].copy()
            eligible_center[eligible] = image['center'][eligible]
            true_matching += _matching_count(eligible_center)
            placebo_matching += _matching_count(placebo)

        if eligible_total is None:
            eligible_total = eligible_count
            true_local_total = true_local
            true_matching_total = true_matching
        elif (eligible_count != eligible_total or
              true_local != true_local_total or
              true_matching != true_matching_total):
            raise ControlError('placebo eligibility changed across seeds')
        runs.append({
            'seed': int(seed),
            'local_rescue': int(placebo_local),
            'matching': int(placebo_matching),
        })
    return {
        'eligibility': 'same-image same-class GT groups with at least 2 GT',
        'eligible_geometry_miss': int(eligible_total or 0),
        'true_center_local_rescue': int(true_local_total or 0),
        'true_center_matching': int(true_matching_total or 0),
        'runs': runs,
        'local_summary': _summary(row['local_rescue'] for row in runs),
        'matching_summary': _summary(row['matching'] for row in runs),
    }


def analyze_records(records: Sequence[PreparedRecord],
                    query_counts: Sequence[int],
                    local_candidate_counts: Sequence[int],
                    seeds: Sequence[int],
                    iou_threshold: float = 0.5):
    """Analyze prepared records without model loading, inference, or GPU."""
    records = tuple(records)
    seeds = tuple(seeds)
    query_counts = tuple(query_counts)
    local_candidate_counts = tuple(local_candidate_counts)
    if not records:
        raise ControlError('at least one prepared record is required')
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ControlError('seeds must be nonempty integers')
    if (type(iou_threshold) not in (int, float) or
            not 0.0 <= iou_threshold <= 1.0):
        raise ControlError('IoU threshold must be in [0, 1]')
    query_count = len(records[0].boxes)
    if any(len(record.boxes) != query_count for record in records):
        raise ControlError('all records must have the same query count')
    nested_query_subsets(query_count, query_counts, seeds[0])
    if (not local_candidate_counts or
            any(type(value) is not int or value <= 0
                for value in local_candidate_counts)):
        raise ControlError('local candidate counts must be positive integers')

    images = tuple(_record_edges(record, iou_threshold) for record in records)
    anchors = {
        'gt': int(sum(len(image['record'].gt_boxes) for image in images)),
        'geometry_miss': int(sum(
            image['geometry_miss'].sum() for image in images)),
        'local_center_rescue': int(sum(np.count_nonzero(
            image['center'][image['geometry_miss']].any(axis=1))
            for image in images)),
        'baseline_matching': int(sum(
            _matching_count(image['baseline']) for image in images)),
        'center_matching': int(sum(
            _matching_count(image['center']) for image in images)),
    }
    local_curve = _local_candidate_curve(
        images, local_candidate_counts, seeds)
    local_curve['all'] = {
        'rescued': anchors['local_center_rescue'],
    }
    return {
        'anchors': anchors,
        'candidate_matching': _matching_curve(
            images, query_counts, seeds, query_count),
        'local_candidate_rescue': local_curve,
        'placebo': _placebo_control(images, seeds, iou_threshold),
    }


def require_anchors(actual: Mapping, expected: Mapping) -> None:
    """Fail closed unless every expected canonical anchor is exact."""
    mismatches = {
        key: {'expected': value, 'actual': actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ControlError('canonical anchor mismatch: {}'.format(mismatches))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_dense400(dump_path: Path,
                  manifest_path: Path,
                  queries_per_image: int = 600,
                  num_classes: int = 18):
    """Load the dump once and prepare only manifest-selected records."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ControlError('failed to load selection manifest') from error
    if not isinstance(manifest, Mapping):
        raise ControlError('selection manifest must be an object')
    entries = manifest.get('records')
    if not isinstance(entries, list) or not entries:
        raise ControlError('selection manifest records must be nonempty')
    if manifest.get('selected_total') != len(entries):
        raise ControlError('selection manifest total mismatch')

    raw = load_cpu(dump_path)
    selected = []
    seen_indices = set()
    seen_ids = set()
    gt_total = 0
    try:
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ControlError(
                    'selection manifest entry must be an object')
            index = entry.get('dataset_index')
            image_id = entry.get('img_id')
            gt_count = entry.get('gt_count')
            if (type(index) is not int or not 0 <= index < len(raw) or
                    type(gt_count) is not int or gt_count < 0):
                raise ControlError(
                    'invalid selection manifest index or GT count')
            if index in seen_indices or image_id in seen_ids:
                raise ControlError('duplicate selection manifest identity')
            record = raw[index]
            if record.get('img_id') != image_id:
                raise ControlError(
                    'selection manifest image identity mismatch')
            actual_gt = len(as_tensor(record['gt_instances']['bboxes']))
            if actual_gt != gt_count:
                raise ControlError('selection manifest GT count mismatch')
            seen_indices.add(index)
            seen_ids.add(image_id)
            selected.append(record)
            gt_total += gt_count
        if manifest.get('selected_gt_total') != gt_total:
            raise ControlError('selection manifest GT total mismatch')
        return prepare_records(
            selected,
            queries_per_image=queries_per_image,
            num_classes=num_classes), manifest
    finally:
        del raw


def _parse_counts(value: str):
    try:
        counts = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'counts must be comma-separated integers') from error
    if not counts:
        raise argparse.ArgumentTypeError('counts must not be empty')
    return counts


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-counts', type=_parse_counts,
                        default=(100, 200, 400, 600))
    parser.add_argument('--local-candidate-counts', type=_parse_counts,
                        default=(1, 5, 10, 25, 50, 100))
    parser.add_argument('--seed-base', type=int, default=20260723)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    return parser


def _publish_json(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ControlError('output already exists: {}'.format(path))
    encoded = (json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + '\n').encode('utf-8')
    descriptor, temporary = tempfile.mkstemp(
        prefix='.{}.'.format(path.name), suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ControlError('CUDA_VISIBLE_DEVICES must be empty')
    if type(args.num_seeds) is not int or args.num_seeds <= 0:
        raise ControlError('number of seeds must be positive')
    torch.set_num_threads(1)
    records, manifest = load_dense400(args.dump, args.manifest)
    seeds = tuple(args.seed_base + offset for offset in range(args.num_seeds))
    controls = analyze_records(
        records,
        query_counts=args.query_counts,
        local_candidate_counts=args.local_candidate_counts,
        seeds=seeds,
        iou_threshold=args.iou_threshold,
    )
    require_anchors(controls['anchors'], CANONICAL_ANCHORS)
    payload = {
        'schema_version': 1,
        'status': 'cpu_only_gt_assisted_controls_not_ap_or_causal',
        'inputs': {
            'dump': str(args.dump),
            'dump_sha256': _file_sha256(args.dump),
            'manifest': str(args.manifest),
            'manifest_sha256': _file_sha256(args.manifest),
            'selection_content_sha256': manifest.get(
                'selection_content_sha256'),
            'queries_per_image': 600,
            'iou_threshold': args.iou_threshold,
            'seeds': list(seeds),
        },
        'definitions': {
            'candidate_matching': 'Nested globally sampled query IDs; '
                                  'maximum same-label bipartite matching.',
            'local_candidate_rescue': 'Per-GT random same-label candidates '
                                      'under target-center replacement.',
            'placebo': 'Within-image, within-class cyclically deranged GT '
                       'center donor evaluated against the original GT.',
        },
        'limitations': [
            'GT-assisted edge replacements are not deployable predictions.',
            'Matching ignores score order and is not AP.',
            'Candidate deletion does not emulate training a smaller-query '
            'model.',
            'A shuffled-center negative control does not establish causality '
            'or learnability.',
            'Results are limited to Dense400, one E24 checkpoint, and IoU '
            '0.5.',
        ],
        'controls': controls,
    }
    _publish_json(args.output, payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = run(args)
    except (ControlError, MemoryError, OSError) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        return 2
    print(json.dumps({
        'output': str(args.output),
        'anchors': payload['controls']['anchors'],
        'output_sha256': _file_sha256(args.output),
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
