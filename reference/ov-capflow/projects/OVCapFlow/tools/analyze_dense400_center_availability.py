#!/usr/bin/env python3
"""Measure query-center availability for Dense400 geometry misses on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_dense400_center_controls import (  # noqa: E402
    ControlError,
    PreparedRecord,
    _file_sha256,
    _matching_count,
    _publish_json,
    _record_edges,
    load_dense400,
    prepare_records,
    require_anchors,
)


CANONICAL_ANCHORS = {'geometry_miss': 61068}
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def normalized_center_distances(query_centers: np.ndarray,
                                gt_boxes: np.ndarray) -> np.ndarray:
    """Return GT-local normalized Chebyshev center distances, shape GxQ."""
    query_centers = np.asarray(query_centers, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    if query_centers.ndim != 2 or query_centers.shape[1:] != (2,):
        raise ControlError('query centers must have shape (Q, 2)')
    if gt_boxes.ndim != 2 or gt_boxes.shape[1:] != (5,):
        raise ControlError('GT boxes must have shape (G, 5)')
    if (not np.isfinite(query_centers).all() or
            not np.isfinite(gt_boxes).all()):
        raise ControlError('center availability inputs must be finite')
    if np.any(gt_boxes[:, 2:4] <= 0):
        raise ControlError('GT widths and heights must be positive')

    delta = query_centers[None, :, :] - gt_boxes[:, None, :2]
    cosine = np.cos(gt_boxes[:, 4])[:, None]
    sine = np.sin(gt_boxes[:, 4])[:, None]
    local_x = cosine * delta[:, :, 0] + sine * delta[:, :, 1]
    local_y = -sine * delta[:, :, 0] + cosine * delta[:, :, 1]
    normalized_x = 2.0 * np.abs(local_x) / gt_boxes[:, None, 2]
    normalized_y = 2.0 * np.abs(local_y) / gt_boxes[:, None, 3]
    return np.maximum(normalized_x, normalized_y)


def _quantile_summary(values: Sequence[float]):
    values = np.asarray(tuple(values), dtype=np.float64)
    finite = values[np.isfinite(values)]
    quantiles = {}
    if len(finite):
        measured = np.quantile(finite, QUANTILES)
        quantiles = {
            'p{}'.format(int(round(fraction * 100))): float(value)
            for fraction, value in zip(QUANTILES, measured)
        }
    return {
        'finite': int(len(finite)),
        'missing_candidates': int(len(values) - len(finite)),
        'quantiles': quantiles,
    }


def _rate(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


def analyze_records(records: Sequence[PreparedRecord],
                    iou_threshold: float = 0.5):
    """Analyze center availability only for canonical geometry-miss GT."""
    records = tuple(records)
    if not records:
        raise ControlError('at least one prepared record is required')

    geometry_total = 0
    any_available_total = 0
    same_available_total = 0
    all_matching_total = 0
    same_matching_total = 0
    min_any_values = []
    min_same_values = []
    per_image = []

    for record in records:
        image = _record_edges(record, iou_threshold)
        gt_ids = np.flatnonzero(image['geometry_miss'])
        geometry_count = len(gt_ids)
        if geometry_count:
            distances = normalized_center_distances(
                record.boxes[:, :2], record.gt_boxes[gt_ids])
            inside = distances <= 1.0
            same_label = (
                record.gt_labels[gt_ids, None] == record.labels[None, :])
            same_inside = inside & same_label
            any_available = int(np.count_nonzero(inside.any(axis=1)))
            same_available = int(np.count_nonzero(
                same_inside.any(axis=1)))
            all_matching = _matching_count(inside)
            same_matching = _matching_count(same_inside)
            min_any = distances.min(axis=1)
            min_same = np.where(
                same_label, distances, np.inf).min(axis=1)
            min_any_values.extend(min_any.tolist())
            min_same_values.extend(min_same.tolist())
        else:
            any_available = 0
            same_available = 0
            all_matching = 0
            same_matching = 0

        geometry_total += geometry_count
        any_available_total += any_available
        same_available_total += same_available
        all_matching_total += all_matching
        same_matching_total += same_matching
        per_image.append({
            'img_id': str(record.img_id),
            'geometry_miss': int(geometry_count),
            'any_label_available': int(any_available),
            'same_label_available': int(same_available),
            'all_label_matching': int(all_matching),
            'same_label_matching': int(same_matching),
        })

    return {
        'anchors': {'geometry_miss': int(geometry_total)},
        'availability': {
            'any_label': {
                'available': int(any_available_total),
                'rate': _rate(any_available_total, geometry_total),
            },
            'same_label': {
                'available': int(same_available_total),
                'rate': _rate(same_available_total, geometry_total),
            },
        },
        'matching': {
            'all_label': int(all_matching_total),
            'all_label_rate': _rate(all_matching_total, geometry_total),
            'same_label': int(same_matching_total),
            'same_label_rate': _rate(same_matching_total, geometry_total),
        },
        'min_normalized_chebyshev': {
            'any_label': _quantile_summary(min_any_values),
            'same_label': _quantile_summary(min_same_values),
        },
        'per_image': per_image,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    return parser


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ControlError('CUDA_VISIBLE_DEVICES must be empty')
    torch.set_num_threads(1)
    records, manifest = load_dense400(args.dump, args.manifest)
    controls = analyze_records(records, iou_threshold=args.iou_threshold)
    require_anchors(controls['anchors'], CANONICAL_ANCHORS)
    payload = {
        'schema_version': 1,
        'status': 'working_cpu_only_center_availability_not_ap_or_causal',
        'inputs': {
            'dump': str(args.dump),
            'dump_sha256': _file_sha256(args.dump),
            'manifest': str(args.manifest),
            'manifest_sha256': _file_sha256(args.manifest),
            'selection_content_sha256': manifest.get(
                'selection_content_sha256'),
            'queries_per_image': 600,
            'iou_threshold': args.iou_threshold,
        },
        'definition': {
            'normalized_chebyshev':
                'max(2*abs(dx_local)/GT_w, 2*abs(dy_local)/GT_h)',
            'inside': 'normalized_chebyshev <= 1',
            'scope': 'canonical geometry-miss GT only',
            'same_label': 'fixed predicted query label equals GT label',
            'matching': 'per-image maximum bipartite center-inside matching',
            'quantiles': 'finite per-GT minima; missing same-label candidate '
                         'counts are reported separately',
        },
        'limitations': [
            'Center-inside availability is not IoU reachability or AP.',
            'Existing final centers do not identify a learnable allocation '
            'mechanism.',
            'Matching ignores scores, box extent, angle, and decoder '
            'dynamics.',
            'Results are limited to Dense400, one E24 dump, and geometry '
            'misses defined at IoU 0.5.',
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
