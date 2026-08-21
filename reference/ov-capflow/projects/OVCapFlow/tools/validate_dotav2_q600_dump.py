#!/usr/bin/env python3
"""Validate a raw DOTA2 fixed-Q prediction dump on CPU."""

import argparse
import io
import json
import pickle
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch


class DumpValidationError(ValueError, AssertionError):
    """A fail-closed dump error compatible with both validator contracts."""


class CpuUnpickler(pickle.Unpickler):
    """Map serialized PyTorch storage bytes to CPU and change nothing else."""

    def find_class(self, module: str, name: str) -> Any:
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda payload: torch.load(
                io.BytesIO(payload), map_location='cpu')
        return super().find_class(module, name)


def load_cpu(path: Path) -> Sequence[dict]:
    """Load an MMEngine DumpResults pickle without requiring visible GPUs."""
    with Path(path).open('rb') as stream:
        return CpuUnpickler(stream).load()


INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def as_tensor(value: Any, field: str = 'value') -> torch.Tensor:
    """Unwrap an MMRotate container and fail closed on malformed values."""
    try:
        tensor = value if isinstance(value, torch.Tensor) else value.tensor
    except Exception as error:
        raise DumpValidationError(
            '{} must be a torch.Tensor or tensor container'.format(
                field)) from error
    if not isinstance(tensor, torch.Tensor):
        raise DumpValidationError(
            '{} must be a torch.Tensor or tensor container'.format(field))
    return tensor


def _require_positive_integer(name: str, value: int) -> None:
    if (not isinstance(value, Integral) or isinstance(value, bool)
            or value <= 0):
        raise DumpValidationError(
            '{} must be a positive integer'.format(name))


def _require_mapping(value: Any, field: str) -> Mapping:
    if not isinstance(value, ABCMapping):
        raise DumpValidationError('{} must be a mapping'.format(field))
    return value


def _require_keys(
        value: Mapping, required: set, field: str, index: int) -> None:
    missing = sorted(required.difference(value.keys()))
    if missing:
        raise DumpValidationError(
            'missing {} keys at index {}: {}'.format(
                field, index, ', '.join(missing)))


def _require_shape(
        tensor: torch.Tensor, expected: tuple, field: str, index: int) -> None:
    if tuple(tensor.shape) != expected:
        raise DumpValidationError(
            '{} shape at index {}: expected {}, got {}'.format(
                field, index, expected, tuple(tensor.shape)))


def _require_cpu(tensor: torch.Tensor, field: str, index: int) -> None:
    if tensor.device.type != 'cpu':
        raise DumpValidationError(
            '{} at index {} must be on CPU'.format(field, index))


def _require_finite(tensor: torch.Tensor, field: str, index: int) -> None:
    try:
        finite = bool(torch.isfinite(tensor).all())
    except (RuntimeError, TypeError) as error:
        raise DumpValidationError(
            '{} at index {} must support a finite-value check'.format(
                field, index)) from error
    if not finite:
        raise DumpValidationError(
            'non-finite {} at index {}'.format(field, index))


def _require_integer_labels(
        labels: torch.Tensor, field: str, index: int,
        num_classes: int) -> None:
    if labels.dtype not in INTEGER_DTYPES:
        raise DumpValidationError(
            '{} must have integer dtype (excluding bool) at index {}'.format(
                field, index))
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        singular = 'prediction label' if field == 'prediction labels' else (
            'GT label')
        raise DumpValidationError(
            '{} out of range at index {}'.format(singular, index))


def validate_records(
        records: Sequence[dict],
        expected_records: int = 13833,
        queries_per_image: int = 600,
        num_classes: int = 18) -> Dict[str, Any]:
    """Validate record identity, fixed row shapes, devices, and finiteness."""
    _require_positive_integer('expected_records', expected_records)
    _require_positive_integer('queries_per_image', queries_per_image)
    _require_positive_integer('num_classes', num_classes)
    if (not isinstance(records, ABCSequence)
            or isinstance(records, (str, bytes, bytearray))):
        raise DumpValidationError('records must be a sequence')
    try:
        record_count = len(records)
    except (TypeError, OverflowError) as error:
        raise DumpValidationError(
            'records must have a finite length') from error
    if record_count != expected_records:
        raise DumpValidationError(
            'record count: expected {}, got {}'.format(
                expected_records, record_count))

    image_ids = set()
    rows = 0
    for index, record in enumerate(records):
        record = _require_mapping(record, 'record at index {}'.format(index))
        _require_keys(
            record, {'img_id', 'pred_instances', 'gt_instances'}, 'record',
            index)
        image_id = record['img_id']
        try:
            duplicate = image_id in image_ids
            if not duplicate:
                image_ids.add(image_id)
        except (TypeError, ValueError) as error:
            raise DumpValidationError(
                'image id at index {} must be hashable'.format(index)) from error
        if duplicate:
            raise DumpValidationError(
                'duplicate image id: {}'.format(image_id))

        pred = _require_mapping(
            record['pred_instances'],
            'pred_instances at index {}'.format(index))
        _require_keys(
            pred, {'bboxes', 'scores', 'labels'}, 'prediction', index)
        boxes = as_tensor(pred['bboxes'], 'prediction boxes')
        scores = as_tensor(pred['scores'], 'prediction scores')
        labels = as_tensor(pred['labels'], 'prediction labels')
        expected_box_shape = (queries_per_image, 5)
        expected_vector_shape = (queries_per_image,)
        _require_shape(boxes, expected_box_shape, 'prediction box', index)
        _require_shape(scores, expected_vector_shape, 'prediction score', index)
        _require_shape(labels, expected_vector_shape, 'prediction label', index)
        _require_cpu(boxes, 'prediction boxes', index)
        _require_cpu(scores, 'prediction scores', index)
        _require_cpu(labels, 'prediction labels', index)
        _require_finite(boxes, 'prediction boxes', index)
        _require_finite(scores, 'prediction scores', index)
        _require_integer_labels(labels, 'prediction labels', index, num_classes)

        gt = _require_mapping(
            record['gt_instances'], 'gt_instances at index {}'.format(index))
        _require_keys(gt, {'bboxes', 'labels'}, 'GT', index)
        gt_boxes = as_tensor(gt['bboxes'], 'GT boxes')
        gt_labels = as_tensor(gt['labels'], 'GT labels')
        if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 5:
            raise DumpValidationError(
                'GT box shape at index {}: expected [N, 5], got {}'.format(
                    index, tuple(gt_boxes.shape)))
        if gt_labels.ndim != 1:
            raise DumpValidationError(
                'GT label shape at index {}: expected [N], got {}'.format(
                    index, tuple(gt_labels.shape)))
        if gt_boxes.shape[0] != gt_labels.shape[0]:
            raise DumpValidationError(
                'GT box/label length mismatch at index {}: {} versus {}'.format(
                    index, gt_boxes.shape[0], gt_labels.shape[0]))
        _require_cpu(gt_boxes, 'GT boxes', index)
        _require_cpu(gt_labels, 'GT labels', index)
        _require_finite(gt_boxes, 'GT boxes', index)
        _require_integer_labels(gt_labels, 'GT labels', index, num_classes)
        rows += int(boxes.shape[0])

    expected_rows = expected_records * queries_per_image
    if rows != expected_rows:
        raise DumpValidationError(
            'prediction row count: expected {}, got {}'.format(
                expected_rows, rows))
    return {
        'records': len(records),
        'unique_image_ids': len(image_ids),
        'prediction_rows': rows,
        'all_cpu_finite': True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--expected-records', type=int, default=13833)
    parser.add_argument('--queries-per-image', type=int, default=600)
    parser.add_argument('--num-classes', type=int, default=18)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = validate_records(
        load_cpu(args.dump),
        expected_records=args.expected_records,
        queries_per_image=args.queries_per_image,
        num_classes=args.num_classes,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
