"""Pure authority for selecting the P0-A diagnostic primary objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


_REQUIRED_ROW_KEYS = frozenset({
    'annotation_sha256', 'box', 'class_name', 'object_id', 'overlap',
    'scene_id', 'size',
})
_SHA256_RE = re.compile(r'[0-9a-f]{64}\Z')
_AREA_MIN = 43.5
_AREA_MAX = 3265.0
_MAX_OVERLAP = 0.05
_MINIMUM_CLASS_COUNT = 5
_ZERO_OBSERVED_CLASSES = ('container-crane', 'helicopter', 'helipad')
_MISSING = object()


class P0AAuthorityError(ValueError):
    """Raised when P0-A primary-selection input violates the sealed policy."""


@dataclass(frozen=True)
class PrimarySelection:
    """Immutable result of applying the fixed P0-A primary-object policy."""

    policy: Mapping[str, Any]
    primary_rows: tuple[Mapping[str, Any], ...]
    primary_object_count: int
    supported_classes: tuple[str, ...]
    non_primary_diagnostics: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one value as compact, key-sorted UTF-8 JSON followed by LF."""
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False,
    ) + '\n').encode('utf-8')


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of bytes-like input."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file, streamed in bounded chunks."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise P0AAuthorityError('chunk_size must be a positive integer')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P0AAuthorityError(f'{label} must be a nonempty string')
    return value


def _require_finite_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P0AAuthorityError(f'{label} must be a finite nonnegative number')
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise P0AAuthorityError(f'{label} must be a finite nonnegative number')
    return number


def _validate_box(value: Any) -> tuple[float, float, float, float, float]:
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or len(value) != 5):
        raise P0AAuthorityError('box must contain exactly five finite numbers')
    numbers: list[float] = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, Real):
            raise P0AAuthorityError('box must contain exactly five finite numbers')
        converted = float(number)
        if not math.isfinite(converted):
            raise P0AAuthorityError('box must contain exactly five finite numbers')
        numbers.append(converted)
    return tuple(numbers)  # type: ignore[return-value]


def _freeze_row(row: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise P0AAuthorityError(f'row {index} must be a mapping')
    if frozenset(row) != _REQUIRED_ROW_KEYS or len(row) != len(_REQUIRED_ROW_KEYS):
        raise P0AAuthorityError(f'row {index} must contain exactly seven required keys')
    annotation_sha256 = row['annotation_sha256']
    if not isinstance(annotation_sha256, str) or not _SHA256_RE.fullmatch(annotation_sha256):
        raise P0AAuthorityError(
            f'row {index} annotation_sha256 must be a lowercase SHA-256 digest')
    return MappingProxyType({
        'annotation_sha256': annotation_sha256,
        'box': _validate_box(row['box']),
        'class_name': _require_nonempty_string(row['class_name'], f'row {index} class_name'),
        'object_id': _require_nonempty_string(row['object_id'], f'row {index} object_id'),
        'overlap': _require_finite_nonnegative_number(row['overlap'], f'row {index} overlap'),
        'scene_id': _require_nonempty_string(row['scene_id'], f'row {index} scene_id'),
        'size': _require_finite_nonnegative_number(row['size'], f'row {index} size'),
    })


def _sort_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(sorted(rows, key=lambda row: (row['scene_id'], row['object_id'])))


def _fixed_policy() -> Mapping[str, Any]:
    return MappingProxyType({
        'area_min': _AREA_MIN,
        'area_max': _AREA_MAX,
        'max_overlap_iou': _MAX_OVERLAP,
        'min_objects_per_class': _MINIMUM_CLASS_COUNT,
        'selection_basis': 'p0a_inventory_gt_quantiles_v1',
    })


def select_primary_objects(
        rows: Iterable[Mapping[str, Any]], *,
        zero_observed_classes: Any = _MISSING,
        **policy_overrides: Any) -> PrimarySelection:
    """Select immutable P0-A primary objects using the non-overridable policy."""
    if policy_overrides:
        names = ', '.join(sorted(policy_overrides))
        raise P0AAuthorityError(f'fixed P0-A policy cannot be overridden: {names}')
    if zero_observed_classes is _MISSING:
        raise P0AAuthorityError('zero_observed_classes declaration is required')
    if (type(zero_observed_classes) is not tuple
            or zero_observed_classes != _ZERO_OBSERVED_CLASSES):
        raise P0AAuthorityError(
            'zero_observed_classes must exactly declare '
            "('container-crane', 'helicopter', 'helipad')")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise P0AAuthorityError('rows must be an iterable of mappings')

    frozen_rows = tuple(_freeze_row(row, index) for index, row in enumerate(rows))
    object_ids = tuple(row['object_id'] for row in frozen_rows)
    if len(set(object_ids)) != len(object_ids):
        raise P0AAuthorityError('object_id values must be unique')
    observed_classes = {row['class_name'] for row in frozen_rows}
    zero_observed = set(_ZERO_OBSERVED_CLASSES)
    if observed_classes & zero_observed:
        raise P0AAuthorityError('declared zero-observed classes cannot appear in rows')

    geometry_rows = []
    geometry_excluded = []
    for row in frozen_rows:
        if _AREA_MIN <= row['size'] <= _AREA_MAX and row['overlap'] <= _MAX_OVERLAP:
            geometry_rows.append(row)
        else:
            geometry_excluded.append(row)

    class_counts: dict[str, int] = {}
    for row in geometry_rows:
        class_name = row['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    supported_classes = tuple(sorted(
        class_name for class_name, count in class_counts.items()
        if count >= _MINIMUM_CLASS_COUNT))
    supported = set(supported_classes)
    primary_rows = _sort_rows(row for row in geometry_rows if row['class_name'] in supported)
    unsupported_rows = _sort_rows(
        row for row in geometry_rows if row['class_name'] not in supported)
    diagnostics = MappingProxyType({
        'class_counts': MappingProxyType(dict(sorted(class_counts.items()))),
        'geometry_excluded': _sort_rows(geometry_excluded),
        'unsupported_class': unsupported_rows,
    })
    return PrimarySelection(
        policy=_fixed_policy(),
        primary_rows=primary_rows,
        primary_object_count=len(primary_rows),
        supported_classes=supported_classes,
        non_primary_diagnostics=diagnostics,
    )


__all__ = [
    'P0AAuthorityError', 'PrimarySelection', 'canonical_json_bytes',
    'sha256_bytes', 'sha256_file', 'select_primary_objects',
]
