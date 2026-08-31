"""Pure, deterministic DOTA parsing and P0-A diagnostic inventory artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


_SHA256_RE = re.compile(r'[0-9a-f]{64}\Z')
_SOURCE_PLAN_SCHEMA = 'risc-n0-set-orbit-scene-plan-v1'
_CANDIDATE_PLAN_SCHEMA = 'ovd-orbit-p0-candidate-plan-v1'
_DIAGNOSTICS_SCHEMA = 'risc-n0-set-orbit-p0a-inventory-diagnostics-v1'
_RECEIPT_SCHEMA = 'risc-n0-set-orbit-p0a-inventory-receipt-v1'
_SELECTED_FOLDS = frozenset({'c4_a', 'c4_b'})
_FORBIDDEN_SCENE_ID = 'P0148'
_QUANTILE_LABELS = ((0.0, '0.0'), (0.25, '0.25'), (0.5, '0.5'),
                    (0.75, '0.75'), (1.0, '1.0'))


class P0AInventoryError(ValueError):
    """Raised when a P0-A inventory input violates its sealed contract."""


class _CanonicalJsonMapping(dict[str, Any]):
    """A canonical JSON mapping carrying the digest of its exact source bytes."""

    def __init__(self, value: Mapping[str, Any], source_sha256: str) -> None:
        super().__init__(value)
        self.source_sha256 = source_sha256


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one canonical JSON record as compact, sorted UTF-8 plus one LF."""
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False,
    ) + '\n').encode('utf-8')


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 hex digest for a bytes-like value."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading its contents at once."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise P0AInventoryError('chunk_size must be a positive integer')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_annotation_snapshot(path: str | Path) -> tuple[bytes, str]:
    """Stream one annotation once and return the exact parsed bytes and digest."""
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            chunks.append(chunk)
    return b''.join(chunks), digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise P0AInventoryError('JSON constants must be finite')


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P0AInventoryError(f'{label} must be a mapping')
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P0AInventoryError(f'{label} must be a nonempty string')
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise P0AInventoryError(f'{label} must be a nonnegative integer')
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise P0AInventoryError(f'{label} must be a lowercase SHA-256 hex digest')
    return value


def load_canonical_json(path: str | Path, *, label: str) -> Mapping[str, Any]:
    """Load a mapping only when its source bytes are canonical JSON."""
    label = _require_nonempty_string(label, 'label')
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise P0AInventoryError(f'{label} could not read {source}') from error
    try:
        value = json.loads(raw.decode('utf-8'), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, P0AInventoryError) as error:
        raise P0AInventoryError(f'{label} must contain valid canonical JSON') from error
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise P0AInventoryError(f'{label} must contain valid canonical JSON') from error
    if canonical != raw:
        raise P0AInventoryError(f'{label} bytes must be canonical JSON')
    return _CanonicalJsonMapping(_require_mapping(value, label), sha256_bytes(raw))


def normalize_rbox(box: Sequence[float]) -> list[float]:
    """Return a finite [cx, cy, w, h, theta] with w >= h and canonical theta."""
    if (isinstance(box, (str, bytes)) or not isinstance(box, Sequence)
            or len(box) != 5):
        raise P0AInventoryError('box must contain exactly five values')
    values = []
    for value in box:
        if isinstance(value, bool):
            raise P0AInventoryError('box values must be finite numbers')
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise P0AInventoryError('box values must be finite numbers') from error
        if not math.isfinite(number):
            raise P0AInventoryError('box values must be finite numbers')
        values.append(number)
    center_x, center_y, width, height, theta = values
    if width <= 0 or height <= 0:
        raise P0AInventoryError('box width and height must be positive')
    if width < height:
        width, height = height, width
        theta += math.pi / 2
    theta = math.fmod(theta + math.pi / 2, math.pi)
    if theta < 0:
        theta += math.pi
    theta -= math.pi / 2
    return [center_x, center_y, width, height, theta]


def _polygon_from(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        if 'polygon' in value:
            value = value['polygon']
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise P0AInventoryError('polygon must be a sequence of points')
    if len(value) < 3:
        raise P0AInventoryError('polygon must contain at least three points')
    polygon = []
    for point in value:
        if (isinstance(point, (str, bytes)) or not isinstance(point, Sequence)
                or len(point) != 2):
            raise P0AInventoryError('each polygon point must contain two coordinates')
        coordinates = []
        for coordinate in point:
            if isinstance(coordinate, bool):
                raise P0AInventoryError('polygon coordinates must be finite')
            try:
                number = float(coordinate)
            except (TypeError, ValueError) as error:
                raise P0AInventoryError(
                    'polygon coordinates must be finite') from error
            if not math.isfinite(number):
                raise P0AInventoryError('polygon coordinates must be finite')
            coordinates.append(number)
        polygon.append((coordinates[0], coordinates[1]))
    return polygon


def _signed_polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return sum(
        first[0] * second[1] - first[1] * second[0]
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    ) / 2


def polygon_area(polygon: Any) -> float:
    """Return the absolute shoelace area of a finite polygon."""
    return abs(_signed_polygon_area(_polygon_from(polygon)))


def _cross(
        start: tuple[float, float], end: tuple[float, float],
        point: tuple[float, float]) -> float:
    return ((end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0]))


def _line_intersection(
        subject_start: tuple[float, float], subject_end: tuple[float, float],
        clip_start: tuple[float, float], clip_end: tuple[float, float],
) -> tuple[float, float]:
    subject_dx = subject_end[0] - subject_start[0]
    subject_dy = subject_end[1] - subject_start[1]
    clip_dx = clip_end[0] - clip_start[0]
    clip_dy = clip_end[1] - clip_start[1]
    denominator = subject_dx * clip_dy - subject_dy * clip_dx
    if denominator == 0.0:
        return subject_end
    offset_x = clip_start[0] - subject_start[0]
    offset_y = clip_start[1] - subject_start[1]
    fraction = (offset_x * clip_dy - offset_y * clip_dx) / denominator
    return (
        subject_start[0] + fraction * subject_dx,
        subject_start[1] + fraction * subject_dy,
    )


def _clip_convex_polygon(
        subject: list[tuple[float, float]],
        clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    orientation = _signed_polygon_area(clip)
    if orientation == 0.0:
        return []
    sign = 1.0 if orientation > 0 else -1.0
    output = subject
    for clip_start, clip_end in zip(clip, clip[1:] + clip[:1]):
        if not output:
            return []
        input_polygon = output
        output = []
        previous = input_polygon[-1]
        previous_inside = sign * _cross(clip_start, clip_end, previous) >= 0.0
        for current in input_polygon:
            current_inside = sign * _cross(clip_start, clip_end, current) >= 0.0
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(
                        previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(
                    previous, current, clip_start, clip_end))
            previous = current
            previous_inside = current_inside
    return output


def convex_iou(first: Any, second: Any) -> float:
    """Return the IoU of two convex polygons using Sutherland-Hodgman clipping."""
    first_polygon = _polygon_from(first)
    second_polygon = _polygon_from(second)
    first_area = abs(_signed_polygon_area(first_polygon))
    second_area = abs(_signed_polygon_area(second_polygon))
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    intersection = _clip_convex_polygon(first_polygon, second_polygon)
    intersection_area = abs(_signed_polygon_area(intersection)) if intersection else 0.0
    if intersection_area <= 0.0:
        return 0.0
    return intersection_area / (first_area + second_area - intersection_area)


def _line_error(line_number: int, message: str) -> P0AInventoryError:
    return P0AInventoryError(f'line {line_number}: {message}')


def _validate_annotation_sha256(annotation_sha256: str) -> None:
    if not isinstance(annotation_sha256, str) or not _SHA256_RE.fullmatch(annotation_sha256):
        raise P0AInventoryError('annotation_sha256 must be a lowercase SHA-256 hex digest')


def _validate_convex_quadrilateral(
        polygon: Sequence[tuple[float, float]], *, line_number: int) -> None:
    cross_products = [
        _cross(polygon[index], polygon[(index + 1) % 4], polygon[(index + 2) % 4])
        for index in range(4)
    ]
    if (any(cross == 0.0 for cross in cross_products)
            or not (all(cross > 0 for cross in cross_products)
                    or all(cross < 0 for cross in cross_products))):
        raise _line_error(line_number, 'quadrilateral must be strictly convex')


def parse_dota_line(
        line: str, *, line_number: int, scene_id: str, annotation_sha256: str,
        vocabulary: Sequence[str],
) -> dict[str, Any]:
    """Parse one DOTA line into a sealed, geometry-checked inventory record."""
    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1:
        raise P0AInventoryError('line_number must be a positive integer')
    if not isinstance(scene_id, str) or not scene_id:
        raise _line_error(line_number, 'scene_id must be a nonempty string')
    _validate_annotation_sha256(annotation_sha256)
    if isinstance(vocabulary, (str, bytes)) or not isinstance(vocabulary, Sequence):
        raise _line_error(line_number, 'vocabulary must be a sequence of class names')
    if not isinstance(line, str):
        raise _line_error(line_number, 'annotation must be text')
    tokens = line.split()
    if len(tokens) != 10:
        raise _line_error(line_number, 'DOTA annotation must contain ten tokens')

    coordinates = []
    for token in tokens[:8]:
        try:
            coordinate = float(token)
        except ValueError as error:
            raise _line_error(line_number, 'coordinates must be finite numbers') from error
        if not math.isfinite(coordinate):
            raise _line_error(line_number, 'coordinates must be finite numbers')
        coordinates.append(coordinate)
    polygon = tuple(
        (coordinates[index], coordinates[index + 1]) for index in range(0, 8, 2))
    _validate_convex_quadrilateral(polygon, line_number=line_number)

    class_name = tokens[8]
    if class_name not in vocabulary:
        raise _line_error(line_number, f'unknown class {class_name!r}')
    try:
        difficulty = int(tokens[9])
    except ValueError as error:
        raise _line_error(line_number, 'difficulty must be 0, 1, or 2') from error
    if str(difficulty) != tokens[9] or difficulty not in (0, 1, 2):
        raise _line_error(line_number, 'difficulty must be 0, 1, or 2')

    width = math.dist(polygon[0], polygon[1])
    height = math.dist(polygon[1], polygon[2])
    if width <= 0.0 or height <= 0.0:
        raise _line_error(line_number, 'quadrilateral sides must be nonzero')
    size = polygon_area(polygon)
    raw_box = [
        sum(point[0] for point in polygon) / 4,
        sum(point[1] for point in polygon) / 4,
        width,
        height,
        math.atan2(
            polygon[1][1] - polygon[0][1], polygon[1][0] - polygon[0][0]),
    ]
    if not all(math.isfinite(value) for value in raw_box):
        raise _line_error(line_number, 'derived box values must be finite')
    box = normalize_rbox(raw_box)
    if not all(math.isfinite(value) for value in box):
        raise _line_error(line_number, 'derived box values must be finite')
    if not math.isfinite(size) or size <= 0.0:
        raise _line_error(line_number, 'quadrilateral area must be finite and positive')
    return {
        'scene_id': scene_id,
        'object_id': f'{scene_id}:{line_number}',
        'class_name': class_name,
        'annotation_sha256': annotation_sha256,
        'box': box,
        'size': size,
        'difficulty': difficulty,
        'polygon': polygon,
    }


def _source_record(record: Any, index: int) -> dict[str, Any]:
    """Validate and copy the source fields needed by a later diagnostic plan."""
    context = f'source plan records[{index}]'
    record = _require_mapping(record, context)
    required = (
        'fold_id', 'scene_id', 'scene_rank', 'image_path', 'image_sha256',
        'annotation_path', 'annotation_sha256',
    )
    for field in required:
        if field not in record:
            raise P0AInventoryError(f'{context}.{field} is required')
    return {
        'fold_id': _require_nonempty_string(record['fold_id'], f'{context}.fold_id'),
        'scene_id': _require_nonempty_string(record['scene_id'], f'{context}.scene_id'),
        'scene_rank': _require_nonnegative_int(
            record['scene_rank'], f'{context}.scene_rank'),
        'image_path': _require_nonempty_string(
            record['image_path'], f'{context}.image_path'),
        'image_sha256': _require_sha256(
            record['image_sha256'], f'{context}.image_sha256'),
        'annotation_path': _require_nonempty_string(
            record['annotation_path'], f'{context}.annotation_path'),
        'annotation_sha256': _require_sha256(
            record['annotation_sha256'], f'{context}.annotation_sha256'),
    }


def validate_source_plan(source_plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return the sorted, exactly-80 C4 records from a sealed source plan."""
    source_plan = _require_mapping(source_plan, 'source plan')
    if source_plan.get('schema') != _SOURCE_PLAN_SCHEMA:
        raise P0AInventoryError(
            f'source plan.schema must be {_SOURCE_PLAN_SCHEMA!r}')
    unique_scene_count = source_plan.get('unique_scene_count')
    if (isinstance(unique_scene_count, bool)
            or not isinstance(unique_scene_count, int)
            or unique_scene_count != 160):
        raise P0AInventoryError('source plan.unique_scene_count must be exactly 160')
    records = source_plan.get('records')
    if not isinstance(records, list):
        raise P0AInventoryError('source plan.records must be a list')
    if len(records) != 160:
        raise P0AInventoryError('source plan.records length must be exactly 160')

    selected: list[dict[str, Any]] = []
    scene_ids: set[str] = set()
    for index, value in enumerate(records):
        record = _source_record(value, index)
        scene_id = record['scene_id']
        if scene_id == _FORBIDDEN_SCENE_ID:
            raise P0AInventoryError('source plan must not contain P0148')
        if scene_id in scene_ids:
            raise P0AInventoryError('source plan scene_id values must be unique')
        scene_ids.add(scene_id)
        if record['fold_id'] in _SELECTED_FOLDS:
            selected.append(record)
    if len(selected) != 80:
        raise P0AInventoryError('source plan must contain exactly 80 C4 scenes')
    if len(scene_ids) != 160:
        raise P0AInventoryError('source plan must contain exactly 160 distinct scene_id values')
    fold_counts = {fold_id: 0 for fold_id in sorted(_SELECTED_FOLDS)}
    for record in selected:
        fold_counts[record['fold_id']] += 1
    if fold_counts != {'c4_a': 40, 'c4_b': 40}:
        raise P0AInventoryError(
            'source plan must contain exactly 40 scenes from each C4 fold')
    selected.sort(key=lambda row: (
        row['fold_id'], row['scene_rank'], row['scene_id']))
    return tuple(selected)


def validate_source_manifest(source_input_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate and return the immutable 18-name full diagnostic vocabulary."""
    source_input_manifest = _require_mapping(
        source_input_manifest, 'source input manifest')
    support = _require_mapping(
        source_input_manifest.get('support'), 'source input manifest.support')
    class_order = support.get('class_order')
    if not isinstance(class_order, list) or len(class_order) != 18:
        raise P0AInventoryError(
            'source input manifest.support.class_order must contain exactly 18 names')
    vocabulary = tuple(
        _require_nonempty_string(
            name, f'source input manifest.support.class_order[{index}]')
        for index, name in enumerate(class_order))
    if len(set(vocabulary)) != len(vocabulary):
        raise P0AInventoryError(
            'source input manifest.support.class_order names must be unique')
    return vocabulary


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b''.join(canonical_json_bytes(row) for row in rows)


def _quantiles(values: Sequence[float], label: str) -> dict[str, float]:
    """Return the fixed five linear quantiles with finite canonical values."""
    if not values:
        raise P0AInventoryError(f'{label} quantiles require at least one retained object')
    ordered = sorted(values)
    if not all(math.isfinite(value) for value in ordered):
        raise P0AInventoryError(f'{label} quantiles must be finite')
    result: dict[str, float] = {}
    last = len(ordered) - 1
    for quantile, quantile_label in _QUANTILE_LABELS:
        position = last * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
        if not math.isfinite(value):
            raise P0AInventoryError(f'{label} quantiles must be finite')
        result[quantile_label] = value
    return result


def _public_object_row(record: Mapping[str, Any], overlap: float) -> dict[str, Any]:
    """Discard parser-only polygon and difficulty data before artifact encoding."""
    if not math.isfinite(overlap) or overlap < 0.0 or overlap > 1.0:
        raise P0AInventoryError('derived object overlap must be finite in [0, 1]')
    return {
        'annotation_sha256': record['annotation_sha256'],
        'box': record['box'],
        'class_name': record['class_name'],
        'object_id': record['object_id'],
        'overlap': overlap,
        'scene_id': record['scene_id'],
        'size': record['size'],
    }


def _read_verified_annotation(record: Mapping[str, Any]) -> list[tuple[int, str]]:
    """Hash one annotation before any parser sees its text; never read images."""
    scene_id = record['scene_id']
    path = Path(record['annotation_path'])
    try:
        snapshot, observed_hash = read_annotation_snapshot(path)
    except (OSError, ValueError) as error:
        raise P0AInventoryError(f'annotation {scene_id} hash mismatch') from error
    if observed_hash != record['annotation_sha256']:
        raise P0AInventoryError(f'annotation {scene_id} hash mismatch')
    try:
        text = snapshot.decode('utf-8')
    except UnicodeError as error:
        raise P0AInventoryError(f'annotation {scene_id} could not read') from error
    return [
        (line_number, line)
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]


def _candidate_record(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'scene_id': source['scene_id'],
        'split': 'diagnostic',
        'image_path': source['image_path'],
        'image_sha256': source['image_sha256'],
        'annotation_path': source['annotation_path'],
        'annotation_sha256': source['annotation_sha256'],
    }


def _inventory_diagnostics(
        *, rows: Sequence[Mapping[str, Any]], selected_scene_count: int,
        difficulty_2_object_count: int, annotation_verification_count: int,
) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    for row in rows:
        class_name = row['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return {
        'schema': _DIAGNOSTICS_SCHEMA,
        'status': 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD',
        'selected_scene_count': selected_scene_count,
        'retained_object_count': len(rows),
        'difficulty_2_object_count': difficulty_2_object_count,
        'annotation_verification_count': annotation_verification_count,
        'class_counts': {name: class_counts[name] for name in sorted(class_counts)},
        'size_quantiles': _quantiles([row['size'] for row in rows], 'size'),
        'overlap_quantiles': _quantiles([row['overlap'] for row in rows], 'overlap'),
    }


def _result_markdown(diagnostics: Mapping[str, Any]) -> bytes:
    """Return the Chinese, count-only human receipt for diagnostic preparation."""
    return (
        '# P0-A 诊断对象清单摘要\n\n'
        '状态：P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD\n\n'
        f"候选场景数：{diagnostics['selected_scene_count']}\n"
        f"保留对象数：{diagnostics['retained_object_count']}\n"
        f"难度 2 跳过对象数：{diagnostics['difficulty_2_object_count']}\n"
        f"已核验标注数：{diagnostics['annotation_verification_count']}\n"
        f"保留类别数：{len(diagnostics['class_counts'])}\n\n"
        '未运行模型、检查点或 GPU；本文件仅记录诊断输入清单计数。\n'
    ).encode('utf-8')


def build_inventory_artifacts(
        *, source_plan: Mapping[str, Any], source_plan_sha256: str,
        source_input_manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    """Build the deterministic P0-A package without reading images or a model."""
    source_plan = _require_mapping(source_plan, 'source plan')
    source_plan_sha256 = _require_sha256(
        source_plan_sha256, 'source_plan_sha256')
    try:
        observed_plan_sha256 = sha256_bytes(canonical_json_bytes(source_plan))
    except (TypeError, ValueError, UnicodeError) as error:
        raise P0AInventoryError('source plan cannot be canonically hashed') from error
    if source_plan_sha256 != observed_plan_sha256:
        raise P0AInventoryError('source_plan_sha256 must match source plan bytes')

    selected = validate_source_plan(source_plan)
    vocabulary = validate_source_manifest(source_input_manifest)
    retained_rows: list[dict[str, Any]] = []
    difficulty_2_object_count = 0
    annotation_verification_count = 0
    for source in selected:
        parsed_retained: list[dict[str, Any]] = []
        for line_number, line in _read_verified_annotation(source):
            parsed = parse_dota_line(
                line,
                line_number=line_number,
                scene_id=source['scene_id'],
                annotation_sha256=source['annotation_sha256'],
                vocabulary=vocabulary,
            )
            if parsed['difficulty'] == 2:
                difficulty_2_object_count += 1
            else:
                parsed_retained.append(parsed)
        annotation_verification_count += 1
        for record in parsed_retained:
            overlap = max((
                convex_iou(record['polygon'], other['polygon'])
                for other in parsed_retained if other is not record
            ), default=0.0)
            retained_rows.append(_public_object_row(record, overlap))

    retained_rows.sort(key=lambda row: (row['scene_id'], row['object_id']))
    candidate_plan = {
        'schema': _CANDIDATE_PLAN_SCHEMA,
        'records': [_candidate_record(source) for source in selected],
    }
    diagnostics = _inventory_diagnostics(
        rows=retained_rows,
        selected_scene_count=len(selected),
        difficulty_2_object_count=difficulty_2_object_count,
        annotation_verification_count=annotation_verification_count,
    )
    artifacts = {
        'candidate_scene_plan.json': canonical_json_bytes(candidate_plan),
        'object_inventory.jsonl': _canonical_jsonl_bytes(retained_rows),
        'inventory_diagnostics.json': canonical_json_bytes(diagnostics),
        'result.md': _result_markdown(diagnostics),
    }
    artifacts['receipt.json'] = canonical_json_bytes({
        'schema': _RECEIPT_SCHEMA,
        'status': 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD',
        'artifact_sha256': {
            name: sha256_bytes(artifacts[name]) for name in sorted(artifacts)
        },
    })
    return artifacts


def build_inventory_failure_artifacts(error: P0AInventoryError) -> dict[str, bytes]:
    """Build the minimal fail-stop package without returning source contents."""
    if not isinstance(error, P0AInventoryError):
        raise TypeError('error must be a P0AInventoryError')
    message = str(error)
    status = 'P0_INPUT_FAIL_STOP'
    return {
        'receipt.json': canonical_json_bytes({
            'schema': _RECEIPT_SCHEMA,
            'status': status,
            'error': message,
        }),
        'inventory_diagnostics.json': canonical_json_bytes({
            'schema': _DIAGNOSTICS_SCHEMA,
            'status': status,
            'error': message,
        }),
        'result.md': (
            '# P0-A 诊断对象清单失败\n\n'
            f'诊断输入失败：{message}\n\n'
            '未运行模型、检查点或 GPU。\n'
        ).encode('utf-8'),
    }


def _validated_inventory_artifact_items(
        artifacts: Mapping[str, bytes]) -> tuple[tuple[str, bytes], ...]:
    """Return safe, nonempty flat artifact items for atomic publication."""
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise P0AInventoryError('artifacts must be a nonempty mapping')
    items: list[tuple[str, bytes]] = []
    names: set[str] = set()
    for name, payload in artifacts.items():
        if not isinstance(name, str) or not name:
            raise P0AInventoryError('artifact names must be nonempty strings')
        path = Path(name)
        if (path.is_absolute() or path.name != name or name in {'.', '..'}
                or '/' in name or '\\' in name or '\x00' in name):
            raise P0AInventoryError(f'unsafe artifact name: {name!r}')
        if name in names:
            raise P0AInventoryError(f'duplicate artifact name: {name!r}')
        if not isinstance(payload, bytes) or not payload:
            raise P0AInventoryError(f'artifact {name!r} must be nonempty bytes')
        names.add(name)
        items.append((name, payload))
    return tuple(items)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically rename a same-parent directory without replacing its target."""
    if os.name != 'posix':
        raise P0AInventoryError('no-replace directory publication is unavailable')
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise P0AInventoryError(
            'no-replace directory publication is unavailable') from error
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
            -100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def _raise_if_output_exists(output_dir: Path) -> None:
    try:
        output_dir.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(output_dir))


def publish_inventory_artifacts(
        output_dir: Path | str, artifacts: Mapping[str, bytes]) -> None:
    """Atomically publish one P0-A package without overwriting an output path."""
    items = _validated_inventory_artifact_items(artifacts)
    target = Path(output_dir)
    if not target.name or target.name in {'.', '..'}:
        raise P0AInventoryError('output_dir must name a new directory')
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    _raise_if_output_exists(target)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{target.name}.tmp-', dir=parent))
    try:
        for name, payload in items:
            destination = temporary / name
            with destination.open('xb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    'P0AInventoryError', 'canonical_json_bytes', 'sha256_bytes', 'sha256_file',
    'read_annotation_snapshot', 'load_canonical_json', 'normalize_rbox',
    'polygon_area', 'convex_iou',
    'parse_dota_line', 'validate_source_plan', 'validate_source_manifest',
    'build_inventory_artifacts', 'build_inventory_failure_artifacts',
    'publish_inventory_artifacts',
]
