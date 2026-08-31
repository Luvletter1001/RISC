"""Pure, deterministic DOTA annotation parsing for the P0-A inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


_SHA256_RE = re.compile(r'[0-9a-f]{64}\Z')


class P0AInventoryError(ValueError):
    """Raised when a P0-A inventory input violates its sealed contract."""


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


__all__ = [
    'P0AInventoryError', 'canonical_json_bytes', 'sha256_bytes', 'sha256_file',
    'normalize_rbox', 'polygon_area', 'convex_iou', 'parse_dota_line',
]
