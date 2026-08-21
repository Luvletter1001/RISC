"""Spatial-region to dense feature-point assignment for FOCUS losses."""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Iterable, Sequence


Point = tuple[float, float]


def parse_polygon(value: Any) -> list[Point]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
    return [(float(x), float(y)) for x, y in value]


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _blank_mask(height: int, width: int) -> list[list[bool]]:
    return [[False for _ in range(width)] for _ in range(height)]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class FocusSpatialTargetAssigner:
    """Assign verified spatial targets to feature-grid centers.

    The implementation is framework-light on purpose so scripts/tests can run
    without importing MMRotate. Detector integration can convert masks to
    tensors at the call site.
    """

    def __init__(
            self,
            max_points_per_target: int = 16,
            max_focus_points_per_image: int = 256) -> None:
        self.max_points_per_target = int(max_points_per_target)
        self.max_focus_points_per_image = int(max_focus_points_per_image)

    def choose_level(self, target: dict[str, Any], strides: Sequence[int]) -> int:
        area = max(_as_float(target.get("raw_box_area")), 1.0)
        scale = math.sqrt(area)
        for idx, stride in enumerate(strides):
            if scale <= max(float(stride) * 8.0, 1.0):
                return idx
        return max(0, len(strides) - 1)

    def _candidate_points(
            self,
            polygon: Sequence[Point],
            center: Point,
            feat_size: tuple[int, int],
            stride: int) -> list[dict[str, Any]]:
        height, width = feat_size
        out: list[dict[str, Any]] = []
        for gy in range(height):
            for gx in range(width):
                px = (gx + 0.5) * stride
                py = (gy + 0.5) * stride
                inside = point_in_polygon(px, py, polygon)
                if not inside:
                    continue
                dist = math.hypot(px - center[0], py - center[1])
                out.append({
                    "grid_x": gx,
                    "grid_y": gy,
                    "point_x": px,
                    "point_y": py,
                    "distance_to_center": dist,
                    "inside_polygon": True,
                })
        out.sort(key=lambda row: row["distance_to_center"])
        return out[:self.max_points_per_target]

    def assign(
            self,
            targets: Iterable[dict[str, Any]],
            featmap_sizes: Sequence[tuple[int, int]],
            strides: Sequence[int],
            image_size: tuple[int, int] | None = None,
            valid_mask: Any | None = None) -> dict[str, Any]:
        del image_size, valid_mask
        anti_masks = [_blank_mask(h, w) for h, w in featmap_sizes]
        preserve_masks = [_blank_mask(h, w) for h, w in featmap_sizes]
        weight_masks = [
            [[0.0 for _ in range(w)] for _ in range(h)]
            for h, w in featmap_sizes
        ]
        debug_rows: list[dict[str, Any]] = []
        total_points = 0

        for target in targets:
            if str(target.get("valid_for_loss", "")).lower() != "true":
                continue
            role = str(target.get("target_role", ""))
            if role not in {"anti_negative", "preserve_positive"}:
                continue
            polygon = parse_polygon(target.get("target_polygon", ""))
            center = (
                _as_float(target.get("target_center_x")),
                _as_float(target.get("target_center_y")),
            )
            level = self.choose_level(target, [int(s) for s in strides])
            candidates = self._candidate_points(
                polygon, center, featmap_sizes[level], int(strides[level]))
            for item in candidates:
                if total_points >= self.max_focus_points_per_image:
                    break
                gy = int(item["grid_y"])
                gx = int(item["grid_x"])
                conflict = ""
                if role == "preserve_positive":
                    if anti_masks[level][gy][gx]:
                        conflict = "preserve_over_anti"
                    anti_masks[level][gy][gx] = False
                    preserve_masks[level][gy][gx] = True
                elif role == "anti_negative" and not preserve_masks[level][gy][gx]:
                    anti_masks[level][gy][gx] = True
                elif role == "anti_negative":
                    conflict = "preserve_over_anti"
                weight_masks[level][gy][gx] = 1.0
                debug_rows.append({
                    "focus_target_id": target.get("focus_target_id", ""),
                    "level": level,
                    "grid_x": gx,
                    "grid_y": gy,
                    "point_x": item["point_x"],
                    "point_y": item["point_y"],
                    "inside_polygon": item["inside_polygon"],
                    "distance_to_center": item["distance_to_center"],
                    "assigned_role": role,
                    "weight": 1.0,
                    "conflict_resolution": conflict,
                    "valid": True,
                })
                total_points += 1

        return {
            "anti_mask_by_level": anti_masks,
            "preserve_mask_by_level": preserve_masks,
            "target_weights_by_level": weight_masks,
            "target_meta_by_level": [],
            "debug_rows": debug_rows,
            "mapping_coverage": {
                "assigned_points": total_points,
            },
        }
