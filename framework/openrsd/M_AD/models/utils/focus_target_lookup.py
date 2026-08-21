"""Lookup and transform FOCUS spatial targets for detector batches."""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_ROLES = {"anti_negative", "preserve_positive"}
DEFAULT_COORDINATE_FRAME = "rotated_angle_sweep"
DEBUG_FIELDS = [
    "batch_index",
    "img_path",
    "parsed_tile_id",
    "parsed_angle",
    "matched_targets",
    "anti_targets",
    "preserve_targets",
    "unmatched_reason",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        if value in {"", None}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_path_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _norm_path_key(path: str) -> str:
    return str(Path(path)).replace("\\", "/")


def _stem_from_path(path: str) -> str:
    if not path:
        return ""
    return Path(path).stem


def _parse_angle_from_path(path: str) -> int | None:
    match = re.search(r"angle[_-](\d{1,3})", path)
    if not match:
        return None
    return _safe_int(match.group(1))


def _extract_hw(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def parse_polygon(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
    return [(float(x), float(y)) for x, y in value]


class FocusTargetLookup:
    """Match spatial target rows to batch image metadata.

    The target CSV is authored in the original rotated angle-sweep image frame.
    This lookup supports deterministic resize plus top-left-origin padding. It
    rejects transforms such as random flip or rotation because those would
    require full polygon synchronization from the pipeline.
    """

    def __init__(
            self,
            target_csv: str | Path,
            coordinate_frame: str = DEFAULT_COORDINATE_FRAME) -> None:
        self.target_csv = Path(target_csv)
        self.coordinate_frame = str(coordinate_frame)
        self.rows = self._load_rows(self.target_csv)
        self.index = self._build_index(self.rows)
        self.last_debug_rows: list[dict[str, Any]] = []

    def _load_rows(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return [
            row for row in rows
            if _truthy(row.get("valid_for_loss"))
            and row.get("target_role") in SUPPORTED_ROLES
        ]

    def _build_index(
            self,
            rows: Iterable[dict[str, Any]]
    ) -> dict[str, dict[Any, list[dict[str, Any]]]]:
        index: dict[str, dict[Any, list[dict[str, Any]]]] = {
            "image_path": defaultdict(list),
            "basename": defaultdict(list),
            "stem": defaultdict(list),
            "tile_angle": defaultdict(list),
            "tile_id": defaultdict(list),
        }
        for row in rows:
            image_path = _as_path_text(row.get("image_path"))
            if image_path:
                norm_path = _norm_path_key(image_path)
                index["image_path"][norm_path].append(row)
                index["basename"][Path(norm_path).name].append(row)
                index["stem"][Path(norm_path).stem].append(row)
            tile_id = _as_path_text(row.get("tile_id")) or _stem_from_path(
                image_path)
            angle = _safe_int(row.get("angle"))
            if tile_id:
                index["tile_id"][tile_id].append(row)
            if tile_id and angle is not None:
                index["tile_angle"][(tile_id, angle)].append(row)
        return index

    def parse_image_identity(self, meta: dict[str, Any]) -> dict[str, Any]:
        img_path = (
            _as_path_text(meta.get("img_path"))
            or _as_path_text(meta.get("filename"))
            or _as_path_text(meta.get("ori_filename"))
            or _as_path_text(meta.get("img_id")))
        img_id = _as_path_text(meta.get("img_id"))
        tile_id = _as_path_text(meta.get("tile_id"))
        if not tile_id:
            tile_id = img_id or _stem_from_path(img_path)
        angle = _safe_int(meta.get("angle"))
        if angle is None:
            angle = _parse_angle_from_path(img_path)
        return {
            "img_path": img_path,
            "img_id": img_id,
            "tile_id": tile_id,
            "angle": angle,
            "basename": Path(img_path).name if img_path else "",
            "stem": _stem_from_path(img_path),
        }

    def _candidate_rows(
            self,
            identity: dict[str, Any]) -> list[dict[str, Any]]:
        img_path = identity.get("img_path", "")
        tile_id = identity.get("tile_id", "")
        angle = identity.get("angle")
        candidate_sets: list[list[dict[str, Any]]] = []
        if img_path:
            candidate_sets.append(
                self.index["image_path"].get(_norm_path_key(img_path), []))
        if tile_id and angle is not None:
            candidate_sets.append(
                self.index["tile_angle"].get((tile_id, int(angle)), []))
        if identity.get("basename"):
            candidate_sets.append(
                self.index["basename"].get(identity["basename"], []))
        if identity.get("stem"):
            candidate_sets.append(self.index["stem"].get(identity["stem"], []))
        if tile_id and angle is None:
            candidate_sets.append(self.index["tile_id"].get(tile_id, []))

        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for candidates in candidate_sets:
            for row in candidates:
                key = str(row.get("focus_target_id") or id(row))
                if key in seen:
                    continue
                seen.add(key)
                out.append(row)
            if out:
                break
        return out

    def _unsupported_transform_reason(self, meta: dict[str, Any]) -> str:
        if _truthy(meta.get("flip")):
            return "augmentation_not_supported"
        for key in (
                "rotate",
                "rotation",
                "rotation_angle",
                "rotate_angle",
                "random_rotate_angle"):
            value = meta.get(key)
            if value not in {None, "", 0, 0.0, "0", "0.0"}:
                return "augmentation_not_supported"
        return ""

    def _scale_factors(self, meta: dict[str, Any]) -> tuple[float, float]:
        scale_factor = meta.get("scale_factor")
        if hasattr(scale_factor, "tolist"):
            scale_factor = scale_factor.tolist()
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) == 1:
                sx = sy = _safe_float(scale_factor[0], 1.0)
            else:
                sx = _safe_float(scale_factor[0], 1.0)
                sy = _safe_float(scale_factor[1], sx)
            return sx, sy
        if scale_factor not in {None, ""}:
            sx = sy = _safe_float(scale_factor, 1.0)
            return sx, sy

        ori_hw = _extract_hw(meta.get("ori_shape"))
        img_hw = _extract_hw(meta.get("img_shape"))
        if ori_hw and img_hw and ori_hw[0] > 0 and ori_hw[1] > 0:
            return img_hw[1] / ori_hw[1], img_hw[0] / ori_hw[0]
        return 1.0, 1.0

    def _image_bounds(self, meta: dict[str, Any]) -> tuple[float, float] | None:
        img_hw = _extract_hw(meta.get("img_shape"))
        if not img_hw:
            return None
        height, width = img_hw[:2]
        return width, height

    def _transform_row(
            self,
            row: dict[str, Any],
            meta: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        if row.get("coordinate_frame") != self.coordinate_frame:
            return None, "coordinate_frame_mismatch"
        transform_reason = self._unsupported_transform_reason(meta)
        if transform_reason:
            return None, transform_reason
        try:
            polygon = parse_polygon(row.get("target_polygon", ""))
        except Exception:
            return None, "invalid_polygon"
        if len(polygon) < 3:
            return None, "invalid_polygon"

        sx, sy = self._scale_factors(meta)
        transformed_polygon = [
            (float(x) * sx, float(y) * sy) for x, y in polygon
        ]
        bounds = self._image_bounds(meta)
        if bounds is not None:
            width, height = bounds
            if any(
                    x < -1e-3 or y < -1e-3
                    or x > width + 1e-3 or y > height + 1e-3
                    for x, y in transformed_polygon):
                return None, "target_outside_image"

        out = dict(row)
        out["target_polygon"] = transformed_polygon
        out["target_center_x"] = _safe_float(row.get("target_center_x")) * sx
        out["target_center_y"] = _safe_float(row.get("target_center_y")) * sy
        out["raw_box_area"] = _safe_float(row.get("raw_box_area")) * sx * sy
        out["valid_for_loss"] = "true"
        out["transform_scale_x"] = sx
        out["transform_scale_y"] = sy
        return out, ""

    def match_batch(
            self,
            batch_img_metas: Iterable[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        batch_targets: list[dict[str, Any]] = []
        debug_rows: list[dict[str, Any]] = []
        for batch_index, meta in enumerate(batch_img_metas):
            identity = self.parse_image_identity(meta)
            reasons: list[str] = []
            raw_candidates: list[dict[str, Any]] = []
            if not identity["img_path"] and not identity["tile_id"]:
                reasons.append("no_image_key")
            else:
                raw_candidates = self._candidate_rows(identity)
                if not raw_candidates:
                    reasons.append("no_matching_targets")

            anti_targets: list[dict[str, Any]] = []
            preserve_targets: list[dict[str, Any]] = []
            for row in raw_candidates:
                transformed, reason = self._transform_row(row, meta)
                if reason:
                    reasons.append(reason)
                    continue
                assert transformed is not None
                if transformed["target_role"] == "anti_negative":
                    anti_targets.append(transformed)
                elif transformed["target_role"] == "preserve_positive":
                    preserve_targets.append(transformed)

            if anti_targets or preserve_targets:
                unmatched_reason = ""
            else:
                unmatched_reason = reasons[0] if reasons else "no_matching_targets"

            image_key = identity["img_path"] or identity["tile_id"]
            batch_targets.append({
                "batch_index": batch_index,
                "image_key": image_key,
                "img_path": identity["img_path"],
                "parsed_tile_id": identity["tile_id"],
                "parsed_angle": identity["angle"],
                "anti_targets": anti_targets,
                "preserve_targets": preserve_targets,
                "target_meta": anti_targets + preserve_targets,
                "unmatched_reason": unmatched_reason,
            })
            debug_rows.append({
                "batch_index": batch_index,
                "img_path": identity["img_path"],
                "parsed_tile_id": identity["tile_id"],
                "parsed_angle": identity["angle"],
                "matched_targets": len(raw_candidates),
                "anti_targets": len(anti_targets),
                "preserve_targets": len(preserve_targets),
                "unmatched_reason": unmatched_reason,
            })
        self.last_debug_rows = debug_rows
        return batch_targets, debug_rows

    def write_debug_csv(self, path: str | Path,
                        rows: list[dict[str, Any]] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DEBUG_FIELDS)
            writer.writeheader()
            for row in rows if rows is not None else self.last_debug_rows:
                writer.writerow({field: row.get(field, "") for field in DEBUG_FIELDS})

    def to_jsonable_targets(
            self,
            batch_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in batch_targets:
            converted = dict(item)
            for key in ("anti_targets", "preserve_targets", "target_meta"):
                rows = []
                for row in converted.get(key, []):
                    row_out = dict(row)
                    row_out["target_polygon"] = [
                        [float(x), float(y)]
                        for x, y in row_out.get("target_polygon", [])
                    ]
                    rows.append(row_out)
                converted[key] = rows
            out.append(converted)
        return json.loads(json.dumps(out))
