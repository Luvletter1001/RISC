#!/usr/bin/env python
"""Filter E-P2 pre-NMS logits to the target object path.

The model-side E-P2 dump records top hard-negative locations. For causal/path
evidence, the auditor needs the same target object across scale steps rather
than whichever dense location currently has the largest hard-negative logit.
This script joins the raw pre-NMS dump with the E9/E-P2 variant CSV, then keeps
the candidate box with maximum IoU to the variant target polygon for each
``case_id + control_type + class pair`` group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M_AD.models.utils.ep2_path_probe_dump import parse_ep2_metadata_from_path


EXTRA_FIELDNAMES = [
    "source_control_type",
    "path_variant",
    "target_iou",
    "target_qbox",
    "candidate_qbox",
    "target_pair_from_e9",
    "e9_variant",
    "e9_variant_image_path",
]
PATH_VARIANT_CONTROLS = {
    "original_tile",
    "neutral_same_scale",
    "neutral_pred_class_scale",
}


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_fieldnames(rows, extra=()):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for key in extra:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = collect_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value):
    value = float(value)
    if value.is_integer():
        return f"{value:.1f}"
    text = f"{value:.12g}"
    return "0" if text == "-0" else text


def parse_float_list(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    text = str(value or "")
    return [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def as_points(qbox: Sequence[float]) -> list[tuple[float, float]]:
    values = parse_float_list(qbox)
    if len(values) != 8:
        raise ValueError(f"qbox needs 8 numbers, got {len(values)}: {qbox!r}")
    return [(values[idx], values[idx + 1]) for idx in range(0, 8, 2)]


def flatten_points(points: Iterable[Sequence[float]]) -> list[float]:
    values = []
    for x, y in points:
        values.extend([float(x), float(y)])
    return values


def signed_polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(signed_polygon_area(points))


def ensure_ccw(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    if signed_polygon_area(pts) < 0:
        pts.reverse()
    return pts


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _inside(point, edge_start, edge_end, eps=1e-9):
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end
    return cross(bx - ax, by - ay, px - ax, py - ay) >= -eps


def _line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    denom = cross(dx1, dy1, dx2, dy2)
    if abs(denom) < 1e-12:
        return p2
    t = cross(x3 - x1, y3 - y1, dx2, dy2) / denom
    return x1 + t * dx1, y1 + t * dy1


def convex_polygon_intersection(subject, clip):
    output = ensure_ccw(subject)
    clip = ensure_ccw(clip)
    for idx, edge_start in enumerate(clip):
        edge_end = clip[(idx + 1) % len(clip)]
        current = output
        output = []
        if not current:
            break
        previous = current[-1]
        for point in current:
            point_inside = _inside(point, edge_start, edge_end)
            previous_inside = _inside(previous, edge_start, edge_end)
            if point_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, point, edge_start,
                                           edge_end))
                output.append(point)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, point, edge_start, edge_end))
            previous = point
    return output


def polygon_iou(qbox_a, qbox_b) -> float:
    poly_a = ensure_ccw(as_points(qbox_a))
    poly_b = ensure_ccw(as_points(qbox_b))
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter = polygon_area(convex_polygon_intersection(poly_a, poly_b))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def rbox_to_qbox(cx, cy, width, height, angle) -> list[float]:
    cx = float(cx)
    cy = float(cy)
    width = abs(float(width))
    height = abs(float(height))
    angle = float(angle)
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    vec1 = (width / 2.0 * cos_value, width / 2.0 * sin_value)
    vec2 = (-height / 2.0 * sin_value, height / 2.0 * cos_value)
    points = [
        (cx + vec1[0] + vec2[0], cy + vec1[1] + vec2[1]),
        (cx + vec1[0] - vec2[0], cy + vec1[1] - vec2[1]),
        (cx - vec1[0] - vec2[0], cy - vec1[1] - vec2[1]),
        (cx - vec1[0] + vec2[0], cy - vec1[1] + vec2[1]),
    ]
    return flatten_points(points)


def row_pair(row):
    if row.get("pair"):
        return row["pair"]
    if row.get("gt_class") and row.get("hardneg_class"):
        return f"{row['gt_class']}->{row['hardneg_class']}"
    return ""


def e9_raw_pair(row):
    gt_class = row.get("target_gt_class", "")
    hardneg_class = row.get("impossible_pred_class", "")
    if gt_class and hardneg_class:
        return f"{gt_class}->{hardneg_class}"
    target_pair = row.get("target_pair", "")
    if "->" in target_pair:
        hardneg_class, gt_class = target_pair.split("->", 1)
        return f"{gt_class}->{hardneg_class}"
    return row_pair(row)


def build_target_index(e9_rows):
    targets = {}
    for row in e9_rows:
        qbox = parse_float_list(row.get("target_qbox", ""))
        if len(qbox) != 8:
            continue
        meta = parse_ep2_metadata_from_path(row.get("variant_image_path"))
        case_id = (
            meta.get("ep2_case_id") or row.get("case_id")
            or row.get("object_id") or row.get("source_crop_id") or "")
        control_type = (
            meta.get("ep2_control_type") or row.get("control_type")
            or row.get("variant") or "clean")
        pair = e9_raw_pair(row)
        if not case_id or not control_type or not pair:
            continue
        targets[(case_id, control_type, pair)] = {
            "target_qbox": qbox,
            "target_pair_from_e9": row.get("target_pair", ""),
            "e9_variant": row.get("variant", control_type),
            "e9_variant_image_path": row.get("variant_image_path", ""),
        }
    return targets


def row_qbox(row):
    values = [
        row.get("bbox_cx"),
        row.get("bbox_cy"),
        row.get("bbox_w"),
        row.get("bbox_h"),
        row.get("bbox_angle", 0),
    ]
    if any(value in {None, ""} for value in values[:4]):
        return None
    return rbox_to_qbox(*(to_float(value) for value in values))


def group_raw_rows(raw_rows):
    groups = defaultdict(list)
    for row in raw_rows:
        key = (row.get("case_id", ""), row.get("control_type", "clean"),
               row_pair(row))
        if all(key):
            groups[key].append(row)
    return groups


def filter_ep2_pre_nms_logits_by_target_iou(raw_csv,
                                            e9_csv,
                                            output_csv,
                                            min_iou=0.1):
    raw_rows = read_csv_rows(raw_csv)
    e9_rows = read_csv_rows(e9_csv)
    targets = build_target_index(e9_rows)
    groups = group_raw_rows(raw_rows)
    filtered_rows = []
    groups_seen = 0
    unmatched_groups = 0
    for key, rows in sorted(groups.items()):
        target = targets.get(key)
        if target is None:
            continue
        groups_seen += 1
        best_row = None
        best_iou = -1.0
        best_qbox = None
        for row in rows:
            candidate_qbox = row_qbox(row)
            if candidate_qbox is None:
                continue
            iou = polygon_iou(candidate_qbox, target["target_qbox"])
            if iou > best_iou:
                best_iou = iou
                best_row = row
                best_qbox = candidate_qbox
        if best_row is None or best_iou < float(min_iou):
            unmatched_groups += 1
            continue
        out_row = dict(best_row)
        source_control_type = out_row.get("control_type", "")
        path_variant = target["e9_variant"] or source_control_type
        if source_control_type in PATH_VARIANT_CONTROLS:
            out_row["control_type"] = "clean"
        out_row.update({
            "source_control_type": source_control_type,
            "path_variant": path_variant,
            "target_iou": fmt(best_iou),
            "target_qbox": " ".join(fmt(v) for v in target["target_qbox"]),
            "candidate_qbox": " ".join(fmt(v) for v in best_qbox),
            "target_pair_from_e9": target["target_pair_from_e9"],
            "e9_variant": target["e9_variant"],
            "e9_variant_image_path": target["e9_variant_image_path"],
        })
        filtered_rows.append(out_row)

    fieldnames = collect_fieldnames(raw_rows, EXTRA_FIELDNAMES)
    write_csv_rows(output_csv, filtered_rows, fieldnames=fieldnames)
    return {
        "raw_csv": str(raw_csv),
        "e9_csv": str(e9_csv),
        "output_csv": str(output_csv),
        "raw_rows": len(raw_rows),
        "e9_rows": len(e9_rows),
        "target_groups": len(targets),
        "groups_seen": groups_seen,
        "rows_written": len(filtered_rows),
        "unmatched_groups": unmatched_groups,
        "min_iou": float(min_iou),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep target-object pre-NMS logits by E9 target IoU.")
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--e9-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-iou", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = filter_ep2_pre_nms_logits_by_target_iou(
        args.raw_csv, args.e9_csv, args.output_csv, min_iou=args.min_iou)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
