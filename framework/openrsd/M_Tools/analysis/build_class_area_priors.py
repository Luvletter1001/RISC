#!/usr/bin/env python
"""Build class-conditional area priors from DOTA-style annotation txt files."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_PERCENTILES = (1, 5, 10, 50, 90, 95, 99)


def qbox_area(values):
    pts = [float(v) for v in values]
    if len(pts) != 8:
        raise ValueError(f"Expected 8 qbox values, got {len(pts)}")
    xs = pts[0::2]
    ys = pts[1::2]
    total = 0.0
    for idx in range(4):
        nxt = (idx + 1) % 4
        total += xs[idx] * ys[nxt] - xs[nxt] * ys[idx]
    return abs(total) * 0.5


def qbox_edge_lengths(values):
    pts = [float(v) for v in values]
    if len(pts) != 8:
        raise ValueError(f"Expected 8 qbox values, got {len(pts)}")
    xs = pts[0::2]
    ys = pts[1::2]
    lengths = []
    for idx in range(4):
        nxt = (idx + 1) % 4
        dx = xs[nxt] - xs[idx]
        dy = ys[nxt] - ys[idx]
        lengths.append(math.hypot(dx, dy))
    return lengths


def qbox_log_aspect(values):
    lengths = [length for length in qbox_edge_lengths(values) if length > 0]
    if len(lengths) != 4:
        return 0.0
    long_side = max(lengths)
    short_side = max(min(lengths), 1e-6)
    return math.log(long_side / short_side)


def qbox_right_angle_error(values):
    pts = [float(v) for v in values]
    if len(pts) != 8:
        raise ValueError(f"Expected 8 qbox values, got {len(pts)}")
    xs = pts[0::2]
    ys = pts[1::2]
    errors = []
    for idx in range(4):
        prev_idx = (idx - 1) % 4
        next_idx = (idx + 1) % 4
        v1x = xs[prev_idx] - xs[idx]
        v1y = ys[prev_idx] - ys[idx]
        v2x = xs[next_idx] - xs[idx]
        v2y = ys[next_idx] - ys[idx]
        denom = math.hypot(v1x, v1y) * math.hypot(v2x, v2y)
        if denom <= 1e-6:
            continue
        errors.append(abs((v1x * v2x + v1y * v2y) / denom))
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def percentile(values, pct):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def _parse_class_names(class_names):
    if class_names is None:
        return None
    if isinstance(class_names, str):
        return {item.strip() for item in class_names.split(",") if item.strip()}
    return set(class_names)


def canonical_tile_id(tile_id):
    tile_id = str(tile_id)
    if tile_id.startswith("angle_") and "__" in tile_id:
        return tile_id.split("__", 1)[1]
    return tile_id


def source_id_from_tile_id(tile_id):
    tile_id = canonical_tile_id(tile_id)
    if "__" in tile_id:
        return tile_id.split("__", 1)[0]
    return tile_id


def annotation_file_ids(ann_dir):
    tile_ids = set()
    source_ids = set()
    for ann_file in sorted(Path(ann_dir).glob("*.txt")):
        tile_id = canonical_tile_id(ann_file.stem)
        tile_ids.add(tile_id)
        source_ids.add(source_id_from_tile_id(tile_id))
    return tile_ids, source_ids


def audit_annotation_overlap(prior_ann_dir, eval_ann_dir, example_limit=10):
    prior_tiles, prior_sources = annotation_file_ids(prior_ann_dir)
    eval_tiles, eval_sources = annotation_file_ids(eval_ann_dir)
    tile_overlap = sorted(prior_tiles & eval_tiles)
    source_overlap = sorted(prior_sources & eval_sources)
    return {
        "prior_ann_dir": str(Path(prior_ann_dir)),
        "eval_ann_dir": str(Path(eval_ann_dir)),
        "prior_tile_count": len(prior_tiles),
        "eval_tile_count": len(eval_tiles),
        "tile_overlap_count": len(tile_overlap),
        "eval_tile_overlap_rate": (
            len(tile_overlap) / len(eval_tiles) if eval_tiles else 0.0),
        "prior_source_count": len(prior_sources),
        "eval_source_count": len(eval_sources),
        "source_overlap_count": len(source_overlap),
        "eval_source_overlap_rate": (
            len(source_overlap) / len(eval_sources) if eval_sources else 0.0),
        "tile_overlap_examples": tile_overlap[:example_limit],
        "source_overlap_examples": source_overlap[:example_limit],
    }


def _exclude_sets(exclude_ann_dir):
    if exclude_ann_dir is None:
        return set(), set()
    return annotation_file_ids(exclude_ann_dir)


def _should_exclude_ann_file(ann_file, excluded_tiles, excluded_sources,
                             exclude_level):
    if exclude_level == "none":
        return False
    tile_id = canonical_tile_id(ann_file.stem)
    if exclude_level == "tile":
        return tile_id in excluded_tiles
    if exclude_level == "source":
        return source_id_from_tile_id(tile_id) in excluded_sources
    raise ValueError(f"Unsupported exclude_level: {exclude_level}")


def iter_dota_geometry(ann_dir, class_names=None, diff_thr=100,
                       exclude_ann_dir=None, exclude_level="none"):
    ann_dir = Path(ann_dir)
    allowed = _parse_class_names(class_names)
    excluded_tiles, excluded_sources = _exclude_sets(exclude_ann_dir)
    for ann_file in sorted(ann_dir.glob("*.txt")):
        if _should_exclude_ann_file(
                ann_file, excluded_tiles, excluded_sources, exclude_level):
            continue
        for raw in ann_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.split()
            if len(parts) < 9:
                continue
            cls = parts[8]
            if allowed is not None and cls not in allowed:
                continue
            difficulty = int(parts[9]) if len(parts) > 9 else 0
            if difficulty > diff_thr:
                continue
            area = qbox_area(parts[:8])
            if area <= 0:
                continue
            yield cls, {
                "area": area,
                "log_aspect": qbox_log_aspect(parts[:8]),
                "right_angle_error": qbox_right_angle_error(parts[:8]),
            }


def iter_dota_areas(ann_dir, class_names=None, diff_thr=100,
                    exclude_ann_dir=None, exclude_level="none"):
    for cls, geometry in iter_dota_geometry(
            ann_dir, class_names, diff_thr, exclude_ann_dir, exclude_level):
        yield cls, geometry["area"]


def build_class_area_prior_rows(ann_dir, class_names=None, diff_thr=100,
                                exclude_ann_dir=None, exclude_level="none"):
    by_class = defaultdict(list)
    for cls, geometry in iter_dota_geometry(
            ann_dir, class_names, diff_thr, exclude_ann_dir, exclude_level):
        by_class[cls].append(geometry)

    rows = []
    for cls in sorted(by_class):
        geometries = by_class[cls]
        areas = sorted(item["area"] for item in geometries)
        log_areas = [math.log(area) for area in areas]
        log_aspects = [item["log_aspect"] for item in geometries]
        angle_errors = [item["right_angle_error"] for item in geometries]
        log_mean = sum(log_areas) / len(log_areas)
        if len(log_areas) > 1:
            log_var = sum((x - log_mean) ** 2 for x in log_areas) / len(log_areas)
            log_std = math.sqrt(log_var)
        else:
            log_std = 0.0
        log_aspect_mean = sum(log_aspects) / len(log_aspects)
        if len(log_aspects) > 1:
            log_aspect_var = (
                sum((x - log_aspect_mean) ** 2 for x in log_aspects)
                / len(log_aspects))
            log_aspect_std = math.sqrt(log_aspect_var)
        else:
            log_aspect_std = 0.0
        rows.append({
            "class": cls,
            "count": len(areas),
            "min_area": areas[0],
            "p01_area": percentile(areas, 1),
            "p05_area": percentile(areas, 5),
            "p10_area": percentile(areas, 10),
            "median_area": percentile(areas, 50),
            "p90_area": percentile(areas, 90),
            "p95_area": percentile(areas, 95),
            "p99_area": percentile(areas, 99),
            "max_area": areas[-1],
            "log_area_mean": log_mean,
            "log_area_std": log_std,
            "log_aspect_mean": log_aspect_mean,
            "log_aspect_std": log_aspect_std,
            "p50_log_aspect": percentile(log_aspects, 50),
            "p90_log_aspect": percentile(log_aspects, 90),
            "right_angle_error_mean": sum(angle_errors) / len(angle_errors),
            "right_angle_error_p95": percentile(angle_errors, 95),
            "source_ann_dir": str(Path(ann_dir)),
            "diff_thr": int(diff_thr),
            "exclude_ann_dir": str(Path(exclude_ann_dir)) if exclude_ann_dir else "",
            "exclude_level": exclude_level,
        })
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class",
        "count",
        "min_area",
        "p01_area",
        "p05_area",
        "p10_area",
        "median_area",
        "p90_area",
        "p95_area",
        "p99_area",
        "max_area",
        "log_area_mean",
        "log_area_std",
        "log_aspect_mean",
        "log_aspect_std",
        "p50_log_aspect",
        "p90_log_aspect",
        "right_angle_error_mean",
        "right_angle_error_p95",
        "source_ann_dir",
        "diff_thr",
        "exclude_ann_dir",
        "exclude_level",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build class area priors from DOTA annotation txt files.")
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--diff-thr", type=int, default=100)
    parser.add_argument("--exclude-ann-dir", default=None)
    parser.add_argument(
        "--exclude-level",
        choices=["none", "tile", "source"],
        default="none")
    parser.add_argument("--audit-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_class_area_prior_rows(
        ann_dir=args.ann_dir,
        class_names=args.class_names,
        diff_thr=args.diff_thr,
        exclude_ann_dir=args.exclude_ann_dir,
        exclude_level=args.exclude_level,
    )
    write_csv(args.out_csv, rows)
    summary = {
        "ann_dir": args.ann_dir,
        "out_csv": args.out_csv,
        "class_count": len(rows),
        "object_count": sum(row["count"] for row in rows),
        "diff_thr": args.diff_thr,
        "exclude_ann_dir": args.exclude_ann_dir or "",
        "exclude_level": args.exclude_level,
        "classes": [row["class"] for row in rows],
    }
    if args.exclude_ann_dir:
        audit = audit_annotation_overlap(args.ann_dir, args.exclude_ann_dir)
        summary["overlap_audit"] = audit
        if args.audit_json:
            Path(args.audit_json).write_text(
                json.dumps(audit, indent=2, ensure_ascii=False) + os.linesep,
                encoding="utf-8")
    summary_json = args.summary_json
    if summary_json is None:
        summary_json = str(Path(args.out_csv).with_suffix(".summary.json"))
    Path(summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
