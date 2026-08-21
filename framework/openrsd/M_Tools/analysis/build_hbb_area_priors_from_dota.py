#!/usr/bin/env python
"""Build class-conditional HBB area priors from DOTA-style qbox labels."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def qbox_hbb_area(values):
    pts = [float(v) for v in values]
    if len(pts) != 8:
        raise ValueError(f"Expected 8 qbox values, got {len(pts)}")
    xs = pts[0::2]
    ys = pts[1::2]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return max(width, 0.0) * max(height, 0.0)


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


def parse_class_names(raw):
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def iter_areas(ann_dir, class_names=None, diff_thr=100):
    ann_dir = Path(ann_dir)
    allowed = parse_class_names(class_names)
    for ann_file in sorted(ann_dir.glob("*.txt")):
        for raw in ann_file.read_text(
                encoding="utf-8", errors="ignore").splitlines():
            parts = raw.split()
            if len(parts) < 9:
                continue
            cls = parts[8]
            if allowed is not None and cls not in allowed:
                continue
            difficulty = int(parts[9]) if len(parts) > 9 else 0
            if difficulty > diff_thr:
                continue
            area = qbox_hbb_area(parts[:8])
            if area > 0:
                yield cls, area


def build_rows(ann_dir, class_names=None, diff_thr=100):
    by_class = defaultdict(list)
    for cls, area in iter_areas(ann_dir, class_names, diff_thr):
        by_class[cls].append(area)

    rows = []
    for cls in sorted(by_class):
        areas = sorted(by_class[cls])
        log_areas = [math.log(area) for area in areas]
        log_mean = sum(log_areas) / len(log_areas)
        if len(log_areas) > 1:
            log_var = sum((x - log_mean) ** 2 for x in log_areas) / len(log_areas)
            log_std = math.sqrt(log_var)
        else:
            log_std = 0.0
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
            "source_ann_dir": str(Path(ann_dir)),
            "diff_thr": int(diff_thr),
            "area_mode": "enclosing_hbb_from_qbox",
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
        "source_ann_dir",
        "diff_thr",
        "area_mode",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--diff-thr", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(args.ann_dir, args.class_names, args.diff_thr)
    write_csv(args.out_csv, rows)
    summary = {
        "ann_dir": args.ann_dir,
        "out_csv": args.out_csv,
        "class_count": len(rows),
        "object_count": sum(row["count"] for row in rows),
        "diff_thr": int(args.diff_thr),
        "area_mode": "enclosing_hbb_from_qbox",
        "classes": [row["class"] for row in rows],
    }
    summary_json = args.summary_json or str(Path(args.out_csv).with_suffix(".summary.json"))
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
