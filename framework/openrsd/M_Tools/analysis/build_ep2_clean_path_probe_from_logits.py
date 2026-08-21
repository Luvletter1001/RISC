#!/usr/bin/env python
"""Normalize pre-NMS path logits into E-P2-clean probe CSV.

This script is the bridge between a model-side pre-NMS logit dump and
``evaluate_ep2_clean_path_intervention.py``. It does not run a detector. It
expects one row per case/path step/control with class-specific logits and
Gaussian z values, then writes canonical columns:

``logit_gt``, ``logit_hardneg``, ``z_gt``, ``z_hardneg``,
``hardneg_margin`` and ``support_advantage``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
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


def parse_pair(row):
    if row.get("gt_class") and row.get("hardneg_class"):
        return row["gt_class"], row["hardneg_class"], (
            row.get("pair") or f"{row['gt_class']}->{row['hardneg_class']}")
    pair = row.get("pair", "")
    if "->" not in pair:
        raise ValueError(
            "Each row needs either gt_class/hardneg_class or pair=a->b")
    gt_class, hardneg_class = pair.split("->", 1)
    return gt_class, hardneg_class, pair


def key_variants(prefix, class_name):
    safe = re.sub(r"[^0-9A-Za-z]+", "_", class_name).strip("_")
    return [
        f"{prefix}::{class_name}",
        f"{prefix}:{class_name}",
        f"{prefix}_{class_name}",
        f"{prefix}_{safe}",
    ]


def get_class_value(row, prefix, class_name, aliases=()):
    for key in aliases:
        if key in row and row[key] != "":
            return row[key]
    for key in key_variants(prefix, class_name):
        if key in row and row[key] != "":
            return row[key]
    raise KeyError(
        f"missing {prefix} value for class {class_name!r}; tried "
        f"{key_variants(prefix, class_name)}")


def get_z_value(row, class_name, gt_or_hardneg):
    abs_alias = f"abs_z_{gt_or_hardneg}"
    z_alias = f"z_{gt_or_hardneg}"
    if abs_alias in row and row[abs_alias] != "":
        abs_z = abs(to_float(row[abs_alias]))
        z_value = to_float(row.get(z_alias), abs_z)
        return z_value, abs_z
    try:
        abs_z = abs(to_float(get_class_value(row, "abs_z", class_name)))
        z_value = to_float(row.get(z_alias), abs_z)
        return z_value, abs_z
    except KeyError:
        pass
    z_raw = get_class_value(row, "z", class_name, aliases=(z_alias,))
    z_value = to_float(z_raw)
    return z_value, abs(z_value)


def build_path_probe_rows(raw_rows):
    rows = []
    for row in raw_rows:
        gt_class, hardneg_class, pair = parse_pair(row)
        logit_gt = to_float(get_class_value(
            row, "logit", gt_class, aliases=("logit_gt",)))
        logit_hardneg = to_float(get_class_value(
            row, "logit", hardneg_class, aliases=("logit_hardneg",)))
        z_gt, abs_z_gt = get_z_value(row, gt_class, "gt")
        z_hardneg, abs_z_hardneg = get_z_value(
            row, hardneg_class, "hardneg")
        hardneg_margin = logit_hardneg - logit_gt
        support_advantage = abs_z_gt - abs_z_hardneg
        rows.append({
            "case_id": row.get("case_id", ""),
            "pair": pair,
            "gt_class": gt_class,
            "hardneg_class": hardneg_class,
            "control_type": row.get("control_type", "clean") or "clean",
            "context_id": row.get("context_id", ""),
            "object_id": row.get("object_id", row.get("source_crop_id", "")),
            "path_step": row.get("path_step", ""),
            "log_area": row.get("log_area", ""),
            "logit_gt": fmt(logit_gt),
            "logit_hardneg": fmt(logit_hardneg),
            "z_gt": fmt(z_gt),
            "z_hardneg": fmt(z_hardneg),
            "abs_z_gt": fmt(abs_z_gt),
            "abs_z_hardneg": fmt(abs_z_hardneg),
            "hardneg_margin": fmt(hardneg_margin),
            "support_advantage": fmt(support_advantage),
        })
    return rows


def build_path_probe_from_logits(input_csv, output_csv):
    raw_rows = read_csv_rows(input_csv)
    rows = build_path_probe_rows(raw_rows)
    write_csv(output_csv, rows)
    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_loaded": len(raw_rows),
        "rows_written": len(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize pre-NMS logits into E-P2-clean path CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print(json.dumps(
        build_path_probe_from_logits(args.input_csv, args.output_csv),
        ensure_ascii=False))


if __name__ == "__main__":
    main()
