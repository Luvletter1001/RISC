#!/usr/bin/env python
"""Append CPU-only E-P2 shuffled-prior controls to a clean path CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SUPPORT_FIELDS = [
    "z_gt",
    "z_hardneg",
    "abs_z_gt",
    "abs_z_hardneg",
    "support_advantage",
]
EXTRA_FIELDS = ["control_note"]


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


def group_clean_cases(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("control_type") != "clean":
            continue
        case_id = row.get("case_id", "")
        pair = row.get("pair", "")
        if case_id and pair:
            grouped[(case_id, pair)].append(row)
    return grouped


def sort_path_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            to_float(row.get("path_step"), 0.0),
            to_float(row.get("log_area"), 0.0),
        ),
    )


def build_shuffled_prior_rows(rows, control_type="shuffled_prior"):
    controls = []
    for _, group in sorted(group_clean_cases(rows).items()):
        ordered = sort_path_rows(group)
        if len(ordered) < 2:
            continue
        reversed_support = list(reversed(ordered))
        for row, support_source in zip(ordered, reversed_support):
            out = dict(row)
            out["control_type"] = control_type
            out["control_note"] = "support_path_reversed"
            for field in SUPPORT_FIELDS:
                if field in support_source:
                    out[field] = support_source.get(field, "")
            controls.append(out)
    return controls


def append_shuffled_prior_control(input_csv,
                                  output_csv,
                                  control_type="shuffled_prior"):
    rows = read_csv_rows(input_csv)
    controls = build_shuffled_prior_rows(rows, control_type=control_type)
    all_rows = rows + controls
    write_csv_rows(
        output_csv,
        all_rows,
        fieldnames=collect_fieldnames(all_rows, EXTRA_FIELDS),
    )
    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "input_rows": len(rows),
        "control_rows": len(controls),
        "rows_written": len(all_rows),
        "control_type": control_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append shuffled-prior controls to E-P2 clean path CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--control-type", default="shuffled_prior")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = append_shuffled_prior_control(
        args.input_csv, args.output_csv, control_type=args.control_type)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
