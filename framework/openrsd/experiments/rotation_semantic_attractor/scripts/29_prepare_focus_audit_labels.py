#!/usr/bin/env python3
"""Prepare corrected-FSV labels for FOCUS-OVD hard-negative training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "experiments/rotation_semantic_attractor/reports/visual_summary/audit/"
    "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv")
DEFAULT_OUT_DIR = Path("resultmd/exp_focus_ovd_20260608/audit_labels")

FALSE_SV_LABELS = {"non_vehicle_background", "non_vehicle_object_conflict"}
TRUE_VEHICLE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}
AMBIGUOUS_LABELS = {"", "ambiguous", "invalid_visualization"}
EXCLUDED_CATEGORIES = {"degenerate_large_sv_box", "padding_artifact"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify(row: dict[str, str]) -> tuple[str, str]:
    category = row.get("audit_category", "").strip()
    label = row.get("human_label", "").strip()
    valid = row.get("valid_for_human_audit", "").strip().lower()

    if category in EXCLUDED_CATEGORIES:
        return "excluded_failure_mode", category
    if valid not in {"true", "1", "yes"}:
        return "ambiguous_or_invalid", "not_valid_for_human_audit"
    if category != "valid_unmatched_sv":
        return "reference_only", category or "non_valid_unmatched_sv"
    if label in FALSE_SV_LABELS:
        return "corrected_false_sv", "audited_non_vehicle"
    if label in TRUE_VEHICLE_LABELS:
        return "annotation_missing_true_vehicle", "audited_true_vehicle"
    if label in AMBIGUOUS_LABELS:
        return "ambiguous_or_invalid", label or "missing_human_label"
    return "ambiguous_or_invalid", f"unsupported_label:{label}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in rows:
        focus_label, reason = classify(row)
        counts[focus_label] += 1
        reasons[reason] += 1
        out = dict(row)
        out["focus_label"] = focus_label
        out["focus_reason"] = reason
        out["use_as_hard_negative"] = str(
            focus_label == "corrected_false_sv").lower()
        out["use_as_preserve_positive"] = str(
            focus_label == "annotation_missing_true_vehicle").lower()
        out["excluded_from_hard_negative"] = str(
            focus_label != "corrected_false_sv").lower()
        out_rows.append(out)

    fields = list(out_rows[0].keys()) if out_rows else [
        "crop_id", "focus_label", "focus_reason"]
    out_csv = args.out_dir / "focus_audit_label_index.csv"
    write_rows(out_csv, out_rows, fields)

    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(out_csv),
        "total_rows": len(rows),
        "focus_label_counts": dict(counts),
        "reason_counts": dict(reasons),
        "training_policy": {
            "hard_negative": "corrected_false_sv only",
            "preserve_positive": "annotation_missing_true_vehicle only",
            "excluded": "ambiguous_or_invalid, reference_only, excluded_failure_mode",
            "degenerate_large_sv_and_padding": "excluded from hard negatives",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "focus_audit_label_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# FOCUS-OVD Audit Label Index",
        "",
        f"- Input CSV: `{args.input_csv}`",
        f"- Output CSV: `{out_csv}`",
        f"- Total rows: {len(rows)}",
        "",
        "| focus_label | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(counts.items()):
        md.append(f"| {key} | {value} |")
    md.extend([
        "",
        "Hard negatives are restricted to `corrected_false_sv`.",
        "`annotation_missing_true_vehicle` rows are preserve positives, not negatives.",
        "`degenerate_large_sv_box` and `padding_artifact` rows are excluded from hard negatives.",
    ])
    (args.out_dir / "focus_audit_label_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
