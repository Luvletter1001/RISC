#!/usr/bin/env python3
"""Validate verified expanded manual labels for the tonight audit workspace."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_LABELS = {
    "valid_unmatched_sv": {
        "true_vehicle_annot_missing",
        "true_vehicle_annot_present_but_missed_match",
        "non_vehicle_background",
        "non_vehicle_object_conflict",
        "ambiguous",
        "invalid_visualization",
    },
    "true_sv_positive_control": {
        "true_vehicle_annot_present",
        "true_vehicle_annot_present_but_missed_match",
        "ambiguous",
        "invalid_visualization",
        "non_vehicle_background",
        "non_vehicle_object_conflict",
    },
    "degenerate_large_sv_box": {
        "actual_large_sv_degenerate_prediction",
        "padding_or_boundary_related",
        "large_context_texture",
        "object_conflict",
        "uncertain",
    },
    "padding_artifact": {
        "valid_padding_artifact",
        "not_padding_artifact",
        "uncertain",
    },
    "strict_object_flip": {
        "good_qualitative_case",
        "not_clear",
        "wrong_or_ambiguous",
    },
}

VALID_CONFIDENCE = {"", "high", "medium", "low"}
REQUIRED_CATEGORIES = {
    "valid_unmatched_sv": 300,
    "strict_object_flip": 100,
    "true_sv_positive_control": 100,
    "degenerate_large_sv_box": 150,
    "padding_artifact": 100,
}

SUBSET_FILES = [
    "valid_unmatched_sv_300_for_corrected_fsv.csv",
    "true_sv_positive_control_100_for_qc.csv",
    "degenerate_large_sv_box_60_spotcheck.csv",
    "padding_artifact_30_spotcheck.csv",
    "strict_object_flip_30_qualitative_candidates.csv",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def md_table(rows: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    if headers is None:
        headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def merge_subset_labels(rows: list[dict[str, str]], manual_audit_dir: Path | None) -> tuple[list[dict[str, str]], list[str]]:
    if not manual_audit_dir:
        return rows, []
    by_id = {r.get("crop_id", ""): dict(r) for r in rows}
    sources = []
    for name in SUBSET_FILES:
        path = manual_audit_dir / name
        if not path.exists():
            continue
        subset_rows = read_csv(path)
        used = 0
        for row in subset_rows:
            cid = row.get("crop_id", "")
            if cid not in by_id:
                continue
            if row.get("human_label", "").strip():
                by_id[cid]["human_label"] = row.get("human_label", "").strip()
                by_id[cid]["human_confidence"] = row.get("human_confidence", "").strip()
                by_id[cid]["human_notes"] = row.get("human_notes", "").strip()
                used += 1
        if used:
            sources.append(f"{path}:{used}")
    return list(by_id.values()), sources


def validate(
    label_csv: Path,
    min_valid_unmatched_labels: int,
    allow_partial: bool,
    manual_audit_dir: Path | None = None,
) -> dict[str, Any]:
    rows = read_csv(label_csv)
    rows, effective_sources = merge_subset_labels(rows, manual_audit_dir)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if label_csv.name == "human_sv_crop_audit_labels.csv":
        errors.append({"type": "old_label_csv_rejected", "path": str(label_csv)})
    if "VERIFIED_EXPANDED" not in label_csv.name:
        errors.append({"type": "not_verified_expanded_label_csv", "path": str(label_csv)})
    if not label_csv.exists():
        errors.append({"type": "label_csv_missing", "path": str(label_csv)})

    crop_ids = [r.get("crop_id", "") for r in rows]
    duplicate_ids = sorted([k for k, v in Counter(crop_ids).items() if k and v > 1])
    if duplicate_ids:
        errors.append({"type": "duplicate_crop_id", "count": len(duplicate_ids), "sample": duplicate_ids[:20]})

    category_counts = Counter(r.get("audit_category", "") for r in rows)
    for category, expected in REQUIRED_CATEGORIES.items():
        if category_counts.get(category, 0) != expected:
            errors.append({
                "type": "category_count_mismatch",
                "category": category,
                "expected": expected,
                "actual": category_counts.get(category, 0),
            })

    missing_paths = []
    invalid_labels = []
    invalid_conf = []
    degenerate_mislabels = []
    padding_labels = []
    corrected_forbidden = []
    for row in rows:
        cid = row.get("crop_id", "")
        cat = row.get("audit_category", "")
        label = row.get("human_label", "").strip()
        conf = row.get("human_confidence", "").strip()
        if label and label not in VALID_LABELS.get(cat, set()):
            invalid_labels.append({"crop_id": cid, "audit_category": cat, "human_label": label})
        if conf not in VALID_CONFIDENCE:
            invalid_conf.append({"crop_id": cid, "human_confidence": conf})
        for col in ["image_path_full_tile", "image_path_zoom", "image_path_gt_context", "metadata_json"]:
            p = Path(row.get(col, "") or "__missing__")
            if not p.exists():
                missing_paths.append({"crop_id": cid, "column": col, "path": str(p)})
        if cat == "degenerate_large_sv_box" and label in {
            "true_vehicle_annot_missing",
            "true_vehicle_annot_present_but_missed_match",
        }:
            degenerate_mislabels.append({"crop_id": cid, "human_label": label})
        if cat == "padding_artifact" and label:
            padding_labels.append({"crop_id": cid, "human_label": label})
        if cat in {"degenerate_large_sv_box", "padding_artifact", "strict_object_flip", "true_sv_positive_control"} and label in {
            "true_vehicle_annot_missing",
            "true_vehicle_annot_present_but_missed_match",
            "non_vehicle_background",
            "non_vehicle_object_conflict",
        }:
            corrected_forbidden.append({"crop_id": cid, "audit_category": cat, "human_label": label})

    if invalid_labels:
        errors.append({"type": "invalid_human_label", "count": len(invalid_labels), "sample": invalid_labels[:30]})
    if invalid_conf:
        errors.append({"type": "invalid_human_confidence", "count": len(invalid_conf), "sample": invalid_conf[:30]})
    if missing_paths:
        errors.append({"type": "missing_image_or_metadata_path", "count": len(missing_paths), "sample": missing_paths[:30]})
    if degenerate_mislabels:
        errors.append({"type": "degenerate_large_sv_labeled_as_true_vehicle_for_correction", "count": len(degenerate_mislabels), "sample": degenerate_mislabels[:30]})

    valid_unmatched = [r for r in rows if r.get("audit_category") == "valid_unmatched_sv"]
    bad_valid_flag = [r.get("crop_id", "") for r in valid_unmatched if not is_true(r.get("valid_for_human_audit", ""))]
    bad_coord = [r.get("crop_id", "") for r in valid_unmatched if not is_true(r.get("coordinate_frame_consistent", ""))]
    if bad_valid_flag:
        errors.append({"type": "valid_unmatched_not_valid_for_human", "count": len(bad_valid_flag), "sample": bad_valid_flag[:30]})
    if bad_coord:
        errors.append({"type": "valid_unmatched_coordinate_mismatch", "count": len(bad_coord), "sample": bad_coord[:30]})
    if padding_labels:
        warnings.append({"type": "padding_artifact_labels_present_not_for_corrected_fsv", "count": len(padding_labels), "sample": padding_labels[:30]})
    if corrected_forbidden:
        warnings.append({"type": "non_corrected_categories_have_corrected_fsv_style_labels", "count": len(corrected_forbidden), "sample": corrected_forbidden[:30]})

    labeled_by_category = Counter()
    label_distribution = Counter()
    for row in rows:
        label = row.get("human_label", "").strip()
        if label:
            labeled_by_category[row.get("audit_category", "")] += 1
            label_distribution[(row.get("audit_category", ""), label)] += 1

    valid_labeled = labeled_by_category.get("valid_unmatched_sv", 0)
    true_control_labeled = labeled_by_category.get("true_sv_positive_control", 0)
    if errors:
        status = "BLOCKED"
    elif valid_labeled >= min_valid_unmatched_labels:
        status = "READY_FOR_CORRECTED_FSV"
    elif valid_labeled > 0:
        status = "PARTIAL_LABELS" if allow_partial else "BLOCKED"
    else:
        status = "HUMAN_LABELS_REQUIRED"

    distribution_rows = []
    for category in sorted(REQUIRED_CATEGORIES):
        total = category_counts.get(category, 0)
        labeled = labeled_by_category.get(category, 0)
        distribution_rows.append({
            "audit_category": category,
            "human_label": "__LABELED_TOTAL__",
            "count": labeled,
            "category_total": total,
        })
        for label in sorted(VALID_LABELS.get(category, [])):
            distribution_rows.append({
                "audit_category": category,
                "human_label": label,
                "count": label_distribution.get((category, label), 0),
                "category_total": total,
            })

    return {
        "status": status,
        "label_csv": str(label_csv),
        "row_count": len(rows),
        "category_counts": dict(category_counts),
        "valid_unmatched_labeled": valid_labeled,
        "true_sv_positive_control_labeled": true_control_labeled,
        "min_valid_unmatched_labels": min_valid_unmatched_labels,
        "allow_partial": allow_partial,
        "effective_subset_label_sources": effective_sources,
        "errors": errors,
        "warnings": warnings,
        "distribution_rows": distribution_rows,
        "effective_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-valid-unmatched-labels", type=int, default=300)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--manual-audit-dir", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manual_audit_dir = Path(args.manual_audit_dir) if args.manual_audit_dir else output_dir
    result = validate(Path(args.label_csv), args.min_valid_unmatched_labels, args.allow_partial, manual_audit_dir)
    distribution_rows = result.pop("distribution_rows")
    effective_rows = result.pop("effective_rows")

    write_json(output_dir / "label_validation_report.json", result)
    write_csv(output_dir / "label_distribution.csv", distribution_rows)
    write_csv(output_dir / "label_validation_effective_labels.csv", effective_rows)
    md = [
        "# Label Validation Report",
        "",
        f"- status: `{result['status']}`",
        f"- label_csv: `{result['label_csv']}`",
        f"- row_count: `{result['row_count']}`",
        f"- valid_unmatched_labeled: `{result['valid_unmatched_labeled']}`",
        f"- true_sv_positive_control_labeled: `{result['true_sv_positive_control_labeled']}`",
        f"- effective_subset_label_sources: `{result['effective_subset_label_sources']}`",
        "",
        "## Category Counts",
        "",
        md_table([
            {"audit_category": k, "count": v, "expected": REQUIRED_CATEGORIES.get(k, "")}
            for k, v in sorted(result["category_counts"].items())
        ]),
        "",
        "## Errors",
        "",
        md_table(result["errors"]),
        "",
        "## Warnings",
        "",
        md_table(result["warnings"]),
    ]
    (output_dir / "label_validation_report.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "valid_unmatched_labeled", "true_sv_positive_control_labeled"]}, ensure_ascii=False))
    return 0 if result["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
