#!/usr/bin/env python3
"""Run corrected-FSV estimation after verified expanded manual labels exist."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


TRUE_VEHICLE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}
CORRECTED_FALSE_LABELS = {
    "non_vehicle_background",
    "non_vehicle_object_conflict",
}
EXCLUDED_LABELS = {"ambiguous", "invalid_visualization"}
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


def is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def merge_subset_labels(rows: list[dict[str, str]], manual_audit_dir: Path) -> list[dict[str, str]]:
    by_id = {r.get("crop_id", ""): dict(r) for r in rows}
    for name in SUBSET_FILES:
        path = manual_audit_dir / name
        if not path.exists():
            continue
        for row in read_csv(path):
            cid = row.get("crop_id", "")
            if cid in by_id and row.get("human_label", "").strip():
                by_id[cid]["human_label"] = row.get("human_label", "").strip()
                by_id[cid]["human_confidence"] = row.get("human_confidence", "").strip()
                by_id[cid]["human_notes"] = row.get("human_notes", "").strip()
    return list(by_id.values())


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def bootstrap_rate(labels: list[str], seed: int, iters: int = 10000) -> dict[str, Any]:
    decidable = [x for x in labels if x in TRUE_VEHICLE_LABELS or x in CORRECTED_FALSE_LABELS]
    if not decidable:
        return {"status": "BLOCKED_NO_DECIDABLE_LABELS", "iters": 0}
    rng = random.Random(seed)
    rates = []
    n = len(decidable)
    for _ in range(iters):
        sample = [decidable[rng.randrange(n)] for _ in range(n)]
        false_count = sum(1 for x in sample if x in CORRECTED_FALSE_LABELS)
        rates.append(false_count / n)
    return {
        "status": "DONE",
        "iters": iters,
        "metric": "sample_corrected_false_rate_among_decidable_valid_unmatched_sv",
        "ci95_low": percentile(rates, 0.025),
        "ci95_high": percentile(rates, 0.975),
    }


def run_validation(repo_root: Path, label_csv: Path, output_dir: Path, manual_audit_dir: Path) -> dict[str, Any]:
    validator = repo_root / "experiments/rotation_semantic_attractor/scripts/26_validate_tonight_manual_labels.py"
    validation_dir = output_dir / "validation"
    cmd = [
        sys.executable,
        str(validator),
        "--label-csv",
        str(label_csv),
        "--output-dir",
        str(validation_dir),
        "--manual-audit-dir",
        str(manual_audit_dir),
        "--allow-partial",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report_path = validation_dir / "label_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "status": "BLOCKED",
        "errors": [{"type": "validation_script_failed", "stdout": proc.stdout, "stderr": proc.stderr}],
    }
    report["validation_returncode"] = proc.returncode
    return report


def run_legacy_19(repo_root: Path, label_csv: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    script19 = repo_root / "experiments/rotation_semantic_attractor/scripts/19_audit_unannotated_sv_contamination.py"
    legacy_dir = output_dir / "legacy_19_audit"
    cmd = [
        sys.executable,
        str(script19),
        "--repo-root",
        str(repo_root),
        "--human-label-csv",
        str(label_csv),
        "--output-dir",
        str(legacy_dir),
        "--sample-size",
        "500",
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output_dir": str(legacy_dir),
    }


def estimate(label_csv: Path, manual_audit_dir: Path, status: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = merge_subset_labels(read_csv(label_csv), manual_audit_dir)
    usable = [
        r for r in rows
        if r.get("audit_category") == "valid_unmatched_sv"
        and is_true(r.get("valid_for_human_audit", ""))
        and is_true(r.get("coordinate_frame_consistent", ""))
        and r.get("human_label", "").strip()
    ]
    labels = [r.get("human_label", "").strip() for r in usable]
    labeled = len(labels)
    true_vehicle = sum(1 for x in labels if x in TRUE_VEHICLE_LABELS)
    corrected_false = sum(1 for x in labels if x in CORRECTED_FALSE_LABELS)
    ambiguous = sum(1 for x in labels if x == "ambiguous")
    invalid_visualization = sum(1 for x in labels if x == "invalid_visualization")
    decidable = true_vehicle + corrected_false
    if labeled == 0:
        corrected_status = "HUMAN_LABELS_REQUIRED"
    elif labeled < 100:
        corrected_status = "WARNING_SMALL_SAMPLE"
    elif labeled < 200:
        corrected_status = "SANITY_ONLY_CORRECTED_FSV"
    elif labeled < 300:
        corrected_status = "MAIN_ESTIMATE_WITH_WIDE_CI"
    else:
        corrected_status = "PAPER_READY_CORRECTED_FSV"
    if status == "BLOCKED":
        corrected_status = "BLOCKED"
    rate = corrected_false / decidable if decidable else None
    annotation_missing_rate = true_vehicle / decidable if decidable else None
    ci = bootstrap_rate(labels, seed) if decidable else {"status": "BLOCKED_NO_DECIDABLE_LABELS", "iters": 0}
    summary = {
        "status": corrected_status,
        "validation_status": status,
        "label_csv": str(label_csv),
        "total_valid_unmatched_rows": sum(1 for r in rows if r.get("audit_category") == "valid_unmatched_sv"),
        "labeled_valid_unmatched_rows": labeled,
        "decidable_valid_unmatched_rows": decidable,
        "true_vehicle_deducted_count": true_vehicle,
        "corrected_false_sv_count": corrected_false,
        "ambiguous_excluded_count": ambiguous,
        "invalid_visualization_excluded_count": invalid_visualization,
        "annotation_missing_or_missed_match_rate": annotation_missing_rate,
        "sample_corrected_false_rate_among_decidable": rate,
        "paper_caveat": "This is a sampled verified-expanded corrected-FSV estimate; it is not AP/mAP and excludes ambiguous/invalid rows.",
    }
    return summary, ci


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    label_csv = Path(args.label_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_audit_dir = output_dir.parent

    validation = run_validation(repo_root, label_csv, output_dir, manual_audit_dir)
    if validation.get("status") == "BLOCKED":
        summary = {
            "status": "BLOCKED",
            "reason": "label validation blocked corrected-FSV",
            "validation": validation,
        }
        write_csv(output_dir / "corrected_fsv_verified_expanded.csv", [summary])
        write_json(output_dir / "corrected_fsv_bootstrap_ci.json", {"status": "BLOCKED"})
        (output_dir / "corrected_fsv_verified_expanded_summary.md").write_text(
            "# Corrected-FSV Verified Expanded Summary\n\n"
            "- status: `BLOCKED`\n"
            "- reason: label validation blocked corrected-FSV.\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "BLOCKED"}, ensure_ascii=False))
        return 0

    legacy = run_legacy_19(repo_root, label_csv, output_dir, args.seed)
    summary, ci = estimate(label_csv, manual_audit_dir, validation.get("status", "BLOCKED"), args.seed)
    summary["legacy_19_audit"] = legacy.get("output_dir", "")
    write_csv(output_dir / "corrected_fsv_verified_expanded.csv", [summary])
    write_json(output_dir / "corrected_fsv_bootstrap_ci.json", ci)
    md = [
        "# Corrected-FSV Verified Expanded Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_status: `{summary['validation_status']}`",
        f"- label_csv: `{summary['label_csv']}`",
        f"- total_valid_unmatched_rows: `{summary['total_valid_unmatched_rows']}`",
        f"- labeled_valid_unmatched_rows: `{summary['labeled_valid_unmatched_rows']}`",
        f"- decidable_valid_unmatched_rows: `{summary['decidable_valid_unmatched_rows']}`",
        f"- true_vehicle_deducted_count: `{summary['true_vehicle_deducted_count']}`",
        f"- corrected_false_sv_count: `{summary['corrected_false_sv_count']}`",
        f"- ambiguous_excluded_count: `{summary['ambiguous_excluded_count']}`",
        f"- invalid_visualization_excluded_count: `{summary['invalid_visualization_excluded_count']}`",
        f"- annotation_missing_or_missed_match_rate: `{summary['annotation_missing_or_missed_match_rate']}`",
        f"- sample_corrected_false_rate_among_decidable: `{summary['sample_corrected_false_rate_among_decidable']}`",
        "",
        "This estimate uses only verified `valid_unmatched_sv` rows with non-empty human labels. It excludes ambiguous and invalid visualization rows, and it never uses degenerate, padding, strict-object-flip, or true-SV-control rows for corrected-FSV.",
    ]
    (output_dir / "corrected_fsv_verified_expanded_summary.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "labeled": summary["labeled_valid_unmatched_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
