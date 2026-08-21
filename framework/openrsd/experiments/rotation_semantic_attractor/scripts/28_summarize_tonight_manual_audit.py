#!/usr/bin/env python3
"""Summarize tonight's verified expanded manual audit labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def merge_subset_labels(master: list[dict[str, str]], output_dir: Path) -> list[dict[str, str]]:
    by_id = {r.get("crop_id", ""): dict(r) for r in master}
    subset_files = [
        "valid_unmatched_sv_300_for_corrected_fsv.csv",
        "true_sv_positive_control_100_for_qc.csv",
        "degenerate_large_sv_box_60_spotcheck.csv",
        "padding_artifact_30_spotcheck.csv",
        "strict_object_flip_30_qualitative_candidates.csv",
    ]
    for name in subset_files:
        for row in read_csv(output_dir / name):
            cid = row.get("crop_id", "")
            if cid in by_id and row.get("human_label", "").strip():
                by_id[cid]["human_label"] = row.get("human_label", "").strip()
                by_id[cid]["human_confidence"] = row.get("human_confidence", "").strip()
                by_id[cid]["human_notes"] = row.get("human_notes", "").strip()
    return list(by_id.values())


def label_counts(rows: list[dict[str, str]], category: str) -> Counter[str]:
    c = Counter()
    for row in rows:
        if row.get("audit_category") == category:
            label = row.get("human_label", "").strip()
            if label:
                c[label] += 1
    return c


def summarize_category(rows: list[dict[str, str]], category: str) -> dict[str, Any]:
    subset = [r for r in rows if r.get("audit_category") == category]
    labels = label_counts(rows, category)
    out: dict[str, Any] = {
        "total": len(subset),
        "labeled": sum(labels.values()),
    }
    out.update(dict(labels))
    return out


def read_corrected(corrected_fsv_dir: Path) -> dict[str, Any]:
    rows = read_csv(corrected_fsv_dir / "corrected_fsv_verified_expanded.csv")
    if not rows:
        return {"status": "HUMAN_LABELS_REQUIRED"}
    row = rows[0]
    return {
        "status": row.get("status", ""),
        "sample_corrected_false_rate_among_decidable": row.get("sample_corrected_false_rate_among_decidable", ""),
        "annotation_missing_or_missed_match_rate": row.get("annotation_missing_or_missed_match_rate", ""),
        "labeled_valid_unmatched_rows": row.get("labeled_valid_unmatched_rows", ""),
        "decidable_valid_unmatched_rows": row.get("decidable_valid_unmatched_rows", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--corrected-fsv-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = merge_subset_labels(read_csv(Path(args.label_csv)), output_dir)
    corrected = read_corrected(Path(args.corrected_fsv_dir))

    valid = summarize_category(rows, "valid_unmatched_sv")
    true_control = summarize_category(rows, "true_sv_positive_control")
    degenerate = summarize_category(rows, "degenerate_large_sv_box")
    padding = summarize_category(rows, "padding_artifact")
    strict = summarize_category(rows, "strict_object_flip")

    true_vehicle = valid.get("true_vehicle_annot_missing", 0) + valid.get("true_vehicle_annot_present_but_missed_match", 0)
    valid_decidable = true_vehicle + valid.get("non_vehicle_background", 0) + valid.get("non_vehicle_object_conflict", 0)
    annotation_missing_rate = true_vehicle / valid_decidable if valid_decidable else None
    clear_true = true_control.get("true_vehicle_annot_present", 0) + true_control.get("true_vehicle_annot_present_but_missed_match", 0)
    suspicious = true_control.get("non_vehicle_background", 0) + true_control.get("non_vehicle_object_conflict", 0) + true_control.get("invalid_visualization", 0)
    if true_control["labeled"] == 0:
        qc_verdict = "PENDING"
    elif clear_true / max(1, true_control["labeled"]) >= 0.9:
        qc_verdict = "PASS"
    else:
        qc_verdict = "REVIEW_NEEDED"
    good_strict_ids = [
        r.get("crop_id", "") for r in rows
        if r.get("audit_category") == "strict_object_flip" and r.get("human_label", "").strip() == "good_qualitative_case"
    ]

    summary = {
        "valid_unmatched_sv": {
            **valid,
            "true_vehicle_annot_missing": valid.get("true_vehicle_annot_missing", 0),
            "true_vehicle_annot_present_but_missed_match": valid.get("true_vehicle_annot_present_but_missed_match", 0),
            "non_vehicle_background": valid.get("non_vehicle_background", 0),
            "non_vehicle_object_conflict": valid.get("non_vehicle_object_conflict", 0),
            "ambiguous": valid.get("ambiguous", 0),
            "invalid_visualization": valid.get("invalid_visualization", 0),
            "annotation_missing_rate": annotation_missing_rate,
            "corrected_fsv_status": corrected.get("status", "HUMAN_LABELS_REQUIRED"),
            "corrected_fsv_value": corrected.get("sample_corrected_false_rate_among_decidable", ""),
            "corrected_fsv_ci": "see corrected_fsv_bootstrap_ci.json if available",
        },
        "true_sv_positive_control": {
            **true_control,
            "clear_true_vehicle_count": clear_true,
            "ambiguous_count": true_control.get("ambiguous", 0),
            "suspicious_count": suspicious,
            "qc_verdict": qc_verdict,
        },
        "degenerate_large_sv_box": {
            **degenerate,
            "actual_large_sv_degenerate_prediction": degenerate.get("actual_large_sv_degenerate_prediction", 0),
            "padding_or_boundary_related": degenerate.get("padding_or_boundary_related", 0),
            "large_context_texture": degenerate.get("large_context_texture", 0),
            "object_conflict": degenerate.get("object_conflict", 0),
            "uncertain": degenerate.get("uncertain", 0),
            "interpretation": "PENDING" if degenerate["labeled"] == 0 else "manual spot-check completed",
        },
        "padding_artifact": {
            **padding,
            "valid_padding_artifact": padding.get("valid_padding_artifact", 0),
            "not_padding_artifact": padding.get("not_padding_artifact", 0),
            "uncertain": padding.get("uncertain", 0),
        },
        "strict_object_flip": {
            **strict,
            "good_qualitative_case": strict.get("good_qualitative_case", 0),
            "not_clear": strict.get("not_clear", 0),
            "wrong_or_ambiguous": strict.get("wrong_or_ambiguous", 0),
            "candidate_crop_ids_for_paper_figures": good_strict_ids,
        },
        "claim_update": {
            "allowed_claims": [
                "false-SV / unmatched-SV burden under annotated-GT matching",
                "support/class embedding causally modulates SV burden",
                "DeHub reduces SV prediction burden",
                "degenerate large SV boxes are a separate failure mode if confirmed",
            ],
            "forbidden_claims": [
                "all unmatched SV are hallucinations",
                "all no-SV risk-group images contain no real vehicles",
                "DeHub is safe",
                "DeHub improves AP",
                "context alone proves vehicle hallucination",
                "closed-set models have open-vocabulary embedding attractors",
            ],
            "claims_requiring_ap_gt_evaluator": [
                "AP50/mAP improvement",
                "DeHub true-SV preservation",
                "open-vocab AP-level performance",
            ],
        },
    }

    write_json(output_dir / "manual_audit_summary_tonight.json", summary)
    md = [
        "# Manual Audit Summary Tonight",
        "",
        "## valid_unmatched_sv",
        "",
        md_table([summary["valid_unmatched_sv"]]),
        "",
        "## true_sv_positive_control",
        "",
        md_table([summary["true_sv_positive_control"]]),
        "",
        "## degenerate_large_sv_box",
        "",
        md_table([summary["degenerate_large_sv_box"]]),
        "",
        "## padding_artifact",
        "",
        md_table([summary["padding_artifact"]]),
        "",
        "## strict_object_flip",
        "",
        md_table([{k: v for k, v in summary["strict_object_flip"].items() if k != "candidate_crop_ids_for_paper_figures"}]),
        "",
        "Candidate crop ids for paper figures:",
        "",
        ", ".join(good_strict_ids) if good_strict_ids else "_None yet._",
        "",
        "## Claim Update",
        "",
        "### Allowed Claims",
        "",
        "\n".join(f"- {x}" for x in summary["claim_update"]["allowed_claims"]),
        "",
        "### Forbidden Claims",
        "",
        "\n".join(f"- {x}" for x in summary["claim_update"]["forbidden_claims"]),
        "",
        "### Claims Requiring AP/GT Evaluator",
        "",
        "\n".join(f"- {x}" for x in summary["claim_update"]["claims_requiring_ap_gt_evaluator"]),
    ]
    (output_dir / "manual_audit_summary_tonight.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({
        "valid_unmatched_labeled": valid["labeled"],
        "corrected_fsv_status": corrected.get("status", "HUMAN_LABELS_REQUIRED"),
        "qc_verdict": qc_verdict,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
