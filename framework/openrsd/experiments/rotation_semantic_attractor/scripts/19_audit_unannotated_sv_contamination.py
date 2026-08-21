#!/usr/bin/env python3
"""Audit the risk that unmatched small-vehicle predictions are true unannotated vehicles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LABEL_COLUMNS = [
    "crop_id",
    "image_path",
    "tile_id",
    "angle",
    "model",
    "checkpoint",
    "pred_score",
    "pred_box",
    "best_gt_class",
    "best_gt_iou",
    "risk_group",
    "auto_label",
    "human_label",
    "human_confidence",
    "human_notes",
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


def md_table(rows: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    if headers is None:
        headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def detect_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    metrics_root = Path(args.metrics_root) if args.metrics_root else Path("")
    prediction_root = Path(args.prediction_root) if args.prediction_root else Path("")
    gt_root = Path(args.gt_root) if args.gt_root else Path("")
    image_root = Path(args.image_root) if args.image_root else Path("")

    closed_table = Path("/data/zcy/OpenRSD_artifacts/closedset_scientific_audit_20260531/metrics/paper_table_closedset_false_sv_benchmark.csv")
    open_rows = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle/metrics/open_vocab_benchmark_rows_merged.csv")
    context_rows = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_context_counterfactual_s3_12angle/metrics/context_counterfactual_rows.csv")
    dehub_rows = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_dehub_safety_s3_12angle/metrics/dehub_safety_rows_merged.csv")
    dehub_scorecard = Path(args.repo_root) / "resultmd/exp_rotation_semantic_attractor/evidence_closure_20260601/dehub_safety_scorecard.csv"
    ap_blockers = Path(args.repo_root) / "resultmd/exp_rotation_semantic_attractor/evidence_closure_20260601/open_vocab_ap_blockers.md"

    return {
        "closedset_false_sv_table": str(closed_table),
        "closedset_false_sv_table_exists": closed_table.exists(),
        "openvocab_rows": str(open_rows),
        "openvocab_rows_exists": open_rows.exists(),
        "context_rows": str(context_rows),
        "context_rows_exists": context_rows.exists(),
        "dehub_rows": str(dehub_rows),
        "dehub_rows_exists": dehub_rows.exists(),
        "dehub_scorecard": str(dehub_scorecard),
        "dehub_scorecard_exists": dehub_scorecard.exists(),
        "ap_blockers": str(ap_blockers),
        "ap_blockers_exists": ap_blockers.exists(),
        "prediction_root": str(prediction_root),
        "prediction_root_exists": prediction_root.exists(),
        "gt_root": str(gt_root),
        "gt_root_exists": gt_root.exists(),
        "image_root": str(image_root),
        "image_root_exists": image_root.exists(),
        "metrics_root": str(metrics_root),
        "metrics_root_exists": metrics_root.exists(),
    }


def build_corrected_placeholder(output_dir: Path, human_label_csv: Path | None) -> tuple[str, list[dict[str, Any]]]:
    if human_label_csv and human_label_csv.exists():
        allowed_names = {
            "human_sv_crop_audit_template_VERIFIED.csv",
            "human_sv_crop_audit_template_VERIFIED_FAST.csv",
            "human_sv_crop_audit_template_VERIFIED_EXPANDED.csv",
        }
        if human_label_csv.name not in allowed_names:
            rows = [{
                "status": "ERROR_OLD_UNVERIFIED_CROP_PACK",
                "human_label_csv": str(human_label_csv),
                "corrected_FSV": "BLOCKED_BY_INVALID_CROPS",
                "required_input": "human_sv_crop_audit_template_VERIFIED.csv, human_sv_crop_audit_template_VERIFIED_FAST.csv, or human_sv_crop_audit_template_VERIFIED_EXPANDED.csv",
                "required_filters": "audit_category == valid_unmatched_sv; valid_for_human_audit == true; coordinate_frame_consistent == true; exclude degenerate_large_sv_box, padding_artifact, invalid_visualization, coordinate_mismatch, hbb_fallback_without_label, crop_window_drawn_instead_of_prediction",
            }]
            return "ERROR_OLD_UNVERIFIED_CROP_PACK", rows
        labels = read_csv(human_label_csv)
        labeled_count = sum(1 for r in labels if r.get("human_label"))
        usable = [
            r for r in labels
            if r.get("audit_category") == "valid_unmatched_sv"
            and str(r.get("valid_for_human_audit", "")).lower() == "true"
            and str(r.get("coordinate_frame_consistent", "true")).lower() == "true"
            and r.get("auto_label") not in {
                "degenerate_large_sv_box",
                "padding_artifact",
                "invalid_visualization",
                "coordinate_mismatch",
                "hbb_fallback_without_label",
                "crop_window_drawn_instead_of_prediction",
            }
            if r.get("human_label") in {
                "true_vehicle_annot_missing",
                "true_vehicle_annot_present_but_missed_match",
                "non_vehicle_background",
                "non_vehicle_object_conflict",
            }
        ]
        if labeled_count == 0:
            usable_verified = sum(
                1 for r in labels
                if r.get("audit_category") == "valid_unmatched_sv"
                and str(r.get("valid_for_human_audit", "")).lower() == "true"
                and str(r.get("coordinate_frame_consistent", "true")).lower() == "true"
            )
            status = "WARNING_SMALL_SAMPLE" if human_label_csv.name.endswith("_FAST.csv") and usable_verified < 100 else "HUMAN_LABELS_REQUIRED"
            rows = [{
                "status": status,
                "human_label_csv": str(human_label_csv),
                "usable_verified_rows": usable_verified,
                "corrected_FSV": "HUMAN_LABELS_REQUIRED",
                "required_next_step": "fill human_label/human_confidence in the verified template before estimating corrected-FSV",
            }]
            return status, rows
        if len(usable) < 100:
            true_vehicle = sum(
                1 for r in usable
                if r.get("human_label") in {"true_vehicle_annot_missing", "true_vehicle_annot_present_but_missed_match"}
            )
            rate = true_vehicle / max(1, len(usable))
            rows = [{
                "status": "SANITY_ONLY_CORRECTED_FSV",
                "human_label_csv": str(human_label_csv),
                "usable_labels": len(usable),
                "annotation_missing_or_missed_match_rate": f"{rate:.6f}",
                "paper_use": "BLOCKED_SMALL_SAMPLE",
                "required_next_step": "label at least 100 verified valid_unmatched_sv rows for stronger estimate; >=200 for main estimate",
            }]
            return "SANITY_ONLY_CORRECTED_FSV", rows
        true_vehicle = sum(
            1 for r in usable
            if r.get("human_label") in {"true_vehicle_annot_missing", "true_vehicle_annot_present_but_missed_match"}
        )
        rate = true_vehicle / max(1, len(usable))
        status = "PAPER_READY_CORRECTED_FSV" if len(usable) >= 200 else "SANITY_ONLY_CORRECTED_FSV"
        rows = [{
            "status": status,
            "human_label_csv": str(human_label_csv),
            "usable_labels": len(usable),
            "annotation_missing_or_missed_match_rate": f"{rate:.6f}",
            "note": "Bootstrap CI and stratum-level scaling should be added before final paper use; >=300 labels is preferred.",
        }]
        return status, rows
    rows = [{
        "status": "HUMAN_LABELS_REQUIRED",
        "human_label_csv": str(human_label_csv) if human_label_csv else "",
        "corrected_FSV": "BLOCKED",
        "corrected_BG_FSV": "BLOCKED",
        "corrected_ObjectFlip_SV": "BLOCKED",
        "corrected_DeHub_delta": "BLOCKED",
        "required_next_step": "fill human_sv_crop_audit_template.csv or provide human_sv_crop_audit_labels.csv",
    }]
    return "HUMAN_LABELS_REQUIRED", rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--prediction-root", default="")
    parser.add_argument("--gt-root", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--metrics-root", default="")
    parser.add_argument("--human-label-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    artifact_status = detect_artifacts(args)
    human_label_csv = Path(args.human_label_csv) if args.human_label_csv else audit_dir / "human_sv_crop_audit_labels.csv"
    corrected_status, corrected_rows = build_corrected_placeholder(output_dir, human_label_csv)

    template_path = audit_dir / "human_sv_crop_audit_template.csv"
    if not template_path.exists():
        write_csv(template_path, [], LABEL_COLUMNS)

    corrected_path = audit_dir / "corrected_fsv_if_human_labels_available.csv"
    write_csv(corrected_path, corrected_rows)
    write_md(
        audit_dir / "corrected_fsv_summary.md",
        "# Corrected-FSV Summary\n\n"
        f"- status: `{corrected_status}`\n"
        "- corrected-FSV is not reported as complete without human labels and stratum-level estimation.\n"
    )

    sample_manifest_path = audit_dir / "unannotated_sv_sample_manifest.csv"
    if not sample_manifest_path.exists():
        write_csv(sample_manifest_path, [], LABEL_COLUMNS + ["sample_status", "blocker"])

    already_controlled = [
        {"control": "FR_SV / FSV / BG_FSV / ObjectFlip_SV", "status": "PARTIAL_CONTROL", "meaning": "separates unmatched burden from strict object-conflict and valid-mask variants"},
        {"control": "Abs_FalseSV and SV_Pred_per_img", "status": "PARTIAL_CONTROL", "meaning": "avoids ranking only by FSV denominator"},
        {"control": "true_sv_rich positive control", "status": "PARTIAL_CONTROL", "meaning": "shows model behavior when annotated SV exists"},
        {"control": "strict object flip", "status": "PARTIAL_CONTROL", "meaning": "requires overlap with non-SV/LV GT, reducing but not eliminating missing-label risk"},
        {"control": "context NOT_APPLICABLE split", "status": "PARTIAL_CONTROL", "meaning": "does not force object-level counterfactual when usable SV polygon is absent"},
    ]
    unresolved = [
        {"risk": "no annotated SV does not mean no true SV", "status": "NOT_EXCLUDED"},
        {"risk": "unmatched SV can include annotation-missing real vehicles", "status": "NOT_EXCLUDED"},
        {"risk": "BG_FSV can overestimate hallucination under incomplete GT", "status": "NOT_EXCLUDED"},
        {"risk": "DeHub true-SV preservation needs raw predictions plus GT/human matching", "status": "BLOCKED"},
        {"risk": "row-level burden cannot replace human crop audit", "status": "NOT_EXCLUDED"},
    ]

    audit = {
        "status": "NOT_FULLY_EXCLUDED",
        "short_answer": "Unannotated true small-vehicle contamination has not been fully excluded; it is partially controlled by multiple diagnostics and still requires human crop audit or a label-refined subset.",
        "artifact_status": artifact_status,
        "already_controlled_by": already_controlled,
        "still_not_excluded": unresolved,
        "human_audit_template": str(template_path),
        "sample_manifest": str(sample_manifest_path),
        "corrected_fsv_status": corrected_status,
        "corrected_fsv_path": str(corrected_path),
        "allowed_wording": "unmatched-SV burden under annotated-GT matching, with annotation-missing true-SV risk not fully excluded",
        "forbidden_wording": "all unmatched SV predictions are hallucinations",
    }
    write_json(audit_dir / "unannotated_sv_contamination_audit.json", audit)

    md = [
        "# Unannotated True-SV Contamination Audit",
        "",
        "## Short Answer",
        "",
        "**No. The possibility that some unmatched small-vehicle predictions are true but unannotated vehicles has not been fully excluded.**",
        "",
        "Current experiments partially control the risk, but they do not eliminate it. The safe term is `unmatched-SV burden` or `false-SV burden under annotated-GT matching`, not universal hallucination.",
        "",
        "## Already Partially Controlled By",
        "",
        md_table(already_controlled),
        "",
        "## Still Not Excluded",
        "",
        md_table(unresolved),
        "",
        "## Required Next Audit",
        "",
        "- Use the generated crop audit template to label high-confidence unmatched SV crops.",
        "- Estimate annotation-missing rate and corrected-FSV with bootstrap CI after human labels exist.",
        "- Report corrected-FSV as sample-estimated, not as full GT truth.",
        "",
        "## Generated Files",
        "",
        f"- Human audit template: `{template_path}`",
        f"- Sample manifest: `{sample_manifest_path}`",
        f"- Corrected-FSV placeholder: `{corrected_path}`",
        "",
        f"corrected-FSV status: `{corrected_status}`",
    ]
    write_md(audit_dir / "unannotated_sv_contamination_audit.md", "\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
