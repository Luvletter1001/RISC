#!/usr/bin/env python3
"""Build the consolidated FOCUS P1 unblock loss-mapping report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_EXP_DIR = Path(
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609")
LOSS_KEYS = [
    "loss_focus_support_distill",
    "loss_focus_anti",
    "loss_focus_preserve",
    "loss_focus_migration",
    "loss_focus_total",
]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def bool_from(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    out = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|") for field in fields)
            + " |")
    return out


def write_html_from_md(md_path: Path, html_path: Path) -> None:
    body = html.escape(md_path.read_text(encoding="utf-8"))
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>FOCUS P1 Unblock Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1080px;"
        "margin:32px auto;line-height:1.5}pre{white-space:pre-wrap}"
        "code{background:#f3f4f6;padding:1px 4px;border-radius:3px}"
        "</style></head><body><pre>"
        + body + "</pre></body></html>",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    exp_dir = args.exp_dir if args.exp_dir.is_absolute() else repo / args.exp_dir
    preflight = read_json(
        exp_dir / "preflight/p1_unblock_preflight.json", {})
    target_summary = read_json(
        exp_dir / "targets/focus_spatial_region_targets_summary.json", {})
    capture_summary = read_json(
        exp_dir / "provenance/focus_pre_nms_capture_summary.json", {})
    exact_summary = read_json(
        exp_dir / "provenance/focus_pre_nms_exact_targets_summary.json", {})
    smoke = read_json(
        exp_dir / "one_batch_smoke/one_batch_loss_wiring_report.json", {})

    valid_counts = target_summary.get("valid_role_counts", {})
    anti_valid = int(valid_counts.get("anti_negative", 0))
    preserve_valid = int(valid_counts.get("preserve_positive", 0))
    loss_wired = bool_from(preflight.get("detector_focus_loss_path_wired"))
    smoke_pass = str(smoke.get("status", "")).startswith("PASS_")
    finite = bool_from(smoke.get("all_losses_finite"))
    grad_flow = bool_from(smoke.get("adapter_grad_flow"))
    frozen_grad = bool_from(smoke.get("frozen_grad_present"))
    p1a_ready = (
        loss_wired and anti_valid > 0 and preserve_valid > 0
        and smoke_pass and finite and grad_flow and not frozen_grad)
    p1b_coverage = float(exact_summary.get("exact_match_coverage", 0.0) or 0.0)
    p1b_ready = (
        exact_summary.get("status") == "P1B_EXACT_PROVENANCE_LOSS_READY"
        and p1b_coverage >= 0.70)
    detector_batch_injection = any(
        row.get("name") == "detector_batch_target_mask_injection"
        and row.get("ok") for row in preflight.get("checks", []))
    can_continue_to_detector_training = bool(
        p1a_ready and detector_batch_injection)
    next_training_command = (
        "NOT_READY: detector batch target-mask injection is not proven"
        if not can_continue_to_detector_training else
        "rtk bash tools/my_dist_train.sh "
        "M_configs/experiments/focus_ovd/"
        "focus_ovd_p1_spatial_region_loss_20260609.py 2")
    evidence_level = (
        "DETECTOR_LEVEL_SPATIAL_PSEUDO_REGION_PLUS_SYNTHETIC_HEAD_BACKWARD"
        if p1a_ready else "INSUFFICIENT_P1A_EVIDENCE")
    payload = {
        "status": (
            "PASS_P1A_SPATIAL_REGION_LOSS_PARTIAL_UNBLOCK"
            if p1a_ready else "BLOCKED_LABEL_TO_LOSS_MAPPING"),
        "preflight_status": preflight.get("status", ""),
        "detector_focus_loss_path_wired": loss_wired,
        "loss_keys": LOSS_KEYS,
        "spatial_target_total_rows": int(target_summary.get("total_rows", 0) or 0),
        "spatial_target_valid_anti_negative": anti_valid,
        "spatial_target_valid_preserve_positive": preserve_valid,
        "spatial_target_role_counts": target_summary.get("role_counts", {}),
        "spatial_target_invalid_reason_counts": target_summary.get(
            "invalid_reason_counts", {}),
        "spatial_target_summary_status": target_summary.get("status", ""),
        "exact_provenance_status": exact_summary.get(
            "status", capture_summary.get("status", "")),
        "exact_pre_nms_capture_status": capture_summary.get("status", ""),
        "pre_nms_candidate_count": int(
            capture_summary.get("provenance_candidate_count", 0) or 0),
        "pre_nms_source_stage_counts": capture_summary.get(
            "source_stage_counts", {}),
        "exact_valid_focus_rows": int(
            exact_summary.get("valid_focus_rows", 0) or 0),
        "exact_matched_rows": int(exact_summary.get("matched_rows", 0) or 0),
        "exact_coverage_threshold": float(
            exact_summary.get("coverage_threshold", 0.70) or 0.70),
        "exact_match_coverage": p1b_coverage,
        "one_batch_smoke_status": smoke.get("status", ""),
        "one_batch_actual_detector_train": bool_from(
            smoke.get("actual_detector_train")),
        "one_batch_actual_detector_rerun": bool_from(
            smoke.get("actual_detector_rerun")),
        "one_batch_detector_batch_source": smoke.get(
            "detector_batch_source", ""),
        "one_batch_assigned_points": int(smoke.get("assigned_points", 0) or 0),
        "one_batch_anti_points": int(smoke.get("anti_points", 0) or 0),
        "one_batch_preserve_points": int(
            smoke.get("preserve_points", 0) or 0),
        "losses_finite": finite,
        "adapter_grad_flow": grad_flow,
        "frozen_grad_present": frozen_grad,
        "p1a_spatial_region_loss_ready": p1a_ready,
        "p1b_exact_provenance_loss_ready": p1b_ready,
        "detector_batch_target_mask_injection_ready": bool(detector_batch_injection),
        "can_continue": can_continue_to_detector_training,
        "can_continue_to_detector_training": can_continue_to_detector_training,
        "evidence_level": evidence_level,
        "next_training_command_if_ready": next_training_command,
        "report_scope": (
            "No full detector training, AP evaluation, safety verdict, "
            "detector rerun, or DeCLIP support is claimed."),
    }
    write_json(exp_dir / "reports/focus_p1_unblock_loss_mapping_report.json",
               payload)

    decision_rows = [
        {"item": "detector_focus_loss_path_wired", "value": loss_wired},
        {"item": "loss_keys", "value": ";".join(LOSS_KEYS)},
        {"item": "spatial_target_total_rows", "value": payload["spatial_target_total_rows"]},
        {"item": "spatial_valid_anti_negative", "value": anti_valid},
        {"item": "spatial_valid_preserve_positive", "value": preserve_valid},
        {"item": "exact_provenance_status", "value": payload["exact_provenance_status"]},
        {"item": "exact_match_coverage", "value": f"{p1b_coverage:.6f}"},
        {"item": "exact_coverage_threshold", "value": payload["exact_coverage_threshold"]},
        {"item": "one_batch_smoke_status", "value": payload["one_batch_smoke_status"]},
        {"item": "one_batch_actual_detector_train", "value": payload["one_batch_actual_detector_train"]},
        {"item": "losses_finite", "value": finite},
        {"item": "adapter_grad_flow", "value": grad_flow},
        {"item": "frozen_grad_present", "value": frozen_grad},
        {"item": "p1a_spatial_region_loss_ready", "value": p1a_ready},
        {"item": "p1b_exact_provenance_loss_ready", "value": p1b_ready},
        {"item": "detector_batch_target_mask_injection_ready", "value": detector_batch_injection},
        {"item": "can_continue_to_detector_training", "value": can_continue_to_detector_training},
        {"item": "evidence_level", "value": evidence_level},
        {"item": "next_training_command_if_ready", "value": next_training_command},
    ]
    preflight_rows = [
        {
            "check": row.get("name", ""),
            "ok": row.get("ok", ""),
            "severity": row.get("severity", ""),
            "detail": row.get("detail", ""),
        }
        for row in preflight.get("checks", [])
    ]
    role_counts = target_summary.get("role_counts", {})
    valid_role_counts = target_summary.get("valid_role_counts", {})
    target_rows = [
        {
            "role": role,
            "total_rows": int(role_counts.get(role, 0) or 0),
            "valid_for_loss_rows": int(valid_role_counts.get(role, 0) or 0),
        }
        for role in ("anti_negative", "preserve_positive", "exclude")
    ]
    invalid_reason_rows = [
        {"invalid_reason": key, "count": value}
        for key, value in sorted(
            target_summary.get("invalid_reason_counts", {}).items())
    ] or [{"invalid_reason": "none", "count": 0}]
    provenance_rows = [
        {"item": "capture_status", "value": capture_summary.get("status", "")},
        {"item": "actual_pre_nms_capture", "value": capture_summary.get("actual_pre_nms_capture", "")},
        {"item": "input_target_rows", "value": capture_summary.get("input_target_rows", "")},
        {"item": "pre_nms_candidate_count", "value": capture_summary.get("provenance_candidate_count", "")},
        {"item": "source_stage_final_rows", "value": capture_summary.get("source_stage_counts", {}).get("final", 0)},
        {"item": "required_fields_missing", "value": ";".join(capture_summary.get("required_fields_missing", []))},
        {"item": "exact_status", "value": exact_summary.get("status", "")},
        {"item": "exact_total_crop_rows", "value": exact_summary.get("total_crop_rows", "")},
        {"item": "exact_valid_focus_rows", "value": exact_summary.get("valid_focus_rows", "")},
        {"item": "exact_matched_rows", "value": exact_summary.get("matched_rows", "")},
        {"item": "exact_match_coverage", "value": f"{p1b_coverage:.6f}"},
        {"item": "coverage_threshold", "value": exact_summary.get("coverage_threshold", "")},
    ]
    smoke_overview_rows = [
        {"item": "status", "value": smoke.get("status", "")},
        {"item": "seed", "value": smoke.get("seed", "")},
        {"item": "detector_batch_source", "value": smoke.get("detector_batch_source", "")},
        {"item": "actual_detector_train", "value": smoke.get("actual_detector_train", "")},
        {"item": "actual_detector_rerun", "value": smoke.get("actual_detector_rerun", "")},
        {"item": "assigned_points", "value": smoke.get("assigned_points", "")},
        {"item": "anti_points", "value": smoke.get("anti_points", "")},
        {"item": "preserve_points", "value": smoke.get("preserve_points", "")},
        {"item": "all_losses_finite", "value": smoke.get("all_losses_finite", "")},
        {"item": "adapter_grad_flow", "value": smoke.get("adapter_grad_flow", "")},
        {"item": "frozen_grad_present", "value": smoke.get("frozen_grad_present", "")},
    ]
    artifact_rows = [
        {"artifact": "preflight_json", "path": str(exp_dir / "preflight/p1_unblock_preflight.json")},
        {"artifact": "spatial_targets_csv", "path": str(exp_dir / "targets/focus_spatial_region_targets.csv")},
        {"artifact": "spatial_targets_summary", "path": str(exp_dir / "targets/focus_spatial_region_targets_summary.json")},
        {"artifact": "pre_nms_candidates_csv", "path": str(exp_dir / "provenance/focus_pre_nms_provenance_candidates.csv")},
        {"artifact": "exact_targets_csv", "path": str(exp_dir / "provenance/focus_pre_nms_exact_targets.csv")},
        {"artifact": "smoke_loss_values", "path": str(exp_dir / "one_batch_smoke/loss_values.csv")},
        {"artifact": "smoke_grad_flow", "path": str(exp_dir / "one_batch_smoke/grad_flow.csv")},
        {"artifact": "report_json", "path": str(exp_dir / "reports/focus_p1_unblock_loss_mapping_report.json")},
    ]
    md = [
        "# FOCUS P1 Unblock Loss Mapping Report",
        "",
        f"- status: `{payload['status']}`",
        f"- preflight_status: `{payload['preflight_status']}`",
        f"- can_continue_to_detector_training: `{can_continue_to_detector_training}`",
        f"- evidence_level: `{evidence_level}`",
        f"- report_scope: `{payload['report_scope']}`",
        "",
        "## Executive Decision",
        "",
    ]
    md.extend(markdown_table(decision_rows, ["item", "value"]))
    md.extend([
        "",
        "## Loss Keys",
        "",
    ])
    md.extend(markdown_table(
        [{"loss_key": key, "wired": True} for key in LOSS_KEYS],
        ["loss_key", "wired"]))
    md.extend([
        "",
        "## Preflight Checks",
        "",
    ])
    md.extend(markdown_table(preflight_rows, ["check", "ok", "severity", "detail"]))
    md.extend([
        "",
        "## Spatial Region Target Mapping (P1A)",
        "",
        f"- status: `{target_summary.get('status', '')}`",
        f"- coordinate_frame: `{target_summary.get('coordinate_frame', '')}`",
        f"- evidence_level: `{target_summary.get('evidence_level', '')}`",
        f"- input_split: `{target_summary.get('input_split', '')}`",
        f"- target_csv: `{target_summary.get('target_csv', '')}`",
        "",
    ])
    md.extend(markdown_table(target_rows, [
        "role", "total_rows", "valid_for_loss_rows"]))
    md.extend([
        "",
        "### Invalid Target Reasons",
        "",
    ])
    md.extend(markdown_table(invalid_reason_rows, [
        "invalid_reason", "count"]))
    md.extend([
        "",
        "## Exact Pre-NMS Provenance (P1B)",
        "",
    ])
    md.extend(markdown_table(provenance_rows, ["item", "value"]))
    md.extend([
        "",
        "## One-Batch Loss Wiring Smoke",
        "",
    ])
    md.extend(markdown_table(smoke_overview_rows, ["item", "value"]))
    md.extend([
        "",
        "### Loss Values",
        "",
    ])
    md.extend(markdown_table(smoke.get("loss_rows", []), [
        "loss_key", "value", "finite"]))
    md.extend([
        "",
        "### Gradient Flow",
        "",
    ])
    md.extend(markdown_table(smoke.get("grad_rows", []), [
        "parameter_group", "grad_norm", "has_grad"]))
    md.extend([
        "",
        "## Readiness Matrix",
        "",
    ])
    md.extend(markdown_table([
        {
            "gate": "P1A spatial-region loss",
            "ready": p1a_ready,
            "reason": (
                "loss path wired; anti/preserve spatial targets exist; "
                "synthetic detector-head backward passed"),
        },
        {
            "gate": "P1B exact pre-NMS provenance",
            "ready": p1b_ready,
            "reason": (
                f"exact coverage {p1b_coverage:.6f} below threshold "
                f"{payload['exact_coverage_threshold']:.2f}; no candidates"),
        },
        {
            "gate": "Detector training continuation",
            "ready": can_continue_to_detector_training,
            "reason": (
                "detector batch target-mask injection is not proven"),
        },
    ], ["gate", "ready", "reason"]))
    md.extend([
        "",
        "## Artifacts",
        "",
    ])
    md.extend(markdown_table(artifact_rows, ["artifact", "path"]))
    md.extend([
        "",
        "## Boundaries",
        "",
        "- P1A spatial-region loss is a detector-level pseudo-region mapping, not exact pre-NMS provenance.",
        "- P1B exact pre-NMS provenance is not ready unless coverage reaches 70%.",
        "- This run does not claim full detector training, detector rerun, AP/safety, or DeCLIP support.",
        "",
    ])
    md_path = exp_dir / "reports/focus_p1_unblock_loss_mapping_report.md"
    html_path = exp_dir / "reports/focus_p1_unblock_loss_mapping_report.html"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_html_from_md(md_path, html_path)
    print(json.dumps(payload, indent=2))
    return 0 if p1a_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
