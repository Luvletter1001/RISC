#!/usr/bin/env python3
"""Build consolidated FOCUS P1A realbatch unblock report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from focus_p1a_realbatch_common import (
    bool_from,
    ensure_exp_tree,
    html_from_markdown,
    md_table,
    read_json,
    resolve,
    write_json,
)


DEFAULT_EXP = Path("resultmd/exp_focus_ovd_p1a_realbatch_unblock_20260609")
NEXT_TRAINING_COMMAND = (
    "rtk bash tools/my_dist_train.sh "
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py 2 "
    "--cfg-options "
    "model.bbox_head.focus_losses.enable=True "
    "model.bbox_head.focus_losses.target_mapping_mode=spatial_region "
    "model.bbox_head.focus_losses.spatial_target_csv="
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609/"
    "targets/focus_spatial_region_targets.csv")


def finite_positive(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return value == value and value != float("inf") and value > 0.0


def build_payload(exp_dir: Path) -> dict[str, Any]:
    preflight = read_json(exp_dir / "preflight/p1a_realbatch_preflight.json", {})
    dataset = read_json(exp_dir / "configs/p1a_realbatch_smoke_dataset.json", {})
    smoke = read_json(exp_dir / "one_batch_real/realbatch_smoke_report.json", {})
    overlays = read_json(exp_dir / "debug_overlays/metadata.json", {})
    preflight_after = read_json(exp_dir / "preflight_after_realbatch/p1_preflight.json", {})

    actual_detector_train = bool_from(smoke.get("actual_detector_train"))
    anti_targets = int(smoke.get("num_focus_anti_targets", 0) or 0)
    preserve_targets = int(smoke.get("num_focus_preserve_targets", 0) or 0)
    anti_points = int(smoke.get("num_focus_anti_points", 0) or 0)
    preserve_points = int(smoke.get("num_focus_preserve_points", 0) or 0)
    loss_focus_anti = float(smoke.get("loss_focus_anti", 0.0) or 0.0)
    loss_focus_preserve = float(smoke.get("loss_focus_preserve", 0.0) or 0.0)
    loss_focus_total = float(smoke.get("loss_focus_total", 0.0) or 0.0)
    adapter_grad_flow = bool_from(smoke.get("adapter_grad_flow"))
    frozen_grad_present = bool_from(smoke.get("frozen_grad_present"))
    checksum_changed = bool_from(smoke.get("adapter_checksum_changed_after_step"))
    detector_batch_ready = (
        str(smoke.get("status", "")).startswith("PASS_")
        and actual_detector_train
        and smoke.get("real_batch_source") == "dataloader_real_batch"
        and anti_targets > 0
        and preserve_targets > 0
        and anti_points > 0
        and preserve_points > 0
        and finite_positive(loss_focus_anti)
        and finite_positive(loss_focus_preserve)
        and loss_focus_total == loss_focus_total
        and adapter_grad_flow
        and not frozen_grad_present
        and checksum_changed
        and bool_from(smoke.get("declip_support_disabled"))
        and bool_from(smoke.get("native_support_preserved")))
    can_continue = bool(
        detector_batch_ready
        or bool_from(preflight_after.get("can_continue_to_detector_training")))
    evidence_level = (
        "DETECTOR_LEVEL_SPATIAL_PSEUDO_REGION_REAL_BATCH"
        if detector_batch_ready else "INSUFFICIENT_P1A_REAL_BATCH_EVIDENCE")
    return {
        "status": (
            "PASS_P1A_REALBATCH_UNBLOCK"
            if detector_batch_ready else "BLOCKED_P1A_REALBATCH_UNBLOCK"),
        "preflight_status": preflight.get("status", ""),
        "dataset_status": dataset.get("status", ""),
        "one_batch_status": smoke.get("status", ""),
        "overlay_status": overlays.get("status", ""),
        "actual_detector_train": actual_detector_train,
        "real_batch_source": smoke.get("real_batch_source", ""),
        "num_focus_images_with_targets": int(smoke.get("num_focus_images_with_targets", 0) or 0),
        "num_focus_anti_targets": anti_targets,
        "num_focus_preserve_targets": preserve_targets,
        "num_focus_anti_points": anti_points,
        "num_focus_preserve_points": preserve_points,
        "focus_assignment_coverage": float(smoke.get("focus_assignment_coverage", 0.0) or 0.0),
        "loss_focus_support_distill": float(smoke.get("loss_focus_support_distill", 0.0) or 0.0),
        "loss_focus_anti": loss_focus_anti,
        "loss_focus_preserve": loss_focus_preserve,
        "loss_focus_migration": float(smoke.get("loss_focus_migration", 0.0) or 0.0),
        "loss_focus_total": loss_focus_total,
        "adapter_grad_flow": adapter_grad_flow,
        "frozen_grad_present": frozen_grad_present,
        "adapter_checksum_changed_after_step": checksum_changed,
        "declip_support_disabled": bool_from(smoke.get("declip_support_disabled")),
        "native_support_preserved": bool_from(smoke.get("native_support_preserved")),
        "support_delta_norm_ratio": float(smoke.get("support_delta_norm_ratio", 0.0) or 0.0),
        "detector_batch_target_mask_injection_ready": detector_batch_ready,
        "can_continue_to_detector_training": can_continue,
        "evidence_level": evidence_level,
        "debug_overlay_path": str(exp_dir / "debug_overlays"),
        "next_detector_training_command": (
            NEXT_TRAINING_COMMAND if detector_batch_ready
            else "NOT_READY: realbatch smoke did not satisfy all P1A gates"),
        "answers": {
            "real_detector_batch_matched_spatial_targets": bool(
                preflight.get("batch_match_status") == "PASS"
                or int(smoke.get("num_focus_images_with_targets", 0) or 0) > 0),
            "anti_preserve_targets_entered_batch": anti_targets > 0 and preserve_targets > 0,
            "anti_preserve_feature_points_nonzero": anti_points > 0 and preserve_points > 0,
            "loss_focus_anti_nonzero": finite_positive(loss_focus_anti),
            "loss_focus_preserve_nonzero": finite_positive(loss_focus_preserve),
            "gradient_reached_adapter": adapter_grad_flow,
            "frozen_base_has_no_grad": not frozen_grad_present,
            "can_continue_p1_detector_training": can_continue,
            "current_evidence_level": evidence_level,
            "next_training_command": (
                NEXT_TRAINING_COMMAND if detector_batch_ready
                else "NOT_READY"),
        },
        "artifacts": {
            "preflight_json": str(exp_dir / "preflight/p1a_realbatch_preflight.json"),
            "smoke_dataset_json": str(exp_dir / "configs/p1a_realbatch_smoke_dataset.json"),
            "realbatch_smoke_report_json": str(exp_dir / "one_batch_real/realbatch_smoke_report.json"),
            "overlay_metadata_json": str(exp_dir / "debug_overlays/metadata.json"),
            "preflight_after_realbatch_json": str(exp_dir / "preflight_after_realbatch/p1_preflight.json"),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    decision_rows = [
        {"item": "actual_detector_train", "value": payload["actual_detector_train"]},
        {"item": "real_batch_source", "value": payload["real_batch_source"]},
        {"item": "num_focus_anti_targets", "value": payload["num_focus_anti_targets"]},
        {"item": "num_focus_preserve_targets", "value": payload["num_focus_preserve_targets"]},
        {"item": "num_focus_anti_points", "value": payload["num_focus_anti_points"]},
        {"item": "num_focus_preserve_points", "value": payload["num_focus_preserve_points"]},
        {"item": "loss_focus_anti", "value": payload["loss_focus_anti"]},
        {"item": "loss_focus_preserve", "value": payload["loss_focus_preserve"]},
        {"item": "loss_focus_total", "value": payload["loss_focus_total"]},
        {"item": "adapter_grad_flow", "value": payload["adapter_grad_flow"]},
        {"item": "frozen_grad_present", "value": payload["frozen_grad_present"]},
        {"item": "adapter_checksum_changed_after_step", "value": payload["adapter_checksum_changed_after_step"]},
        {"item": "detector_batch_target_mask_injection_ready", "value": payload["detector_batch_target_mask_injection_ready"]},
        {"item": "can_continue_to_detector_training", "value": payload["can_continue_to_detector_training"]},
        {"item": "evidence_level", "value": payload["evidence_level"]},
        {"item": "debug_overlay_path", "value": payload["debug_overlay_path"]},
        {"item": "next_detector_training_command", "value": payload["next_detector_training_command"]},
    ]
    answer_rows = [
        {"question": key, "answer": value}
        for key, value in payload["answers"].items()
    ]
    lines = [
        "# FOCUS P1A Realbatch Unblock Report",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        "",
        "## Required Final Fields",
        "",
    ]
    lines.extend(md_table(decision_rows, ["item", "value"]))
    lines.extend(["", "## Required Questions", ""])
    lines.extend(md_table(answer_rows, ["question", "answer"]))
    lines.extend(["", "## Artifacts", ""])
    lines.extend(md_table([
        {"artifact": key, "path": value}
        for key, value in payload["artifacts"].items()
    ], ["artifact", "path"]))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    ensure_exp_tree(exp_dir)
    payload = build_payload(exp_dir)
    write_json(exp_dir / "reports/focus_p1a_realbatch_unblock_report.json", payload)
    md_path = exp_dir / "reports/focus_p1a_realbatch_unblock_report.md"
    html_path = exp_dir / "reports/focus_p1a_realbatch_unblock_report.html"
    write_markdown(md_path, payload)
    html_from_markdown(md_path, html_path)
    print(json.dumps({
        "status": payload["status"],
        "detector_batch_target_mask_injection_ready": payload["detector_batch_target_mask_injection_ready"],
        "can_continue_to_detector_training": payload["can_continue_to_detector_training"],
        "report": str(md_path),
    }, indent=2))
    return 0 if payload["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
