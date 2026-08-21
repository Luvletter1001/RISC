#!/usr/bin/env python3
"""FOCUS-OVD P1 detector-level validation preflight.

This script is intentionally conservative.  If P0 crop labels cannot be
converted into detector training loss masks, it blocks instead of producing
fake detector-level evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXP_DIR = Path("resultmd/exp_focus_ovd_p1_detector_validation_20260609")
DEFAULT_UNBLOCK_EXP = Path(
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609")
DEFAULT_BASELINE_CKPT = Path(
    "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth")
DEFAULT_SUPPORT_PKL = Path(
    "data/DOTA2_1024_500/ss_train/"
    "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl")
DEFAULT_CORRECTED_FSV_CSV = Path(
    "experiments/rotation_semantic_attractor/reports/visual_summary/audit/"
    "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv")
DEFAULT_FOCUS_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
DEFAULT_DENSE_HEAD = Path("M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py")
DEFAULT_DETECTOR = Path("M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py")
DEFAULT_LOSS_HELPER = Path("M_AD/models/losses/focus_attractor_losses.py")

P1_SUBDIRS = [
    "preflight", "configs", "loss_wiring", "train", "eval", "ap_eval",
    "safety", "figures", "tables", "reports", "manifests",
]

P1_VARIANTS = [
    "P1_V00_baseline_detector_eval",
    "P1_V01_focus_zero_detector_eval",
    "P1_V02_random_orientation_detector",
    "P1_V03_direction_text_prompt_only_detector_negative_control",
    "P1_V23_detector_anti_plus_preserve",
    "P1_V24_detector_full_focus_core",
    "P1_V24_no_orientation_detector",
    "P1_V24_random_orientation_detector",
    "P1_V24_no_support_distill_detector",
    "P1_V24_no_preserve_detector",
    "P1_DeHub_reference_if_available",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_check(name: str, ok: bool, detail: str,
               severity: str = "blocking") -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "detail": str(detail),
        "severity": severity,
    }


def ensure_p1_tree(exp_dir: Path) -> None:
    for subdir in P1_SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def safe_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def inspect_detector_loss_wiring(dense_head_path: Path) -> dict[str, Any]:
    if not dense_head_path.exists():
        return make_check("detector_focus_loss_path_wired", False,
                          f"missing dense head: {dense_head_path}")
    text = dense_head_path.read_text(encoding="utf-8")
    has_config = "focus_losses" in text
    has_loss_names = all(name in text for name in (
        "loss_focus_support_distill",
        "loss_focus_anti",
        "loss_focus_preserve",
        "loss_focus_total",
    ))
    has_helper_call = "anti_attractor_loss(" in text and "preserve_loss(" in text
    ok = has_config and has_loss_names and has_helper_call
    detail = (
        "focus loss branch present"
        if ok else
        "missing bbox_head.focus_losses branch and detector loss calls")
    return make_check("detector_focus_loss_path_wired", ok, detail)


def inspect_training_logits_availability(dense_head_path: Path) -> dict[str, Any]:
    if not dense_head_path.exists():
        return make_check("training_logits_and_support_available", False,
                          f"missing dense head: {dense_head_path}")
    text = dense_head_path.read_text(encoding="utf-8")
    ok = all(token in text for token in (
        "def loss_by_feat",
        "cls_scores",
        "support_feats",
        "support_labels",
    ))
    detail = (
        "loss_by_feat receives cls_scores/support_feats/support_labels"
        if ok else
        "loss_by_feat inputs do not expose both dense logits and support")
    return make_check("training_logits_and_support_available", ok, detail)


def inspect_focus_adapter_exists(dense_head_path: Path) -> dict[str, Any]:
    if not dense_head_path.exists():
        return make_check("focus_adapter_exists_in_model", False,
                          f"missing dense head: {dense_head_path}")
    text = dense_head_path.read_text(encoding="utf-8")
    ok = "focus_support_adapter" in text and "FourierSupportResidualAdapter" in text
    return make_check(
        "focus_adapter_exists_in_model", ok,
        "focus adapter path present" if ok else "focus adapter path missing")


def inspect_label_to_loss_mapping(
        train_split_path: Path,
        mapper_paths: Iterable[Path] | None = None) -> dict[str, Any]:
    if not train_split_path.exists():
        return make_check("label_to_loss_mapping", False,
                          f"missing train split: {train_split_path}")
    payload = read_json(train_split_path)
    rows = payload.get("rows", [])
    if not rows:
        return make_check("label_to_loss_mapping", False, "train split has no rows")

    exact_fields = {
        "prediction_id", "train_sample_id", "feature_level", "grid_x", "grid_y",
    }
    exact = all(exact_fields.issubset(set(row.keys())) for row in rows)
    if exact:
        return make_check("label_to_loss_mapping", True,
                          "exact_training_loss_mask_fields_present")

    tile_angle = all({"crop_id", "tile_id", "angle"}.issubset(set(row.keys()))
                     for row in rows)
    mapper_exists = any(path.exists() for path in (mapper_paths or []))
    if tile_angle and mapper_exists:
        return make_check("label_to_loss_mapping", True,
                          "explicit_tile_angle_mapper_available")
    if tile_angle:
        return make_check("label_to_loss_mapping", False,
                          "tile_angle_proxy_only_no_training_loss_mask")
    return make_check("label_to_loss_mapping", False,
                      "missing_exact_or_tile_angle_mapping_fields")


def inspect_p0_candidates(p0_exp: Path) -> dict[str, Any]:
    safety = p0_exp / "safety/p0_safety_gate_summary.csv"
    if not safety.exists():
        return make_check("p0_v23_v24_effective", False,
                          f"missing P0 safety CSV: {safety}")
    rows = read_csv_rows(safety)
    verdicts = {row.get("variant_id"): row.get("verdict") for row in rows}
    ok = (
        verdicts.get("V23_anti_plus_preserve") == "EFFECTIVE_CANDIDATE"
        and verdicts.get("V24_full_focus_core") == "EFFECTIVE_CANDIDATE")
    return make_check("p0_v23_v24_effective", ok, json.dumps(verdicts))


def inspect_p0_proxy_only(p0_exp: Path) -> dict[str, Any]:
    eval_csv = p0_exp / "eval/p0_eval_all_variants.csv"
    if not eval_csv.exists():
        return make_check("p0_is_proxy_only", False,
                          f"missing P0 eval CSV: {eval_csv}")
    rows = read_csv_rows(eval_csv)
    detector_train_values = {row.get("train_status") for row in rows}
    detector_rerun_values = {row.get("actual_detector_rerun") for row in rows}
    ok = "false" in detector_rerun_values and "DONE_SMALL_PROXY_TRAIN" in detector_train_values
    return make_check(
        "p0_is_proxy_only", ok,
        "P0 eval rows carry actual_detector_rerun=false and proxy train status"
        if ok else f"unexpected P0 detector flags: {detector_rerun_values}")


def inspect_split_distinct(train_split: Path, eval_split: Path) -> dict[str, Any]:
    if not train_split.exists() or not eval_split.exists():
        return make_check("train_eval_split_distinct", False,
                          "train/eval split file missing")
    train_rows = read_json(train_split).get("rows", [])
    eval_rows = read_json(eval_split).get("rows", [])
    train_ids = {row.get("crop_id") for row in train_rows}
    eval_ids = {row.get("crop_id") for row in eval_rows}
    overlap = sorted(v for v in train_ids & eval_ids if v)
    return make_check("train_eval_split_distinct", len(overlap) == 0,
                      f"overlap_count={len(overlap)}")


def inspect_declip_disabled(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return make_check("declip_support_disabled", False,
                          f"missing config: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    bad = "use_declip_support=True" in text or "DeCLIP" in text
    return make_check("declip_support_disabled", not bad,
                      "DeCLIP support not enabled" if not bad else "DeCLIP reference found")


def inspect_audit_metric_availability(audit_csv: Path) -> list[dict[str, Any]]:
    if not audit_csv.exists():
        missing = f"missing audit CSV: {audit_csv}"
        return [
            make_check("true_sv_gt_matching_available", False, missing),
            make_check("degenerate_large_sv_metric_available", False, missing),
        ]
    rows = read_csv_rows(audit_csv)
    columns = set(rows[0].keys()) if rows else set()
    true_gt = {"best_gt_class", "best_gt_iou"}.issubset(columns)
    degenerate = any(row.get("audit_category") == "degenerate_large_sv_box"
                     for row in rows)
    return [
        make_check("true_sv_gt_matching_available", true_gt,
                   "best_gt_class/best_gt_iou columns present"),
        make_check("degenerate_large_sv_metric_available", degenerate,
                   "degenerate_large_sv_box rows present"),
    ]


def derive_p1_status(checks: list[dict[str, Any]],
                     actual_detector_train: bool,
                     actual_detector_rerun: bool) -> str:
    del actual_detector_train, actual_detector_rerun
    by_name = {check["name"]: check for check in checks}
    if not by_name.get("label_to_loss_mapping", {}).get("ok", False):
        return "BLOCKED_LABEL_TO_LOSS_MAPPING"
    if not by_name.get("detector_focus_loss_path_wired", {}).get("ok", False):
        return "BLOCKED_LABEL_TO_LOSS_MAPPING"
    if any(not check["ok"] and check.get("severity") == "blocking"
           for check in checks):
        return "FAIL_PREFLIGHT"
    return "READY_FOR_LOSS_WIRING_SMOKE"


def inspect_unblock_exp(unblock_exp: Path) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    report_json = unblock_exp / "reports/focus_p1_unblock_loss_mapping_report.json"
    if not report_json.exists():
        return [
            make_check("p1_unblock_report_exists", False,
                       f"missing {report_json}", severity="warning")
        ], None, {}
    payload = read_json(report_json)
    p1a_ready = bool(payload.get("p1a_spatial_region_loss_ready", False))
    p1b_ready = bool(payload.get("p1b_exact_provenance_loss_ready", False))
    checks = [
        make_check("p1_unblock_report_exists", True, str(report_json),
                   severity="warning"),
        make_check("p1a_spatial_region_loss_ready", p1a_ready,
                   payload.get("evidence_level", ""),
                   severity="warning"),
        make_check("p1b_exact_provenance_loss_ready", p1b_ready,
                   f"coverage={payload.get('exact_match_coverage', 0.0)}",
                   severity="warning"),
        make_check("detector_batch_target_mask_injection_ready",
                   bool(payload.get("detector_batch_target_mask_injection_ready", False)),
                   "required before claiming actual detector training",
                   severity="warning"),
    ]
    if p1b_ready:
        return checks, "PASS_P1B_EXACT_PROVENANCE_LOSS", payload
    if p1a_ready:
        return checks, "PASS_P1A_SPATIAL_REGION_LOSS", payload
    return checks, None, payload


def inspect_realbatch_exp(realbatch_exp: Path) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    report_json = realbatch_exp / "one_batch_real/realbatch_smoke_report.json"
    if not report_json.exists():
        return [
            make_check("p1a_realbatch_smoke_report_exists", False,
                       f"missing {report_json}", severity="warning"),
            make_check("detector_batch_target_mask_injection_ready", False,
                       "realbatch smoke report missing", severity="warning"),
        ], False, {}
    payload = read_json(report_json)
    actual_detector_train = bool(payload.get("actual_detector_train", False))
    anti_points = int(payload.get("num_focus_anti_points", 0) or 0)
    preserve_points = int(payload.get("num_focus_preserve_points", 0) or 0)
    loss_focus_anti = float(payload.get("loss_focus_anti", 0.0) or 0.0)
    loss_focus_preserve = float(payload.get("loss_focus_preserve", 0.0) or 0.0)
    adapter_grad_flow = bool(payload.get("adapter_grad_flow", False))
    frozen_grad_present = bool(payload.get("frozen_grad_present", False))
    detector_batch_ready = (
        str(payload.get("status", "")).startswith("PASS_")
        and actual_detector_train
        and anti_points > 0
        and preserve_points > 0
        and loss_focus_anti > 0.0
        and loss_focus_preserve > 0.0
        and adapter_grad_flow
        and not frozen_grad_present)
    checks = [
        make_check("p1a_realbatch_smoke_report_exists", True,
                   str(report_json), severity="warning"),
        make_check("p1a_realbatch_smoke_pass",
                   str(payload.get("status", "")).startswith("PASS_"),
                   payload.get("status", ""), severity="warning"),
        make_check("one_batch_actual_detector_train", actual_detector_train,
                   payload.get("real_batch_source", ""), severity="warning"),
        make_check("realbatch_anti_points_gt_0", anti_points > 0,
                   anti_points, severity="warning"),
        make_check("realbatch_preserve_points_gt_0", preserve_points > 0,
                   preserve_points, severity="warning"),
        make_check("realbatch_loss_focus_anti_gt_0", loss_focus_anti > 0.0,
                   loss_focus_anti, severity="warning"),
        make_check("realbatch_loss_focus_preserve_gt_0", loss_focus_preserve > 0.0,
                   loss_focus_preserve, severity="warning"),
        make_check("realbatch_adapter_grad_flow", adapter_grad_flow,
                   adapter_grad_flow, severity="warning"),
        make_check("realbatch_frozen_grad_absent", not frozen_grad_present,
                   frozen_grad_present, severity="warning"),
        make_check("detector_batch_target_mask_injection_ready",
                   detector_batch_ready,
                   "realbatch one-step satisfies detector target-mask gate",
                   severity="warning"),
    ]
    return checks, detector_batch_ready, payload


def write_preflight_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FOCUS-OVD P1 Detector-Level Preflight",
        "",
        f"- status: `{payload['status']}`",
        f"- actual_detector_train: `{payload['actual_detector_train']}`",
        f"- actual_detector_rerun: `{payload['actual_detector_rerun']}`",
        "",
        "## Decision",
        "",
    ]
    if payload["status"] == "BLOCKED_LABEL_TO_LOSS_MAPPING":
        lines.extend([
            "P1 is blocked before detector training.",
            "",
            "The current code has P0 proxy evidence, but it does not yet prove a real detector-level label-to-loss mapping.",
            "No P1 training, detector rerun, safety verdict, or P2 promotion should be claimed from this state.",
            "",
        ])
    elif payload["status"] == "PASS_P1A_SPATIAL_REGION_LOSS":
        lines.extend([
            "P1A spatial-region detector loss mapping is available.",
            "",
            "This is detector-level pseudo-region evidence, not exact pre-NMS provenance and not an AP/safety result.",
            "",
        ])
    elif payload["status"] == "PASS_P1B_EXACT_PROVENANCE_LOSS":
        lines.extend([
            "P1B exact pre-NMS provenance mapping is available.",
            "",
        ])
    else:
        lines.extend(["P1 preflight did not block.", ""])
    lines.extend([
        "## Checks",
        "",
        "| check | ok | severity | detail |",
        "| --- | --- | --- | --- |",
    ])
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(
            f"| {check['name']} | {check['ok']} | {check['severity']} | `{detail}` |")
    lines.extend(["", "## Stop Rule", ""])
    if payload["status"] == "BLOCKED_LABEL_TO_LOSS_MAPPING":
        lines.append(
            "Because loss wiring and label-to-loss mapping cannot both be proven, scripts 58-63 and any detector training/eval are intentionally not run.")
    else:
        lines.append(
            "This preflight does not claim actual detector training/eval; it only reports whether the loss-mapping gate is open.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--p0-exp", type=Path,
                        default=Path("resultmd/exp_focus_ovd_p0_train_eval_20260609"))
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "preflight")
    parser.add_argument("--unblock-exp", type=Path, default=None,
                        help="Optional P1 unblock experiment dir from scripts 64-69.")
    parser.add_argument("--realbatch-exp", type=Path, default=None,
                        help="Optional P1A realbatch experiment dir from scripts 70-74.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    p0_exp = (repo_root / args.p0_exp).resolve() if not args.p0_exp.is_absolute() else args.p0_exp
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    exp_dir = output_dir.parent
    ensure_p1_tree(exp_dir)

    train_split = p0_exp / "configs/p0_train_split.json"
    eval_split = p0_exp / "configs/p0_eval_split.json"
    safety_split = p0_exp / "configs/p0_safety_split.json"
    dense_head = repo_root / DEFAULT_DENSE_HEAD
    focus_config = repo_root / DEFAULT_FOCUS_CONFIG
    corrected_fsv_csv = repo_root / DEFAULT_CORRECTED_FSV_CSV

    checks = [
        make_check("p0_report_exists",
                   (p0_exp / "reports/focus_p0_train_eval_report.md").exists(),
                   str(p0_exp / "reports/focus_p0_train_eval_report.md")),
        inspect_p0_candidates(p0_exp),
        inspect_p0_proxy_only(p0_exp),
        inspect_focus_adapter_exists(dense_head),
        inspect_detector_loss_wiring(dense_head),
        inspect_training_logits_availability(dense_head),
        inspect_label_to_loss_mapping(train_split, mapper_paths=[
            repo_root / "experiments/rotation_semantic_attractor/scripts/focus_label_to_loss_mapping.py",
            repo_root / "experiments/rotation_semantic_attractor/scripts/58_focus_p1_loss_wiring_smoke.py",
        ]),
        inspect_split_distinct(train_split, eval_split),
        inspect_declip_disabled(focus_config),
        make_check("native_support_pkl_exists", (repo_root / DEFAULT_SUPPORT_PKL).exists(),
                   str(repo_root / DEFAULT_SUPPORT_PKL)),
        make_check("baseline_checkpoint_exists", (repo_root / DEFAULT_BASELINE_CKPT).exists(),
                   str(repo_root / DEFAULT_BASELINE_CKPT)),
        make_check("ap_evaluator_availability_checked",
                   (repo_root / "tools/test_rotate.py").exists()
                   or (repo_root / "tools/analysis_tools/eval_metric.py").exists(),
                   "tools/test_rotate.py or tools/analysis_tools/eval_metric.py"),
        make_check("class_migration_metric_available",
                   (p0_exp / "safety/p0_safety_gate_summary.csv").exists(),
                   str(p0_exp / "safety/p0_safety_gate_summary.csv")),
    ]
    checks.extend(inspect_audit_metric_availability(corrected_fsv_csv))

    status = derive_p1_status(
        checks, actual_detector_train=False, actual_detector_rerun=False)
    unblock_payload = {}
    if args.unblock_exp is not None:
        unblock_exp = ((repo_root / args.unblock_exp).resolve()
                       if not args.unblock_exp.is_absolute()
                       else args.unblock_exp)
        unblock_checks, unblock_status, unblock_payload = inspect_unblock_exp(
            unblock_exp)
        checks.extend(unblock_checks)
        if unblock_status is not None:
            status = unblock_status
    manifest = {
        "git_commit": safe_git_commit(repo_root),
        "baseline_checkpoint": str(repo_root / DEFAULT_BASELINE_CKPT),
        "support_pkl": str(repo_root / DEFAULT_SUPPORT_PKL),
        "corrected_fsv_csv": str(corrected_fsv_csv),
        "train_split": str(train_split),
        "eval_split": str(eval_split),
        "safety_split": str(safety_split),
        "variants": P1_VARIANTS,
        "actual_detector_train": False,
        "actual_detector_rerun": False,
        "status": status,
        "unblock_report_status": unblock_payload.get("status", ""),
        "unblock_evidence_level": unblock_payload.get("evidence_level", ""),
    }
    realbatch_payload = {}
    detector_batch_ready = False
    if args.realbatch_exp is not None:
        realbatch_exp = ((repo_root / args.realbatch_exp).resolve()
                         if not args.realbatch_exp.is_absolute()
                         else args.realbatch_exp)
        realbatch_checks, detector_batch_ready, realbatch_payload = (
            inspect_realbatch_exp(realbatch_exp))
        checks.extend(realbatch_checks)
        if detector_batch_ready:
            status = "PASS_P1A_SPATIAL_REGION_REAL_BATCH"
            manifest["status"] = status
            manifest["actual_detector_train"] = True
            manifest["actual_detector_rerun"] = False
            manifest["unblock_evidence_level"] = (
                "DETECTOR_LEVEL_SPATIAL_PSEUDO_REGION_REAL_BATCH")
    original_blocking_failed = [
        check["name"] for check in checks
        if not check["ok"] and check["severity"] == "blocking"
    ]
    effective_blocking_failed = (
        [] if status.startswith("PASS_") else original_blocking_failed)
    payload = {
        **manifest,
        "p0_exp": str(p0_exp),
        "checks": checks,
        "blocking_failed": effective_blocking_failed,
        "original_blocking_failed": original_blocking_failed,
        "detector_batch_target_mask_injection_ready": bool(detector_batch_ready),
        "can_continue_to_detector_training": bool(detector_batch_ready),
        "evidence_level": (
            "DETECTOR_LEVEL_SPATIAL_PSEUDO_REGION_REAL_BATCH"
            if detector_batch_ready else
            unblock_payload.get("evidence_level", "INSUFFICIENT_P1_EVIDENCE")),
        "realbatch_report_status": realbatch_payload.get("status", ""),
        "realbatch_actual_detector_train": bool(
            realbatch_payload.get("actual_detector_train", False)),
        "realbatch_num_focus_anti_points": int(
            realbatch_payload.get("num_focus_anti_points", 0) or 0),
        "realbatch_num_focus_preserve_points": int(
            realbatch_payload.get("num_focus_preserve_points", 0) or 0),
        "realbatch_loss_focus_anti": float(
            realbatch_payload.get("loss_focus_anti", 0.0) or 0.0),
        "realbatch_loss_focus_preserve": float(
            realbatch_payload.get("loss_focus_preserve", 0.0) or 0.0),
    }
    write_json(output_dir / "p1_preflight.json", payload)
    write_json(exp_dir / "manifest.json", manifest)
    write_json(exp_dir / "manifests/p1_manifest.json", manifest)
    write_preflight_markdown(output_dir / "p1_preflight.md", payload)
    print(json.dumps({
        "status": status,
        "actual_detector_train": bool(detector_batch_ready),
        "actual_detector_rerun": False,
        "detector_batch_target_mask_injection_ready": bool(detector_batch_ready),
        "can_continue_to_detector_training": bool(detector_batch_ready),
        "blocking_failed": payload["blocking_failed"],
    }, indent=2))
    return 0 if (status == "READY_FOR_LOSS_WIRING_SMOKE"
                 or status.startswith("PASS_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
