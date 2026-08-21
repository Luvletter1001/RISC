#!/usr/bin/env python3
"""Run FOCUS-OVD P0 small proxy evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    append_manifest,
    baseline_eval_metrics,
    build_stratified_splits,
    ensure_exp_tree,
    eval_proxy_metrics,
    labeled_rows,
    markdown_table,
    parse_variants,
    read_csv_rows,
    read_json,
    write_csv_rows,
    write_json,
)


SUMMARY_FIELDS = [
    "variant_id", "eval_status", "train_status", "angle_status",
    "actual_detector_rerun", "small_train_type", "corrected_FSV",
    "corrected_FSV_delta", "corrected_FSV_reduction", "dense_sv_ratio",
    "dense_sv_ratio_delta", "SV_pred_per_img", "LV_pred_per_img",
    "det/img", "true_SV_positive_control_retention",
    "annotation_missing_true_vehicle_retention", "migration_mass_ratio",
    "top_migrated_classes", "degenerate_large_sv_ratio",
    "padding_artifact_sv_rate", "support_delta_norm_ratio",
    "support_inter_class_cos_mean", "support_inter_class_cos_max",
    "alpha", "kappa_gate_active_rate", "orientation_confidence_mean",
    "sv_logit_delta", "top1_shift_rate",
]


def parse_angles(value: str) -> list[int]:
    return [int(x) for x in value.replace(",", " ").split() if x.strip()]


def train_status(train_dir: Path, variant_id: str) -> str:
    path = train_dir / variant_id / "train_status.json"
    if not path.exists():
        return "NOT_TRAIN_REQUIRED" if variant_id in {
            "V00_baseline", "V01_focus_zero", "V02_focus_random_orientation",
            "V03_direction_text_prompt_only", "V10_orientation_probe_only"} else "MISSING_TRAIN_STATUS"
    try:
        return str(read_json(path).get("status", "UNKNOWN"))
    except Exception:
        return "TRAIN_STATUS_READ_ERROR"


def write_variant_outputs(out_dir: Path, metrics: dict, split_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_dir / "eval_summary.csv", [metrics], SUMMARY_FIELDS)
    md = [f"# {metrics['variant_id']} P0 Eval Summary", ""]
    md.extend(markdown_table([metrics], SUMMARY_FIELDS))
    (out_dir / "eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sample_rows = []
    for row in split_rows[:50]:
        sample_rows.append({
            "crop_id": row.get("crop_id", ""),
            "audit_category": row.get("audit_category", ""),
            "focus_label": row.get("focus_label", ""),
            "variant_id": metrics["variant_id"],
            "proxy_eval": "true",
        })
    write_csv_rows(out_dir / "eval_rows.csv", sample_rows)
    write_csv_rows(out_dir / "corrected_fsv_eval.csv", [{
        "variant_id": metrics["variant_id"],
        "corrected_FSV": metrics.get("corrected_FSV", ""),
        "corrected_FSV_delta": metrics.get("corrected_FSV_delta", ""),
        "corrected_false_sv_rejection_rate": metrics.get("corrected_false_sv_rejection_rate", ""),
        "annotation_missing_true_vehicle_retention": metrics.get("annotation_missing_true_vehicle_retention", ""),
    }])
    write_csv_rows(out_dir / "safety_metrics.csv", [{
        key: metrics.get(key, "") for key in [
            "variant_id", "true_SV_positive_control_retention",
            "annotation_missing_true_vehicle_retention", "det/img",
            "lowrisk_det_inflation", "degenerate_large_sv_ratio",
            "padding_artifact_sv_rate", "det_explosion_flag",
        ]
    }])
    write_csv_rows(out_dir / "support_geometry_eval.csv", [{
        key: metrics.get(key, "") for key in [
            "variant_id", "support_delta_norm_ratio",
            "support_inter_class_cos_mean", "support_inter_class_cos_max",
            "alpha",
        ]
    }])
    write_csv_rows(out_dir / "migration_metrics.csv", [{
        key: metrics.get(key, "") for key in [
            "variant_id", "migration_mass_ratio", "top_migrated_classes",
            "class_JS", "class_KL",
        ]
    }])
    write_csv_rows(out_dir / "dense_sv_metrics.csv", [{
        key: metrics.get(key, "") for key in [
            "variant_id", "dense_sv_ratio", "dense_sv_ratio_delta",
            "preNMS_FR_SV", "postNMS_FR_SV", "SV_pred_per_img",
        ]
    }])
    (out_dir / "qualitative_examples").mkdir(exist_ok=True)
    write_json(out_dir / "eval_manifest.json", metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-index", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--angles", default="0,30,60,90,120,150,180,210,240,270,300,330")
    args = parser.parse_args()

    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    _index = read_csv_rows(args.config_index)
    split = build_stratified_splits(labeled_rows(), seed=20260609, train_ratio=0.8)
    baseline = baseline_eval_metrics(split)
    angles = parse_angles(args.angles)
    evaluated = []
    for variant_id in parse_variants(args.variants):
        status = train_status(args.train_dir, variant_id)
        metrics = eval_proxy_metrics(
            variant_id, baseline, split, angle_count=len(angles),
            train_status=status)
        if variant_id == "V00_baseline":
            metrics.update({
                "corrected_FSV": baseline["corrected_FSV"],
                "corrected_FSV_delta": 0.0,
                "corrected_FSV_reduction": 0.0,
                "dense_sv_ratio": baseline["dense_sv_ratio"],
                "dense_sv_ratio_delta": 0.0,
            })
        write_variant_outputs(args.output_dir / variant_id, metrics, split["eval"])
        evaluated.append(metrics)
    write_csv_rows(args.output_dir / "p0_eval_all_variants.csv", evaluated, SUMMARY_FIELDS)
    append_manifest(
        exp_dir, args.repo_root.resolve(),
        {"stage": "p0_eval", "status": "WRITTEN", "variants": [m["variant_id"] for m in evaluated], "angles": angles})
    print(json.dumps({"evaluated": [m["variant_id"] for m in evaluated], "angles": len(angles)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
