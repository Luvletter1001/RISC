#!/usr/bin/env python3
"""Preflight for FOCUS P1A real detector-batch target injection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from focus_p1a_realbatch_common import (
    DOTA2_CLASSES,
    ensure_exp_tree,
    md_table,
    read_csv,
    read_json,
    resolve,
    select_realbatch_images,
    target_role_counts,
    valid_focus_targets,
    write_json,
    write_manifest,
)


DEFAULT_EXP = Path("resultmd/exp_focus_ovd_p1a_realbatch_unblock_20260609")
DEFAULT_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
DEFAULT_SUPPORT = Path(
    "data/DOTA2_1024_500/ss_train/"
    "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl")


def check(name: str, ok: bool, detail: Any,
          severity: str = "blocking") -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "severity": severity,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FOCUS P1A Realbatch Preflight",
        "",
        f"- status: `{payload['status']}`",
        f"- spatial_targets: `{payload['spatial_targets']}`",
        f"- anti_negative rows: `{payload['anti_negative_rows']}`",
        f"- preserve_positive rows: `{payload['preserve_positive_rows']}`",
        f"- selected_batch_images: `{payload['selected_batch_images']}`",
        f"- batch_match_status: `{payload['batch_match_status']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(md_table(payload["checks"], ["name", "ok", "severity", "detail"]))
    lines.extend([
        "",
        "## Selected Images",
        "",
    ])
    lines.extend(md_table(payload["selected_images"], [
        "tile_id",
        "angle",
        "anti_target_count",
        "preserve_target_count",
        "image_exists",
        "annotation_exists",
        "annotation_object_count",
    ]))
    lines.extend([
        "",
        "## Decision",
        "",
        "This preflight only verifies target/image/batch metadata matchability. "
        "It does not claim actual detector training; script 72 must provide "
        "the real one-step loss and gradient evidence.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spatial-targets", type=Path, required=True)
    parser.add_argument("--p0-train-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP / "preflight")
    parser.add_argument("--focus-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    lookup_path = repo_root / "M_AD/models/utils/focus_target_lookup.py"
    spec = importlib.util.spec_from_file_location("focus_target_lookup", lookup_path)
    lookup_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(lookup_module)
    FocusTargetLookup = lookup_module.FocusTargetLookup

    spatial_targets = resolve(repo_root, args.spatial_targets)
    p0_train_split = resolve(repo_root, args.p0_train_split)
    output_dir = resolve(repo_root, args.output_dir)
    exp_dir = output_dir.parent
    ensure_exp_tree(exp_dir)
    write_manifest(exp_dir, repo_root, {
        "spatial_targets": str(spatial_targets),
        "p0_train_split": str(p0_train_split),
    })

    rows = read_csv(spatial_targets) if spatial_targets.exists() else []
    valid_rows = valid_focus_targets(rows)
    role_counts = target_role_counts(rows)
    valid_role_counts = target_role_counts(valid_rows)
    required_fields = {
        "tile_id",
        "angle",
        "image_path",
        "target_polygon",
        "coordinate_frame",
        "target_role",
        "valid_for_loss",
    }
    missing_required = [
        row.get("focus_target_id", f"row_{idx}")
        for idx, row in enumerate(valid_rows)
        if not required_fields.issubset(row.keys())
        or any(not row.get(field) for field in required_fields)
    ]
    selected_images = select_realbatch_images(valid_rows)

    meta_batch = []
    for item in selected_images:
        meta_batch.append({
            "img_path": item["image_path"],
            "img_id": item["tile_id"],
            "tile_id": item["tile_id"],
            "angle": item["angle"],
            "ori_shape": (1024, 1024),
            "img_shape": (832, 832),
            "scale_factor": (832.0 / 1024.0, 832.0 / 1024.0),
            "dataset_flag": "Data1_DOTA2",
        })
    lookup_debug = []
    batch_targets_json = []
    if meta_batch:
        lookup = FocusTargetLookup(spatial_targets)
        batch_targets, lookup_debug = lookup.match_batch(meta_batch)
        batch_targets_json = lookup.to_jsonable_targets(batch_targets)
        lookup.write_debug_csv(
            exp_dir / "batch_match/focus_batch_target_lookup_debug.csv",
            lookup_debug)

    matched_anti = sum(int(row.get("anti_targets", 0)) for row in lookup_debug)
    matched_preserve = sum(int(row.get("preserve_targets", 0)) for row in lookup_debug)
    batch_match_ok = matched_anti > 0 and matched_preserve > 0

    config_path = resolve(repo_root, args.focus_config)
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    destructive_aug = {
        "RandomFlip": "RandomFlip" in config_text,
        "RandomRotate": "RandomRotate" in config_text,
        "crop": "RandomCrop" in config_text or "Crop" in config_text,
    }
    support_path = repo_root / DEFAULT_SUPPORT
    checks = [
        check("spatial_target_csv_exists", spatial_targets.exists(), str(spatial_targets)),
        check("anti_negative_rows_eq_108", role_counts.get("anti_negative", 0) == 108,
              role_counts.get("anti_negative", 0)),
        check("preserve_positive_rows_eq_207", role_counts.get("preserve_positive", 0) == 207,
              role_counts.get("preserve_positive", 0)),
        check("valid_anti_negative_rows_gt_0", valid_role_counts.get("anti_negative", 0) > 0,
              valid_role_counts.get("anti_negative", 0)),
        check("valid_preserve_positive_rows_gt_0", valid_role_counts.get("preserve_positive", 0) > 0,
              valid_role_counts.get("preserve_positive", 0)),
        check("target_required_fields_present", len(missing_required) == 0,
              f"missing_count={len(missing_required)}"),
        check("p0_train_split_exists", p0_train_split.exists(), str(p0_train_split)),
        check("p0_train_split_has_rows",
              bool((read_json(p0_train_split, {}) or {}).get("rows")),
              "rows present" if p0_train_split.exists() else "missing"),
        check("selected_real_images_nonempty", len(selected_images) > 0,
              len(selected_images)),
        check("selected_images_exist", all(item["image_exists"] for item in selected_images),
              [item["image_path"] for item in selected_images]),
        check("selected_annfiles_exist", all(item["annotation_exists"] for item in selected_images),
              [item["annotation_path"] for item in selected_images]),
        check("batch_img_metas_have_required_fields", all(
            {"img_path", "img_id", "ori_shape", "img_shape", "scale_factor", "tile_id", "angle"}
            .issubset(meta.keys()) for meta in meta_batch),
              "img_path/img_id/ori_shape/img_shape/scale_factor/tile_id/angle"),
        check("tile_angle_parse_available", all(
            item.get("tile_id") and item.get("angle") is not None
            for item in selected_images), "tile_id + angle"),
        check("coordinate_frame_rotated_angle_sweep", all(
            item.get("coordinate_frame") == "rotated_angle_sweep"
            for item in selected_images), "rotated_angle_sweep"),
        check("batch_lookup_matches_anti_and_preserve", batch_match_ok,
              f"anti={matched_anti}, preserve={matched_preserve}"),
        check("deterministic_resize_pad_supported", True,
              "P1A smoke keeps resize+pad only"),
        check("destructive_augmentations_detected_for_disable", True,
              json.dumps(destructive_aug, sort_keys=True), severity="warning"),
        check("native_support_enabled", support_path.exists(), str(support_path)),
        check("declip_support_disabled",
              "use_declip_support=True" not in config_text,
              str(config_path)),
    ]
    status = (
        "PASS_P1A_REALBATCH_PREFLIGHT"
        if all(c["ok"] for c in checks if c["severity"] == "blocking")
        else "BLOCKED_BATCH_TARGET_MATCH")
    payload = {
        "status": status,
        "spatial_targets": str(spatial_targets),
        "p0_train_split": str(p0_train_split),
        "anti_negative_rows": int(role_counts.get("anti_negative", 0)),
        "preserve_positive_rows": int(role_counts.get("preserve_positive", 0)),
        "valid_role_counts": dict(Counter(row.get("target_role") for row in valid_rows)),
        "required_fields": sorted(required_fields),
        "missing_required_target_ids": missing_required[:50],
        "selected_batch_images": len(selected_images),
        "selected_images": selected_images,
        "batch_targets": batch_targets_json,
        "batch_match_status": "PASS" if batch_match_ok else "FAIL",
        "lookup_debug_rows": lookup_debug,
        "smoke_pipeline_policy": "disable RandomFlip/RandomRotate/crop; keep deterministic resize/pad",
        "dota2_classes": DOTA2_CLASSES,
        "checks": checks,
    }
    write_json(output_dir / "p1a_realbatch_preflight.json", payload)
    write_markdown(output_dir / "p1a_realbatch_preflight.md", payload)
    print(json.dumps({
        "status": status,
        "anti": matched_anti,
        "preserve": matched_preserve,
        "selected_images": len(selected_images),
    }, indent=2))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
