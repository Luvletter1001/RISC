#!/usr/bin/env python3
"""Build spatial-region detector loss targets from verified FOCUS crop labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_EXP_DIR = Path(
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609")
DEFAULT_SPLIT = Path(
    "resultmd/exp_focus_ovd_p0_train_eval_20260609/configs/"
    "p0_train_split.json")
FALSE_LABELS = {"non_vehicle_background", "non_vehicle_object_conflict"}
TRUE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}
EXCLUDE_CATEGORIES = {"degenerate_large_sv_box", "padding_artifact"}
FIELDS = [
    "focus_target_id", "crop_id", "audit_category", "split_role",
    "tile_id", "angle", "model_name", "source_stage", "source_raw_path",
    "source_raw_index", "image_path", "metadata_json", "human_label",
    "focus_label", "target_role", "valid_for_loss", "invalid_reason",
    "coordinate_frame", "coordinate_frame_consistent", "target_polygon",
    "target_center_x", "target_center_y", "raw_box_area", "score",
    "pred_class", "padding_overlap_ratio", "valid_mask_ratio_inside_box",
    "use_as_hard_negative", "use_as_preserve_positive",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]],
              fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(split_path: Path) -> list[dict[str, Any]]:
    payload = read_json(split_path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"{split_path} does not contain a rows list")
    return rows


def metadata_for(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row.get("metadata_json", ""))
    if not path.exists():
        return {}
    return read_json(path)


def raw_record(meta: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    value = meta.get("raw_prediction_record") or row.get("raw_prediction_record")
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return {}


def center_from_record(record: dict[str, Any],
                       polygon: list[list[float]]) -> tuple[float, float]:
    box = record.get("box") or []
    if len(box) >= 2:
        return float(box[0]), float(box[1])
    if polygon:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.0, 0.0


def coordinate_frame(meta: dict[str, Any]) -> str:
    image_path = str(meta.get("image_path", ""))
    raw_path = str(meta.get("raw_path", ""))
    if "angle_sweep" in image_path or "/angle_" in raw_path:
        return "rotated_angle_sweep"
    return "unknown"


def target_role(row: dict[str, Any]) -> tuple[str, str]:
    audit_category = str(row.get("audit_category", ""))
    human_label = str(row.get("human_label", ""))
    focus_label = str(row.get("focus_label", ""))
    if audit_category in EXCLUDE_CATEGORIES:
        return "exclude", f"excluded_category:{audit_category}"
    if human_label in {"ambiguous", "invalid_visualization"}:
        return "exclude", f"excluded_human_label:{human_label}"
    if audit_category == "true_sv_positive_control":
        return "preserve_positive", ""
    if human_label in TRUE_LABELS or focus_label == "annotation_missing_true_vehicle":
        return "preserve_positive", ""
    if (human_label in FALSE_LABELS
            or focus_label == "corrected_false_sv"
            or as_bool(row.get("use_as_hard_negative"))):
        return "anti_negative", ""
    return "exclude", "not_a_focus_loss_target"


def build_target(row: dict[str, Any], idx: int) -> dict[str, Any]:
    meta = metadata_for(row)
    record = raw_record(meta, row)
    polygon = record.get("polygon") or []
    cx, cy = center_from_record(record, polygon)
    role, invalid_reason = target_role(row)
    coord_ok = as_bool(meta.get(
        "coordinate_frame_consistent",
        row.get("coordinate_frame_consistent", "false")))
    pad = as_float(meta.get("padding_overlap_ratio",
                            row.get("padding_overlap_ratio")), default=1.0)
    valid_mask = as_float(meta.get("valid_mask_ratio_inside_box",
                                   row.get("valid_mask_ratio_inside_box")),
                          default=0.0)
    area = as_float(meta.get("raw_box_area", row.get("raw_box_area")),
                    default=0.0)
    invalid_reasons = [invalid_reason] if invalid_reason else []
    if not polygon:
        invalid_reasons.append("missing_polygon")
    if not coord_ok:
        invalid_reasons.append("coordinate_frame_inconsistent")
    if pad > 0.25:
        invalid_reasons.append(f"padding_overlap_gt_0.25:{pad:.6f}")
    if valid_mask < 0.75:
        invalid_reasons.append(f"valid_mask_lt_0.75:{valid_mask:.6f}")
    if area <= 1.0:
        invalid_reasons.append(f"area_too_small:{area:.6f}")
    valid_for_loss = role in {"anti_negative", "preserve_positive"} and not invalid_reasons
    return {
        "focus_target_id": f"focus_p1a_{idx:05d}",
        "crop_id": row.get("crop_id", ""),
        "audit_category": row.get("audit_category", ""),
        "split_role": row.get("split_role", ""),
        "tile_id": row.get("tile_id", meta.get("tile_id", "")),
        "angle": row.get("angle", meta.get("angle", "")),
        "model_name": row.get("model_name", meta.get("model_name", "")),
        "source_stage": record.get("stage", ""),
        "source_raw_path": meta.get("raw_path", ""),
        "source_raw_index": meta.get("raw_index", record.get("raw_index", "")),
        "image_path": meta.get("image_path", ""),
        "metadata_json": row.get("metadata_json", ""),
        "human_label": row.get("human_label", ""),
        "focus_label": row.get("focus_label", ""),
        "target_role": role,
        "valid_for_loss": str(bool(valid_for_loss)).lower(),
        "invalid_reason": ";".join(v for v in invalid_reasons if v),
        "coordinate_frame": coordinate_frame(meta),
        "coordinate_frame_consistent": str(bool(coord_ok)).lower(),
        "target_polygon": json.dumps(polygon),
        "target_center_x": f"{cx:.6f}",
        "target_center_y": f"{cy:.6f}",
        "raw_box_area": f"{area:.6f}",
        "score": row.get("score", meta.get("score", record.get("score", ""))),
        "pred_class": row.get("pred_class", record.get("class_name", "")),
        "padding_overlap_ratio": f"{pad:.6f}",
        "valid_mask_ratio_inside_box": f"{valid_mask:.6f}",
        "use_as_hard_negative": row.get("use_as_hard_negative", ""),
        "use_as_preserve_positive": row.get("use_as_preserve_positive", ""),
    }


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    out = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(f, "")).replace("|", "\\|")
                                      for f in fields) + " |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    split = args.input_split if args.input_split.is_absolute() else repo / args.input_split
    exp_dir = args.exp_dir if args.exp_dir.is_absolute() else repo / args.exp_dir
    rows = load_rows(split)
    targets = [build_target(row, idx) for idx, row in enumerate(rows)]
    target_dir = exp_dir / "targets"
    csv_path = target_dir / "focus_spatial_region_targets.csv"
    write_csv(csv_path, targets)

    role_counts = Counter(row["target_role"] for row in targets)
    valid_counts = Counter(row["target_role"] for row in targets
                           if row["valid_for_loss"] == "true")
    invalid_counts = Counter()
    for row in targets:
        for reason in str(row["invalid_reason"]).split(";"):
            if reason:
                invalid_counts[reason.split(":")[0]] += 1
    summary = {
        "status": (
            "P1A_SPATIAL_REGION_TARGETS_READY"
            if valid_counts["anti_negative"] > 0
            and valid_counts["preserve_positive"] > 0
            else "P1A_SPATIAL_REGION_TARGETS_INSUFFICIENT"),
        "input_split": str(split),
        "target_csv": str(csv_path),
        "total_rows": len(targets),
        "role_counts": dict(role_counts),
        "valid_role_counts": dict(valid_counts),
        "invalid_reason_counts": dict(invalid_counts),
        "coordinate_frame": "rotated_angle_sweep",
        "evidence_level": "DETECTOR_LEVEL_SPATIAL_PSEUDO_REGION",
    }
    write_json(target_dir / "focus_spatial_region_targets.json", targets)
    write_json(target_dir / "focus_spatial_region_targets_summary.json", summary)
    md = [
        "# FOCUS P1 Spatial Region Targets",
        "",
        f"- status: `{summary['status']}`",
        f"- total_rows: `{summary['total_rows']}`",
        f"- anti_negative_valid: `{summary['valid_role_counts'].get('anti_negative', 0)}`",
        f"- preserve_positive_valid: `{summary['valid_role_counts'].get('preserve_positive', 0)}`",
        f"- evidence_level: `{summary['evidence_level']}`",
        "",
        "## Counts",
        "",
    ]
    count_rows = [
        {
            "role": role,
            "total": role_counts.get(role, 0),
            "valid": valid_counts.get(role, 0),
        }
        for role in ("anti_negative", "preserve_positive", "exclude")
    ]
    md.extend(markdown_table(count_rows, ["role", "total", "valid"]))
    md.append("")
    (target_dir / "focus_spatial_region_targets_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "P1A_SPATIAL_REGION_TARGETS_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
