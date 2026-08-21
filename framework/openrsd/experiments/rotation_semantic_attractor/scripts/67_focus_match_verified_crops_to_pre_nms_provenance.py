#!/usr/bin/env python3
"""Match verified crop audit rows to captured pre-NMS provenance candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FALSE_LABELS = {"non_vehicle_background", "non_vehicle_object_conflict"}
TRUE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]],
                   fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_box(value: Any) -> list[float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        value = json.loads(text)
    return [float(v) for v in value]


def crop_box(row: dict[str, Any]) -> list[float]:
    record = row.get("raw_prediction_record", "")
    if not record and row.get("metadata_json"):
        meta = json.load(open(row["metadata_json"], "r", encoding="utf-8"))
        record = meta.get("raw_prediction_record", "")
    if isinstance(record, str):
        record = json.loads(record)
    return parse_box(record.get("box", []))


def to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    if len(box) >= 5:
        cx, cy, w, h = box[:4]
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
    if len(box) >= 4:
        x1, y1, x2, y2 = box[:4]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return 0.0, 0.0, 0.0, 0.0


def hbb_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = to_xyxy(box_a)
    bx1, by1, bx2, by2 = to_xyxy(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def target_role(row: dict[str, Any]) -> str:
    existing = str(row.get("target_role", ""))
    if existing in {"anti_negative", "preserve_positive", "exclude"}:
        return existing
    if row.get("focus_label") == "corrected_false_sv" or row.get("human_label") in FALSE_LABELS:
        return "anti_negative"
    if (row.get("focus_label") == "annotation_missing_true_vehicle"
            or row.get("human_label") in TRUE_LABELS
            or row.get("audit_category") == "true_sv_positive_control"):
        return "preserve_positive"
    return "exclude"


def match_crop_to_provenance(
        crop: dict[str, Any],
        candidates: list[dict[str, Any]],
        iou_thr: float = 0.80,
        score_tol: float = 1e-4) -> dict[str, str]:
    crop_b = crop_box(crop)
    crop_score = float(crop.get("score") or 0.0)
    role = target_role(crop)
    best: tuple[float, float, dict[str, Any]] | None = None
    for cand in candidates:
        if str(cand.get("tile_id")) != str(crop.get("tile_id")):
            continue
        if int(float(cand.get("angle", -999))) != int(float(crop.get("angle", -998))):
            continue
        if str(cand.get("class_name", "")).lower().replace("_", "-") != "small-vehicle":
            continue
        cand_box = parse_box(cand.get("box", "[]"))
        iou = hbb_iou(crop_b, cand_box)
        score_diff = abs(float(cand.get("score") or 0.0) - crop_score)
        if iou < iou_thr or score_diff > score_tol:
            continue
        key = (iou, -score_diff)
        if best is None or key > (best[0], best[1]):
            best = (iou, -score_diff, cand)
    if best is None:
        return {
            "crop_id": str(crop.get("crop_id", "")),
            "prediction_id": "",
            "feature_level": "",
            "grid_x": "",
            "grid_y": "",
            "anchor_id_or_point_id": "",
            "target_role": role,
            "match_iou": "0.000000",
            "score_diff": "",
            "valid_for_exact_loss": "false",
            "invalid_reason": "no_matching_pre_nms_candidate",
        }
    iou, neg_score_diff, cand = best
    return {
        "crop_id": str(crop.get("crop_id", "")),
        "prediction_id": str(cand.get("prediction_id", "")),
        "feature_level": str(cand.get("feature_level", "")),
        "grid_x": str(cand.get("grid_x", "")),
        "grid_y": str(cand.get("grid_y", "")),
        "anchor_id_or_point_id": str(cand.get("anchor_id_or_point_id", "")),
        "target_role": role,
        "match_iou": f"{iou:.6f}",
        "score_diff": f"{abs(neg_score_diff):.6f}",
        "valid_for_exact_loss": "true",
        "invalid_reason": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--human-label-csv", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    human_label_csv = args.human_label_csv if args.human_label_csv.is_absolute() else repo / args.human_label_csv
    provenance_csv = args.provenance if args.provenance.is_absolute() else repo / args.provenance
    out_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    crops = read_csv_rows(human_label_csv)
    provenance = read_csv_rows(provenance_csv)
    rows = [match_crop_to_provenance(row, provenance) for row in crops]
    fields = [
        "crop_id", "prediction_id", "feature_level", "grid_x", "grid_y",
        "anchor_id_or_point_id", "target_role", "match_iou", "score_diff",
        "valid_for_exact_loss", "invalid_reason",
    ]
    write_csv_rows(out_dir / "focus_pre_nms_exact_targets.csv", rows, fields)
    valid_focus = [r for r in rows if r["target_role"] != "exclude"]
    matched = [r for r in valid_focus if r["valid_for_exact_loss"] == "true"]
    coverage = len(matched) / len(valid_focus) if valid_focus else 0.0
    status = (
        "P1B_EXACT_PROVENANCE_LOSS_READY"
        if coverage >= 0.70 else "P1B_DIAGNOSTIC_INSUFFICIENT_COVERAGE")
    summary = {
        "status": status,
        "human_label_csv": str(human_label_csv),
        "provenance_csv": str(provenance_csv),
        "exact_targets_csv": str(out_dir / "focus_pre_nms_exact_targets.csv"),
        "total_crop_rows": len(crops),
        "valid_focus_rows": len(valid_focus),
        "matched_rows": len(matched),
        "exact_match_coverage": coverage,
        "coverage_threshold": 0.70,
        "evidence_level": (
            "EXACT_PRE_NMS_PROVENANCE"
            if coverage >= 0.70 else "INSUFFICIENT_EXACT_PRE_NMS_PROVENANCE"),
    }
    (out_dir / "focus_pre_nms_exact_targets_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = [
        "# FOCUS P1 Pre-NMS Exact Target Matching",
        "",
        f"- status: `{status}`",
        f"- valid_focus_rows: `{len(valid_focus)}`",
        f"- matched_rows: `{len(matched)}`",
        f"- exact_match_coverage: `{coverage:.6f}`",
    ]
    (out_dir / "focus_pre_nms_exact_targets_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "coverage": coverage}, indent=2))
    return 0 if status == "P1B_EXACT_PROVENANCE_LOSS_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
