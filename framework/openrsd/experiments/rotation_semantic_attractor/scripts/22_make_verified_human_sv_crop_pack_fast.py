#!/usr/bin/env python3
"""Build a verified-fast replacement human SV crop pack from integrity audit rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


TARGETS = {
    "valid_unmatched_sv": 50,
    "strict_object_flip": 30,
    "true_sv_positive_control": 30,
    "degenerate_large_sv_box": 30,
    "padding_artifact": 30,
}

HEADERS = [
    "crop_id",
    "audit_category",
    "image_path_full_tile",
    "image_path_zoom",
    "image_path_gt_context",
    "metadata_json",
    "model_name",
    "tile_id",
    "angle",
    "score",
    "pred_class",
    "raw_box_type",
    "raw_box_angle",
    "raw_box_area",
    "valid_mask_ratio_inside_box",
    "padding_overlap_ratio",
    "best_gt_class",
    "best_gt_iou",
    "auto_label",
    "valid_for_human_audit",
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def points_from_record(record: dict[str, Any]) -> list[tuple[float, float]]:
    poly = record.get("polygon")
    if poly:
        return [(float(p[0]), float(p[1])) for p in poly]
    box = record.get("box") or []
    if len(box) >= 8:
        return [(float(box[i]), float(box[i + 1])) for i in range(0, 8, 2)]
    if len(box) == 5:
        cx, cy, w, h, a = [float(x) for x in box]
        ca, sa = math.cos(a), math.sin(a)
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in corners]
    return []


def parse_dota_ann(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            pts = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 8, 2)]
        except Exception:
            continue
        rows.append({"points": pts, "class_name": parts[8]})
    return rows


def source_paths(run_root: Path, model: str, tile: str, angle: int) -> dict[str, Path]:
    return {
        "valid_mask": run_root / "valid_masks" / model / tile / f"angle_{angle:03d}.png",
        "rotated_gt": run_root / "rotated_gt" / model / tile / f"{tile}_angle_{angle:03d}.txt",
    }


def data_image_path(data_root: Path, tile: str, angle: int) -> Path:
    return data_root / f"angle_{angle:03d}" / "images" / f"{tile}.png"


def data_gt_path(data_root: Path, tile: str, angle: int) -> Path:
    return data_root / f"angle_{angle:03d}" / "annfiles" / f"{tile}.txt"


def bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def draw_overlay(
    image_path: Path,
    out_path: Path,
    pred_points: list[tuple[float, float]],
    gt_rows: list[dict[str, Any]] | None,
    title: str,
    zoom_center: tuple[float, float] | None = None,
    zoom_size: int | None = None,
    draw_hbb_fallback: bool = False,
) -> bool:
    if not image_path.exists() or not pred_points:
        return False
    img = Image.open(image_path).convert("RGB")
    if zoom_center and zoom_size:
        cx, cy = zoom_center
        cx = min(max(cx, 0), max(0, img.size[0] - 1))
        cy = min(max(cy, 0), max(0, img.size[1] - 1))
        half = zoom_size // 2
        left = max(0, int(round(cx - half)))
        upper = max(0, int(round(cy - half)))
        right = min(img.size[0], int(round(cx + half)))
        lower = min(img.size[1], int(round(cy + half)))
        if right <= left:
            right = min(img.size[0], left + 1)
        if lower <= upper:
            lower = min(img.size[1], upper + 1)
        crop_box = (left, upper, right, lower)
        offset = (crop_box[0], crop_box[1])
        img = img.crop(crop_box)
    else:
        offset = (0, 0)
    ox, oy = offset
    draw = ImageDraw.Draw(img, "RGBA")
    if gt_rows:
        for gt in gt_rows:
            pts = [(x - ox, y - oy) for x, y in gt["points"]]
            color = (0, 255, 0, 180) if gt["class_name"] == "small-vehicle" else (0, 160, 255, 120)
            draw.line(pts + [pts[0]], fill=color, width=2)
    if draw_hbb_fallback:
        b = bounds(pred_points)
        if b:
            x1, y1, x2, y2 = b
            draw.rectangle((x1 - ox, y1 - oy, x2 - ox, y2 - oy), outline=(255, 0, 0, 255), width=4)
            title += " HBB_FALLBACK_RENDERING"
    else:
        pts = [(x - ox, y - oy) for x, y in pred_points]
        draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=4)
    draw.rectangle((0, 0, img.size[0], 42), fill=(0, 0, 0, 165))
    draw.text((5, 5), title[:190], fill=(255, 255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def row_category(row: dict[str, str]) -> str | None:
    raw_ok = row.get("source_prediction_found") == "true" and row.get("raw_box_type") in {"obb", "polygon"}
    mask_ok = as_float(row.get("valid_mask_ratio_inside_box", ""), 0.0) >= 0.85
    pad_ok = as_float(row.get("padding_overlap_ratio", ""), 1.0) <= 0.10
    center_ok = row.get("box_center_in_valid_region") == "true"
    oversized = row.get("is_oversized_for_small_vehicle") == "true"
    best_cls = row.get("best_gt_class", "")
    best_iou = as_float(row.get("best_gt_iou", ""), 0.0)
    nearest_sv = as_float(row.get("nearest_sv_gt_iou", ""), 0.0)
    risk = row.get("risk_group", "")
    if raw_ok and (not mask_ok or not pad_ok or not center_ok):
        return "padding_artifact"
    if raw_ok and oversized:
        return "degenerate_large_sv_box"
    if raw_ok and mask_ok and pad_ok and center_ok and risk == "true_sv_rich" and nearest_sv > 0.10:
        return "true_sv_positive_control"
    if raw_ok and mask_ok and pad_ok and center_ok and best_cls not in {"", "NO_GT_OVERLAP", "small-vehicle", "large-vehicle"} and best_iou > 0.10:
        return "strict_object_flip"
    if raw_ok and mask_ok and pad_ok and center_ok and risk == "no_annotated_sv_gt" and nearest_sv < 0.10 and not oversized:
        return "valid_unmatched_sv"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", default="experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle")
    parser.add_argument("--visual-summary-dir", required=True)
    parser.add_argument("--data-image-root", default="data/DOTA1_1024_500/angle_sweep_val/realistic")
    parser.add_argument("--data-gt-root", default="data/DOTA1_1024_500/angle_sweep_val/realistic")
    args = parser.parse_args()

    root = Path(args.visual_summary_dir)
    audit_dir = root / "audit"
    pack_root = root / "crops_verified_fast"
    run_root = Path(args.run_root)
    image_root = Path(args.data_image_root)
    gt_root = Path(args.data_gt_root)
    audit_rows = read_csv(audit_dir / "urgent_crop_pack_integrity_audit.csv")
    counts: Counter[str] = Counter()
    out_rows: list[dict[str, Any]] = []
    for row in audit_rows:
        cat = row_category(row)
        if cat is None or counts[cat] >= TARGETS[cat]:
            continue
        raw_record = parse_jsonish(row.get("raw_prediction_record", ""))
        if not isinstance(raw_record, dict):
            continue
        pred_points = points_from_record(raw_record)
        if not pred_points:
            continue
        model = row["model_name"]
        tile = row["tile_id"]
        angle = int(as_float(row["angle"], 0))
        image_path = data_image_path(image_root, tile, angle)
        sp = source_paths(run_root, model, tile, angle)
        gt_path = sp["rotated_gt"] if sp["rotated_gt"].exists() else data_gt_path(gt_root, tile, angle)
        gt_rows = parse_dota_ann(gt_path)
        center = (sum(x for x, _ in pred_points) / len(pred_points), sum(y for _, y in pred_points) / len(pred_points))
        crop_id = f"{cat}_{counts[cat]:04d}"
        sample_dir = pack_root / cat / crop_id
        title = (
            f"{model} {tile} angle={angle} score={row['score']} pred=small-vehicle "
            f"raw={row['raw_box_type']} angle={row['raw_box_angle']} area={row['raw_box_area']} "
            f"valid={row['valid_mask_ratio_inside_box']} pad={row['padding_overlap_ratio']} "
            f"gt={row['best_gt_class']} iou={row['best_gt_iou']} cat={cat}"
        )
        full_tile = sample_dir / "full_tile_raw_polygon_overlay.png"
        zoom = sample_dir / "zoom_fixed_256_overlay.png"
        gt_context = sample_dir / "gt_context_overlay.png"
        ok1 = draw_overlay(image_path, full_tile, pred_points, None, title)
        ok2 = draw_overlay(image_path, zoom, pred_points, None, title, zoom_center=center, zoom_size=256)
        ok3 = draw_overlay(image_path, gt_context, pred_points, gt_rows, title, zoom_center=center, zoom_size=512)
        if not (ok1 and ok2 and ok3):
            continue
        meta = {
            "crop_id": crop_id,
            "audit_category": cat,
            "model_name": model,
            "tile_id": tile,
            "angle": angle,
            "score": row["score"],
            "pred_class": row["pred_class"],
            "raw_box_type": row["raw_box_type"],
            "raw_box_angle": row["raw_box_angle"],
            "raw_box_area": row["raw_box_area"],
            "valid_mask_ratio_inside_box": row["valid_mask_ratio_inside_box"],
            "padding_overlap_ratio": row["padding_overlap_ratio"],
            "best_gt_class": row["best_gt_class"],
            "best_gt_iou": row["best_gt_iou"],
            "source_crop_id": row["crop_id"],
            "source_prediction_path": row["source_prediction_path"],
        }
        meta_path = sample_dir / "metadata.json"
        write_json(meta_path, meta)
        valid_for_human = cat in {"valid_unmatched_sv", "strict_object_flip", "true_sv_positive_control"}
        out_rows.append({
            "crop_id": crop_id,
            "audit_category": cat,
            "image_path_full_tile": str(full_tile),
            "image_path_zoom": str(zoom),
            "image_path_gt_context": str(gt_context),
            "metadata_json": str(meta_path),
            "model_name": model,
            "tile_id": tile,
            "angle": angle,
            "score": row["score"],
            "pred_class": row["pred_class"],
            "raw_box_type": row["raw_box_type"],
            "raw_box_angle": row["raw_box_angle"],
            "raw_box_area": row["raw_box_area"],
            "valid_mask_ratio_inside_box": row["valid_mask_ratio_inside_box"],
            "padding_overlap_ratio": row["padding_overlap_ratio"],
            "best_gt_class": row["best_gt_class"],
            "best_gt_iou": row["best_gt_iou"],
            "auto_label": cat,
            "valid_for_human_audit": "true" if valid_for_human else "false",
            "human_label": "",
            "human_confidence": "",
            "human_notes": "",
        })
        counts[cat] += 1
        if all(counts[k] >= v for k, v in TARGETS.items()):
            break
    template = audit_dir / "human_sv_crop_audit_template_VERIFIED_FAST.csv"
    write_csv(template, out_rows, HEADERS)
    status = {
        "status": "READY_FOR_HUMAN_LABELS" if any(r["audit_category"] == "valid_unmatched_sv" for r in out_rows) else "PARTIAL_NO_VALID_UNMATCHED",
        "pack_root": str(pack_root),
        "template": str(template),
        "counts": dict(counts),
        "targets": TARGETS,
        "total_rows": len(out_rows),
    }
    write_json(audit_dir / "verified_fast_crop_pack_status.json", status)
    write_md(
        audit_dir / "verified_fast_crop_pack_status.md",
        "# Verified Fast Crop Pack Status\n\n"
        f"- status: `{status['status']}`\n"
        f"- pack_root: `{pack_root}`\n"
        f"- template: `{template}`\n\n"
        "| category | count | target |\n| --- | --- | --- |\n"
        + "\n".join(f"| {k} | {counts.get(k, 0)} | {v} |" for k, v in TARGETS.items()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
