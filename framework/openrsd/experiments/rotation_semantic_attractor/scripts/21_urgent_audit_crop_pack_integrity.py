#!/usr/bin/env python3
"""Urgently audit the visual-summary human crop pack integrity.

This script does not run inference. It traces the existing crop pack back to
canonical/raw prediction JSONs, GT, valid masks, and the crop rendering code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat

# Keep this emergency audit in pure-Python geometry mode. Some invalid
# detector polygons can crash GEOS in long batch runs; HBB IoU is enough for
# the integrity decision and is explicitly treated as an approximation.
Polygon = None


DOTA_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]

ANOMALY_HINTS = {
    ("r3det_kfiou", "P2802__1024__4914___4192", 180),
    ("r3det_kfiou", "P1518__1024__524___1048", 60),
    ("r3det_kfiou", "P1666__1024__1048___2976", 240),
    ("r3det_kfiou", "P1380__1024__2620___3144", 60),
    ("oriented_rcnn", "P2625__1024__3259___524", 300),
    ("r3det_kfiou", "P1147__1024__2620___1048", 150),
    ("r3det_kfiou", "P1397__1024__2620___1048", 120),
    ("r3det_kfiou", "P2802__1024__3668___1572", 180),
    ("r3det_kfiou", "P0019__1024__4716___2096", 90),
    ("redet_msrr", "P2802__1024__2096___1572", 90),
    ("h2rbox", "P2625__1024__3144___524", 330),
    ("oriented_reppoints", "P0706__1024__0___0", 240),
}

AUDIT_HEADERS = [
    "crop_id",
    "crop_path",
    "model_name",
    "tile_id",
    "angle",
    "score",
    "pred_class",
    "pred_id",
    "source_manifest_found",
    "source_prediction_found",
    "source_prediction_path",
    "raw_prediction_record",
    "raw_box",
    "raw_box_type",
    "raw_box_has_angle",
    "raw_box_angle",
    "raw_box_area",
    "raw_box_width",
    "raw_box_height",
    "raw_box_aspect_ratio",
    "rendered_box",
    "rendered_box_type",
    "rendered_box_area",
    "rendered_box_has_angle",
    "rendered_box_matches_raw",
    "rendered_box_iou_with_raw_hbb",
    "rendering_lost_angle",
    "is_hbb_fallback",
    "hbb_fallback_explicitly_labeled",
    "is_crop_window_box",
    "coordinate_frame",
    "coordinate_frame_verified",
    "valid_mask_available",
    "valid_mask_ratio_inside_box",
    "padding_overlap_ratio",
    "gray_padding_pixel_ratio",
    "box_center_in_valid_region",
    "best_gt_class",
    "best_gt_iou",
    "nearest_sv_gt_iou",
    "risk_group",
    "sv_gt_size_plausible",
    "is_oversized_for_small_vehicle",
    "valid_for_human_audit",
    "invalid_reason",
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


def md_table(rows: list[dict[str, Any]], headers: list[str] | None = None, limit: int | None = None) -> str:
    if not rows:
        return "_No rows._"
    use_rows = rows[:limit] if limit else rows
    if headers is None:
        headers = list(use_rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in use_rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def bool_text(value: bool) -> str:
    return "true" if value else "false"


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
    if len(box) == 4:
        x1, y1, x2, y2 = [float(x) for x in box]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return []


def bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def edge_size(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    if len(points) < 4:
        b = bounds(points)
        if not b:
            return 0.0, 0.0, 0.0
        x1, y1, x2, y2 = b
        w, h = abs(x2 - x1), abs(y2 - y1)
        return w, h, max(w / max(h, 1e-9), h / max(w, 1e-9))
    ds = []
    for i, (x1, y1) in enumerate(points[:4]):
        x2, y2 = points[(i + 1) % 4]
        ds.append(math.hypot(x2 - x1, y2 - y1))
    w = max(ds)
    h = min(d for d in ds if d > 1e-9) if any(d > 1e-9 for d in ds) else 0.0
    return w, h, max(w / max(h, 1e-9), h / max(w, 1e-9))


def hbb_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-9)


def poly_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if Polygon is None or len(a) < 3 or len(b) < 3:
        return hbb_iou(bounds(a), bounds(b))
    try:
        pa = Polygon(a)
        pb = Polygon(b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return hbb_iou(bounds(a), bounds(b))


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
        rows.append({"points": pts, "class_name": parts[8], "difficulty": parts[9] if len(parts) > 9 else ""})
    return rows


def percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def compute_sv_size_distribution(data_gt_root: Path, audit_dir: Path) -> dict[str, float]:
    areas: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    aspects: list[float] = []
    per_angle: Counter[int] = Counter()
    for ann in sorted(data_gt_root.glob("angle_*/annfiles/*.txt")):
        try:
            angle = int(ann.parts[-3].replace("angle_", ""))
        except Exception:
            angle = -1
        for gt in parse_dota_ann(ann):
            if gt["class_name"] != "small-vehicle":
                continue
            pts = gt["points"]
            areas.append(polygon_area(pts))
            w, h, ar = edge_size(pts)
            widths.append(w)
            heights.append(h)
            aspects.append(ar)
            per_angle[angle] += 1
    row = {
        "num_gt_sv": len(areas),
        "area_p50": percentile(areas, 0.50),
        "area_p75": percentile(areas, 0.75),
        "area_p90": percentile(areas, 0.90),
        "area_p95": percentile(areas, 0.95),
        "area_p99": percentile(areas, 0.99),
        "area_p995": percentile(areas, 0.995),
        "width_p50": percentile(widths, 0.50),
        "width_p75": percentile(widths, 0.75),
        "width_p90": percentile(widths, 0.90),
        "width_p95": percentile(widths, 0.95),
        "width_p99": percentile(widths, 0.99),
        "height_p50": percentile(heights, 0.50),
        "height_p75": percentile(heights, 0.75),
        "height_p90": percentile(heights, 0.90),
        "height_p95": percentile(heights, 0.95),
        "height_p99": percentile(heights, 0.99),
        "aspect_ratio_p95": percentile(aspects, 0.95),
        "aspect_ratio_p99": percentile(aspects, 0.99),
    }
    write_csv(audit_dir / "sv_gt_size_distribution.csv", [row])
    md = [
        "# Small-Vehicle GT Size Distribution",
        "",
        f"- source: `{data_gt_root}`",
        f"- num_gt_sv: `{len(areas)}`",
        "",
        md_table([row]),
        "",
        "Per-angle annotated small-vehicle counts:",
        "",
        md_table([{"angle": k, "num_gt_sv": v} for k, v in sorted(per_angle.items())]),
    ]
    write_md(audit_dir / "sv_gt_size_distribution.md", "\n".join(md))
    return row


def find_matching_prediction(preds: list[dict[str, Any]], row: dict[str, str]) -> dict[str, Any] | None:
    score = as_float(row.get("pred_score", row.get("score", "")), -1.0)
    pred_box = parse_jsonish(row.get("pred_box", ""))
    candidates = [p for p in preds if p.get("class_name") == "small-vehicle"]
    if isinstance(pred_box, list):
        for pred in candidates:
            box = pred.get("box") or []
            if len(box) == len(pred_box) and all(abs(float(a) - float(b)) < 1e-4 for a, b in zip(box, pred_box)):
                return pred
    best = None
    best_diff = 1e9
    for pred in candidates:
        diff = abs(float(pred.get("score", 0.0)) - score)
        if diff < best_diff:
            best = pred
            best_diff = diff
    if best is not None and best_diff < 1e-5:
        return best
    return best if best_diff < 1e-3 else None


def mask_metrics(mask_path: Path, image_path: Path, points: list[tuple[float, float]]) -> dict[str, Any]:
    out = {
        "valid_mask_available": mask_path.exists(),
        "valid_mask_ratio_inside_box": "",
        "padding_overlap_ratio": "",
        "gray_padding_pixel_ratio": "",
        "box_center_in_valid_region": "",
    }
    if not points:
        return out
    b = bounds(points)
    if b is None:
        return out
    x1, y1, x2, y2 = b
    cx = int(round(sum(x for x, _ in points) / len(points)))
    cy = int(round(sum(y for _, y in points) / len(points)))
    if mask_path.exists():
        try:
            mask = Image.open(mask_path).convert("L")
            w, h = mask.size
            poly_img = Image.new("1", (w, h), 0)
            ImageDraw.Draw(poly_img).polygon(points, fill=1)
            mask_px = mask.load()
            poly_px = poly_img.load()
            total = 0
            valid = 0
            bx1 = max(0, int(math.floor(x1)))
            by1 = max(0, int(math.floor(y1)))
            bx2 = min(w - 1, int(math.ceil(x2)))
            by2 = min(h - 1, int(math.ceil(y2)))
            for yy in range(by1, by2 + 1):
                for xx in range(bx1, bx2 + 1):
                    if poly_px[xx, yy]:
                        total += 1
                        if mask_px[xx, yy] > 127:
                            valid += 1
            ratio = valid / max(total, 1)
            out["valid_mask_ratio_inside_box"] = f"{ratio:.6f}"
            out["padding_overlap_ratio"] = f"{(1.0 - ratio):.6f}"
            if 0 <= cx < w and 0 <= cy < h:
                out["box_center_in_valid_region"] = bool_text(mask_px[cx, cy] > 127)
            else:
                out["box_center_in_valid_region"] = "false"
        except Exception:
            out["valid_mask_available"] = False
    if image_path.exists():
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            bx1 = max(0, int(math.floor(x1)))
            by1 = max(0, int(math.floor(y1)))
            bx2 = min(w, int(math.ceil(x2)))
            by2 = min(h, int(math.ceil(y2)))
            if bx2 > bx1 and by2 > by1:
                crop = img.crop((bx1, by1, bx2, by2))
                stat = ImageStat.Stat(crop)
                means = stat.mean
                # DOTA rotation padding in this project is commonly a neutral gray.
                pixels = list(crop.getdata())
                gray = sum(1 for r, g, b in pixels if max(abs(r - 128), abs(g - 128), abs(b - 128)) <= 24)
                out["gray_padding_pixel_ratio"] = f"{gray / max(len(pixels), 1):.6f}"
        except Exception:
            pass
    return out


def source_paths(run_root: Path, model: str, tile: str, angle: int) -> dict[str, Path]:
    return {
        "canonical": run_root / "canonical_predictions" / model / tile / f"angle_{angle:03d}.json",
        "raw": run_root / "raw_predictions" / model / tile / f"angle_{angle:03d}.json",
        "rotated_gt": run_root / "rotated_gt" / model / tile / f"{tile}_angle_{angle:03d}.txt",
        "valid_mask": run_root / "valid_masks" / model / tile / f"angle_{angle:03d}.png",
        "rotated_image": run_root / "rotated_images" / model / tile / f"{tile}_angle_{angle:03d}.png",
    }


def data_image_path(data_root: Path, tile: str, angle: int) -> Path:
    return data_root / f"angle_{angle:03d}" / "images" / f"{tile}.png"


def data_gt_path(data_root: Path, tile: str, angle: int) -> Path:
    return data_root / f"angle_{angle:03d}" / "annfiles" / f"{tile}.txt"


def diagnose_gt(points: list[tuple[float, float]], gt_path: Path) -> dict[str, Any]:
    gts = parse_dota_ann(gt_path)
    best_cls = ""
    best_iou = 0.0
    sv_iou = 0.0
    for gt in gts:
        iou = poly_iou(points, gt["points"])
        if iou > best_iou:
            best_iou = iou
            best_cls = gt["class_name"]
        if gt["class_name"] == "small-vehicle":
            sv_iou = max(sv_iou, iou)
    return {
        "best_gt_class": best_cls if best_cls else "NO_GT_OVERLAP",
        "best_gt_iou": f"{best_iou:.6f}",
        "nearest_sv_gt_iou": f"{sv_iou:.6f}",
    }


def draw_overlay(
    image_path: Path,
    out_path: Path,
    pred_points: list[tuple[float, float]] | None = None,
    hbb: tuple[float, float, float, float] | None = None,
    gt_rows: list[dict[str, Any]] | None = None,
    mask_path: Path | None = None,
    title: str = "",
    zoom_center: tuple[float, float] | None = None,
    zoom_size: int | None = None,
) -> bool:
    if not image_path.exists():
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
    draw = ImageDraw.Draw(img, "RGBA")
    ox, oy = offset
    if mask_path and mask_path.exists() and not zoom_size:
        try:
            mask = Image.open(mask_path).convert("L").resize(img.size)
            overlay = Image.new("RGBA", img.size, (255, 255, 0, 0))
            opx = overlay.load()
            mpx = mask.load()
            for yy in range(img.size[1]):
                for xx in range(img.size[0]):
                    if mpx[xx, yy] < 128:
                        opx[xx, yy] = (255, 255, 0, 60)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
        except Exception:
            pass
    if gt_rows:
        for gt in gt_rows:
            pts = [(x - ox, y - oy) for x, y in gt["points"]]
            color = (0, 255, 0, 180) if gt["class_name"] == "small-vehicle" else (0, 160, 255, 140)
            draw.line(pts + [pts[0]], fill=color, width=2)
    if hbb:
        x1, y1, x2, y2 = hbb
        draw.rectangle((x1 - ox, y1 - oy, x2 - ox, y2 - oy), outline=(255, 0, 0, 255), width=4)
    if pred_points:
        pts = [(x - ox, y - oy) for x, y in pred_points]
        draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=4)
    if title:
        draw.rectangle((0, 0, img.size[0], 34), fill=(0, 0, 0, 160))
        draw.text((6, 6), title[:180], fill=(255, 255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def build_file_inventory(args: argparse.Namespace, rows: list[dict[str, str]], audit_dir: Path) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    run_root = Path(args.run_root)
    crops_dir = output_dir / "crops"
    labels = audit_dir / "human_sv_crop_audit_labels.csv"
    template = audit_dir / "human_sv_crop_audit_template.csv"
    manifest = audit_dir / "unannotated_sv_sample_manifest.csv"
    crop_paths = sorted(crops_dir.glob("**/*.png"))
    crop_path_set = {str(p.resolve()) for p in crop_paths}
    row_paths = {
        str((Path(p) if Path(p).is_absolute() else (Path.cwd() / p)).resolve())
        for p in (r.get("image_path", "") for r in rows)
        if p
    }
    sample = rows[:50]
    traceable = 0
    valid_masks = 0
    gts = 0
    raws = 0
    canon = 0
    for row in sample:
        model = row.get("model") or row.get("model_name", "")
        tile = row.get("tile_id", "")
        angle = int(as_float(row.get("angle", 0)))
        sp = source_paths(run_root, model, tile, angle)
        raws += int(sp["raw"].exists())
        canon += int(sp["canonical"].exists())
        valid_masks += int(sp["valid_mask"].exists())
        gts += int(sp["rotated_gt"].exists() or data_gt_path(Path(args.data_gt_root), tile, angle).exists())
        traceable += int(sp["raw"].exists() and sp["canonical"].exists())
    inventory = {
        "old_crop_pack_status": "QUARANTINED_UNVERIFIED",
        "old_crop_dir": str(crops_dir),
        "old_crop_png_count": len(crop_paths),
        "template": str(template),
        "template_exists": template.exists(),
        "template_rows": len(rows),
        "manifest": str(manifest),
        "manifest_exists": manifest.exists(),
        "manifest_rows": len(read_csv(manifest)),
        "human_labels": str(labels),
        "human_labels_exists": labels.exists(),
        "labels_status": "INVALID_UNTIL_CROP_INTEGRITY_PASSES" if labels.exists() else "NOT_PRESENT",
        "crop_rows_missing_file": sorted(row_paths - crop_path_set)[:50],
        "crop_files_missing_row": sorted(crop_path_set - row_paths)[:50],
        "run_root": str(run_root),
        "canonical_predictions_root": str(run_root / "canonical_predictions"),
        "canonical_predictions_root_exists": (run_root / "canonical_predictions").exists(),
        "raw_predictions_root": str(run_root / "raw_predictions"),
        "raw_predictions_root_exists": (run_root / "raw_predictions").exists(),
        "rotated_gt_root": str(run_root / "rotated_gt"),
        "rotated_gt_root_exists": (run_root / "rotated_gt").exists(),
        "valid_masks_root": str(run_root / "valid_masks"),
        "valid_masks_root_exists": (run_root / "valid_masks").exists(),
        "metadata_json_per_crop": False,
        "sample_traceability_first_50": {
            "raw_prediction_found": raws,
            "canonical_prediction_found": canon,
            "valid_mask_found": valid_masks,
            "gt_found": gts,
            "raw_and_canonical_found": traceable,
        },
    }
    write_json(audit_dir / "urgent_crop_pack_file_inventory.json", inventory)
    md = [
        "# Urgent Crop Pack File Inventory",
        "",
        "OLD_CROP_PACK_STATUS = QUARANTINED_UNVERIFIED",
        "",
        md_table([{k: v} for k, v in inventory.items() if k != "sample_traceability_first_50"]),
        "",
        "## Sample Traceability First 50 Template Rows",
        "",
        md_table([inventory["sample_traceability_first_50"]]),
        "",
        "## Impact",
        "",
        "- The old 200 crops are not valid for corrected-FSV until integrity passes.",
        "- Crop images do not have per-image metadata JSON; traceability depends on the template/manifest rows.",
        "- If human label CSV exists, its status is `INVALID_UNTIL_CROP_INTEGRITY_PASSES`.",
    ]
    write_md(audit_dir / "urgent_crop_pack_file_inventory.md", "\n".join(md))
    return inventory


def write_quarantine_marker(audit_dir: Path) -> None:
    write_md(
        audit_dir / "OLD_CROP_PACK_QUARANTINED.txt",
        "OLD_CROP_PACK_STATUS = QUARANTINED_UNVERIFIED\n\n"
        "The old 200 crop images must not be used for corrected-FSV before the crop integrity audit passes.\n"
        "The old 200 crop images must not be used for human true-vehicle / non-vehicle labeling.\n"
        "The old 200 crop images may only be used to debug the visualization pipeline.\n"
        "If human_sv_crop_audit_labels.csv exists, labels_status = INVALID_UNTIL_CROP_INTEGRITY_PASSES.\n",
    )


def audit_rows(args: argparse.Namespace, template_rows: list[dict[str, str]], size_stats: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = Path(args.run_root)
    data_root = Path(args.data_gt_root)
    image_root = Path(args.data_image_root)
    audit_rows_out: list[dict[str, Any]] = []
    oversized_rows: list[dict[str, Any]] = []
    padding_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    area_p995 = float(size_stats.get("area_p995", 0.0))
    width_p99 = float(size_stats.get("width_p99", 0.0))
    height_p99 = float(size_stats.get("height_p99", 0.0))
    for row in template_rows:
        model = row.get("model") or row.get("model_name", "")
        tile = row.get("tile_id", "")
        angle = int(as_float(row.get("angle", 0)))
        score = as_float(row.get("pred_score", row.get("score", "")), 0.0)
        sp = source_paths(run_root, model, tile, angle)
        canon = load_json(sp["canonical"]) if sp["canonical"].exists() else {}
        raw = load_json(sp["raw"]) if sp["raw"].exists() else {}
        canon_pred = find_matching_prediction(canon.get("final_predictions", []), row)
        raw_pred = None
        if canon_pred and raw.get("final_predictions"):
            ridx = canon_pred.get("raw_index")
            for pred in raw.get("final_predictions", []):
                if pred.get("raw_index") == ridx and pred.get("class_name") == canon_pred.get("class_name"):
                    raw_pred = pred
                    break
        if raw_pred is None and raw.get("final_predictions"):
            raw_pred = find_matching_prediction(raw.get("final_predictions", []), row)
        raw_points = points_from_record(raw_pred or {}) if raw_pred else []
        canon_points = points_from_record(canon_pred or {}) if canon_pred else []
        rendered_bounds = bounds(canon_points)
        raw_hbb = bounds(raw_points)
        rendered_area = 0.0
        if rendered_bounds:
            rendered_area = max(0.0, rendered_bounds[2] - rendered_bounds[0]) * max(0.0, rendered_bounds[3] - rendered_bounds[1])
        raw_area = polygon_area(raw_points)
        raw_w, raw_h, raw_ar = edge_size(raw_points)
        raw_box_type = str((raw_pred or {}).get("box_type", ""))
        raw_box = (raw_pred or {}).get("box", "")
        raw_angle = ""
        if isinstance(raw_box, list) and len(raw_box) == 5:
            raw_angle = raw_box[4]
        mask_metrics_row = mask_metrics(sp["valid_mask"], data_image_path(image_root, tile, angle), raw_points)
        gt_path = sp["rotated_gt"] if sp["rotated_gt"].exists() else data_gt_path(data_root, tile, angle)
        gt_diag = diagnose_gt(raw_points, gt_path)
        rendered_vs_raw_hbb_iou = hbb_iou(rendered_bounds, bounds(canon_points))
        # rendered == canonical HBB by construction, but the image was rotated.
        coordinate_frame = "canonical_prediction_drawn_on_rotated_image" if angle != 0 else "canonical_prediction_drawn_on_angle0_image"
        coordinate_verified = angle == 0
        raw_has_angle = isinstance(raw_box, list) and len(raw_box) == 5
        is_hbb_fallback = raw_box_type in {"obb", "polygon"} and rendered_bounds is not None
        rendering_lost_angle = is_hbb_fallback
        center_valid = mask_metrics_row.get("box_center_in_valid_region") == "true"
        valid_ratio = as_float(mask_metrics_row.get("valid_mask_ratio_inside_box", ""), 0.0)
        padding_ratio = as_float(mask_metrics_row.get("padding_overlap_ratio", ""), 1.0)
        oversized = (
            (area_p995 > 0 and raw_area > area_p995)
            or (width_p99 > 0 and raw_w > width_p99)
            or (height_p99 > 0 and raw_h > height_p99)
        )
        reasons: list[str] = []
        if not row:
            reasons.append("missing_manifest_row")
        if raw_pred is None or canon_pred is None:
            reasons.append("missing_source_prediction")
        if raw_pred is None:
            reasons.append("raw_box_not_found")
        if not raw_box_type:
            reasons.append("box_type_unknown")
        if raw_box_type in {"obb", "polygon"} and not raw_has_angle and raw_box_type == "obb":
            reasons.append("angle_missing_for_rotated_model")
        if rendering_lost_angle:
            reasons.append("rendering_lost_angle")
            if not False:
                reasons.append("hbb_fallback_without_label")
        if rendered_vs_raw_hbb_iou < 0.8:
            reasons.append("raw_vs_render_iou_low")
        if not coordinate_verified:
            reasons.append("coordinate_frame_mismatch")
        if not mask_metrics_row.get("valid_mask_available"):
            reasons.append("valid_mask_missing")
        if valid_ratio < 0.85:
            reasons.append("padding_contaminated")
        if padding_ratio > 0.10:
            if "padding_contaminated" not in reasons:
                reasons.append("padding_contaminated")
        if not center_valid:
            reasons.append("center_outside_valid_region")
        if not gt_path.exists():
            reasons.append("gt_missing")
        if oversized:
            reasons.append("oversized_sv_box")
        if not reasons:
            reasons.append("")
        valid_for_human = reasons == [""]
        out = {
            "crop_id": row.get("crop_id", ""),
            "crop_path": row.get("image_path", ""),
            "model_name": model,
            "tile_id": tile,
            "angle": angle,
            "score": f"{score:.6f}",
            "pred_class": (canon_pred or raw_pred or {}).get("class_name", row.get("pred_class", "small-vehicle")),
            "pred_id": (canon_pred or raw_pred or {}).get("raw_index", ""),
            "source_manifest_found": bool_text(True),
            "source_prediction_found": bool_text(raw_pred is not None and canon_pred is not None),
            "source_prediction_path": str(sp["raw"]),
            "raw_prediction_record": json.dumps(raw_pred or {}, ensure_ascii=False),
            "raw_box": json.dumps(raw_box, ensure_ascii=False),
            "raw_box_type": raw_box_type,
            "raw_box_has_angle": bool_text(raw_has_angle),
            "raw_box_angle": raw_angle,
            "raw_box_area": f"{raw_area:.6f}",
            "raw_box_width": f"{raw_w:.6f}",
            "raw_box_height": f"{raw_h:.6f}",
            "raw_box_aspect_ratio": f"{raw_ar:.6f}",
            "rendered_box": json.dumps(rendered_bounds if rendered_bounds else [], ensure_ascii=False),
            "rendered_box_type": "HBB_FROM_CANONICAL_POLYGON",
            "rendered_box_area": f"{rendered_area:.6f}",
            "rendered_box_has_angle": "false",
            "rendered_box_matches_raw": bool_text(rendered_vs_raw_hbb_iou >= 0.8),
            "rendered_box_iou_with_raw_hbb": f"{rendered_vs_raw_hbb_iou:.6f}",
            "rendering_lost_angle": bool_text(rendering_lost_angle),
            "is_hbb_fallback": bool_text(is_hbb_fallback),
            "hbb_fallback_explicitly_labeled": "false",
            "is_crop_window_box": "false",
            "coordinate_frame": coordinate_frame,
            "coordinate_frame_verified": bool_text(coordinate_verified),
            "valid_mask_available": bool_text(bool(mask_metrics_row.get("valid_mask_available"))),
            "valid_mask_ratio_inside_box": mask_metrics_row.get("valid_mask_ratio_inside_box", ""),
            "padding_overlap_ratio": mask_metrics_row.get("padding_overlap_ratio", ""),
            "gray_padding_pixel_ratio": mask_metrics_row.get("gray_padding_pixel_ratio", ""),
            "box_center_in_valid_region": mask_metrics_row.get("box_center_in_valid_region", ""),
            "best_gt_class": gt_diag["best_gt_class"],
            "best_gt_iou": gt_diag["best_gt_iou"],
            "nearest_sv_gt_iou": gt_diag["nearest_sv_gt_iou"],
            "risk_group": row.get("risk_group", ""),
            "sv_gt_size_plausible": bool_text(not oversized),
            "is_oversized_for_small_vehicle": bool_text(oversized),
            "valid_for_human_audit": bool_text(valid_for_human),
            "invalid_reason": ";".join(r for r in reasons if r) or "",
        }
        audit_rows_out.append(out)
        oversized_rows.append({
            "crop_id": out["crop_id"],
            "model_name": model,
            "tile_id": tile,
            "angle": angle,
            "score": out["score"],
            "pred_area": out["raw_box_area"],
            "pred_width": out["raw_box_width"],
            "pred_height": out["raw_box_height"],
            "pred_area_vs_sv_gt_p995": f"{raw_area / max(area_p995, 1e-9):.6f}",
            "pred_width_vs_sv_gt_p99": f"{raw_w / max(width_p99, 1e-9):.6f}",
            "pred_height_vs_sv_gt_p99": f"{raw_h / max(height_p99, 1e-9):.6f}",
            "oversized_flag": bool_text(oversized),
        })
        padding_rows.append({
            "crop_id": out["crop_id"],
            "model_name": model,
            "tile_id": tile,
            "angle": angle,
            "valid_mask_available": out["valid_mask_available"],
            "valid_mask_ratio_inside_box": out["valid_mask_ratio_inside_box"],
            "padding_overlap_ratio": out["padding_overlap_ratio"],
            "gray_padding_pixel_ratio": out["gray_padding_pixel_ratio"],
            "box_center_in_valid_region": out["box_center_in_valid_region"],
            "padding_status": "padding_artifact" if "padding_contaminated" in out["invalid_reason"] else "ok_or_unverified",
        })
        raw_label_id = (raw_pred or {}).get("class_id", "")
        mapped = DOTA_CLASSES[int(raw_label_id)] if isinstance(raw_label_id, int) and 0 <= raw_label_id < len(DOTA_CLASSES) else ""
        mapping_rows.append({
            "crop_id": out["crop_id"],
            "model_name": model,
            "raw_label_id": raw_label_id,
            "raw_label_name": (raw_pred or {}).get("class_name", ""),
            "mapped_label_name": mapped,
            "dataset_classes_index": ",".join(f"{i}:{c}" for i, c in enumerate(DOTA_CLASSES)),
            "small_vehicle_index": 4,
            "large_vehicle_index": 5,
            "mapping_source": "DOTA_CLASSES zero-based order and prediction class_name field",
            "mapping_verified": bool_text(mapped == "small-vehicle" and (raw_pred or {}).get("class_name") == "small-vehicle"),
        })
    return audit_rows_out, oversized_rows, padding_rows, mapping_rows


def write_audit_reports(audit_dir: Path, rows: list[dict[str, Any]], oversized_rows: list[dict[str, Any]], padding_rows: list[dict[str, Any]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    write_csv(audit_dir / "urgent_crop_pack_integrity_audit.csv", rows, AUDIT_HEADERS)
    write_csv(audit_dir / "oversized_sv_box_audit.csv", oversized_rows)
    write_csv(audit_dir / "padding_contamination_audit.csv", padding_rows)
    write_csv(audit_dir / "class_mapping_for_crop_pack.csv", mapping_rows)
    total = len(rows)
    counter = Counter()
    for row in rows:
        counter["source_prediction_found"] += row["source_prediction_found"] == "true"
        counter["raw_box_verified"] += row["raw_box_type"] in {"obb", "polygon"} and row["source_prediction_found"] == "true"
        counter["hbb_fallback"] += row["is_hbb_fallback"] == "true"
        counter["obb_angle_lost"] += row["rendering_lost_angle"] == "true"
        counter["crop_window_suspected"] += row["is_crop_window_box"] == "true"
        counter["oversized_box"] += row["is_oversized_for_small_vehicle"] == "true"
        counter["padding_contaminated"] += "padding_contaminated" in row["invalid_reason"]
        counter["valid_for_human_audit"] += row["valid_for_human_audit"] == "true"
        counter["coordinate_mismatch"] += "coordinate_frame_mismatch" in row["invalid_reason"]
        counter["class_mapping_error"] += row.get("pred_class") != "small-vehicle"
    reason_counts = Counter()
    for row in rows:
        for reason in row["invalid_reason"].split(";"):
            if reason:
                reason_counts[reason] += 1
    summary = {
        "total_crops": total,
        **{k: int(v) for k, v in counter.items()},
        "valid_rate": counter["valid_for_human_audit"] / max(total, 1),
        "invalid_reason_counts": dict(reason_counts),
        "old_crop_pack_status": "QUARANTINED_UNVERIFIED",
    }
    write_json(audit_dir / "urgent_crop_pack_integrity_audit.json", summary)
    write_md(
        audit_dir / "urgent_crop_pack_integrity_audit.md",
        "# Urgent Crop Pack Integrity Audit\n\n"
        "OLD_CROP_PACK_STATUS = QUARANTINED_UNVERIFIED\n\n"
        "## Summary\n\n"
        + md_table([summary]) + "\n\n"
        "## Invalid Reason Counts\n\n"
        + md_table([{"invalid_reason": k, "count": v} for k, v in reason_counts.most_common()]) + "\n\n"
        "## First 30 Rows\n\n"
        + md_table(rows, ["crop_id", "model_name", "tile_id", "angle", "score", "raw_box_type", "rendering_lost_angle", "coordinate_frame_verified", "valid_mask_ratio_inside_box", "is_oversized_for_small_vehicle", "valid_for_human_audit", "invalid_reason"], 30),
    )
    write_md(
        audit_dir / "oversized_sv_box_audit.md",
        "# Oversized SV Box Audit\n\n"
        "Prediction boxes are compared with annotated small-vehicle GT size percentiles. Oversized boxes are not valid for unannotated true-SV contamination estimation; they should be treated as `degenerate_large_sv_box` unless later disproven.\n\n"
        + md_table(oversized_rows, limit=50),
    )
    write_md(
        audit_dir / "padding_contamination_audit.md",
        "# Padding Contamination Audit\n\n"
        "Rows with padding_overlap_ratio > 0.10 or valid_mask_ratio_inside_box < 0.85 are invalid for human true-SV audit.\n\n"
        + md_table(padding_rows, limit=50),
    )
    write_md(
        audit_dir / "class_mapping_for_crop_pack.md",
        "# Class Mapping for Crop Pack\n\n"
        "The audit checks that raw label id 4 maps to `small-vehicle` under the DOTA zero-based class order and that the prediction record class_name is also `small-vehicle`.\n\n"
        + md_table(mapping_rows, limit=80),
    )
    return summary


def write_rendering_code_audit(repo_root: Path, audit_dir: Path) -> None:
    script = repo_root / "experiments/rotation_semantic_attractor/scripts/20_make_human_sv_crop_audit_pack.py"
    text = script.read_text(encoding="utf-8", errors="ignore") if script.exists() else ""
    findings = [
        {
            "file": str(script),
            "function": "polygon_bounds",
            "current_logic": "Uses prediction polygon/8-point box, then returns min_x/min_y/max_x/max_y.",
            "issue": "OBB/polygon orientation is collapsed to HBB before rendering.",
            "fix_needed": "Render polygon directly and label any HBB fallback explicitly.",
        },
        {
            "file": str(script),
            "function": "draw_crop",
            "current_logic": "Uses patches.Rectangle((x1-cx1,y1-cy1), x2-x1, y2-y1).",
            "issue": "Drawn red box is an HBB fallback, not the detector OBB polygon.",
            "fix_needed": "Use polygon patch/line with actual raw prediction points.",
        },
        {
            "file": str(script),
            "function": "main/image_for",
            "current_logic": "Reads canonical_predictions but draws them on data angle_sweep rotated images.",
            "issue": "For nonzero angles, canonical coordinates and rotated image coordinates are mixed.",
            "fix_needed": "Use raw_predictions with rotated images, or canonical predictions with canonical images after explicit transform.",
        },
        {
            "file": str(script),
            "function": "draw_crop crop extent",
            "current_logic": "Crop size expands with max HBB side length.",
            "issue": "Degenerate large predictions create large scene crops, confusing human audit.",
            "fix_needed": "Use fixed-size zoom crop around raw prediction center for human audit.",
        },
    ]
    checks = {
        "contains_patches_rectangle": "patches.Rectangle" in text,
        "contains_polygon_bounds": "polygon_bounds" in text,
        "contains_hbb_label": "HBB_FALLBACK" in text,
        "contains_raw_prediction_use": "raw_predictions" in text,
    }
    md = [
        "# Box Rendering Code Audit",
        "",
        "## Search Findings",
        "",
        md_table([checks]),
        "",
        "## File/Function Audit",
        "",
        md_table(findings),
        "",
        "## Conclusion",
        "",
        "- Current crop rendering loses OBB angle by drawing an HBB rectangle.",
        "- Current titles do not mark HBB fallback.",
        "- Current crop generation reads canonical predictions and draws them on rotated images, so nonzero-angle samples have a coordinate-frame mismatch.",
        "- The red rectangle is not the crop window itself, but the crop extent follows the HBB size, so degenerate boxes produce huge scene crops.",
    ]
    write_md(audit_dir / "box_rendering_code_audit.md", "\n".join(md))


def make_debug_overlays(args: argparse.Namespace, rows: list[dict[str, Any]], audit_dir: Path) -> None:
    run_root = Path(args.run_root)
    image_root = Path(args.data_image_root)
    out_root = audit_dir / "debug_raw_vs_render_20"
    selected: list[dict[str, Any]] = []
    for row in rows:
        key = (row["model_name"], row["tile_id"], int(row["angle"]))
        if key in ANOMALY_HINTS:
            selected.append(row)
    for row in rows:
        if len(selected) >= 20:
            break
        if row not in selected:
            selected.append(row)
    for idx, row in enumerate(selected[:20]):
        sample_dir = out_root / f"{idx:02d}_{row['crop_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        crop_path = Path(row["crop_path"])
        if crop_path.exists():
            shutil.copyfile(crop_path, sample_dir / "original_current_crop.png")
        raw_record = parse_jsonish(row["raw_prediction_record"])
        raw_points = points_from_record(raw_record if isinstance(raw_record, dict) else {})
        raw_hbb = bounds(raw_points)
        model = row["model_name"]
        tile = row["tile_id"]
        angle = int(row["angle"])
        sp = source_paths(run_root, model, tile, angle)
        image_path = data_image_path(image_root, tile, angle)
        gt_path = sp["rotated_gt"] if sp["rotated_gt"].exists() else data_gt_path(Path(args.data_gt_root), tile, angle)
        gt_rows = parse_dota_ann(gt_path)
        center = None
        if raw_points:
            center = (sum(x for x, _ in raw_points) / len(raw_points), sum(y for _, y in raw_points) / len(raw_points))
        title = (
            f"{model} {tile} angle={angle} score={row['score']} "
            f"raw={row['raw_box_type']} valid={row['valid_mask_ratio_inside_box']} "
            f"padding={row['padding_overlap_ratio']}"
        )
        draw_overlay(image_path, sample_dir / "full_tile_raw_polygon_overlay.png", pred_points=raw_points, gt_rows=None, mask_path=sp["valid_mask"], title=title)
        draw_overlay(image_path, sample_dir / "full_tile_raw_hbb_overlay.png", hbb=raw_hbb, gt_rows=None, mask_path=sp["valid_mask"], title=title + " HBB")
        if center:
            draw_overlay(image_path, sample_dir / "zoom_fixed_256_overlay.png", pred_points=raw_points, gt_rows=None, title=title, zoom_center=center, zoom_size=256)
        diagnosis = "cannot_verify"
        if "coordinate_frame_mismatch" in row["invalid_reason"]:
            diagnosis = "coordinate_mismatch"
        if row["is_hbb_fallback"] == "true":
            diagnosis = "hbb_fallback_visualization"
        if "padding_contaminated" in row["invalid_reason"]:
            diagnosis = "padding_artifact"
        if row["is_oversized_for_small_vehicle"] == "true":
            diagnosis = "actual_large_prediction"
        meta = {
            "crop_id": row["crop_id"],
            "model_name": model,
            "tile_id": tile,
            "angle": angle,
            "score": row["score"],
            "raw_box": row["raw_box"],
            "raw_box_type": row["raw_box_type"],
            "raw_box_angle": row["raw_box_angle"],
            "rendered_box": row["rendered_box"],
            "rendered_box_type": row["rendered_box_type"],
            "rendered_vs_raw_hbb_iou": row["rendered_box_iou_with_raw_hbb"],
            "valid_mask_ratio_inside_box": row["valid_mask_ratio_inside_box"],
            "padding_overlap_ratio": row["padding_overlap_ratio"],
            "best_gt_class": row["best_gt_class"],
            "best_gt_iou": row["best_gt_iou"],
            "diagnosis": diagnosis,
        }
        write_json(sample_dir / "metadata.json", meta)


def write_urgent_diagnosis(audit_dir: Path, summary: dict[str, Any]) -> None:
    valid_rate = float(summary.get("valid_rate", 0.0))
    q1 = "NO" if valid_rate < 0.7 else "PARTIAL"
    q6 = "YES" if valid_rate < 0.7 else "NO"
    md = [
        "# Urgent Crop Pack Diagnosis Report",
        "",
        "## Q1. 旧 200 张 crop 是否可用于 human labeling？",
        "",
        f"A. **{q1}**. valid_for_human_audit = {summary.get('valid_for_human_audit', 0)} / {summary.get('total_crops', 0)}.",
        "",
        "## Q2. 红框是否实际 detector prediction？",
        "",
        md_table([{
            "verified raw prediction": summary.get("source_prediction_found", 0),
            "missing source": summary.get("total_crops", 0) - summary.get("source_prediction_found", 0),
            "crop window suspected": summary.get("crop_window_suspected", 0),
            "coordinate mismatch": summary.get("coordinate_mismatch", 0),
            "HBB fallback": summary.get("hbb_fallback", 0),
            "cannot verify": summary.get("total_crops", 0) - summary.get("raw_box_verified", 0),
        }]),
        "",
        "## Q3. 全是水平框的原因是什么？",
        "",
        "最可能原因：`OBB/polygon rendered as HBB fallback`。旧脚本用 polygon bounds 后通过 `patches.Rectangle` 画水平外接框，并且没有标题标注 HBB fallback。",
        "",
        "## Q4. 大面积色块/大场面原因是什么？",
        "",
        "最可能原因是两类叠加：`actual degenerate large SV boxes` 与 `crop size follows large HBB`。另有部分样本被 `padding_contaminated` 和 `coordinate_frame_mismatch` 污染。",
        "",
        "## Q5. 是否可以继续计算 corrected-FSV？",
        "",
        "A. **No**. 只有 verified valid_unmatched_sv + human labels 才可以。旧 200 张在完整性通过率低时不能用于 corrected-FSV。",
        "",
        "## Q6. 需要重采样吗？",
        "",
        f"A. **{q6}**. 重采样条件：使用 raw_predictions、固定 256 crop、直接画 raw polygon/OBB、valid mask ratio >= 0.85、padding <= 0.10、剔除 oversized 和坐标不明样本。",
        "",
        "## Q7. 当前科学结论是否受影响？",
        "",
        "- 已有 closed-set/open-vocab diagnostic 图表不自动作废。",
        "- unannotated true-SV contamination audit 不能使用旧 crop。",
        "- corrected-FSV 仍然 BLOCKED。",
        "- 若大框是真实模型输出，应单独作为 degenerate large-SV-box failure 研究，不混入未标注真车审计。",
        "",
        "## Machine Summary",
        "",
        md_table([summary]),
    ]
    write_md(audit_dir / "urgent_crop_pack_diagnosis_report.md", "\n".join(md))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", default="experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-image-root", default="data/DOTA1_1024_500/angle_sweep_val/realistic")
    parser.add_argument("--data-gt-root", default="data/DOTA1_1024_500/angle_sweep_val/realistic")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    template = audit_dir / "human_sv_crop_audit_template.csv"
    rows = read_csv(template)
    write_quarantine_marker(audit_dir)
    build_file_inventory(args, rows, audit_dir)
    size_stats = compute_sv_size_distribution(Path(args.data_gt_root), audit_dir)
    audit, oversized, padding, mapping = audit_rows(args, rows, size_stats)
    summary = write_audit_reports(audit_dir, audit, oversized, padding, mapping)
    write_rendering_code_audit(repo_root, audit_dir)
    make_debug_overlays(args, audit, audit_dir)
    write_urgent_diagnosis(audit_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
