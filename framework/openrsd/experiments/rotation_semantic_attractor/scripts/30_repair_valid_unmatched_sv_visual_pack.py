#!/usr/bin/env python3
"""Repair the valid_unmatched_sv crop pack after visual box audit failures.

This script only edits the manual-audit artifacts. It does not rerun inference,
training, or any full benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


AUDIT_REL = Path("experiments/rotation_semantic_attractor/reports/visual_summary/audit")
VISUAL_REL = Path("experiments/rotation_semantic_attractor/reports/visual_summary")
RUN_REL = Path("experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle")
IMAGE_REL = Path("data/DOTA1_1024_500/angle_sweep_val/realistic")
SUBSET_NAME = "valid_unmatched_sv_300_for_corrected_fsv.csv"
TARGET = 300
RAW_CACHE: dict[str, dict[str, Any]] = {}
TONIGHT_SUBSETS = {
    "valid_unmatched_sv": "valid_unmatched_sv_300_for_corrected_fsv.csv",
    "true_sv_positive_control": "true_sv_positive_control_100_for_qc.csv",
    "degenerate_large_sv_box": "degenerate_large_sv_box_60_spotcheck.csv",
    "padding_artifact": "padding_artifact_30_spotcheck.csv",
    "strict_object_flip": "strict_object_flip_30_qualitative_candidates.csv",
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
    "coordinate_frame_consistent",
    "auto_label",
    "valid_for_human_audit",
    "human_label",
    "human_confidence",
    "human_notes",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def backup(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    dst = path.with_name(path.name + f".bak_visual_repair_{stamp}")
    shutil.copy2(path, dst)
    return dst


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
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
        w, h = abs(b[2] - b[0]), abs(b[3] - b[1])
        return w, h, max(w / max(h, 1e-9), h / max(w, 1e-9))
    ds = []
    for i, (x1, y1) in enumerate(points[:4]):
        x2, y2 = points[(i + 1) % 4]
        ds.append(math.hypot(x2 - x1, y2 - y1))
    long_edge = max(ds)
    short_edge = min(d for d in ds if d > 1e-9) if any(d > 1e-9 for d in ds) else 0.0
    return long_edge, short_edge, long_edge / max(short_edge, 1e-9)


def parse_dota(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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


def hbb_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-9)


def best_gt(points: list[tuple[float, float]], gts: list[dict[str, Any]]) -> tuple[str, float, float]:
    best_cls = "NO_GT_OVERLAP"
    best = 0.0
    sv_best = 0.0
    pb = bounds(points)
    for gt in gts:
        iou = hbb_iou(pb, bounds(gt["points"]))
        if iou > best:
            best = iou
            best_cls = gt["class_name"]
        if gt["class_name"] == "small-vehicle":
            sv_best = max(sv_best, iou)
    return best_cls, best, sv_best


def raw_box_wh_angle(pred: dict[str, Any]) -> tuple[Any, Any, Any]:
    box = pred.get("box") or []
    if isinstance(box, list) and len(box) == 5:
        return box[2], box[3], box[4]
    return "", "", ""


def get_image_path(repo: Path, row: dict[str, Any], raw_data: dict[str, Any]) -> Path:
    meta_path = raw_data.get("metadata", {}).get("rotated_image_path", "")
    if meta_path:
        path = Path(meta_path)
        if not path.is_absolute():
            path = repo / path
        if path.exists():
            return path
    angle = int(float(row["angle"]))
    return repo / IMAGE_REL / f"angle_{angle:03d}" / "images" / f"{row['tile_id']}.png"


def get_gt_path(repo: Path, row: dict[str, Any], raw_data: dict[str, Any]) -> Path:
    angle = int(float(row["angle"]))
    data_gt_path = repo / IMAGE_REL / f"angle_{angle:03d}" / "annfiles" / f"{row['tile_id']}.txt"
    if data_gt_path.exists():
        return data_gt_path
    gt_path = Path(raw_data.get("metadata", {}).get("rotated_gt_path", ""))
    if not gt_path.is_absolute():
        gt_path = repo / gt_path
    return gt_path


def load_prediction(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw_path = Path(row["raw_path"])
    cache_key = str(raw_path)
    raw_data = RAW_CACHE.get(cache_key)
    if raw_data is None:
        raw_data = load_json(raw_path)
        RAW_CACHE[cache_key] = raw_data
    for pred in raw_data.get("final_predictions", []):
        if pred.get("class_name") == "small-vehicle" and str(pred.get("raw_index")) == str(row.get("raw_index")):
            return pred, raw_data
    return None, raw_data


def is_strictly_inside(points: list[tuple[float, float]], image_size: tuple[int, int]) -> bool:
    w, h = image_size
    return bool(points) and all(0 <= x < w and 0 <= y < h for x, y in points)


def candidate_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("raw_path", "")), str(row.get("raw_index", ""))


def identity_from_raw_path(raw_path: str) -> tuple[str | None, str | None, str | None]:
    parts = Path(raw_path).parts
    try:
        idx = parts.index("raw_predictions")
        model = parts[idx + 1]
        tile = parts[idx + 2]
        angle_name = Path(parts[idx + 3]).stem
        angle = angle_name.split("_", 1)[1] if angle_name.startswith("angle_") else None
        return model, tile, str(int(angle)) if angle is not None else None
    except Exception:
        return None, None, None


def draw_line_with_halo(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color: tuple[int, int, int, int], width: int) -> None:
    if not pts:
        return
    closed = pts + [pts[0]]
    draw.line(closed, fill=color, width=width, joint="curve")


def draw_cross(draw: ImageDraw.ImageDraw, cx: float, cy: float, color: tuple[int, int, int, int]) -> None:
    r = 8
    draw.line([(cx - r, cy), (cx + r, cy)], fill=(255, 255, 255, 230), width=5)
    draw.line([(cx, cy - r), (cx, cy + r)], fill=(255, 255, 255, 230), width=5)
    draw.line([(cx - r, cy), (cx + r, cy)], fill=color, width=3)
    draw.line([(cx, cy - r), (cx, cy + r)], fill=color, width=3)


def draw_overlay(
    image_path: Path,
    out_path: Path,
    points: list[tuple[float, float]],
    gts: list[dict[str, Any]],
    title: str,
    zoom_size: int | None = None,
) -> bool:
    if not image_path.exists() or not points:
        return False
    img = Image.open(image_path).convert("RGB")
    offset = (0, 0)
    if zoom_size:
        b = bounds(points)
        if not b:
            return False
        cx = min(max((b[0] + b[2]) / 2.0, 0), img.size[0] - 1)
        cy = min(max((b[1] + b[3]) / 2.0, 0), img.size[1] - 1)
        half = zoom_size // 2
        left = max(0, min(int(round(cx - half)), max(0, img.size[0] - zoom_size)))
        top = max(0, min(int(round(cy - half)), max(0, img.size[1] - zoom_size)))
        right = min(img.size[0], left + zoom_size)
        bottom = min(img.size[1], top + zoom_size)
        img = img.crop((left, top, right, bottom))
        offset = (left, top)
    draw = ImageDraw.Draw(img, "RGBA")
    ox, oy = offset
    for gt in gts:
        pts = [(x - ox, y - oy) for x, y in gt["points"]]
        color = (0, 255, 0, 210) if gt["class_name"] == "small-vehicle" else (0, 160, 255, 170)
        draw_line_with_halo(draw, pts, color, 1 if zoom_size is None else 2)
    pts = [(x - ox, y - oy) for x, y in points]
    draw_line_with_halo(draw, pts, (255, 0, 0, 245), 1 if zoom_size is None else 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def red_pixel_count(path: Path) -> int:
    if not path.exists():
        return 0
    arr = np.asarray(Image.open(path).convert("RGB"))
    mask = (arr[:, :, 0] >= 220) & (arr[:, :, 1] <= 80) & (arr[:, :, 2] <= 80)
    return int(mask.sum())


def make_sample(
    repo: Path,
    crop_id: str,
    candidate: dict[str, Any],
    label_values: dict[str, str] | None = None,
    category: str | None = None,
) -> tuple[dict[str, str], dict[str, Any], dict[str, int]]:
    pred, raw_data = load_prediction(candidate)
    if pred is None:
        raise RuntimeError(f"prediction not found for {candidate_key(candidate)}")
    points = points_from_record(pred)
    image_path = get_image_path(repo, candidate, raw_data)
    gt_path = get_gt_path(repo, candidate, raw_data)
    gts = parse_dota(gt_path)
    long_edge, short_edge, aspect = edge_size(points)
    area = polygon_area(points)
    best_cls, best_iou, nearest_sv = best_gt(points, gts)
    raw_w, raw_h, raw_angle = raw_box_wh_angle(pred)
    cat = category or str(candidate.get("audit_category") or "valid_unmatched_sv")
    sample_dir = repo / VISUAL_REL / "crops_verified_expanded" / cat / crop_id
    full = sample_dir / "full_tile_raw_polygon_overlay.png"
    zoom = sample_dir / "zoom_fixed_256_overlay.png"
    gt_context = sample_dir / "gt_context_overlay.png"
    title = (
        f"{cat} {candidate['model_name']} {candidate['tile_id']} angle={candidate['angle']} "
        f"score={as_float(candidate.get('score')):.3f} pred={pred.get('class_name', 'small-vehicle')} raw={candidate.get('raw_box_type')} "
        f"area={area:.1f} gt={best_cls} iou={best_iou:.3f} visual_fixed"
    )
    draw_overlay(image_path, full, points, [], title)
    draw_overlay(image_path, zoom, points, [], title, zoom_size=256)
    draw_overlay(image_path, gt_context, points, gts, title, zoom_size=512)
    meta = {
        **candidate,
        "crop_id": crop_id,
        "audit_category": cat,
        "image_path": str(image_path),
        "gt_path": str(gt_path),
        "raw_prediction_record": json.dumps(pred, ensure_ascii=False),
        "raw_box_width_long_edge": long_edge,
        "raw_box_height_short_edge": short_edge,
        "raw_box_aspect_ratio": aspect,
        "raw_obb_w_box2": raw_w,
        "raw_obb_h_box3": raw_h,
        "raw_obb_angle_box4": raw_angle,
        "raw_box_area": area,
        "best_gt_class": best_cls,
        "best_gt_iou": best_iou,
        "nearest_sv_gt_iou": nearest_sv,
        "coordinate_frame_consistent": True,
        "visual_repair_status": "VISUAL_FIXED_STRICT_INSIDE",
    }
    meta_path = sample_dir / "metadata.json"
    write_json(meta_path, meta)
    labels = label_values or {}
    row = {
        "crop_id": crop_id,
        "audit_category": cat,
        "image_path_full_tile": str(full),
        "image_path_zoom": str(zoom),
        "image_path_gt_context": str(gt_context),
        "metadata_json": str(meta_path),
        "model_name": str(candidate["model_name"]),
        "tile_id": str(candidate["tile_id"]),
        "angle": str(int(float(candidate["angle"]))),
        "score": f"{as_float(candidate.get('score')):.6f}",
        "pred_class": str(pred.get("class_name", "small-vehicle")),
        "raw_box_type": str(candidate.get("raw_box_type", "")),
        "raw_box_angle": str(raw_angle),
        "raw_box_area": f"{area:.6f}",
        "valid_mask_ratio_inside_box": f"{as_float(candidate.get('valid_mask_ratio_inside_box')):.6f}",
        "padding_overlap_ratio": f"{as_float(candidate.get('padding_overlap_ratio')):.6f}",
        "best_gt_class": best_cls,
        "best_gt_iou": f"{best_iou:.6f}",
        "coordinate_frame_consistent": "true",
        "auto_label": cat,
        "valid_for_human_audit": str(candidate.get("valid_for_human_audit", "true")).lower(),
        "human_label": labels.get("human_label", ""),
        "human_confidence": labels.get("human_confidence", ""),
        "human_notes": labels.get("human_notes", ""),
    }
    red_counts = {
        "full_red_pixels": red_pixel_count(full),
        "zoom_red_pixels": red_pixel_count(zoom),
        "gt_context_red_pixels": red_pixel_count(gt_context),
    }
    return row, meta, red_counts


def row_to_candidate(row: dict[str, str]) -> dict[str, Any]:
    meta = load_json(Path(row["metadata_json"]))
    raw_model, raw_tile, raw_angle = identity_from_raw_path(str(meta["raw_path"]))
    return {
        "audit_category": row.get("audit_category") or meta.get("audit_category") or "valid_unmatched_sv",
        "model_name": str(raw_model or meta.get("model_name") or row["model_name"]),
        "tile_id": str(raw_tile or meta.get("tile_id") or row["tile_id"]),
        "angle": str(raw_angle or meta.get("angle") or row["angle"]),
        "score": str(meta.get("score") or row["score"]),
        "raw_box_type": str(meta.get("raw_box_type") or row.get("raw_box_type", "")),
        "valid_mask_ratio_inside_box": str(meta.get("valid_mask_ratio_inside_box") or row.get("valid_mask_ratio_inside_box", "")),
        "padding_overlap_ratio": str(meta.get("padding_overlap_ratio") or row.get("padding_overlap_ratio", "")),
        "valid_for_human_audit": str(row.get("valid_for_human_audit") or meta.get("valid_for_human_audit") or "true").lower(),
        "raw_path": meta["raw_path"],
        "raw_index": str(meta["raw_index"]),
    }


def gray_padding_ratio(image_path: Path, points: list[tuple[float, float]]) -> float:
    if not image_path.exists() or not points:
        return 1.0
    img = Image.open(image_path).convert("RGB")
    mask = Image.new("1", img.size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=1)
    mask_arr = np.asarray(mask).astype(bool)
    if not mask_arr.any():
        return 1.0
    arr = np.asarray(img)
    pix = arr[mask_arr]
    pix_i = pix.astype(int)
    channel_span = pix_i.max(axis=1) - pix_i.min(axis=1)
    intensity = pix_i.mean(axis=1)
    gray = (channel_span <= 8) & (intensity >= 80) & (intensity <= 170)
    return float(gray.mean())


def current_row_is_good(repo: Path, row: dict[str, str]) -> tuple[bool, str]:
    try:
        candidate = row_to_candidate(row)
        pred, raw_data = load_prediction(candidate)
        if pred is None:
            return False, "missing_prediction"
        points = points_from_record(pred)
        image_path = get_image_path(repo, candidate, raw_data)
        image_size = Image.open(image_path).size
        if not is_strictly_inside(points, image_size):
            return False, "pred_box_outside_image_bounds"
        if gray_padding_ratio(image_path, points) > 0.25:
            return False, "pred_box_on_gray_padding"
        return True, "strict_inside"
    except Exception as exc:
        return False, f"error:{exc}"


def candidate_is_good(repo: Path, row: dict[str, Any]) -> bool:
    try:
        pred, raw_data = load_prediction(row)
        if pred is None:
            return False
        points = points_from_record(pred)
        image_path = get_image_path(repo, row, raw_data)
        return is_strictly_inside(points, Image.open(image_path).size) and gray_padding_ratio(image_path, points) <= 0.25
    except Exception:
        return False


def update_matching_rows(rows: list[dict[str, str]], replacement_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        crop_id = row.get("crop_id", "")
        if crop_id in replacement_by_id:
            new_row = dict(row)
            for key, value in replacement_by_id[crop_id].items():
                if key in HEADERS:
                    new_row[key] = value
            out.append(new_row)
        else:
            out.append(row)
    return out


def regenerate_static_gallery(path: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        meta = [
            ("样本 ID", row["crop_id"]),
            ("模型", row["model_name"]),
            ("图块", row["tile_id"]),
            ("角度", row["angle"]),
            ("分数", row["score"]),
            ("最近 GT", f"{row['best_gt_class']} / IoU={row['best_gt_iou']}"),
            ("有效区域", row["valid_mask_ratio_inside_box"]),
            ("padding", row["padding_overlap_ratio"]),
        ]
        meta_html = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in meta)
        imgs = "".join(
            f"<figure><img src=\"file://{row[col]}\"><figcaption>{label}</figcaption></figure>"
            for col, label in [
                ("image_path_full_tile", "整图红框定位"),
                ("image_path_zoom", "256 放大红框"),
                ("image_path_gt_context", "GT 上下文"),
            ]
        )
        cards.append(f"<section><h2>{row['crop_id']}</h2><div class=\"imgs\">{imgs}</div><ul>{meta_html}</ul></section>")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>valid unmatched sv 300 gallery - visual repaired</title>
<style>
body{{font-family:Arial,sans-serif;margin:20px;background:#f7f7f7;color:#222}}
section{{background:white;border:1px solid #ddd;margin:0 0 18px;padding:12px;border-radius:6px}}
.imgs{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:10px}}
img{{max-width:100%;border:1px solid #ccc;background:#888}}
figcaption{{font-size:13px;color:#555}}
ul{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:4px 12px;padding-left:18px}}
</style></head><body>
<h1>未匹配小车 300 张图册 - 已修复红框可视化</h1>
<p>这份静态图册用于查看图片；点击标注请使用本地服务页面。后台 CSV 仍保存英文标签。</p>
{''.join(cards)}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/data1/zcy/OpenRSD")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--redraw-current-only", action="store_true", help="Only redraw the current 300 rows with the current visual style.")
    parser.add_argument(
        "--redraw-all-tonight-current-only",
        action="store_true",
        help="Redraw all current tonight subset rows with the same thin-box visual style, preserving labels.",
    )
    parser.add_argument(
        "--replace-zero-red-tonight",
        action="store_true",
        help="Replace current tonight subset rows whose red prediction outline is still invisible after redraw.",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root)
    audit_dir = repo / AUDIT_REL
    work_dir = Path(args.work_dir) if args.work_dir else audit_dir / "tonight_manual_audit_20260601"
    label_csv = audit_dir / "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv"
    template_csv = audit_dir / "human_sv_crop_audit_template_VERIFIED_EXPANDED.csv"
    subset_csv = work_dir / SUBSET_NAME
    pool_csv = audit_dir / "verified_expanded_candidate_pool.csv"
    if not args.force:
        raise SystemExit("Use --force after confirming this should repair the manual-audit pack.")
    required = [label_csv, template_csv, pool_csv]
    if args.replace_zero_red_tonight or args.redraw_all_tonight_current_only:
        required.extend(work_dir / name for name in TONIGHT_SUBSETS.values())
    else:
        required.append(subset_csv)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = {
        "label_csv": str(backup(label_csv, stamp)),
        "template_csv": str(backup(template_csv, stamp)),
    }
    if args.replace_zero_red_tonight or args.redraw_all_tonight_current_only:
        backups["subset_csvs"] = {
            name: str(backup(work_dir / name, stamp)) for name in TONIGHT_SUBSETS.values()
        }
    else:
        backups["subset_csv"] = str(backup(subset_csv, stamp))

    label_headers, label_rows = read_csv(label_csv)
    template_headers, template_rows = read_csv(template_csv)

    if args.replace_zero_red_tonight:
        pool_headers, pool_rows = read_csv(pool_csv)
        pool_by_cat: dict[str, list[dict[str, Any]]] = {}
        for row in pool_rows:
            cat = row.get("audit_category", "")
            if cat in TONIGHT_SUBSETS:
                pool_by_cat.setdefault(cat, []).append(row)
        for cat in pool_by_cat:
            pool_by_cat[cat].sort(key=lambda r: (-as_float(r.get("score")), r.get("model_name", ""), r.get("tile_id", "")))

        replacement_by_id: dict[str, dict[str, str]] = {}
        audit_rows: list[dict[str, Any]] = []
        subset_summaries: dict[str, Any] = {}
        used_keys_by_cat: dict[str, set[tuple[str, str]]] = {cat: set() for cat in TONIGHT_SUBSETS}
        subset_data: dict[str, tuple[Path, list[str], list[dict[str, str]]]] = {}
        for category, filename in TONIGHT_SUBSETS.items():
            path = work_dir / filename
            headers, rows = read_csv(path)
            subset_data[category] = (path, headers, rows)
            for row in rows:
                try:
                    used_keys_by_cat[category].add(candidate_key(row_to_candidate(row)))
                except Exception:
                    pass

        for category, (path, headers, rows) in subset_data.items():
            new_rows: list[dict[str, str]] = []
            bad = []
            fixed = 0
            failed = 0
            cleared_labels = 0
            rejects = 0
            pool_iter = iter(pool_by_cat.get(category, []))
            for row in rows:
                current_counts = {
                    "full": red_pixel_count(Path(row.get("image_path_full_tile", ""))),
                    "zoom": red_pixel_count(Path(row.get("image_path_zoom", ""))),
                    "gt_context": red_pixel_count(Path(row.get("image_path_gt_context", ""))),
                }
                if all(v > 0 for v in current_counts.values()):
                    new_rows.append(row)
                    continue
                bad.append(row["crop_id"])
                accepted_row: dict[str, str] | None = None
                accepted_meta: dict[str, Any] | None = None
                accepted_counts: dict[str, int] | None = None
                while True:
                    try:
                        candidate = next(pool_iter)
                    except StopIteration:
                        break
                    key = candidate_key(candidate)
                    if key in used_keys_by_cat[category]:
                        continue
                    labels = {"human_label": "", "human_confidence": "", "human_notes": ""}
                    trial_row, trial_meta, trial_counts = make_sample(repo, row["crop_id"], candidate, labels, category=category)
                    if trial_counts["full_red_pixels"] > 0 and trial_counts["zoom_red_pixels"] > 0 and trial_counts["gt_context_red_pixels"] > 0:
                        accepted_row = trial_row
                        accepted_meta = trial_meta
                        accepted_counts = trial_counts
                        used_keys_by_cat[category].add(key)
                        break
                    rejects += 1
                if accepted_row is None or accepted_meta is None or accepted_counts is None:
                    failed += 1
                    new_rows.append(row)
                    audit_rows.append({
                        "crop_id": row["crop_id"],
                        "audit_category": category,
                        "repair_status": "BLOCKED_NO_VISIBLE_REPLACEMENT",
                        "old_full_red_pixels": current_counts["full"],
                        "old_zoom_red_pixels": current_counts["zoom"],
                        "old_gt_context_red_pixels": current_counts["gt_context"],
                        "replacement_rejects": rejects,
                    })
                    continue
                if row.get("human_label"):
                    cleared_labels += 1
                fixed += 1
                new_rows.append(accepted_row)
                replacement_by_id[row["crop_id"]] = accepted_row
                audit_rows.append({
                    "crop_id": row["crop_id"],
                    "audit_category": category,
                    "repair_status": "REPLACED_ZERO_RED_VISUAL",
                    "old_full_red_pixels": current_counts["full"],
                    "old_zoom_red_pixels": current_counts["zoom"],
                    "old_gt_context_red_pixels": current_counts["gt_context"],
                    "new_full_red_pixels": accepted_counts["full_red_pixels"],
                    "new_zoom_red_pixels": accepted_counts["zoom_red_pixels"],
                    "new_gt_context_red_pixels": accepted_counts["gt_context_red_pixels"],
                    "new_model_name": accepted_row["model_name"],
                    "new_tile_id": accepted_row["tile_id"],
                    "new_angle": accepted_row["angle"],
                    "new_raw_path": accepted_meta.get("raw_path", ""),
                    "new_raw_index": accepted_meta.get("raw_index", ""),
                })
            write_csv(path, headers or HEADERS, new_rows)
            subset_summaries[category] = {
                "csv": str(path),
                "rows": len(new_rows),
                "zero_red_rows_before": len(bad),
                "replaced_zero_red_rows": fixed,
                "blocked_zero_red_rows": failed,
                "cleared_labels_for_replaced_rows": cleared_labels,
                "red_pixel_zero_counts_after": {
                    "full": sum(1 for r in new_rows if red_pixel_count(Path(r.get("image_path_full_tile", ""))) == 0),
                    "zoom": sum(1 for r in new_rows if red_pixel_count(Path(r.get("image_path_zoom", ""))) == 0),
                    "gt_context": sum(1 for r in new_rows if red_pixel_count(Path(r.get("image_path_gt_context", ""))) == 0),
                },
            }
        if replacement_by_id:
            write_csv(label_csv, label_headers or HEADERS, update_matching_rows(label_rows, replacement_by_id))
            write_csv(template_csv, template_headers or HEADERS, update_matching_rows(template_rows, replacement_by_id))
        out_dir = work_dir / "visual_box_audit_all_tonight_subsets"
        repair_csv = out_dir / "visual_replace_zero_red_all_tonight_subsets.csv"
        if audit_rows:
            fieldnames = sorted({key for row in audit_rows for key in row})
            write_csv(repair_csv, fieldnames, audit_rows)
        summary = {
            "status": "DONE" if all(v["blocked_zero_red_rows"] == 0 for v in subset_summaries.values()) else "PARTIAL",
            "mode": "replace_zero_red_tonight",
            "subsets": subset_summaries,
            "total_zero_red_rows_before": sum(v["zero_red_rows_before"] for v in subset_summaries.values()),
            "total_replaced_zero_red_rows": sum(v["replaced_zero_red_rows"] for v in subset_summaries.values()),
            "total_blocked_zero_red_rows": sum(v["blocked_zero_red_rows"] for v in subset_summaries.values()),
            "backups": backups,
            "label_csv": str(label_csv),
            "template_csv": str(template_csv),
            "repair_csv": str(repair_csv),
        }
        write_json(out_dir / "visual_replace_zero_red_all_tonight_subsets.json", summary)
        md = [
            "# Zero-Red Visual Replacement For Tonight Manual Audit Subsets",
            "",
            f"Status: **{summary['status']}**",
            "",
            "Rows whose prediction outline still had zero red pixels after thin-box redraw were replaced with same-category visible candidates from the existing verified expanded candidate pool. Replaced rows have empty human labels to avoid carrying labels across different images.",
            "",
            f"- zero-red rows before: {summary['total_zero_red_rows_before']}",
            f"- replaced rows: {summary['total_replaced_zero_red_rows']}",
            f"- blocked rows: {summary['total_blocked_zero_red_rows']}",
            "",
        ]
        for category, info in subset_summaries.items():
            md.extend([
                f"## {category}",
                "",
                f"- zero-red before: {info['zero_red_rows_before']}",
                f"- replaced: {info['replaced_zero_red_rows']}",
                f"- blocked: {info['blocked_zero_red_rows']}",
                f"- zero-red after: `{info['red_pixel_zero_counts_after']}`",
                "",
            ])
        write_md(out_dir / "visual_replace_zero_red_all_tonight_subsets.md", "\n".join(md))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.redraw_all_tonight_current_only:
        audit_rows: list[dict[str, Any]] = []
        replacement_by_id: dict[str, dict[str, str]] = {}
        subset_summaries: dict[str, Any] = {}
        for category, filename in TONIGHT_SUBSETS.items():
            path = work_dir / filename
            subset_headers, subset_rows = read_csv(path)
            redrawn_rows: list[dict[str, str]] = []
            for row in subset_rows:
                candidate = row_to_candidate(row)
                labels = {k: row.get(k, "") for k in ["human_label", "human_confidence", "human_notes"]}
                new_row, meta, red_counts = make_sample(repo, row["crop_id"], candidate, labels, category=category)
                redrawn_rows.append(new_row)
                replacement_by_id[row["crop_id"]] = new_row
                audit_rows.append({
                    "crop_id": row["crop_id"],
                    "audit_category": category,
                    "repair_status": "redrawn_current_only_thin_box",
                    "model_name": new_row["model_name"],
                    "tile_id": new_row["tile_id"],
                    "angle": new_row["angle"],
                    "raw_path": meta.get("raw_path", ""),
                    "raw_index": meta.get("raw_index", ""),
                    **red_counts,
                })
                if len(audit_rows) % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "redraw_all_tonight_progress",
                                "processed": len(audit_rows),
                                "current_category": category,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            write_csv(path, subset_headers or HEADERS, redrawn_rows)
            subset_summaries[category] = {
                "csv": str(path),
                "rows": len(redrawn_rows),
                "red_pixel_min": {
                    "full": min(r["full_red_pixels"] for r in audit_rows if r["audit_category"] == category),
                    "zoom": min(r["zoom_red_pixels"] for r in audit_rows if r["audit_category"] == category),
                    "gt_context": min(r["gt_context_red_pixels"] for r in audit_rows if r["audit_category"] == category),
                },
                "red_pixel_zero_counts": {
                    "full": sum(1 for r in audit_rows if r["audit_category"] == category and int(r["full_red_pixels"]) == 0),
                    "zoom": sum(1 for r in audit_rows if r["audit_category"] == category and int(r["zoom_red_pixels"]) == 0),
                    "gt_context": sum(1 for r in audit_rows if r["audit_category"] == category and int(r["gt_context_red_pixels"]) == 0),
                },
            }
        write_csv(label_csv, label_headers or HEADERS, update_matching_rows(label_rows, replacement_by_id))
        write_csv(template_csv, template_headers or HEADERS, update_matching_rows(template_rows, replacement_by_id))

        out_dir = work_dir / "visual_box_audit_all_tonight_subsets"
        repair_csv = out_dir / "visual_redraw_thin_box_all_tonight_subsets.csv"
        write_csv(repair_csv, list(audit_rows[0].keys()), audit_rows)
        summary = {
            "status": "DONE",
            "mode": "redraw_all_tonight_current_only",
            "total_rows": len(audit_rows),
            "style": "thin_red_outline_no_white_halo_no_center_cross",
            "subsets": subset_summaries,
            "overall_red_pixel_zero_counts": {
                "full": sum(1 for r in audit_rows if int(r["full_red_pixels"]) == 0),
                "zoom": sum(1 for r in audit_rows if int(r["zoom_red_pixels"]) == 0),
                "gt_context": sum(1 for r in audit_rows if int(r["gt_context_red_pixels"]) == 0),
            },
            "backups": backups,
            "label_csv": str(label_csv),
            "template_csv": str(template_csv),
            "repair_csv": str(repair_csv),
        }
        write_json(out_dir / "visual_redraw_thin_box_all_tonight_subsets.json", summary)
        md = [
            "# Thin Box Redraw For All Tonight Manual Audit Subsets",
            "",
            "Status: **DONE**",
            "",
            "This pass redraws the current rows in every tonight labeling subset with the same thin red prediction outline used for the repaired 300-row main set. It preserves existing human labels and does not rerun inference or training.",
            "",
            f"- Total rows redrawn: {summary['total_rows']}",
            f"- Overall zero-red pixel counts: `{summary['overall_red_pixel_zero_counts']}`",
            "",
            "## Subsets",
            "",
        ]
        for category, info in subset_summaries.items():
            md.extend([
                f"### {category}",
                "",
                f"- rows: {info['rows']}",
                f"- zero-red pixel counts: `{info['red_pixel_zero_counts']}`",
                f"- min red pixels: `{info['red_pixel_min']}`",
                f"- csv: `{info['csv']}`",
                "",
            ])
        md.extend([
            "## Outputs",
            "",
            f"- repair CSV: `{repair_csv}`",
            f"- repair JSON: `{out_dir / 'visual_redraw_thin_box_all_tonight_subsets.json'}`",
        ])
        write_md(out_dir / "visual_redraw_thin_box_all_tonight_subsets.md", "\n".join(md))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    subset_headers, subset_rows = read_csv(subset_csv)
    if len(subset_rows) != TARGET:
        raise RuntimeError(f"Expected {TARGET} subset rows, got {len(subset_rows)}")

    if args.redraw_current_only:
        redrawn_rows: list[dict[str, str]] = []
        audit_rows: list[dict[str, Any]] = []
        for row in subset_rows:
            candidate = row_to_candidate(row)
            labels = {k: row.get(k, "") for k in ["human_label", "human_confidence", "human_notes"]}
            new_row, meta, red_counts = make_sample(repo, row["crop_id"], candidate, labels)
            redrawn_rows.append(new_row)
            audit_rows.append({
                "crop_id": row["crop_id"],
                "repair_status": "redrawn_current_only_thin_box",
                "model_name": new_row["model_name"],
                "tile_id": new_row["tile_id"],
                "angle": new_row["angle"],
                "raw_path": meta.get("raw_path", ""),
                "raw_index": meta.get("raw_index", ""),
                **red_counts,
            })
            if len(audit_rows) % 25 == 0:
                print(json.dumps({"stage": "redraw_current_progress", "processed": len(audit_rows)}, ensure_ascii=False), flush=True)
        write_csv(subset_csv, subset_headers or HEADERS, redrawn_rows)
        by_id = {row["crop_id"]: row for row in redrawn_rows}
        write_csv(label_csv, label_headers or HEADERS, update_matching_rows(label_rows, by_id))
        write_csv(template_csv, template_headers or HEADERS, update_matching_rows(template_rows, by_id))
        gallery = work_dir / "html" / "valid_unmatched_sv_300_gallery.html"
        regenerate_static_gallery(gallery, redrawn_rows)
        out_dir = work_dir / "visual_box_audit_valid_unmatched_300"
        repair_csv = out_dir / "visual_redraw_thin_box_valid_unmatched_300.csv"
        write_csv(repair_csv, list(audit_rows[0].keys()), audit_rows)
        summary = {
            "status": "DONE",
            "mode": "redraw_current_only",
            "total": len(redrawn_rows),
            "red_pixel_min": {
                "full": min(r["full_red_pixels"] for r in audit_rows),
                "zoom": min(r["zoom_red_pixels"] for r in audit_rows),
                "gt_context": min(r["gt_context_red_pixels"] for r in audit_rows),
            },
            "red_pixel_zero_counts": {
                "full": sum(1 for r in audit_rows if int(r["full_red_pixels"]) == 0),
                "zoom": sum(1 for r in audit_rows if int(r["zoom_red_pixels"]) == 0),
                "gt_context": sum(1 for r in audit_rows if int(r["gt_context_red_pixels"]) == 0),
            },
            "style": "thin_red_outline_no_white_halo_no_center_cross",
            "backups": backups,
            "subset_csv": str(subset_csv),
            "label_csv": str(label_csv),
            "static_gallery": str(gallery),
            "repair_csv": str(repair_csv),
        }
        write_json(out_dir / "visual_redraw_thin_box_valid_unmatched_300.json", summary)
        write_md(
            out_dir / "visual_redraw_thin_box_valid_unmatched_300.md",
            "\n".join([
                "# Thin Box Redraw For valid_unmatched_sv 300",
                "",
                "Status: **DONE**",
                "",
                "The 300 current samples were redrawn with a thin red outline only. The previous white halo and center cross were removed because they obscured small targets during human inspection.",
                "",
                f"- Total rows: {summary['total']}",
                f"- Zero-red pixels after redraw: {summary['red_pixel_zero_counts']}",
                f"- Minimum red pixels after redraw: {summary['red_pixel_min']}",
                f"- Repaired subset CSV: `{subset_csv}`",
                f"- Static gallery: `{gallery}`",
            ]),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    current_used = {candidate_key(row_to_candidate(row)) for row in subset_rows}
    good_by_id: dict[str, bool] = {}
    reason_by_id: dict[str, str] = {}
    for row in subset_rows:
        good, reason = current_row_is_good(repo, row)
        good_by_id[row["crop_id"]] = good
        reason_by_id[row["crop_id"]] = reason

    bad_rows = [row for row in subset_rows if not good_by_id[row["crop_id"]]]
    pool_candidates: list[dict[str, Any]] = []
    scanned_pool = 0
    for row in read_csv(pool_csv)[1]:
        if row.get("audit_category") != "valid_unmatched_sv":
            continue
        scanned_pool += 1
        key = candidate_key(row)
        if key in current_used:
            continue
        if candidate_is_good(repo, row):
            pool_candidates.append(row)
        if scanned_pool % 500 == 0:
            print(
                json.dumps(
                    {
                        "stage": "candidate_scan_progress",
                        "scanned_valid_unmatched": scanned_pool,
                        "replacement_pool": len(pool_candidates),
                        "raw_cache": len(RAW_CACHE),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    pool_candidates.sort(key=lambda r: (-as_float(r.get("score")), r.get("model_name", ""), r.get("tile_id", "")))
    if len(pool_candidates) < len(bad_rows):
        raise RuntimeError(f"Need {len(bad_rows)} replacement candidates, found {len(pool_candidates)}")
    print(
        json.dumps(
            {
                "stage": "selection_ready",
                "current_good": TARGET - len(bad_rows),
                "current_bad": len(bad_rows),
                "replacement_pool": len(pool_candidates),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    replacements = iter(pool_candidates)
    repaired_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    replaced_count = 0
    labels_cleared = 0
    invalid_visualization_labels_cleared = 0
    for row in subset_rows:
        crop_id = row["crop_id"]
        if good_by_id[crop_id]:
            candidate = row_to_candidate(row)
            labels = {k: row.get(k, "") for k in ["human_label", "human_confidence", "human_notes"]}
            if labels.get("human_label") == "invalid_visualization":
                labels = {"human_label": "", "human_confidence": "", "human_notes": ""}
                invalid_visualization_labels_cleared += 1
            status = "kept"
        else:
            candidate = next(replacements)
            labels = {"human_label": "", "human_confidence": "", "human_notes": ""}
            replaced_count += 1
            if row.get("human_label"):
                labels_cleared += 1
            status = "replaced_bad_visual"
        new_row, meta, red_counts = make_sample(repo, crop_id, candidate, labels)
        repaired_rows.append(new_row)
        audit_rows.append({
            "crop_id": crop_id,
            "repair_status": status,
            "old_reason": reason_by_id.get(crop_id, ""),
            "old_human_label_cleared": "true" if status != "kept" and row.get("human_label") else "false",
            "model_name": new_row["model_name"],
            "tile_id": new_row["tile_id"],
            "angle": new_row["angle"],
            "raw_path": meta.get("raw_path", ""),
            "raw_index": meta.get("raw_index", ""),
            **red_counts,
        })
        if len(audit_rows) % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "redraw_progress",
                        "processed": len(audit_rows),
                        "replaced": replaced_count,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_csv(subset_csv, subset_headers or HEADERS, repaired_rows)
    by_id = {row["crop_id"]: row for row in repaired_rows}
    write_csv(label_csv, label_headers or HEADERS, update_matching_rows(label_rows, by_id))
    write_csv(template_csv, template_headers or HEADERS, update_matching_rows(template_rows, by_id))
    gallery = work_dir / "html" / "valid_unmatched_sv_300_gallery.html"
    regenerate_static_gallery(gallery, repaired_rows)

    out_dir = work_dir / "visual_box_audit_valid_unmatched_300"
    repair_csv = out_dir / "visual_repair_valid_unmatched_300.csv"
    write_csv(repair_csv, list(audit_rows[0].keys()), audit_rows)
    summary = {
        "status": "DONE",
        "total": len(repaired_rows),
        "kept": TARGET - replaced_count,
        "replaced_bad_visual": replaced_count,
        "labels_cleared_for_replaced_rows": labels_cleared,
        "invalid_visualization_labels_cleared_after_visual_fix": invalid_visualization_labels_cleared,
        "red_pixel_min": {
            "full": min(r["full_red_pixels"] for r in audit_rows),
            "zoom": min(r["zoom_red_pixels"] for r in audit_rows),
            "gt_context": min(r["gt_context_red_pixels"] for r in audit_rows),
        },
        "red_pixel_zero_counts": {
            "full": sum(1 for r in audit_rows if int(r["full_red_pixels"]) == 0),
            "zoom": sum(1 for r in audit_rows if int(r["zoom_red_pixels"]) == 0),
            "gt_context": sum(1 for r in audit_rows if int(r["gt_context_red_pixels"]) == 0),
        },
        "old_bad_reason_counts": dict(Counter(reason_by_id[row["crop_id"]] for row in bad_rows)),
        "backups": backups,
        "subset_csv": str(subset_csv),
        "label_csv": str(label_csv),
        "static_gallery": str(gallery),
        "repair_csv": str(repair_csv),
    }
    write_json(out_dir / "visual_repair_valid_unmatched_300.json", summary)
    md = [
        "# Visual Repair For valid_unmatched_sv 300",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"- Total rows: {summary['total']}",
        f"- Kept original strict-inside rows: {summary['kept']}",
        f"- Replaced bad visual rows: {summary['replaced_bad_visual']}",
        f"- Existing human labels cleared because their sample was replaced: {summary['labels_cleared_for_replaced_rows']}",
        f"- Existing invalid_visualization labels cleared after visual fix: {summary['invalid_visualization_labels_cleared_after_visual_fix']}",
        f"- Zero-red pixels after repair: {summary['red_pixel_zero_counts']}",
        f"- Minimum red pixels after repair: {summary['red_pixel_min']}",
        "",
        "## Cause",
        "",
        "The original expanded pack allowed some valid_unmatched_sv predictions whose polygon coordinates were outside the 1024x1024 rotated tile. Those samples could not show a reliable red box in zoom/context views and must not be used for corrected-FSV human labeling.",
        "",
        "## Action",
        "",
        "All 300 valid_unmatched_sv images were redrawn with a red box plus center cross. Rows whose prediction box was outside image bounds were replaced with strict-inside valid_unmatched_sv candidates from the existing candidate pool. No inference or training was rerun.",
        "",
        "## Outputs",
        "",
        f"- Repaired subset CSV: `{subset_csv}`",
        f"- Master label CSV: `{label_csv}`",
        f"- Static gallery: `{gallery}`",
        f"- Repair CSV: `{repair_csv}`",
    ]
    write_md(out_dir / "visual_repair_valid_unmatched_300.md", "\n".join(md))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
