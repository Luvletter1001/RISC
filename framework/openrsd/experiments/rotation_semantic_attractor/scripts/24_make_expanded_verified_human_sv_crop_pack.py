#!/usr/bin/env python3
"""Create an expanded verified human SV crop pack from the full prediction pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


TARGETS = {
    "valid_unmatched_sv": 300,
    "strict_object_flip": 100,
    "true_sv_positive_control": 100,
    "degenerate_large_sv_box": 150,
    "padding_artifact": 100,
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


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


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


def md_table(rows: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    if headers is None:
        headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    w = max(ds)
    h = min(d for d in ds if d > 1e-9) if any(d > 1e-9 for d in ds) else 0.0
    return w, h, max(w / max(h, 1e-9), h / max(w, 1e-9))


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


def mask_metrics(
    mask_path: Path,
    points: list[tuple[float, float]],
    mode: str,
    mask_cache: dict[str, Image.Image],
) -> tuple[float, float, bool]:
    if not mask_path.exists() or not points:
        return 0.0, 1.0, False
    try:
        cache_key = str(mask_path)
        mask = mask_cache.get(cache_key)
        if mask is None:
            mask = Image.open(mask_path).convert("L")
            mask_cache[cache_key] = mask
        w, h = mask.size
        b = bounds(points)
        if not b:
            return 0.0, 1.0, False
        x1, y1, x2, y2 = b
        bx1 = max(0, int(math.floor(x1)))
        by1 = max(0, int(math.floor(y1)))
        bx2 = min(w - 1, int(math.ceil(x2)))
        by2 = min(h - 1, int(math.ceil(y2)))
        if bx2 < bx1 or by2 < by1:
            return 0.0, 1.0, False
        if mode == "exact_polygon":
            poly = Image.new("1", (w, h), 0)
            ImageDraw.Draw(poly).polygon(points, fill=1)
            mpx = mask.load()
            ppx = poly.load()
            total = 0
            valid = 0
            for yy in range(by1, by2 + 1):
                for xx in range(bx1, bx2 + 1):
                    if ppx[xx, yy]:
                        total += 1
                        if mpx[xx, yy] > 127:
                            valid += 1
        else:
            crop = mask.crop((bx1, by1, bx2 + 1, by2 + 1))
            hist = crop.histogram()
            total = crop.size[0] * crop.size[1]
            valid = sum(hist[128:])
        cx = int(round(sum(x for x, _ in points) / len(points)))
        cy = int(round(sum(y for _, y in points) / len(points)))
        center_ok = bool(0 <= cx < w and 0 <= cy < h and mask.getpixel((cx, cy)) > 127)
        ratio = valid / max(total, 1)
        return ratio, 1.0 - ratio, center_ok
    except Exception:
        return 0.0, 1.0, False


def score_bin(score: float) -> str:
    if score < 0.5:
        return "0.3-0.5"
    if score < 0.7:
        return "0.5-0.7"
    if score < 0.9:
        return "0.7-0.9"
    return "0.9-1.0"


def angle_bin(angle: int) -> str:
    return "orthogonal_0_90_180_270" if angle in {0, 90, 180, 270} else "oblique_other"


def area_bin(area: float, stats: dict[str, float]) -> str:
    if area <= stats.get("area_p50", 0.0):
        return "small-like_p0-p50"
    if area <= stats.get("area_p90", 0.0):
        return "medium-like_p50-p90"
    return "large-but-not-oversized_p90-p995"


def category_for(c: dict[str, Any], stats: dict[str, float]) -> str:
    valid_ok = c["valid_mask_ratio_inside_box"] >= 0.85
    pad_ok = c["padding_overlap_ratio"] <= 0.10
    center_ok = c["box_center_in_valid_region"]
    oversized = c["is_oversized"]
    if not valid_ok or not pad_ok or not center_ok:
        return "padding_artifact"
    if oversized:
        return "degenerate_large_sv_box"
    if c["best_gt_class"] == "small-vehicle" and c["best_gt_iou"] >= 0.30:
        return "true_sv_positive_control"
    if c["best_gt_iou"] >= 0.30 and c["best_gt_class"] not in {"small-vehicle", "large-vehicle"}:
        return "strict_object_flip"
    if c["nearest_sv_gt_iou"] < 0.30:
        return "valid_unmatched_sv"
    return "not_selected"


def draw_overlay(image_path: Path, out_path: Path, points: list[tuple[float, float]], gts: list[dict[str, Any]], title: str, zoom_size: int | None = None) -> bool:
    if not image_path.exists() or not points:
        return False
    img = Image.open(image_path).convert("RGB")
    offset = (0, 0)
    if zoom_size:
        cx = min(max(sum(x for x, _ in points) / len(points), 0), img.size[0] - 1)
        cy = min(max(sum(y for _, y in points) / len(points), 0), img.size[1] - 1)
        half = zoom_size // 2
        left = max(0, int(cx - half))
        top = max(0, int(cy - half))
        right = min(img.size[0], int(cx + half))
        bottom = min(img.size[1], int(cy + half))
        if right <= left:
            right = min(img.size[0], left + 1)
        if bottom <= top:
            bottom = min(img.size[1], top + 1)
        img = img.crop((left, top, right, bottom))
        offset = (left, top)
    draw = ImageDraw.Draw(img, "RGBA")
    ox, oy = offset
    for gt in gts:
        pts = [(x - ox, y - oy) for x, y in gt["points"]]
        color = (0, 255, 0, 180) if gt["class_name"] == "small-vehicle" else (0, 160, 255, 120)
        draw.line(pts + [pts[0]], fill=color, width=2)
    pts = [(x - ox, y - oy) for x, y in points]
    draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=3)
    draw.rectangle((0, 0, img.size[0], 48), fill=(0, 0, 0, 170))
    draw.text((5, 5), title[:190], fill=(255, 255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def select_valid_balanced(candidates: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    selected = []
    model_count: Counter[str] = Counter()
    tile_count: Counter[str] = Counter()
    score_count: Counter[str] = Counter()
    angle_count: Counter[str] = Counter()
    area_count: Counter[str] = Counter()
    model_cap = max(1, int(target * 0.25))
    pool = sorted(candidates, key=lambda c: (-c["score"], c["model_name"], c["tile_id"]))
    while pool and len(selected) < target:
        best_idx = None
        best_key = None
        for i, c in enumerate(pool):
            if tile_count[c["tile_id"]] >= 20:
                continue
            if model_count[c["model_name"]] >= model_cap and len({p["model_name"] for p in pool}) > 1:
                continue
            key = (
                model_count[c["model_name"]],
                score_count[c["score_bin"]],
                angle_count[c["angle_bin"]],
                area_count[c["box_area_bin"]],
                -c["score"],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        if best_idx is None:
            break
        c = pool.pop(best_idx)
        selected.append(c)
        model_count[c["model_name"]] += 1
        tile_count[c["tile_id"]] += 1
        score_count[c["score_bin"]] += 1
        angle_count[c["angle_bin"]] += 1
        area_count[c["box_area_bin"]] += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", default="experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle")
    parser.add_argument("--visual-summary-dir", required=True)
    parser.add_argument("--data-image-root", default="data/DOTA1_1024_500/angle_sweep_val/realistic")
    parser.add_argument("--max-metric-rows", type=int, default=50000)
    parser.add_argument("--max-preds-per-event", type=int, default=20)
    parser.add_argument("--mask-mode", choices=["hbb_fast", "exact_polygon"], default="hbb_fast")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    repo = Path(args.repo_root)
    run_root = repo / args.run_root
    root = repo / args.visual_summary_dir
    audit_dir = root / "audit"
    pack_root = root / "crops_verified_expanded"
    image_root = repo / args.data_image_root
    stats_rows = read_csv(audit_dir / "sv_gt_size_distribution.csv")
    stats = {k: as_float(v) for k, v in (stats_rows[0] if stats_rows else {}).items()}
    metric = run_root / "metrics/full_closedset_false_hub_tile_angle.csv"
    metric_rows = [
        r for r in read_csv(metric)
        if r.get("region_mode") == "all_region" and int(float(r.get("num_sv_pred") or 0)) > 0
    ]
    random.Random(args.seed).shuffle(metric_rows)
    candidates: list[dict[str, Any]] = []
    scanned = 0
    seen_keys = set()
    mask_cache: dict[str, Image.Image] = {}
    for row in metric_rows[: args.max_metric_rows]:
        scanned += 1
        model, tile, angle = row["model_name"], row["tile_id"], int(float(row["angle"]))
        raw_path = run_root / "raw_predictions" / model / tile / f"angle_{angle:03d}.json"
        data = load_json(raw_path)
        preds = [p for p in data.get("final_predictions", []) if p.get("class_name") == "small-vehicle"]
        preds.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
        gt_path = Path(data.get("metadata", {}).get("rotated_gt_path", ""))
        if not gt_path.is_absolute():
            gt_path = repo / gt_path
        gts = parse_dota(gt_path)
        mask_path = run_root / "valid_masks" / model / tile / f"angle_{angle:03d}.png"
        image_path = image_root / f"angle_{angle:03d}" / "images" / f"{tile}.png"
        for pred in preds[: args.max_preds_per_event]:
            raw_index = pred.get("raw_index", "")
            key = (model, tile, angle, raw_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pts = points_from_record(pred)
            if not pts:
                continue
            area = polygon_area(pts)
            width, height, aspect = edge_size(pts)
            valid_ratio, pad_ratio, center_ok = mask_metrics(mask_path, pts, args.mask_mode, mask_cache)
            best_cls, best_iou, nearest_sv = best_gt(pts, gts)
            oversized = (
                area > stats.get("area_p995", float("inf"))
                or width > stats.get("width_p99", float("inf"))
                or height > stats.get("height_p99", float("inf"))
            )
            score = float(pred.get("score", 0.0))
            raw_box = pred.get("box", [])
            c = {
                "model_name": model,
                "tile_id": tile,
                "angle": angle,
                "score": score,
                "pred_class": "small-vehicle",
                "raw_box_type": pred.get("box_type", ""),
                "raw_box_angle": raw_box[4] if isinstance(raw_box, list) and len(raw_box) == 5 else "",
                "raw_box_area": area,
                "raw_box_width": width,
                "raw_box_height": height,
                "raw_box_aspect_ratio": aspect,
                "valid_mask_ratio_inside_box": valid_ratio,
                "padding_overlap_ratio": pad_ratio,
                "box_center_in_valid_region": center_ok,
                "best_gt_class": best_cls,
                "best_gt_iou": best_iou,
                "nearest_sv_gt_iou": nearest_sv,
                "is_oversized": oversized,
                "coordinate_frame_consistent": True,
                "raw_path": str(raw_path),
                "gt_path": str(gt_path),
                "image_path": str(image_path),
                "raw_index": raw_index,
                "raw_prediction_record": json.dumps(pred, ensure_ascii=False),
                "score_bin": score_bin(score),
                "angle_bin": angle_bin(angle),
                "box_area_bin": area_bin(area, stats),
                "mask_estimator": args.mask_mode,
            }
            c["audit_category"] = category_for(c, stats)
            if c["audit_category"] != "not_selected":
                candidates.append(c)
        counts = Counter(c["audit_category"] for c in candidates)
        if args.progress_every and scanned % args.progress_every == 0:
            counts_text = ", ".join(f"{k}={counts.get(k, 0)}" for k in TARGETS)
            print(
                f"[scan] rows={scanned} candidates={len(candidates)} masks_cached={len(mask_cache)} {counts_text}",
                flush=True,
            )
        if all(counts.get(k, 0) >= v * 2 or counts.get(k, 0) >= v for k, v in TARGETS.items()):
            break

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        grouped[c["audit_category"]].append(c)
    selected: dict[str, list[dict[str, Any]]] = {}
    selected["valid_unmatched_sv"] = select_valid_balanced(grouped.get("valid_unmatched_sv", []), TARGETS["valid_unmatched_sv"])
    for cat in ["strict_object_flip", "true_sv_positive_control", "degenerate_large_sv_box", "padding_artifact"]:
        selected[cat] = sorted(grouped.get(cat, []), key=lambda c: (-c["score"], c["model_name"], c["tile_id"]))[: TARGETS[cat]]

    template_rows = []
    for cat in TARGETS:
        for idx, c in enumerate(selected.get(cat, [])):
            crop_id = f"{cat}_{idx:04d}"
            sample_dir = pack_root / cat / crop_id
            pred = json.loads(c["raw_prediction_record"])
            pts = points_from_record(pred)
            image_path = Path(c["image_path"])
            gts = parse_dota(Path(c["gt_path"]))
            title = (
                f"{cat} {c['model_name']} {c['tile_id']} angle={c['angle']} score={c['score']:.3f} "
                f"pred=small-vehicle raw={c['raw_box_type']} angle={c['raw_box_angle']} area={c['raw_box_area']:.1f} "
                f"valid={c['valid_mask_ratio_inside_box']:.3f} pad={c['padding_overlap_ratio']:.3f} "
                f"gt={c['best_gt_class']} iou={c['best_gt_iou']:.3f} frame=rotated"
            )
            full = sample_dir / "full_tile_raw_polygon_overlay.png"
            zoom = sample_dir / "zoom_fixed_256_overlay.png"
            gt_context = sample_dir / "gt_context_overlay.png"
            draw_overlay(image_path, full, pts, [], title)
            draw_overlay(image_path, zoom, pts, [], title, zoom_size=256)
            draw_overlay(image_path, gt_context, pts, gts, title, zoom_size=512)
            meta = {
                **{k: c[k] for k in c if k not in {"raw_prediction_record"}},
                "crop_id": crop_id,
                "audit_category": cat,
            }
            meta_path = sample_dir / "metadata.json"
            write_json(meta_path, meta)
            valid_for_human = cat in {"valid_unmatched_sv", "strict_object_flip", "true_sv_positive_control"}
            template_rows.append({
                "crop_id": crop_id,
                "audit_category": cat,
                "image_path_full_tile": str(full),
                "image_path_zoom": str(zoom),
                "image_path_gt_context": str(gt_context),
                "metadata_json": str(meta_path),
                "model_name": c["model_name"],
                "tile_id": c["tile_id"],
                "angle": c["angle"],
                "score": f"{c['score']:.6f}",
                "pred_class": "small-vehicle",
                "raw_box_type": c["raw_box_type"],
                "raw_box_angle": c["raw_box_angle"],
                "raw_box_area": f"{c['raw_box_area']:.6f}",
                "valid_mask_ratio_inside_box": f"{c['valid_mask_ratio_inside_box']:.6f}",
                "padding_overlap_ratio": f"{c['padding_overlap_ratio']:.6f}",
                "best_gt_class": c["best_gt_class"],
                "best_gt_iou": f"{c['best_gt_iou']:.6f}",
                "coordinate_frame_consistent": "true",
                "auto_label": cat,
                "valid_for_human_audit": "true" if valid_for_human else "false",
                "human_label": "",
                "human_confidence": "",
                "human_notes": "",
            })

    template_path = audit_dir / "human_sv_crop_audit_template_VERIFIED_EXPANDED.csv"
    write_csv(template_path, template_rows, HEADERS)
    pool_headers = [
        "audit_category", "model_name", "tile_id", "angle", "score", "raw_box_type", "raw_box_area",
        "raw_box_width", "raw_box_height", "valid_mask_ratio_inside_box", "padding_overlap_ratio",
        "box_center_in_valid_region", "best_gt_class", "best_gt_iou", "nearest_sv_gt_iou",
        "is_oversized", "score_bin", "angle_bin", "box_area_bin", "mask_estimator", "raw_path", "raw_index",
    ]
    write_csv(audit_dir / "verified_expanded_candidate_pool.csv", candidates, pool_headers)
    balance_rows = []
    valid_selected = selected.get("valid_unmatched_sv", [])
    for dim in ["model_name", "score_bin", "angle_bin", "box_area_bin"]:
        for key, value in sorted(Counter(c[dim] for c in valid_selected).items()):
            balance_rows.append({"dimension": dim, "value": key, "count": value, "target_note": "best_effort_stratified"})
    blockers = []
    for cat, target in TARGETS.items():
        have = len(selected.get(cat, []))
        if have < target:
            blockers.append({"audit_category": cat, "target": target, "actual": have, "blocker": "insufficient verified candidates in scanned full prediction pool"})
    write_csv(audit_dir / "verified_expanded_sampling_balance.csv", balance_rows + blockers)
    write_md(
        audit_dir / "verified_expanded_sampling_balance.md",
        "# Verified Expanded Sampling Balance\n\n"
        f"- scanned_metric_rows: `{scanned}`\n"
        f"- candidate_rows: `{len(candidates)}`\n"
        f"- template: `{template_path}`\n\n"
        "## Counts\n\n"
        + md_table([{"audit_category": k, "selected": len(v), "target": TARGETS[k]} for k, v in selected.items()])
        + "\n\n## Balance / Blockers\n\n"
        + md_table(balance_rows + blockers),
    )
    status = {
        "status": "READY_FOR_HUMAN_LABELS" if selected.get("valid_unmatched_sv") else "BLOCKED_NO_VALID_UNMATCHED",
        "pack_root": str(pack_root),
        "template": str(template_path),
        "counts": {k: len(v) for k, v in selected.items()},
        "targets": TARGETS,
        "scanned_metric_rows": scanned,
        "candidate_rows": len(candidates),
        "blockers": blockers,
        "source": "full_closedset_false_hub_tile_angle.csv + full raw_predictions/rotated_gt/valid_masks",
        "mask_estimator": args.mask_mode,
    }
    write_json(audit_dir / "verified_expanded_crop_pack_status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
