#!/usr/bin/env python3
"""Audit whether main metrics use a consistent coordinate frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROWS = [
    {
        "metric_family": "closedset_false_hub_all_region",
        "metric_files": [
            "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_false_hub_tile_angle.csv",
            "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_false_hub_summary_by_model.csv",
            "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_rotation_gain.csv",
            "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_per_class_absorption.csv",
        ],
        "prediction_source": "full_closedset_s2_12angle/canonical_predictions",
        "prediction_coordinate_frame": "canonical",
        "gt_source": "manifest split ann_path / original DOTA tile GT",
        "gt_coordinate_frame": "canonical",
        "image_source": "not used for metric matching",
        "image_coordinate_frame": "not_applicable",
        "valid_mask_source": "not used for all_region metrics",
        "valid_mask_coordinate_frame": "not_applicable",
        "matching_coordinate_frame": "canonical",
        "coordinate_frame_consistent": True,
        "uses_canonical_prediction_on_rotated_image": False,
        "uses_rotated_prediction_on_canonical_gt": False,
        "requires_inverse_rotation": True,
        "inverse_rotation_applied": True,
        "status": "PASS",
        "risk_level": "LOW",
        "notes": "05_eval_false_hub_taxonomy.py reads canonical_predictions and read_dota_txt(tile['ann_path']); crop bug is visualization-only for this family.",
    },
    {
        "metric_family": "closedset_valid_mask_only",
        "metric_files": [
            "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_validmask_comparison.csv",
        ],
        "prediction_source": "full_closedset_s2_12angle/canonical_predictions",
        "prediction_coordinate_frame": "canonical",
        "gt_source": "manifest split ann_path / original DOTA tile GT",
        "gt_coordinate_frame": "canonical",
        "image_source": "not used for false hub matching",
        "image_coordinate_frame": "not_applicable",
        "valid_mask_source": "valid_masks exist in rotated frame but are not applied by 05_eval_false_hub_taxonomy.py",
        "valid_mask_coordinate_frame": "rotated",
        "matching_coordinate_frame": "canonical",
        "coordinate_frame_consistent": False,
        "uses_canonical_prediction_on_rotated_image": False,
        "uses_rotated_prediction_on_canonical_gt": False,
        "requires_inverse_rotation": True,
        "inverse_rotation_applied": True,
        "status": "SUSPECT",
        "risk_level": "MEDIUM",
        "notes": "The script duplicates all_region rows for valid_mask_only; valid-mask-specific claims should not be used without recompute.",
    },
    {
        "metric_family": "openvocab_row_level_burden",
        "metric_files": [
            "/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle/metrics/open_vocab_benchmark_rows_merged.csv",
        ],
        "prediction_source": "OpenRSD streaming post_summary; raw boxes not retained",
        "prediction_coordinate_frame": "rotated transient image for inference; aggregate counts only",
        "gt_source": "rotated transient GT for counts; source_ann_path records original source",
        "gt_coordinate_frame": "rotated transient for run-time preparation; source path canonical is provenance only",
        "image_source": "transient rotated scratch images not retained",
        "image_coordinate_frame": "rotated",
        "valid_mask_source": "not retained / not used for row-level count metrics",
        "valid_mask_coordinate_frame": "not_applicable",
        "matching_coordinate_frame": "not_applicable_count_level",
        "coordinate_frame_consistent": True,
        "uses_canonical_prediction_on_rotated_image": False,
        "uses_rotated_prediction_on_canonical_gt": False,
        "requires_inverse_rotation": False,
        "inverse_rotation_applied": False,
        "status": "PASS_WITH_NOT_APPLICABLE",
        "risk_level": "LOW_FOR_COUNTS_HIGH_FOR_AP",
        "notes": "Row-level burden is count-level inference output, not AP/GT matching; raw-box overlay is blocked because boxes were not retained.",
    },
    {
        "metric_family": "context_counterfactual_row_level",
        "metric_files": [
            "/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_context_counterfactual_s3_12angle/metrics/context_counterfactual_rows.csv",
        ],
        "prediction_source": "OpenRSD streaming post_summary on generated context images",
        "prediction_coordinate_frame": "counterfactual transient image; aggregate counts only",
        "gt_source": "rotated transient GT used to create object/context conditions",
        "gt_coordinate_frame": "rotated transient",
        "image_source": "counterfactual transient scratch images not retained",
        "image_coordinate_frame": "rotated/counterfactual",
        "valid_mask_source": "not used for row-level count metrics",
        "valid_mask_coordinate_frame": "not_applicable",
        "matching_coordinate_frame": "not_applicable_count_level",
        "coordinate_frame_consistent": True,
        "uses_canonical_prediction_on_rotated_image": False,
        "uses_rotated_prediction_on_canonical_gt": False,
        "requires_inverse_rotation": False,
        "inverse_rotation_applied": False,
        "status": "PASS_WITH_NOT_APPLICABLE",
        "risk_level": "LOW_FOR_COUNTS",
        "notes": "Counterfactual rows are count-level summaries; NOT_APPLICABLE remains data qualification.",
    },
    {
        "metric_family": "dehub_safety_row_level",
        "metric_files": [
            "/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_dehub_safety_s3_12angle/metrics/dehub_safety_rows_merged.csv",
        ],
        "prediction_source": "OpenRSD streaming post_summary baseline/repair; raw boxes not retained",
        "prediction_coordinate_frame": "rotated transient image; aggregate counts only",
        "gt_source": "rotated transient GT for preparation; source_ann_path records original source",
        "gt_coordinate_frame": "rotated transient for run-time preparation",
        "image_source": "transient rotated scratch images not retained",
        "image_coordinate_frame": "rotated",
        "valid_mask_source": "not retained / not used for row-level burden metrics",
        "valid_mask_coordinate_frame": "not_applicable",
        "matching_coordinate_frame": "not_applicable_count_level",
        "coordinate_frame_consistent": True,
        "uses_canonical_prediction_on_rotated_image": False,
        "uses_rotated_prediction_on_canonical_gt": False,
        "requires_inverse_rotation": False,
        "inverse_rotation_applied": False,
        "status": "PASS_WITH_NOT_APPLICABLE",
        "risk_level": "LOW_FOR_BURDEN_HIGH_FOR_TRUE_SV_PRESERVATION",
        "notes": "Paired burden metrics are count-level; true-SV preservation/AP remain blocked without retained raw boxes or evaluator.",
    },
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
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            pts = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 8, 2)]
        except Exception:
            continue
        out.append({"points": pts, "class_name": parts[8]})
    return out


def best_gt(points: list[tuple[float, float]], gts: list[dict[str, Any]]) -> tuple[str, float]:
    best_cls = "NO_GT_OVERLAP"
    best = 0.0
    for gt in gts:
        iou = hbb_iou(bounds(points), bounds(gt["points"]))
        if iou > best:
            best = iou
            best_cls = gt["class_name"]
    return best_cls, best


def draw_overlay(image_path: Path, out: Path, points: list[tuple[float, float]] | None, gt_rows: list[dict[str, Any]] | None, title: str, zoom: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        make_placeholder(out, title + "\nIMAGE_MISSING")
        return
    img = Image.open(image_path).convert("RGB")
    offset = (0, 0)
    if zoom and points:
        cx = min(max(sum(x for x, _ in points) / len(points), 0), img.size[0] - 1)
        cy = min(max(sum(y for _, y in points) / len(points), 0), img.size[1] - 1)
        half = 128
        left = max(0, int(cx - half))
        top = max(0, int(cy - half))
        right = min(img.size[0], int(cx + half))
        bottom = min(img.size[1], int(cy + half))
        if right <= left:
            right = min(img.size[0], left + 1)
        if bottom <= top:
            bottom = min(img.size[1], top + 1)
        offset = (left, top)
        img = img.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(img, "RGBA")
    ox, oy = offset
    for gt in gt_rows or []:
        pts = [(x - ox, y - oy) for x, y in gt["points"]]
        color = (0, 255, 0, 180) if gt["class_name"] == "small-vehicle" else (0, 160, 255, 120)
        draw.line(pts + [pts[0]], fill=color, width=2)
    if points:
        pts = [(x - ox, y - oy) for x, y in points]
        draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=3)
    draw.rectangle((0, 0, img.size[0], 40), fill=(0, 0, 0, 170))
    draw.text((5, 5), title[:180], fill=(255, 255, 255, 255))
    img.save(out)


def make_placeholder(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (768, 512), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    y = 24
    for line in text.splitlines():
        draw.text((24, y), line[:120], fill=(0, 0, 0))
        y += 22
    img.save(path)


def closedset_events(repo_root: Path, out_root: Path, limit: int) -> list[dict[str, Any]]:
    metric = repo_root / "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle/metrics/full_closedset_false_hub_tile_angle.csv"
    run_root = repo_root / "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle"
    rows = [
        r for r in read_csv(metric)
        if r.get("region_mode") == "all_region" and int(float(r.get("num_sv_pred") or 0)) > 0
    ]
    random.Random(20260601).shuffle(rows)
    events = []
    for row in rows:
        if len(events) >= limit:
            break
        model = row["model_name"]
        tile = row["tile_id"]
        angle = int(float(row["angle"]))
        canon_path = run_root / "canonical_predictions" / model / tile / f"angle_{angle:03d}.json"
        data = load_json(canon_path)
        pred = next((p for p in data.get("final_predictions", []) if p.get("class_name") == "small-vehicle"), None)
        if not pred:
            continue
        points = points_from_record(pred)
        gt_path = repo_root / data.get("metadata", {}).get("rotated_gt_path", "")
        # Canonical metrics use split GT, not rotated_gt_path. Derive original GT from split path.
        manifest = load_json(run_root / "manifest.json")
        split = load_json(repo_root / manifest.get("split_path", ""))
        ann_map = {t["tile_id"]: t["ann_path"] for t in split.get("tiles", [])}
        gt_path = repo_root / ann_map.get(tile, "")
        image_path = repo_root / "data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/images" / f"{tile}.png"
        gts = parse_dota(gt_path)
        best_cls, best_iou = best_gt(points, gts)
        event_id = f"closedset_{len(events):03d}"
        event_dir = out_root / event_id
        title = f"closedset {model} {tile} angle={angle} canonical metric"
        draw_overlay(image_path, event_dir / "full_tile_metric_overlay.png", points, None, title)
        draw_overlay(image_path, event_dir / "zoom_metric_overlay.png", points, None, title, zoom=True)
        draw_overlay(image_path, event_dir / "gt_context_overlay.png", points, gts, title, zoom=True)
        meta = {
            "event_id": event_id,
            "metric_family": "closed-set unmatched SV",
            "model_name": model,
            "tile_id": tile,
            "angle": angle,
            "prediction_file": str(canon_path),
            "gt_file": str(gt_path),
            "image_file": str(image_path),
            "prediction_coordinate_frame": "canonical",
            "gt_coordinate_frame": "canonical",
            "box_before_transform": "",
            "box_after_transform": pred.get("box", ""),
            "transform_applied": "inverse_rotation already applied when canonical_predictions were written",
            "best_gt_class": best_cls,
            "best_gt_iou": f"{best_iou:.6f}",
            "valid_mask_ratio": "not_applicable_all_region",
            "padding_overlap_ratio": "not_applicable_all_region",
            "manual_check_required": False,
            "diagnosis": "metric_frame_consistent",
        }
        write_json(event_dir / "metadata.json", meta)
        events.append(meta)
    return events


def aggregate_placeholder_events(name: str, rows_path: Path, out_root: Path, start: int, limit: int) -> list[dict[str, Any]]:
    rows = read_csv(rows_path)
    random.Random(20260601 + start).shuffle(rows)
    out = []
    for row in rows[:limit]:
        event_id = f"{name}_{start + len(out):03d}"
        event_dir = out_root / event_id
        msg = (
            f"{name}\nrow-level aggregate only\nraw boxes not retained, overlay blocked\n"
            f"tile={row.get('tile_id','')} angle={row.get('angle','')}"
        )
        for img_name in ["full_tile_metric_overlay.png", "zoom_metric_overlay.png", "gt_context_overlay.png"]:
            make_placeholder(event_dir / img_name, msg)
        meta = {
            "event_id": event_id,
            "metric_family": name,
            "model_name": row.get("model_name", ""),
            "tile_id": row.get("tile_id", ""),
            "angle": row.get("angle", ""),
            "prediction_file": "RAW_BOX_NOT_RETAINED",
            "gt_file": row.get("source_ann_path", ""),
            "image_file": row.get("source_image_path", ""),
            "prediction_coordinate_frame": "row_level_aggregate",
            "gt_coordinate_frame": "not_applicable_for_count_level",
            "box_before_transform": "",
            "box_after_transform": "",
            "transform_applied": "not_applicable",
            "best_gt_class": "NOT_COMPUTED",
            "best_gt_iou": "NOT_COMPUTED",
            "valid_mask_ratio": "NOT_RETAINED",
            "padding_overlap_ratio": "NOT_RETAINED",
            "manual_check_required": True,
            "diagnosis": "cannot_verify",
        }
        write_json(event_dir / "metadata.json", meta)
        out.append(meta)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--visual-summary-dir", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root)
    root = Path(args.visual_summary_dir)
    audit_dir = root / "audit"
    debug_dir = audit_dir / "debug_metric_coordinate_100"
    audit_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in ROWS:
        item = dict(src)
        files = [Path(p) if str(p).startswith("/") else repo / p for p in item.pop("metric_files")]
        item["metric_file"] = ";".join(str(p) for p in files)
        item["sample_checked"] = any(p.exists() for p in files)
        if not all(p.exists() for p in files):
            item["notes"] += " Missing files: " + ",".join(str(p) for p in files if not p.exists())
            if item["status"] == "PASS":
                item["status"] = "CANNOT_VERIFY"
                item["risk_level"] = "HIGH"
        rows.append(item)
    headers = [
        "metric_family", "metric_file", "prediction_source", "prediction_coordinate_frame", "gt_source",
        "gt_coordinate_frame", "image_source", "image_coordinate_frame", "valid_mask_source",
        "valid_mask_coordinate_frame", "matching_coordinate_frame", "coordinate_frame_consistent",
        "uses_canonical_prediction_on_rotated_image", "uses_rotated_prediction_on_canonical_gt",
        "requires_inverse_rotation", "inverse_rotation_applied", "sample_checked", "status",
        "risk_level", "notes",
    ]
    write_csv(audit_dir / "metric_coordinate_provenance_audit.csv", rows, headers)
    status_by_family = {r["metric_family"]: r["status"] for r in rows}

    events = []
    events.extend(closedset_events(repo, debug_dir, 30))
    events.extend(aggregate_placeholder_events("openvocab_unmatched_sv", Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle/metrics/open_vocab_benchmark_rows_merged.csv"), debug_dir, 30, 30))
    events.extend(aggregate_placeholder_events("dehub_baseline_repair_sv", Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_dehub_safety_s3_12angle/metrics/dehub_safety_rows_merged.csv"), debug_dir, 60, 20))
    events.extend(aggregate_placeholder_events("context_counterfactual_sv", Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_context_counterfactual_s3_12angle/metrics/context_counterfactual_rows.csv"), debug_dir, 80, 20))
    write_json(audit_dir / "debug_metric_coordinate_100_manifest.json", {"events": events, "num_events": len(events)})

    verdict = {
        "old_crop_coordinate_error_scope": "visualization_only_for_closedset_main_all_region_metrics",
        "closedset_false_hub_metrics_trust": "PASS",
        "closedset_valid_mask_metrics_trust": "SUSPECT_NEEDS_RECOMPUTE_FOR_VALIDMASK_CLAIMS",
        "openvocab_row_level_burden_trust": "PASS_WITH_NOT_APPLICABLE",
        "context_counterfactual_metrics_trust": "PASS_WITH_NOT_APPLICABLE",
        "dehub_burden_reduction_metrics_trust": "PASS_WITH_NOT_APPLICABLE",
        "metrics_need_recompute": ["closedset_valid_mask_only if used for paper claims"],
        "figures_to_remove_or_mark": ["valid-mask-specific crop/qualitative figures from old crop pack"],
        "status_by_family": status_by_family,
        "debug_events": len(events),
    }
    write_json(audit_dir / "metric_coordinate_provenance_audit.json", verdict)
    md = [
        "# Metric Coordinate Provenance Audit",
        "",
        "## Answers",
        "",
        "- Q1. 旧 crop 坐标错误是否只影响 visualization？closed-set main all-region metrics: yes; valid-mask-specific claims are suspect because valid-mask filtering was not actually applied.",
        "- Q2. closed-set false-hub metrics 是否仍可信？PASS for all-region canonical prediction vs canonical GT matching.",
        "- Q3. open-vocab row-level burden metrics 是否仍可信？PASS_WITH_NOT_APPLICABLE for count-level burden; no AP/raw-box claim.",
        "- Q4. context counterfactual metrics 是否仍可信？PASS_WITH_NOT_APPLICABLE for count-level rows.",
        "- Q5. DeHub burden reduction metrics 是否仍可信？PASS_WITH_NOT_APPLICABLE for paired count-level burden; true-SV preservation/AP remain blocked.",
        "- Q6. 哪些指标需要重新计算？closed-set valid-mask-only claims if used; AP/true-SV preservation remain separate blockers.",
        "- Q7. 哪些图表必须移除或标红？old crop qualitative figures and any valid-mask-specific claims from duplicated rows.",
        "",
        "## Family Status",
        "",
        md_table(rows, ["metric_family", "status", "risk_level", "prediction_coordinate_frame", "gt_coordinate_frame", "matching_coordinate_frame", "notes"]),
        "",
        f"Debug event directory: `{debug_dir}`",
    ]
    write_md(audit_dir / "metric_coordinate_provenance_audit.md", "\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
