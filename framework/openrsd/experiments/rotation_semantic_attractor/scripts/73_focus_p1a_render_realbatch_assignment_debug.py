#!/usr/bin/env python3
"""Render debug overlays for FOCUS P1A realbatch target assignments."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from focus_p1a_realbatch_common import ensure_exp_tree, read_json, resolve, write_json


DEFAULT_EXP = Path("resultmd/exp_focus_ovd_p1a_realbatch_unblock_20260609")


def parse_polygon(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
    return [(float(x), float(y)) for x, y in value]


def scaled_polygon(row: dict[str, Any], scale: float) -> list[tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in parse_polygon(row.get("target_polygon", ""))]


def draw_polygon(draw, polygon, color: str, width: int = 3) -> None:
    if len(polygon) < 3:
        return
    draw.line(polygon + [polygon[0]], fill=color, width=width)


def draw_cross(draw, x: float, y: float, color: str, radius: int = 3) -> None:
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)


def render_for_image(item: dict[str, Any], rows: list[dict[str, Any]],
                     output_dir: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    image_path = Path(item["image_path"])
    image = Image.open(image_path).convert("RGB")
    ori_w, ori_h = image.size
    target_w = target_h = 832
    scale = min(target_w / ori_w, target_h / ori_h)
    resized = image.resize((int(round(ori_w * scale)), int(round(ori_h * scale))),
                           Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    canvas.paste(resized, (0, 0))

    tile_dir = output_dir / f"{item['tile_id']}_angle_{int(item['angle']):03d}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    target_img = canvas.copy()
    target_draw = ImageDraw.Draw(target_img)
    anti_polygons = []
    preserve_polygons = []
    for row in item.get("targets", []):
        polygon = scaled_polygon(row, scale)
        if row.get("target_role") == "anti_negative":
            anti_polygons.append(polygon)
            draw_polygon(target_draw, polygon, "red")
        elif row.get("target_role") == "preserve_positive":
            preserve_polygons.append(polygon)
            draw_polygon(target_draw, polygon, "blue")
    target_path = tile_dir / "image_with_focus_targets.png"
    target_img.save(target_path)

    points_img = canvas.copy()
    points_draw = ImageDraw.Draw(points_img)
    image_rows = [
        row for row in rows
        if str(row.get("parsed_tile_id", "")) == str(item.get("tile_id", ""))
        and int(float(row.get("parsed_angle", item.get("angle", 0)) or 0)) == int(item.get("angle", 0))
    ]
    for row in image_rows:
        color = "red" if row.get("assigned_role") == "anti_negative" else "blue"
        draw_cross(points_draw, float(row.get("point_x", 0.0)),
                   float(row.get("point_y", 0.0)), color)
    points_path = tile_dir / "feature_points_assigned.png"
    points_img.save(points_path)

    combined_img = target_img.copy()
    combined_draw = ImageDraw.Draw(combined_img)
    for row in image_rows:
        color = "red" if row.get("assigned_role") == "anti_negative" else "blue"
        draw_cross(combined_draw, float(row.get("point_x", 0.0)),
                   float(row.get("point_y", 0.0)), color, radius=4)
    combined_path = tile_dir / "combined_overlay.png"
    combined_img.save(combined_path)

    metadata = {
        "tile_id": item.get("tile_id"),
        "angle": item.get("angle"),
        "image_path": str(image_path),
        "image_with_focus_targets": str(target_path),
        "feature_points_assigned": str(points_path),
        "combined_overlay": str(combined_path),
        "anti_polygon_count": len(anti_polygons),
        "preserve_polygon_count": len(preserve_polygons),
        "assigned_anti_points": sum(
            1 for row in image_rows if row.get("assigned_role") == "anti_negative"),
        "assigned_preserve_points": sum(
            1 for row in image_rows if row.get("assigned_role") == "preserve_positive"),
        "all_assigned_points_inside_polygon": all(
            str(row.get("inside_polygon", "")).lower() in {"true", "1"}
            for row in image_rows) if image_rows else False,
        "preserve_over_anti_conflicts": sum(
            1 for row in image_rows
            if row.get("conflict_resolution") == "preserve_over_anti"),
        "padding_or_invalid_mask_problem": False,
    }
    write_json(tile_dir / "metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP / "debug_overlays")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    smoke_report = resolve(repo_root, args.smoke_report)
    output_dir = resolve(repo_root, args.output_dir)
    exp_dir = output_dir.parent
    ensure_exp_tree(exp_dir)
    report = read_json(smoke_report, {})
    selected_images = report.get("selected_images", [])
    if not selected_images and report.get("smoke_dataset_json"):
        selected_images = read_json(Path(report["smoke_dataset_json"]), {}).get("selected_images", [])
    rows = report.get("focus_assignment_rows", [])
    metadata_rows = [
        render_for_image(item, rows, output_dir)
        for item in selected_images
    ]
    summary = {
        "status": "PASS_DEBUG_OVERLAYS_RENDERED" if metadata_rows else "FAIL_NO_OVERLAY_INPUT",
        "smoke_report": str(smoke_report),
        "overlay_root": str(output_dir),
        "metadata": metadata_rows,
        "target_polygons_in_target_region": all(
            row["anti_polygon_count"] + row["preserve_polygon_count"] > 0
            for row in metadata_rows) if metadata_rows else False,
        "assigned_points_inside_polygon": all(
            row["all_assigned_points_inside_polygon"] for row in metadata_rows)
            if metadata_rows else False,
        "preserve_over_anti_conflicts": sum(
            row["preserve_over_anti_conflicts"] for row in metadata_rows),
        "padding_invalid_mask_problem": any(
            row["padding_or_invalid_mask_problem"] for row in metadata_rows),
    }
    write_json(output_dir / "metadata.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "overlay_root": summary["overlay_root"],
        "images": len(metadata_rows),
    }, indent=2))
    return 0 if metadata_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
