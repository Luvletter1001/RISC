#!/usr/bin/env python3
"""Build DOTA1 1024/500 split annfiles without writing cropped images.

This is a lightweight companion to ``prepare_dota1_1024_500_split.py`` for
offline audits that already have tile-level predictions. It preserves the same
tile naming and object inclusion logic, but writes only ``annfiles/*.txt``.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Polygon = None
shapely_box = None


@dataclass
class Obj:
    poly: np.ndarray
    cls_name: str
    difficult: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--prediction-pkl", type=Path, default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--gap", type=int, default=500)
    parser.add_argument("--img-rate-thr", type=float, default=0.6)
    parser.add_argument("--iof-thr", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-shapely",
        action="store_true",
        help="Use polygon/window IOF via shapely instead of center inclusion.")
    return parser.parse_args()


def canonical_img_id(img_id: object) -> str:
    text = str(img_id)
    if text.startswith("angle_") and "__" in text:
        return text.split("__", 1)[1]
    return text


def load_prediction_tile_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("rb") as f:
        predictions = pickle.load(f)
    out = set()
    for sample in predictions:
        if isinstance(sample, dict):
            img_id = sample.get("img_id")
        else:
            img_id = getattr(sample, "img_id", None)
            if img_id is None:
                metainfo = getattr(sample, "metainfo", {}) or {}
                img_id = metainfo.get("img_id")
        if img_id is not None:
            out.add(canonical_img_id(img_id))
    return out


def parse_tile_id(tile_id: str) -> tuple[str, int, int] | None:
    parts = tile_id.split("__")
    if len(parts) != 4:
        return None
    stem, size_text, x_text, y_text = parts
    try:
        int(size_text)
        x = int(x_text)
        y = int(y_text.lstrip("_"))
    except ValueError:
        return None
    return stem, x, y


def group_requested_tiles(tile_ids: set[str] | None) -> dict[str, set[str]] | None:
    if tile_ids is None:
        return None
    grouped: dict[str, set[str]] = {}
    for tile_id in tile_ids:
        parsed = parse_tile_id(tile_id)
        if parsed is None:
            continue
        stem, _x, _y = parsed
        grouped.setdefault(stem, set()).add(tile_id)
    return grouped


def load_ann(path: Path) -> list[Obj]:
    objs = []
    if not path.exists():
        return objs
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("imagesource") or parts[0].startswith("gsd"):
                continue
            if len(parts) < 9:
                continue
            diff = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0
            objs.append(Obj(np.asarray([float(x) for x in parts[:8]], dtype=np.float32), parts[8], diff))
    return objs


def windows_for(width: int, height: int, size: int, gap: int, img_rate_thr: float) -> list[tuple[int, int, int, int]]:
    step = size - gap
    x_num = 1 if width <= size else math.ceil((width - size) / step + 1)
    y_num = 1 if height <= size else math.ceil((height - size) / step + 1)
    xs = [step * i for i in range(x_num)]
    ys = [step * i for i in range(y_num)]
    if len(xs) > 1 and xs[-1] + size > width:
        xs[-1] = width - size
    if len(ys) > 1 and ys[-1] + size > height:
        ys[-1] = height - size
    out = []
    for x in xs:
        for y in ys:
            x2, y2 = x + size, y + size
            ix1, iy1 = max(0, x), max(0, y)
            ix2, iy2 = min(width, x2), min(height, y2)
            rate = max(0, ix2 - ix1) * max(0, iy2 - iy1) / float(size * size)
            if rate >= img_rate_thr:
                out.append((int(x), int(y), int(x2), int(y2)))
    if not out:
        out.append((0, 0, size, size))
    return out


def object_iof(poly: np.ndarray, window: tuple[int, int, int, int]) -> float:
    pts = poly.reshape(-1, 2)
    x1, y1, x2, y2 = window
    if Polygon is not None and shapely_box is not None:
        polygon = Polygon(pts)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            return 0.0
        inter = polygon.intersection(shapely_box(x1, y1, x2, y2)).area
        return float(inter / max(polygon.area, 1e-6))
    cx, cy = pts.mean(axis=0)
    return 1.0 if x1 <= cx <= x2 and y1 <= cy <= y2 else 0.0


def translate(poly: np.ndarray, x: int, y: int) -> np.ndarray:
    return poly + np.asarray([-x, -y] * 4, dtype=np.float32)


def write_ann(
    objs: list[Obj],
    window: tuple[int, int, int, int],
    patch_id: str,
    out_ann_dir: Path,
    iof_thr: float,
) -> int:
    x1, y1, _x2, _y2 = window
    kept = []
    for obj in objs:
        iof = object_iof(obj.poly, window)
        if iof >= iof_thr:
            diff = obj.difficult if iof >= 0.999 else 2
            kept.append((translate(obj.poly, x1, y1), obj.cls_name, diff))
    with (out_ann_dir / f"{patch_id}.txt").open("w", encoding="utf-8") as f:
        for poly, cls_name, diff in kept:
            f.write(" ".join(f"{float(v):.1f}" for v in poly.tolist()) + f" {cls_name} {diff}\n")
    return len(kept)


def main() -> None:
    args = parse_args()
    global Polygon, shapely_box
    if args.use_shapely:
        from shapely.geometry import Polygon as _Polygon
        from shapely.geometry import box as _box

        Polygon = _Polygon
        shapely_box = _box

    requested_tiles = load_prediction_tile_ids(args.prediction_pkl)
    requested_by_image = group_requested_tiles(requested_tiles)
    out_ann_dir = args.save_dir / "annfiles"
    if out_ann_dir.exists() and any(out_ann_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_ann_dir} exists; pass --overwrite to reuse it")
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([
        p for p in args.img_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    ])
    if requested_by_image is not None:
        img_paths = [p for p in img_paths if p.stem in requested_by_image]
    if args.limit > 0:
        img_paths = img_paths[:args.limit]

    generated_tiles: set[str] = set()
    total_objs = 0
    for idx, img_path in enumerate(img_paths, 1):
        img_id = img_path.stem
        objs = load_ann(args.ann_dir / f"{img_id}.txt")
        allowed = requested_by_image.get(img_id) if requested_by_image is not None else None
        if allowed is not None:
            parsed_windows = []
            for patch_id in allowed:
                parsed = parse_tile_id(patch_id)
                if parsed is None:
                    continue
                _stem, x1, y1 = parsed
                parsed_windows.append((x1, y1, x1 + args.size, y1 + args.size))
            windows = sorted(parsed_windows, key=lambda item: (item[0], item[1]))
        else:
            with Image.open(img_path) as img:
                windows = windows_for(img.width, img.height, args.size, args.gap, args.img_rate_thr)
        for x1, y1, x2, _y2 in windows:
            patch_id = f"{img_id}__{args.size}__{x1}___{y1}"
            if allowed is not None and patch_id not in allowed:
                continue
            obj_count = write_ann(
                objs,
                (x1, y1, x2, y1 + args.size),
                patch_id,
                out_ann_dir,
                args.iof_thr,
            )
            generated_tiles.add(patch_id)
            total_objs += obj_count
        if idx % 50 == 0 or idx == len(img_paths):
            print(f"{idx}/{len(img_paths)} images, annfiles={len(generated_tiles)}, objects={total_objs}", flush=True)

    missing_tiles = sorted((requested_tiles or set()) - generated_tiles)
    summary = {
        "img_dir": str(args.img_dir),
        "ann_dir": str(args.ann_dir),
        "save_dir": str(args.save_dir),
        "prediction_pkl": str(args.prediction_pkl) if args.prediction_pkl else None,
        "size": args.size,
        "gap": args.gap,
        "img_rate_thr": args.img_rate_thr,
        "iof_thr": args.iof_thr,
        "use_shapely": args.use_shapely,
        "raw_images_scanned": len(img_paths),
        "requested_tiles": len(requested_tiles) if requested_tiles is not None else None,
        "generated_annfiles": len(generated_tiles),
        "objects_written": total_objs,
        "missing_requested_tiles": len(missing_tiles),
        "missing_requested_tiles_sample": missing_tiles[:50],
    }
    summary_path = args.save_dir / "annfiles_only_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
