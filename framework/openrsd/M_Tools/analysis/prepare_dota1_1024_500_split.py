#!/usr/bin/env python3
"""Prepare a DOTA1 1024/500 split without relying on cv2 multiprocessing."""

from __future__ import annotations

import argparse
import math
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
    parser.add_argument('--img-dir', type=Path, required=True)
    parser.add_argument('--ann-dir', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--gap', type=int, default=500)
    parser.add_argument('--img-rate-thr', type=float, default=0.6)
    parser.add_argument('--iof-thr', type=float, default=0.7)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--use-shapely',
        action='store_true',
        help='Use polygon/window IOF via shapely. Default uses center-point inclusion to avoid binary crashes.')
    return parser.parse_args()


def load_ann(path: Path) -> list[Obj]:
    objs = []
    if not path.exists():
        return objs
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('imagesource') or parts[0].startswith('gsd'):
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


def save_patch(img: Image.Image,
               objs: list[Obj],
               window: tuple[int, int, int, int],
               patch_id: str,
               out_img_dir: Path,
               out_ann_dir: Path,
               size: int,
               iof_thr: float) -> tuple[int, int]:
    x1, y1, x2, y2 = window
    crop = img.crop((x1, y1, min(x2, img.width), min(y2, img.height)))
    if crop.size != (size, size):
        canvas = Image.new(img.mode, (size, size), 0)
        canvas.paste(crop, (0, 0))
        crop = canvas
    crop.save(out_img_dir / f'{patch_id}.png')

    kept = []
    for obj in objs:
        iof = object_iof(obj.poly, window)
        if iof >= iof_thr:
            diff = obj.difficult if iof >= 0.999 else 2
            kept.append((translate(obj.poly, x1, y1), obj.cls_name, diff))
    with (out_ann_dir / f'{patch_id}.txt').open('w', encoding='utf-8') as f:
        for poly, cls_name, diff in kept:
            f.write(' '.join(f'{float(v):.1f}' for v in poly.tolist()) + f' {cls_name} {diff}\n')
    return 1, len(kept)


def main() -> None:
    args = parse_args()
    global Polygon, shapely_box
    if args.use_shapely:
        from shapely.geometry import Polygon as _Polygon
        from shapely.geometry import box as _box
        Polygon = _Polygon
        shapely_box = _box
    if args.save_dir.exists() and any(args.save_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{args.save_dir} exists; pass --overwrite to reuse it')
    out_img_dir = args.save_dir / 'images'
    out_ann_dir = args.save_dir / 'annfiles'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in args.img_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}])
    if args.limit > 0:
        img_paths = img_paths[:args.limit]
    total_patches = 0
    total_objs = 0
    for idx, img_path in enumerate(img_paths, 1):
        img_id = img_path.stem
        objs = load_ann(args.ann_dir / f'{img_id}.txt')
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            windows = windows_for(img.width, img.height, args.size, args.gap, args.img_rate_thr)
            for x1, y1, x2, _y2 in windows:
                patch_id = f'{img_id}__{args.size}__{x1}___{y1}'
                patch_count, obj_count = save_patch(
                    img, objs, (x1, y1, x2, y1 + args.size), patch_id,
                    out_img_dir, out_ann_dir, args.size, args.iof_thr)
                total_patches += patch_count
                total_objs += obj_count
        if idx % 25 == 0 or idx == len(img_paths):
            print(f'{idx}/{len(img_paths)} images, patches={total_patches}, objects={total_objs}', flush=True)
    print(f'Finished: images={len(img_paths)}, patches={total_patches}, objects={total_objs}')
    print(f'Output: {args.save_dir}')


if __name__ == '__main__':
    main()
