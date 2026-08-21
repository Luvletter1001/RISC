#!/usr/bin/env python
"""Safe single-process DOTA raw-image splitter for 1024/500 rebuilds.

This is a conservative fallback for environments where the stock MMRotate
``tools/data/dota2/split/img_split.py`` crashes. It writes the same split
layout expected by MMRotate/OpenRSD:

    save_dir/images/<stem>__1024__x___y.png
    save_dir/annfiles/<stem>__1024__x___y.txt

The implementation intentionally avoids Shapely and multiprocessing. Polygon
clipping is Sutherland-Hodgman against the window rectangle, using IOF over the
original polygon area to match the DOTA split semantics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


Point = tuple[float, float]
Rect = tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--gap", type=int, default=500)
    parser.add_argument("--img-rate-thr", type=float, default=0.6)
    parser.add_argument("--iof-thr", type=float, default=0.7)
    parser.add_argument("--padding-rgb", nargs=3, type=int, default=[124, 116, 104])
    parser.add_argument("--save-ext", default=".png")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="Skip images larger than this many pixels. 0 means keep all.",
    )
    parser.add_argument(
        "--stem-list",
        default=None,
        help="Optional text file with one image stem per line.",
    )
    return parser.parse_args()


def polygon_area(poly: Sequence[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def _clip_edge(poly: Sequence[Point], inside, intersect) -> list[Point]:
    if not poly:
        return []
    out: list[Point] = []
    prev = poly[-1]
    prev_inside = inside(prev)
    for cur in poly:
        cur_inside = inside(cur)
        if cur_inside:
            if not prev_inside:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_inside:
            out.append(intersect(prev, cur))
        prev, prev_inside = cur, cur_inside
    return out


def clip_polygon_to_rect(poly: Sequence[Point], rect: Rect) -> list[Point]:
    x1, y1, x2, y2 = rect

    def ix_at_x(a: Point, b: Point, x: float) -> Point:
        ax, ay = a
        bx, by = b
        if bx == ax:
            return (x, ay)
        t = (x - ax) / (bx - ax)
        return (x, ay + t * (by - ay))

    def ix_at_y(a: Point, b: Point, y: float) -> Point:
        ax, ay = a
        bx, by = b
        if by == ay:
            return (ax, y)
        t = (y - ay) / (by - ay)
        return (ax + t * (bx - ax), y)

    clipped = list(poly)
    clipped = _clip_edge(clipped, lambda p: p[0] >= x1, lambda a, b: ix_at_x(a, b, x1))
    clipped = _clip_edge(clipped, lambda p: p[0] <= x2, lambda a, b: ix_at_x(a, b, x2))
    clipped = _clip_edge(clipped, lambda p: p[1] >= y1, lambda a, b: ix_at_y(a, b, y1))
    clipped = _clip_edge(clipped, lambda p: p[1] <= y2, lambda a, b: ix_at_y(a, b, y2))
    return clipped


def get_sliding_windows(
    width: int,
    height: int,
    size: int,
    gap: int,
    img_rate_thr: float,
) -> list[Rect]:
    step = size - gap
    if step <= 0:
        raise ValueError(f"size must be larger than gap, got {size}/{gap}")

    x_num = 1 if width <= size else math.ceil((width - size) / step + 1)
    y_num = 1 if height <= size else math.ceil((height - size) / step + 1)
    x_starts = [step * i for i in range(x_num)]
    y_starts = [step * i for i in range(y_num)]
    if len(x_starts) > 1 and x_starts[-1] + size > width:
        x_starts[-1] = width - size
    if len(y_starts) > 1 and y_starts[-1] + size > height:
        y_starts[-1] = height - size

    windows: list[Rect] = []
    rates: list[float] = []
    for x in x_starts:
        for y in y_starts:
            x2, y2 = x + size, y + size
            inside_w = max(0, min(x2, width) - max(x, 0))
            inside_h = max(0, min(y2, height) - max(y, 0))
            rate = (inside_w * inside_h) / float(size * size)
            windows.append((x, y, x2, y2))
            rates.append(rate)
    if not any(rate > img_rate_thr for rate in rates):
        best = max(rates)
        return [win for win, rate in zip(windows, rates) if abs(rate - best) < 0.01]
    return [win for win, rate in zip(windows, rates) if rate > img_rate_thr]


def load_dota_objects(path: Path) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    if not path.exists():
        return objects
    for line_no, raw in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        parts = raw.strip().split()
        if not parts or parts[0].startswith(("imagesource", "gsd")):
            continue
        if len(parts) < 9:
            raise ValueError(f"Invalid DOTA line in {path}:{line_no}: {raw}")
        coords = [float(v) for v in parts[:8]]
        objects.append(
            {
                "poly": list(zip(coords[0::2], coords[1::2])),
                "label": parts[8],
                "diff": parts[9] if len(parts) > 9 else "0",
            }
        )
    return objects


def project_objects_to_window(
    objects: Iterable[dict[str, object]],
    window: Rect,
    iof_thr: float,
) -> list[str]:
    x1, y1, x2, y2 = window
    lines: list[str] = []
    for obj in objects:
        poly = obj["poly"]
        assert isinstance(poly, list)
        area = polygon_area(poly)
        if area <= 0:
            continue
        clipped = clip_polygon_to_rect(poly, window)
        clipped_area = polygon_area(clipped)
        if clipped_area / area < iof_thr or len(clipped) < 3:
            continue

        # Match the MMRotate/BboxToolkit DOTA split behavior: the IOF test
        # decides whether to keep an object, but the written annotation remains
        # the original oriented polygon translated into patch coordinates.
        translated = [(px - x1, py - y1) for px, py in poly]
        coords = " ".join(f"{v:.1f}" for point in translated for v in point)
        diff = str(obj["diff"]) if clipped_area >= area - 1e-3 else "2"
        lines.append(f"{coords} {obj['label']} {diff}")
    return lines


def crop_with_padding(img: Image.Image, window: Rect, size: int, padding_rgb: Sequence[int]) -> Image.Image:
    x1, y1, x2, y2 = window
    crop_box = (max(x1, 0), max(y1, 0), min(x2, img.width), min(y2, img.height))
    tile = Image.new("RGB", (size, size), tuple(padding_rgb))
    crop = img.crop(crop_box).convert("RGB")
    tile.paste(crop, (max(0, -x1), max(0, -y1)))
    return tile


def split_dataset(args: argparse.Namespace) -> dict[str, object]:
    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    save_dir = Path(args.save_dir)
    if save_dir.exists():
        raise FileExistsError(f"{save_dir} already exists")
    image_out = save_dir / "images"
    ann_out = save_dir / "annfiles"
    image_out.mkdir(parents=True)
    ann_out.mkdir(parents=True)

    allowed_stems = None
    if args.stem_list:
        allowed_stems = {
            line.strip()
            for line in Path(args.stem_list).read_text().splitlines()
            if line.strip()
        }

    img_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if allowed_stems is not None:
        img_paths = [p for p in img_paths if p.stem in allowed_stems]
    if args.limit > 0:
        img_paths = img_paths[: args.limit]

    done = skipped_large = total_tiles = total_objects = 0
    skipped: list[str] = []
    for img_path in img_paths:
        with Image.open(img_path) as img:
            pixels = img.width * img.height
            if args.max_pixels and pixels > args.max_pixels:
                skipped_large += 1
                skipped.append(img_path.name)
                continue
            objects = load_dota_objects(ann_dir / f"{img_path.stem}.txt")
            windows = get_sliding_windows(
                img.width,
                img.height,
                args.size,
                args.gap,
                args.img_rate_thr,
            )
            for window in windows:
                x1, y1, x2, y2 = window
                tile_id = f"{img_path.stem}__{x2 - x1}__{x1}___{y1}"
                tile = crop_with_padding(img, window, args.size, args.padding_rgb)
                tile.save(image_out / f"{tile_id}{args.save_ext}")
                lines = project_objects_to_window(objects, window, args.iof_thr)
                (ann_out / f"{tile_id}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
                total_tiles += 1
                total_objects += len(lines)
            done += 1
            print(
                json.dumps(
                    {
                        "image": img_path.name,
                        "width": img.width,
                        "height": img.height,
                        "objects": len(objects),
                        "tiles": len(windows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "img_dir": str(img_dir),
        "ann_dir": str(ann_dir),
        "save_dir": str(save_dir),
        "processed_images": done,
        "skipped_large_images": skipped_large,
        "skipped_large_image_names": skipped,
        "total_tiles": total_tiles,
        "total_objects_in_tiles": total_objects,
        "size": args.size,
        "gap": args.gap,
        "iof_thr": args.iof_thr,
        "img_rate_thr": args.img_rate_thr,
    }
    (save_dir / "safe_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    split_dataset(parse_args())


if __name__ == "__main__":
    main()
