#!/usr/bin/env python
"""Build DOTA annfiles from uploaded tile names and raw DOTA v2 labels.

The uploaded OpenRSD DOTA2 train package already contains sliced image tiles,
but not DOTA-style ``annfiles``. This script reconstructs tile annotations by
projecting raw DOTA polygons into each tile window.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


Point = tuple[float, float]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class TileInfo:
    source_id: str
    tile_size: int
    x: int
    y: int
    stem: str


@dataclass(frozen=True)
class DotaObject:
    poly: list[Point]
    label: str
    difficulty: str


TILE_RE = re.compile(r"^(?P<src>.+?)__(?P<size>\d+)__(?P<x>\d+)___(?P<y>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-label-zip", required=True)
    parser.add_argument("--tile-list", required=True)
    parser.add_argument("--out-ann-dir", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--iof-thr", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_tile_stem(stem: str) -> TileInfo:
    match = TILE_RE.match(stem)
    if not match:
        raise ValueError(f"Invalid DOTA tile stem: {stem}")
    return TileInfo(
        source_id=match.group("src"),
        tile_size=int(match.group("size")),
        x=int(match.group("x")),
        y=int(match.group("y")),
        stem=stem,
    )


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
        prev = cur
        prev_inside = cur_inside
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


def load_raw_labels(raw_label_zip: Path) -> dict[str, list[DotaObject]]:
    labels: dict[str, list[DotaObject]] = {}
    with zipfile.ZipFile(raw_label_zip) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.endswith(".txt"):
                continue
            stem = Path(member).stem
            text = zf.read(member).decode("utf-8", errors="ignore")
            objects: list[DotaObject] = []
            for line_no, raw in enumerate(text.splitlines(), 1):
                parts = raw.strip().split()
                if not parts or parts[0].startswith(("imagesource", "gsd")):
                    continue
                if len(parts) < 9:
                    raise ValueError(f"Invalid DOTA line in {member}:{line_no}: {raw}")
                coords = [float(v) for v in parts[:8]]
                objects.append(
                    DotaObject(
                        poly=list(zip(coords[0::2], coords[1::2])),
                        label=parts[8],
                        difficulty=parts[9] if len(parts) > 9 else "0",
                    )
                )
            labels[stem] = objects
    return labels


def _fmt_num(value: float) -> str:
    rounded = round(float(value), 6)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def project_objects_to_tile(
    objects: Iterable[DotaObject],
    tile: TileInfo,
    *,
    base_size: int = 1024,
    iof_thr: float = 0.7,
) -> list[str]:
    scale = tile.tile_size / float(base_size)
    window = (float(tile.x), float(tile.y), float(tile.x + tile.tile_size), float(tile.y + tile.tile_size))
    lines: list[str] = []
    for obj in objects:
        scaled = [(x * scale, y * scale) for x, y in obj.poly]
        area = polygon_area(scaled)
        if area <= 0:
            continue
        clipped = clip_polygon_to_rect(scaled, window)
        clipped_area = polygon_area(clipped)
        if len(clipped) < 3 or clipped_area / area < iof_thr:
            continue

        translated = [(x - tile.x, y - tile.y) for x, y in scaled]
        coords = " ".join(_fmt_num(v) for point in translated for v in point)
        difficulty = obj.difficulty if clipped_area >= area - 1e-3 else "2"
        lines.append(f"{coords} {obj.label} {difficulty}")
    return lines


def read_tile_list(tile_list: Path, tile_size: int) -> list[TileInfo]:
    tiles: list[TileInfo] = []
    for raw in tile_list.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item:
            continue
        stem = Path(item).stem
        tile = parse_tile_stem(stem)
        if tile_size > 0 and tile.tile_size != tile_size:
            continue
        tiles.append(tile)
    return tiles


def build_annfiles(args: argparse.Namespace) -> dict[str, object]:
    raw_label_zip = Path(args.raw_label_zip)
    tile_list = Path(args.tile_list)
    out_ann_dir = Path(args.out_ann_dir)
    if out_ann_dir.exists() and any(out_ann_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_ann_dir} is not empty; pass --overwrite")
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    labels = load_raw_labels(raw_label_zip)
    tiles = read_tile_list(tile_list, args.tile_size)
    if args.limit > 0:
        tiles = tiles[: args.limit]

    class_counts: Counter[str] = Counter()
    missing_sources: Counter[str] = Counter()
    nonempty_tiles = 0
    object_count = 0
    for index, tile in enumerate(tiles, 1):
        objects = labels.get(tile.source_id)
        if objects is None:
            missing_sources[tile.source_id] += 1
            lines: list[str] = []
        else:
            lines = project_objects_to_tile(
                objects,
                tile,
                base_size=args.base_size,
                iof_thr=args.iof_thr,
            )
        if lines:
            nonempty_tiles += 1
            object_count += len(lines)
            for line in lines:
                class_counts[line.split()[8]] += 1
        (out_ann_dir / f"{tile.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        if index % 5000 == 0:
            print(json.dumps({"processed_tiles": index, "objects": object_count}, ensure_ascii=False), flush=True)

    summary = {
        "raw_label_zip": str(raw_label_zip),
        "tile_list": str(tile_list),
        "out_ann_dir": str(out_ann_dir),
        "tile_size": args.tile_size,
        "base_size": args.base_size,
        "iof_thr": args.iof_thr,
        "tiles": len(tiles),
        "nonempty_tiles": nonempty_tiles,
        "empty_tiles": len(tiles) - nonempty_tiles,
        "object_count": object_count,
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "missing_source_count": len(missing_sources),
        "missing_source_tile_count": int(sum(missing_sources.values())),
        "missing_source_examples": dict(missing_sources.most_common(20)),
    }
    if args.summary_path:
        Path(args.summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    build_annfiles(parse_args())


if __name__ == "__main__":
    main()
