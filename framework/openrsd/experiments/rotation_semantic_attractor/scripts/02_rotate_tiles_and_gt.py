#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from common import add_common_args, parse_angles, read_jsonish
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt, records_to_polygons, write_dota_txt
from experiments.rotation_semantic_attractor.src.rotation import rotate_image_and_polygons


def rotate_split(split_path: Path, out_dir: Path, angles: list[int], limit: int = 0) -> None:
    split = read_jsonish(split_path)
    tiles = split["tiles"][: limit or None]
    for tile in tiles:
        image = Image.open(tile["image_path"]).convert("RGB")
        records = read_dota_txt(tile["ann_path"])
        polygons = records_to_polygons(records)
        for angle in angles:
            base = out_dir / tile["tile_id"] / f"angle_{angle:03d}"
            mask_path = base / "valid_mask.png"
            result = rotate_image_and_polygons(image, polygons, angle_deg=angle, expand=True, output_mask_path=mask_path)
            (base / "images").mkdir(parents=True, exist_ok=True)
            (base / "annfiles").mkdir(parents=True, exist_ok=True)
            result.image.save(base / "images" / f"{tile['tile_id']}_rot{angle:03d}.png")
            rotated_records = []
            for rec, poly in zip(records, result.polygons):
                new_rec = dict(rec)
                new_rec["polygon"] = poly
                rotated_records.append(new_rec)
            write_dota_txt(base / "annfiles" / f"{tile['tile_id']}_rot{angle:03d}.txt", rotated_records)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    angles = parse_angles(args.angles, default=[0, 90])
    out_dir = Path(args.output_dir or Path(args.split).parent / "rotated_tiles")
    if not args.dry_run:
        rotate_split(Path(args.split), out_dir, angles, limit=args.limit)
    print(f"rotated_output={out_dir}")


if __name__ == "__main__":
    main()
