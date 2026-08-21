from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..dota_io import read_dota_txt
from ..constants import SMALL_VEHICLE


def class_polygons(ann_path: str | Path, class_name: str = SMALL_VEHICLE) -> list[list[tuple[float, float]]]:
    polygons = []
    for record in read_dota_txt(ann_path):
        if record.get("class_name") != class_name or "polygon" not in record:
            continue
        polygons.append([(float(x), float(y)) for x, y in record["polygon"]])
    return polygons


def polygon_mask(size: tuple[int, int], polygons: list[list[tuple[float, float]]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return mask


def image_median_color(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    median = np.median(arr.reshape(-1, 3), axis=0)
    return tuple(int(v) for v in median)


def write_context_counterfactual_images(
    image_path: str | Path,
    ann_path: str | Path,
    output_dir: str | Path,
    *,
    target_class: str = SMALL_VEHICLE,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    polygons = class_polygons(ann_path, target_class)
    if not polygons:
        return {
            "target_class": target_class,
            "num_target_polygons": 0,
            "conditions": {},
        }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = polygon_mask(image.size, polygons)
    fill = image_median_color(image)

    object_only = Image.new("RGB", image.size, fill)
    object_only.paste(image, mask=mask)
    object_only_path = out_dir / "object_only.png"
    object_only.save(object_only_path)

    context_only = image.copy()
    draw = ImageDraw.Draw(context_only)
    for polygon in polygons:
        draw.polygon(polygon, fill=fill)
    context_only_path = out_dir / "context_only.png"
    context_only.save(context_only_path)

    mask_path = out_dir / "target_mask.png"
    mask.save(mask_path)
    return {
        "target_class": target_class,
        "num_target_polygons": len(polygons),
        "mask_path": str(mask_path),
        "conditions": {
            "object_only": str(object_only_path),
            "context_only": str(context_only_path),
        },
    }
