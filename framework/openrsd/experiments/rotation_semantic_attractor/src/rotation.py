from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image


@dataclass
class RotationTransform:
    angle_deg: float
    original_size: tuple[int, int]
    rotated_size: tuple[int, int]
    forward: np.ndarray
    inverse: np.ndarray
    expand: bool


@dataclass
class RotationResult:
    image: Image.Image
    polygons: List[List[List[float]]]
    valid_mask: Image.Image
    transform: RotationTransform


def _rotation_matrix(width: int, height: int, angle_deg: float, expand: bool) -> tuple[np.ndarray, tuple[int, int]]:
    angle = math.radians(angle_deg % 360)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # PIL-style counter-clockwise rotation in image coordinates.
    base = np.array(
        [
            [cos_a, sin_a, cx - cos_a * cx - sin_a * cy],
            [-sin_a, cos_a, cy + sin_a * cx - cos_a * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if not expand:
        return base, (width, height)

    corners = np.array(
        [[0.0, 0.0, 1.0], [width - 1.0, 0.0, 1.0], [width - 1.0, height - 1.0, 1.0], [0.0, height - 1.0, 1.0]],
        dtype=np.float64,
    ).T
    rotated = base @ corners
    min_x, min_y = rotated[0].min(), rotated[1].min()
    max_x, max_y = rotated[0].max(), rotated[1].max()
    new_w = int(math.ceil(max_x - min_x + 1.0))
    new_h = int(math.ceil(max_y - min_y + 1.0))
    translate = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    return translate @ base, (new_w, new_h)


def build_rotation_transform(width: int, height: int, angle_deg: float, expand: bool = True) -> RotationTransform:
    forward, rotated_size = _rotation_matrix(width, height, angle_deg, expand)
    return RotationTransform(
        angle_deg=float(angle_deg % 360),
        original_size=(int(width), int(height)),
        rotated_size=rotated_size,
        forward=forward,
        inverse=np.linalg.inv(forward),
        expand=expand,
    )


def apply_transform_to_polygon(polygon: Sequence[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((arr.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([arr, ones], axis=1).T
    out = (matrix @ homogeneous).T
    return out[:, :2]


def rotate_image_and_polygons(
    image: Image.Image,
    polygons: Sequence[Sequence[Sequence[float]]],
    angle_deg: float,
    expand: bool = True,
    fill_color: tuple[int, int, int] = (0, 0, 0),
    output_mask_path: str | Path | None = None,
) -> RotationResult:
    transform = build_rotation_transform(image.width, image.height, angle_deg, expand=expand)
    rotated_image = image.rotate(angle_deg, resample=Image.BILINEAR, expand=expand, fillcolor=fill_color)
    valid_mask = Image.new("L", image.size, color=255)
    rotated_mask = valid_mask.rotate(angle_deg, resample=Image.NEAREST, expand=expand, fillcolor=0)

    rotated_polygons = [
        apply_transform_to_polygon(poly, transform.forward).tolist()
        for poly in polygons
    ]

    if output_mask_path is not None:
        out = Path(output_mask_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rotated_mask.save(out)

    return RotationResult(
        image=rotated_image,
        polygons=rotated_polygons,
        valid_mask=rotated_mask,
        transform=transform,
    )

