from pathlib import Path

import numpy as np
from PIL import Image

from experiments.rotation_semantic_attractor.src.rotation import (
    apply_transform_to_polygon,
    build_rotation_transform,
    rotate_image_and_polygons,
)


def test_polygon_round_trip_expand_canvas():
    polygon = np.array(
        [[100.0, 100.0], [220.0, 100.0], [220.0, 180.0], [100.0, 180.0]],
        dtype=np.float64,
    )

    transform = build_rotation_transform(width=512, height=384, angle_deg=37, expand=True)
    rotated = apply_transform_to_polygon(polygon, transform.forward)
    restored = apply_transform_to_polygon(rotated, transform.inverse)

    assert np.max(np.abs(restored - polygon)) < 1e-3


def test_zero_degree_rotation_keeps_gt_and_image_size(tmp_path: Path):
    image = Image.new("RGB", (64, 48), color=(10, 20, 30))
    polygon = [[10.0, 8.0], [30.0, 8.0], [30.0, 20.0], [10.0, 20.0]]

    result = rotate_image_and_polygons(
        image=image,
        polygons=[polygon],
        angle_deg=0,
        expand=True,
        output_mask_path=tmp_path / "valid_mask.png",
    )

    assert result.image.size == (64, 48)
    assert np.allclose(np.asarray(result.polygons[0]), np.asarray(polygon), atol=1e-6)
    assert result.valid_mask.size == (64, 48)
    assert np.asarray(result.valid_mask).sum() > 0


def test_valid_mask_non_empty_for_nonzero_rotation(tmp_path: Path):
    image = Image.new("RGB", (64, 48), color=(255, 255, 255))

    result = rotate_image_and_polygons(
        image=image,
        polygons=[],
        angle_deg=90,
        expand=True,
        output_mask_path=tmp_path / "valid_mask.png",
    )

    assert result.valid_mask.size == result.image.size
    assert np.asarray(result.valid_mask).sum() > 0
    assert (tmp_path / "valid_mask.png").exists()
