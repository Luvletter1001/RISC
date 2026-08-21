from pathlib import Path

from PIL import Image

from experiments.rotation_semantic_attractor.src.metrics.context_metrics import (
    write_context_counterfactual_images,
)


def test_write_context_counterfactual_images_creates_real_images(tmp_path: Path):
    image_path = tmp_path / "tile.png"
    ann_path = tmp_path / "tile.txt"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)
    ann_path.write_text("2 2 8 2 8 8 2 8 small-vehicle 0\n")

    result = write_context_counterfactual_images(image_path, ann_path, tmp_path / "cf")

    assert result["num_target_polygons"] == 1
    assert set(result["conditions"]) == {"object_only", "context_only"}
    for path in result["conditions"].values():
        assert Path(path).exists()
        with Image.open(path) as image:
            assert image.size == (16, 16)
