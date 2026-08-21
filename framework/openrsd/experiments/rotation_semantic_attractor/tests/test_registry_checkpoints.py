import json
from pathlib import Path


def test_ai4rs_missing_models_have_checkpoint_urls_and_local_targets():
    registry_path = Path("experiments/rotation_semantic_attractor/configs/model_registry.yaml")
    registry = json.loads(registry_path.read_text())["models"]

    expected = {
        "roi_trans": "roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth",
        "s2anet": "s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth",
        "rotated_fcos": "rotated_fcos_r50_fpn_1x_dota_le90-d87568ed.pth",
        "rotated_atss": "rotated_atss_obb_r50_fpn_1x_dota_le90-e029ca06.pth",
    }

    for model_key, filename in expected.items():
        cfg = registry[model_key]
        assert cfg["checkpoint"].endswith(f"/weights/{filename}")
        assert cfg["checkpoint_url"].startswith("https://download.openmmlab.com/mmrotate/v0.1.0/")
        assert cfg["checkpoint_url"].endswith(filename)

