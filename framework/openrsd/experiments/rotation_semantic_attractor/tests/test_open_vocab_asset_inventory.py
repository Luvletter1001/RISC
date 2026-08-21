from pathlib import Path

from experiments.rotation_semantic_attractor.src.model_adapters.open_vocab_asset_inventory import (
    classify_asset,
    find_runnable_open_vocab_pairs,
    scan_open_vocab_assets,
)


def test_open_vocab_asset_inventory_classifies_config_checkpoint_and_support(tmp_path: Path):
    config = tmp_path / "A10_flex_rtm_v3_1_formal.py"
    checkpoint = tmp_path / "epoch_24_weights_only.pth"
    support = tmp_path / "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    config.write_text("model = dict(type='OpenRTMDet', support_feat_dict={})\n")
    checkpoint.write_bytes(b"0" * (1024 * 1024 + 1))
    support.write_bytes(b"support")

    assert classify_asset(config)["asset_type"] == "config"
    assert classify_asset(checkpoint)["asset_type"] == "checkpoint"
    assert classify_asset(support)["asset_type"] == "support_pkl"

    assets = scan_open_vocab_assets([tmp_path])
    assert {asset["asset_type"] for asset in assets} >= {"config", "checkpoint", "support_pkl"}


def test_open_vocab_inventory_finds_runnable_openrsd_pair(tmp_path: Path):
    config = tmp_path / "A10_flex_rtm_v3_1_formal.py"
    checkpoint = tmp_path / "epoch_24_weights_only.pth"
    support = tmp_path / "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    config.write_text("model = dict(type='OpenRTMDet', support_feat_dict={'Data1_DOTA1': 'support.pkl'})\n")
    checkpoint.write_bytes(b"0" * (1024 * 1024 + 1))
    support.write_bytes(b"support")

    pairs = find_runnable_open_vocab_pairs(scan_open_vocab_assets([tmp_path]))

    assert pairs
    assert pairs[0]["candidate_model_family"] == "openrsd"
    assert pairs[0]["usable"] is True


def test_open_vocab_inventory_reports_blocker_when_checkpoint_missing(tmp_path: Path):
    config = tmp_path / "A10_flex_rtm_v3_1_formal.py"
    support = tmp_path / "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    config.write_text("model = dict(type='OpenRTMDet')\n")
    support.write_bytes(b"support")

    pairs = find_runnable_open_vocab_pairs(scan_open_vocab_assets([tmp_path]))

    assert not pairs
