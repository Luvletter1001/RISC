from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_preflight():
    path = Path(
        "experiments/rotation_semantic_attractor/scripts/"
        "57_focus_p1_preflight.py")
    spec = importlib.util.spec_from_file_location("focus_p1_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_p1_preflight_blocks_without_focus_loss_branch_and_loss_masks(tmp_path):
    preflight = _load_preflight()
    dense_head = tmp_path / "head.py"
    dense_head.write_text(
        "def loss_by_feat():\n"
        "    losses = {}\n"
        "    losses['loss_cls'] = 0\n"
        "    return losses\n",
        encoding="utf-8")
    train_split = tmp_path / "p0_train_split.json"
    train_split.write_text(
        '{"rows": [{"crop_id": "c0", "tile_id": "t0", "angle": "0", '
        '"focus_label": "corrected_false_sv"}]}',
        encoding="utf-8")

    loss_check = preflight.inspect_detector_loss_wiring(dense_head)
    mapping_check = preflight.inspect_label_to_loss_mapping(
        train_split, mapper_paths=[])
    status = preflight.derive_p1_status(
        [loss_check, mapping_check],
        actual_detector_train=False,
        actual_detector_rerun=False)

    assert loss_check["ok"] is False
    assert mapping_check["ok"] is False
    assert mapping_check["detail"] == "tile_angle_proxy_only_no_training_loss_mask"
    assert status == "BLOCKED_LABEL_TO_LOSS_MAPPING"


def test_p1_preflight_accepts_explicit_training_loss_mask_mapping(tmp_path):
    preflight = _load_preflight()
    train_split = tmp_path / "p0_train_split.json"
    train_split.write_text(
        '{"rows": [{"crop_id": "c0", "tile_id": "t0", "angle": "0", '
        '"prediction_id": "p0", "train_sample_id": "s0", '
        '"feature_level": "p3", "grid_x": "4", "grid_y": "5", '
        '"focus_label": "corrected_false_sv"}]}',
        encoding="utf-8")

    mapping_check = preflight.inspect_label_to_loss_mapping(
        train_split, mapper_paths=[])

    assert mapping_check["ok"] is True
    assert mapping_check["detail"] == "exact_training_loss_mask_fields_present"
