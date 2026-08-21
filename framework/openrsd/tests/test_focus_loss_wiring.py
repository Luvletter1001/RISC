from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_targets():
    path = Path("M_AD/models/utils/focus_loss_targets.py")
    spec = importlib.util.spec_from_file_location("focus_loss_targets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_focus_loss_config_defaults_to_disabled_and_zero_weights():
    mod = _load_targets()
    cfg = mod.normalize_focus_losses_config(None)

    assert cfg["enable"] is False
    assert cfg["target_mapping_mode"] == "none"
    assert cfg["support_distill_weight"] == 0.0
    assert cfg["anti_attractor_weight"] == 0.0
    assert cfg["preserve_weight"] == 0.0


def test_focus_loss_config_requires_mapping_for_anti_or_preserve():
    mod = _load_targets()
    cfg = mod.normalize_focus_losses_config({
        "enable": True,
        "target_mapping_mode": "none",
        "anti_attractor_weight": 0.05,
        "preserve_weight": 0.05,
    })

    status = mod.validate_focus_loss_targets_config(cfg)

    assert status["ok"] is False
    assert status["status"] == "BLOCKED_TARGET_MAPPING_REQUIRED"


def test_old_crop_labels_without_spatial_or_exact_fields_are_rejected():
    mod = _load_targets()
    row = {
        "crop_id": "valid_unmatched_sv_0000",
        "tile_id": "tile",
        "angle": "30",
        "focus_label": "corrected_false_sv",
    }

    assert mod.has_spatial_region_fields(row) is False
    assert mod.has_exact_provenance_fields(row) is False
