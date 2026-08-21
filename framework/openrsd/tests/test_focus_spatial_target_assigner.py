from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_assigner():
    path = Path("M_AD/models/utils/focus_spatial_target_assigner.py")
    spec = importlib.util.spec_from_file_location("focus_spatial_target_assigner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spatial_assigner_maps_polygon_to_feature_centers_and_limits_points():
    mod = _load_assigner()
    assigner = mod.FocusSpatialTargetAssigner(max_points_per_target=4)
    targets = [{
        "focus_target_id": "t0",
        "target_polygon": "[[15,15],[49,15],[49,49],[15,49]]",
        "target_center_x": "32",
        "target_center_y": "32",
        "raw_box_area": "1156",
        "target_role": "anti_negative",
        "valid_for_loss": "true",
    }]

    result = assigner.assign(
        targets=targets,
        featmap_sizes=[(8, 8), (4, 4)],
        strides=[8, 16],
        image_size=(64, 64))

    anti = result["anti_mask_by_level"]
    assert sum(sum(row) for row in anti[0]) == 4
    assert sum(sum(row) for row in anti[1]) == 0
    assert all(row["assigned_role"] == "anti_negative" for row in result["debug_rows"])


def test_preserve_overrides_anti_on_conflict():
    mod = _load_assigner()
    assigner = mod.FocusSpatialTargetAssigner(max_points_per_target=4)
    targets = [
        {
            "focus_target_id": "anti",
            "target_polygon": "[[16,16],[48,16],[48,48],[16,48]]",
            "target_center_x": "32",
            "target_center_y": "32",
            "raw_box_area": "1024",
            "target_role": "anti_negative",
            "valid_for_loss": "true",
        },
        {
            "focus_target_id": "pos",
            "target_polygon": "[[16,16],[48,16],[48,48],[16,48]]",
            "target_center_x": "32",
            "target_center_y": "32",
            "raw_box_area": "1024",
            "target_role": "preserve_positive",
            "valid_for_loss": "true",
        },
    ]

    result = assigner.assign(targets, [(8, 8)], [8], (64, 64))

    anti_count = sum(sum(row) for row in result["anti_mask_by_level"][0])
    preserve_count = sum(sum(row) for row in result["preserve_mask_by_level"][0])
    assert anti_count == 0
    assert preserve_count == 4
    assert any(r["conflict_resolution"] == "preserve_over_anti" for r in result["debug_rows"])


def test_small_spatial_target_uses_finest_level():
    mod = _load_assigner()
    assigner = mod.FocusSpatialTargetAssigner(max_points_per_target=4)
    targets = [{
        "focus_target_id": "small",
        "target_polygon": "[[456,544],[480,544],[480,568],[456,568]]",
        "target_center_x": "468",
        "target_center_y": "556",
        "raw_box_area": "144",
        "target_role": "anti_negative",
        "valid_for_loss": "true",
    }]

    result = assigner.assign(
        targets=targets,
        featmap_sizes=[(104, 104), (52, 52), (26, 26)],
        strides=[8, 16, 32],
        image_size=(832, 832))

    assert sum(sum(row) for row in result["anti_mask_by_level"][0]) > 0
    assert sum(sum(row) for row in result["anti_mask_by_level"][2]) == 0
    assert all(row["level"] == 0 for row in result["debug_rows"])
