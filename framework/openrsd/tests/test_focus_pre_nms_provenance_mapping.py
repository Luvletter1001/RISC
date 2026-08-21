from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_matcher():
    path = Path(
        "experiments/rotation_semantic_attractor/scripts/"
        "67_focus_match_verified_crops_to_pre_nms_provenance.py")
    spec = importlib.util.spec_from_file_location("focus_pre_nms_matcher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pre_nms_matcher_requires_tile_angle_class_iou_and_score():
    matcher = _load_matcher()
    crop = {
        "crop_id": "c0",
        "tile_id": "tile",
        "angle": "30",
        "score": "0.9000",
        "focus_label": "corrected_false_sv",
        "raw_prediction_record": '{"box": [10, 10, 20, 20], "box_type": "xyxy"}',
    }
    candidates = [
        {
            "prediction_id": "bad_angle",
            "tile_id": "tile",
            "angle": "60",
            "class_name": "small-vehicle",
            "score": "0.9000",
            "box": "[10, 10, 20, 20]",
            "feature_level": "0",
            "grid_x": "1",
            "grid_y": "2",
            "anchor_id_or_point_id": "p",
        },
        {
            "prediction_id": "best",
            "tile_id": "tile",
            "angle": "30",
            "class_name": "small-vehicle",
            "score": "0.9000",
            "box": "[10, 10, 20, 20]",
            "feature_level": "0",
            "grid_x": "1",
            "grid_y": "2",
            "anchor_id_or_point_id": "p",
        },
    ]

    match = matcher.match_crop_to_provenance(crop, candidates, iou_thr=0.8)

    assert match["prediction_id"] == "best"
    assert match["valid_for_exact_loss"] == "true"
    assert float(match["match_iou"]) == 1.0


def test_pre_nms_matcher_marks_unmatched_when_coverage_missing():
    matcher = _load_matcher()
    crop = {
        "crop_id": "c0",
        "tile_id": "tile",
        "angle": "30",
        "score": "0.9",
        "focus_label": "corrected_false_sv",
        "raw_prediction_record": '{"box": [10, 10, 20, 20], "box_type": "xyxy"}',
    }

    match = matcher.match_crop_to_provenance(crop, [], iou_thr=0.8)

    assert match["valid_for_exact_loss"] == "false"
    assert match["invalid_reason"] == "no_matching_pre_nms_candidate"
