from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_lookup():
    path = Path("M_AD/models/utils/focus_target_lookup.py")
    spec = importlib.util.spec_from_file_location("focus_target_lookup", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_targets(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "focus_target_id",
        "tile_id",
        "angle",
        "image_path",
        "target_role",
        "valid_for_loss",
        "coordinate_frame",
        "target_polygon",
        "target_center_x",
        "target_center_y",
        "raw_box_area",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_lookup_matches_tile_angle_and_transforms_resize(tmp_path):
    mod = _load_lookup()
    csv_path = tmp_path / "targets.csv"
    _write_targets(csv_path, [
        {
            "focus_target_id": "anti-0",
            "tile_id": "P0001__1024__0___0",
            "angle": "90",
            "image_path": (
                "/data/DOTA1_1024_500/angle_sweep_val/realistic/"
                "angle_090/images/P0001__1024__0___0.png"),
            "target_role": "anti_negative",
            "valid_for_loss": "true",
            "coordinate_frame": "rotated_angle_sweep",
            "target_polygon": "[[10,20],[30,20],[30,40],[10,40]]",
            "target_center_x": "20",
            "target_center_y": "30",
            "raw_box_area": "400",
        },
        {
            "focus_target_id": "preserve-0",
            "tile_id": "P0001__1024__0___0",
            "angle": "90",
            "image_path": (
                "/data/DOTA1_1024_500/angle_sweep_val/realistic/"
                "angle_090/images/P0001__1024__0___0.png"),
            "target_role": "preserve_positive",
            "valid_for_loss": "true",
            "coordinate_frame": "rotated_angle_sweep",
            "target_polygon": "[[50,60],[70,60],[70,80],[50,80]]",
            "target_center_x": "60",
            "target_center_y": "70",
            "raw_box_area": "400",
        },
    ])

    lookup = mod.FocusTargetLookup(csv_path)
    batch_targets, debug_rows = lookup.match_batch([
        {
            "img_path": (
                "/work/angle_090/images/P0001__1024__0___0.png"),
            "img_id": "P0001__1024__0___0",
            "ori_shape": (100, 100),
            "img_shape": (50, 50),
            "scale_factor": (0.5, 0.5),
        }
    ])

    assert debug_rows[0]["parsed_tile_id"] == "P0001__1024__0___0"
    assert debug_rows[0]["parsed_angle"] == 90
    assert debug_rows[0]["matched_targets"] == 2
    assert debug_rows[0]["anti_targets"] == 1
    assert debug_rows[0]["preserve_targets"] == 1
    assert debug_rows[0]["unmatched_reason"] == ""
    assert batch_targets[0]["anti_targets"][0]["target_polygon"][0] == (5.0, 10.0)
    assert batch_targets[0]["anti_targets"][0]["target_center_x"] == 10.0
    assert batch_targets[0]["preserve_targets"][0]["target_polygon"][2] == (35.0, 40.0)


def test_lookup_rejects_unsupported_flip_transform(tmp_path):
    mod = _load_lookup()
    csv_path = tmp_path / "targets.csv"
    _write_targets(csv_path, [
        {
            "focus_target_id": "anti-0",
            "tile_id": "P0002__1024__0___0",
            "angle": "180",
            "image_path": (
                "/data/DOTA1_1024_500/angle_sweep_val/realistic/"
                "angle_180/images/P0002__1024__0___0.png"),
            "target_role": "anti_negative",
            "valid_for_loss": "true",
            "coordinate_frame": "rotated_angle_sweep",
            "target_polygon": "[[10,10],[20,10],[20,20],[10,20]]",
            "target_center_x": "15",
            "target_center_y": "15",
            "raw_box_area": "100",
        },
    ])

    lookup = mod.FocusTargetLookup(csv_path)
    batch_targets, debug_rows = lookup.match_batch([
        {
            "img_path": "/work/angle_180/images/P0002__1024__0___0.png",
            "img_id": "P0002__1024__0___0",
            "ori_shape": (100, 100),
            "img_shape": (100, 100),
            "scale_factor": (1.0, 1.0),
            "flip": True,
            "flip_direction": "horizontal",
        }
    ])

    assert batch_targets[0]["anti_targets"] == []
    assert debug_rows[0]["matched_targets"] == 1
    assert debug_rows[0]["unmatched_reason"] == "augmentation_not_supported"


def test_lookup_reports_no_matching_targets_with_parsed_angle(tmp_path):
    mod = _load_lookup()
    csv_path = tmp_path / "targets.csv"
    _write_targets(csv_path, [
        {
            "focus_target_id": "anti-0",
            "tile_id": "P0003__1024__0___0",
            "angle": "0",
            "image_path": (
                "/data/DOTA1_1024_500/angle_sweep_val/realistic/"
                "angle_000/images/P0003__1024__0___0.png"),
            "target_role": "anti_negative",
            "valid_for_loss": "true",
            "coordinate_frame": "rotated_angle_sweep",
            "target_polygon": "[[10,10],[20,10],[20,20],[10,20]]",
            "target_center_x": "15",
            "target_center_y": "15",
            "raw_box_area": "100",
        },
    ])

    lookup = mod.FocusTargetLookup(csv_path)
    batch_targets, debug_rows = lookup.match_batch([
        {
            "filename": "/work/angle_270/images/P9999__1024__0___0.png",
            "img_id": "P9999__1024__0___0",
            "ori_shape": (100, 100),
            "img_shape": (100, 100),
        }
    ])

    assert batch_targets[0]["parsed_angle"] == 270
    assert debug_rows[0]["matched_targets"] == 0
    assert debug_rows[0]["unmatched_reason"] == "no_matching_targets"
