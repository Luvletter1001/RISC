import csv
import math
from pathlib import Path

import pytest
import torch

from M_AD.models.utils.ep2_path_probe_dump import (
    append_ep2_path_probe_rows,
    build_ep2_path_probe_rows,
    parse_ep2_metadata_from_path,
)


def test_build_ep2_path_probe_rows_records_raw_logits_and_gaussian_z():
    rows = build_ep2_path_probe_rows(
        cls_logits=torch.tensor([
            [0.2, 1.5, -0.1],
            [2.0, 0.4, 0.1],
        ]),
        decoded_bboxes=torch.tensor([
            [10.0, 10.0, 10.0, 10.0, 0.0],
            [20.0, 20.0, 100.0, 100.0, 0.0],
        ]),
        class_names=("small-vehicle", "tennis-court", "plane"),
        class_pairs=[("small-vehicle", "tennis-court")],
        log_area_mean=torch.tensor([math.log(100.0), math.log(10000.0), 0.0]),
        log_area_std=torch.tensor([0.5, 0.5, 1.0]),
        valid_mask=torch.tensor([True, True, False]),
        img_meta={
            "img_id": "tile_001_step2",
            "img_path": "/tmp/tile_001_step2.png",
            "ep2_case_id": "case_001",
            "ep2_context_id": "tile_001",
            "ep2_object_id": "obj_001",
            "ep2_control_type": "clean",
            "ep2_path_step": 2,
        },
        level_idx=3,
        max_locations=1,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["case_id"] == "case_001"
    assert row["control_type"] == "clean"
    assert row["context_id"] == "tile_001"
    assert row["object_id"] == "obj_001"
    assert row["path_step"] == 2
    assert row["pair"] == "small-vehicle->tennis-court"
    assert row["location_idx"] == 0
    assert row["level_idx"] == 3
    assert row["bbox_cx"] == pytest.approx(10.0)
    assert row["bbox_cy"] == pytest.approx(10.0)
    assert row["bbox_w"] == pytest.approx(10.0)
    assert row["bbox_h"] == pytest.approx(10.0)
    assert row["bbox_angle"] == pytest.approx(0.0)
    assert row["logit_gt"] == pytest.approx(0.2)
    assert row["logit_hardneg"] == pytest.approx(1.5)
    assert row["hardneg_margin"] == pytest.approx(1.3)
    assert row["z_gt"] == pytest.approx(0.0)
    assert row["z_hardneg"] == pytest.approx(-9.2103405, rel=1e-5)
    assert row["support_advantage"] == pytest.approx(-9.2103405, rel=1e-5)


def test_append_ep2_path_probe_rows_writes_header_once(tmp_path: Path):
    output_csv = tmp_path / "ep2_raw_logits.csv"
    rows = [{
        "case_id": "case_001",
        "control_type": "clean",
        "context_id": "tile_001",
        "object_id": "obj_001",
        "pair": "a->b",
        "gt_class": "a",
        "hardneg_class": "b",
        "path_step": 0,
        "log_area": 1.0,
        "logit_gt": 0.1,
        "logit_hardneg": 0.2,
        "z_gt": 0.0,
        "z_hardneg": 1.0,
        "abs_z_gt": 0.0,
        "abs_z_hardneg": 1.0,
        "hardneg_margin": 0.1,
        "support_advantage": -1.0,
    }]

    append_ep2_path_probe_rows(output_csv, rows)
    append_ep2_path_probe_rows(output_csv, rows)

    with output_csv.open(newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert len(loaded) == 2
    assert loaded[0]["case_id"] == "case_001"


def test_parse_ep2_metadata_from_path_reads_encoded_variant_name():
    meta = parse_ep2_metadata_from_path(
        "/tmp/ep2case-case001__ep2ctx-tile001__ep2obj-obj001__"
        "ep2ctrl-clean__ep2step-02.jpg")

    assert meta == {
        "ep2_case_id": "case001",
        "ep2_context_id": "tile001",
        "ep2_object_id": "obj001",
        "ep2_control_type": "clean",
        "ep2_path_step": "02",
    }


def test_parse_ep2_metadata_from_path_keeps_underscore_control_name():
    meta = parse_ep2_metadata_from_path(
        "/tmp/ep2case-p4_s3c-0007-P0001-1024-0-0__"
        "ep2ctx-P0001-1024-0-0__ep2obj-p4_s3c-0007__"
        "ep2ctrl-neutral_same_scale__ep2step-01.jpg")

    assert meta["ep2_control_type"] == "neutral_same_scale"
