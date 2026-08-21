import csv
import subprocess
import sys
from pathlib import Path

import pytest

from M_Tools.analysis.filter_ep2_pre_nms_logits_by_target_iou import (
    filter_ep2_pre_nms_logits_by_target_iou,
    polygon_iou,
    rbox_to_qbox,
)


def write_rows(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ep2_path(case_id, control_type, step="00"):
    return (
        f"ep2case-{case_id}__ep2ctx-tile_a__ep2obj-obj_a"
        f"__ep2ctrl-{control_type}__ep2step-{step}.jpg"
    )


def test_rbox_to_qbox_and_polygon_iou_cover_axis_aligned_target():
    qbox = rbox_to_qbox(10, 10, 4, 6, 0)

    assert qbox == pytest.approx([12, 13, 12, 7, 8, 7, 8, 13])
    assert polygon_iou(qbox, [8, 7, 12, 7, 12, 13, 8, 13]) == pytest.approx(1.0)


def test_filter_keeps_target_iou_candidate_not_highest_logit(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    e9_csv = tmp_path / "e9.csv"
    out_csv = tmp_path / "filtered.csv"
    case_id = "p4_ovd_ep2_s3c-0000-P1189"
    target_qbox = "8 7 12 7 12 13 8 13"
    write_rows(e9_csv, [{
        "model": "p4_ovd_ep2_s3c",
        "variant": "original_tile",
        "target_gt_class": "tennis-court",
        "impossible_pred_class": "small-vehicle",
        "target_pair": "small-vehicle->tennis-court",
        "target_qbox": target_qbox,
        "variant_image_path": ep2_path(case_id, "original_tile"),
    }])
    write_rows(raw_csv, [
        {
            "case_id": case_id,
            "control_type": "original_tile",
            "pair": "tennis-court->small-vehicle",
            "gt_class": "tennis-court",
            "hardneg_class": "small-vehicle",
            "path_step": "00",
            "logit_gt": "0.0",
            "logit_hardneg": "20.0",
            "z_gt": "4.0",
            "z_hardneg": "1.0",
            "bbox_cx": "100",
            "bbox_cy": "100",
            "bbox_w": "4",
            "bbox_h": "6",
            "bbox_angle": "0",
        },
        {
            "case_id": case_id,
            "control_type": "original_tile",
            "pair": "tennis-court->small-vehicle",
            "gt_class": "tennis-court",
            "hardneg_class": "small-vehicle",
            "path_step": "00",
            "logit_gt": "1.0",
            "logit_hardneg": "2.0",
            "z_gt": "3.0",
            "z_hardneg": "2.0",
            "bbox_cx": "10",
            "bbox_cy": "10",
            "bbox_w": "4",
            "bbox_h": "6",
            "bbox_angle": "0",
        },
    ])

    summary = filter_ep2_pre_nms_logits_by_target_iou(
        raw_csv, e9_csv, out_csv, min_iou=0.5)

    rows = read_rows(out_csv)
    assert summary["rows_written"] == 1
    assert rows[0]["logit_hardneg"] == "2.0"
    assert float(rows[0]["target_iou"]) == pytest.approx(1.0)
    assert rows[0]["target_pair_from_e9"] == "small-vehicle->tennis-court"
    assert rows[0]["control_type"] == "clean"
    assert rows[0]["path_variant"] == "original_tile"
    assert rows[0]["source_control_type"] == "original_tile"


def test_filter_reports_unmatched_groups_below_min_iou(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    e9_csv = tmp_path / "e9.csv"
    out_csv = tmp_path / "filtered.csv"
    case_id = "c1"
    write_rows(e9_csv, [{
        "target_gt_class": "ship",
        "impossible_pred_class": "harbor",
        "target_qbox": "0 0 10 0 10 10 0 10",
        "variant_image_path": ep2_path(case_id, "neutral_same_scale", "01"),
    }])
    write_rows(raw_csv, [{
        "case_id": case_id,
        "control_type": "neutral_same_scale",
        "pair": "ship->harbor",
        "bbox_cx": "100",
        "bbox_cy": "100",
        "bbox_w": "10",
        "bbox_h": "10",
        "bbox_angle": "0",
    }])

    summary = filter_ep2_pre_nms_logits_by_target_iou(
        raw_csv, e9_csv, out_csv, min_iou=0.5)

    assert summary["groups_seen"] == 1
    assert summary["rows_written"] == 0
    assert summary["unmatched_groups"] == 1


def test_filter_keeps_real_control_type_for_randomized_controls(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    e9_csv = tmp_path / "e9.csv"
    out_csv = tmp_path / "filtered.csv"
    case_id = "c_control"
    write_rows(e9_csv, [{
        "target_gt_class": "ship",
        "impossible_pred_class": "harbor",
        "target_qbox": "8 7 12 7 12 13 8 13",
        "variant_image_path": ep2_path(case_id, "shuffled_prior", "00"),
    }])
    write_rows(raw_csv, [{
        "case_id": case_id,
        "control_type": "shuffled_prior",
        "pair": "ship->harbor",
        "bbox_cx": "10",
        "bbox_cy": "10",
        "bbox_w": "4",
        "bbox_h": "6",
        "bbox_angle": "0",
    }])

    filter_ep2_pre_nms_logits_by_target_iou(
        raw_csv, e9_csv, out_csv, min_iou=0.5)

    rows = read_rows(out_csv)
    assert rows[0]["control_type"] == "shuffled_prior"
    assert rows[0]["path_variant"] == "shuffled_prior"
    assert rows[0]["source_control_type"] == "shuffled_prior"


def test_cli_runs_from_repo_root(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    e9_csv = tmp_path / "e9.csv"
    out_csv = tmp_path / "filtered.csv"
    case_id = "c_cli"
    write_rows(e9_csv, [{
        "target_gt_class": "ship",
        "impossible_pred_class": "harbor",
        "target_qbox": "8 7 12 7 12 13 8 13",
        "variant_image_path": ep2_path(case_id, "original_tile", "00"),
    }])
    write_rows(raw_csv, [{
        "case_id": case_id,
        "control_type": "original_tile",
        "pair": "ship->harbor",
        "bbox_cx": "10",
        "bbox_cy": "10",
        "bbox_w": "4",
        "bbox_h": "6",
        "bbox_angle": "0",
    }])

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "M_Tools/analysis/filter_ep2_pre_nms_logits_by_target_iou.py",
            "--raw-csv",
            str(raw_csv),
            "--e9-csv",
            str(e9_csv),
            "--output-csv",
            str(out_csv),
            "--min-iou",
            "0.5",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"rows_written": 1' in result.stdout
