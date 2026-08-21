import csv
import math

import pytest

from M_Tools.analysis.build_class_area_priors import (
    audit_annotation_overlap,
    build_class_area_prior_rows,
    qbox_area,
    qbox_log_aspect,
    qbox_right_angle_error,
    source_id_from_tile_id,
)


def write_ann(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_qbox_area_uses_polygon_area():
    assert qbox_area([0, 0, 4, 0, 4, 5, 0, 5]) == 20.0


def test_qbox_geometry_descriptors_are_rotation_and_order_invariant():
    wide = [0, 0, 8, 0, 8, 2, 0, 2]
    tall_reordered = [8, 2, 0, 2, 0, 0, 8, 0]

    assert qbox_log_aspect(wide) == pytest.approx(math.log(4.0))
    assert qbox_log_aspect(tall_reordered) == pytest.approx(math.log(4.0))
    assert qbox_right_angle_error(wide) == pytest.approx(0.0)


def test_build_class_area_prior_rows_from_dota_txt(tmp_path):
    ann_dir = tmp_path / "annfiles"
    ann_dir.mkdir()
    write_ann(
        ann_dir / "a.txt",
        [
            "0 0 10 0 10 10 0 10 small-vehicle 0",
            "0 0 20 0 20 20 0 20 small-vehicle 0",
            "0 0 40 0 40 40 0 40 plane 0",
            "0 0 8 0 8 8 0 8 ignored-class 0",
            "0 0 100 0 100 100 0 100 plane 200",
        ],
    )
    write_ann(
        ann_dir / "b.txt",
        ["0 0 30 0 30 30 0 30 plane 0"],
    )

    rows = build_class_area_prior_rows(
        ann_dir=ann_dir,
        class_names=["small-vehicle", "plane"],
        diff_thr=100,
    )

    by_class = {row["class"]: row for row in rows}
    assert set(by_class) == {"small-vehicle", "plane"}
    assert by_class["small-vehicle"]["count"] == 2
    assert by_class["small-vehicle"]["median_area"] == 250.0
    assert by_class["plane"]["count"] == 2
    assert by_class["plane"]["median_area"] == 1250.0
    assert by_class["plane"]["source_ann_dir"] == str(ann_dir)
    assert by_class["plane"]["log_area_mean"] == (
        math.log(1600.0) + math.log(900.0)) / 2.0
    assert by_class["plane"]["log_area_std"] > 0.0
    assert by_class["small-vehicle"]["log_aspect_mean"] == pytest.approx(0.0)
    assert by_class["plane"]["log_aspect_std"] == pytest.approx(0.0)
    assert by_class["plane"]["right_angle_error_mean"] == pytest.approx(0.0)


def test_source_id_from_tile_id_handles_dota_and_angle_prefixes():
    assert source_id_from_tile_id("P0001__1024__0___0") == "P0001"
    assert source_id_from_tile_id("angle_030__P0001__1024__0___0") == "P0001"
    assert source_id_from_tile_id("plain_image") == "plain_image"


def test_audit_annotation_overlap_reports_tile_and_source_overlap(tmp_path):
    prior_dir = tmp_path / "prior"
    eval_dir = tmp_path / "eval"
    prior_dir.mkdir()
    eval_dir.mkdir()
    write_ann(prior_dir / "P0001__1024__0___0.txt", [])
    write_ann(prior_dir / "P0001__1024__512___0.txt", [])
    write_ann(prior_dir / "P0002__1024__0___0.txt", [])
    write_ann(eval_dir / "P0001__1024__0___0.txt", [])
    write_ann(eval_dir / "P0003__1024__0___0.txt", [])

    audit = audit_annotation_overlap(prior_dir, eval_dir)

    assert audit["prior_tile_count"] == 3
    assert audit["eval_tile_count"] == 2
    assert audit["tile_overlap_count"] == 1
    assert audit["source_overlap_count"] == 1
    assert audit["eval_tile_overlap_rate"] == 0.5
    assert audit["eval_source_overlap_rate"] == 0.5


def test_build_class_area_prior_rows_can_exclude_eval_tiles_or_sources(tmp_path):
    prior_dir = tmp_path / "prior"
    eval_dir = tmp_path / "eval"
    prior_dir.mkdir()
    eval_dir.mkdir()
    write_ann(
        prior_dir / "P0001__1024__0___0.txt",
        ["0 0 10 0 10 10 0 10 small-vehicle 0"],
    )
    write_ann(
        prior_dir / "P0001__1024__512___0.txt",
        ["0 0 20 0 20 20 0 20 small-vehicle 0"],
    )
    write_ann(
        prior_dir / "P0002__1024__0___0.txt",
        ["0 0 30 0 30 30 0 30 plane 0"],
    )
    write_ann(eval_dir / "P0001__1024__0___0.txt", [])

    tile_rows = build_class_area_prior_rows(
        ann_dir=prior_dir,
        class_names=["small-vehicle", "plane"],
        exclude_ann_dir=eval_dir,
        exclude_level="tile",
    )
    source_rows = build_class_area_prior_rows(
        ann_dir=prior_dir,
        class_names=["small-vehicle", "plane"],
        exclude_ann_dir=eval_dir,
        exclude_level="source",
    )

    tile_by_class = {row["class"]: row for row in tile_rows}
    source_by_class = {row["class"]: row for row in source_rows}
    assert tile_by_class["small-vehicle"]["count"] == 1
    assert tile_by_class["small-vehicle"]["median_area"] == 400.0
    assert "small-vehicle" not in source_by_class
    assert source_by_class["plane"]["count"] == 1
