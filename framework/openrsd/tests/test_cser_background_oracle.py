import numpy as np
import json
import subprocess
import sys
from pathlib import Path

from M_Tools.analysis.evaluate_cser_background_oracle import (
    background_mask,
    evaluate_samples,
    filter_sample,
    load_angle_prediction_records,
    parse_dota_annotation,
    summarize_removed_predictions,
)


def _sample():
    return {
        "image_id": "synthetic",
        "boxes": np.asarray(
            [[10, 10, 4, 4, 0], [40, 40, 4, 4, 0]], dtype=np.float32),
        "scores": np.asarray([0.8, 0.9], dtype=np.float32),
        "labels": np.asarray([4, 4], dtype=np.int64),
        "gt_boxes": np.asarray([[10, 10, 4, 4, 0]], dtype=np.float32),
        "gt_labels": np.asarray([4], dtype=np.int64),
        "gt_boxes_ignore": np.zeros((0, 5), dtype=np.float32),
        "gt_labels_ignore": np.zeros((0,), dtype=np.int64),
    }


def test_background_mask_uses_any_gt_iou():
    sample = _sample()

    mask = background_mask(
        sample["boxes"], sample["gt_boxes"], iou_thr=0.1)

    assert mask.tolist() == [False, True]


def test_disabled_oracle_is_exact_baseline():
    sample = _sample()

    kept = filter_sample(sample, mode="baseline", bg_iou_thr=0.1)

    np.testing.assert_array_equal(kept["boxes"], sample["boxes"])
    np.testing.assert_array_equal(kept["scores"], sample["scores"])
    np.testing.assert_array_equal(kept["labels"], sample["labels"])


def test_background_oracle_improves_ap_without_removing_tp():
    sample = _sample()
    class_names = ("c0", "c1", "c2", "c3", "small-vehicle")
    baseline = filter_sample(sample, mode="baseline", bg_iou_thr=0.1)
    oracle = filter_sample(sample, mode="bg_oracle", bg_iou_thr=0.1)

    baseline_eval = evaluate_samples([baseline], class_names, nproc=1)
    oracle_eval = evaluate_samples([oracle], class_names, nproc=1)
    removed = summarize_removed_predictions([sample], [oracle])

    assert baseline_eval["map50"] < oracle_eval["map50"]
    assert oracle_eval["map50"] == 1.0
    assert removed["removed_prediction_count"] == 1
    assert removed["removed_tp_count"] == 0


def test_jsonl_loader_keeps_only_requested_angle(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "image_id": "tile_rot000", "tile_id": "tile", "angle": 0,
            "class_id": 10, "class_name": "small-vehicle", "score": 0.8,
            "rbox": [10, 10, 4, 4, 0],
        },
        {
            "image_id": "tile_rot030", "tile_id": "tile", "angle": 30,
            "class_id": 10, "class_name": "small-vehicle", "score": 0.9,
            "rbox": [11, 11, 4, 4, 0],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    records = load_angle_prediction_records(path, angle=0)

    assert list(records) == ["tile_rot000"]
    assert records["tile_rot000"]["boxes"].shape == (1, 5)
    assert records["tile_rot000"]["labels"].tolist() == [4]


def test_dota_parser_separates_difficult_instances(tmp_path):
    path = tmp_path / "tile.txt"
    path.write_text(
        "8 8 12 8 12 12 8 12 small-vehicle 0\n"
        "38 38 42 38 42 42 38 42 small-vehicle 1\n")

    parsed = parse_dota_annotation(
        path, class_to_label={"small-vehicle": 4}, diff_thr=0)

    assert parsed["gt_boxes"].shape == (1, 5)
    assert parsed["gt_labels"].tolist() == [4]
    assert parsed["gt_boxes_ignore"].shape == (1, 5)
    assert parsed["gt_labels_ignore"].tolist() == [4]


def test_dota_parser_uses_mmrotate_qbox_to_rbox_conversion(tmp_path):
    path = tmp_path / "irregular.txt"
    path.write_text(
        "0 0 6 0 4 4 0 4 bridge 0\n"
        "10 10 14 10 13 13 10 14 bridge 1\n")

    parsed = parse_dota_annotation(
        path, class_to_label={"bridge": 2}, diff_thr=0)

    np.testing.assert_allclose(
        parsed["gt_boxes"],
        np.asarray([[3, 2, 4, 6, np.pi / 2]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        parsed["gt_boxes_ignore"],
        np.asarray([[12, 12, 4, 4, np.pi / 2]], dtype=np.float32),
        atol=1e-6,
    )


def test_direct_script_entrypoint_can_resolve_repo_imports():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "M_Tools/analysis/evaluate_cser_background_oracle.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
