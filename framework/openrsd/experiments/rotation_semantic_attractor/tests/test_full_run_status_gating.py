import importlib
import json
import sys
from pathlib import Path

from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus

sys.path.insert(0, str(Path("experiments/rotation_semantic_attractor/scripts").resolve()))
false_hub = importlib.import_module("experiments.rotation_semantic_attractor.scripts.05_eval_false_hub_taxonomy")


def _full_manifest(model_status):
    return {
        "angles": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        "args": {"limit": 0},
        "model_status": model_status,
    }


def test_false_hub_done_full_requires_model_completion():
    split = {"split_name": "S2_final_test"}
    meta = false_hub._false_hub_status(
        split,
        _full_manifest({"rotated_retinanet_msrr": {"status": ExperimentStatus.DONE_FULL}}),
        "rotated_retinanet_msrr",
    )
    assert meta["status"] == ExperimentStatus.DONE_FULL

    meta = false_hub._false_hub_status(
        split,
        _full_manifest({"rotated_retinanet_msrr": {"status": "RUNNING"}}),
        "rotated_retinanet_msrr",
    )
    assert meta["status"] == ExperimentStatus.NOT_RUN
    assert not meta["include_in_main_table"]


def test_false_hub_failed_model_is_not_promoted_to_done_full():
    meta = false_hub._false_hub_status(
        {"split_name": "S2_final_test"},
        _full_manifest({"r3det_kfiou": {"status": ExperimentStatus.FAILED}}),
        "r3det_kfiou",
    )
    assert meta["status"] == ExperimentStatus.FAILED
    assert not meta["is_scientific_result"]


def test_false_hub_parallel_rows_match_serial_rows(tmp_path):
    pred_dir = tmp_path / "canonical_predictions" / "toy_model" / "tile_a"
    pred_dir.mkdir(parents=True)
    pred_path = pred_dir / "angle_000.json"
    pred_path.write_text(
        json.dumps(
            {
                "final_predictions": [
                    {
                        "stage": "final",
                        "class_name": "small-vehicle",
                        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    }
                ]
            }
        )
    )
    task = false_hub._prediction_task(
        pred_path,
        [{"class_name": "small-vehicle", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
        0.3,
        {"status": ExperimentStatus.DONE_FULL},
    )

    serial_rows, serial_absorption = false_hub._compute_prediction_tasks([task], workers=1)
    parallel_rows, parallel_absorption = false_hub._compute_prediction_tasks([task], workers=2)

    assert parallel_rows == serial_rows
    assert parallel_absorption == serial_absorption
    assert [row["region_mode"] for row in serial_rows] == ["all_region", "valid_mask_only"]
    assert serial_rows[0]["status"] == ExperimentStatus.DONE_FULL
