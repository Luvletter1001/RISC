import importlib
import json
import sys
from pathlib import Path

from experiments.rotation_semantic_attractor.src.metrics.stage_metrics import compute_stage_row
from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus

sys.path.insert(0, str(Path("experiments/rotation_semantic_attractor/scripts").resolve()))
stage_script = importlib.import_module("experiments.rotation_semantic_attractor.scripts.06_stage_decomposition")


def test_stage_row_reports_partial_support_when_pre_nms_and_dense_are_available():
    raw_output = {
        "final_predictions": [
            {"class_name": "small-vehicle"},
            {"class_name": "ship"},
        ],
        "pre_nms_predictions": [
            {"class_name": "small-vehicle"},
            {"class_name": "small-vehicle"},
            {"class_name": "ship"},
            {"class_name": "plane"},
        ],
        "dense_logits": {
            "small_vehicle_topk_mean_score": 0.42,
            "small_vehicle_max_score": 0.91,
        },
        "metadata": {
            "capability_profile": {
                "model_family": "closed_set",
                "architecture_type": "dense_head",
                "query_logits_status": ExperimentStatus.NOT_APPLICABLE,
                "query_logits_reason": "closed-set dense/head detector does not use query logits",
            },
        },
    }

    row = compute_stage_row("toy", "tile", 90, raw_output)

    assert row["pre_nms_fr_sv"] == 0.5
    assert row["post_nms_fr_sv"] == 0.5
    assert row["nms_amp_sv"] == 0.0
    assert row["dense_or_query_sv"] == 0.42
    assert row["query_logits_status"] == ExperimentStatus.NOT_APPLICABLE
    assert row["status"] == ExperimentStatus.DONE_SMOKE


def test_stage_parallel_rows_match_serial_rows(tmp_path):
    raw_dir = tmp_path / "raw_predictions" / "toy_model" / "tile_a"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "angle_000.json"
    raw_path.write_text(
        json.dumps(
            {
                "final_predictions": [{"class_name": "small-vehicle"}],
                "metadata": {"capability_profile": {"model_family": "closed_set"}},
            }
        )
    )

    serial_rows = stage_script._compute_stage_paths([raw_path], workers=1)
    parallel_rows = stage_script._compute_stage_paths([raw_path], workers=2)

    assert parallel_rows == serial_rows
    assert serial_rows[0]["model_name"] == "toy_model"
