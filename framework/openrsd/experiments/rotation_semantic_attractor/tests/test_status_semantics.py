import json
from pathlib import Path

from experiments.rotation_semantic_attractor.src.metrics.stage_metrics import compute_stage_row
from experiments.rotation_semantic_attractor.src.model_adapters.capabilities import build_capability_profile
from experiments.rotation_semantic_attractor.src.utils.status import (
    CLAIM_NONE,
    ExperimentStatus,
    count_statuses,
    status_metadata,
)


def test_closed_set_query_logits_are_not_applicable_and_not_failure():
    registry = json.loads(Path("experiments/rotation_semantic_attractor/configs/model_registry.yaml").read_text())["models"]
    capability = build_capability_profile("rotated_retinanet_msrr", registry["rotated_retinanet_msrr"])

    assert capability["model_family"] == "closed_set"
    assert capability["query_logits_status"] == ExperimentStatus.NOT_APPLICABLE
    assert capability["query_logits_reason"] == "closed-set dense/head detector does not use query logits"

    counts = count_statuses([{"status": ExperimentStatus.NOT_APPLICABLE}, {"status": ExperimentStatus.FAILED}])
    assert counts["failure_count"] == 1
    assert counts["not_applicable_count"] == 1


def test_stage_row_uses_not_applicable_for_closed_set_query_logits():
    raw_output = {
        "final_predictions": [{"class_name": "small-vehicle"}, {"class_name": "ship"}],
        "pre_nms_predictions": [{"class_name": "small-vehicle"}, {"class_name": "ship"}],
        "dense_logits": {"small_vehicle_topk_mean_score": 0.25},
        "metadata": {
            "capability_profile": {
                "model_family": "closed_set",
                "architecture_type": "dense_head",
                "query_logits_status": ExperimentStatus.NOT_APPLICABLE,
                "query_logits_reason": "closed-set dense/head detector does not use query logits",
            }
        },
    }

    row = compute_stage_row("toy", "tile", 0, raw_output)

    assert row["query_logits_status"] == ExperimentStatus.NOT_APPLICABLE
    assert row["query_logits_reason"] == "closed-set dense/head detector does not use query logits"
    assert row["status"] == ExperimentStatus.DONE_SMOKE
    assert "query_logits_not_available_for_closed_set" not in str(row)


def test_stage_row_keeps_closed_set_query_logits_not_applicable_without_metadata():
    raw_output = {
        "final_predictions": [{"class_name": "small-vehicle"}, {"class_name": "ship"}],
        "pre_nms_predictions": [{"class_name": "small-vehicle"}, {"class_name": "ship"}],
        "dense_logits": {"small_vehicle_topk_mean_score": 0.25},
    }

    row = compute_stage_row("rotated_retinanet_msrr", "tile", 0, raw_output)

    assert row["model_family"] == "closed_set"
    assert row["query_logits_status"] == ExperimentStatus.NOT_APPLICABLE
    assert row["query_logits_reason"] == "closed-set dense/head detector does not use query logits"
    assert row["status"] == ExperimentStatus.DONE_SMOKE


def test_not_selected_and_proxy_status_metadata_are_not_scientific_failures():
    not_selected = status_metadata(
        ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE,
        "open-vocabulary model was not selected in this smoke run",
        claim_level=CLAIM_NONE,
    )
    proxy = status_metadata(
        ExperimentStatus.SMOKE_PROXY,
        "schema proxy only",
        claim_level=CLAIM_NONE,
    )

    assert not_selected["is_scientific_result"] is False
    assert not_selected["include_in_main_table"] is False
    assert proxy["is_proxy"] is True
    assert proxy["include_in_main_table"] is False

    counts = count_statuses([not_selected, proxy, {"status": ExperimentStatus.FAILED}])
    assert counts["failure_count"] == 1
    assert counts["not_selected_count"] == 1
    assert counts["proxy_count"] == 1
