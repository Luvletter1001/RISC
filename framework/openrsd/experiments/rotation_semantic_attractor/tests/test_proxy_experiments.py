from experiments.rotation_semantic_attractor.src.metrics.proxy_experiments import (
    closedset_intervention_rows,
    dehub_safety_rows,
)
from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus


def test_closedset_intervention_rows_are_measured_proxies_not_not_run():
    predictions = [
        {"class_name": "small-vehicle", "score": 0.8},
        {"class_name": "ship", "score": 0.7},
        {"class_name": "small-vehicle", "score": 0.2},
    ]

    rows = closedset_intervention_rows(
        model_name="toy",
        tile_id="tile",
        angle=0,
        final_predictions=predictions,
        pre_nms_predictions=predictions,
    )

    assert rows
    assert {row["status"] for row in rows} == {ExperimentStatus.SMOKE_PROXY}
    assert {row["is_scientific_result"] for row in rows} == {False}
    assert {row["include_in_main_table"] for row in rows} == {False}
    assert any(row["intervention_type"] == "zero_small_vehicle_classifier_channel" and row["final_fr_sv"] == 0.0 for row in rows)


def test_dehub_safety_rows_use_smoke_metrics_when_available():
    rows = dehub_safety_rows(
        method="toy",
        false_hub_summary={
            "mean_final_fsv": "0.25",
            "mean_bg_fsv": "0.10",
            "mean_true_sv_recall": "0.80",
        },
        stage_summary={"mean_post_nms_fr_sv": "0.30"},
    )

    assert rows[0]["status"] == ExperimentStatus.SCHEMA_ONLY
    assert rows[0]["include_in_repair_table"] is False
    assert rows[0]["Final_FSV"] == 0.25
    assert rows[0]["BG_FSV"] == 0.10
    assert rows[0]["True_SV_Recall"] == 0.80
