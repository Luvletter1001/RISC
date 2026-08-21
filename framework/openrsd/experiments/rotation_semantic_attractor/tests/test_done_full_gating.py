from experiments.rotation_semantic_attractor.src.utils.status import (
    ExperimentStatus,
    can_be_done_full,
)


def test_smoke_proxy_and_schema_only_cannot_be_done_full():
    for status in (ExperimentStatus.DONE_SMOKE, ExperimentStatus.SMOKE_PROXY, ExperimentStatus.SCHEMA_ONLY):
        row = {
            "status": status,
            "split_name": "S2_final_test",
            "angle_set": "0,30,60,90,120,150,180,210,240,270,300,330",
            "actual_inference_run": True,
            "has_final_predictions": True,
            "has_gt_matching": True,
            "has_false_sv_taxonomy": True,
        }

        assert not can_be_done_full("false_hub_benchmark", row)


def test_false_hub_benchmark_done_full_requires_full_split_and_taxonomy():
    row = {
        "status": ExperimentStatus.DONE_FULL,
        "split_name": "S2_final_test",
        "angle_set": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        "actual_inference_run": True,
        "has_final_predictions": True,
        "has_gt_matching": True,
        "has_false_sv_taxonomy": True,
        "is_smoke_limit": False,
        "is_proxy": False,
    }

    assert can_be_done_full("false_hub_benchmark", row)

    row["has_gt_matching"] = False
    assert not can_be_done_full("false_hub_benchmark", row)


def test_stage_decomposition_requires_two_real_stages_not_proxy():
    row = {
        "status": ExperimentStatus.DONE_FULL,
        "actual_inference_run": True,
        "dense_logits_status": ExperimentStatus.DONE_FULL,
        "query_logits_status": ExperimentStatus.NOT_APPLICABLE,
        "pre_nms_status": ExperimentStatus.DONE_FULL,
        "post_nms_status": ExperimentStatus.DONE_FULL,
        "is_proxy": False,
    }

    assert can_be_done_full("stage_decomposition", row)

    row["pre_nms_status"] = ExperimentStatus.SMOKE_PROXY
    row["post_nms_status"] = ExperimentStatus.SCHEMA_ONLY
    assert not can_be_done_full("stage_decomposition", row)
