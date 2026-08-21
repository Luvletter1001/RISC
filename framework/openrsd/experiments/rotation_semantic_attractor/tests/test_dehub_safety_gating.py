from experiments.rotation_semantic_attractor.src.utils.status import (
    ExperimentStatus,
    can_be_done_full,
)


def test_dehub_safety_requires_baseline_and_repair_checkpoint():
    row = {
        "status": ExperimentStatus.DONE_FULL,
        "baseline_checkpoint": "/tmp/base.pth",
        "repair_checkpoint": "/tmp/repair.pth",
        "same_split": True,
        "same_angles": True,
        "same_threshold": True,
        "actual_inference_run": True,
        "has_false_hub_metrics": True,
        "has_true_sv_preservation": True,
        "has_det_per_img": True,
        "has_class_distribution_js_kl": True,
        "has_lowrisk_inflation": True,
        "has_ap50_or_proxy": True,
        "is_schema_only": False,
    }

    assert can_be_done_full("dehub_safety", row)

    row["repair_checkpoint"] = ""
    assert not can_be_done_full("dehub_safety", row)


def test_dehub_schema_only_cannot_enter_repair_table():
    row = {
        "status": ExperimentStatus.SCHEMA_ONLY,
        "baseline_checkpoint": "/tmp/base.pth",
        "repair_checkpoint": "/tmp/repair.pth",
        "same_split": True,
        "actual_inference_run": False,
        "is_schema_only": True,
    }

    assert not can_be_done_full("dehub_safety", row)
