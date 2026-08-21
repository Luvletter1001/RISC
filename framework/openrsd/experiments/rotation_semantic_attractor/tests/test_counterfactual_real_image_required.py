from experiments.rotation_semantic_attractor.src.utils.status import (
    ExperimentStatus,
    can_be_done_full,
)


def test_context_counterfactual_requires_real_modified_image_and_rerun():
    row = {
        "status": ExperimentStatus.DONE_FULL,
        "modified_image_path": "/tmp/cf.png",
        "reran_inference": True,
        "counterfactual_is_real_image_level": True,
        "has_paired_comparison": True,
        "completed_conditions": ["object_only", "context_only"],
        "is_proxy": False,
    }

    assert can_be_done_full("context_counterfactual", row)

    row["modified_image_path"] = ""
    assert not can_be_done_full("context_counterfactual", row)


def test_context_counterfactual_rejects_ledger_proxy():
    row = {
        "status": ExperimentStatus.SMOKE_PROXY,
        "modified_image_path": "",
        "reran_inference": False,
        "counterfactual_is_real_image_level": False,
        "completed_conditions": ["object_only", "context_only"],
        "is_proxy": True,
    }

    assert not can_be_done_full("context_counterfactual", row)
