from experiments.rotation_semantic_attractor.src.utils.status import (
    ExperimentStatus,
    can_be_done_full,
)


def test_open_vocab_embedding_intervention_requires_modified_embedding_checksum():
    row = {
        "status": ExperimentStatus.DONE_FULL,
        "has_open_vocab_config": True,
        "has_checkpoint": True,
        "has_prompt_or_class_embedding_path": True,
        "is_embedding_level": True,
        "actual_embedding_modified": True,
        "original_embedding_checksum": "abc",
        "modified_embedding_checksum": "def",
        "reran_inference": True,
        "has_paired_comparison": True,
        "is_prompt_only": False,
        "modified_tensor_name": "visual_support_mapping.weight",
    }

    assert can_be_done_full("open_vocab_embedding_intervention", row)

    row["modified_embedding_checksum"] = "abc"
    assert not can_be_done_full("open_vocab_embedding_intervention", row)


def test_prompt_only_is_not_embedding_level_intervention():
    row = {
        "status": ExperimentStatus.DONE_FULL,
        "has_open_vocab_config": True,
        "has_checkpoint": True,
        "has_prompt_or_class_embedding_path": True,
        "is_embedding_level": True,
        "actual_embedding_modified": True,
        "original_embedding_checksum": "abc",
        "modified_embedding_checksum": "def",
        "reran_inference": True,
        "has_paired_comparison": True,
        "is_prompt_only": True,
        "modified_tensor_name": "prompt_text",
    }

    assert not can_be_done_full("open_vocab_embedding_intervention", row)


def test_closedset_classifier_channel_intervention_requires_actual_logit_or_weight_change():
    row = {
        "status": ExperimentStatus.DONE_FULL,
        "actual_logit_modified": True,
        "actual_weight_modified": False,
        "reran_inference": True,
        "has_paired_comparison": True,
        "modified_layer_name": "bbox_head.cls_convs.0",
        "modified_tensor_name": "cls_logits",
        "modified_class_index": 10,
        "is_proxy": False,
    }

    assert can_be_done_full("closedset_classifier_channel_intervention", row)

    row["actual_logit_modified"] = False
    row["actual_weight_modified"] = False
    assert not can_be_done_full("closedset_classifier_channel_intervention", row)
