from experiments.rotation_semantic_attractor.src.model_adapters.closedset_intervention import (
    classifier_channel_indices,
)


def test_classifier_channel_indices_for_anchor_head():
    assert classifier_channel_indices(45, 15, 10) == [10, 25, 40]


def test_classifier_channel_indices_for_class_only_head():
    assert classifier_channel_indices(15, 15, 10) == [10]
