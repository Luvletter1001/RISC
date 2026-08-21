from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from experiments.rotation_semantic_attractor.src.model_adapters.open_vocab_hooks import (
    apply_openrsd_visual_support_intervention,
    tensor_checksum,
)
from experiments.rotation_semantic_attractor.src.model_adapters.openrsd_hook_registry import (
    classify_openrsd_module,
    select_openrsd_hook_modules,
)


class _Module:
    pass


def test_classify_openrsd_module_names():
    assert classify_openrsd_module("bbox_head.rtm_cls_heads.0") == "dense_logits"
    assert classify_openrsd_module("visual_support_mapping") == "visual_support_embedding"
    assert classify_openrsd_module("text_support_mapping") == "text_embedding"
    assert classify_openrsd_module("aux_bbox_head.some_score") == "alignment_score"
    assert classify_openrsd_module("backbone") is None


def test_select_openrsd_hook_modules_prioritizes_real_targets():
    named = [
        ("backbone", _Module()),
        ("bbox_head.rtm_cls_heads.0", _Module()),
        ("visual_support_mapping", _Module()),
        ("text_support_mapping", _Module()),
    ]
    rows = select_openrsd_hook_modules(named)
    assert [row["hook_target"] for row in rows] == [
        "dense_logits",
        "visual_support_embedding",
        "text_embedding",
    ]
    assert rows[0]["module_name"] == "bbox_head.rtm_cls_heads.0"


def test_visual_support_intervention_changes_small_vehicle_checksum():
    labels = np.array([10, 10, 6, 6])
    feats = np.arange(16, dtype=np.float32).reshape(4, 4)
    before = tensor_checksum(feats)
    changed, meta = apply_openrsd_visual_support_intervention(
        feats.copy(),
        labels,
        intervention="zero_sv",
        small_vehicle_id=10,
        large_vehicle_id=6,
    )

    assert meta["actual_embedding_modified"] is True
    assert meta["original_embedding_checksum"] == before
    assert meta["modified_embedding_checksum"] != before
    assert np.all(changed[labels == 10] == 0)
