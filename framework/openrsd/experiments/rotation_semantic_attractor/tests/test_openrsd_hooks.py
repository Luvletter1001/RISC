from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
nn = torch.nn

from experiments.rotation_semantic_attractor.src.model_adapters.open_vocab_hooks import (
    apply_openrsd_visual_support_intervention,
    tensor_checksum,
)
from experiments.rotation_semantic_attractor.src.model_adapters.openrsd_hook_registry import (
    RISCReadoutHookRecorder,
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


class _TinySemanticHead(nn.Module):
    def forward(self, embedding, support, labels):
        del labels
        return embedding[:, :2] + support.mean().to(embedding.dtype)


class _TinyBBoxHead(nn.Module):
    def __init__(self, with_objectness=True):
        super().__init__()
        self.rtm_cls = nn.ModuleList([nn.Identity()])
        self.risc_final_readout = nn.Identity()
        self.rtm_cls_heads = nn.ModuleList([_TinySemanticHead()])
        self.rtm_reg = nn.ModuleList([nn.Identity()])
        self.rtm_ang = nn.ModuleList([nn.Identity()])
        if with_objectness:
            self.rtm_obj = nn.ModuleList([nn.Identity()])

    def forward(self, value, support, labels):
        parent = self.rtm_cls[0](value)
        adapted = self.risc_final_readout(parent)
        semantic = self.rtm_cls_heads[0](adapted, support, labels)
        bbox = self.rtm_reg[0](value)
        angle = self.rtm_ang[0](value[:, :1])
        objectness = (
            self.rtm_obj[0](value[:, :1])
            if hasattr(self, 'rtm_obj') else None)
        return semantic, bbox, angle, objectness


class _TinyModel(nn.Module):
    def __init__(self, with_objectness=True):
        super().__init__()
        self.bbox_head = _TinyBBoxHead(with_objectness=with_objectness)

    def forward(self, value, support, labels):
        return self.bbox_head(value, support, labels)


def _assert_cpu_detached_tree(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu'
        assert value.requires_grad is False
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_cpu_detached_tree(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_cpu_detached_tree(item)


def test_risc_readout_recorder_captures_ordered_full_tensor_events():
    model = _TinyModel()
    recorder = RISCReadoutHookRecorder()
    recorder.register(model)
    value = torch.randn(1, 4, 2, 2, requires_grad=True)
    support = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    expected = model(value, support, labels)

    observed = model(value, support, labels)
    snapshot = recorder.snapshot()
    counts = recorder.validate_complete()
    recorder.close()

    for actual, reference in zip(observed, expected):
        if actual is None:
            assert reference is None
        else:
            assert torch.equal(actual, reference)
    assert counts == {
        'parent_embedding': 2,
        'final_readout_adapter': 2,
        'semantic_readout': 2,
        'bbox_regression': 2,
        'angle_prediction': 2,
        'objectness': 2,
    }
    assert [event['hook_target'] for event in snapshot['events'][:6]] == [
        'parent_embedding',
        'final_readout_adapter',
        'semantic_readout',
        'bbox_regression',
        'angle_prediction',
        'objectness',
    ]
    semantic_event = snapshot['events'][2]
    assert len(semantic_event['inputs']) == 3
    assert torch.equal(semantic_event['inputs'][1], support.detach())
    assert torch.equal(semantic_event['inputs'][2], labels)
    _assert_cpu_detached_tree(snapshot)


def test_risc_readout_recorder_records_structural_objectness_absence():
    model = _TinyModel(with_objectness=False)
    recorder = RISCReadoutHookRecorder()
    recorder.register(model)
    recorder.clear()
    model(
        torch.randn(1, 4, 2, 2),
        torch.randn(3, 4),
        torch.tensor([0, 1, 2]))

    counts = recorder.validate_complete()
    snapshot = recorder.snapshot()

    assert counts['objectness'] == 0
    assert snapshot['structurally_absent_targets'] == ['objectness']

    recorder.close()


def test_risc_readout_recorder_rejects_duplicate_registration():
    recorder = RISCReadoutHookRecorder()
    model = _TinyModel()
    recorder.register(model)

    with pytest.raises(RuntimeError, match='already registered'):
        recorder.register(model)

    recorder.close()


def test_risc_readout_snapshot_rejects_partial_feature_level_capture():
    model = _TinyModel()
    model.bbox_head.rtm_cls.append(nn.Identity())
    model.bbox_head.rtm_cls_heads.append(_TinySemanticHead())
    model.bbox_head.rtm_reg.append(nn.Identity())
    model.bbox_head.rtm_ang.append(nn.Identity())
    model.bbox_head.rtm_obj.append(nn.Identity())
    recorder = RISCReadoutHookRecorder()
    recorder.register(model)
    model(
        torch.randn(1, 4, 2, 2),
        torch.randn(3, 4),
        torch.tensor([0, 1, 2]))

    with pytest.raises(RuntimeError, match=r'bbox_head.*\.1'):
        recorder.snapshot()

    recorder.close()
