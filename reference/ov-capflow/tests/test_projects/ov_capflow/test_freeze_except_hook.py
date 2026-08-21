import pytest
import torch
import torch.nn as nn

from projects.OVCapFlow.ov_capflow.freeze_except_hook import (
    FreezeExceptHook, apply_freeze_except)
from projects.OVCapFlow.ov_capflow.ov_capflow import OVCapFlow


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_fusion = nn.Linear(4, 4)


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(), ToyLayer()])
        self.null_reservoir = nn.Linear(4, 1)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.decoder = ToyDecoder()


def test_freeze_except_hook_is_fail_closed():
    model = ToyModel()
    hook = FreezeExceptHook(
        trainable_patterns=[r'^decoder\.layers\.\d+\.semantic_fusion\.'])
    hook.apply(model)
    trainable = [name for name, value in model.named_parameters()
                 if value.requires_grad]
    assert trainable
    assert all('.semantic_fusion.' in name for name in trainable)


def test_freeze_except_hook_rejects_empty_match():
    with pytest.raises(RuntimeError, match='matched no parameters'):
        FreezeExceptHook([r'^missing\.']).apply(ToyModel())


def test_pure_freeze_helper_returns_exact_matches():
    model = ToyModel()
    matched = apply_freeze_except(
        model, [r'^decoder\.layers\.0\.semantic_fusion\.(weight|bias)$'])
    assert matched == [
        'decoder.layers.0.semantic_fusion.weight',
        'decoder.layers.0.semantic_fusion.bias',
    ]
    assert [name for name, value in model.named_parameters()
            if value.requires_grad] == matched


def test_pure_freeze_helper_rejects_invalid_and_empty_patterns():
    with pytest.raises(RuntimeError, match='matched no parameters'):
        apply_freeze_except(ToyModel(), [r'^missing\.'])
    with pytest.raises(ValueError, match='non-empty'):
        apply_freeze_except(ToyModel(), [])


def test_constructor_freezes_only_after_super_builds_every_module(monkeypatch):
    def fake_parent_init(instance, **kwargs):
        nn.Module.__init__(instance)
        instance.backbone = nn.Linear(4, 4)
        instance.bbox_head = nn.Module()
        instance.bbox_head.existence_residual = nn.Linear(256, 1)

    monkeypatch.setattr(
        'projects.OVCapFlow.ov_capflow.ov_capflow.'
        'RotatedGroundingDINO.__init__', fake_parent_init)
    model = OVCapFlow(
        train_query_groups=3,
        freeze_except_patterns=[
            r'^bbox_head\.existence_residual\.(weight|bias)$'])
    trainable = [name for name, value in model.named_parameters()
                 if value.requires_grad]
    assert trainable == [
        'bbox_head.existence_residual.weight',
        'bbox_head.existence_residual.bias',
    ]
    assert sum(value.numel() for value in model.parameters()
               if value.requires_grad) == 257
    assert not any(
        value.requires_grad for value in model.backbone.parameters())


def test_constructor_freeze_is_fail_closed_after_parent_init(monkeypatch):
    called = []

    def fake_parent_init(instance, **kwargs):
        nn.Module.__init__(instance)
        instance.parent_parameter = nn.Parameter(torch.ones(1))
        called.append(True)

    monkeypatch.setattr(
        'projects.OVCapFlow.ov_capflow.ov_capflow.'
        'RotatedGroundingDINO.__init__', fake_parent_init)
    with pytest.raises(RuntimeError, match='matched no parameters'):
        OVCapFlow(freeze_except_patterns=[r'^missing\.'])
    assert called == [True]
