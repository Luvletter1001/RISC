from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module():
    return _load_module(
        "focus_text_anchor_calibration_zero",
        "M_AD/models/utils/focus_text_anchor_calibration.py")


def test_zero_tac_is_equivalent_to_focus_logits_for_all_levels():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=5,
        enable=True,
        use_alpha=True,
        use_beta=True,
        alpha_init=0.0,
        beta_init=0.0)
    levels = [
        torch.randn(2, 5, 8, 8),
        torch.randn(2, 5, 4, 4),
        torch.randn(2, 5, 2, 2),
    ]

    calibrated = [layer(logits) for logits in levels]

    for before, after in zip(levels, calibrated):
        assert torch.allclose(after, before, atol=0.0)


def test_tac_never_mutates_support_embeddings():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=3,
        enable=True,
        use_alpha=True,
        use_beta=True,
        apply_to_classes=[0])
    with torch.no_grad():
        layer.raw_alpha[0] = 1.0
        layer.raw_beta[0] = 1.0
    logits = torch.randn(1, 3, 2, 2)
    support = torch.randn(1, 6, 256)
    support_before = support.clone()

    _ = layer(logits)

    assert torch.allclose(support, support_before, atol=0.0)


def test_anchor_regularizer_is_zero_at_initialization():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=3,
        enable=True,
        use_alpha=True,
        use_beta=False,
        anchor_weight=0.01)
    logits = torch.randn(1, 3, 2, 2)

    reg = layer.anchor_regularizer(logits)

    assert float(reg.detach()) == 0.0
    reg.backward()
    assert layer.raw_alpha.grad is not None
