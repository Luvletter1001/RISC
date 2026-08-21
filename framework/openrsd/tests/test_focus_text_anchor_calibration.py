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
        "focus_text_anchor_calibration",
        "M_AD/models/utils/focus_text_anchor_calibration.py")


def test_disabled_returns_identical_logits():
    mod = _module()
    layer = mod.TextAnchorCalibration(num_classes=4, enable=False)
    logits = torch.randn(2, 4, 3, 5)

    out = layer(logits)

    assert torch.allclose(out, logits, atol=0.0)
    assert out.data_ptr() == logits.data_ptr()


def test_zero_init_enabled_returns_identical_logits_and_has_alpha_grad():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=4,
        enable=True,
        use_alpha=True,
        use_beta=False,
        apply_to_classes=None,
        alpha_init=0.0)
    logits = torch.randn(2, 4, 3, 5, requires_grad=True)

    out = layer(logits)
    loss = out[:, 1].sum()
    loss.backward()

    assert torch.allclose(out, logits, atol=0.0)
    assert layer.raw_alpha.grad is not None
    assert float(layer.raw_alpha.grad.abs().sum()) > 0.0


def test_apply_to_classes_only_changes_monitored_classes():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=4,
        enable=True,
        use_alpha=True,
        use_beta=True,
        apply_to_classes=["small-vehicle", "ship"],
        class_names=["small-vehicle", "large-vehicle", "ship", "plane"],
        alpha_bound=0.05,
        beta_bound=0.20)
    with torch.no_grad():
        layer.raw_alpha.fill_(2.0)
        layer.raw_beta.fill_(2.0)
    logits = torch.ones(1, 4, 2, 2)

    out = layer(logits)

    assert torch.all(out[:, 0] > logits[:, 0])
    assert torch.allclose(out[:, 1], logits[:, 1], atol=0.0)
    assert torch.all(out[:, 2] > logits[:, 2])
    assert torch.allclose(out[:, 3], logits[:, 3], atol=0.0)
    debug = layer.debug_state()
    assert debug["calibrated_class_list"] == ["small-vehicle", "ship"]


def test_episodic_class_names_map_local_channels_to_global_classes():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=4,
        enable=True,
        use_alpha=True,
        use_beta=True,
        apply_to_classes=["small-vehicle", "ship"],
        class_names=["small-vehicle", "large-vehicle", "ship", "plane"],
        alpha_bound=0.05,
        beta_bound=0.20)
    with torch.no_grad():
        layer.raw_alpha.fill_(2.0)
        layer.raw_beta.fill_(2.0)
    logits = torch.ones(1, 3, 2, 2)

    out = layer.forward_for_class_names(
        logits, ["large-vehicle", "ship", "plane"])

    assert torch.allclose(out[:, 0], logits[:, 0], atol=0.0)
    assert torch.all(out[:, 1] > logits[:, 1])
    assert torch.allclose(out[:, 2], logits[:, 2], atol=0.0)


def test_incompatible_non_class_logits_are_identity():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=18,
        enable=True,
        use_alpha=True,
        use_beta=True,
        alpha_bound=0.05,
        beta_bound=0.20)
    with torch.no_grad():
        layer.raw_alpha.fill_(2.0)
        layer.raw_beta.fill_(2.0)
    logits = torch.randn(2, 5, 4, 4)

    out = layer(logits)

    assert torch.allclose(out, logits, atol=0.0)
    assert out.data_ptr() == logits.data_ptr()
    assert layer.last_debug["skipped_incompatible_logits"] is True
    assert layer.last_debug["incompatible_logits_shape"] == [2, 5, 4, 4]


def test_alpha_beta_bounds_are_respected():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=3,
        enable=True,
        use_alpha=True,
        use_beta=True,
        alpha_bound=0.05,
        beta_bound=0.20)
    with torch.no_grad():
        layer.raw_alpha.fill_(100.0)
        layer.raw_beta.fill_(-100.0)

    alpha = layer.alpha_values()
    beta = layer.beta_values()

    assert torch.all(alpha <= 0.05)
    assert torch.all(alpha >= -0.05)
    assert torch.all(beta <= 0.20)
    assert torch.all(beta >= -0.20)
    assert torch.isclose(alpha.abs().max(), torch.tensor(0.05), atol=1e-6)
    assert torch.isclose(beta.abs().max(), torch.tensor(0.20), atol=1e-6)


def test_use_beta_false_keeps_beta_fixed_and_not_trainable():
    mod = _module()
    layer = mod.TextAnchorCalibration(
        num_classes=3,
        enable=True,
        use_alpha=True,
        use_beta=False)

    named_params = dict(layer.named_parameters())
    named_buffers = dict(layer.named_buffers())

    assert "raw_alpha" in named_params
    assert "raw_beta" not in named_params
    assert "raw_beta" in named_buffers
    assert torch.all(layer.beta_values() == 0)
