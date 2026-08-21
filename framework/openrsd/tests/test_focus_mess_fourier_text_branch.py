from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module():
    return _load_module(
        "focus_mess_fourier_text_branch",
        "M_AD/models/utils/focus_mess_fourier_text_branch.py")


def test_cyclic_mix_commutes_with_c8_roll():
    mod = _module()
    branch = mod.MessDetTextDownsampleBranch(
        support_dim=4,
        code_dim=6,
        group_order=8,
        low_rank=3,
        alpha_m_init=0.0,
        alpha_m_max=0.02)
    orbit = torch.randn(2, 5, 3, 8, 4)

    mixed_after_roll = branch._cyclic_mix(torch.roll(orbit, shifts=2, dims=-2))
    rolled_after_mix = torch.roll(branch._cyclic_mix(orbit), shifts=2, dims=-2)

    assert torch.allclose(mixed_after_roll, rolled_after_mix, atol=1e-6)


def test_messdet_text_branch_zero_alpha_keeps_support_and_reports_c8_shape():
    torch.manual_seed(3)
    mod = _module()
    branch = mod.MessDetTextDownsampleBranch(
        support_dim=4,
        code_dim=6,
        group_order=8,
        low_rank=3,
        alpha_m_init=0.0,
        alpha_m_max=0.02,
        apply_to_class_ids=(13,))
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[13, 2, 13]])

    out, debug = branch(
        base,
        cue,
        confidence,
        support_labels=labels)

    expected = base[:, None].expand(1, 2, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)
    assert tuple(debug["mess_equivariant_shape"]) == (1, 3, 8, 3)
    assert float(debug["mess_delta_norm"].max()) == 0.0
    assert torch.equal(debug["class_mask"], torch.tensor([True, False, True]))


def test_messdet_text_branch_active_alpha_masks_non_target_supports():
    torch.manual_seed(5)
    mod = _module()
    branch = mod.MessDetTextDownsampleBranch(
        support_dim=4,
        code_dim=6,
        group_order=8,
        low_rank=3,
        alpha_m_init=0.0,
        alpha_m_max=0.02,
        apply_to_class_ids=(13,))
    with torch.no_grad():
        branch.alpha_m.fill_(0.02)
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[13, 2, 13]])

    _out, debug = branch(
        base,
        cue,
        confidence,
        support_labels=labels)

    assert float(debug["mess_delta_norm"][..., 1].max()) == 0.0
    assert float(debug["mess_delta_norm"][..., [0, 2]].max()) > 0.0


def test_dual_text_adapter_zero_alpha_keeps_support_and_backprops_to_alpha():
    torch.manual_seed(7)
    mod = _module()
    adapter = mod.FocusMessFourierDualTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.02,
        max_delta_norm_ratio=0.03,
        mess_cfg=dict(group_order=8, low_rank=3, alpha_m_max=0.02),
        fourier_cfg=dict(low_rank=3, alpha_t_max=0.02),
        fusion_cfg=dict(max_branch_weight=0.5))
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[13, 2, 13]])

    out, debug = adapter(
        base,
        cue,
        confidence,
        support_labels=labels)
    expected = base[:, None].expand(1, 2, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)
    assert float(debug["text_delta_norm"].max()) == 0.0
    assert 0.0 <= debug["text_branch_fusion"]["mess_branch_weight"] <= 0.5

    out[..., 0, :].sum().backward()
    assert adapter.alpha_t.grad is not None


def test_dual_text_adapter_no_target_keeps_parameters_in_graph_for_ddp():
    torch.manual_seed(11)
    mod = _module()
    adapter = mod.FocusMessFourierDualTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.02,
        max_delta_norm_ratio=0.03,
        mess_cfg=dict(group_order=8, low_rank=3, alpha_m_max=0.02),
        fourier_cfg=dict(low_rank=3, alpha_t_max=0.02),
        fusion_cfg=dict(max_branch_weight=0.5))
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[2, 3, 4]])

    out, debug = adapter(
        base,
        cue,
        confidence,
        support_labels=labels)
    expected = base[:, None].expand(1, 2, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)
    assert float(debug["text_delta_norm"].max()) == 0.0

    out.sum().backward()
    missing = [
        name for name, param in adapter.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []


def test_build_focus_text_adapter_preserves_default_fourier_type():
    mod = _module()
    adapter = mod.build_focus_text_adapter(
        eqtext_cfg=dict(enable=True, low_rank=3),
        support_dim=4,
        code_dim=6,
        class_names=("small-vehicle", "ship"))

    assert adapter.__class__.__name__ == "FourierEquivariantTextAdapter"
