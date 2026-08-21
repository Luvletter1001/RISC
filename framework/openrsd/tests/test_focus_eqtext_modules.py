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


def test_eqtext_adapter_zero_init_keeps_normalized_text_support():
    mod = _load_module(
        "focus_eqtext_adapter",
        "M_AD/models/utils/focus_eqtext_adapter.py")
    adapter = mod.FourierEquivariantTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.05,
        low_rank=3)
    base = F.normalize(torch.randn(1, 2, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    class_ids = torch.tensor([[13, 2, 13]])

    out, debug = adapter(base, cue, confidence, class_ids=class_ids)

    assert torch.allclose(out, base, atol=1e-6)
    assert float(adapter.alpha_t.detach()) == 0.0
    assert adapter.alpha_t_max <= 0.05
    assert float(debug["text_delta_norm"].max()) == 0.0
    assert float(debug["text_anchor_loss_raw"].detach()) == 0.0
    assert "text_interclass_cos" in debug


def test_eqtext_adapter_masks_non_target_classes_when_alpha_active():
    mod = _load_module(
        "focus_eqtext_adapter",
        "M_AD/models/utils/focus_eqtext_adapter.py")
    adapter = mod.FourierEquivariantTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.05,
        low_rank=3)
    with torch.no_grad():
        adapter.alpha_t.fill_(0.05)
    base = F.normalize(torch.randn(1, 2, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    class_ids = torch.tensor([[13, 2, 13]])

    _out, debug = adapter(base, cue, confidence, class_ids=class_ids)

    assert torch.all(debug["class_mask"] == torch.tensor([True, False, True]))
    assert float(debug["text_delta_norm"][..., 1].max()) == 0.0
    assert float(debug["text_delta_norm"][..., [0, 2]].max()) > 0.0
    assert float(debug["text_anchor_loss_raw"].detach()) > 0.0


def test_eqtext_interclass_cos_uses_class_prototypes_not_same_class_shots():
    mod = _load_module(
        "focus_eqtext_adapter_proto_cos",
        "M_AD/models/utils/focus_eqtext_adapter.py")
    adapter = mod.FourierEquivariantTextAdapter(
        support_dim=4,
        code_dim=6,
        alpha_t_init=0.0,
        alpha_t_max=0.05,
        low_rank=3)
    base = F.normalize(torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]]), dim=-1)
    cue = torch.randn(1, 2, 6)
    labels = torch.tensor([[0, 0, 1, 1]])

    _out, debug = adapter(
        base,
        cue,
        torch.ones(1, 2),
        support_labels=labels)

    assert float(debug["text_interclass_cos_max"].max()) == 0.0


def test_eqtext_adapter_alpha_zero_still_backpropagates_to_alpha():
    mod = _load_module(
        "focus_eqtext_adapter_grad",
        "M_AD/models/utils/focus_eqtext_adapter.py")
    adapter = mod.FourierEquivariantTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.05,
        low_rank=3)
    base = F.normalize(torch.randn(1, 2, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    class_ids = torch.tensor([[13, 2, 13]])

    out, _debug = adapter(base, cue, confidence, class_ids=class_ids)
    assert torch.allclose(out, base, atol=1e-6)
    loss = out[..., 0, :].sum()
    loss.backward()

    assert adapter.alpha_t.grad is not None


def test_visual_support_adapter_alpha_zero_still_backpropagates_to_alpha():
    mod = _load_module(
        "focus_support_adapter_grad",
        "M_AD/models/utils/focus_support_adapter.py")
    adapter = mod.FourierSupportResidualAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_classes=("small-vehicle",),
        alpha_init=0.0,
        alpha_max=0.1,
        low_rank=3)
    base = F.normalize(torch.randn(1, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    confidence = torch.ones(1, 2)
    labels = torch.tensor([[0, 1, 0]])

    out, _debug = adapter(
        base,
        cue,
        confidence,
        support_labels=labels,
        class_names=("small-vehicle", "ship"))
    expanded = base[:, None].expand_as(out)
    assert torch.allclose(out, expanded, atol=1e-6)
    out[..., 0, :].sum().backward()

    assert adapter.alpha.grad is not None


def test_dual_fusion_disabled_and_zero_alpha_are_exact_visual_fallbacks():
    mod = _load_module(
        "focus_dual_support_fusion",
        "M_AD/models/utils/focus_dual_support_fusion.py")
    fusion = mod.FocusDualSupportFusion(
        visual_weight_init=0.9,
        text_weight_init=0.1,
        max_text_weight=0.2)
    visual = F.normalize(torch.randn(1, 5, 4), dim=-1)
    text = F.normalize(torch.randn(1, 5, 4), dim=-1)

    disabled, disabled_debug = fusion(visual, text, text_enabled=False)
    zero, zero_debug = fusion(
        visual,
        text,
        text_enabled=True,
        visual_alpha=torch.tensor(0.0),
        text_alpha=torch.tensor(0.0))
    active, active_debug = fusion(
        visual,
        text,
        text_enabled=True,
        visual_alpha=torch.tensor(0.05),
        text_alpha=torch.tensor(0.05))

    assert torch.allclose(disabled, visual, atol=0.0)
    assert torch.allclose(zero, visual, atol=0.0)
    assert disabled_debug["dual_text_weight"] == 0.0
    assert zero_debug["dual_text_weight"] == 0.0
    assert 0.0 < active_debug["dual_text_weight"] <= 0.2
    assert torch.allclose(active.norm(dim=-1), torch.ones_like(active[..., 0]), atol=1e-5)


def test_dual_fusion_zero_alpha_straight_through_keeps_value_and_grad():
    mod = _load_module(
        "focus_dual_support_fusion_grad",
        "M_AD/models/utils/focus_dual_support_fusion.py")
    fusion = mod.FocusDualSupportFusion(
        visual_weight_init=0.9,
        text_weight_init=0.1,
        max_text_weight=0.2)
    visual = F.normalize(torch.randn(1, 5, 4), dim=-1)
    text = F.normalize(torch.randn(1, 5, 4), dim=-1).requires_grad_(True)
    visual_alpha = torch.nn.Parameter(torch.tensor(0.0))
    text_alpha = torch.nn.Parameter(torch.tensor(0.0))

    out, debug = fusion(
        visual,
        text,
        text_enabled=True,
        visual_alpha=visual_alpha,
        text_alpha=text_alpha)
    assert torch.allclose(out, visual, atol=0.0)
    assert debug["dual_fallback"] == "zero_alpha_straight_through"
    assert 0.0 < debug["dual_text_weight"] <= 0.2
    out.sum().backward()

    assert text.grad is not None
    assert float(text.grad.abs().sum()) > 0.0


def test_dual_fusion_broadcasts_visual_support_for_text_only_path():
    mod = _load_module(
        "focus_dual_support_fusion_broadcast",
        "M_AD/models/utils/focus_dual_support_fusion.py")
    fusion = mod.FocusDualSupportFusion(
        visual_weight_init=0.9,
        text_weight_init=0.1,
        max_text_weight=0.2)
    visual = F.normalize(torch.randn(1, 5, 4), dim=-1)
    text = F.normalize(torch.randn(1, 7, 5, 4), dim=-1)

    out, debug = fusion(
        visual,
        text,
        text_enabled=True,
        visual_alpha=torch.tensor(0.05),
        text_alpha=torch.tensor(0.05))

    assert out.shape == text.shape
    assert 0.0 < debug["dual_text_weight"] <= 0.2


def test_eqtext_losses_have_expected_guardrail_direction():
    losses = _load_module(
        "focus_eqtext_losses",
        "M_AD/models/losses/focus_eqtext_losses.py")
    base = F.normalize(torch.randn(4, 8), dim=-1)
    drifted = F.normalize(base + 0.1 * torch.randn(4, 8), dim=-1)
    same_logits = torch.randn(2, 3, 4)
    sv_high = torch.tensor([0.8, 0.7])
    sv_low = torch.tensor([0.1, 0.2])
    neg_logits = torch.tensor([[0.2, 0.3], [0.4, 0.3]])
    collapsed = F.normalize(torch.ones(3, 8), dim=-1)
    separated = F.normalize(torch.eye(3, 8), dim=-1)

    assert losses.loss_focus_text_anchor(base, base).item() == 0.0
    assert losses.loss_focus_text_anchor(drifted, base).item() > 0.0
    assert losses.loss_focus_eqtext_consistency(same_logits, same_logits).item() == 0.0
    assert losses.loss_focus_text_negative_margin(sv_high, neg_logits).item() > 0.0
    assert losses.loss_focus_text_negative_margin(sv_low, neg_logits).item() == 0.0
    assert losses.loss_focus_text_proto_separation(collapsed).item() > 0.0
    assert losses.loss_focus_text_proto_separation(separated).item() == 0.0


def test_negative_text_bank_is_auxiliary_only_and_keeps_required_prompts():
    mod = _load_module(
        "focus_negative_text_bank",
        "M_AD/models/utils/focus_negative_text_bank.py")
    bank = mod.FocusNegativeTextBank.default()

    assert "small vehicle viewed from above" in bank.positive_prompts
    assert "not padding border" in bank.negative_prompts
    assert bank.auxiliary_only is True
    assert bank.replaces_class_support is False
    assert len(bank.positive_prompts) == 4
    assert len(bank.negative_prompts) == 8
