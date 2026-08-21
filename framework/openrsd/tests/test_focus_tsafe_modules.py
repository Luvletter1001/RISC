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


def test_tsafe_shadow_adapter_zero_init_is_non_invasive():
    mod = _load_module(
        "focus_tsafe_text_adapter",
        "M_AD/models/utils/focus_tsafe_text_adapter.py")
    adapter = mod.FourierShadowTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.01,
        low_rank=3)
    base = F.normalize(torch.randn(1, 2, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    class_ids = torch.tensor([[13, 2, 13]])

    out, debug = adapter(base, cue, class_ids=class_ids)

    assert torch.allclose(out, base, atol=1e-6)
    assert float(adapter.alpha_t.detach()) == 0.0
    assert adapter.alpha_t_max <= 0.01
    assert debug["final_logit_effect"] is False
    assert float(debug["text_delta_norm"].max()) == 0.0
    assert "text_interclass_cos_max" in debug


def test_tsafe_shadow_adapter_active_delta_is_bounded_and_class_gated():
    mod = _load_module(
        "focus_tsafe_text_adapter_active",
        "M_AD/models/utils/focus_tsafe_text_adapter.py")
    adapter = mod.FourierShadowTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.01,
        max_delta_norm_ratio=0.02,
        low_rank=3)
    with torch.no_grad():
        adapter.alpha_t.fill_(0.01)
    base = F.normalize(torch.randn(1, 2, 3, 4), dim=-1)
    cue = torch.randn(1, 2, 6)
    class_ids = torch.tensor([[13, 2, 13]])

    _out, debug = adapter(base, cue, class_ids=class_ids)

    assert torch.all(debug["class_mask"] == torch.tensor([True, False, True]))
    assert float(debug["text_delta_norm"][..., 1].max()) == 0.0
    assert float(debug["text_delta_norm"][..., [0, 2]].max()) > 0.0
    assert float(debug["text_delta_norm"].max()) <= 0.020001


def test_tsafe_negative_bank_is_auxiliary_and_has_required_prompts():
    mod = _load_module(
        "focus_tsafe_negative_bank",
        "M_AD/models/utils/focus_tsafe_negative_bank.py")
    bank = mod.FocusTSafeNegativeBank.default()

    assert "small vehicle viewed from above" in bank.positive_prompts
    assert "not padding border" in bank.negative_prompts
    assert bank.auxiliary_only is True
    assert bank.replaces_class_support is False
    assert len(bank.positive_prompts) == 4
    assert len(bank.negative_prompts) == 8
    embeds = bank.deterministic_embeddings(dim=8)
    assert embeds["positive"].shape == (4, 8)
    assert embeds["negative"].shape == (8, 8)


def test_tsafe_losses_default_zero_and_directional_when_enabled():
    losses = _load_module(
        "focus_tsafe_text_losses",
        "M_AD/models/losses/focus_tsafe_text_losses.py")
    base = F.normalize(torch.randn(4, 8), dim=-1)
    drifted = F.normalize(base + 0.1 * torch.randn(4, 8), dim=-1)
    same_a = torch.randn(2, 3, 4)
    same_b = same_a.clone()
    sv_high = torch.tensor([0.8, 0.7])
    sv_low = torch.tensor([0.1, 0.2])
    neg_logits = torch.tensor([[0.2, 0.3], [0.4, 0.3]])
    collapsed = F.normalize(torch.ones(3, 8), dim=-1)
    separated = F.normalize(torch.eye(3, 8), dim=-1)

    assert losses.loss_tsafe_text_anchor(drifted, base, weight=0.0).item() == 0.0
    assert losses.loss_tsafe_rotation_equivariance(
        same_a, same_b, weight=0.0).item() == 0.0
    assert losses.loss_tsafe_negative_margin(
        sv_high, neg_logits, weight=0.0).item() == 0.0
    assert losses.loss_tsafe_proto_separation(
        collapsed, weight=0.0).item() == 0.0

    assert losses.loss_tsafe_text_anchor(drifted, base, weight=1.0).item() > 0.0
    assert losses.loss_tsafe_rotation_equivariance(
        same_a, same_b, weight=1.0).item() == 0.0
    assert losses.loss_tsafe_negative_margin(
        sv_high, neg_logits, weight=1.0).item() > 0.0
    assert losses.loss_tsafe_negative_margin(
        sv_low, neg_logits, weight=1.0).item() == 0.0
    assert losses.loss_tsafe_proto_separation(
        collapsed, weight=1.0).item() > 0.0
    assert losses.loss_tsafe_proto_separation(
        separated, weight=1.0).item() == 0.0


def test_tsafe_shadow_decision_rules_are_conservative():
    mod = _load_module(
        "tsafe_shadow_signal_analysis",
        "experiments/rotation_semantic_attractor/scripts/"
        "115_tsafe_shadow_signal_analysis.py")

    assert mod.decide_shadow_status(auc=0.70, degenerate_high_rate=0.1,
                                    padding_high_rate=0.1)["status"] == "SHADOW_SIGNAL_PASS"
    assert mod.decide_shadow_status(auc=0.50, degenerate_high_rate=0.1,
                                    padding_high_rate=0.1)["status"] == "SHADOW_NO_SIGNAL"
    assert mod.decide_shadow_status(auc=0.70, degenerate_high_rate=0.7,
                                    padding_high_rate=0.1)["status"] == "SHADOW_UNSAFE_AS_GATE"


def test_tsafe_offline_calibration_is_one_way_down_only():
    mod = _load_module(
        "tsafe_offline_tiny_calibration",
        "experiments/rotation_semantic_attractor/scripts/"
        "117_tsafe_offline_tiny_calibration.py")
    focus_scores = torch.tensor([0.9, 0.5, 0.2])
    eqtext = torch.tensor([0.1, 0.4, 0.7])
    neg_max = torch.tensor([0.8, 0.4, 0.1])

    adjusted = mod.apply_logit_downweight(
        focus_scores,
        eqtext_sv_similarity=eqtext,
        max_negative_text_similarity=neg_max,
        lambda_value=0.005)

    assert torch.all(adjusted <= focus_scores)
    assert adjusted[0] < focus_scores[0]
    assert adjusted[1] == focus_scores[1]
    assert adjusted[2] == focus_scores[2]
