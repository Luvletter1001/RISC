import torch
import torch.nn as nn
import torch.nn.functional as F

from M_AD.models.utils.focus_contrastive_embed import (
    OrientationConditionedContrastiveEmbed,
)
from M_AD.models.utils.focus_dual_support_fusion import FocusDualSupportFusion
from M_AD.models.utils.focus_fourier_orientation import FourierOrientationLearner
from M_AD.models.utils.focus_mess_fourier_text_branch import (
    FocusMessFourierDualTextAdapter,
)
from M_AD.models.utils.focus_support_adapter import FourierSupportResidualAdapter


def _baseline_matching_scores(matching_scores, support_labels, support_shot,
                              num_classes, num_in_classes):
    batch, positions, support_count = matching_scores.shape
    labels = support_labels[:, None, :].expand(batch, positions, support_count)
    scores = matching_scores.clone()
    scores[labels < 0] = -10
    in_len = num_in_classes * support_shot
    cls_scores = torch.full(
        (batch, positions, num_classes),
        float("-inf"),
        device=scores.device)
    cls_scores[:, :, :num_in_classes] = scores[:, :, :in_len].reshape(
        batch, positions, num_in_classes, support_shot).max(dim=-1).values
    if support_count > in_len:
        cls_scores[:, :, -1] = scores[:, :, in_len:].max(dim=-1).values
    return cls_scores


def _baseline_logits(pred_embeds, support_feats, support_labels, log_scale,
                     bias, support_shot, num_classes, num_in_classes):
    batch, dim, height, width = pred_embeds.shape
    x = pred_embeds.permute(0, 2, 3, 1).reshape(batch, height * width, dim)
    w = F.normalize(support_feats, dim=-1)
    scaled = (x @ w.transpose(-1, -2)) * log_scale.exp() + bias
    cls = _baseline_matching_scores(
        scaled, support_labels, support_shot, num_classes, num_in_classes)
    return cls.reshape(batch, height, width, cls.shape[-1]).permute(0, 3, 1, 2)


def test_dual_text_zero_weight_matches_focus_ovd_visual_baseline_exactly():
    torch.manual_seed(11)
    batch, dim, height, width = 1, 8, 3, 2
    support_shot = 2
    num_classes = 3
    pred_embeds = torch.randn(batch, dim, height, width)
    support_feats = torch.randn(batch, num_classes * support_shot, dim)
    text_feats = support_feats.clone()
    support_labels = torch.tensor([[0, 0, 1, 1, 2, 2]])
    identity = nn.Identity()
    log_scale = nn.Parameter(torch.tensor([-1.0]))
    bias = nn.Parameter(torch.tensor([-4.0]))
    orientation = FourierOrientationLearner(
        patch_size=3,
        num_angle_bins=18,
        harmonic_orders=(2, 4))
    visual_adapter = FourierSupportResidualAdapter(
        support_dim=dim,
        code_dim=4,
        apply_to_classes=("small-vehicle",),
        alpha_init=0.0)
    text_adapter = FocusMessFourierDualTextAdapter(
        support_dim=dim,
        code_dim=4,
        apply_to_classes=("small-vehicle",),
        alpha_t_init=0.0,
        alpha_t_max=0.02,
        max_delta_norm_ratio=0.03,
        mess_cfg=dict(low_rank=4),
        fourier_cfg=dict(low_rank=4, alpha_t_max=0.02),
        fusion_cfg=dict(max_branch_weight=0.5))
    fusion = FocusDualSupportFusion(
        visual_weight_init=0.98,
        text_weight_init=0.02,
        max_text_weight=0.0)
    focus = OrientationConditionedContrastiveEmbed()

    actual, debug = focus(
        pred_embeds,
        support_feats,
        support_labels,
        visual_fc=identity,
        text_fc=identity,
        log_scale=log_scale,
        bias=bias,
        orientation_learner=orientation,
        support_adapter=visual_adapter,
        text_support_adapter=text_adapter,
        dual_support_fusion=fusion,
        visual_support_feats=support_feats,
        text_support_feats=text_feats,
        eqtext_enabled=True,
        dual_fusion_enabled=True,
        head_residual_enabled=True,
        return_focus_debug=True,
        support_shot=support_shot,
        num_classes=num_classes,
        num_in_classes=num_classes,
        align_style="labelled",
        focus_class_names=("plane", "small-vehicle", "ship"))
    expected = _baseline_logits(
        pred_embeds,
        support_feats,
        support_labels,
        log_scale,
        bias,
        support_shot,
        num_classes,
        num_classes)

    assert torch.equal(actual, expected)
    assert debug["dual_fusion"]["dual_text_weight"] == 0.0
    assert debug["eqtext"]["text_branch_fusion"]["mess_branch_weight"] <= 0.5


def test_dual_text_nonzero_config_remains_bounded():
    fusion = FocusDualSupportFusion(
        visual_weight_init=0.5,
        text_weight_init=0.5,
        max_text_weight=0.02)
    visual = F.normalize(torch.randn(1, 6, 8), dim=-1)
    text = F.normalize(torch.randn(1, 6, 8), dim=-1)

    out, debug = fusion(
        visual,
        text,
        text_enabled=True,
        visual_alpha=torch.tensor(0.05),
        text_alpha=torch.tensor(0.02))

    assert out.shape == visual.shape
    assert 0.0 < debug["dual_text_weight"] <= 0.02
    assert torch.allclose(
        out.norm(dim=-1), torch.ones_like(out[..., 0]), atol=1e-5)
