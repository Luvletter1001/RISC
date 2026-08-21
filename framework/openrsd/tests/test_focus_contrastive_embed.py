import torch
import torch.nn as nn
import torch.nn.functional as F

from M_AD.models.utils.focus_contrastive_embed import (
    OrientationConditionedContrastiveEmbed,
)
from M_AD.models.utils.focus_fourier_orientation import FourierOrientationLearner
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
        device=scores.device,
    )
    in_scores = scores[:, :, :in_len].reshape(
        batch, positions, num_in_classes, support_shot)
    cls_scores[:, :, :num_in_classes] = in_scores.max(dim=-1).values
    if support_count > in_len:
        cls_scores[:, :, -1] = scores[:, :, in_len:].max(dim=-1).values
    return cls_scores


def _baseline_logits(pred_embeds, support_feats, support_labels, log_scale, bias,
                     support_shot, num_classes, num_in_classes):
    batch, dim, height, width = pred_embeds.shape
    x = pred_embeds.permute(0, 2, 3, 1).reshape(batch, height * width, dim)
    w = F.normalize(support_feats, dim=-1)
    match = x @ w.transpose(-1, -2)
    scaled = match * log_scale.exp() + bias
    cls = _baseline_matching_scores(
        scaled,
        support_labels,
        support_shot=support_shot,
        num_classes=num_classes,
        num_in_classes=num_in_classes,
    )
    return cls.reshape(batch, height, width, cls.shape[-1]).permute(0, 3, 1, 2)


def test_zero_residual_focus_logits_match_baseline_exactly():
    torch.manual_seed(7)
    batch, dim, height, width = 2, 8, 3, 2
    support_shot = 2
    num_classes = 3
    pred_embeds = torch.randn(batch, dim, height, width)
    support_feats = torch.randn(batch, num_classes * support_shot, dim)
    support_labels = torch.tensor([[0, 0, 1, 1, 2, 2],
                                   [0, 0, 1, 1, 2, 2]])
    log_scale = nn.Parameter(torch.tensor([-1.0]))
    bias = nn.Parameter(torch.tensor([-4.0]))
    identity = nn.Identity()
    orientation = FourierOrientationLearner(
        patch_size=3,
        num_angle_bins=18,
        harmonic_orders=(2, 4),
    )
    adapter = FourierSupportResidualAdapter(
        support_dim=dim,
        code_dim=4,
        class_names=None,
        apply_to_classes=("small-vehicle",),
        alpha_init=0.0,
    )

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
        support_adapter=adapter,
        return_focus_debug=True,
        support_shot=support_shot,
        num_classes=num_classes,
        num_in_classes=num_classes,
        align_style="labelled",
        focus_class_names=("plane", "small-vehicle", "ship"),
    )
    expected = _baseline_logits(
        pred_embeds,
        support_feats,
        support_labels,
        log_scale,
        bias,
        support_shot=support_shot,
        num_classes=num_classes,
        num_in_classes=num_classes,
    )

    assert torch.equal(actual, expected)
    assert debug["enabled"] is True
    assert torch.max(debug["adapter"]["delta_norm_ratio"]) == 0
