import math

import torch
from mmengine.registry import init_default_scope

from M_AD.models.dense_heads.sgp_rtmdet_head import (
    SGPRotatedRTMDetSepBNHead,
    diagonal_gaussian_product_logit,
    geometry_logprob_from_distances,
)


def test_diagonal_gaussian_product_prefers_matching_support_mean():
    query_mu = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    query_logvar = torch.zeros_like(query_mu)
    support_mu = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    support_logvar = torch.zeros_like(support_mu)

    logits = diagonal_gaussian_product_logit(
        query_mu, query_logvar, support_mu, support_logvar)

    assert logits.shape == (2, 2)
    assert logits[0, 0] > logits[0, 1] + 5.0
    assert logits[1, 1] > logits[1, 0] + 3.0


def test_geometry_logprob_prefers_matching_predicted_box_size():
    distances = torch.tensor([
        [10.0, 5.0, 10.0, 5.0],
        [40.0, 40.0, 40.0, 40.0],
    ])
    class_mean = torch.tensor([
        [math.log(200.0), math.log(2.0)],
        [math.log(6400.0), math.log(1.0)],
        [0.0, 0.0],
    ])
    class_std = torch.full_like(class_mean, 0.10)
    valid_mask = torch.tensor([True, True, False])

    scores = geometry_logprob_from_distances(
        distances, class_mean, class_std, valid_mask=valid_mask)

    assert scores.shape == (2, 3)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]
    assert torch.allclose(scores[:, 2], torch.zeros(2))


def test_sgp_head_forward_returns_class_logits_without_rtm_cls():
    init_default_scope('mmrotate')
    head = SGPRotatedRTMDetSepBNHead(
        num_classes=3,
        in_channels=4,
        stacked_convs=1,
        feat_channels=8,
        anchor_generator=dict(type='mmdet.MlvlPointGenerator',
                              offset=0,
                              strides=[8]),
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        angle_version='le90',
        sgp_head=dict(
            embed_channels=4,
            semantic_weight=1.0,
            geometry_weight=0.0,
            objectness_weight=0.0,
            logit_scale_init=1.0,
        ),
    )
    head.eval()

    with torch.no_grad():
        cls_scores, bbox_preds, angle_preds = head((torch.randn(2, 4, 4, 4), ))

    assert cls_scores[0].shape == (2, 3, 4, 4)
    assert bbox_preds[0].shape == (2, 4, 4, 4)
    assert angle_preds[0].shape == (2, 1, 4, 4)
    assert head.sgp_debug['used_rtm_cls'] is False


def test_sgp_head_v2_support_and_logvar_warm_start():
    init_default_scope('mmrotate')
    head = SGPRotatedRTMDetSepBNHead(
        num_classes=3,
        in_channels=4,
        stacked_convs=1,
        feat_channels=8,
        anchor_generator=dict(type='mmdet.MlvlPointGenerator',
                              offset=0,
                              strides=[8]),
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        angle_version='le90',
        sgp_head=dict(
            embed_channels=4,
            support_mu_init='orthogonal',
            support_mu_scale=2.0,
            support_mu_init_std=0.0,
            support_logvar_init=-2.0,
            query_logvar_bias_init=-2.0,
            prototype_cosine_weight=4.0,
        ),
    )
    head.init_weights()

    expected = torch.eye(3, 4) * 2.0
    assert torch.allclose(head.sgp_support_mu.detach(), expected)
    assert torch.allclose(head.sgp_support_logvar.detach(),
                          torch.full((3, 4), -2.0))
    assert torch.allclose(head.sgp_query_logvar[0].bias.detach(),
                          torch.full((4, ), -2.0))

    with torch.no_grad():
        cls_scores, _, _ = head((torch.randn(1, 4, 2, 2), ))
    assert cls_scores[0].shape == (1, 3, 2, 2)
    assert head.sgp_debug['prototype_cosine_weight'] == 4.0
