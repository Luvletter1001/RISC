import torch
from mmengine.registry import init_default_scope

from M_AD.models.dense_heads.p13_fusion_decoder_head import (
    P13FusionDecoderRotatedRTMDetSepBNHead,
    build_orthogonal_support_tokens,
)


def test_p13_orthogonal_support_tokens_have_expected_prefix():
    support = build_orthogonal_support_tokens(
        num_classes=3, embed_channels=5, scale=2.0, tail_std=0.0)

    assert support.shape == (3, 5)
    assert torch.allclose(support[:, :3], torch.eye(3) * 2.0)
    assert torch.allclose(support[:, 3:], torch.zeros(3, 2))


def test_p13_head_forward_returns_decoder_outputs_without_rtm_cls():
    init_default_scope('mmrotate')
    head = P13FusionDecoderRotatedRTMDetSepBNHead(
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
        p13_decoder=dict(
            embed_channels=4,
            support_init='orthogonal',
            support_scale=2.0,
            support_init_std=0.0,
            cls_weight=1.0,
            objectness_weight=0.25,
            bbox_delta_weight=0.05,
            angle_delta_weight=0.05,
            logit_scale_init=4.0,
        ),
    )
    head.init_weights()
    head.eval()

    expected_support = torch.eye(3, 4) * 2.0
    assert torch.allclose(head.p13_support_tokens.detach(), expected_support)

    with torch.no_grad():
        cls_scores, bbox_preds, angle_preds = head((torch.randn(2, 4, 4, 4), ))

    assert cls_scores[0].shape == (2, 3, 4, 4)
    assert bbox_preds[0].shape == (2, 4, 4, 4)
    assert angle_preds[0].shape == (2, 1, 4, 4)
    assert head.p13_debug['used_rtm_cls'] is False
    assert head.p13_debug['direct_cls_from_fusion_decoder'] is True
    assert head.p13_debug['direct_bbox_residual_from_fusion_decoder'] is True
    assert head.p13_debug['bbox_delta_weight'] == 0.05
