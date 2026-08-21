import torch

from M_AD.models.necks.gaussian_support_pafpn import (
    GaussianSupportCSPNeXtPAFPN,
    GaussianSupportCrossLevelChannelFusion,
)


def _tiny_inputs():
    return (
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    )


def _fusion_cfg(mode="per_level_channel", zero_init_adapter=True):
    return dict(
        mode=mode,
        support_means=[[2.0, 0.1], [5.0, 1.2], [8.0, 0.4]],
        support_stds=[[0.5, 0.2], [0.7, 0.4], [0.8, 0.3]],
        valid_mask=[True, True, False],
        token_dim=8,
        temperature=0.75,
        gamma_init=0.05,
        residual_scale=0.25,
        zero_init_adapter=zero_init_adapter,
    )


def test_gaussian_support_pafpn_per_level_keeps_output_contract():
    torch.manual_seed(23)
    neck = GaussianSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        gaussian_support_fusion=_fusion_cfg("per_level_channel"),
    )
    neck.eval()

    with torch.no_grad():
        outs = neck(_tiny_inputs())

    assert len(outs) == 3
    assert [out.shape[1] for out in outs] == [12, 12, 12]
    assert [tuple(out.shape[-2:]) for out in outs] == [
        (16, 16),
        (8, 8),
        (4, 4),
    ]


def test_gaussian_support_pafpn_per_level_can_select_levels():
    torch.manual_seed(41)
    base_neck = GaussianSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        gaussian_support_fusion=dict(enable=False),
    )
    selective_neck = GaussianSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        gaussian_support_fusion=dict(
            **_fusion_cfg("per_level_channel", zero_init_adapter=False),
            enabled_levels=[2],
        ),
    )
    selective_neck.load_state_dict(base_neck.state_dict(), strict=False)
    base_neck.eval()
    selective_neck.eval()
    inputs = _tiny_inputs()

    with torch.no_grad():
        base_outs = base_neck(inputs)
        selective_outs = selective_neck(inputs)

    assert torch.allclose(selective_outs[0], base_outs[0])
    assert torch.allclose(selective_outs[1], base_outs[1])
    assert not torch.allclose(selective_outs[2], base_outs[2])


def test_cross_level_fusion_is_identity_when_zero_initialized():
    torch.manual_seed(29)
    fusion = GaussianSupportCrossLevelChannelFusion(
        out_channels=6,
        num_levels=3,
        support_means=torch.tensor([[2.0, 0.1], [5.0, 1.2]]),
        support_stds=torch.tensor([[0.5, 0.2], [0.7, 0.4]]),
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.1,
        zero_init_adapter=True,
    )
    feats = (
        torch.randn(2, 6, 8, 8),
        torch.randn(2, 6, 4, 4),
        torch.randn(2, 6, 2, 2),
    )

    outs, debug = fusion(feats, return_debug=True)

    assert len(outs) == 3
    assert all(torch.allclose(out, feat) for out, feat in zip(outs, feats))
    assert debug["attention"].shape == (2, 3, 2)
    assert torch.all(debug["attention"][..., 1] > 0)


def test_cross_level_fusion_uses_gaussian_priors():
    torch.manual_seed(31)
    kwargs = dict(
        out_channels=6,
        num_levels=3,
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.2,
        residual_scale=0.5,
        zero_init_adapter=False,
    )
    fusion_a = GaussianSupportCrossLevelChannelFusion(
        support_means=torch.tensor([[1.0, 0.1], [7.0, 1.4]]),
        support_stds=torch.tensor([[0.4, 0.2], [0.8, 0.5]]),
        **kwargs,
    )
    fusion_b = GaussianSupportCrossLevelChannelFusion(
        support_means=torch.tensor([[3.0, 0.6], [11.0, 2.2]]),
        support_stds=torch.tensor([[1.4, 0.9], [1.8, 1.1]]),
        **kwargs,
    )
    fusion_b.load_state_dict(
        {
            key: value
            for key, value in fusion_a.state_dict().items()
            if not key.startswith("support_")
        },
        strict=False,
    )
    feats = (
        torch.randn(1, 6, 8, 8),
        torch.randn(1, 6, 4, 4),
        torch.randn(1, 6, 2, 2),
    )

    out_a = fusion_a(feats)
    out_b = fusion_b(feats)

    assert any(
        not torch.allclose(level_a, level_b)
        for level_a, level_b in zip(out_a, out_b)
    )


def test_gaussian_support_pafpn_cross_level_mode_keeps_output_contract():
    torch.manual_seed(37)
    neck = GaussianSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        gaussian_support_fusion=_fusion_cfg("cross_level_channel"),
    )
    neck.eval()

    with torch.no_grad():
        outs = neck(_tiny_inputs())

    assert len(outs) == 3
    assert [out.shape[1] for out in outs] == [12, 12, 12]
