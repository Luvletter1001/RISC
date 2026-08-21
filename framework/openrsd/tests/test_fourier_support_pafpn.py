import torch

from M_AD.models.necks.fourier_support_pafpn import (
    FourierGatedGaussianChannelAdapter,
    FourierSupportCSPNeXtPAFPN,
    FourierSupportChannelAdapter,
)


def _tiny_inputs():
    return (
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    )


def _fourier_cfg(**overrides):
    cfg = dict(
        mode="fourier_channel",
        enabled_levels=[2],
        fft_size=8,
        hidden_dim=16,
        gamma_init=0.05,
        residual_scale=0.25,
        zero_init_adapter=False,
    )
    cfg.update(overrides)
    return cfg


def _gaussian_cfg(**overrides):
    cfg = dict(
        support_means=[[2.0, 0.1], [5.0, 1.2], [8.0, 0.4]],
        support_stds=[[0.5, 0.2], [0.7, 0.4], [0.8, 0.3]],
        valid_mask=[True, True, False],
        token_dim=8,
        temperature=0.75,
        gamma_init=0.05,
        residual_scale=0.25,
        zero_init_adapter=True,
    )
    cfg.update(overrides)
    return cfg


def test_fourier_adapter_descriptor_separates_frequency_patterns():
    adapter = FourierSupportChannelAdapter(
        in_channels=4,
        fft_size=16,
        hidden_dim=8,
        gamma_init=0.0,
    )
    smooth = torch.ones(1, 4, 16, 16)
    checker = torch.ones(1, 4, 16, 16)
    checker[..., ::2, 1::2] = -1
    checker[..., 1::2, ::2] = -1

    _, smooth_debug = adapter(smooth, return_debug=True)
    _, checker_debug = adapter(checker, return_debug=True)

    assert smooth_debug["fourier_descriptor"].shape == (1, 8)
    assert not torch.allclose(
        smooth_debug["fourier_descriptor"],
        checker_debug["fourier_descriptor"],
    )


def test_fourier_neck_zero_init_is_identity_against_same_pafpn():
    torch.manual_seed(101)
    base_neck = FourierSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        fourier_support_fusion=dict(enable=False),
    )
    zero_neck = FourierSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        fourier_support_fusion=_fourier_cfg(
            enabled_levels=[0, 1, 2],
            zero_init_adapter=True,
            gamma_init=0.1,
        ),
    )
    zero_neck.load_state_dict(base_neck.state_dict(), strict=False)
    base_neck.eval()
    zero_neck.eval()
    inputs = _tiny_inputs()

    with torch.no_grad():
        base_outs = base_neck(inputs)
        zero_outs = zero_neck(inputs)

    assert len(zero_outs) == 3
    assert [out.shape[1] for out in zero_outs] == [12, 12, 12]
    assert all(
        torch.allclose(zero_out, base_out)
        for zero_out, base_out in zip(zero_outs, base_outs)
    )


def test_fourier_neck_can_select_p5_only():
    torch.manual_seed(103)
    base_neck = FourierSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        fourier_support_fusion=dict(enable=False),
    )
    selective_neck = FourierSupportCSPNeXtPAFPN(
        in_channels=[8, 16, 32],
        out_channels=12,
        num_csp_blocks=1,
        norm_cfg=dict(type="BN"),
        fourier_support_fusion=_fourier_cfg(enabled_levels=[2]),
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


def test_fourier_gated_gaussian_adapter_keeps_gaussian_identity_safe():
    torch.manual_seed(107)
    adapter = FourierGatedGaussianChannelAdapter(
        in_channels=6,
        fft_size=8,
        hidden_dim=8,
        gaussian_support_cfg=_gaussian_cfg(zero_init_adapter=True),
    )
    x = torch.randn(2, 6, 8, 8)

    out, debug = adapter(x, return_debug=True)

    assert torch.allclose(out, x)
    assert debug["fourier_confidence"].shape == (2, 1, 1, 1)
    assert torch.all(debug["fourier_confidence"] >= 0)
    assert torch.all(debug["fourier_confidence"] <= 1)
