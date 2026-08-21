import torch

from M_AD.models.backbones.gaussian_support_cspnext import (
    GaussianSupportCSPNeXt,
    GaussianSupportGlobalChannelAdapter,
    GaussianSupportTokenGate,
)


def test_gaussian_support_token_gate_preserves_shape_and_zero_init_identity():
    torch.manual_seed(7)
    gate = GaussianSupportTokenGate(
        in_channels=8,
        support_means=torch.tensor([[2.0, 0.1], [5.0, 1.2]]),
        support_stds=torch.tensor([[0.5, 0.2], [0.7, 0.4]]),
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.0,
    )
    x = torch.randn(2, 8, 4, 5)

    y, debug = gate(x, return_debug=True)

    assert y.shape == x.shape
    assert torch.allclose(y, x)
    assert debug["attention"].shape == (2, 20, 2)
    assert torch.allclose(
        debug["attention"].sum(dim=-1),
        torch.ones(2, 20),
        atol=1e-5,
    )


def test_gaussian_support_token_gate_masks_invalid_support_tokens():
    torch.manual_seed(11)
    gate = GaussianSupportTokenGate(
        in_channels=4,
        support_means=torch.tensor([[2.0, 0.1], [5.0, 1.2], [8.0, 0.3]]),
        support_stds=torch.tensor([[0.5, 0.2], [0.7, 0.4], [0.6, 0.3]]),
        valid_mask=torch.tensor([True, False, True]),
        token_dim=4,
        gamma_init=0.0,
    )
    x = torch.randn(1, 4, 3, 3)

    _, debug = gate(x, return_debug=True)

    assert torch.all(debug["attention"][..., 1] == 0)
    assert torch.allclose(
        debug["attention"][..., [0, 2]].sum(dim=-1),
        torch.ones(1, 9),
        atol=1e-5,
    )


def test_gaussian_support_token_gate_uses_gaussian_priors_not_plain_adapter():
    torch.manual_seed(13)
    support_means = torch.tensor([[1.0, 0.1], [7.0, 1.4]])
    support_stds = torch.tensor([[0.4, 0.2], [0.8, 0.5]])
    gate_a = GaussianSupportTokenGate(
        in_channels=6,
        support_means=support_means,
        support_stds=support_stds,
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.2,
    )
    shifted_support_means = torch.tensor([[3.0, 0.6], [11.0, 2.2]])
    shifted_support_stds = torch.tensor([[1.4, 0.9], [1.8, 1.1]])
    gate_b = GaussianSupportTokenGate(
        in_channels=6,
        support_means=shifted_support_means,
        support_stds=shifted_support_stds,
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.2,
    )
    gate_b.load_state_dict(
        {
            key: value
            for key, value in gate_a.state_dict().items()
            if not key.startswith("support_")
        },
        strict=False,
    )
    x = torch.randn(1, 6, 3, 4)

    y_a = gate_a(x)
    y_b = gate_b(x)

    assert not torch.allclose(y_a, y_b)


def test_gaussian_support_global_channel_adapter_is_identity_at_zero_init():
    torch.manual_seed(17)
    adapter = GaussianSupportGlobalChannelAdapter(
        in_channels=8,
        support_means=torch.tensor([[2.0, 0.1], [5.0, 1.2]]),
        support_stds=torch.tensor([[0.5, 0.2], [0.7, 0.4]]),
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.1,
        zero_init_adapter=True,
    )
    x = torch.randn(2, 8, 4, 5)

    y, debug = adapter(x, return_debug=True)

    assert y.shape == x.shape
    assert torch.allclose(y, x)
    assert debug["attention"].shape == (2, 2)
    assert debug["channel_delta"].shape == (2, 8, 1, 1)


def test_gaussian_support_global_channel_adapter_uses_support_priors():
    torch.manual_seed(19)
    support_means = torch.tensor([[1.0, 0.1], [7.0, 1.4]])
    support_stds = torch.tensor([[0.4, 0.2], [0.8, 0.5]])
    adapter_a = GaussianSupportGlobalChannelAdapter(
        in_channels=6,
        support_means=support_means,
        support_stds=support_stds,
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.2,
        zero_init_adapter=False,
    )
    adapter_b = GaussianSupportGlobalChannelAdapter(
        in_channels=6,
        support_means=torch.tensor([[3.0, 0.6], [11.0, 2.2]]),
        support_stds=torch.tensor([[1.4, 0.9], [1.8, 1.1]]),
        valid_mask=torch.tensor([True, True]),
        token_dim=4,
        gamma_init=0.2,
        zero_init_adapter=False,
    )
    adapter_b.load_state_dict(
        {
            key: value
            for key, value in adapter_a.state_dict().items()
            if not key.startswith("support_")
        },
        strict=False,
    )
    x = torch.randn(1, 6, 3, 4)

    y_a = adapter_a(x)
    y_b = adapter_b(x)

    assert not torch.allclose(y_a, y_b)


def test_gaussian_support_cspnext_keeps_backbone_output_contract():
    model = GaussianSupportCSPNeXt(
        arch="P5",
        deepen_factor=0.33,
        widen_factor=0.25,
        out_indices=(2, 3, 4),
        norm_cfg=dict(type="BN"),
        gaussian_support_adapter=dict(
            support_means=[[2.0, 0.1], [5.0, 1.2]],
            support_stds=[[0.5, 0.2], [0.7, 0.4]],
            valid_mask=[True, True],
            token_dim=8,
            gamma_init=0.0,
        ),
    )
    model.eval()

    with torch.no_grad():
        outs = model(torch.randn(1, 3, 128, 128))

    assert len(outs) == 3
    assert [out.shape[1] for out in outs] == [64, 128, 256]
    assert [tuple(out.shape[-2:]) for out in outs] == [
        (16, 16),
        (8, 8),
        (4, 4),
    ]


def test_gaussian_support_cspnext_can_use_global_channel_adapter_on_c4_c5_only():
    model = GaussianSupportCSPNeXt(
        arch="P5",
        deepen_factor=0.33,
        widen_factor=0.25,
        out_indices=(2, 3, 4),
        norm_cfg=dict(type="BN"),
        gaussian_support_adapter=dict(
            adapter_type="global_channel",
            support_means=[[2.0, 0.1], [5.0, 1.2]],
            support_stds=[[0.5, 0.2], [0.7, 0.4]],
            valid_mask=[True, True],
            stages=(3, 4),
            token_dim=8,
            gamma_init=0.0,
        ),
    )
    model.eval()

    with torch.no_grad():
        outs = model(torch.randn(1, 3, 128, 128))

    assert len(outs) == 3
    assert set(model.gaussian_support_gates.keys()) == {"3", "4"}
    assert [out.shape[1] for out in outs] == [64, 128, 256]
