import pytest
import torch

from M_AD.models.utils.risc_final_readout import RISCFinalReadoutAdapter


def build_adapter(**overrides):
    config = dict(
        embed_dims=4,
        rank=2,
        enabled=True,
        init_alpha=0.0,
        max_alpha=0.1,
        max_delta_norm_ratio=0.05,
        init_seed=20260822,
    )
    config.update(overrides)
    return RISCFinalReadoutAdapter(**config)


def test_zero_alpha_is_bitwise_identity():
    module = build_adapter()
    value = torch.randn(2, 4, 3, 5)

    output = module(value)

    assert torch.equal(output, value)
    assert module.debug_state()['alpha'] == 0.0
    assert module.debug_state()['max_delta_norm_ratio'] == 0.0


def test_disabled_mode_returns_original_tensor():
    module = build_adapter(enabled=False)
    value = torch.randn(1, 4, 2, 2)

    assert module(value) is value
    assert module.debug_state()['enabled'] is False


def test_construction_preserves_global_cpu_rng():
    torch.manual_seed(9)
    before = torch.random.get_rng_state().clone()

    build_adapter()

    assert torch.equal(torch.random.get_rng_state(), before)


def test_nonzero_residual_is_bounded_per_location():
    module = build_adapter(max_delta_norm_ratio=0.01)
    with torch.no_grad():
        module.raw_alpha.fill_(torch.atanh(torch.tensor(0.5)))
    value = torch.randn(2, 4, 3, 5)

    output = module(value)
    ratio = (output - value).norm(dim=1) / value.norm(dim=1).clamp_min(1e-8)

    assert not torch.equal(output, value)
    assert ratio.max().item() <= 0.010001
    assert module.debug_state()['max_delta_norm_ratio'] <= 0.010001


def test_zero_alpha_receives_nonzero_finite_gradient():
    module = build_adapter()
    value = torch.arange(1, 1 + 2 * 4 * 2 * 2, dtype=torch.float32)
    value = value.reshape(2, 4, 2, 2)
    weights = torch.linspace(-0.7, 1.3, value.numel()).reshape_as(value)

    output = module(value)
    (output * weights).sum().backward()

    assert module.raw_alpha.grad is not None
    assert torch.isfinite(module.raw_alpha.grad)
    assert module.raw_alpha.grad.abs().item() > 0.0


@pytest.mark.parametrize('shape', [(2, 4, 3), (2, 3, 4, 5)])
def test_adapter_rejects_non_nchw_or_wrong_channel_shape(shape):
    module = build_adapter()

    with pytest.raises(ValueError, match='NCHW|embed_dims'):
        module(torch.zeros(shape))


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'embed_dims': 0}, 'embed_dims'),
        ({'rank': 0}, 'rank'),
        ({'rank': 5}, 'rank'),
        ({'enabled': 1}, 'enabled'),
        ({'max_alpha': 0.0}, 'max_alpha'),
        ({'init_alpha': 0.2}, 'init_alpha'),
        ({'max_delta_norm_ratio': 0.0}, 'max_delta_norm_ratio'),
    ],
)
def test_adapter_rejects_invalid_contract(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_adapter(**overrides)
