from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine import Config

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import (
    OpenRotatedRTMDetSepBNHead,
)
from M_AD.models.utils.risc_final_readout import RISCFinalReadoutAdapter


PROJECT_ROOT = Path(__file__).parents[1]
LEGACY_A10_CONFIG = (
    PROJECT_ROOT / 'M_configs' / 'Step2_A10_Large_Pretrain_Stage3'
    / 'A10_flex_rtm_v3_1_formal.py')
S0_CONFIG = (
    PROJECT_ROOT / 'M_configs' / 'Diagnostics'
    / 'risc_openrsd_a10_final_readout_s0.py')


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


def test_construction_never_reseeds_cuda_rng(monkeypatch):
    cuda_seed_calls = []
    monkeypatch.setattr(
        torch.cuda,
        'manual_seed_all',
        lambda seed: cuda_seed_calls.append(seed))

    build_adapter()

    assert cuda_seed_calls == []


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


def test_residual_ratio_bound_uses_actual_low_norm_embedding():
    module = build_adapter(max_delta_norm_ratio=0.05)
    with torch.no_grad():
        module.raw_alpha.fill_(torch.atanh(torch.tensor(0.5)))
    value = torch.tensor([1.0, -1.0, 2.0, -2.0]).reshape(1, 4, 1, 1)
    value = value * 1e-10

    output = module(value)
    ratio = (output - value).norm(dim=1) / value.norm(dim=1)

    assert ratio.max().item() <= 0.050001
    assert module.debug_state()['max_delta_norm_ratio'] <= 0.050001


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


def test_head_helper_is_noop_when_adapter_is_absent():
    head = SimpleNamespace(risc_final_readout=None)
    value = torch.randn(1, 4, 2, 2)

    output = OpenRotatedRTMDetSepBNHead._apply_risc_final_readout(
        head, value)

    assert output is value


def test_head_helper_applies_adapter_only_to_supplied_semantic_tensor():
    adapter = build_adapter()
    with torch.no_grad():
        adapter.raw_alpha.fill_(torch.atanh(torch.tensor(0.5)))
    head = SimpleNamespace(risc_final_readout=adapter)
    semantic = torch.randn(1, 4, 2, 2)

    adapted = OpenRotatedRTMDetSepBNHead._apply_risc_final_readout(
        head, semantic)

    assert not torch.equal(adapted, semantic)


def test_legacy_a10_config_omits_risc_adapter():
    config = Config.fromfile(
        LEGACY_A10_CONFIG, import_custom_modules=False)

    assert 'risc_final_readout' not in config.model.bbox_head


def build_head(config_path):
    config = Config.fromfile(config_path, import_custom_modules=False)
    register_all_modules(init_default_scope=True)
    head_config = config.model.bbox_head.copy()
    head_config.train_cfg = None
    head_config.norm_cfg = dict(type='BN')
    return MODELS.build(head_config)


def build_s0_head():
    return build_head(S0_CONFIG)


def test_s0_config_builds_rank8_zero_alpha_adapter():
    head = build_s0_head()

    assert isinstance(head.risc_final_readout, RISCFinalReadoutAdapter)
    assert head.risc_final_readout.enabled is True
    assert head.risc_final_readout.rank == 8
    assert head.risc_final_readout.raw_alpha.item() == 0.0


def test_head_forward_keeps_parent_embedding_for_existing_auxiliary_losses():
    head = build_s0_head()
    head.eval()
    with torch.no_grad():
        head.risc_final_readout.raw_alpha.fill_(
            torch.atanh(torch.tensor(0.5)))
    parent_outputs = []
    semantic_inputs = []
    parent_handle = head.rtm_cls[0].register_forward_hook(
        lambda module, inputs, output: parent_outputs.append(output.detach()))
    semantic_handle = head.rtm_cls_heads[0].register_forward_hook(
        lambda module, inputs, output: semantic_inputs.append(
            inputs[0].detach()))
    features = tuple(
        torch.randn(1, 256, size, size)
        for size in (4, 2, 1, 1, 1))
    support = torch.randn(1, 2, 256)
    labels = torch.tensor([[0, 1]])

    with torch.no_grad():
        outputs = head(
            features,
            support,
            labels,
            labels,
            align_style='labelled',
            num_classes=2,
            num_in_classes=2,
            support_shot=1)
    parent_handle.remove()
    semantic_handle.remove()

    returned_embedding = outputs[3][0]
    assert torch.equal(returned_embedding, parent_outputs[0])
    assert not torch.equal(semantic_inputs[0], parent_outputs[0])


def _assert_tensor_tree_equal(lhs, rhs):
    assert type(lhs) is type(rhs)
    if isinstance(lhs, torch.Tensor):
        assert torch.equal(lhs, rhs)
        return
    if isinstance(lhs, (tuple, list)):
        assert len(lhs) == len(rhs)
        for lhs_item, rhs_item in zip(lhs, rhs):
            _assert_tensor_tree_equal(lhs_item, rhs_item)


def test_zero_alpha_s0_head_is_bitwise_parent_equivalent():
    torch.manual_seed(20260822)
    parent = build_head(LEGACY_A10_CONFIG)
    parent_state = parent.state_dict()
    torch.manual_seed(20260822)
    candidate = build_s0_head()
    candidate_state = candidate.state_dict()

    for name, value in parent_state.items():
        assert torch.equal(candidate_state[name], value), name

    parent.eval()
    candidate.eval()
    features = tuple(
        torch.randn(1, 256, size, size)
        for size in (4, 2, 1, 1, 1))
    support = torch.randn(1, 2, 256)
    labels = torch.tensor([[0, 1]])
    kwargs = dict(
        align_style='labelled',
        num_classes=2,
        num_in_classes=2,
        support_shot=1)
    with torch.no_grad():
        np.random.seed(20260822)
        parent_outputs = parent(
            features, support, labels, labels, **kwargs)
        np.random.seed(20260822)
        candidate_outputs = candidate(
            features, support, labels, labels, **kwargs)

    _assert_tensor_tree_equal(candidate_outputs, parent_outputs)
