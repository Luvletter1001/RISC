import torch
from torch import nn

from M_AD.models.utils.counter_support_evidence_ratio import (
    CounterSupportEvidenceRatio,
)
from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import (
    OpenRotatedRTMDetSepBNHead,
)


def _inputs():
    positive = torch.tensor(
        [[[[1.0, 0.5]], [[0.2, 0.8]]]], dtype=torch.float32)
    queries = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[0.5, 0.5]], [[0.2, 0.8]]]],
        dtype=torch.float32,
    )
    supports = torch.tensor(
        [[
            [1.0, 0.0, 0.5, 0.0],
            [0.8, 0.2, 0.4, 0.1],
            [0.0, 1.0, 0.0, 0.5],
            [0.2, 0.8, 0.1, 0.4],
        ]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    return positive, queries, supports, labels


def test_disabled_adapter_is_exact_identity():
    positive, queries, supports, labels = _inputs()
    adapter = CounterSupportEvidenceRatio(
        embed_dims=4, rank=2, enable=False)

    adjusted = adapter(
        positive, queries, supports, labels, visual_fc=nn.Identity())

    assert adjusted is positive


def test_zero_strength_adapter_is_exact_identity():
    positive, queries, supports, labels = _inputs()
    adapter = CounterSupportEvidenceRatio(
        embed_dims=4, rank=2, enable=True, init_strength=0.0)

    adjusted = adapter(
        positive, queries, supports, labels, visual_fc=nn.Identity())

    torch.testing.assert_close(adjusted, positive, rtol=0, atol=0)


def test_positive_strength_only_subtracts_counter_evidence():
    positive, queries, supports, labels = _inputs()
    adapter = CounterSupportEvidenceRatio(
        embed_dims=4, rank=2, enable=True, init_strength=0.5)

    adjusted = adapter(
        positive, queries, supports, labels, visual_fc=nn.Identity())

    assert torch.all(adjusted <= positive)
    assert torch.any(adjusted < positive)


def test_negative_raw_strength_cannot_add_counter_evidence():
    positive, queries, supports, labels = _inputs()
    adapter = CounterSupportEvidenceRatio(
        embed_dims=4, rank=2, enable=True, init_strength=-0.5)

    adjusted = adapter(
        positive, queries, supports, labels, visual_fc=nn.Identity())

    torch.testing.assert_close(adjusted, positive, rtol=0, atol=0)


def test_counter_evidence_depends_on_runtime_support():
    positive, queries, supports, labels = _inputs()
    adapter = CounterSupportEvidenceRatio(
        embed_dims=4, rank=2, enable=True, init_strength=0.5)
    swapped = supports.flip(dims=(1,))

    first = adapter(
        positive, queries, supports, labels, visual_fc=nn.Identity())
    second = adapter(
        positive, queries, swapped, labels, visual_fc=nn.Identity())

    assert not torch.equal(first, second)


def test_detection_loss_backpropagates_to_adapter_parameters():
    positive, queries, supports, labels = _inputs()
    adapter = CounterSupportEvidenceRatio(
        embed_dims=4, rank=2, enable=True, init_strength=0.5)

    adjusted = adapter(
        positive, queries, supports, labels, visual_fc=nn.Identity())
    adjusted.square().mean().backward()

    gradients = {
        name: parameter.grad
        for name, parameter in adapter.named_parameters()
    }
    assert gradients
    assert all(gradient is not None for gradient in gradients.values())
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
    assert any(torch.count_nonzero(gradient) for gradient in gradients.values())


def _bare_head(counter_support_ratio):
    head = OpenRotatedRTMDetSepBNHead.__new__(OpenRotatedRTMDetSepBNHead)
    nn.Module.__init__(head)
    head.embed_dims = 4
    head.visual_fc = nn.Identity()
    head.counter_support_ratio_cfg = dict(counter_support_ratio or {})
    head._init_counter_support_ratio()
    return head


def test_head_builds_one_shared_disabled_adapter_by_default():
    head = _bare_head(None)

    assert isinstance(
        head.counter_support_ratio, CounterSupportEvidenceRatio)
    assert head.counter_support_ratio.enable is False


def test_head_applies_configured_shared_adapter_to_logits():
    positive, queries, supports, labels = _inputs()
    head = _bare_head({
        "enable": True,
        "rank": 2,
        "init_strength": 0.5,
    })

    adjusted = head._apply_counter_support_ratio(
        positive, queries, supports, labels)

    assert adjusted.shape == positive.shape
    assert torch.all(adjusted <= positive)
    assert list(name for name, _ in head.named_modules()).count(
        "counter_support_ratio") == 1
