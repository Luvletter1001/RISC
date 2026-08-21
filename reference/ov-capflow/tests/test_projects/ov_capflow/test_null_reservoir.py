import torch

from projects.OVCapFlow.ov_capflow.null_reservoir import (
    ExplicitNullReservoir,
    null_reservoir_losses,
)


def test_null_probability_is_bounded_and_keeps_every_query():
    module = ExplicitNullReservoir(embed_dims=8)
    query = torch.randn(2, 5, 8)
    logits = module(query)
    probability = logits.sigmoid()
    assert logits.shape == (2, 5)
    assert torch.all((probability > 0) & (probability < 1))


def test_null_losses_are_finite_for_mixed_and_empty_groups():
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    matched = torch.tensor([[True, False, False]])
    target_count = torch.tensor([1.0])
    gate = torch.tensor([[0.5, 0.2, 0.1]], requires_grad=True)
    losses = null_reservoir_losses(
        null_logits=logits,
        matched_mask=matched,
        target_count=target_count,
        gate_strength=gate,
        matched_weight=1.0,
        unmatched_weight=1.0,
        mass_weight=0.1,
        gate_order_weight=0.1,
        gate_margin=0.0)
    sum(losses.values()).backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert logits.grad is not None
    assert gate.grad is not None

    empty_matched = torch.zeros(1, 3, dtype=torch.bool)
    empty_losses = null_reservoir_losses(
        null_logits=logits.detach(),
        matched_mask=empty_matched,
        target_count=torch.tensor([0.0]))
    assert all(torch.isfinite(value) for value in empty_losses.values())
