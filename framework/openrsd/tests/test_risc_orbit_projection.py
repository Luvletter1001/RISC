import torch
import torch.nn.functional as F

from M_AD.models.utils.risc_orbit_projection import RISCOrbitLowRankProjection


def _inputs():
    support = torch.tensor(
        [[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 2.0]])
    code = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    confidence = torch.ones(1, 2)
    return support, code, confidence


def test_zero_init_is_exact_identity():
    support, code, confidence = _inputs()
    module = RISCOrbitLowRankProjection(
        support_dim=4, code_dim=2, rank=2, init_strength=0.0)

    output, debug = module(support, code, confidence)
    expected = support[None, None].expand(1, 2, 3, 4)

    assert torch.equal(output, expected)
    assert debug["projection_strength"].item() == 0.0
    assert debug["delta_norm_ratio"].max().item() == 0.0


def test_projection_is_shared_and_matches_low_rank_formula():
    support, code, confidence = _inputs()
    module = RISCOrbitLowRankProjection(
        support_dim=4,
        code_dim=2,
        rank=2,
        init_strength=0.0,
        max_strength=1.0,
        max_delta_norm_ratio=1.0,
        normalize_after_projection=False,
    )
    with torch.no_grad():
        module.basis.zero_()
        module.basis[0, 0] = 1.0
        module.basis[1, 1] = 1.0
        module.code_to_rank.weight.zero_()
        module.code_to_rank.bias.zero_()  # sigmoid -> 0.5 for both ranks
        module.raw_strength.fill_(torch.atanh(torch.tensor(0.4)))

    output, debug = module(support, code, confidence)
    expected = support.clone()
    expected[:, :2] *= 0.8  # 1 - strength(0.4) * gate(0.5)
    expected = expected[None, None].expand(1, 2, 3, 4)

    assert torch.allclose(output, expected, atol=1e-6)
    assert debug["class_mask"].tolist() == [True, True, True]
    assert torch.allclose(debug["rank_gate_mean"], torch.tensor(0.5))


def test_real_dot_product_loss_reaches_zero_initialized_strength():
    support, code, confidence = _inputs()
    module = RISCOrbitLowRankProjection(
        support_dim=4,
        code_dim=2,
        rank=2,
        init_strength=0.0,
        normalize_after_projection=False,
    )
    query = torch.tensor([[[1.0, -0.5, 0.25, 0.0], [0.2, 1.0, 0.0, -0.5]]])

    conditioned, _ = module(support, code, confidence)
    logits = torch.matmul(
        query.unsqueeze(-2), F.normalize(conditioned, dim=-1).transpose(-1, -2)
    ).squeeze(-2)
    loss = logits[..., 0].sum() - logits[..., 1].sum()
    loss.backward()

    assert module.raw_strength.grad is not None
    assert torch.isfinite(module.raw_strength.grad)
    assert module.raw_strength.grad.abs().item() > 0.0


def test_projection_respects_norm_bound_and_invalid_support_mask():
    support, code, confidence = _inputs()
    labels = torch.tensor([[0, -1, 2]])
    module = RISCOrbitLowRankProjection(
        support_dim=4,
        code_dim=2,
        rank=2,
        init_strength=0.9,
        max_strength=1.0,
        max_delta_norm_ratio=0.01,
        normalize_after_projection=False,
    )

    output, debug = module(
        support, code, confidence, support_labels=labels)
    expected_invalid = support[1].view(1, 1, 4).expand(1, 2, 4)

    assert debug["delta_norm_ratio"].max().item() <= 0.010001
    assert debug["class_mask"].tolist() == [True, False, True]
    assert torch.equal(output[:, :, 1], expected_invalid)
