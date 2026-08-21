import torch
import torch.nn.functional as F

from M_AD.models.utils.focus_support_adapter import FourierSupportResidualAdapter


def _inputs():
    support = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        dim=-1,
    )
    code = torch.randn(2, 5, 6)
    confidence = torch.ones(2, 5)
    return support, code, confidence


def test_zero_init_residual_is_baseline_equivalent():
    support, code, confidence = _inputs()
    adapter = FourierSupportResidualAdapter(
        support_dim=4,
        code_dim=6,
        class_names=["small-vehicle", "ship", "large-vehicle"],
        apply_to_classes=["small-vehicle"],
        alpha_init=0.0,
    )

    conditioned, debug = adapter(support, code, confidence)

    expected = support[None, None].expand(2, 5, 3, 4)
    assert torch.allclose(conditioned, expected, atol=1e-6)
    assert debug["delta_norm_ratio"].max() == 0


def test_alpha_zero_residual_keeps_trainable_graph():
    support, code, confidence = _inputs()
    adapter = FourierSupportResidualAdapter(
        support_dim=4,
        code_dim=6,
        class_names=["small-vehicle", "ship", "large-vehicle"],
        apply_to_classes=["small-vehicle"],
        alpha_init=0.0,
    )

    conditioned, _ = adapter(support, code, confidence)
    loss = conditioned[:, :, 0, 0].sum()
    loss.backward()

    assert conditioned.requires_grad
    assert adapter.alpha.grad is not None
    assert adapter.alpha.grad.abs() > 0


def test_only_selected_class_receives_residual_when_nonzero():
    support, code, confidence = _inputs()
    adapter = FourierSupportResidualAdapter(
        support_dim=4,
        code_dim=6,
        class_names=["small-vehicle", "ship", "large-vehicle"],
        apply_to_classes=["small-vehicle"],
        alpha_init=0.2,
        max_delta_norm_ratio=0.5,
    )
    with torch.no_grad():
        adapter.delta[-1].weight.fill_(0.05)
        adapter.delta[-1].bias.fill_(0.05)

    conditioned, debug = adapter(support, code, confidence)

    assert not torch.allclose(conditioned[:, :, 0], support[0])
    assert torch.allclose(conditioned[:, :, 1], support[1], atol=1e-6)
    assert torch.allclose(conditioned[:, :, 2], support[2], atol=1e-6)
    assert debug["class_mask"].tolist() == [True, False, False]


def test_confidence_zero_disables_residual():
    support, code, confidence = _inputs()
    adapter = FourierSupportResidualAdapter(
        support_dim=4,
        code_dim=6,
        class_names=["small-vehicle", "ship", "large-vehicle"],
        apply_to_classes=["small-vehicle"],
        alpha_init=0.2,
    )
    with torch.no_grad():
        adapter.delta[-1].weight.fill_(0.05)
        adapter.delta[-1].bias.fill_(0.05)

    conditioned, _ = adapter(support, code, confidence * 0.0)

    expected = support[None, None].expand(2, 5, 3, 4)
    assert torch.allclose(conditioned, expected, atol=1e-6)


def test_delta_norm_ratio_is_clipped():
    support, code, confidence = _inputs()
    adapter = FourierSupportResidualAdapter(
        support_dim=4,
        code_dim=6,
        class_names=["small-vehicle", "ship", "large-vehicle"],
        apply_to_classes=["small-vehicle"],
        alpha_init=1.0,
        max_delta_norm_ratio=0.01,
    )
    with torch.no_grad():
        adapter.delta[-1].weight.fill_(1.0)
        adapter.delta[-1].bias.fill_(1.0)

    _, debug = adapter(support, code, confidence)

    assert debug["delta_norm_ratio"].max() <= 0.0101
    assert torch.isfinite(debug["inter_class_cos_before"]).all()
    assert torch.isfinite(debug["inter_class_cos_after"]).all()


def test_runtime_class_names_select_target_class_when_init_has_no_names():
    adapter = FourierSupportResidualAdapter(
        support_dim=4,
        code_dim=2,
        class_names=None,
        apply_to_classes=("small-vehicle",),
        alpha_init=1.0,
        alpha_max=1.0,
        max_delta_norm_ratio=1.0,
        normalize_after_residual=False,
    )
    with torch.no_grad():
        adapter.delta[-1].bias.fill_(0.1)

    support = torch.ones(3, 4)
    code = torch.ones(1, 2, 2)
    confidence = torch.ones(1, 2)

    conditioned, debug = adapter(
        support,
        code,
        confidence,
        class_names=("plane", "small-vehicle", "ship"),
    )

    assert debug["class_mask"].tolist() == [False, True, False]
    assert torch.allclose(conditioned[:, :, 0], support[0].expand(1, 2, 4))
    assert not torch.allclose(conditioned[:, :, 1], support[1].expand(1, 2, 4))
    assert torch.allclose(conditioned[:, :, 2], support[2].expand(1, 2, 4))
