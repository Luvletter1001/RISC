import pytest
import torch
import torch.nn.functional as F
from mmdet.models.losses import FocalLoss

from projects.OVCapFlow.ov_capflow.calibration import (
    balanced_group_classification_loss,
    calibrate_selected_log_scores,
    grounding_logits_to_class_log_scores,
    select_from_class_log_scores,
)


def _loss_module():
    return FocalLoss(
        use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean')


def test_balanced_loss_is_invariant_to_repeated_unmatched_queries():
    logits = torch.tensor([[[2.0], [-2.0], [-2.0]]], requires_grad=True)
    labels = torch.tensor([[[1.0], [0.0], [0.0]]])
    matched = torch.tensor([[True, False, False]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    weights = torch.ones(1, 3)
    loss_a, stats_a = balanced_group_classification_loss(
        _loss_module(), logits, labels, weights, valid, matched, 1.0, 1.0)

    logits_b = torch.cat([logits, logits[:, 1:].repeat(1, 4, 1)], dim=1)
    labels_b = torch.cat([labels, labels[:, 1:].repeat(1, 4, 1)], dim=1)
    matched_b = torch.cat(
        [matched, torch.zeros(1, 8, dtype=torch.bool)], dim=1)
    valid_b = torch.ones_like(labels_b, dtype=torch.bool)
    weights_b = torch.ones(1, 11)
    loss_b, stats_b = balanced_group_classification_loss(
        _loss_module(), logits_b, labels_b, weights_b, valid_b, matched_b,
        1.0, 1.0)

    torch.testing.assert_close(loss_a, loss_b)
    assert stats_a['matched_queries'] == stats_b['matched_queries'] == 1
    assert stats_b['unmatched_queries'] == 10


def test_balanced_loss_handles_empty_group_and_backpropagates():
    logits = torch.tensor([[[-1.0], [-2.0]]], requires_grad=True)
    labels = torch.zeros_like(logits)
    matched = torch.zeros(1, 2, dtype=torch.bool)
    valid = torch.ones_like(labels, dtype=torch.bool)
    loss, stats = balanced_group_classification_loss(
        _loss_module(), logits, labels, torch.ones(1, 2), valid, matched,
        1.0, 1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert stats['matched_queries'] == 0


def test_log_readout_is_label_invariant_to_common_power():
    logits = torch.tensor([
        [-80.0, -79.0, -120.0, -121.0],
        [-5.0, -6.0, -4.0, -7.0],
    ], dtype=torch.float16)
    positive_map = {1: [0, 1], 2: [2, 3]}
    log_scores = grounding_logits_to_class_log_scores(logits, positive_map)
    _, labels_a = select_from_class_log_scores(log_scores)
    _, labels_b = select_from_class_log_scores(log_scores * 32.0)
    torch.testing.assert_close(labels_a, labels_b)
    assert torch.isfinite(log_scores).all()


def test_calibration_remains_finite_with_null_and_capacity():
    selected = torch.tensor([-120.0, -2.0], dtype=torch.float16)
    null_logits = torch.tensor([20.0, -20.0], dtype=torch.float16)
    capacity = torch.tensor([1e-4, 0.9], dtype=torch.float16)
    scores = calibrate_selected_log_scores(
        selected, null_logits=null_logits, capacity=capacity,
        temperature=1.0, power=32.0)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
    assert torch.all(scores > 0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_zero_log_residual_is_bitwise_parent_identity_at_extremes(dtype):
    selected = torch.tensor(
        [-65504.0, -120.0, -80.0, -2.0, 0.0, 20.0], dtype=dtype)
    zeros = torch.zeros_like(selected)
    centered_zero = (
        F.logsigmoid(zeros.float()) - F.logsigmoid(zeros.float()))

    omitted = calibrate_selected_log_scores(
        selected, temperature=0.5, power=32.0)
    explicit = calibrate_selected_log_scores(
        selected,
        temperature=0.5,
        power=32.0,
        log_residual=centered_zero)

    assert torch.equal(omitted, explicit)


def test_nonzero_log_residual_changes_only_returned_scores():
    selected = torch.tensor([-3.0, -1.0], dtype=torch.float16)
    residual = torch.tensor([0.5, -0.25], dtype=torch.float32)
    selected_before = selected.clone()
    residual_before = residual.clone()

    parent = calibrate_selected_log_scores(selected)
    adapted = calibrate_selected_log_scores(
        selected, log_residual=residual)

    assert not torch.equal(parent, adapted)
    assert torch.equal(selected, selected_before)
    assert torch.equal(residual, residual_before)


def test_log_residual_is_added_after_legacy_calibration_before_clamp_exp():
    selected = torch.tensor([-1.0, -40.0], dtype=torch.float16)
    null_logits = torch.tensor([0.25, -0.5], dtype=torch.float16)
    capacity = torch.tensor([0.5, 0.25], dtype=torch.float16)
    residual = torch.tensor([0.4, -10.0], dtype=torch.float32)

    scores = calibrate_selected_log_scores(
        selected,
        null_logits=null_logits,
        capacity=capacity,
        temperature=4.0,
        power=2.0,
        log_residual=residual)
    expected = selected.float() * (2.0 / 4.0)
    expected = expected + F.logsigmoid(-null_logits.float())
    expected = expected + capacity.float().clamp(min=1e-8).log()
    expected = expected + residual.float()
    expected = expected.clamp(min=-80.0, max=0.0).exp()

    assert torch.equal(scores, expected)


def test_omitted_log_residual_preserves_legacy_null_capacity_bits():
    selected = torch.tensor([-120.0, -2.0], dtype=torch.float16)
    null_logits = torch.tensor([20.0, -20.0], dtype=torch.float16)
    capacity = torch.tensor([1e-4, 0.9], dtype=torch.float16)

    scores = calibrate_selected_log_scores(
        selected,
        null_logits=null_logits,
        capacity=capacity,
        temperature=1.0,
        power=32.0)
    expected = selected.float() * 32.0
    expected = expected + F.logsigmoid(-null_logits.float())
    expected = expected + capacity.float().clamp(min=1e-8).log()
    expected = expected.clamp(min=-80.0, max=0.0).exp()

    assert torch.equal(scores, expected)


def test_log_residual_shape_must_match_selected_scores():
    selected = torch.tensor([-3.0, -1.0])

    with pytest.raises(
            ValueError,
            match='log_residual must match selected_log_scores'):
        calibrate_selected_log_scores(
            selected, log_residual=torch.zeros(2, 1))
