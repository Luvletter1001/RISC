import ast
import inspect
import math
import textwrap

import pytest
import torch

from projects.OVCapFlow.ov_capflow.existence_residual import (
    ExistenceResidual,
    centered_existence_log_residual,
    grouped_existence_bce_loss,
)


def test_constructor_preserves_rng_and_creates_exact_zero_parameters():
    rng_before = torch.get_rng_state().clone()
    residual = ExistenceResidual()
    rng_after = torch.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert [name for name, _ in residual.named_parameters()] == [
        'weight', 'bias'
    ]
    assert residual.weight.shape == (1, 256)
    assert residual.bias.shape == (1, )
    assert sum(parameter.numel() for parameter in residual.parameters()) == 257
    assert torch.equal(residual.weight, torch.zeros_like(residual.weight))
    assert torch.equal(residual.bias, torch.zeros_like(residual.bias))


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_centered_residual_is_exact_positive_zero_at_zero(dtype):
    finfo = torch.finfo(dtype)
    logits = torch.tensor(
        [finfo.min, -1.0, 0.0, 1.0, finfo.max], dtype=dtype)

    residual = centered_existence_log_residual(logits)

    assert residual.dtype == torch.float32
    assert residual.shape == logits.shape
    assert residual[2].item() == 0.0
    assert not torch.signbit(residual[2])
    assert torch.equal(residual[2], torch.zeros((), dtype=torch.float32))
    assert torch.isfinite(residual).all()


def test_grouped_loss_matches_mixed_strata_hand_calculation():
    log_three = math.log(3.0)
    logits = torch.tensor(
        [[[0.0, log_three, log_three, -log_three]]], requires_grad=True)
    matched = torch.tensor([[[True, True, False, False]]])

    loss = grouped_existence_bce_loss(logits, matched)

    matched_mean = (math.log(2.0) + math.log(4.0 / 3.0)) / 2.0
    unmatched_mean = (math.log(4.0) + math.log(4.0 / 3.0)) / 2.0
    expected = (matched_mean + unmatched_mean) / 2.0
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float32))


@pytest.mark.parametrize('matched_value', [False, True])
def test_grouped_loss_empty_or_all_matched_is_finite_and_backpropagates(
        matched_value):
    logits = torch.tensor(
        [[[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]]], requires_grad=True)
    matched = torch.full(logits.shape, matched_value, dtype=torch.bool)

    loss = grouped_existence_bce_loss(logits, matched)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_grouped_loss_correct_infinite_logits_are_exact_zero_and_backward():
    logits = torch.tensor([[[float('inf'), -float('inf')]]],
                          requires_grad=True)
    matched = torch.tensor([[[True, False]]])

    loss = grouped_existence_bce_loss(logits, matched)
    loss.backward()

    assert torch.equal(loss, torch.zeros((), dtype=torch.float32))
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_grouped_loss_wrong_infinite_logits_never_produce_nan():
    logits = torch.tensor([[[-float('inf'), float('inf')]]],
                          requires_grad=True)
    matched = torch.tensor([[[True, False]]])

    loss = grouped_existence_bce_loss(logits, matched)
    loss.backward()

    assert torch.isinf(loss)
    assert not torch.isnan(loss)
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


@pytest.mark.parametrize('shape', [(0, 3, 600), (1, 0, 600), (1, 3, 0)])
def test_grouped_loss_rejects_empty_dimensions(shape):
    logits = torch.zeros(shape)
    matched = torch.zeros(shape, dtype=torch.bool)

    with pytest.raises(ValueError, match='dimensions must be non-empty'):
        grouped_existence_bce_loss(logits, matched)


def test_grouped_loss_rejects_invalid_shape_or_dtype():
    with pytest.raises(ValueError, match='shape \\[batch, groups, queries\\]'):
        grouped_existence_bce_loss(
            torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match='matched_mask must match logits'):
        grouped_existence_bce_loss(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 2, dtype=torch.bool))
    with pytest.raises(TypeError, match='matched_mask must be boolean'):
        grouped_existence_bce_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))


def test_grouped_loss_has_only_metadata_validation_branches_and_no_item():
    source = textwrap.dedent(inspect.getsource(grouped_existence_bce_loss))
    tree = ast.parse(source)
    function = tree.body[0]

    assert not any(
        isinstance(node, ast.Attribute) and node.attr == 'item'
        for node in ast.walk(tree))
    numerical_start = next(
        index for index, statement in enumerate(function.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(node, ast.Attribute) and node.attr == 'float'
            for node in ast.walk(statement)))
    metadata_branches = function.body[:numerical_start]
    assert len(metadata_branches) == 4
    assert all(isinstance(branch, ast.If) for branch in metadata_branches)
    assert not any(
        isinstance(node, (ast.If, ast.IfExp))
        for statement in function.body[numerical_start:]
        for node in ast.walk(statement))
    for branch in metadata_branches:
        assert not any(isinstance(node, ast.Call) for node in ast.walk(branch.test))
        accessed_attributes = {
            node.attr
            for node in ast.walk(branch.test)
            if isinstance(node, ast.Attribute)
        }
        assert accessed_attributes <= {'ndim', 'shape', 'dtype', 'bool'}


def test_zero_weight_adamw_step_leaves_control_head_bitwise_zero():
    residual = ExistenceResidual()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=1e-4)
    hidden = torch.ones(1, 2, 3, 256)
    matched = torch.tensor([[[True, False, False], [False, True, False]]])
    weight_before = residual.weight.detach().clone()
    bias_before = residual.bias.detach().clone()

    logits = residual(hidden).squeeze(-1)
    control_loss = 0.0 * grouped_existence_bce_loss(logits, matched)
    control_loss.backward()
    optimizer.step()

    assert torch.equal(residual.weight, weight_before)
    assert torch.equal(residual.bias, bias_before)
