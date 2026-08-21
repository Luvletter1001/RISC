import math

import pytest
import torch

from M_AD.models.utils.gaussian_semantic_scale import (
    gaussian_pos_hardneg_consistency_loss,
)


def test_g3_pos_hardneg_loss_penalizes_incompatible_high_logit_negative():
    cls_logits = torch.tensor([[0.0, 2.0, -1.0]])
    labels = torch.tensor([0])
    bbox_targets = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assign_metrics = torch.tensor([1.0])
    log_area_mean = torch.tensor([
        math.log(100.0),
        math.log(10000.0),
        math.log(100.0),
    ])
    log_area_std = torch.tensor([0.5, 0.5, 0.5])
    valid = torch.tensor([True, True, False])

    loss, debug = gaussian_pos_hardneg_consistency_loss(
        cls_logits=cls_logits,
        labels=labels,
        bbox_targets=bbox_targets,
        assign_metrics=assign_metrics,
        log_area_mean=log_area_mean,
        log_area_std=log_area_std,
        valid_mask=valid,
        margin=0.5,
        min_logprob_gap=1.0,
    )

    assert loss.item() > 0
    assert debug["num_pos"] == 1
    assert debug["num_hardneg"] == 1
    assert debug["num_candidate_pairs"] == 1
    assert debug["num_pos_with_hardneg"] == 1
    assert debug["active_violation_count"] == 1
    assert debug["hardneg_per_pos"] == pytest.approx(1.0)


def test_g3_pos_hardneg_loss_decreases_when_gt_logit_increases():
    labels = torch.tensor([0])
    bbox_targets = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assign_metrics = torch.tensor([1.0])
    log_area_mean = torch.tensor([math.log(100.0), math.log(10000.0)])
    log_area_std = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, True])

    low_gt_loss, _ = gaussian_pos_hardneg_consistency_loss(
        cls_logits=torch.tensor([[0.0, 2.0]]),
        labels=labels,
        bbox_targets=bbox_targets,
        assign_metrics=assign_metrics,
        log_area_mean=log_area_mean,
        log_area_std=log_area_std,
        valid_mask=valid,
        margin=0.5,
        min_logprob_gap=1.0,
    )
    high_gt_loss, _ = gaussian_pos_hardneg_consistency_loss(
        cls_logits=torch.tensor([[4.0, 2.0]]),
        labels=labels,
        bbox_targets=bbox_targets,
        assign_metrics=assign_metrics,
        log_area_mean=log_area_mean,
        log_area_std=log_area_std,
        valid_mask=valid,
        margin=0.5,
        min_logprob_gap=1.0,
    )

    assert high_gt_loss.item() < low_gt_loss.item()
    assert high_gt_loss.item() == pytest.approx(0.0)


def test_g3_pos_hardneg_loss_ignores_scale_compatible_negative():
    cls_logits = torch.tensor([[0.0, 2.0]])
    labels = torch.tensor([0])
    bbox_targets = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assign_metrics = torch.tensor([1.0])
    log_area_mean = torch.tensor([math.log(100.0), math.log(100.0)])
    log_area_std = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, True])

    loss, debug = gaussian_pos_hardneg_consistency_loss(
        cls_logits=cls_logits,
        labels=labels,
        bbox_targets=bbox_targets,
        assign_metrics=assign_metrics,
        log_area_mean=log_area_mean,
        log_area_std=log_area_std,
        valid_mask=valid,
        margin=0.5,
        min_logprob_gap=1.0,
    )

    assert loss.item() == pytest.approx(0.0)
    assert debug["num_hardneg"] == 0
    assert debug["num_candidate_pairs"] == 0
    assert debug["active_violation_count"] == 0


def test_g3_pos_hardneg_loss_ignores_background_label_without_indexing_error():
    cls_logits = torch.tensor([[0.0, 2.0]])
    labels = torch.tensor([2])
    bbox_targets = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assign_metrics = torch.tensor([1.0])
    log_area_mean = torch.tensor([math.log(100.0), math.log(10000.0)])
    log_area_std = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, True])

    loss, debug = gaussian_pos_hardneg_consistency_loss(
        cls_logits=cls_logits,
        labels=labels,
        bbox_targets=bbox_targets,
        assign_metrics=assign_metrics,
        log_area_mean=log_area_mean,
        log_area_std=log_area_std,
        valid_mask=valid,
        margin=0.5,
        min_logprob_gap=1.0,
    )

    assert loss.item() == pytest.approx(0.0)
    assert debug["num_pos"] == 0


def test_g3_v2_ignores_low_logit_hardneg_for_ap_sensitive_gate():
    cls_logits = torch.tensor([[0.0, 2.0]])
    labels = torch.tensor([0])
    bbox_targets = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assign_metrics = torch.tensor([1.0])
    log_area_mean = torch.tensor([math.log(100.0), math.log(10000.0)])
    log_area_std = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, True])

    loss, debug = gaussian_pos_hardneg_consistency_loss(
        cls_logits=cls_logits,
        labels=labels,
        bbox_targets=bbox_targets,
        assign_metrics=assign_metrics,
        log_area_mean=log_area_mean,
        log_area_std=log_area_std,
        valid_mask=valid,
        margin=0.5,
        min_logprob_gap=1.0,
        min_hardneg_logit=3.0,
    )

    assert loss.item() == pytest.approx(0.0)
    assert debug["num_hardneg"] == 0
    assert debug["num_ap_sensitive_pos"] == 0


def test_g3_v2_ignores_gt_prior_outlier_positive():
    cls_logits = torch.tensor([[0.0, 4.0]])
    labels = torch.tensor([0])
    bbox_targets = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assign_metrics = torch.tensor([1.0])
    log_area_mean = torch.tensor([math.log(10000.0), math.log(1.0)])
    log_area_std = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, True])

    loss, debug = gaussian_pos_hardneg_consistency_loss(
        cls_logits=cls_logits,
        labels=labels,
        bbox_targets=bbox_targets,
        assign_metrics=assign_metrics,
        log_area_mean=log_area_mean,
        log_area_std=log_area_std,
        valid_mask=valid,
        margin=0.5,
        min_logprob_gap=1.0,
        max_gt_abs_z=2.0,
    )

    assert loss.item() == pytest.approx(0.0)
    assert debug["num_hardneg"] == 0
    assert debug["num_ap_sensitive_pos"] == 0
