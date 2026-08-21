import math

import pytest
import torch

from M_AD.models.utils.gaussian_semantic_scale import (
    GaussianSemanticScaleDensityHead,
    gaussian_geometry_support_stats,
    gaussian_positive_scale_consistency_loss,
    gaussian_support_negative_focal_loss,
    load_class_geometry_priors,
    semantic_scale_density_delta_outlier_loss,
    semantic_scale_density_nll_loss,
    semantic_scale_density_pair_margin_loss,
    semantic_scale_density_positive_delta_loss,
)


def test_density_head_outputs_trainable_likelihood_and_nonpositive_delta():
    head = GaussianSemanticScaleDensityHead(
        in_channels=4,
        num_classes=2,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    features = torch.ones((1, 4, 2, 2), dtype=torch.float32)
    prior_mean = torch.tensor([math.log(100.0), math.log(10000.0)])
    prior_std = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, True])
    log_area = torch.tensor([
        math.log(100.0),
        math.log(10000.0),
        math.log(100.0),
        math.log(10000.0),
    ])

    out = head(features, log_area, prior_mean, prior_std, valid)

    assert out["log_prob"].shape == (4, 2)
    assert out["uncertainty"].shape == (4, 2)
    assert out["logit_delta"].shape == (4, 2)
    assert torch.all(out["uncertainty"] >= 0)
    assert torch.all(out["logit_delta"] <= 0)
    assert torch.all(out["logit_delta"] >= -0.5)


def test_density_head_clamps_positive_delta_when_enabled():
    head = GaussianSemanticScaleDensityHead(
        in_channels=4,
        num_classes=2,
        hidden_channels=4,
        max_delta_abs=0.25,
        nonpositive_delta=False,
    )
    with torch.no_grad():
        head.net[-1].bias[2::3].fill_(1.0)
    features = torch.ones((1, 4, 1, 1), dtype=torch.float32)
    prior_mean = torch.tensor([math.log(100.0), math.log(10000.0)])
    prior_std = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, True])
    log_area = torch.tensor([math.log(100.0)])

    out = head(features, log_area, prior_mean, prior_std, valid)

    assert torch.all(out["logit_delta"] <= 0.25)
    assert out["logit_delta"][0, 0].item() == pytest.approx(0.25)


def test_load_class_geometry_priors_reads_log_area_and_log_aspect(tmp_path):
    prior_csv = tmp_path / "geometry_priors.csv"
    prior_csv.write_text(
        "\n".join([
            "class,log_area_mean,log_area_std,log_aspect_mean,log_aspect_std",
            "ship,8.0,0.5,0.7,0.2",
            "storage,6.0,0.3,0.0,0.1",
        ]) + "\n",
        encoding="utf-8",
    )

    means, stds, valid, class_names = load_class_geometry_priors(
        prior_csv,
        class_names=["storage", "ship", "plane"],
        num_classes=3,
    )

    assert class_names == ["storage", "ship", "plane"]
    assert means.shape == (3, 2)
    assert stds.shape == (3, 2)
    assert means[0].tolist() == pytest.approx([6.0, 0.0])
    assert means[1].tolist() == pytest.approx([8.0, 0.7])
    assert stds[0].tolist() == pytest.approx([0.3, 0.1])
    assert stds[1].tolist() == pytest.approx([0.5, 0.2])
    assert valid.tolist() == [True, True, False]


def test_geometry_support_log_prob_prefers_matching_aspect_prior():
    geometry = torch.tensor([
        [math.log(100.0), 0.0],
        [math.log(100.0), math.log(4.0)],
    ])
    means = torch.tensor([
        [math.log(100.0), 0.0],
        [math.log(100.0), math.log(4.0)],
    ])
    stds = torch.tensor([
        [0.5, 0.2],
        [0.5, 0.2],
    ])
    valid = torch.tensor([True, True])

    stats = gaussian_geometry_support_stats(geometry, means, stds, valid)

    assert stats["log_prob"].shape == (2, 2)
    assert stats["z"].shape == (2, 2, 2)
    assert stats["mahalanobis"].shape == (2, 2)
    assert stats["log_prob"][0, 0] > stats["log_prob"][0, 1]
    assert stats["log_prob"][1, 1] > stats["log_prob"][1, 0]


def test_density_nll_loss_uses_gt_class_only_and_reports_debug():
    log_prob = torch.tensor([
        [-0.2, -5.0],
        [-4.0, -0.3],
        [-0.1, -3.0],
    ])
    labels = torch.tensor([0, 1, 2])
    assign_metrics = torch.tensor([1.0, 0.5, 1.0])
    valid = torch.tensor([True, True])

    loss, debug = semantic_scale_density_nll_loss(
        log_prob=log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
    )

    expected = (0.2 * 1.0 + 0.3 * 0.5) / 1.5
    assert loss.item() == pytest.approx(expected)
    assert debug["num_pos"] == 2
    assert debug["mean_gt_log_prob"] == (-0.25)


def test_density_delta_outlier_loss_trains_low_support_negative_delta():
    logit_delta = torch.tensor([
        [0.0, -0.05],
        [0.0, -0.30],
    ], requires_grad=True)
    log_prob = torch.tensor([
        [-0.2, -5.0],
        [-0.3, -4.5],
    ])
    labels = torch.tensor([0, 0])
    assign_metrics = torch.tensor([1.0, 0.5])
    valid = torch.tensor([True, True])

    loss, debug = semantic_scale_density_delta_outlier_loss(
        logit_delta=logit_delta,
        log_prob=log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
        min_logprob_gap=1.0,
        target_negative_delta=0.25,
        gt_keep_weight=0.1,
        max_hardneg=1,
    )

    assert loss.item() > 0
    assert debug["num_pos"] == 2
    assert debug["num_hardneg"] == 2
    assert debug["mean_hardneg_delta"] == pytest.approx(-0.175)
    loss.backward()
    assert logit_delta.grad[0, 1] > 0
    assert logit_delta.grad[1, 1] == pytest.approx(0.0)
    assert logit_delta.grad[:, 0].abs().sum() == pytest.approx(0.0)


def test_density_delta_outlier_loss_zero_when_no_hard_negative():
    logit_delta = torch.zeros((1, 2), requires_grad=True)
    log_prob = torch.tensor([[-0.2, -0.4]])
    labels = torch.tensor([0])
    assign_metrics = torch.tensor([1.0])
    valid = torch.tensor([True, True])

    loss, debug = semantic_scale_density_delta_outlier_loss(
        logit_delta=logit_delta,
        log_prob=log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
        min_logprob_gap=1.0,
    )

    assert loss.item() == pytest.approx(0.0)
    assert debug["num_pos"] == 1
    assert debug["num_hardneg"] == 0


def test_density_delta_outlier_loss_ignores_low_score_hard_negative():
    logit_delta = torch.tensor([[0.0, -0.05]], requires_grad=True)
    log_prob = torch.tensor([[-0.2, -5.0]])
    cls_logits = torch.tensor([[0.0, torch.logit(torch.tensor(0.10))]])
    labels = torch.tensor([0])
    assign_metrics = torch.tensor([1.0])
    valid = torch.tensor([True, True])

    loss, debug = semantic_scale_density_delta_outlier_loss(
        logit_delta=logit_delta,
        log_prob=log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
        cls_logits=cls_logits,
        min_hardneg_score=0.50,
        min_logprob_gap=1.0,
        target_negative_delta=0.25,
    )

    assert loss.item() == pytest.approx(0.0)
    assert debug["num_candidate_pairs"] == 0
    assert debug["num_score_gated_pairs"] == 1


def test_density_delta_outlier_loss_keeps_high_score_hard_negative():
    logit_delta = torch.tensor([[0.0, -0.05]], requires_grad=True)
    log_prob = torch.tensor([[-0.2, -5.0]])
    cls_logits = torch.tensor([[0.0, torch.logit(torch.tensor(0.90))]])
    labels = torch.tensor([0])
    assign_metrics = torch.tensor([1.0])
    valid = torch.tensor([True, True])

    loss, debug = semantic_scale_density_delta_outlier_loss(
        logit_delta=logit_delta,
        log_prob=log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
        cls_logits=cls_logits,
        min_hardneg_score=0.50,
        min_logprob_gap=1.0,
        target_negative_delta=0.25,
    )

    assert loss.item() == pytest.approx(0.20)
    assert debug["num_candidate_pairs"] == 1
    assert debug["num_score_gated_pairs"] == 0


def test_density_positive_delta_loss_pushes_supported_gt_delta_upward():
    logit_delta = torch.tensor([[0.0, -0.2]], requires_grad=True)
    log_prob = torch.tensor([[-0.2, -5.0]])
    labels = torch.tensor([0])
    assign_metrics = torch.tensor([1.0])
    valid = torch.tensor([True, True])

    loss, debug = semantic_scale_density_positive_delta_loss(
        logit_delta=logit_delta,
        log_prob=log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
        target_positive_delta=0.20,
    )

    assert loss.item() == pytest.approx(0.20)
    assert debug["num_pos"] == 1
    assert debug["num_supported_pos"] == 1
    assert debug["mean_gt_delta"] == pytest.approx(0.0)
    loss.backward()
    assert logit_delta.grad[0, 0] < 0
    assert logit_delta.grad[0, 1].item() == pytest.approx(0.0)


def test_density_pair_margin_loss_protects_gt_against_geometry_hardneg():
    logit_delta = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    log_prob = torch.tensor([[-0.2, -5.0, -0.4]])
    cls_logits = torch.tensor([[1.0, 1.3, -4.0]])
    labels = torch.tensor([0])
    assign_metrics = torch.tensor([1.0])
    valid = torch.tensor([True, True, True])

    loss, debug = semantic_scale_density_pair_margin_loss(
        logit_delta=logit_delta,
        log_prob=log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
        cls_logits=cls_logits,
        min_hardneg_score=0.50,
        min_logprob_gap=1.0,
        margin=0.20,
        max_hardneg=1,
    )

    assert loss.item() == pytest.approx(0.50)
    assert debug["num_pos"] == 1
    assert debug["num_hardneg"] == 1
    assert debug["num_candidate_pairs"] == 1
    assert debug["active_violation_count"] == 1
    assert debug["mean_pair_margin"] == pytest.approx(-0.30)
    loss.backward()
    assert logit_delta.grad[0, 0] < 0
    assert logit_delta.grad[0, 1] > 0
    assert logit_delta.grad[0, 2].item() == pytest.approx(0.0)


def test_support_negative_focal_loss_penalizes_low_support_high_score_negatives():
    cls_logits = torch.tensor([
        [torch.logit(torch.tensor(0.90)), torch.logit(torch.tensor(0.10))],
        [torch.logit(torch.tensor(0.20)), torch.logit(torch.tensor(0.80))],
        [torch.logit(torch.tensor(0.95)), torch.logit(torch.tensor(0.20))],
    ], requires_grad=True)
    log_prob = torch.tensor([
        [-0.2, -4.5],
        [-0.4, -0.3],
        [-0.2, -5.0],
    ])
    labels = torch.tensor([2, 2, 0])
    valid = torch.tensor([True, True])

    gated_loss, gated_debug = gaussian_support_negative_focal_loss(
        cls_logits=cls_logits,
        labels=labels,
        log_prob=log_prob,
        valid_mask=valid,
        min_score=0.50,
        min_logprob_gap=1.0,
        gamma=0.0,
        max_extra_weight=0.0,
        max_hardneg=1,
    )

    assert gated_loss.item() == pytest.approx(0.0)
    assert gated_debug["num_candidate_pairs"] == 1
    assert gated_debug["num_score_gated_pairs"] == 1
    assert gated_debug["num_selected_pairs"] == 0

    loss, debug = gaussian_support_negative_focal_loss(
        cls_logits=cls_logits,
        labels=labels,
        log_prob=log_prob,
        valid_mask=valid,
        min_score=0.05,
        min_logprob_gap=1.0,
        gamma=0.0,
        max_extra_weight=0.0,
        max_hardneg=1,
    )

    assert loss.item() == pytest.approx(
        torch.nn.functional.softplus(cls_logits[0, 1]).item())
    assert debug["num_negative_locations"] == 2
    assert debug["num_candidate_pairs"] == 1
    assert debug["num_selected_pairs"] == 1
    assert debug["num_positive_locations"] == 1
    assert debug["mean_selected_score"] == pytest.approx(0.10)
    assert debug["mean_logprob_gap"] == pytest.approx(4.3)
    loss.backward()
    assert cls_logits.grad[0, 1] > 0
    assert cls_logits.grad[0, 0].item() == pytest.approx(0.0)
    assert cls_logits.grad[1].abs().sum().item() == pytest.approx(0.0)
    assert cls_logits.grad[2].abs().sum().item() == pytest.approx(0.0)


def test_positive_scale_consistency_penalizes_predicted_support_drop_only():
    target_log_prob = torch.tensor([
        [-0.2, -2.0],
        [-0.4, -0.3],
        [-0.1, -4.0],
    ])
    pred_log_prob = torch.tensor([
        [-1.4, -2.0],
        [-0.7, -0.3],
        [-5.0, -0.2],
    ], requires_grad=True)
    labels = torch.tensor([0, 0, 2])
    assign_metrics = torch.tensor([1.0, 1.0, 0.0])
    valid = torch.tensor([True, True])

    loss, debug = gaussian_positive_scale_consistency_loss(
        target_log_prob=target_log_prob,
        pred_log_prob=pred_log_prob,
        labels=labels,
        assign_metrics=assign_metrics,
        valid_mask=valid,
        max_logprob_drop=0.5,
        min_target_logprob=None,
    )

    assert loss.item() == pytest.approx((1.2 - 0.5) / 2.0)
    assert debug["num_pos"] == 2
    assert debug["num_supported_pos"] == 2
    assert debug["active_violation_count"] == 1
    assert debug["mean_logprob_drop"] == pytest.approx(0.75)
    loss.backward()
    assert pred_log_prob.grad[0, 0] < 0
    assert pred_log_prob.grad[0, 1].item() == pytest.approx(0.0)
    assert pred_log_prob.grad[1].abs().sum().item() == pytest.approx(0.0)
    assert pred_log_prob.grad[2].abs().sum().item() == pytest.approx(0.0)
