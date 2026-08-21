import math
from types import SimpleNamespace

import pytest
import torch

from M_AD.models.dense_heads.gs3c_rtmdet_head import GSRRotatedRTMDetSepBNHead
from M_AD.models.utils.gaussian_semantic_scale import (
    GaussianScaleLogitAdapter,
    GaussianSemanticScaleDensityHead,
    gaussian_ap_constrained_support_ranking_projection_loss,
)


class IdentityAngleCoder:
    def decode(self, angle_pred, keepdim=True):
        return angle_pred


class IdentityBboxCoder:
    def decode(self, priors, bbox_pred, max_shape=None):
        return bbox_pred


def build_dummy_closed_set_head(enable=True):
    return SimpleNamespace(
        cls_out_channels=3,
        gaussian_semantic_scale_enable=enable,
        gaussian_semantic_scale_mode="continuous_logit_energy",
        gaussian_semantic_scale_z0=4.0,
        gaussian_semantic_scale_beta=math.log(4.0),
        gaussian_semantic_consistency_enable=False,
        gaussian_semantic_consistency_weight=0.05,
        gaussian_semantic_consistency_margin=0.5,
        gaussian_semantic_consistency_min_logprob_gap=1.0,
        gaussian_semantic_consistency_max_hardneg=1,
        gaussian_semantic_scale_adapter_nonpositive_delta=True,
        gaussian_semantic_density_enable=False,
        gaussian_semantic_density_loss_weight=0.05,
        gaussian_semantic_density_log_stats=False,
        gaussian_semantic_density_apply_logit_delta=False,
        gaussian_semantic_density_logit_delta_source="head",
        gaussian_semantic_density_logprob_delta_beta=1.0,
        gaussian_semantic_density_logprob_delta_threshold=-8.0,
        gaussian_semantic_density_delta_loss_enable=False,
        gaussian_semantic_density_delta_loss_weight=0.02,
        gaussian_semantic_density_delta_loss_min_logprob_gap=1.0,
        gaussian_semantic_density_delta_loss_min_hardneg_score=None,
        gaussian_semantic_density_delta_loss_min_hardneg_logit=None,
        gaussian_semantic_density_delta_loss_target_negative_delta=0.25,
        gaussian_semantic_density_delta_loss_gt_keep_weight=0.1,
        gaussian_semantic_density_delta_loss_max_hardneg=1,
        gaussian_semantic_density_positive_delta_loss_enable=False,
        gaussian_semantic_density_positive_delta_loss_weight=0.01,
        gaussian_semantic_density_positive_delta_loss_target=0.10,
        gaussian_semantic_density_positive_delta_loss_min_gt_logprob=None,
        gaussian_semantic_density_positive_delta_loss_min_gt_score=None,
        gaussian_semantic_density_pair_margin_loss_enable=False,
        gaussian_semantic_density_pair_margin_loss_weight=0.01,
        gaussian_semantic_density_pair_margin_loss_margin=0.20,
        gaussian_semantic_density_pair_margin_loss_min_logprob_gap=1.0,
        gaussian_semantic_density_pair_margin_loss_min_hardneg_score=None,
        gaussian_semantic_density_pair_margin_loss_max_hardneg=1,
        gaussian_semantic_density_support_negative_loss_enable=False,
        gaussian_semantic_density_support_negative_loss_weight=0.01,
        gaussian_semantic_density_support_negative_loss_min_score=0.05,
        gaussian_semantic_density_support_negative_loss_min_logprob_gap=1.0,
        gaussian_semantic_density_support_negative_loss_gamma=2.0,
        gaussian_semantic_density_support_negative_loss_gap_scale=8.0,
        gaussian_semantic_density_support_negative_loss_max_extra_weight=2.0,
        gaussian_semantic_density_support_negative_loss_max_hardneg=1,
        gaussian_semantic_density_scale_consistency_loss_enable=False,
        gaussian_semantic_density_scale_consistency_loss_weight=0.01,
        gaussian_semantic_density_scale_consistency_loss_max_logprob_drop=0.5,
        gaussian_semantic_density_scale_consistency_loss_min_target_logprob=None,
        gaussian_semantic_level_routing_projection_loss_enable=False,
        gaussian_semantic_level_routing_projection_loss_weight=0.01,
        gaussian_semantic_level_routing_projection_loss_max_drop=0.05,
        gaussian_semantic_level_routing_projection_loss_min_assign_metric=0.0,
        gaussian_semantic_level_routing_log_stats=False,
        gaussian_semantic_ap_safe_support_projection_enable=False,
        gaussian_semantic_ap_safe_support_projection_loss_weight=0.015,
        gaussian_semantic_ap_safe_support_projection_min_logprob_gap=1.5,
        gaussian_semantic_ap_safe_support_projection_protect_margin=0.20,
        gaussian_semantic_ap_safe_support_projection_rank_margin=0.05,
        gaussian_semantic_ap_safe_support_projection_margin_temperature=0.25,
        gaussian_semantic_ap_safe_support_projection_support_temperature=1.0,
        gaussian_semantic_ap_safe_support_projection_min_hardneg_score=0.05,
        gaussian_semantic_ap_safe_support_projection_min_hardneg_logit=None,
        gaussian_semantic_ap_safe_support_projection_max_hardneg=1,
        gaussian_semantic_ap_safe_support_projection_max_budget=1.0,
        gaussian_semantic_ap_safe_support_projection_assign_metric_power=1.0,
        gaussian_semantic_ap_safe_support_projection_detach_gt_logit=True,
        gaussian_semantic_ap_safe_support_projection_use_geometry_logprob=False,
        gaussian_semantic_ap_safe_support_projection_log_stats=False,
        gaussian_semantic_ap_safe_ranking_projection_enable=False,
        gaussian_semantic_ap_safe_ranking_projection_loss_weight=0.015,
        gaussian_semantic_ap_safe_ranking_projection_min_logprob_gap=1.5,
        gaussian_semantic_ap_safe_ranking_projection_protect_margin=0.20,
        gaussian_semantic_ap_safe_ranking_projection_margin_temperature=0.25,
        gaussian_semantic_ap_safe_ranking_projection_support_temperature=1.0,
        gaussian_semantic_ap_safe_ranking_projection_min_hardneg_score=0.05,
        gaussian_semantic_ap_safe_ranking_projection_min_hardneg_logit=None,
        gaussian_semantic_ap_safe_ranking_projection_max_hardneg=1,
        gaussian_semantic_ap_safe_ranking_projection_max_budget=1.0,
        gaussian_semantic_ap_safe_ranking_projection_assign_metric_power=1.0,
        gaussian_semantic_ap_safe_ranking_projection_use_geometry_logprob=False,
        gaussian_semantic_ap_safe_ranking_projection_log_stats=False,
        gaussian_semantic_density_use_geometry_logprob=False,
        gaussian_semantic_density_head=None,
        gaussian_semantic_geometry_enable=False,
        gaussian_semantic_geometry_mean=torch.zeros((3, 2)),
        gaussian_semantic_geometry_std=torch.ones((3, 2)),
        gaussian_semantic_geometry_valid_mask=torch.tensor([
            False, False, False]),
        gaussian_semantic_scale_adapter_max_delta_abs=0.0,
        gaussian_semantic_scale_adapter_min_abs_z=0.0,
        gaussian_semantic_scale_adapter_min_score=0.0,
        gaussian_semantic_scale_domain_mode="closed_set",
        gaussian_semantic_scale_adapter=None,
        gaussian_semantic_valid_mask=torch.tensor([True, True, False]),
        gaussian_semantic_log_area_mean=torch.tensor([
            math.log(100.0),
            math.log(10000.0),
            math.log(10000.0),
        ]),
        gaussian_semantic_log_area_std=torch.tensor([0.5, 0.5, 0.5]),
        gaussian_semantic_scale_class_names=(
            "small-vehicle",
            "plane",
            "unknown-no-prior",
        ),
        gaussian_semantic_scale_debug={
            "enable": enable,
            "domain_mode": "closed_set",
            "gaussian_energy_mode": "continuous_logit_energy",
            "num_classes": 3,
            "num_valid_priors": 2,
        },
        angle_coder=IdentityAngleCoder(),
        bbox_coder=IdentityBboxCoder(),
    )


def test_closed_set_gs3c_head_downweights_pre_topk_scores():
    head = build_dummy_closed_set_head()
    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy"},
        level_idx=0,
    )

    assert calibrated[0, 0] < 0.30
    assert calibrated[0, 1] < scores[0, 1]
    assert calibrated[0, 1] > 0.75
    assert calibrated[0, 2].item() == pytest.approx(0.70)
    assert head._last_gaussian_semantic_scale_debug["domain_mode"] == "closed_set"
    assert (
        head._last_gaussian_semantic_scale_debug["gaussian_energy_mode"]
        == "continuous_logit_energy"
    )


def test_closed_set_gs3c_classwise_z0_relaxes_selected_class():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_scale_z0_per_class = torch.tensor([4.0, 6.0, 4.0])
    head.gaussian_semantic_scale_beta_per_class = torch.tensor([
        math.log(4.0),
        math.log(4.0),
        math.log(4.0),
    ])
    scores = torch.tensor([[0.90, 0.80, 0.70]])
    plane_z5_side = math.sqrt(10000.0 * math.exp(2.5))
    bbox_pred = torch.tensor([[0.0, 0.0, plane_z5_side, plane_z5_side]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy"},
        level_idx=0,
    )

    assert calibrated[0, 0] < 0.05
    assert calibrated[0, 1] > 0.70
    assert calibrated[0, 2].item() == pytest.approx(0.70)


def test_closed_set_gs3c_head_enable_false_is_bitwise_noop():
    head = build_dummy_closed_set_head(enable=False)
    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy"},
        level_idx=0,
    )

    assert torch.equal(calibrated, scores)


def test_closed_set_level_routing_bias_penalizes_wrong_fpn_level():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_level_routing_enable = True
    head.gaussian_semantic_level_routing_learnable_gate_enable = False
    head.gaussian_semantic_level_routing_weight = 0.5
    head.gaussian_semantic_level_routing_max_bias_abs = 2.0
    head.gaussian_semantic_level_routing_bias = torch.tensor([
        [0.0, -2.0, 0.0],
        [-2.0, 0.0, 0.0],
    ])
    logits = torch.zeros((1, 3))

    routed_small = (
        GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_level_routing(
            head, logits, level_idx=0))
    routed_large = (
        GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_level_routing(
            head, logits, level_idx=1))

    assert routed_small[0, 0].item() == pytest.approx(0.0)
    assert routed_small[0, 1].item() == pytest.approx(-2.0)
    assert routed_small[0, 2].item() == pytest.approx(0.0)
    assert routed_large[0, 0].item() == pytest.approx(-2.0)
    assert routed_large[0, 1].item() == pytest.approx(0.0)
    assert routed_large[0, 2].item() == pytest.approx(0.0)


def test_closed_set_level_routing_learnable_gate_scales_bias():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_level_routing_enable = True
    head.gaussian_semantic_level_routing_learnable_gate_enable = True
    head.gaussian_semantic_level_routing_gate_logit = torch.tensor(0.0)
    head.gaussian_semantic_level_routing_bias = torch.tensor([[-2.0, 0.0, 0.0]])
    logits = torch.zeros((1, 3))

    routed = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_level_routing(
        head, logits, level_idx=0)

    assert routed[0, 0].item() == pytest.approx(-1.0)
    assert routed[0, 1].item() == pytest.approx(0.0)


def test_closed_set_level_routing_projection_loss_protects_positive_logits():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_level_routing_enable = True
    head.gaussian_semantic_level_routing_projection_loss_enable = True
    head.gaussian_semantic_level_routing_projection_loss_weight = 2.0
    head.gaussian_semantic_level_routing_projection_loss_max_drop = 0.25
    raw = torch.zeros((1, 3, 1, 2))
    routed = raw.clone()
    routed[:, 0, 0, 0] = -1.0
    routed[:, 1, 0, 1] = -0.1
    labels = [torch.tensor([[0, 1]])]
    assign_metrics = [torch.tensor([[1.0, 1.0]])]

    loss = (
        GSRRotatedRTMDetSepBNHead
        ._loss_gaussian_semantic_level_routing_projection(
            head, [raw], [routed], labels, assign_metrics))

    assert loss.item() == pytest.approx(0.75)
    debug = head._last_gaussian_semantic_level_routing_projection_debug
    assert debug["num_pos"] == 2
    assert debug["active_violation_count"] == 1


def test_closed_set_ap_safe_support_projection_protects_ap_critical_positive():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_ap_safe_support_projection_enable = True
    head.gaussian_semantic_ap_safe_support_projection_loss_weight = 1.0
    head.gaussian_semantic_ap_safe_support_projection_min_logprob_gap = 1.5
    head.gaussian_semantic_ap_safe_support_projection_protect_margin = 0.20
    head.gaussian_semantic_ap_safe_support_projection_min_hardneg_score = 0.05
    cls_scores = [torch.tensor([[[[1.0]], [[0.95]], [[-4.0]]]])]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = (
        GSRRotatedRTMDetSepBNHead
        ._loss_gaussian_semantic_ap_safe_support_projection(
            head, cls_scores, labels_list, bbox_targets_list,
            assign_metrics_list))

    assert loss.item() == pytest.approx(0.0)
    debug = head._last_gaussian_semantic_ap_safe_support_projection_debug
    assert debug["num_hardneg"] == 1
    assert debug["active_projection_count"] == 0
    assert debug["mean_safe_budget"] == pytest.approx(0.0)


def test_closed_set_ap_safe_support_projection_pushes_safe_low_support_rival_only():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_ap_safe_support_projection_enable = True
    head.gaussian_semantic_ap_safe_support_projection_loss_weight = 1.0
    head.gaussian_semantic_ap_safe_support_projection_min_logprob_gap = 1.5
    head.gaussian_semantic_ap_safe_support_projection_protect_margin = 0.20
    head.gaussian_semantic_ap_safe_support_projection_rank_margin = 0.05
    head.gaussian_semantic_ap_safe_support_projection_margin_temperature = 0.25
    head.gaussian_semantic_ap_safe_support_projection_min_hardneg_score = 0.05
    cls_score = torch.tensor(
        [[[[1.00]], [[0.40]], [[-4.00]]]], requires_grad=True)
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = (
        GSRRotatedRTMDetSepBNHead
        ._loss_gaussian_semantic_ap_safe_support_projection(
            head, [cls_score], labels_list, bbox_targets_list,
            assign_metrics_list))
    loss.backward()

    debug = head._last_gaussian_semantic_ap_safe_support_projection_debug
    assert loss.item() > 0.0
    assert debug["active_projection_count"] == 1
    assert cls_score.grad[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert cls_score.grad[0, 1, 0, 0].item() > 0.0
    assert cls_score.grad[0, 2, 0, 0].item() == pytest.approx(0.0)


def test_p11_ranking_projection_keeps_ap_critical_positive_unchanged():
    logits = torch.tensor([[1.00, 0.95, -4.00]], requires_grad=True)
    labels = torch.tensor([0])
    log_prob = torch.tensor([[0.0, -4.0, -4.0]])
    valid_mask = torch.tensor([True, True, False])
    assign_metrics = torch.tensor([1.0])

    loss, projected, debug = (
        gaussian_ap_constrained_support_ranking_projection_loss(
            cls_logits=logits,
            labels=labels,
            log_prob=log_prob,
            valid_mask=valid_mask,
            assign_metrics=assign_metrics,
            min_logprob_gap=1.5,
            protect_margin=0.20,
            margin_temperature=0.25,
            support_temperature=1.0,
            min_hardneg_score=0.05,
            max_hardneg=1,
            max_budget=1.0,
        ))

    assert torch.equal(projected, logits)
    assert loss.item() == pytest.approx(0.0)
    assert debug["num_candidate_pairs"] == 1
    assert debug["active_projection_count"] == 0
    assert debug["protected_positive_count"] == 1


def test_p11_ranking_projection_demotes_only_safe_low_support_rival():
    logits = torch.tensor([[1.00, 0.20, -4.00]], requires_grad=True)
    labels = torch.tensor([0])
    log_prob = torch.tensor([[0.0, -4.0, -4.0]])
    valid_mask = torch.tensor([True, True, False])
    assign_metrics = torch.tensor([1.0])

    loss, projected, debug = (
        gaussian_ap_constrained_support_ranking_projection_loss(
            cls_logits=logits,
            labels=labels,
            log_prob=log_prob,
            valid_mask=valid_mask,
            assign_metrics=assign_metrics,
            min_logprob_gap=1.5,
            protect_margin=0.20,
            margin_temperature=0.25,
            support_temperature=1.0,
            min_hardneg_score=0.05,
            max_hardneg=1,
            max_budget=1.0,
        ))
    loss.backward()

    assert projected[0, 0].item() == pytest.approx(logits.detach()[0, 0].item())
    assert projected[0, 1].item() < logits.detach()[0, 1].item()
    assert projected[0, 2].item() == pytest.approx(logits.detach()[0, 2].item())
    assert debug["active_projection_count"] == 1
    assert debug["protected_positive_count"] == 1
    assert logits.grad[0, 0].item() == pytest.approx(0.0)
    assert logits.grad[0, 1].item() > 0.0
    assert logits.grad[0, 2].item() == pytest.approx(0.0)


def test_closed_set_p11_ranking_projection_head_wiring_uses_geometry_support():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_log_area_mean = torch.tensor([
        math.log(100.0),
        math.log(100.0),
        math.log(100.0),
    ])
    head.gaussian_semantic_log_area_std = torch.tensor([0.5, 0.5, 0.5])
    head.gaussian_semantic_ap_safe_ranking_projection_enable = True
    head.gaussian_semantic_ap_safe_ranking_projection_loss_weight = 1.0
    head.gaussian_semantic_ap_safe_ranking_projection_min_logprob_gap = 1.5
    head.gaussian_semantic_ap_safe_ranking_projection_protect_margin = 0.20
    head.gaussian_semantic_ap_safe_ranking_projection_margin_temperature = 0.25
    head.gaussian_semantic_ap_safe_ranking_projection_support_temperature = 1.0
    head.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_score = 0.05
    head.gaussian_semantic_ap_safe_ranking_projection_min_hardneg_logit = None
    head.gaussian_semantic_ap_safe_ranking_projection_max_hardneg = 1
    head.gaussian_semantic_ap_safe_ranking_projection_max_budget = 1.0
    head.gaussian_semantic_ap_safe_ranking_projection_assign_metric_power = 1.0
    head.gaussian_semantic_ap_safe_ranking_projection_use_geometry_logprob = True
    head.gaussian_semantic_geometry_enable = True
    head.gaussian_semantic_geometry_mean = torch.tensor([
        [math.log(100.0), 0.0],
        [math.log(100.0), math.log(4.0)],
        [0.0, 0.0],
    ])
    head.gaussian_semantic_geometry_std = torch.tensor([
        [0.5, 0.1],
        [0.5, 0.1],
        [1.0, 1.0],
    ])
    head.gaussian_semantic_geometry_valid_mask = torch.tensor([
        True, True, False])
    cls_score = torch.tensor(
        [[[[1.00]], [[0.20]], [[-4.00]]]], requires_grad=True)
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = (
        GSRRotatedRTMDetSepBNHead
        ._loss_gaussian_semantic_ap_safe_ranking_projection(
            head, [cls_score], labels_list, bbox_targets_list,
            assign_metrics_list))
    loss.backward()

    debug = head._last_gaussian_semantic_ap_safe_ranking_projection_debug
    assert loss.item() > 0.0
    assert debug["geometry_logprob_used"] is True
    assert debug["active_projection_count"] == 1
    assert cls_score.grad[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert cls_score.grad[0, 1, 0, 0].item() > 0.0


def test_closed_set_level_routing_disabled_is_noop_for_feature_map():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_level_routing_enable = False
    logits = torch.randn((2, 3, 4, 5))

    routed = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_level_routing(
        head, logits, level_idx=0)

    assert torch.equal(routed, logits)


def test_closed_set_gs3c_head_loads_fitted_adapter_checkpoint(tmp_path):
    trained_adapter = GaussianScaleLogitAdapter(hidden=4, nonpositive_delta=True)
    with torch.no_grad():
        trained_adapter.net[-1].bias.fill_(1.0)
    ckpt = tmp_path / "adapter.pth"
    torch.save({"state_dict": trained_adapter.state_dict()}, ckpt)

    head = build_dummy_closed_set_head()
    head.gaussian_semantic_scale_mode = "logit_adapter"
    head.gaussian_semantic_scale_beta = 0.0
    head.gaussian_semantic_scale_adapter = GaussianScaleLogitAdapter(
        hidden=4, nonpositive_delta=True)
    head.gaussian_semantic_scale_adapter_max_delta_abs = 0.25

    GSRRotatedRTMDetSepBNHead._load_gaussian_semantic_scale_adapter_checkpoint(
        head, ckpt)

    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy"},
        level_idx=0,
    )

    expected = torch.sigmoid(torch.logit(scores[:, :2]) - 0.25)
    assert torch.allclose(calibrated[:, :2], expected, atol=1e-6)
    assert calibrated[0, 2].item() == pytest.approx(0.70)
    assert (
        head.gaussian_semantic_scale_debug["adapter_checkpoint"]
        == str(ckpt)
    )


def test_closed_set_gs3c_adapter_min_abs_z_gates_low_z_classes(tmp_path):
    trained_adapter = GaussianScaleLogitAdapter(hidden=4, nonpositive_delta=True)
    with torch.no_grad():
        trained_adapter.net[-1].bias.fill_(1.0)
    ckpt = tmp_path / "adapter.pth"
    torch.save({"state_dict": trained_adapter.state_dict()}, ckpt)

    head = build_dummy_closed_set_head()
    head.gaussian_semantic_scale_mode = "logit_adapter"
    head.gaussian_semantic_scale_beta = 0.0
    head.gaussian_semantic_scale_adapter = GaussianScaleLogitAdapter(
        hidden=4, nonpositive_delta=True)
    head.gaussian_semantic_scale_adapter_max_delta_abs = 0.25
    head.gaussian_semantic_scale_adapter_min_abs_z = 5.0

    GSRRotatedRTMDetSepBNHead._load_gaussian_semantic_scale_adapter_checkpoint(
        head, ckpt)

    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy"},
        level_idx=0,
    )

    assert calibrated[0, 0] < scores[0, 0]
    assert calibrated[0, 1].item() == pytest.approx(0.80)
    assert calibrated[0, 2].item() == pytest.approx(0.70)
    assert head._last_gaussian_semantic_scale_debug[
        "last_num_adapter_pairs"] == 1


def test_closed_set_gs3c_g3_loss_penalizes_scale_incompatible_hardneg():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_consistency_enable = True
    head.gaussian_semantic_consistency_weight = 0.5
    cls_scores = [torch.tensor([[[[0.0]], [[2.0]], [[-1.0]]]])]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_semantic_consistency(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() > 0
    assert (
        head._last_gaussian_semantic_consistency_debug["num_hardneg"]
        == 1
    )
    assert (
        head._last_gaussian_semantic_consistency_debug["weighted_loss"]
        == pytest.approx(loss.item())
    )


def test_closed_set_gs3c_g3_loss_disabled_returns_zero():
    head = build_dummy_closed_set_head()
    cls_scores = [torch.tensor([[[[0.0]], [[2.0]], [[-1.0]]]])]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_semantic_consistency(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() == pytest.approx(0.0)
    assert (
        head._last_gaussian_semantic_consistency_debug["num_hardneg"]
        == 0
    )


def test_closed_set_density_loss_uses_gt_class_scale_likelihood():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.5
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
    )
    cls_scores = [torch.zeros((1, 3, 1, 1))]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() > 0
    assert head._last_gaussian_semantic_density_debug["num_pos"] == 1
    assert (
        head._last_gaussian_semantic_density_debug["weighted_loss"]
        == pytest.approx(loss.item())
    )


def test_closed_set_density_delta_loss_adds_train_time_hardneg_signal():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.0
    head.gaussian_semantic_density_delta_loss_enable = True
    head.gaussian_semantic_density_delta_loss_weight = 0.2
    head.gaussian_semantic_density_delta_loss_min_logprob_gap = 1.0
    head.gaussian_semantic_density_delta_loss_target_negative_delta = 0.25
    head.gaussian_semantic_density_delta_loss_gt_keep_weight = 0.1
    head.gaussian_semantic_density_delta_loss_max_hardneg = 1
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    cls_scores = [torch.zeros((1, 3, 1, 1))]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() == pytest.approx(0.2 * 0.25)
    assert head._last_gaussian_semantic_density_debug["delta_num_hardneg"] == 1
    assert (
        head._last_gaussian_semantic_density_debug["delta_weighted_loss"]
        == pytest.approx(loss.item())
    )


def test_closed_set_density_delta_loss_score_gate_skips_low_rank_hardneg():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.0
    head.gaussian_semantic_density_delta_loss_enable = True
    head.gaussian_semantic_density_delta_loss_weight = 0.2
    head.gaussian_semantic_density_delta_loss_min_logprob_gap = 1.0
    head.gaussian_semantic_density_delta_loss_min_hardneg_score = 0.5
    head.gaussian_semantic_density_delta_loss_min_hardneg_logit = None
    head.gaussian_semantic_density_delta_loss_target_negative_delta = 0.25
    head.gaussian_semantic_density_delta_loss_gt_keep_weight = 0.1
    head.gaussian_semantic_density_delta_loss_max_hardneg = 1
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    cls_scores = [torch.tensor([[[[0.0]], [[-4.0]], [[-4.0]]]])]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() == pytest.approx(0.0)
    assert head._last_gaussian_semantic_density_debug["delta_num_hardneg"] == 0
    assert (
        head._last_gaussian_semantic_density_debug["delta_score_gated_pairs"]
        == 1
    )


def test_closed_set_density_delta_loss_can_use_geometry_support_for_hardneg():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_log_area_mean = torch.tensor([
        math.log(100.0),
        math.log(100.0),
        math.log(100.0),
    ])
    head.gaussian_semantic_log_area_std = torch.tensor([0.5, 0.5, 0.5])
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.0
    head.gaussian_semantic_density_delta_loss_enable = True
    head.gaussian_semantic_density_delta_loss_weight = 0.2
    head.gaussian_semantic_density_delta_loss_min_logprob_gap = 1.0
    head.gaussian_semantic_density_delta_loss_target_negative_delta = 0.25
    head.gaussian_semantic_density_delta_loss_gt_keep_weight = 0.1
    head.gaussian_semantic_density_delta_loss_max_hardneg = 1
    head.gaussian_semantic_density_use_geometry_logprob = True
    head.gaussian_semantic_geometry_enable = True
    head.gaussian_semantic_geometry_mean = torch.tensor([
        [math.log(100.0), 0.0],
        [math.log(100.0), math.log(4.0)],
        [0.0, 0.0],
    ])
    head.gaussian_semantic_geometry_std = torch.tensor([
        [0.5, 0.1],
        [0.5, 0.1],
        [1.0, 1.0],
    ])
    head.gaussian_semantic_geometry_valid_mask = torch.tensor([
        True, True, False])
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    cls_scores = [torch.zeros((1, 3, 1, 1))]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() == pytest.approx(0.2 * 0.25)
    assert head._last_gaussian_semantic_density_debug["delta_num_hardneg"] == 1
    assert (
        head._last_gaussian_semantic_density_debug[
            "geometry_logprob_used"] is True
    )


def test_closed_set_density_pair_margin_loss_uses_geometry_support():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_log_area_mean = torch.tensor([
        math.log(100.0),
        math.log(100.0),
        math.log(100.0),
    ])
    head.gaussian_semantic_log_area_std = torch.tensor([0.5, 0.5, 0.5])
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.0
    head.gaussian_semantic_density_delta_loss_enable = False
    head.gaussian_semantic_density_positive_delta_loss_enable = False
    head.gaussian_semantic_density_pair_margin_loss_enable = True
    head.gaussian_semantic_density_pair_margin_loss_weight = 0.4
    head.gaussian_semantic_density_pair_margin_loss_margin = 0.2
    head.gaussian_semantic_density_pair_margin_loss_min_logprob_gap = 1.0
    head.gaussian_semantic_density_pair_margin_loss_min_hardneg_score = 0.5
    head.gaussian_semantic_density_pair_margin_loss_max_hardneg = 1
    head.gaussian_semantic_density_use_geometry_logprob = True
    head.gaussian_semantic_geometry_enable = True
    head.gaussian_semantic_geometry_mean = torch.tensor([
        [math.log(100.0), 0.0],
        [math.log(100.0), math.log(4.0)],
        [0.0, 0.0],
    ])
    head.gaussian_semantic_geometry_std = torch.tensor([
        [0.5, 0.1],
        [0.5, 0.1],
        [1.0, 1.0],
    ])
    head.gaussian_semantic_geometry_valid_mask = torch.tensor([
        True, True, False])
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    cls_scores = [torch.tensor([[[[1.0]], [[1.3]], [[-4.0]]]])]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() == pytest.approx(0.4 * 0.5)
    assert (
        head._last_gaussian_semantic_density_debug[
            "pair_margin_num_hardneg"] == 1
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "pair_margin_active_violation_count"] == 1
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "pair_margin_mean_pair_margin"] == pytest.approx(-0.3)
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "geometry_logprob_used"] is True
    )


def test_closed_set_density_support_negative_loss_uses_predicted_geometry_only_for_negatives():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_log_area_mean = torch.tensor([
        math.log(100.0),
        math.log(100.0),
        math.log(100.0),
    ])
    head.gaussian_semantic_log_area_std = torch.tensor([0.5, 0.5, 0.5])
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.0
    head.gaussian_semantic_density_delta_loss_enable = False
    head.gaussian_semantic_density_positive_delta_loss_enable = False
    head.gaussian_semantic_density_pair_margin_loss_enable = False
    head.gaussian_semantic_density_support_negative_loss_enable = True
    head.gaussian_semantic_density_support_negative_loss_weight = 0.5
    head.gaussian_semantic_density_support_negative_loss_min_score = 0.05
    head.gaussian_semantic_density_support_negative_loss_min_logprob_gap = 1.0
    head.gaussian_semantic_density_support_negative_loss_gamma = 0.0
    head.gaussian_semantic_density_support_negative_loss_gap_scale = 8.0
    head.gaussian_semantic_density_support_negative_loss_max_extra_weight = 0.0
    head.gaussian_semantic_density_support_negative_loss_max_hardneg = 1
    head.gaussian_semantic_density_use_geometry_logprob = True
    head.gaussian_semantic_geometry_enable = True
    head.gaussian_semantic_geometry_mean = torch.tensor([
        [math.log(100.0), 0.0],
        [math.log(100.0), math.log(4.0)],
        [0.0, 0.0],
    ])
    head.gaussian_semantic_geometry_std = torch.tensor([
        [0.5, 0.1],
        [0.5, 0.1],
        [1.0, 1.0],
    ])
    head.gaussian_semantic_geometry_valid_mask = torch.tensor([
        True, True, False])
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    neg_bad_logit = torch.logit(torch.tensor(0.80))
    cls_scores = [torch.tensor([[
        [[torch.logit(torch.tensor(0.10)), torch.logit(torch.tensor(0.10))]],
        [[neg_bad_logit, torch.logit(torch.tensor(0.90))]],
        [[-4.0, -4.0]],
    ]])]
    labels_list = [torch.tensor([3, 0])]
    bbox_targets_list = [torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 10.0, 0.0],
    ])]
    pred_bboxes_list = [torch.tensor([[
        [0.0, 0.0, 10.0, 10.0, 0.0],
        [0.0, 0.0, 10.0, 10.0, 0.0],
    ]])]
    assign_metrics_list = [torch.tensor([0.0, 1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head,
        cls_scores,
        labels_list,
        bbox_targets_list,
        assign_metrics_list,
        pred_bboxes_list=pred_bboxes_list)

    assert loss.item() == pytest.approx(
        0.5 * torch.nn.functional.softplus(neg_bad_logit).item())
    assert (
        head._last_gaussian_semantic_density_debug[
            "support_negative_selected_pairs"] == 1
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "support_negative_positive_locations"] == 1
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "geometry_logprob_used"] is True
    )


def test_closed_set_density_scale_consistency_penalizes_positive_predicted_scale_drop():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_log_area_mean = torch.tensor([
        math.log(100.0),
        math.log(10000.0),
        math.log(10000.0),
    ])
    head.gaussian_semantic_log_area_std = torch.tensor([0.5, 0.5, 0.5])
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.0
    head.gaussian_semantic_density_delta_loss_enable = False
    head.gaussian_semantic_density_positive_delta_loss_enable = False
    head.gaussian_semantic_density_pair_margin_loss_enable = False
    head.gaussian_semantic_density_support_negative_loss_enable = False
    head.gaussian_semantic_density_scale_consistency_loss_enable = True
    head.gaussian_semantic_density_scale_consistency_loss_weight = 0.25
    head.gaussian_semantic_density_scale_consistency_loss_max_logprob_drop = 0.5
    head.gaussian_semantic_density_scale_consistency_loss_min_target_logprob = None
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    cls_scores = [torch.tensor([[
        [[0.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ]])]
    labels_list = [torch.tensor([0, 3])]
    bbox_targets_list = [torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])]
    pred_bboxes_list = [torch.tensor([[
        [0.0, 0.0, 20.0, 20.0, 0.0],
        [0.0, 0.0, 100.0, 100.0, 0.0],
    ]])]
    assign_metrics_list = [torch.tensor([1.0, 0.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head,
        cls_scores,
        labels_list,
        bbox_targets_list,
        assign_metrics_list,
        pred_bboxes_list=pred_bboxes_list)

    z = (math.log(400.0) - math.log(100.0)) / 0.5
    expected_drop = 0.5 * z * z
    expected_raw = expected_drop - 0.5
    assert loss.item() == pytest.approx(0.25 * expected_raw)
    assert (
        head._last_gaussian_semantic_density_debug[
            "scale_consistency_supported_pos"] == 1
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "scale_consistency_active_violation_count"] == 1
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "scale_consistency_mean_logprob_drop"] == pytest.approx(
                expected_drop)
    )


def test_closed_set_density_positive_delta_loss_adds_gt_support_signal():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_loss_weight = 0.0
    head.gaussian_semantic_density_positive_delta_loss_enable = True
    head.gaussian_semantic_density_positive_delta_loss_weight = 0.3
    head.gaussian_semantic_density_positive_delta_loss_target = 0.2
    head.gaussian_semantic_density_positive_delta_loss_min_gt_logprob = None
    head.gaussian_semantic_density_positive_delta_loss_min_gt_score = 0.1
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
        nonpositive_delta=False,
    )
    cls_scores = [torch.tensor([[[[2.0]], [[-4.0]], [[-4.0]]]])]
    labels_list = [torch.tensor([0])]
    bbox_targets_list = [torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])]
    assign_metrics_list = [torch.tensor([1.0])]

    loss = GSRRotatedRTMDetSepBNHead._loss_gaussian_scale_density(
        head, cls_scores, labels_list, bbox_targets_list, assign_metrics_list)

    assert loss.item() == pytest.approx(0.3 * 0.2)
    assert (
        head._last_gaussian_semantic_density_debug[
            "positive_delta_supported_pos"] == 1
    )
    assert (
        head._last_gaussian_semantic_density_debug[
            "positive_delta_weighted_loss"] == pytest.approx(loss.item())
    )


def test_closed_set_density_head_delta_can_modulate_pre_topk_scores():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_scale_beta = 0.0
    head.gaussian_semantic_scale_z0_per_class = torch.tensor([4.0, 4.0, 4.0])
    head.gaussian_semantic_scale_beta_per_class = torch.tensor([0.0, 0.0, 0.0])
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_apply_logit_delta = True
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=0.5,
    )
    with torch.no_grad():
        # raw_delta is every third channel in the per-class triplet.
        head.gaussian_semantic_density_head.net[-1].bias[2::3].fill_(1.0)

    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))
    density_features = torch.zeros((1, 3, 1, 1))

    calibrated = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy"},
        level_idx=0,
        density_features=density_features,
    )

    assert calibrated[0, 0] < scores[0, 0]
    assert calibrated[0, 1] < scores[0, 1]
    assert calibrated[0, 2].item() == pytest.approx(0.70)
    assert head._last_gaussian_semantic_scale_debug[
        "last_num_density_delta_pairs"] == 2


def test_closed_set_density_logprob_energy_suppresses_low_support_classes():
    head = build_dummy_closed_set_head()
    head.gaussian_semantic_scale_beta = 0.0
    head.gaussian_semantic_scale_z0_per_class = torch.tensor([4.0, 4.0, 4.0])
    head.gaussian_semantic_scale_beta_per_class = torch.tensor([0.0, 0.0, 0.0])
    head.gaussian_semantic_density_enable = True
    head.gaussian_semantic_density_apply_logit_delta = True
    head.gaussian_semantic_density_logit_delta_source = "log_prob_energy"
    head.gaussian_semantic_density_logprob_delta_beta = 1.0
    head.gaussian_semantic_density_logprob_delta_threshold = -8.0
    head.gaussian_semantic_density_max_delta_abs = 1.0
    head.gaussian_semantic_density_head = GaussianSemanticScaleDensityHead(
        in_channels=3,
        num_classes=3,
        hidden_channels=4,
        max_delta_abs=1.0,
    )

    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))
    density_features = torch.zeros((1, 3, 1, 1))

    calibrated = GSRRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy"},
        level_idx=0,
        density_features=density_features,
    )

    assert calibrated[0, 0].item() == pytest.approx(0.90, abs=5e-4)
    assert calibrated[0, 1] < 0.70
    assert calibrated[0, 2].item() == pytest.approx(0.70)
    assert (
        head._last_gaussian_semantic_scale_debug[
            "density_logit_delta_source"]
        == "log_prob_energy"
    )
