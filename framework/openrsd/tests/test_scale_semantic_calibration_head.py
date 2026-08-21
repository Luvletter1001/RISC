import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import (
    OpenRotatedRTMDetSepBNHead,
)
from M_AD.models.utils.gaussian_semantic_scale import (
    GaussianScaleLogitAdapter,
    build_classwise_gaussian_parameter,
    continuous_gaussian_logit_energy_delta,
)


class IdentityAngleCoder:
    def decode(self, angle_pred, keepdim=True):
        return angle_pred


class IdentityBboxCoder:
    def decode(self, priors, bbox_pred, max_shape=None):
        return bbox_pred


def build_dummy_s3c_head(tmp_path: Path):
    class_names = ("small-vehicle", "plane", "unknown-no-prior")
    return SimpleNamespace(
        scale_semantic_calibration_enable=True,
        scale_semantic_calibration_z_margin=4.0,
        scale_semantic_calibration_lambda=0.25,
        scale_semantic_calibration_mode="score_multiply",
        scale_semantic_calibration_min_score=0.0,
        scale_semantic_valid_mask=torch.tensor([True, True, False]),
        scale_semantic_log_area_mean=torch.tensor([
            math.log(100.0),
            math.log(10000.0),
            math.log(10000.0),
        ]),
        scale_semantic_log_area_std=torch.tensor([0.5, 0.5, 0.5]),
        scale_semantic_calibration_class_names=class_names,
        scale_semantic_calibration_debug={
            "enable": True,
            "num_classes": len(class_names),
        },
        scale_semantic_dump_topk_jsonl=str(tmp_path / "dense_topk.jsonl"),
        scale_semantic_dump_topk_k=4,
        angle_coder=IdentityAngleCoder(),
        bbox_coder=IdentityBboxCoder(),
        _dump_scale_semantic_topk=OpenRotatedRTMDetSepBNHead
        ._dump_scale_semantic_topk,
    )


def call_s3c(head, scores, bbox_pred, angle_pred, priors, tmp_path):
    return OpenRotatedRTMDetSepBNHead._apply_scale_semantic_calibration(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy_img", "img_path": "toy.png"},
        level_idx=0,
    )


def build_dummy_gs3c_head(mode: str = "continuous_logit_energy"):
    class_names = ("small-vehicle", "plane", "unknown-no-prior")
    return SimpleNamespace(
        gaussian_semantic_scale_enable=True,
        gaussian_semantic_scale_mode=mode,
        gaussian_semantic_scale_z0=4.0,
        gaussian_semantic_scale_beta=math.log(4.0),
        gaussian_semantic_scale_adapter_nonpositive_delta=True,
        gaussian_semantic_scale_domain_mode="closed_set",
        gaussian_semantic_scale_adapter=None,
        gaussian_semantic_valid_mask=torch.tensor([True, True, False]),
        gaussian_semantic_log_area_mean=torch.tensor([
            math.log(100.0),
            math.log(10000.0),
            math.log(10000.0),
        ]),
        gaussian_semantic_log_area_std=torch.tensor([0.5, 0.5, 0.5]),
        gaussian_semantic_scale_class_names=class_names,
        gaussian_semantic_scale_debug={
            "enable": True,
            "mode": mode,
            "domain_mode": "closed_set",
            "num_classes": len(class_names),
            "num_valid_priors": 2,
        },
        angle_coder=IdentityAngleCoder(),
        bbox_coder=IdentityBboxCoder(),
    )


def call_gs3c(head, scores, bbox_pred, angle_pred, priors):
    return OpenRotatedRTMDetSepBNHead._apply_gaussian_semantic_scale(
        head,
        scores=scores,
        bbox_pred=bbox_pred,
        angle_pred=angle_pred,
        priors=priors,
        img_shape=(800, 800),
        img_meta={"img_id": "toy_img", "img_path": "toy.png"},
        level_idx=0,
    )


def test_gs3c_continuous_energy_is_monotonic_and_nonpositive():
    z = torch.tensor([[0.0, 2.0, 4.0, 6.0]])
    valid = torch.tensor([[True, True, True, True]])

    delta = continuous_gaussian_logit_energy_delta(
        z, valid, z0=4.0, beta=math.log(4.0))

    assert torch.all(delta <= 0)
    penalty = -delta
    assert penalty[0, 0] < penalty[0, 1] < penalty[0, 2] < penalty[0, 3]

    invalid_delta = continuous_gaussian_logit_energy_delta(
        z, torch.zeros_like(valid), z0=4.0, beta=math.log(4.0))
    assert torch.equal(invalid_delta, torch.zeros_like(invalid_delta))


def test_gs3c_continuous_energy_accepts_classwise_z0_beta_tensors():
    z = torch.tensor([[5.0, 5.0, 7.0]])
    valid = torch.tensor([[True, True, False]])
    z0 = torch.tensor([4.0, 6.0, 4.0])
    beta = torch.tensor([1.0, 2.0, 1.0])

    delta = continuous_gaussian_logit_energy_delta(
        z, valid, z0=z0, beta=beta)

    assert delta.shape == z.shape
    assert delta[0, 0] < delta[0, 1]
    assert delta[0, 2].item() == pytest.approx(0.0)
    assert torch.all(delta <= 0)


def test_build_classwise_gaussian_parameter_maps_names_and_falls_back():
    values = build_classwise_gaussian_parameter(
        scalar_value=4.0,
        classwise_value={"plane": 5.0, "ship": 6.0},
        class_names=("small-vehicle", "plane", "ship"),
        num_classes=3,
        name="z0",
        min_value=0.0,
    )

    assert values.tolist() == pytest.approx([4.0, 5.0, 6.0])


def test_gs3c_adapter_zero_init_never_increases_logits():
    adapter = GaussianScaleLogitAdapter(hidden=8, nonpositive_delta=True)
    z = torch.tensor([[0.0, -2.0, 6.0]])
    log_prob = torch.tensor([[-1.0, -3.0, -9.0]])
    log_std = torch.zeros_like(z)
    valid = torch.tensor([[True, True, False]])

    delta = adapter(
        abs_z=z.abs(),
        signed_z=z,
        gaussian_log_prob=log_prob,
        log_std=log_std,
        valid_mask=valid)

    assert delta.shape == z.shape
    assert torch.all(delta <= 0)
    assert torch.equal(delta[:, -1], torch.zeros_like(delta[:, -1]))
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_pre_topk_gs3c_downweights_continuously_by_gaussian_log_area():
    head = build_dummy_gs3c_head()
    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = call_gs3c(
        head, scores, bbox_pred, angle_pred, priors)

    assert calibrated[0, 0] < 0.30
    assert calibrated[0, 1] < scores[0, 1]
    assert calibrated[0, 1] > 0.75
    assert calibrated[0, 2].item() == pytest.approx(0.70)
    assert (
        head._last_gaussian_semantic_scale_debug["last_num_locations"] == 1)
    assert (
        head._last_gaussian_semantic_scale_debug["gaussian_energy_mode"]
        == "continuous_logit_energy")


def test_ovd_gs3c_classwise_z0_relaxes_selected_class():
    head = build_dummy_gs3c_head()
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

    calibrated = call_gs3c(head, scores, bbox_pred, angle_pred, priors)

    assert calibrated[0, 0] < 0.05
    assert calibrated[0, 1] > 0.70
    assert calibrated[0, 2].item() == pytest.approx(0.70)


def test_pre_topk_gs3c_enable_false_is_bitwise_noop():
    head = build_dummy_gs3c_head()
    head.gaussian_semantic_scale_enable = False
    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = call_gs3c(
        head, scores, bbox_pred, angle_pred, priors)

    assert torch.equal(calibrated, scores)


def test_pre_nms_s3c_downweights_only_scale_implausible_class(tmp_path):
    head = build_dummy_s3c_head(tmp_path)
    head._dump_scale_semantic_topk = (
        OpenRotatedRTMDetSepBNHead._dump_scale_semantic_topk.__get__(head))

    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    calibrated = call_s3c(
        head, scores, bbox_pred, angle_pred, priors, tmp_path)

    assert calibrated[0, 0].item() == pytest.approx(0.225)
    assert calibrated[0, 1].item() == pytest.approx(0.80)
    assert calibrated[0, 2].item() == pytest.approx(0.70)

    assert head._last_scale_semantic_calibration_debug["last_num_flags"] == 1
    assert head._last_scale_semantic_calibration_debug["last_num_locations"] == 1


def test_pre_nms_s3c_writes_dense_topk_before_after_dump(tmp_path):
    head = build_dummy_s3c_head(tmp_path)
    head._dump_scale_semantic_topk = (
        OpenRotatedRTMDetSepBNHead._dump_scale_semantic_topk.__get__(head))

    scores = torch.tensor([[0.90, 0.80, 0.70]])
    bbox_pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    angle_pred = torch.zeros((1, 1))
    priors = torch.zeros((1, 2))

    call_s3c(head, scores, bbox_pred, angle_pred, priors, tmp_path)

    rows = [
        json.loads(line)
        for line in Path(head.scale_semantic_dump_topk_jsonl)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["stage"] for row in rows} == {"before", "after"}

    before_top = next(
        row for row in rows if row["stage"] == "before" and row["rank"] == 1)
    after_top = next(
        row for row in rows if row["stage"] == "after" and row["rank"] == 1)

    assert before_top["class_name"] == "small-vehicle"
    assert before_top["s3c_flagged"] is True
    assert before_top["log_area_z"] >= 4.0
    assert after_top["class_name"] == "plane"
    assert after_top["s3c_flagged"] is False
