import csv
import pickle

import pytest
import torch

from M_Tools.analysis.apply_scale_prior_to_predictions import (
    apply_scale_prior_to_predictions,
    compute_box_areas,
    load_class_area_priors,
)


CLASS_NAMES = ["small-vehicle", "plane"]


def write_priors(path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "p01_area",
                "median_area",
                "p99_area",
                "log_area_mean",
                "log_area_std",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "class": "small-vehicle",
            "p01_area": "50",
            "median_area": "100",
            "p99_area": "200",
            "log_area_mean": "4.605170185988092",
            "log_area_std": "0.5",
        })
        writer.writerow({
            "class": "plane",
            "p01_area": "500",
            "median_area": "1000",
            "p99_area": "2000",
            "log_area_mean": "6.907755278982137",
            "log_area_std": "0.5",
        })


def make_predictions():
    return [{
        "img_id": "tile_1",
        "pred_instances": {
            "bboxes": torch.tensor([
                [0.0, 0.0, 10.0, 10.0, 0.0],
                [0.0, 0.0, 40.0, 40.0, 0.0],
                [0.0, 0.0, 100.0, 100.0, 0.0],
            ]),
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "labels": torch.tensor([0, 0, 1]),
        },
    }]


def test_compute_box_areas_supports_rbox_and_qbox():
    rboxes = torch.tensor([[0.0, 0.0, 4.0, 5.0, 0.0]])
    qboxes = torch.tensor([[0.0, 0.0, 4.0, 0.0, 4.0, 5.0, 0.0, 5.0]])

    assert compute_box_areas(rboxes).tolist() == [20.0]
    assert compute_box_areas(qboxes).tolist() == [20.0]


def test_load_class_area_priors_reads_required_area_columns(tmp_path):
    priors_path = tmp_path / "priors.csv"
    write_priors(priors_path)

    priors = load_class_area_priors(priors_path)

    assert priors["small-vehicle"]["median_area"] == 100.0
    assert priors["plane"]["p99_area"] == 2000.0


def test_prior_outlier_reject_removes_only_flagged_predictions(tmp_path):
    priors_path = tmp_path / "priors.csv"
    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    write_priors(priors_path)
    with in_pkl.open("wb") as f:
        pickle.dump(make_predictions(), f)

    summary = apply_scale_prior_to_predictions(
        input_pkl=in_pkl,
        output_pkl=out_pkl,
        class_area_priors_csv=priors_path,
        class_names=CLASS_NAMES,
        strategy="prior_outlier_reject",
    )

    with out_pkl.open("rb") as f:
        filtered = pickle.load(f)
    pred = filtered[0]["pred_instances"]
    assert pred["scores"].tolist() == [0.8999999761581421]
    assert pred["labels"].tolist() == [0]
    assert summary["detections_before"] == 3
    assert summary["detections_after"] == 1
    assert summary["detections_rejected"] == 2


def test_scale_ratio_reject_uses_predicted_class_median(tmp_path):
    priors_path = tmp_path / "priors.csv"
    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    write_priors(priors_path)
    with in_pkl.open("wb") as f:
        pickle.dump(make_predictions(), f)

    summary = apply_scale_prior_to_predictions(
        input_pkl=in_pkl,
        output_pkl=out_pkl,
        class_area_priors_csv=priors_path,
        class_names=CLASS_NAMES,
        strategy="scale_ratio_reject",
        scale_ratio_thr=4.0,
    )

    with out_pkl.open("rb") as f:
        filtered = pickle.load(f)
    pred = filtered[0]["pred_instances"]
    assert pred["scores"].tolist() == [0.8999999761581421]
    assert pred["labels"].tolist() == [0]
    assert summary["detections_before"] == 3
    assert summary["detections_after"] == 1
    assert summary["detections_rejected"] == 2


def test_prior_outlier_downweight_keeps_boxes_and_reduces_scores(tmp_path):
    priors_path = tmp_path / "priors.csv"
    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    write_priors(priors_path)
    with in_pkl.open("wb") as f:
        pickle.dump(make_predictions(), f)

    summary = apply_scale_prior_to_predictions(
        input_pkl=in_pkl,
        output_pkl=out_pkl,
        class_area_priors_csv=priors_path,
        class_names=CLASS_NAMES,
        strategy="prior_outlier_downweight",
        downweight_lambda=1.0,
    )

    with out_pkl.open("rb") as f:
        filtered = pickle.load(f)
    pred = filtered[0]["pred_instances"]
    assert len(pred["scores"]) == 3
    assert pred["scores"][0].item() == pytest.approx(0.9)
    assert pred["scores"][1].item() < 0.8
    assert pred["scores"][2].item() < 0.7
    assert summary["detections_before"] == 3
    assert summary["detections_after"] == 3
    assert summary["detections_downweighted"] == 2


def test_log_area_calibrate_softly_penalizes_distribution_outliers(tmp_path):
    priors_path = tmp_path / "priors.csv"
    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    write_priors(priors_path)
    with in_pkl.open("wb") as f:
        pickle.dump(make_predictions(), f)

    summary = apply_scale_prior_to_predictions(
        input_pkl=in_pkl,
        output_pkl=out_pkl,
        class_area_priors_csv=priors_path,
        class_names=CLASS_NAMES,
        strategy="log_area_calibrate",
        downweight_lambda=0.5,
        log_area_z_margin=1.0,
    )

    with out_pkl.open("rb") as f:
        calibrated = pickle.load(f)
    pred = calibrated[0]["pred_instances"]
    assert len(pred["scores"]) == 3
    assert pred["scores"][0].item() == pytest.approx(0.9)
    assert pred["scores"][1].item() < 0.8
    assert pred["scores"][2].item() < 0.7
    assert pred["scores"][1].item() > 0.0
    assert summary["detections_after"] == 3
    assert summary["detections_calibrated"] == 2
    assert summary["calibrated_by_class"] == {
        "plane": 1,
        "small-vehicle": 1,
    }


def test_log_area_calibrate_can_target_high_score_outliers_only(tmp_path):
    priors_path = tmp_path / "priors.csv"
    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    write_priors(priors_path)
    with in_pkl.open("wb") as f:
        pickle.dump(make_predictions(), f)

    summary = apply_scale_prior_to_predictions(
        input_pkl=in_pkl,
        output_pkl=out_pkl,
        class_area_priors_csv=priors_path,
        class_names=CLASS_NAMES,
        strategy="log_area_calibrate",
        downweight_lambda=0.5,
        log_area_z_margin=1.0,
        calibration_min_score=0.75,
    )

    with out_pkl.open("rb") as f:
        calibrated = pickle.load(f)
    pred = calibrated[0]["pred_instances"]
    assert pred["scores"][0].item() == pytest.approx(0.9)
    assert pred["scores"][1].item() < 0.8
    assert pred["scores"][2].item() == pytest.approx(0.7)
    assert summary["detections_calibrated"] == 1
    assert summary["calibration_min_score"] == 0.75
    assert summary["calibrated_by_class"] == {"small-vehicle": 1}
