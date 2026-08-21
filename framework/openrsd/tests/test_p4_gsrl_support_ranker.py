import csv
import math
import pickle

import pytest
import torch

from M_Tools.analysis.apply_p4_gsrl_support_ranker import apply_p4_gsrl_ranker


CLASS_NAMES = ["small-vehicle", "plane", "harbor"]


def _write_priors(path):
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
            "log_area_mean": str(math.log(100.0)),
            "log_area_std": "0.5",
        })
        writer.writerow({
            "class": "plane",
            "p01_area": "5000",
            "median_area": "10000",
            "p99_area": "20000",
            "log_area_mean": str(math.log(10000.0)),
            "log_area_std": "0.5",
        })


def _write_predictions(path):
    predictions = [{
        "img_id": "tile_1",
        "pred_instances": {
            "bboxes": torch.tensor([
                [0.0, 0.0, 10.0, 10.0, 0.0],
                [0.0, 0.0, 10.0, 10.0, 0.0],
                [0.0, 0.0, 20.0, 20.0, 0.0],
                [0.0, 0.0, 10.0, 10.0, 0.0],
            ]),
            "scores": torch.tensor([0.50, 0.70, 0.01, 0.60]),
            "labels": torch.tensor([0, 1, 0, 2]),
        },
    }]
    with path.open("wb") as f:
        pickle.dump(predictions, f)


def test_gsrl_boosts_supported_and_penalizes_low_support_scores(tmp_path):
    in_pkl = tmp_path / "predictions.pkl"
    out_pkl = tmp_path / "p4.pkl"
    priors = tmp_path / "priors.csv"
    _write_priors(priors)
    _write_predictions(in_pkl)

    summary = apply_p4_gsrl_ranker(
        input_pkl=in_pkl,
        output_pkl=out_pkl,
        class_area_priors_csv=priors,
        class_names=CLASS_NAMES,
        alpha=0.4,
        beta=0.2,
        reward_z=1.0,
        penalty_z=2.0,
        min_score=0.05,
        max_up_delta=0.5,
        max_down_delta=1.0,
    )

    with out_pkl.open("rb") as f:
        output = pickle.load(f)
    scores = output[0]["pred_instances"]["scores"]

    assert scores[0].item() > 0.50
    assert scores[1].item() < 0.70
    assert scores[2].item() == pytest.approx(0.01)
    assert scores[3].item() == pytest.approx(0.60)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)
    assert summary["scores_boosted"] == 1
    assert summary["scores_penalized"] == 1
    assert summary["scores_changed"] == 2
    assert summary["uses_gt_at_inference"] is False
    assert summary["forbidden_bbox_gaussian_route"] is False
    assert summary["bbox_distance_gaussian_used"] is False


def test_gsrl_keeps_prediction_count_and_boxes_unchanged(tmp_path):
    in_pkl = tmp_path / "predictions.pkl"
    out_pkl = tmp_path / "p4.pkl"
    priors = tmp_path / "priors.csv"
    _write_priors(priors)
    _write_predictions(in_pkl)

    apply_p4_gsrl_ranker(
        input_pkl=in_pkl,
        output_pkl=out_pkl,
        class_area_priors_csv=priors,
        class_names=CLASS_NAMES,
        alpha=0.4,
        beta=0.2,
        reward_z=1.0,
        penalty_z=2.0,
    )

    with in_pkl.open("rb") as f:
        original = pickle.load(f)
    with out_pkl.open("rb") as f:
        updated = pickle.load(f)

    assert len(updated[0]["pred_instances"]["scores"]) == 4
    assert torch.equal(
        updated[0]["pred_instances"]["bboxes"],
        original[0]["pred_instances"]["bboxes"],
    )
