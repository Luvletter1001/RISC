import csv
import math
import pickle

import pytest
import torch

from M_Tools.analysis.apply_p3a_ap_constrained_support_projection import (
    apply_p3a_projection,
    assess_p3a_gate,
)


CLASS_NAMES = ["small-vehicle", "plane"]


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


def _write_ann(path):
    path.write_text(
        "0 0 10 0 10 10 0 10 small-vehicle 0\n",
        encoding="utf-8",
    )


def _write_predictions(path):
    predictions = [{
        "img_id": "tile_1",
        "pred_instances": {
            "bboxes": torch.tensor([
                [5.0, 5.0, 10.0, 10.0, 0.0],
                [5.0, 5.0, 10.0, 10.0, 0.0],
                [50.0, 50.0, 100.0, 100.0, 0.0],
            ]),
            "scores": torch.tensor([0.90, 0.85, 0.70]),
            "labels": torch.tensor([0, 1, 1]),
        },
    }]
    with path.open("wb") as f:
        pickle.dump(predictions, f)


def test_p3a_projection_protects_correct_and_demotes_low_support_wrong(tmp_path):
    pred_path = tmp_path / "predictions.pkl"
    out_path = tmp_path / "projected.pkl"
    ann_dir = tmp_path / "annfiles"
    priors = tmp_path / "priors.csv"
    ann_dir.mkdir()
    _write_predictions(pred_path)
    _write_ann(ann_dir / "tile_1.txt")
    _write_priors(priors)

    summary = apply_p3a_projection(
        input_pkl=pred_path,
        output_pkl=out_path,
        ann_dir=ann_dir,
        class_area_priors_csv=priors,
        class_names=CLASS_NAMES,
        iou_thr=0.5,
        z_thr=4.0,
        lambda_=0.5,
        min_score=0.3,
    )

    with out_path.open("rb") as f:
        projected = pickle.load(f)
    scores = projected[0]["pred_instances"]["scores"].tolist()

    assert scores[0] == pytest.approx(0.90)
    assert scores[1] < 0.85
    assert scores[2] == pytest.approx(0.70)
    assert summary["protected_correct"] == 1
    assert summary["low_support_wrong_candidates"] == 1
    assert summary["scores_changed"] == 1
    assert summary["changed_correct"] == 0
    assert summary["forbidden_bbox_gaussian_route"] is False


def test_p3a_gate_requires_ap_preservation_and_topk_risk_reduction():
    row = {
        "mAP": 0.8630,
        "top5000_precision": 0.7170,
        "top5000_sise_logz": 48,
        "top5000_risk_weighted_logz": 10.0,
        "risk_weighted_logz": 10.0,
        "changed_correct": 0,
    }
    strict_control = {
        "mAP": 0.862878,
        "top5000_precision": 0.7168,
        "top5000_sise_logz": 48,
        "top5000_risk_weighted_logz": 10.4,
        "risk_weighted_logz": 14.2824,
    }

    assert assess_p3a_gate(row, strict_control) == "strict pass"

    row["mAP"] = 0.8620
    assert assess_p3a_gate(row, strict_control) == "fail"

    row["mAP"] = 0.8630
    row["changed_correct"] = 1
    assert assess_p3a_gate(row, strict_control) == "fail"
