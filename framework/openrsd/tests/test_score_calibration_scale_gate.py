from M_Tools.analysis.analyze_score_calibration_scale_gate import (
    enrich_manifest_rows,
    evaluate_gate,
    score_histogram,
    summarize_wrong_pairs,
)


PRIORS = {
    "small-vehicle": {
        "median_area": 100.0,
        "p01_area": 50.0,
        "p99_area": 200.0,
    },
    "plane": {
        "median_area": 1000.0,
        "p01_area": 500.0,
        "p99_area": 2000.0,
    },
    "ship": {
        "median_area": 400.0,
        "p01_area": 100.0,
        "p99_area": 1000.0,
    },
}


def test_summarize_wrong_pairs_exposes_near_one_score_saturation():
    rows = enrich_manifest_rows(
        [
            {
                "kind": "wrong",
                "pred_class": "small-vehicle",
                "gt_class": "plane",
                "score": "0.99995",
                "iou": "0.91",
                "pred_qbox_original": "0 0 40 0 40 40 0 40",
            },
            {
                "kind": "wrong",
                "pred_class": "small-vehicle",
                "gt_class": "plane",
                "score": "0.93",
                "iou": "0.82",
                "pred_qbox_original": "0 0 30 0 30 30 0 30",
            },
            {
                "kind": "wrong",
                "pred_class": "small-vehicle",
                "gt_class": "ship",
                "score": "0.40",
                "iou": "0.80",
                "pred_qbox_original": "0 0 10 0 10 10 0 10",
            },
        ],
        PRIORS,
        scale_ratio_thr=4.0,
    )

    pair_rows = summarize_wrong_pairs(rows, near_one_thr=0.999)
    sv_plane = next(row for row in pair_rows if row["pair"] == "small-vehicle->plane")

    assert sv_plane["count"] == 2
    assert sv_plane["score_ge_0p9"] == 2
    assert sv_plane["score_ge_0p99"] == 1
    assert sv_plane["score_ge_near_one"] == 1
    assert sv_plane["sise_count"] == 2
    assert sv_plane["prior_outlier_count"] == 2
    assert sv_plane["median_score"] == 0.964975


def test_evaluate_gate_quantifies_wrong_reduction_and_correct_false_rejects():
    rows = enrich_manifest_rows(
        [
            {
                "kind": "wrong",
                "pred_class": "small-vehicle",
                "gt_class": "plane",
                "score": "0.99995",
                "iou": "0.91",
                "pred_qbox_original": "0 0 40 0 40 40 0 40",
            },
            {
                "kind": "wrong",
                "pred_class": "small-vehicle",
                "gt_class": "ship",
                "score": "0.40",
                "iou": "0.80",
                "pred_qbox_original": "0 0 10 0 10 10 0 10",
            },
            {
                "kind": "correct",
                "pred_class": "small-vehicle",
                "gt_class": "small-vehicle",
                "score": "0.95",
                "iou": "0.88",
                "pred_qbox_original": "0 0 10 0 10 10 0 10",
            },
            {
                "kind": "correct",
                "pred_class": "plane",
                "gt_class": "plane",
                "score": "0.99",
                "iou": "0.92",
                "pred_qbox_original": "0 0 100 0 100 100 0 100",
            },
        ],
        PRIORS,
        scale_ratio_thr=4.0,
    )

    result = evaluate_gate(rows, gate="scale_ratio", score_thr=0.5)

    assert result["high_conf_wrong_before"] == 1
    assert result["wrong_rejected"] == 1
    assert result["wrong_reduction_rate"] == 1.0
    assert result["high_conf_correct_before"] == 2
    assert result["correct_rejected"] == 1
    assert result["correct_false_reject_rate"] == 0.5
    assert result["gate_precision"] == 0.5


def test_score_histogram_counts_wrong_and_correct_rows_by_bins():
    rows = enrich_manifest_rows(
        [
            {
                "kind": "wrong",
                "pred_class": "small-vehicle",
                "gt_class": "plane",
                "score": "0.99995",
                "iou": "0.91",
                "pred_qbox_original": "0 0 40 0 40 40 0 40",
            },
            {
                "kind": "correct",
                "pred_class": "small-vehicle",
                "gt_class": "small-vehicle",
                "score": "0.49",
                "iou": "0.88",
                "pred_qbox_original": "0 0 10 0 10 10 0 10",
            },
        ],
        PRIORS,
        scale_ratio_thr=4.0,
    )

    hist = score_histogram(rows, bins=[0.0, 0.5, 0.9, 0.99, 1.000001])

    assert {
        (row["kind"], row["score_bin"]): row["count"]
        for row in hist
    } == {
        ("wrong", "[0.99,1.000001)"): 1,
        ("correct", "[0.0,0.5)"): 1,
    }
