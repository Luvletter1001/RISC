import torch

from projects.OVCapFlow.tools.hrsc_mediator_metrics import (
    finalize_mediator_totals,
    summarize_image_mediators,
)


def test_mediator_summary_counts_coverage_duplicates_and_calibration():
    ious = torch.tensor([
        [0.8, 0.1],
        [0.7, 0.2],
        [0.1, 0.9],
        [0.2, 0.1],
    ])
    null_logits = torch.tensor([-10.0, -10.0, -10.0, 10.0])
    semantic_gate = torch.tensor([
        [0.2, -0.2],
        [0.3, -0.3],
        [0.4, -0.4],
        [0.6, -0.6],
    ])

    totals = summarize_image_mediators(
        ious,
        null_logits=null_logits,
        semantic_gate=semantic_gate,
        iou_threshold=0.5)
    metrics = finalize_mediator_totals(totals)

    assert metrics['prediction_count'] == 4
    assert metrics['gt_count'] == 2
    assert metrics['matched_queries'] == 3
    assert metrics['gt_coverage'] == 1.0
    assert metrics['duplicate_extras_per_gt'] == 0.5
    torch.testing.assert_close(
        torch.tensor(metrics['matched_gate_mean']), torch.tensor(0.3))
    torch.testing.assert_close(
        torch.tensor(metrics['unmatched_gate_mean']), torch.tensor(0.6))
    torch.testing.assert_close(
        torch.tensor(metrics['gate_gap']), torch.tensor(0.3))
    assert metrics['null_brier'] < 1e-6


def test_mediator_summary_handles_an_image_without_ground_truth():
    totals = summarize_image_mediators(
        torch.empty(3, 0),
        scores=torch.tensor([0.1, 0.2, 0.3]),
        iou_threshold=0.5)
    metrics = finalize_mediator_totals(totals)

    assert metrics['prediction_count'] == 3
    assert metrics['gt_count'] == 0
    assert metrics['matched_queries'] == 0
    assert metrics['gt_coverage'] is None
    assert metrics['duplicate_extras_per_gt'] is None
    assert metrics['empty_image_count'] == 1
    assert metrics['empty_predictions_per_image'] == 3.0
    torch.testing.assert_close(
        torch.tensor(metrics['empty_foreground_score_mass_mean']),
        torch.tensor(0.6))


def test_nonempty_image_does_not_increment_empty_score_totals():
    totals = summarize_image_mediators(
        torch.tensor([[0.8], [0.1]]),
        scores=torch.tensor([0.9, 0.2]),
        iou_threshold=0.5)
    metrics = finalize_mediator_totals(totals)

    assert metrics['empty_image_count'] == 0
    assert metrics['empty_predictions_per_image'] is None
    assert metrics['empty_foreground_score_mass_mean'] is None
