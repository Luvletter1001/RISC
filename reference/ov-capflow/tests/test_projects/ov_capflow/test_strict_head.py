import torch

from projects.OVCapFlow.ov_capflow.ov_capflow_head import (
    OVCapFlowHead, select_one_class_per_query)


def test_select_one_class_per_query_keeps_every_query_row():
    class_scores = torch.tensor([
        [0.1, 0.2, 0.9, 0.3],
        [0.8, 0.1, 0.2, 0.4],
        [0.7, 0.6, 0.1, 0.2],
    ])
    boxes = torch.arange(15, dtype=torch.float32).reshape(3, 5)

    scores, labels, selected_boxes = select_one_class_per_query(
        class_scores, boxes)

    torch.testing.assert_close(scores, torch.tensor([0.9, 0.8, 0.7]))
    torch.testing.assert_close(labels, torch.tensor([2, 0, 0]))
    torch.testing.assert_close(selected_boxes, boxes)


def test_select_one_class_per_query_does_not_deduplicate_same_class():
    class_scores = torch.tensor([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
    ])
    boxes = torch.randn(3, 5)

    scores, labels, selected_boxes = select_one_class_per_query(
        class_scores, boxes)

    assert scores.shape == (3, )
    assert labels.tolist() == [0, 0, 0]
    assert selected_boxes.shape == (3, 5)


def test_select_one_class_per_query_rejects_mismatched_rows():
    class_scores = torch.randn(3, 2)
    boxes = torch.randn(2, 5)

    try:
        select_one_class_per_query(class_scores, boxes)
    except ValueError as error:
        assert 'same number of queries' in str(error)
    else:
        raise AssertionError('mismatched query rows must raise ValueError')


def test_null_score_calibration_keeps_all_query_rows():
    head = object.__new__(OVCapFlowHead)
    head.readout_cfg = dict(temperature=1.0, power=1.0)
    head.angle_factor = 3.141592653589793
    token_logits = torch.tensor([
        [2.0, -2.0],
        [1.0, -1.0],
        [0.5, -0.5],
    ])
    boxes = torch.randn(3, 5)
    null_logits = torch.tensor([-2.0, 0.0, 2.0])
    results = head._predict_by_feat_single(
        token_logits,
        boxes,
        token_positive_maps={1: [0], 2: [1]},
        img_meta=dict(img_shape=(100, 100), scale_factor=(1.0, 1.0)),
        rescale=True,
        null_logits=null_logits)
    assert results.scores.shape == results.labels.shape == (3, )
    assert results.bboxes.shape == (3, 5)
    assert results.scores[0] > results.scores[2]


def test_predict_preserves_nonmonotonic_query_order():
    head = object.__new__(OVCapFlowHead)
    head.readout_cfg = dict(temperature=1.0, power=1.0)
    head.angle_factor = torch.pi
    token_logits = torch.tensor([
        [0.0, -4.0],
        [4.0, -4.0],
        [-4.0, 2.0],
    ])
    boxes = torch.tensor([
        [0.1, 0.1, 0.1, 0.1, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.0],
        [0.3, 0.3, 0.1, 0.1, 0.0],
    ])

    results = head._predict_by_feat_single(
        token_logits,
        boxes,
        token_positive_maps={1: [0], 2: [1]},
        img_meta=dict(img_shape=(100, 100)),
        rescale=False)

    torch.testing.assert_close(
        results.scores, torch.sigmoid(torch.tensor([0.0, 4.0, 2.0])))
    assert results.labels.tolist() == [0, 0, 1]
    torch.testing.assert_close(
        results.bboxes[:, 0], torch.tensor([10.0, 20.0, 30.0]))
