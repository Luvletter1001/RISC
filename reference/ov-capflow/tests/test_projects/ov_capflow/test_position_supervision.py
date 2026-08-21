import math

import pytest
import torch
import torch.nn.functional as F
from mmdet.models.losses import FocalLoss
from mmengine.structures import InstanceData

import projects.OVCapFlow.ov_capflow.ov_capflow_head as head_module
from projects.GroundingDINO.groundingdino.grounding_dino_head import (
    RotatedGroundingDINOHead, )
from projects.OVCapFlow.ov_capflow.ov_capflow_head import (
    OVCapFlowHead, scale_positive_maps_by_rotated_iou)


def _construct_head(monkeypatch, **kwargs):
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '__init__', lambda self, **kwargs: None)
    return OVCapFlowHead(**kwargs)


def _parent_targets(target_x=0.6):
    labels = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ])
    label_weights = torch.ones(2)
    bbox_targets = torch.tensor([
        [target_x, 0.5, 0.4, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    bbox_weights = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    pos_inds = torch.tensor([0], dtype=torch.long)
    neg_inds = torch.tensor([1], dtype=torch.long)
    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def test_position_supervision_is_disabled_by_default(monkeypatch):
    head = _construct_head(monkeypatch)

    assert head.position_supervised_cfg == {'enabled': False}
    assert head.position_supervised_enabled is False


def test_position_supervision_preserves_matching_group_positional_argument(
        monkeypatch):
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '__init__', lambda self, **kwargs: None)

    head = OVCapFlowHead(None, None, None, 3)

    assert head.matching_query_groups == 3
    assert head.position_supervised_cfg == {'enabled': False}


@pytest.mark.parametrize(
    'position_supervised_cfg, error_type', [
        ({'enabled': 1}, TypeError),
        ({'enabled': False, 'quality_floor': 0.2}, ValueError),
        (['enabled'], TypeError),
    ])
def test_position_supervision_rejects_invalid_config(
        monkeypatch, position_supervised_cfg, error_type):
    with pytest.raises(error_type):
        _construct_head(
            monkeypatch,
            position_supervised_cfg=position_supervised_cfg)


def test_position_supervision_rejects_balanced_classification(monkeypatch):
    with pytest.raises(ValueError, match='balanced'):
        _construct_head(
            monkeypatch,
            balanced_cfg={'enabled': True},
            position_supervised_cfg={'enabled': True})


def test_disabled_target_override_returns_parent_tuple_exactly(monkeypatch):
    parent_targets = _parent_targets()
    monkeypatch.setattr(
        RotatedGroundingDINOHead,
        '_get_targets_single',
        lambda self, *args, **kwargs: parent_targets)
    monkeypatch.setattr(
        head_module,
        'scale_positive_maps_by_rotated_iou',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('position helper must stay disabled')))
    head = _construct_head(monkeypatch)

    result = head._get_targets_single(
        torch.zeros(2, 3),
        torch.zeros(2, 5),
        InstanceData(),
        dict(img_shape=(100, 100)))

    assert result is parent_targets


def test_enabled_empty_gt_returns_parent_tuple_without_iou(monkeypatch):
    parent_targets = (
        torch.zeros(2, 3),
        torch.ones(2),
        torch.zeros(2, 5),
        torch.zeros(2, 5),
        torch.empty(0, dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    monkeypatch.setattr(
        RotatedGroundingDINOHead,
        '_get_targets_single',
        lambda self, *args, **kwargs: parent_targets)
    monkeypatch.setattr(
        head_module,
        'scale_positive_maps_by_rotated_iou',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('empty targets must not call IoU')))
    head = _construct_head(
        monkeypatch, position_supervised_cfg={'enabled': True})

    result = head._get_targets_single(
        torch.zeros(2, 3),
        torch.zeros(2, 5),
        InstanceData(),
        dict(img_shape=(100, 100)))

    assert result is parent_targets
    assert result[0].shape == (2, 3)
    assert torch.isfinite(result[0]).all()


def test_enabled_target_override_scales_only_labels(monkeypatch):
    parent_targets = _parent_targets()
    monkeypatch.setattr(
        RotatedGroundingDINOHead,
        '_get_targets_single',
        lambda self, *args, **kwargs: parent_targets)
    head = _construct_head(
        monkeypatch, position_supervised_cfg={'enabled': True})
    head.angle_factor = math.pi
    bbox_pred = torch.tensor([
        [0.5, 0.5, 0.4, 0.4, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.0],
    ])

    result = head._get_targets_single(
        torch.zeros(2, 3), bbox_pred, InstanceData(),
        dict(img_shape=(100, 100)))

    torch.testing.assert_close(
        result[0], torch.tensor([[0.3, 0.3, 0.0], [0.0, 0.0, 0.0]]),
        rtol=1e-5, atol=1e-6)
    for actual, parent in zip(result[1:], parent_targets[1:]):
        assert actual is parent


@pytest.mark.parametrize('enc_cls_scores, enc_bbox_preds', [
    (torch.zeros(1, 1, 3), None),
    (None, torch.zeros(1, 1, 5)),
    (torch.zeros(1, 1, 3), torch.zeros(1, 1, 5)),
])
def test_position_supervision_excludes_encoder_outputs(
        monkeypatch, enc_cls_scores, enc_bbox_preds):
    monkeypatch.setattr(
        RotatedGroundingDINOHead, 'loss_by_feat',
        lambda self, *args, **kwargs: {})
    head = _construct_head(
        monkeypatch, position_supervised_cfg={'enabled': True})

    with pytest.raises(ValueError, match='encoder'):
        head.loss_by_feat(
            torch.zeros(1, 1, 2, 3),
            torch.zeros(1, 1, 2, 5),
            enc_cls_scores,
            enc_bbox_preds,
            [],
            [],
            dict(num_denoising_queries=0, num_denoising_groups=1))


def test_position_supervision_does_not_override_dn_targets():
    assert '_get_dn_targets_single' not in OVCapFlowHead.__dict__


class _RotatedBoxesStub:

    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def regularize_boxes(self, **kwargs):
        return self


def test_position_supervision_does_not_change_real_dn_loss_route():
    head = object.__new__(OVCapFlowHead)
    torch.nn.Module.__init__(head)
    head.angle_cfg = {}
    head.angle_factor = math.pi
    head.max_text_len = 3
    head.text_masks = torch.ones(1, 3)
    head.bg_cls_weight = 0.1
    head.sync_cls_avg_factor = False
    head.loss_cls = FocalLoss(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0)
    head.loss_bbox = lambda pred, target, weight, avg_factor: pred.sum() * 0
    head.loss_iou = lambda pred, target, weight, avg_factor: pred.sum() * 0
    gt_instances = InstanceData(
        bboxes=_RotatedBoxesStub(torch.tensor([
            [50.0, 50.0, 40.0, 40.0, 0.0],
        ])),
        labels=torch.tensor([0]),
        positive_maps=torch.tensor([[0.5, 0.5, 0.0]]))
    dn_meta = dict(num_denoising_queries=4, num_denoising_groups=2)
    logits = torch.zeros(1, 1, 4, 3)
    boxes = torch.full((1, 1, 4, 5), 0.5)
    boxes[..., 4] = 0.0

    head.position_supervised_enabled = False
    disabled_losses = head._denoising_loss_dict(
        logits, boxes, [gt_instances], [dict(img_shape=(100, 100))],
        dn_meta)
    head.position_supervised_enabled = True
    enabled_losses = head._denoising_loss_dict(
        logits, boxes, [gt_instances], [dict(img_shape=(100, 100))],
        dn_meta)

    assert enabled_losses.keys() == disabled_losses.keys()
    for key in disabled_losses:
        torch.testing.assert_close(enabled_losses[key], disabled_losses[key])


def test_position_supervision_reaches_all_grouped_decoder_layers(monkeypatch):
    monkeypatch.setattr(
        RotatedGroundingDINOHead,
        '_get_targets_single',
        lambda self, *args, **kwargs: _parent_targets(target_x=0.5))
    seen_labels = []

    def record_single_layer_targets(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas):
        targets = self.get_targets(
            [cls_scores[i] for i in range(cls_scores.size(0))],
            [bbox_preds[i] for i in range(bbox_preds.size(0))],
            batch_gt_instances,
            batch_img_metas)
        seen_labels.append(targets[0][0][0].clone())
        zero = cls_scores.sum() * 0
        return zero, zero, zero

    monkeypatch.setattr(
        OVCapFlowHead, 'loss_by_feat_single', record_single_layer_targets)
    monkeypatch.setattr(
        OVCapFlowHead, '_denoising_loss_dict',
        lambda self, *args, **kwargs: {})
    head = object.__new__(OVCapFlowHead)
    torch.nn.Module.__init__(head)
    head.position_supervised_enabled = True
    head.adaptive_dn_enabled = False
    head.matching_query_groups = 3
    head.angle_factor = math.pi
    # This legacy fixture bypasses OVCapFlowHead.__init__, so mirror only the
    # disabled/empty matching-target capture state that __init__ owns.
    head._capture_matching_targets = False
    head._suppress_matching_target_capture = False
    head._captured_matching_mask = None
    head._loss_matching_group_masks = []
    cls_scores = torch.zeros(6, 1, 6, 3)
    bbox_preds = torch.zeros(6, 1, 6, 5)
    bbox_preds[..., 0] = 0.2
    bbox_preds[..., 1] = 0.2
    bbox_preds[..., 2] = 0.1
    bbox_preds[..., 3] = 0.1
    for layer in range(6):
        for group in range(3):
            positive_query = group * 2
            offset = layer * 3 + group
            bbox_preds[layer, 0, positive_query] = torch.tensor(
                [0.5 + offset / 100, 0.5, 0.4, 0.4, 0.0])

    head.loss_by_feat(
        cls_scores,
        bbox_preds,
        enc_cls_scores=None,
        enc_bbox_preds=None,
        batch_gt_instances=[InstanceData()],
        batch_img_metas=[dict(img_shape=(100, 100))],
        dn_meta=dict(
            num_denoising_queries=0,
            num_denoising_groups=1,
            num_matching_query_groups=3,
            num_matching_queries_per_group=2))

    assert len(seen_labels) == 18
    expected_labels = []
    for group in range(3):
        for layer in range(6):
            offset = layer * 3 + group
            quality = (40 - offset) / (40 + offset)
            expected_labels.append(torch.tensor([
                0.5 * quality, 0.5 * quality, 0.0,
            ]))
    for actual, expected in zip(seen_labels, expected_labels):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _partial_overlap_inputs():
    labels = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ])
    bbox_pred = torch.tensor([
        [0.5, 0.5, 0.4, 0.4, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.0],
    ], requires_grad=True)
    bbox_targets = torch.tensor([
        [0.6, 0.5, 0.4, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    return labels, bbox_pred, bbox_targets


def test_partial_overlap_scales_original_map_and_preserves_unmatched_rows():
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()

    scaled = scale_positive_maps_by_rotated_iou(
        labels=labels,
        bbox_pred=bbox_pred,
        bbox_targets=bbox_targets,
        pos_inds=torch.tensor([0]),
        img_meta=dict(img_shape=(100, 100)),
        angle_factor=math.pi)

    torch.testing.assert_close(
        scaled, torch.tensor([[0.3, 0.3, 0.0], [0.0, 0.0, 0.0]]),
        rtol=1e-5, atol=1e-6)
    assert scaled[0, 0] / scaled[0, 1] == 1


def test_perfect_and_near_degenerate_overlap_are_finite():
    labels = torch.tensor([[0.25, 0.75]])
    boxes = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])

    scaled = scale_positive_maps_by_rotated_iou(
        labels, boxes, boxes.clone(), torch.tensor([0]),
        dict(img_shape=(100, 100)), math.pi)

    assert torch.isfinite(scaled).all()
    torch.testing.assert_close(scaled, labels)


def test_zero_positive_returns_same_tensor_without_calling_iou(monkeypatch):
    labels = torch.zeros(3, 4)
    monkeypatch.setattr(
        head_module, 'rbbox_overlaps',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('IoU must not run')))

    scaled = scale_positive_maps_by_rotated_iou(
        labels=labels,
        bbox_pred=torch.zeros(3, 5),
        bbox_targets=torch.zeros(3, 5),
        pos_inds=torch.empty(0, dtype=torch.long),
        img_meta=dict(img_shape=(100, 100)),
        angle_factor=math.pi)

    assert scaled is labels


def test_nonfloating_labels_fail():
    with pytest.raises(ValueError, match='floating'):
        scale_positive_maps_by_rotated_iou(
            labels=torch.tensor([[1, 0]], dtype=torch.long),
            bbox_pred=torch.zeros(1, 5),
            bbox_targets=torch.zeros(1, 5),
            pos_inds=torch.tensor([0]),
            img_meta=dict(img_shape=(100, 100)),
            angle_factor=math.pi)


def test_nonfinite_labels_fail_before_empty_positive_return():
    with pytest.raises(ValueError, match='finite'):
        scale_positive_maps_by_rotated_iou(
            labels=torch.tensor([[float('nan'), 0.0]]),
            bbox_pred=torch.zeros(1, 5),
            bbox_targets=torch.zeros(1, 5),
            pos_inds=torch.empty(0, dtype=torch.long),
            img_meta=dict(img_shape=(100, 100)),
            angle_factor=math.pi)


def test_nonfinite_positive_box_fails_instead_of_repairing():
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()
    bbox_pred = bbox_pred.detach()
    bbox_pred[0, 0] = float('nan')

    with pytest.raises(ValueError, match='finite'):
        scale_positive_maps_by_rotated_iou(
            labels, bbox_pred, bbox_targets, torch.tensor([0]),
            dict(img_shape=(100, 100)), math.pi)


def test_nonfinite_iou_output_fails_instead_of_repairing(monkeypatch):
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()
    monkeypatch.setattr(
        head_module, 'rbbox_overlaps',
        lambda *args, **kwargs: torch.tensor([float('nan')]))

    with pytest.raises(ValueError, match='aligned rotated IoU'):
        scale_positive_maps_by_rotated_iou(
            labels, bbox_pred, bbox_targets, torch.tensor([0]),
            dict(img_shape=(100, 100)), math.pi)


def test_quality_target_has_no_gradient_path_to_bbox_prediction():
    labels, bbox_pred, bbox_targets = _partial_overlap_inputs()
    target = scale_positive_maps_by_rotated_iou(
        labels, bbox_pred, bbox_targets, torch.tensor([0]),
        dict(img_shape=(100, 100)), math.pi)
    logits = torch.zeros_like(target, requires_grad=True)

    F.binary_cross_entropy_with_logits(logits, target).backward()

    assert logits.grad is not None
    assert bbox_pred.grad is None
    assert target.grad_fn is None
