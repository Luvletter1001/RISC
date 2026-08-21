import math

import pytest
import torch
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmengine.structures import InstanceData

from mmrotate.structures.bbox import RotatedBoxes


class _FixedDNAssigner:

    def __init__(self):
        self.seen = None

    def assign(self, pred_instances, dn_instances, gt_instances, dn_meta,
               img_meta):
        self.seen = (pred_instances, dn_instances, gt_instances, dn_meta,
                     img_meta)
        return AssignResult(
            num_gts=2,
            gt_inds=torch.tensor([1, 0, 0, 2]),
            max_overlaps=None,
            labels=torch.tensor([0, -1, -1, 1]))


def test_periodic_angle_noise_wraps_without_changing_other_dimensions():
    from mmrotate.models.layers.transformer.rhino_layers import (
        add_periodic_angle_noise, )

    boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.1, 0.95],
        [0.2, 0.3, 0.1, 0.2, 0.05],
    ])
    noise = torch.tensor([0.10, -0.10])
    noisy = add_periodic_angle_noise(boxes, noise)

    assert torch.equal(noisy[:, :4], boxes[:, :4])
    assert noisy[:, 4].tolist() == pytest.approx([0.05, 0.95])
    assert ((noisy[:, 4] >= 0) & (noisy[:, 4] < 1)).all()


def test_rotated_cdn_angle_noise_is_optional_periodic_and_finite():
    from mmrotate.models.layers.transformer.rhino_layers import (
        RotatedCdnQueryGenerator, )

    common = dict(
        num_classes=3,
        embed_dims=8,
        num_matching_queries=6,
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=False, num_groups=2))
    generator = RotatedCdnQueryGenerator(
        **common, angle_noise_scale=0.25)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.98]])

    torch.manual_seed(7)
    logits = generator.generate_dn_bbox_query(boxes, num_groups=2)
    noisy = logits.sigmoid()

    assert noisy.shape == (4, 5)
    assert torch.isfinite(logits).all()
    assert (noisy[:, 2:4] > 0).all()
    assert ((noisy[:, 4] > 0) & (noisy[:, 4] < 1)).all()
    assert not torch.allclose(noisy[:, 4], boxes[:, 4].expand(4))


def test_adaptive_dn_targets_keep_only_group_consistent_positive_maps():
    from projects.OVCapFlow.ov_capflow.adaptive_dn import (
        build_adaptive_dn_targets_single, )

    assigner = _FixedDNAssigner()
    positive_maps = torch.tensor([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
    ], dtype=torch.float32)
    gt_instances = InstanceData(
        bboxes=RotatedBoxes(torch.tensor([
            [20, 30, 10, 6, 0.1],
            [70, 50, 20, 8, -0.2],
        ], dtype=torch.float32)),
        labels=torch.tensor([0, 1]),
        positive_maps=positive_maps)
    dn_cls = torch.randn(8, 6)
    dn_bbox = torch.rand(8, 5)
    matching_cls = torch.randn(6, 6)
    matching_bbox = torch.rand(6, 5)

    targets = build_adaptive_dn_targets_single(
        dn_cls_score=dn_cls,
        dn_bbox_pred=dn_bbox,
        matching_cls_score=matching_cls,
        matching_bbox_pred=matching_bbox,
        gt_instances=gt_instances,
        img_meta=dict(img_shape=(100, 100)),
        dn_meta=dict(num_denoising_queries=8, num_denoising_groups=2),
        dn_assigner=assigner,
        max_text_len=6,
        angle_factor=math.pi,
        angle_cfg=dict(width_longer=True, start_angle=0))
    labels, label_weights, bbox_targets, bbox_weights, pos, neg = targets

    assert pos.tolist() == [0, 5]
    assert set(neg.tolist()) == {1, 2, 3, 4, 6, 7}
    assert torch.equal(labels[0], positive_maps[0])
    assert torch.equal(labels[5], positive_maps[1])
    assert labels[[1, 2, 3, 4, 6, 7]].count_nonzero() == 0
    assert torch.equal(label_weights, torch.ones(8))
    assert bbox_weights.nonzero(as_tuple=False)[:, 0].unique().tolist() == [0, 5]
    assert torch.isfinite(bbox_targets).all()
    assert assigner.seen[0].scores.shape == (6, 6)
    assert assigner.seen[1].scores.shape == (4, 6)


def test_adaptive_dn_rejects_malformed_group_layout():
    from projects.OVCapFlow.ov_capflow.adaptive_dn import (
        build_adaptive_dn_targets_single, )

    gt_instances = InstanceData(
        bboxes=RotatedBoxes(torch.empty(0, 5)),
        labels=torch.empty(0, dtype=torch.long),
        positive_maps=torch.empty(0, 4))
    with pytest.raises(ValueError, match='divisible'):
        build_adaptive_dn_targets_single(
            torch.empty(7, 4), torch.empty(7, 5),
            torch.empty(6, 4), torch.empty(6, 5), gt_instances,
            dict(img_shape=(100, 100)),
            dict(num_denoising_queries=7, num_denoising_groups=2),
            _FixedDNAssigner(), 4, math.pi,
            dict(width_longer=True, start_angle=0))


def test_head_adaptive_dn_uses_primary_matching_queries(monkeypatch):
    from projects.OVCapFlow.ov_capflow.ov_capflow_head import OVCapFlowHead

    head = OVCapFlowHead.__new__(OVCapFlowHead)
    head.matching_query_groups = 1
    head.adaptive_dn_enabled = True
    matching_calls = []
    adaptive_calls = []

    def matching_loss(self, cls_scores, box_preds, *args, **kwargs):
        matching_calls.append((cls_scores.clone(), box_preds.clone()))
        return {'loss_cls': cls_scores.mean()}

    def adaptive_loss(self, dn_cls, dn_bbox, matching_cls, matching_bbox,
                      *args, **kwargs):
        adaptive_calls.append(
            (dn_cls.clone(), dn_bbox.clone(), matching_cls.clone(),
             matching_bbox.clone()))
        return {'dn_loss_cls': dn_cls.new_tensor(3.0)}

    monkeypatch.setattr(
        OVCapFlowHead, '_matching_loss_by_feat', matching_loss, raising=False)
    monkeypatch.setattr(
        OVCapFlowHead, '_adaptive_denoising_loss_dict', adaptive_loss,
        raising=False)
    cls_scores = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    bbox_preds = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    losses = OVCapFlowHead.loss_by_feat(
        head, cls_scores, bbox_preds, None, None, [], [],
        dict(num_denoising_queries=2, num_denoising_groups=1,
             num_matching_query_groups=1,
             num_matching_queries_per_group=2))

    assert len(matching_calls) == 1
    assert len(adaptive_calls) == 1
    assert adaptive_calls[0][2].shape[-2] == 2
    assert adaptive_calls[0][2][0, 0, 0, 0].item() == 2
    assert losses['dn_loss_cls'].item() == 3.0
