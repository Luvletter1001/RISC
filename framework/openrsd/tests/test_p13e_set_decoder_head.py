import math

import torch
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmrotate.structures import RotatedBoxes

from M_AD.models.dense_heads.p13e_set_decoder_head import (
    P13EDenseSeedGaussianSetHead,
    P13FGaussianPriorSetHead,
    P13GCalibratedGaussianSetHead,
    P13HRankedGaussianPosteriorHead,
    P13IStratifiedLatticePosteriorHead,
    P13JAssignedLatticePosteriorHead,
    P13KBalancedAssignedPosteriorHead,
    P13LClassLockedBalancedPosteriorHead,
    P13MClasswiseRankLockedPosteriorHead,
    P13NTeacherAnchoredPosteriorHead,
    P14RotatedDINOQueryTransportHead,
    P14BReferenceBoxQueryTransportHead,
    P14CClassConsistentQueryTransportHead,
    P14DStratifiedClassQueryTransportHead,
    P14EQualityCalibratedQueryTransportHead,
    P14FDetachedQualityQueryTransportHead,
    P14GGaussianSeedObjectnessQueryTransportHead,
    P14HDecoupledQueryTransportHead,
    P14IGaussianReferenceQueryTransportHead,
    P14JOneToManyQueryTransportHead,
    P14KGroupWiseQueryTransportHead,
    P14LGaussianGeometryQueryTransportHead,
    P14MIterativeRefineQueryTransportHead,
    P14NRankQualityQueryTransportHead,
    P14OQualityIntegratedClassPosteriorHead,
    _gaussian_wasserstein_loss,
)


def _sample_with_gt():
    sample = DetDataSample()
    sample.set_metainfo(
        dict(
            img_id='fake',
            img_shape=(64, 64),
            ori_shape=(64, 64),
            scale_factor=(1.0, 1.0)))
    gt = InstanceData()
    gt.bboxes = RotatedBoxes(
        torch.tensor([[24.0, 28.0, 12.0, 8.0, 0.1]], dtype=torch.float32))
    gt.labels = torch.tensor([1], dtype=torch.long)
    sample.gt_instances = gt
    sample.ignored_instances = InstanceData()
    sample.ignored_instances.bboxes = torch.empty(0, 5)
    sample.ignored_instances.labels = torch.empty(0, dtype=torch.long)
    return sample


def test_p13e_forward_shapes_are_fixed_query_set():
    init_default_scope('mmrotate')
    head = P13EDenseSeedGaussianSetHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=8,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=2.0,
    )
    head.eval()
    feats = (torch.randn(2, 4, 8, 8), torch.randn(2, 4, 4, 4))

    with torch.no_grad():
        cls_logits, box_preds, quality_logits, seed_logits = head(feats)

    assert cls_logits.shape == (2, 8, 3)
    assert box_preds.shape == (2, 8, 5)
    assert quality_logits.shape == (2, 8)
    assert seed_logits.shape == (2, 80)
    assert torch.all((box_preds[..., :4] > 0) & (box_preds[..., :4] <= 1))


def test_p13e_loss_and_predict_are_end_to_end_without_nms():
    init_default_scope('mmrotate')
    head = P13EDenseSeedGaussianSetHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=8,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=2.0,
        max_per_img=8,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    samples = [_sample_with_gt()]

    losses = head.loss(feats, samples)

    for key in ('loss_cls', 'loss_bbox', 'loss_angle', 'loss_quality',
                'loss_seed'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, samples, rescale=True)

    assert len(results) == 1
    assert results[0].bboxes.shape[1] == 5
    assert len(results[0].scores) <= 8
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False


def test_p13f_gaussian_prior_adds_denoising_losses():
    init_default_scope('mmrotate')
    head = P13FGaussianPriorSetHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.5,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.25,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])

    for key in ('loss_dn_cls', 'loss_dn_bbox', 'loss_dn_angle',
                'loss_dn_quality'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        head.predict(feats, [_sample_with_gt()])

    assert head.e2e_debug['uses_gaussian_query_prior'] is True
    assert head.e2e_debug['uses_denoising_queries'] is True


def test_p13g_calibrated_query_layout_and_quality_targets():
    init_default_scope('mmrotate')
    head = P13GCalibratedGaussianSetHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=12,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.2,
        dn_loss_weight=0.5,
        seed_gaussian_sigma=0.08,
        quality_center_sigma=0.12,
        max_per_img=6,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    cls_logits, box_preds, quality_logits, _ = head(feats)

    assert cls_logits.shape == (1, 12, 3)
    assert box_preds.shape == (1, 12, 5)
    assert quality_logits.shape == (1, 12)

    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_seed', 'loss_quality', 'loss_dn_quality'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()])

    assert len(results[0].scores) <= 6
    assert head.e2e_debug['uses_gaussian_seed_heatmap'] is True
    assert head.e2e_debug['uses_calibrated_quality'] is True
    assert head.e2e_debug['uses_per_class_query_layout'] is True


def test_p13h_ranked_posterior_adds_ranking_and_auxiliary_losses():
    init_default_scope('mmrotate')
    head = P13HRankedGaussianPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=12,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.2,
        dn_loss_weight=0.5,
        seed_gaussian_sigma=0.08,
        quality_center_sigma=0.12,
        ranking_loss_weight=0.5,
        aux_recall_loss_weight=0.5,
        box_delta_scale=0.6,
        max_per_img=6,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))

    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_rank', 'loss_aux_recall_cls',
                'loss_aux_recall_quality'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()])

    assert len(results[0].scores) <= 6
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_ranked_posterior'] is True
    assert head.e2e_debug['uses_size_prior'] is True
    assert head.e2e_debug['uses_one_to_many_aux'] is True
    assert head.box_delta_scale == 0.6


def test_p14_query_transport_uses_hard_box_priors_without_nms():
    init_default_scope('mmrotate')
    head = P14RotatedDINOQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_transport_loss_weight=0.75,
        query_transport_topk=3,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    sample = _sample_with_gt()

    query_boxes = torch.tensor(
        [[0.38, 0.44, 0.18, 0.12, 0.10],
         [0.80, 0.80, 0.16, 0.10, -0.20],
         [0.12, 0.18, 0.20, 0.16, 0.30]],
        dtype=torch.float32)
    gt_boxes = torch.tensor(
        [[24.0 / 64.0, 28.0 / 64.0, 12.0 / 64.0, 8.0 / 64.0, 0.10]],
        dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    targets = head._box_query_transport_targets(
        query_boxes,
        gt_boxes,
        gt_labels,
        teacher_bboxes=query_boxes.new_zeros(0, 5),
        teacher_labels=torch.empty(0, dtype=torch.long),
        teacher_scores=query_boxes.new_zeros(0))

    assert targets['positive_mask'].sum().item() == 1
    assert targets['positive_mask'][0].item() is True
    assert torch.allclose(targets['target_boxes'][0], gt_boxes[0])
    assert targets['target_labels'][0].item() == 1

    losses = head.loss(feats, [sample])
    for key in ('loss_query_transport_cls', 'loss_query_transport_bbox',
                'loss_query_transport_angle', 'loss_dn_bbox'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [sample], rescale=True)

    assert len(results[0].scores) == 9
    assert results[0].bboxes.shape[1] == 5
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_box_query_transport'] is True
    assert head.e2e_debug['uses_soft_gaussian_teacher_field'] is False


def test_p14b_uses_reference_box_priors_instead_of_stride_widths():
    init_default_scope('mmrotate')
    class_wh_priors = ((0.45, 0.35), (0.20, 0.12), (0.08, 0.06))
    head = P14BReferenceBoxQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=class_wh_priors,
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()
    with torch.no_grad():
        head.box_delta.weight.zero_()
        head.box_delta.bias.zero_()

    memory = torch.zeros(1, 3, 16)
    query_memory = torch.zeros(1, 3, 16)
    query_coords = torch.tensor([[
        [0.20, 0.25, 0.0],
        [0.50, 0.50, 0.5],
        [0.80, 0.75, 1.0],
    ]], dtype=torch.float32)
    query_strides = torch.full((1, 3, 1), 8.0)
    query_class_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    _, decoded_boxes, _ = head._decode_queries(
        memory,
        query_memory,
        query_coords,
        query_strides,
        query_class_ids=query_class_ids)

    expected_wh = torch.tensor(class_wh_priors, dtype=torch.float32)
    assert torch.allclose(decoded_boxes[0, :, 2:4], expected_wh, atol=1e-4)
    assert decoded_boxes[0, 0, 2] > 8.0 / 64.0 * 2.0

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_bbox', 'loss_query_transport_cls',
                'loss_dn_bbox'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_dab_reference_boxes'] is True
    assert head.e2e_debug['uses_stride_width_reference'] is False


def test_p14c_transport_is_class_consistent_and_ranked():
    init_default_scope('mmrotate')
    head = P14CClassConsistentQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()

    query_boxes = torch.tensor(
        [[0.38, 0.44, 0.18, 0.12, 0.10],
         [0.72, 0.72, 0.18, 0.12, 0.10],
         [0.90, 0.90, 0.18, 0.12, 0.10]],
        dtype=torch.float32)
    gt_boxes = torch.tensor(
        [[24.0 / 64.0, 28.0 / 64.0, 12.0 / 64.0, 8.0 / 64.0, 0.10]],
        dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    query_class_ids = torch.tensor([0, 1, 1], dtype=torch.long)
    targets = head._box_query_transport_targets(
        query_boxes,
        gt_boxes,
        gt_labels,
        teacher_bboxes=query_boxes.new_zeros(0, 5),
        teacher_labels=torch.empty(0, dtype=torch.long),
        teacher_scores=query_boxes.new_zeros(0),
        query_class_ids=query_class_ids)

    assert targets['positive_mask'].tolist() == [False, True, False]
    assert targets['target_labels'][1].item() == 1

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_bbox', 'loss_query_transport_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert results[0].labels.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_class_consistent_transport'] is True
    assert head.e2e_debug['uses_transport_rank_loss'] is True


def test_p14d_repeats_top_spatial_seeds_for_each_class():
    init_default_scope('mmrotate')
    head = P14DStratifiedClassQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    memory = torch.arange(1 * 12 * 16, dtype=torch.float32).view(1, 12, 16)
    coords = torch.stack([
        torch.linspace(0.05, 0.95, 12),
        torch.linspace(0.95, 0.05, 12),
        torch.zeros(12),
    ], dim=-1)
    strides = torch.full((12, 1), 8.0)
    seed_logits = torch.tensor(
        [[0.1, 0.2, 0.9, 0.8, 0.7, 0.0, -0.1, -0.2, -0.3, -0.4,
          -0.5, -0.6]],
        dtype=torch.float32)

    _, seed_coords, _, query_class_ids = head._select_query_seeds(
        memory, coords, strides, seed_logits)

    assert query_class_ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert torch.allclose(seed_coords[0, :3], seed_coords[0, 3:6])
    assert torch.allclose(seed_coords[0, :3], seed_coords[0, 6:9])

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    head.init_weights()
    losses = head.loss(feats, [_sample_with_gt()])
    assert torch.isfinite(losses['loss_query_transport_rank'])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert results[0].labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert head.e2e_debug['uses_stratified_class_queries'] is True
    assert head.e2e_debug['uses_nms'] is False


def test_p14e_geometry_quality_calibrates_same_class_queries():
    init_default_scope('mmrotate')
    head = P14EQualityCalibratedQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        geometry_quality_loss_weight=1.5,
        geometry_quality_cost_scale=0.75,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()

    query_boxes = torch.tensor(
        [[0.375, 0.438, 0.190, 0.125, 0.10],
         [0.620, 0.620, 0.190, 0.125, 0.10],
         [0.375, 0.438, 0.190, 0.125, 0.10]],
        dtype=torch.float32)
    query_class_ids = torch.tensor([1, 1, 2], dtype=torch.long)
    gt_boxes = torch.tensor(
        [[24.0 / 64.0, 28.0 / 64.0, 12.0 / 64.0, 8.0 / 64.0, 0.10]],
        dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)

    quality_targets = head._geometry_quality_targets(
        query_boxes,
        query_class_ids,
        gt_boxes,
        gt_labels,
        teacher_bboxes=query_boxes.new_zeros(0, 5),
        teacher_labels=torch.empty(0, dtype=torch.long),
        teacher_scores=query_boxes.new_zeros(0))

    assert quality_targets[0] > quality_targets[1]
    assert quality_targets[0] > 0.90
    assert quality_targets[2].item() == 0.0

    quality_logits = torch.tensor([-2.0, 2.0, 0.0])
    calibrated_loss = head._geometry_quality_loss(
        quality_logits, quality_targets)
    better_loss = head._geometry_quality_loss(
        torch.tensor([2.0, -2.0, 0.0]), quality_targets)
    assert calibrated_loss > better_loss

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_geo_quality',
                'loss_query_transport_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_geometry_quality_calibration'] is True


def test_p14f_detaches_quality_calibration_from_query_geometry():
    init_default_scope('mmrotate')
    head = P14FDetachedQualityQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        geometry_quality_loss_weight=1.5,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()
    memory = torch.randn(1, 3, 16)
    query_memory = torch.randn(1, 3, 16, requires_grad=True)
    query_coords = torch.tensor([[
        [0.20, 0.25, 0.0],
        [0.50, 0.50, 0.5],
        [0.80, 0.75, 1.0],
    ]], dtype=torch.float32)
    query_strides = torch.full((1, 3, 1), 8.0)
    query_class_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    _, _, quality_logits = head._decode_queries(
        memory,
        query_memory,
        query_coords,
        query_strides,
        query_class_ids=query_class_ids)
    quality_logits.sum().backward()

    assert query_memory.grad is None or torch.allclose(
        query_memory.grad, torch.zeros_like(query_memory.grad))
    assert head.quality.weight.grad is not None
    assert head.quality.weight.grad.abs().sum() > 0

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    assert torch.isfinite(losses['loss_query_transport_geo_quality'])

    with torch.no_grad():
        head.predict(feats, [_sample_with_gt()], rescale=True)

    assert head.e2e_debug['uses_detached_quality_features'] is True
    assert head.e2e_debug['uses_nms'] is False


def test_p14g_uses_gaussian_seed_objectness_in_final_posterior():
    init_default_scope('mmrotate')
    head = P14GGaussianSeedObjectnessQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        seed_gaussian_sigma=0.08,
        seed_score_power=0.5,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()

    coords = torch.tensor(
        [[0.375, 0.4375, 0.0], [0.41, 0.45, 0.0], [0.90, 0.90, 0.0]],
        dtype=torch.float32)
    gt_boxes = torch.tensor(
        [[24.0 / 64.0, 28.0 / 64.0, 12.0 / 64.0, 8.0 / 64.0, 0.10]],
        dtype=torch.float32)
    target = head._gaussian_seed_targets(coords, gt_boxes)
    assert target[0] > 0.99
    assert target[1] > 0.80
    assert target[2] < 1e-4

    good_logits = torch.tensor([4.0, 3.0, -4.0])
    bad_logits = torch.tensor([-4.0, -4.0, 4.0])
    assert head._seed_loss_single(good_logits, gt_boxes, coords) < (
        head._seed_loss_single(bad_logits, gt_boxes, coords))

    memory = torch.randn(1, 4, 16)
    strides = torch.full((4, 1), 8.0)
    seed_logits = torch.tensor([[0.1, 4.0, 0.3, 2.0]])
    _, seed_coords, _, query_class_ids, selected_scores = (
        head._select_query_seeds_with_scores(
            memory, coords=torch.cat([coords, coords[:1]], dim=0),
            strides=strides,
            seed_logits=seed_logits))
    assert selected_scores[0, 0].item() == 4.0
    assert seed_coords.shape[1] == 4
    assert query_class_ids.tolist() == [0, 1, 2, 0]

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_seed', 'loss_query_transport_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_gaussian_seed_heatmap'] is True
    assert head.e2e_debug['uses_encoder_seed_objectness_posterior'] is True


def test_p14h_decouples_classification_and_regression_query_features():
    init_default_scope('mmrotate')
    head = P14HDecoupledQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()
    memory = torch.randn(1, 3, 16)
    query_memory = torch.randn(1, 3, 16)
    query_coords = torch.tensor([[
        [0.20, 0.25, 0.0],
        [0.50, 0.50, 0.5],
        [0.80, 0.75, 1.0],
    ]], dtype=torch.float32)
    query_strides = torch.full((1, 3, 1), 8.0)
    query_class_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    cls_logits, box_preds, quality_logits = head._decode_queries(
        memory,
        query_memory,
        query_coords,
        query_strides,
        query_class_ids=query_class_ids)

    head.zero_grad(set_to_none=True)
    (cls_logits.sum() + quality_logits.sum()).backward(retain_graph=True)
    assert head.cls_query_proj.weight.grad is not None
    assert head.cls_query_proj.weight.grad.abs().sum() > 0
    assert head.reg_query_proj.weight.grad is None or torch.allclose(
        head.reg_query_proj.weight.grad,
        torch.zeros_like(head.reg_query_proj.weight.grad))
    assert head.box_delta.weight.grad is None or torch.allclose(
        head.box_delta.weight.grad,
        torch.zeros_like(head.box_delta.weight.grad))

    head.zero_grad(set_to_none=True)
    box_preds.sum().backward()
    assert head.reg_query_proj.weight.grad is not None
    assert head.reg_query_proj.weight.grad.abs().sum() > 0
    assert head.cls_query_proj.weight.grad is None or torch.allclose(
        head.cls_query_proj.weight.grad,
        torch.zeros_like(head.cls_query_proj.weight.grad))

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_bbox', 'loss_query_transport_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_decoupled_query_features'] is True


def test_p14i_encodes_rotated_reference_boxes_as_gaussian_geometry():
    init_default_scope('mmrotate')
    head = P14IGaussianReferenceQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        gaussian_geometry_weight=0.5,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()

    ref_boxes = torch.tensor(
        [[0.50, 0.50, 0.20, 0.10, 0.00],
         [0.50, 0.50, 0.20, 0.10, math.pi / 2.0],
         [0.50, 0.50, 0.10, 0.20, 0.00]],
        dtype=torch.float32)
    gaussian_features = head._gaussian_reference_features(ref_boxes)
    assert gaussian_features.shape == (3, 6)
    assert not torch.allclose(gaussian_features[0], gaussian_features[1])
    assert not torch.allclose(gaussian_features[0], gaussian_features[2])
    assert torch.all(gaussian_features[:, 2:5].isfinite())

    memory = torch.randn(2, 5, 16)
    query_memory = torch.randn(2, 4, 16)
    query_coords = torch.rand(2, 4, 3)
    query_strides = torch.full((2, 4, 1), 8.0)
    query_class_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    cls_logits, box_preds, quality_logits = head._decode_queries(
        memory,
        query_memory,
        query_coords,
        query_strides,
        query_class_ids=query_class_ids)
    assert cls_logits.shape == (2, 4, 3)
    assert box_preds.shape == (2, 4, 5)
    assert quality_logits.shape == (2, 4)

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_bbox', 'loss_query_transport_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_gaussian_reference_geometry'] is True


def test_p14j_assigns_multiple_same_class_queries_per_gt():
    init_default_scope('mmrotate')
    head = P14JOneToManyQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        aux_otm_loss_weight=0.8,
        aux_otm_topk_per_gt=2,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()

    query_boxes = torch.tensor(
        [[0.375, 0.438, 0.18, 0.12, 0.10],
         [0.395, 0.450, 0.18, 0.12, 0.08],
         [0.380, 0.440, 0.18, 0.12, 0.10],
         [0.900, 0.900, 0.18, 0.12, 0.10]],
        dtype=torch.float32)
    query_class_ids = torch.tensor([1, 1, 2, 1], dtype=torch.long)
    gt_boxes = torch.tensor(
        [[24.0 / 64.0, 28.0 / 64.0, 12.0 / 64.0, 8.0 / 64.0, 0.10]],
        dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    targets = head._one_to_many_transport_targets(
        query_boxes, query_class_ids, gt_boxes, gt_labels)

    assert targets['positive_mask'].tolist() == [True, True, False, False]
    assert torch.allclose(targets['target_boxes'][0], gt_boxes[0])
    assert torch.allclose(targets['target_boxes'][1], gt_boxes[0])

    cls_logits = torch.zeros(4, 3)
    quality_logits = torch.zeros(4)
    losses = head._one_to_many_transport_loss_single(
        cls_logits, query_boxes, quality_logits, query_class_ids, gt_boxes,
        gt_labels)
    for loss in losses:
        assert torch.isfinite(loss)

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    model_losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_otm_cls',
                'loss_query_transport_otm_bbox',
                'loss_query_transport_otm_quality'):
        assert key in model_losses
        assert torch.isfinite(model_losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_one_to_many_auxiliary_transport'] is True


def test_p14k_uses_groupwise_training_and_single_group_inference():
    init_default_scope('mmrotate')
    head = P14KGroupWiseQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=12,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        group_transport_loss_weight=0.6,
        query_group_count=3,
        inference_single_group=True,
        dn_loss_weight=0.5,
        max_per_img=12,
    )
    head.init_weights()

    group_slices = head._query_group_slices(12)
    assert [(s.start, s.stop) for s in group_slices] == [(0, 4), (4, 8),
                                                         (8, 12)]
    assert head._primary_query_slice(12).stop == 4

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_group_cls',
                'loss_query_transport_group_bbox',
                'loss_query_transport_group_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 4
    assert results[0].labels.tolist() == [0, 1, 2, 0]
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_groupwise_query_transport'] is True
    assert head.e2e_debug['inference_single_group'] is True
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False


def test_p14l_adds_gaussian_geometry_loss_for_transport_positives():
    init_default_scope('mmrotate')
    target = torch.tensor([[0.38, 0.44, 0.20, 0.12, 0.10]])
    near = torch.tensor([[0.39, 0.45, 0.20, 0.12, 0.10]])
    far = torch.tensor([[0.80, 0.80, 0.08, 0.08, -0.60]])
    assert _gaussian_wasserstein_loss(near, target).item() < (
        _gaussian_wasserstein_loss(far, target).item())

    head = P14LGaussianGeometryQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.40,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        gwd_loss_weight=2.0,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()

    box_preds = torch.tensor(
        [[0.38, 0.44, 0.20, 0.12, 0.10],
         [0.80, 0.80, 0.08, 0.08, -0.60]],
        dtype=torch.float32,
        requires_grad=True)
    targets = dict(
        positive_mask=torch.tensor([True, False]),
        target_boxes=torch.tensor(
            [[0.375, 0.438, 0.19, 0.125, 0.10],
             [0.00, 0.00, 0.00, 0.00, 0.00]],
            dtype=torch.float32))
    gwd_loss = head._transport_gwd_loss(box_preds, targets)
    assert torch.isfinite(gwd_loss)
    gwd_loss.backward()
    assert box_preds.grad is not None

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    assert 'loss_query_transport_gwd' in losses
    assert torch.isfinite(losses['loss_query_transport_gwd'])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_gaussian_transport_geometry_loss'] is True


def test_p14m_refines_reference_boxes_inside_decoder_layers():
    init_default_scope('mmrotate')
    head = P14MIterativeRefineQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=2,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        center_delta_scale=0.20,
        iterative_center_delta_scale=0.10,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        intermediate_refine_loss_weight=0.4,
        dn_loss_weight=0.5,
        max_per_img=9,
    )
    head.init_weights()
    with torch.no_grad():
        head.box_delta.weight.zero_()
        head.box_delta.bias.copy_(
            torch.tensor([0.5, -0.2, 0.1, -0.1, 0.25]))

    memory = torch.zeros(1, 6, 16)
    query_memory = torch.zeros(1, 4, 16)
    query_coords = torch.tensor([[
        [0.20, 0.25, 0.0],
        [0.50, 0.50, 0.5],
        [0.80, 0.75, 1.0],
        [0.35, 0.70, 0.5],
    ]], dtype=torch.float32)
    query_strides = torch.full((1, 4, 1), 8.0)
    query_class_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    cls_logits, box_preds, quality_logits = head._decode_queries(
        memory,
        query_memory,
        query_coords,
        query_strides,
        query_class_ids=query_class_ids)
    intermediates = head._p14m_last_intermediate

    assert cls_logits.shape == (1, 4, 3)
    assert box_preds.shape == (1, 4, 5)
    assert quality_logits.shape == (1, 4)
    assert len(intermediates['boxes']) == 2
    assert not torch.allclose(intermediates['boxes'][0],
                              intermediates['boxes'][1])
    assert intermediates['boxes'][1][0, 0, 0] > intermediates['boxes'][0][0,
                                                                         0, 0]

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_refine_aux_bbox',
                'loss_query_refine_aux_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_iterative_reference_refinement'] is True


def test_p14n_rank_quality_targets_follow_geometry():
    init_default_scope('mmrotate')
    head = P14NRankQualityQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        rank_quality_floor=0.25,
        max_per_img=9,
    )
    aligned = torch.tensor(
        [[0.375, 0.4375, 0.1875, 0.1250, 0.10],
         [0.700, 0.7200, 0.2600, 0.2000, 0.80]],
        dtype=torch.float32)
    target = torch.tensor(
        [[0.375, 0.4375, 0.1875, 0.1250, 0.10],
         [0.375, 0.4375, 0.1875, 0.1250, 0.10]],
        dtype=torch.float32)

    quality = head._rank_quality_from_boxes(aligned, target)

    assert quality[0] > 0.95
    assert quality[1] < quality[0] * 0.5
    assert quality[1] >= 0.25

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    head.init_weights()
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_query_transport_cls', 'loss_query_transport_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()], rescale=True)

    assert len(results[0].scores) == 9
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_rank_quality_posterior'] is True


def test_p14n_can_keep_binary_class_targets_for_quality_only_ranking():
    init_default_scope('mmrotate')
    head = P14NRankQualityQueryTransportHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=9,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        rank_quality_floor=0.25,
        rank_quality_cls_target_blend=0.0,
        max_per_img=9,
    )
    source_scores = torch.tensor([1.0, 0.4], dtype=torch.float32)
    rank_quality = torch.tensor([0.30, 0.80], dtype=torch.float32)

    cls_target = head._rank_quality_class_target(source_scores, rank_quality)

    assert torch.allclose(cls_target, source_scores)
    head.rank_quality_cls_target_blend = 1.0
    assert torch.allclose(
        head._rank_quality_class_target(source_scores, rank_quality),
        rank_quality)


def test_p14o_predict_uses_quality_integrated_class_score_only():
    init_default_scope('mmrotate')
    head = P14OQualityIntegratedClassPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=3,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        class_wh_priors=((0.45, 0.35), (0.20, 0.12), (0.08, 0.06)),
        learn_reference_wh=False,
        query_transport_loss_weight=0.75,
        transport_rank_loss_weight=0.5,
        rank_quality_floor=0.25,
        max_per_img=3,
    )
    cls_logits = torch.tensor([[
        [2.0, -3.0, -3.0],
        [-3.0, 1.0, -3.0],
        [-3.0, -3.0, 0.5],
    ]])
    box_preds = torch.tensor([[
        [0.25, 0.25, 0.20, 0.10, 0.0],
        [0.50, 0.50, 0.20, 0.10, 0.0],
        [0.75, 0.75, 0.20, 0.10, 0.0],
    ]])
    quality_logits = torch.full((1, 3), -8.0)

    def fake_forward(_x):
        return cls_logits, box_preds, quality_logits, None

    head.forward = fake_forward
    with torch.no_grad():
        results = head.predict((torch.empty(1), ), [_sample_with_gt()])

    expected = torch.sigmoid(torch.tensor([2.0, 1.0, 0.5]))
    multiplied = expected * torch.sigmoid(torch.tensor(-8.0))

    assert torch.allclose(results[0].scores, expected, atol=1e-6)
    assert not torch.allclose(results[0].scores, multiplied, atol=1e-4)
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_quality_integrated_class_posterior'] is True
    assert head.e2e_debug['uses_quality_score_product'] is False


def test_p13i_lattice_posterior_adds_auxiliary_box_pull():
    init_default_scope('mmrotate')
    head = P13IStratifiedLatticePosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.2,
        dn_loss_weight=0.5,
        seed_gaussian_sigma=0.08,
        quality_center_sigma=0.12,
        ranking_loss_weight=0.5,
        aux_recall_loss_weight=0.5,
        aux_box_loss_weight=2.0,
        box_delta_scale=0.6,
        max_per_img=6,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))

    cls_logits, box_preds, quality_logits, _ = head(feats)
    assert cls_logits.shape == (1, 18, 3)
    assert box_preds.shape == (1, 18, 5)
    assert quality_logits.shape == (1, 18)

    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_rank', 'loss_aux_recall_cls',
                'loss_aux_recall_quality', 'loss_aux_recall_bbox',
                'loss_aux_recall_angle'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()])

    assert len(results[0].scores) <= 6
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_stratified_lattice_queries'] is True
    assert head.e2e_debug['uses_auxiliary_box_pull'] is True


def test_p13j_assigned_lattice_uses_query_coords_for_gt_binding():
    init_default_scope('mmrotate')
    head = P13JAssignedLatticePosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.2,
        dn_loss_weight=0.5,
        seed_gaussian_sigma=0.08,
        quality_center_sigma=0.12,
        ranking_loss_weight=0.5,
        assigned_lattice_loss_weight=0.7,
        assigned_lattice_sigma=0.18,
        assigned_topk_per_gt=2,
        box_delta_scale=0.6,
        max_per_img=6,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))

    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_assigned_lattice_cls',
                'loss_assigned_lattice_quality',
                'loss_assigned_lattice_bbox',
                'loss_assigned_lattice_angle',
                'loss_assigned_lattice_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    query_coords = torch.tensor(
        [[0.375, 0.4375, 0.0], [0.90, 0.90, 0.0]], dtype=torch.float32)
    gt_bboxes = torch.tensor(
        [[0.375, 0.4375, 0.20, 0.12, 0.0]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    query_class_ids = torch.tensor([1, 1], dtype=torch.long)
    cls_logits = torch.zeros(2, 3)
    quality_logits = torch.zeros(2)
    shifted_boxes = torch.tensor(
        [[0.80, 0.80, 0.10, 0.10, 0.0], [0.90, 0.90, 0.10, 0.10, 0.0]],
        dtype=torch.float32)

    _, _, assigned_bbox, _, _ = head._assigned_lattice_loss_single(
        cls_logits, shifted_boxes, quality_logits, query_coords,
        query_class_ids, gt_bboxes, gt_labels)

    assert assigned_bbox > 0.5

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()])

    assert len(results[0].scores) <= 6
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_assigned_lattice_gt_binding'] is True


def test_p13k_balances_positive_quality_and_expands_center_delta():
    init_default_scope('mmrotate')
    head = P13KBalancedAssignedPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.2,
        dn_loss_weight=0.5,
        seed_gaussian_sigma=0.08,
        quality_center_sigma=0.12,
        ranking_loss_weight=0.5,
        assigned_lattice_loss_weight=0.7,
        assigned_lattice_sigma=0.12,
        assigned_topk_per_gt=1,
        assigned_positive_quality_gain=4.0,
        center_delta_scale=0.35,
        box_delta_scale=0.6,
        max_per_img=6,
    )
    head.init_weights()
    with torch.no_grad():
        head.box_delta.weight.zero_()
        head.box_delta.bias.zero_()
        head.box_delta.bias[:2] = torch.atanh(
            torch.tensor([0.5, -0.5], dtype=head.box_delta.bias.dtype))

    memory = torch.zeros(1, 2, 16)
    query_memory = torch.zeros(1, 1, 16)
    query_coords = torch.tensor([[[0.5, 0.5, 0.0]]], dtype=torch.float32)
    query_strides = torch.ones(1, 1, 1)
    query_class_ids = torch.tensor([1], dtype=torch.long)

    _, decoded_boxes, _ = head._decode_queries(
        memory,
        query_memory,
        query_coords,
        query_strides,
        query_class_ids=query_class_ids)

    assert torch.allclose(
        decoded_boxes[0, 0, :2],
        torch.tensor([0.675, 0.325]),
        atol=1e-4)

    p13j = P13JAssignedLatticePosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        assigned_lattice_loss_weight=0.7,
        assigned_lattice_sigma=0.12,
        assigned_topk_per_gt=1,
        max_per_img=6,
    )

    query_coords = torch.tensor(
        [[0.375, 0.4375, 0.0], [0.90, 0.90, 0.0],
         [0.10, 0.90, 0.0]],
        dtype=torch.float32)
    gt_bboxes = torch.tensor(
        [[0.375, 0.4375, 0.20, 0.12, 0.0]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    query_class_ids = torch.tensor([1, 1, 2], dtype=torch.long)
    cls_logits = torch.zeros(3, 3)
    quality_logits = torch.full((3,), -4.0)
    box_preds = torch.tensor(
        [[0.80, 0.80, 0.10, 0.10, 0.0],
         [0.90, 0.90, 0.10, 0.10, 0.0],
         [0.10, 0.90, 0.10, 0.10, 0.0]],
        dtype=torch.float32)

    p13j_quality = p13j._assigned_lattice_loss_single(
        cls_logits, box_preds, quality_logits, query_coords,
        query_class_ids, gt_bboxes, gt_labels)[1]
    p13k_quality = head._assigned_lattice_loss_single(
        cls_logits, box_preds, quality_logits, query_coords,
        query_class_ids, gt_bboxes, gt_labels)[1]

    assert p13k_quality > p13j_quality * 1.5

    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_assigned_lattice_cls',
                'loss_assigned_lattice_quality',
                'loss_assigned_lattice_bbox',
                'loss_assigned_lattice_angle',
                'loss_assigned_lattice_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()])

    assert len(results[0].scores) <= 6
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_balanced_assigned_posterior'] is True
    assert head.e2e_debug['uses_expanded_center_delta'] is True
    assert head.center_delta_scale == 0.35


def test_p13l_class_locked_selects_own_class_queries_without_nms():
    init_default_scope('mmrotate')
    head = P13LClassLockedBalancedPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.2,
        dn_loss_weight=0.5,
        seed_gaussian_sigma=0.08,
        quality_center_sigma=0.12,
        ranking_loss_weight=0.5,
        assigned_lattice_loss_weight=0.7,
        assigned_lattice_sigma=0.12,
        assigned_topk_per_gt=1,
        assigned_positive_quality_gain=4.0,
        center_delta_scale=0.35,
        per_class_max_per_img=1,
        score_thr=0.0,
        max_per_img=3,
    )

    scores_per_class = torch.tensor([
        [0.10, 0.99, 0.01],
        [0.20, 0.01, 0.01],
        [0.90, 0.10, 0.01],
        [0.10, 0.30, 0.01],
        [0.90, 0.02, 0.10],
        [0.10, 0.02, 0.40],
    ])
    query_class_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    keep, scores, labels = head._class_locked_select(
        scores_per_class, query_class_ids)

    assert keep.tolist() == [1, 3, 5]
    assert labels.tolist() == [0, 1, 2]
    assert torch.allclose(scores, torch.tensor([0.20, 0.30, 0.40]))

    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    assert torch.isfinite(losses['loss_assigned_lattice_rank'])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()])

    assert len(results[0].scores) <= 3
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_class_locked_inference'] is True
    assert head.e2e_debug['uses_per_class_output_quota'] is True


def test_p13m_classwise_rank_ignores_other_class_hard_negatives():
    init_default_scope('mmrotate')
    baseline = P13LClassLockedBalancedPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        assigned_lattice_sigma=0.12,
        assigned_topk_per_gt=1,
        ranking_margin=0.10,
        score_thr=0.0,
        max_per_img=6,
    )
    head = P13MClasswiseRankLockedPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        assigned_lattice_sigma=0.12,
        assigned_topk_per_gt=1,
        ranking_margin=0.10,
        score_thr=0.0,
        max_per_img=6,
    )

    query_coords = torch.tensor(
        [[0.375, 0.4375, 0.0], [0.90, 0.90, 0.0],
         [0.10, 0.90, 0.0]],
        dtype=torch.float32)
    gt_bboxes = torch.tensor(
        [[0.375, 0.4375, 0.20, 0.12, 0.0]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    query_class_ids = torch.tensor([1, 1, 2], dtype=torch.long)
    cls_logits = torch.zeros(3, 3)
    cls_logits[0, 1] = 0.50
    cls_logits[1, 1] = -4.00
    cls_logits[2, 2] = 8.00
    quality_logits = torch.zeros(3)
    box_preds = torch.tensor(
        [[0.38, 0.44, 0.20, 0.12, 0.0],
         [0.90, 0.90, 0.10, 0.10, 0.0],
         [0.10, 0.90, 0.10, 0.10, 0.0]],
        dtype=torch.float32)

    baseline_rank = baseline._assigned_lattice_loss_single(
        cls_logits, box_preds, quality_logits, query_coords,
        query_class_ids, gt_bboxes, gt_labels)[4]
    classwise_rank = head._assigned_lattice_loss_single(
        cls_logits, box_preds, quality_logits, query_coords,
        query_class_ids, gt_bboxes, gt_labels)[4]

    assert classwise_rank < baseline_rank * 0.5

    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))
    losses = head.loss(feats, [_sample_with_gt()])
    assert torch.isfinite(losses['loss_assigned_lattice_rank'])

    with torch.no_grad():
        head.predict(feats, [_sample_with_gt()])

    assert head.e2e_debug['uses_classwise_assigned_rank'] is True
    assert head.e2e_debug['uses_class_locked_inference'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False


def test_p13n_teacher_targets_keep_gt_priority_over_teacher():
    init_default_scope('mmrotate')
    head = P13NTeacherAnchoredPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        teacher_gaussian_sigma=0.10,
        teacher_score_power=1.0,
        max_per_img=18,
    )

    query_coords = torch.tensor(
        [[0.375, 0.4375, 0.0], [0.75, 0.75, 0.0]],
        dtype=torch.float32)
    query_class_ids = torch.tensor([1, 1], dtype=torch.long)
    gt_bboxes = torch.tensor(
        [[0.375, 0.4375, 0.20, 0.12, 0.0]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    teacher_bboxes = torch.tensor(
        [[0.39, 0.45, 0.55, 0.50, 0.9]], dtype=torch.float32)
    teacher_labels = torch.tensor([1], dtype=torch.long)
    teacher_scores = torch.tensor([0.70], dtype=torch.float32)

    soft_obj, target_boxes, target_labels = head._teacher_augmented_targets(
        query_coords, query_class_ids, gt_bboxes, gt_labels, teacher_bboxes,
        teacher_labels, teacher_scores)

    assert soft_obj[0] > 0.99
    assert target_labels[0].item() == 1
    assert torch.allclose(target_boxes[0], gt_bboxes[0], atol=1e-5)


def test_p13n_teacher_only_signal_creates_finite_loss():
    init_default_scope('mmrotate')
    head = P13NTeacherAnchoredPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        teacher_loss_weight=1.0,
        teacher_gaussian_sigma=0.12,
        max_per_img=18,
    )

    query_coords = torch.tensor(
        [[0.25, 0.25, 0.0], [0.70, 0.70, 0.0]], dtype=torch.float32)
    query_class_ids = torch.tensor([2, 2], dtype=torch.long)
    teacher_bboxes = torch.tensor(
        [[0.25, 0.25, 0.18, 0.16, -0.2]], dtype=torch.float32)
    teacher_labels = torch.tensor([2], dtype=torch.long)
    teacher_scores = torch.tensor([0.90], dtype=torch.float32)
    cls_logits = torch.zeros(2, 3)
    quality_logits = torch.full((2,), -2.0)
    box_preds = torch.tensor(
        [[0.70, 0.70, 0.10, 0.10, 0.0],
         [0.70, 0.70, 0.10, 0.10, 0.0]],
        dtype=torch.float32)

    losses = head._teacher_anchored_loss_single(
        cls_logits, box_preds, quality_logits, query_coords,
        query_class_ids, torch.empty(0, 5), torch.empty(0, dtype=torch.long),
        teacher_bboxes, teacher_labels, teacher_scores)

    for loss in losses:
        assert torch.isfinite(loss)
    assert losses[0] > 0
    assert losses[1] > 0
    assert losses[2] > 0


def test_p13n_predict_outputs_fixed_set_without_nms_or_fallback():
    init_default_scope('mmrotate')
    head = P13NTeacherAnchoredPosteriorHead(
        num_classes=3,
        in_channels=4,
        feat_channels=16,
        num_queries=18,
        num_decoder_layers=1,
        num_heads=4,
        strides=(8, 16),
        image_size=(64, 64),
        logit_scale_init=1.25,
        query_class_prior_weight=0.5,
        own_class_logit_bias=0.2,
        dn_loss_weight=0.5,
        teacher_loss_weight=0.5,
        score_thr=0.99,
        max_per_img=18,
    )
    head.init_weights()
    feats = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4))

    losses = head.loss(feats, [_sample_with_gt()])
    for key in ('loss_teacher_cls', 'loss_teacher_quality',
                'loss_teacher_bbox', 'loss_teacher_angle',
                'loss_teacher_rank'):
        assert key in losses
        assert torch.isfinite(losses[key])

    with torch.no_grad():
        results = head.predict(feats, [_sample_with_gt()])

    assert len(results[0].scores) == 18
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_teacher_training_only'] is True
    assert head.e2e_debug['uses_inference_fallback'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False
    assert head.e2e_debug['uses_per_class_output_quota'] is False
    assert head.e2e_debug['outputs_fixed_query_set'] is True
