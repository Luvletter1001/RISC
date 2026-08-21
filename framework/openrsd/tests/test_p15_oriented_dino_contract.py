import torch
from mmdet.structures import DetDataSample
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmrotate.structures import RotatedBoxes

from M_AD.models.dense_heads.p15_oriented_dino_head import (
    P15OrientedDINOSetHead,
    P15BOrientedDINOSetHead,
    P15CGaussianQualityDINOSetHead,
    P15DLogitFusedGaussianQualityDINOSetHead,
    P15SupportQueryConditioner,
    build_orthogonal_support_tokens,
    gaussian_wasserstein_loss,
)
from M_AD.models.layers.transformer.dinor_layersv2 import CdnRQueryGenerator


def _sample_with_gt():
    sample = DetDataSample()
    sample.set_metainfo(
        dict(
            img_id='p15_fake',
            img_shape=(64, 64),
            ori_shape=(64, 64),
            scale_factor=(1.0, 1.0)))
    gt = InstanceData()
    gt.bboxes = RotatedBoxes(
        torch.tensor([[24.0, 28.0, 12.0, 8.0, 0.10]],
                     dtype=torch.float32))
    gt.labels = torch.tensor([1], dtype=torch.long)
    sample.gt_instances = gt
    sample.ignored_instances = InstanceData()
    sample.ignored_instances.bboxes = torch.empty(0, 5)
    sample.ignored_instances.labels = torch.empty(0, dtype=torch.long)
    return sample


def test_p15_support_tokens_are_class_orthogonal():
    support = build_orthogonal_support_tokens(
        num_classes=4, embed_dims=6, scale=2.0, tail_std=0.0)

    assert support.shape == (4, 6)
    assert torch.allclose(support[:, :4], torch.eye(4) * 2.0)
    assert torch.allclose(support[:, 4:], torch.zeros(4, 2))


def test_p15_query_conditioner_zero_scale_is_identity():
    conditioner = P15SupportQueryConditioner(
        num_classes=3,
        embed_dims=8,
        support_scale=0.0,
        support_init_std=0.0)
    query = torch.randn(2, 5, 8)
    logits = torch.randn(2, 5, 3)

    out = conditioner(query, logits)

    assert torch.allclose(out, query)
    assert conditioner.debug['support_source'] == 'learned_orthogonal'
    assert conditioner.debug['support_conditioned_query_initializer'] is True


def test_p15_query_conditioner_changes_queries_when_enabled():
    conditioner = P15SupportQueryConditioner(
        num_classes=3,
        embed_dims=8,
        support_scale=1.0,
        support_init_std=0.0)
    query = torch.zeros(2, 5, 8)
    logits = torch.zeros(2, 5, 3)
    logits[..., 1] = 6.0

    out = conditioner(query, logits)

    assert out.shape == query.shape
    assert not torch.allclose(out, query)
    assert conditioner.debug['support_conditioning_strength'] > 0.0


def test_p15_query_conditioner_gate_controls_delta_strength():
    conditioner = P15SupportQueryConditioner(
        num_classes=3,
        embed_dims=8,
        support_scale=1.0,
        support_init_std=0.0)
    query = torch.zeros(1, 2, 8)
    logits = torch.zeros(1, 2, 3)
    logits[..., 1] = 6.0
    query_gate = torch.tensor([[[0.0], [1.0]]])

    out = conditioner(query, logits, query_gate=query_gate)

    assert torch.allclose(out[:, 0], query[:, 0])
    assert out[:, 1].norm() > 0.0
    assert conditioner.debug['support_conditioning_strength'] > 0.0


def test_p15_head_predict_outputs_all_queries_without_nms_or_threshold():
    init_default_scope('mmrotate')
    head = P15OrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=7,
        support_query_scale=1.0,
        max_per_img=7)
    head.init_weights()
    hidden_states = torch.randn(1, 1, 7, 16)
    references = [torch.rand(1, 7, 4), torch.rand(1, 7, 4)]
    reference_angles = [
        torch.zeros(1, 7, 1),
        torch.zeros(1, 7, 1),
    ]
    samples = [_sample_with_gt()]

    with torch.no_grad():
        results = head.predict(
            hidden_states=hidden_states,
            references=references,
            reference_angles=reference_angles,
            batch_data_samples=samples,
            rescale=True)

    assert len(results) == 1
    assert results[0].bboxes.tensor.shape == (7, 5)
    assert results[0].scores.shape == (7,)
    assert results[0].labels.shape == (7,)
    assert head.e2e_debug['strict_e2e'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False
    assert head.e2e_debug['outputs_fixed_query_set'] is True


def test_p15_head_loss_is_finite_on_synthetic_rotated_gt():
    init_default_scope('mmrotate')
    head = P15OrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=7,
        support_query_scale=1.0,
        max_per_img=7)
    head.init_weights()
    hidden_states = torch.randn(1, 1, 7, 16)
    references = [torch.rand(1, 7, 4), torch.rand(1, 7, 4)]
    reference_angles = [
        torch.zeros(1, 7, 1),
        torch.zeros(1, 7, 1),
    ]
    enc_outputs_class = torch.randn(1, 7, 3)
    enc_outputs_coord = torch.rand(1, 7, 4)
    enc_outputs_angle = torch.zeros(1, 7, 1)

    losses = head.loss(
        hidden_states=hidden_states,
        references=references,
        reference_angles=reference_angles,
        enc_outputs_class=enc_outputs_class,
        enc_outputs_coord=enc_outputs_coord,
        enc_outputs_angle=enc_outputs_angle,
        batch_data_samples=[_sample_with_gt()],
        dn_meta=None)

    for key in ('loss_cls', 'loss_bbox', 'loss_angle', 'enc_loss_cls',
                'enc_loss_bbox', 'enc_loss_angle'):
        assert key in losses
        assert torch.isfinite(losses[key])


def test_p15b_head_uses_background_softmax_and_objectness():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=7,
        support_query_scale=1.0,
        max_per_img=7)
    head.init_weights()

    assert head.cls_branches[0].out_features == 4
    assert head.obj_branches[0].out_features == 1
    assert head.background_class_index == 3
    assert head.e2e_debug['uses_background_softmax'] is True
    assert head.e2e_debug['uses_query_objectness_posterior'] is True


def test_p15i_condition_matching_queries_can_gate_support_by_objectness():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2,
        support_objectness_gate_power=1.0)
    query = torch.zeros(1, 2, 16)
    class_logits = torch.tensor([[[0.0, 5.0, -2.0, -3.0],
                                  [0.0, 5.0, -2.0, -3.0]]])
    objectness_logits = torch.tensor([[[-8.0], [8.0]]])

    out = head.condition_matching_queries(
        query, class_logits, objectness_logits)

    assert out[:, 1].norm() > out[:, 0].norm()
    assert head.e2e_debug['uses_objectness_gated_support_conditioning'] is True
    assert head.e2e_debug['final_score_formula'] == (
        'objectness_sigmoid_x_foreground_softmax')


def test_p15j_support_gate_warmup_starts_identity_and_ends_with_floor():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2,
        support_objectness_gate_power=0.5,
        support_objectness_gate_warmup_iters=10,
        support_objectness_gate_floor_start=0.5,
        support_objectness_gate_floor_end=0.2)
    objectness_logits = torch.tensor([[[-8.0], [8.0]]])

    start_gate = head._support_query_gate(objectness_logits)
    head._support_gate_step.fill_(10)
    end_gate = head._support_query_gate(objectness_logits)

    prior = objectness_logits.sigmoid().squeeze(-1)
    expected_end = 0.2 + 0.8 * prior.pow(0.5)
    assert torch.allclose(start_gate, torch.ones_like(start_gate))
    assert torch.allclose(end_gate, expected_end)
    assert head.e2e_debug['support_objectness_gate_warmup_iters'] == 10
    assert head.e2e_debug['support_objectness_gate_floor_end'] == 0.2


def test_p15k_dn_query_conditioning_defaults_to_identity():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=5,
        support_query_scale=1.0)
    query = torch.randn(1, 4, 16)
    dn_meta = dict(num_denoising_queries=4, num_denoising_groups=2)

    out = head.condition_denoising_queries(query, [_sample_with_gt()], dn_meta)

    assert torch.allclose(out, query)
    assert head.e2e_debug['uses_support_conditioned_dn_queries'] is False


def test_p15k_dn_query_conditioning_only_changes_positive_slots():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=5,
        support_query_scale=1.0,
        support_dn_query_scale=1.0,
        support_dn_query_logit=6.0)
    query = torch.zeros(1, 4, 16)
    dn_meta = dict(num_denoising_queries=4, num_denoising_groups=2)

    out = head.condition_denoising_queries(query, [_sample_with_gt()], dn_meta)

    assert out.shape == query.shape
    assert out[0, 0].norm() > 0.0
    assert torch.allclose(out[0, 1], query[0, 1])
    assert out[0, 2].norm() > 0.0
    assert torch.allclose(out[0, 3], query[0, 3])
    assert head.e2e_debug['uses_support_conditioned_dn_queries'] is True
    assert head.e2e_debug['support_dn_positive_slots'] == 2


def test_p15b_head_predict_scores_are_objectness_times_foreground_softmax():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2)
    cls_score = torch.tensor([[[0.0, 2.0, -1.0, 4.0],
                               [3.0, -1.0, 0.5, -2.0]]])
    obj_logit = torch.tensor([[[1.0], [-0.5]]])
    bbox_pred = torch.tensor([[[0.5, 0.5, 0.25, 0.25],
                               [0.25, 0.25, 0.20, 0.20]]])
    angle_pred = torch.zeros(1, 2, 1)

    result = head._predict_single(
        cls_score[0],
        obj_logit[0],
        bbox_pred[0],
        angle_pred[0],
        _sample_with_gt().metainfo,
        rescale=True)

    fg_probs = cls_score.softmax(dim=-1)[..., :3]
    expected_scores, expected_labels = fg_probs.max(dim=-1)
    expected_scores = expected_scores * obj_logit.sigmoid().squeeze(-1)
    assert torch.allclose(result.scores, expected_scores[0])
    assert torch.equal(result.labels, expected_labels[0])
    assert result.bboxes.tensor.shape == (2, 5)


def test_p15b_topk_scores_default_matches_objectness_times_foreground_softmax():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2)
    cls_score = torch.tensor([[[0.0, 2.0, -1.0, 4.0],
                               [3.0, -1.0, 0.5, -2.0]]])
    obj_logit = torch.tensor([[[1.0], [-0.5]]])

    scores = head.select_topk_scores(cls_score, obj_logit)

    fg_probs = cls_score.softmax(dim=-1)[..., :3]
    expected = fg_probs.max(dim=-1)[0] * obj_logit.sigmoid().squeeze(-1)
    assert torch.allclose(scores, expected)


def test_p15h_tempered_topk_scores_use_powered_objectness_prior():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2,
        topk_objectness_power=0.5,
        objectness_prior_floor=0.05)
    cls_score = torch.tensor([[[0.0, 2.0, -1.0, 4.0],
                               [3.0, -1.0, 0.5, -2.0]]])
    obj_logit = torch.tensor([[[-8.0], [2.0]]])

    scores = head.select_topk_scores(cls_score, obj_logit)

    fg_probs = cls_score.softmax(dim=-1)[..., :3]
    obj_prior = obj_logit.sigmoid().squeeze(-1).clamp(min=0.05).pow(0.5)
    expected = fg_probs.max(dim=-1)[0] * obj_prior
    assert torch.allclose(scores, expected)
    assert head.e2e_debug['final_score_formula'] == (
        'objectness_sigmoid_x_foreground_softmax')


def test_p15h_matching_power_zero_ignores_objectness_for_assignment():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2,
        matching_objectness_power=0.0)
    sample = _sample_with_gt()
    cls_score = torch.tensor([[0.0, 4.0, -2.0, -3.0],
                              [0.0, 3.0, -2.0, -3.0]])
    bbox_pred = torch.tensor([[24.0 / 64.0, 28.0 / 64.0, 12.0 / 64.0,
                               8.0 / 64.0],
                              [24.0 / 64.0, 28.0 / 64.0, 12.0 / 64.0,
                               8.0 / 64.0]])
    angle_pred = torch.tensor([[0.10], [0.10]])
    obj_logit = torch.tensor([[-8.0], [8.0]])

    assigned_gt, assigned_labels = head._assign_single(
        cls_score, bbox_pred, angle_pred, sample.gt_instances,
        sample.metainfo, obj_logit)

    assert assigned_gt.tolist() == [0, -1]
    assert assigned_labels.tolist() == [1, -1]


def test_p15l_class_balanced_query_selection_defaults_to_global_topk():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=3,
        support_query_scale=1.0)
    class_logits = torch.zeros(1, 5, 4)
    class_logits[0, 0, 0] = 8.0
    class_logits[0, 1, 1] = 7.0
    class_logits[0, 2, 0] = 6.0
    class_logits[0, 3, 0] = 5.0
    class_logits[0, 4, 2] = 2.0
    objectness_logits = torch.ones(1, 5, 1)

    indices = head.select_query_indices(
        class_logits, objectness_logits, num_queries=3)
    global_scores = head.select_topk_scores(class_logits, objectness_logits)

    assert indices.tolist() == torch.topk(global_scores, k=3, dim=1)[1].tolist()
    assert head.e2e_debug['uses_train_class_balanced_query_selection'] is False


def test_p15l_class_balanced_query_selection_reserves_weak_classes():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=3,
        support_query_scale=1.0,
        train_class_balanced_topk_per_class=1)
    class_logits = torch.zeros(1, 5, 4)
    class_logits[0, 0, 0] = 8.0
    class_logits[0, 1, 1] = 7.0
    class_logits[0, 2, 0] = 6.0
    class_logits[0, 3, 0] = 5.0
    class_logits[0, 4, 2] = 2.0
    objectness_logits = torch.ones(1, 5, 1)

    indices = head.select_query_indices(
        class_logits, objectness_logits, num_queries=3)

    assert indices.tolist() == [[0, 1, 4]]
    assert head.e2e_debug['uses_train_class_balanced_query_selection'] is True
    assert head.e2e_debug['train_class_balanced_reserved_queries'] == 3


def test_p15l_e_aux_one2many_defaults_to_disabled():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=3,
        support_query_scale=1.0)

    assert head.aux_one2many_topk == 0
    assert head.aux_one2many_loss_weight == 0.0
    assert head.e2e_debug['uses_aux_one2many_primary_supervision'] is False


def test_p15l_e_aux_one2many_warmup_scales_loss_weight():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=4,
        support_query_scale=1.0,
        aux_one2many_topk=1,
        aux_one2many_loss_weight=0.10,
        aux_one2many_warmup_iters=10)

    assert torch.isclose(
        head._aux_one2many_loss_scale(),
        torch.tensor(0.0))
    head._aux_one2many_step.fill_(10)
    assert torch.isclose(
        head._aux_one2many_loss_scale(),
        torch.tensor(0.10))
    assert head.e2e_debug['uses_aux_one2many_primary_supervision'] is True


def test_p15l_e_aux_one2many_selects_unmatched_low_cost_queries():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=4,
        support_query_scale=1.0,
        aux_one2many_topk=1,
        aux_one2many_loss_weight=0.10)
    total_cost = torch.tensor([[0.0, 9.0],
                               [1.0, 4.0],
                               [2.0, 0.5],
                               [3.0, 0.2]])
    assigned_gt = torch.tensor([0, -1, -1, -1])

    query_indices, gt_indices = head._select_aux_one2many_pairs(
        total_cost, assigned_gt)

    assert query_indices.tolist() == [1, 3]
    assert gt_indices.tolist() == [0, 1]


def test_p15l_e_aux_one2many_keeps_predict_path_strict_e2e():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2,
        aux_one2many_topk=1,
        aux_one2many_loss_weight=0.10)
    cls_score = torch.tensor([[[0.0, 2.0, -1.0, 4.0],
                               [3.0, -1.0, 0.5, -2.0]]])
    obj_logit = torch.tensor([[[1.0], [-0.5]]])
    bbox_pred = torch.tensor([[[0.5, 0.5, 0.25, 0.25],
                               [0.25, 0.25, 0.20, 0.20]]])
    angle_pred = torch.zeros(1, 2, 1)

    result = head._predict_single(
        cls_score[0],
        obj_logit[0],
        bbox_pred[0],
        angle_pred[0],
        _sample_with_gt().metainfo,
        rescale=True)

    fg_probs = cls_score.softmax(dim=-1)[..., :3]
    expected_scores, expected_labels = fg_probs.max(dim=-1)
    expected_scores = expected_scores * obj_logit.sigmoid().squeeze(-1)
    assert torch.allclose(result.scores, expected_scores[0])
    assert torch.equal(result.labels, expected_labels[0])
    assert result.bboxes.tensor.shape == (2, 5)
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False
    assert head.e2e_debug['final_score_formula'] == (
        'objectness_sigmoid_x_foreground_softmax')


def test_p16k_duplicate_rank_defaults_to_disabled():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=3,
        support_query_scale=1.0)

    assert head.duplicate_rank_topk == 0
    assert head.duplicate_rank_loss_weight == 0.0
    assert head.e2e_debug['uses_duplicate_rank_suppression'] is False


def test_p16k_duplicate_rank_warmup_scales_loss_weight():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=4,
        support_query_scale=1.0,
        duplicate_rank_topk=1,
        duplicate_rank_loss_weight=0.10,
        duplicate_rank_warmup_iters=10)

    assert torch.isclose(
        head._duplicate_rank_loss_scale(),
        torch.tensor(0.0))
    head._duplicate_rank_step.fill_(10)
    assert torch.isclose(
        head._duplicate_rank_loss_scale(),
        torch.tensor(0.10))
    assert head.e2e_debug['uses_duplicate_rank_suppression'] is True


def test_p16k_duplicate_rank_selects_near_unmatched_duplicates():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=4,
        support_query_scale=1.0,
        duplicate_rank_topk=1,
        duplicate_rank_loss_weight=0.10,
        duplicate_rank_max_cost=1.0)
    total_cost = torch.tensor([[0.0, 9.0],
                               [0.3, 4.0],
                               [2.2, 0.4],
                               [6.0, 0.1]])
    assigned_gt = torch.tensor([0, -1, -1, 1])

    pos_query, dup_query, dup_gt = head._select_duplicate_rank_pairs(
        total_cost, assigned_gt)

    assert pos_query.tolist() == [0, 3]
    assert dup_query.tolist() == [1, 2]
    assert dup_gt.tolist() == [0, 1]


def test_p16k_duplicate_rank_loss_is_training_only_and_finite():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=3,
        support_query_scale=1.0,
        max_per_img=3,
        duplicate_rank_topk=1,
        duplicate_rank_loss_weight=0.10,
        duplicate_rank_margin=0.15,
        duplicate_rank_max_cost=2.0)
    sample = _sample_with_gt()
    cls_score = torch.tensor([[[0.0, 5.0, -2.0, -4.0],
                               [0.0, 4.5, -2.0, -4.0],
                               [3.0, -2.0, 0.0, -3.0]]])
    obj_logit = torch.tensor([[[4.0], [3.0], [-3.0]]])
    bbox_pred = torch.tensor([[[24.0 / 64.0, 28.0 / 64.0,
                                12.0 / 64.0, 8.0 / 64.0],
                               [25.0 / 64.0, 29.0 / 64.0,
                                12.0 / 64.0, 8.0 / 64.0],
                               [0.75, 0.75, 0.10, 0.10]]])
    angle_pred = torch.tensor([[[0.10], [0.10], [0.0]]])

    losses = head._loss_single(
        cls_score, obj_logit, bbox_pred, angle_pred, [sample])
    result = head._predict_single(
        cls_score[0], obj_logit[0], bbox_pred[0], angle_pred[0],
        sample.metainfo, rescale=True)

    assert 'loss_duplicate_rank' in losses
    assert torch.isfinite(losses['loss_duplicate_rank'])
    assert head.e2e_debug['duplicate_rank_selected_pairs'] == 1
    assert result.bboxes.tensor.shape == (3, 5)
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False


def test_p16o_objectness_budget_defaults_to_disabled():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=3,
        support_query_scale=1.0)

    assert head.matching_duplicate_neg_topk == 0
    assert head.matching_duplicate_neg_loss_weight == 0.0
    assert head.cardinality_loss_weight == 0.0
    assert (
        head.e2e_debug['uses_matching_duplicate_objectness_suppression']
        is False)
    assert head.e2e_debug['uses_objectness_cardinality_budget'] is False


def test_p16t_matching_duplicate_neg_loss_scale_decays_after_warmup():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=4,
        support_query_scale=1.0,
        matching_duplicate_neg_topk=1,
        matching_duplicate_neg_loss_weight=0.20,
        matching_duplicate_neg_warmup_iters=10,
        matching_duplicate_neg_decay_start_iters=20,
        matching_duplicate_neg_decay_end_iters=30,
        matching_duplicate_neg_decay_final_mult=0.5)

    head._matching_duplicate_neg_step.fill_(0)
    assert torch.isclose(
        head._matching_duplicate_neg_loss_scale(),
        torch.tensor(0.0))

    head._matching_duplicate_neg_step.fill_(10)
    assert torch.isclose(
        head._matching_duplicate_neg_loss_scale(),
        torch.tensor(0.20))

    head._matching_duplicate_neg_step.fill_(25)
    assert torch.isclose(
        head._matching_duplicate_neg_loss_scale(),
        torch.tensor(0.15))

    head._matching_duplicate_neg_step.fill_(40)
    assert torch.isclose(
        head._matching_duplicate_neg_loss_scale(),
        torch.tensor(0.10))


def test_p16o_matching_duplicate_neg_selects_low_cost_unmatched_queries():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=4,
        support_query_scale=1.0,
        matching_duplicate_neg_topk=1,
        matching_duplicate_neg_loss_weight=0.10,
        matching_duplicate_neg_max_cost=1.0)
    total_cost = torch.tensor([[0.0, 9.0],
                               [0.3, 4.0],
                               [2.2, 0.4],
                               [6.0, 0.1]])
    assigned_gt = torch.tensor([0, -1, -1, 1])

    duplicate_neg = head._select_matching_duplicate_negatives(
        total_cost, assigned_gt)

    assert duplicate_neg.tolist() == [1, 2]


def test_p16o_cardinality_and_matching_duplicate_losses_are_finite():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=3,
        support_query_scale=1.0,
        max_per_img=3,
        matching_duplicate_neg_topk=1,
        matching_duplicate_neg_loss_weight=0.20,
        matching_duplicate_neg_max_cost=2.0,
        cardinality_loss_weight=0.05)
    sample = _sample_with_gt()
    cls_score = torch.tensor([[[0.0, 5.0, -2.0, -4.0],
                               [0.0, 4.5, -2.0, -4.0],
                               [3.0, -2.0, 0.0, -3.0]]])
    obj_logit = torch.tensor([[[4.0], [3.0], [-3.0]]])
    bbox_pred = torch.tensor([[[24.0 / 64.0, 28.0 / 64.0,
                                12.0 / 64.0, 8.0 / 64.0],
                               [25.0 / 64.0, 29.0 / 64.0,
                                12.0 / 64.0, 8.0 / 64.0],
                               [0.75, 0.75, 0.10, 0.10]]])
    angle_pred = torch.tensor([[[0.10], [0.10], [0.0]]])

    losses = head._loss_single(
        cls_score, obj_logit, bbox_pred, angle_pred, [sample])
    result = head._predict_single(
        cls_score[0], obj_logit[0], bbox_pred[0], angle_pred[0],
        sample.metainfo, rescale=True)

    assert 'loss_matching_dup_obj_neg' in losses
    assert 'loss_cardinality_obj' in losses
    assert torch.isfinite(losses['loss_matching_dup_obj_neg'])
    assert torch.isfinite(losses['loss_cardinality_obj'])
    assert head.e2e_debug['matching_duplicate_neg_selected_queries'] == 1
    assert result.bboxes.tensor.shape == (3, 5)
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_dense_detection_head'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False


def test_p15b_head_loss_splits_denoising_outputs_when_dn_meta_present():
    init_default_scope('mmrotate')
    head = P15BOrientedDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=7,
        support_query_scale=1.0,
        max_per_img=7)
    head.init_weights()
    total_queries = 9
    hidden_states = torch.randn(1, 1, total_queries, 16)
    references = [torch.rand(1, total_queries, 4), torch.rand(1, total_queries, 4)]
    reference_angles = [
        torch.zeros(1, total_queries, 1),
        torch.zeros(1, total_queries, 1),
    ]
    enc_outputs_class = torch.randn(1, 7, 4)
    enc_outputs_objectness = torch.randn(1, 7, 1)
    enc_outputs_coord = torch.rand(1, 7, 4)
    enc_outputs_angle = torch.zeros(1, 7, 1)

    losses = head.loss(
        hidden_states=hidden_states,
        references=references,
        reference_angles=reference_angles,
        enc_outputs_class=enc_outputs_class,
        enc_outputs_coord=enc_outputs_coord,
        enc_outputs_angle=enc_outputs_angle,
        batch_data_samples=[_sample_with_gt()],
        dn_meta=dict(num_denoising_queries=2, num_denoising_groups=1),
        enc_outputs_objectness=enc_outputs_objectness)

    for key in ('loss_cls', 'loss_obj', 'loss_bbox', 'enc_loss_obj',
                'dn_loss_cls', 'dn_loss_obj', 'dn_loss_bbox',
                'dn_loss_angle', 'dn_loss_gwd'):
        assert key in losses
        assert torch.isfinite(losses[key])


def test_p15c_head_predict_scores_include_gaussian_quality_posterior():
    init_default_scope('mmrotate')
    head = P15CGaussianQualityDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2)
    cls_score = torch.tensor([[[0.0, 2.0, -1.0, 4.0],
                               [3.0, -1.0, 0.5, -2.0]]])
    obj_logit = torch.tensor([[[1.0], [-0.5]]])
    quality_logit = torch.tensor([[[2.0], [-2.0]]])
    bbox_pred = torch.tensor([[[0.5, 0.5, 0.25, 0.25],
                               [0.25, 0.25, 0.20, 0.20]]])
    angle_pred = torch.zeros(1, 2, 1)

    result = head._predict_single(
        cls_score[0],
        obj_logit[0],
        quality_logit[0],
        bbox_pred[0],
        angle_pred[0],
        _sample_with_gt().metainfo,
        rescale=True)

    fg_probs = cls_score.softmax(dim=-1)[..., :3]
    expected_scores, expected_labels = fg_probs.max(dim=-1)
    expected_scores = (
        expected_scores * obj_logit.sigmoid().squeeze(-1) *
        quality_logit.sigmoid().squeeze(-1))
    assert torch.allclose(result.scores, expected_scores[0])
    assert torch.equal(result.labels, expected_labels[0])
    assert result.bboxes.tensor.shape == (2, 5)
    assert head.e2e_debug['uses_gaussian_quality_posterior'] is True
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False


def test_p15d_head_predict_scores_fuse_quality_inside_objectness_logit():
    init_default_scope('mmrotate')
    head = P15DLogitFusedGaussianQualityDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2,
        quality_logit_alpha=0.5)
    cls_score = torch.tensor([[[0.0, 2.0, -1.0, 4.0],
                               [3.0, -1.0, 0.5, -2.0]]])
    obj_logit = torch.tensor([[[1.0], [-0.5]]])
    quality_logit = torch.tensor([[[2.0], [-2.0]]])
    bbox_pred = torch.tensor([[[0.5, 0.5, 0.25, 0.25],
                               [0.25, 0.25, 0.20, 0.20]]])
    angle_pred = torch.zeros(1, 2, 1)

    result = head._predict_single(
        cls_score[0],
        obj_logit[0],
        quality_logit[0],
        bbox_pred[0],
        angle_pred[0],
        _sample_with_gt().metainfo,
        rescale=True)

    fg_probs = cls_score.softmax(dim=-1)[..., :3]
    expected_scores, expected_labels = fg_probs.max(dim=-1)
    fused_logit = obj_logit.squeeze(-1) + 0.5 * quality_logit.squeeze(-1)
    expected_scores = expected_scores * fused_logit.sigmoid()
    assert torch.allclose(result.scores, expected_scores[0])
    assert torch.equal(result.labels, expected_labels[0])
    assert result.bboxes.tensor.shape == (2, 5)
    assert head.e2e_debug['uses_nms'] is False
    assert head.e2e_debug['uses_score_threshold_postprocess'] is False
    assert head.e2e_debug['final_score_formula'] == (
        'foreground_softmax_x_sigmoid_objectness_plus_alpha_quality')


def test_p15d_topk_scores_use_same_logit_fused_posterior():
    init_default_scope('mmrotate')
    head = P15DLogitFusedGaussianQualityDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=2,
        support_query_scale=1.0,
        max_per_img=2,
        quality_logit_alpha=0.25)
    cls_score = torch.tensor([[[0.0, 2.0, -1.0, 4.0],
                               [3.0, -1.0, 0.5, -2.0]]])
    obj_logit = torch.tensor([[[1.0], [-0.5]]])
    quality_logit = torch.tensor([[[2.0], [-2.0]]])

    scores = head.select_topk_scores(cls_score, obj_logit, quality_logit)

    fg_probs = cls_score.softmax(dim=-1)[..., :3]
    expected = fg_probs.max(dim=-1)[0]
    fused_logit = obj_logit.squeeze(-1) + 0.25 * quality_logit.squeeze(-1)
    expected = expected * fused_logit.sigmoid()
    assert torch.allclose(scores, expected)


def test_p15c_head_loss_returns_gaussian_quality_terms():
    init_default_scope('mmrotate')
    head = P15CGaussianQualityDINOSetHead(
        num_classes=3,
        embed_dims=16,
        num_reg_fcs=2,
        num_pred_layer=2,
        num_queries=7,
        support_query_scale=1.0,
        max_per_img=7)
    head.init_weights()
    total_queries = 9
    hidden_states = torch.randn(1, 1, total_queries, 16)
    references = [torch.rand(1, total_queries, 4), torch.rand(1, total_queries, 4)]
    reference_angles = [
        torch.zeros(1, total_queries, 1),
        torch.zeros(1, total_queries, 1),
    ]
    enc_outputs_class = torch.randn(1, 7, 4)
    enc_outputs_objectness = torch.randn(1, 7, 1)
    enc_outputs_quality = torch.randn(1, 7, 1)
    enc_outputs_coord = torch.rand(1, 7, 4)
    enc_outputs_angle = torch.zeros(1, 7, 1)

    losses = head.loss(
        hidden_states=hidden_states,
        references=references,
        reference_angles=reference_angles,
        enc_outputs_class=enc_outputs_class,
        enc_outputs_coord=enc_outputs_coord,
        enc_outputs_angle=enc_outputs_angle,
        batch_data_samples=[_sample_with_gt()],
        dn_meta=dict(num_denoising_queries=2, num_denoising_groups=1),
        enc_outputs_objectness=enc_outputs_objectness,
        enc_outputs_quality=enc_outputs_quality)

    for key in ('loss_quality', 'enc_loss_quality', 'dn_loss_quality',
                'loss_cls', 'loss_obj', 'loss_bbox', 'loss_gwd'):
        assert key in losses
        assert torch.isfinite(losses[key])


def test_p15_gaussian_wasserstein_loss_supports_pairwise_cost_matrix():
    pred_boxes = torch.tensor(
        [
            [16.0, 20.0, 8.0, 6.0, 0.0],
            [30.0, 32.0, 10.0, 4.0, 0.2],
            [48.0, 44.0, 12.0, 5.0, -0.3],
        ],
        dtype=torch.float32)
    gt_boxes = torch.tensor(
        [
            [18.0, 22.0, 8.0, 6.0, 0.1],
            [46.0, 42.0, 12.0, 5.0, -0.2],
        ],
        dtype=torch.float32)

    cost = gaussian_wasserstein_loss(pred_boxes[:, None, :],
                                     gt_boxes[None, :, :])

    assert cost.shape == (3, 2)
    assert torch.isfinite(cost).all()


def test_p15_cdnr_query_generator_accepts_rotated_boxes_wrapper():
    generator = CdnRQueryGenerator(
        num_classes=3,
        embed_dims=8,
        num_matching_queries=5,
        label_noise_scale=0.5,
        box_noise_scale=0.4,
        angle_noise_scale=0.2,
        group_cfg=dict(dynamic=False, num_groups=1))

    dn_label, dn_bbox, dn_angle, attn_mask, dn_meta = generator(
        [_sample_with_gt()])

    assert dn_label.shape[-1] == 8
    assert dn_bbox.shape[-1] == 4
    assert dn_angle.shape[-1] == 1
    assert attn_mask.shape[0] == dn_meta['num_denoising_queries'] + 5


def test_p15c_detector_pre_decoder_routes_encoder_quality_logits():
    cfg = Config.fromfile(
        'M_configs/Diagnostics/'
        'hrrsd_p15c_gaussian_quality_overfit100_4gpu_b8_v5.py')
    init_default_scope(cfg.default_scope)
    model = MODELS.build(cfg.model)
    model.train()
    memory = torch.randn(1, 400, model.embed_dims)
    memory_mask = torch.zeros(1, 400, dtype=torch.bool)
    spatial_shapes = torch.tensor([[20, 20]], dtype=torch.long)

    _, head_inputs = model.pre_decoder(
        memory=memory,
        memory_mask=memory_mask,
        spatial_shapes=spatial_shapes,
        batch_data_samples=[_sample_with_gt()])

    assert 'enc_outputs_quality' in head_inputs
    assert head_inputs['enc_outputs_quality'].shape == (
        1, model.num_queries, 1)


def test_p15n_memory_adapter_detector_bypasses_encoder_attention():
    cfg = Config.fromfile(
        'M_configs/Diagnostics/'
        'hrrsd_p15n_l_memory_adapter_pos_topk1_overfit100_4gpu_b8_v5.py')
    init_default_scope(cfg.default_scope)
    model = MODELS.build(cfg.model)
    feat = torch.randn(1, 32, model.embed_dims)
    feat_pos = torch.randn_like(feat)
    feat_mask = torch.zeros(1, 32, dtype=torch.bool)
    spatial_shapes = torch.tensor([[4, 4], [4, 4]], dtype=torch.long)
    level_start_index = torch.tensor([0, 16], dtype=torch.long)
    valid_ratios = torch.ones(1, 2, 2)

    outputs = model.forward_encoder(
        feat=feat,
        feat_mask=feat_mask,
        feat_pos=feat_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios)

    assert outputs['memory'].shape == feat.shape
    assert outputs['memory_mask'] is feat_mask
    assert outputs['spatial_shapes'] is spatial_shapes
    assert model.encoder_attitude_debug['strict_e2e'] is True
    assert model.encoder_attitude_debug['bypasses_deformable_encoder'] is True
    assert model.encoder_attitude_debug[
        'uses_tokenwise_memory_adapter'] is True
    assert not torch.allclose(outputs['memory'], feat + feat_pos)
