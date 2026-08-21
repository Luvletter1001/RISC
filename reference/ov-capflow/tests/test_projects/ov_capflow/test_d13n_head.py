import copy
import math
import types

import pytest
import torch
from mmdet.models.dense_heads import DETRHead
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules
from torch import nn

import projects.OVCapFlow.ov_capflow.ov_capflow_head as head_module
from projects.GroundingDINO.groundingdino.grounding_dino_head import (
    RotatedGroundingDINOHead, )
from projects.OVCapFlow.ov_capflow.existence_residual import ExistenceResidual
from projects.OVCapFlow.ov_capflow.ov_capflow_head import OVCapFlowHead


def _construct_head(monkeypatch, existence_loss_weight=None, **kwargs):
    def initialize_module(self, **unused):
        nn.Module.__init__(self)
        self.embed_dims = unused.get('embed_dims', 256)

    monkeypatch.setattr(
        RotatedGroundingDINOHead, '__init__', initialize_module)
    if (existence_loss_weight is not None and
            'matching_query_groups' not in kwargs):
        kwargs['matching_query_groups'] = 3
    return OVCapFlowHead(
        existence_loss_weight=existence_loss_weight, **kwargs)


def _target_tuple(masks):
    labels = [torch.zeros(mask.numel(), 1) for mask in masks]
    label_weights = [torch.ones(mask.numel()) for mask in masks]
    bbox_targets = [torch.zeros(mask.numel(), 5) for mask in masks]
    bbox_weights = [
        mask[:, None].expand(-1, 5).to(dtype=torch.float32)
        for mask in masks
    ]
    return (labels, label_weights, bbox_targets, bbox_weights,
            sum(int(mask.sum()) for mask in masks),
            sum(int((~mask).sum()) for mask in masks))


def _install_loss_stub(head, masks):
    def loss_by_feat(self, *args, **kwargs):
        self.last_matching_group_masks = masks.detach().clone()
        self.last_matching_mask = self.last_matching_group_masks[:, 0]
        return {'loss_parent': torch.zeros(())}

    head.loss_by_feat = types.MethodType(loss_by_feat, head)


def _install_forward_stub(head, cls_scores, bbox_preds):
    def forward(self, *args, **kwargs):
        return cls_scores, bbox_preds

    head.forward = types.MethodType(forward, head)


def _training_inputs(batch=1, groups=3, queries=600, dn_queries=2,
                     embed_dims=256):
    total = dn_queries + groups * queries
    hidden = torch.zeros(2, batch, total, embed_dims)
    hidden[-1, :, -groups * queries:, 0] = torch.linspace(
        -1.0, 1.0, groups * queries)
    hidden.requires_grad_()
    cls_scores = torch.zeros(1, batch, total, 1)
    bbox_preds = torch.zeros(1, batch, total, 5)
    masks = torch.zeros(batch, groups, queries, dtype=torch.bool)
    masks[..., 0] = True
    dn_meta = dict(
        num_denoising_queries=dn_queries,
        num_denoising_groups=1,
        num_matching_query_groups=groups,
        num_matching_queries_per_group=queries)
    return hidden, cls_scores, bbox_preds, masks, dn_meta


def _call_loss(head, hidden, dn_meta):
    samples = [
        types.SimpleNamespace(
            gt_instances=InstanceData(labels=torch.zeros(1, dtype=torch.long)),
            metainfo=dict(img_shape=(32, 32)))
        for _ in range(hidden.shape[1])
    ]
    return head.loss(
        hidden,
        references=None,
        memory_text=None,
        text_token_mask=torch.ones(hidden.shape[1], 1, dtype=torch.bool),
        enc_outputs_class=None,
        enc_outputs_coord=None,
        batch_data_samples=samples,
        dn_meta=dn_meta)


def test_none_preserves_legacy_state_and_float_arms_share_zero_rng_head(
        monkeypatch):
    legacy = _construct_head(monkeypatch, existence_loss_weight=None)
    assert not hasattr(legacy, 'existence_residual')
    assert not any('existence_residual' in key for key in legacy.state_dict())

    for coefficient in (0.0, 1.0):
        rng_before = torch.get_rng_state().clone()
        head = _construct_head(
            monkeypatch, existence_loss_weight=coefficient)
        assert torch.equal(rng_before, torch.get_rng_state())
        assert isinstance(head.existence_residual, ExistenceResidual)
        assert list(head.existence_residual.state_dict()) == ['weight', 'bias']
        assert sum(p.numel() for p in head.existence_residual.parameters()) == 257
        assert torch.count_nonzero(head.existence_residual.weight) == 0
        assert torch.count_nonzero(head.existence_residual.bias) == 0


def test_enabled_construction_freezes_groups_and_embed_dims(monkeypatch):
    head = _construct_head(
        monkeypatch,
        existence_loss_weight=1.0,
        matching_query_groups=3,
        embed_dims=256)
    assert head.matching_query_groups == 3
    assert head.embed_dims == 256
    assert head.existence_residual.in_features == 256

    for groups in (1, 2, 4):
        with pytest.raises(ValueError, match='matching_query_groups'):
            _construct_head(
                monkeypatch,
                existence_loss_weight=1.0,
                matching_query_groups=groups,
                embed_dims=256)
    for embed_dims in (128, 255, 512):
        with pytest.raises(ValueError, match='embed_dims'):
            _construct_head(
                monkeypatch,
                existence_loss_weight=1.0,
                matching_query_groups=3,
                embed_dims=embed_dims)

    legacy = _construct_head(
        monkeypatch,
        existence_loss_weight=None,
        matching_query_groups=1,
        embed_dims=128)
    assert legacy.matching_query_groups == 1
    assert legacy.embed_dims == 128
    assert not hasattr(legacy, 'existence_residual')


@pytest.mark.parametrize(
    'coefficient',
    [False, True, 0, 1, -0.0, 0.5, -1.0, float('nan'), float('inf'),
     '1.0'])
def test_existence_loss_weight_fails_closed(monkeypatch, coefficient):
    with pytest.raises((TypeError, ValueError), match='existence_loss_weight'):
        _construct_head(monkeypatch, existence_loss_weight=coefficient)


def test_training_uses_only_detached_dn_excluding_tail_and_derives_shape(
        monkeypatch):
    head = _construct_head(
        monkeypatch, existence_loss_weight=1.0, matching_query_groups=3)
    hidden, cls_scores, bbox_preds, masks, dn_meta = _training_inputs()
    _install_forward_stub(head, cls_scores, bbox_preds)
    _install_loss_stub(head, masks)
    seen = []
    hook = head.existence_residual.register_forward_hook(
        lambda module, inputs, output: seen.append(inputs[0]))

    losses = _call_loss(head, hidden, dn_meta)
    hook.remove()

    expected = hidden[-1, :, -1800:, :].reshape(1, 3, 600, 256)
    assert len(seen) == 1
    assert seen[0].shape == (1, 3, 600, 256)
    assert torch.equal(seen[0], expected)
    assert not seen[0].requires_grad
    assert seen[0].grad_fn is None
    assert losses['loss_existence'].requires_grad

    bad_meta = dict(dn_meta, num_matching_queries_per_group=599)
    with pytest.raises(ValueError, match='matching|query'):
        _call_loss(head, hidden, bad_meta)


@pytest.mark.parametrize(
    'groups, queries, embed_dims', [
        (3, 599, 256),
        (3, 601, 256),
        (2, 600, 256),
        (4, 600, 256),
        (3, 600, 128),
        (3, 600, 512),
    ])
def test_training_geometry_fails_closed(
        monkeypatch, groups, queries, embed_dims):
    head = _construct_head(
        monkeypatch, existence_loss_weight=1.0, matching_query_groups=3)
    hidden, cls_scores, bbox_preds, masks, dn_meta = _training_inputs(
        groups=groups, queries=queries, embed_dims=embed_dims)
    _install_forward_stub(head, cls_scores, bbox_preds)
    _install_loss_stub(head, masks)

    with pytest.raises(ValueError, match='D13-N|matching|embed'):
        _call_loss(head, hidden, dn_meta)


def test_actual_final_assignments_are_stacked_distinct_and_count_clamped(
        monkeypatch):
    head = _construct_head(
        monkeypatch, existence_loss_weight=1.0, matching_query_groups=3)
    query_count = 600
    batch_gt_instances = [
        InstanceData(labels=torch.zeros(count, dtype=torch.long))
        for count in (0, 7, 1223)
    ]
    batch_img_metas = [dict(img_shape=(32, 32))] * 3

    def parent_targets(self, cls_scores_list, bbox_preds_list,
                       gt_instances, img_metas):
        group = int(cls_scores_list[0][0, 0].item())
        masks = []
        for gt in gt_instances:
            count = min(len(gt), query_count)
            mask = torch.zeros(query_count, dtype=torch.bool)
            if count:
                start = 0 if count == query_count else group * 10
                mask[start:start + count] = True
            masks.append(mask)
        return _target_tuple(masks)

    def parent_matching_loss(self, cls_scores, bbox_preds,
                             gt_instances, img_metas, ignored=None):
        for layer in range(cls_scores.shape[0]):
            self.get_targets(
                [row for row in cls_scores[layer]],
                [row for row in bbox_preds[layer]], gt_instances, img_metas)
        return {'loss_cls': cls_scores.mean()}

    monkeypatch.setattr(
        RotatedGroundingDINOHead, 'get_targets', parent_targets)
    monkeypatch.setattr(DETRHead, 'loss_by_feat', parent_matching_loss)
    head._denoising_loss_dict = types.MethodType(
        lambda self, *args, **kwargs: {}, head)
    cls_scores = torch.zeros(2, 3, 3 * query_count, 1)
    for group in range(3):
        cls_scores[:, :, group * query_count:(group + 1) * query_count] = group
    bbox_preds = torch.zeros(2, 3, 3 * query_count, 5)

    head.loss_by_feat(
        cls_scores,
        bbox_preds,
        enc_cls_scores=None,
        enc_bbox_preds=None,
        batch_gt_instances=batch_gt_instances,
        batch_img_metas=batch_img_metas,
        dn_meta=dict(
            num_denoising_queries=0,
            num_denoising_groups=0,
            num_matching_query_groups=3,
            num_matching_queries_per_group=query_count))

    captured = head.last_matching_group_masks
    assert captured.shape == (3, 3, 600)
    assert captured.dtype == torch.bool
    assert torch.equal(captured.sum(-1), torch.tensor(
        [[0, 0, 0], [7, 7, 7], [600, 600, 600]]))
    assert not torch.equal(captured[1, 0], captured[1, 1])
    assert not torch.equal(captured[1, 1], captured[1, 2])
    assert torch.equal(head.last_matching_mask, captured[:, 0])


def test_real_hungarian_parent_path_captures_only_final_decoder_assignment():
    register_all_modules()
    cfg = Config.fromfile(
        'configs/ov_capflow/dotav2/'
        'ov_capflow_swin-t_dotav2_cleanstart_q600_base.py')
    head_cfg = copy.deepcopy(cfg.model.bbox_head)
    head_cfg.train_cfg = copy.deepcopy(cfg.model.train_cfg)
    head_cfg.matching_query_groups = 3
    head_cfg.existence_loss_weight = 1.0
    head = MODELS.build(head_cfg)
    head.text_masks = torch.ones(1, 2, dtype=torch.bool)
    head._loss_matching_group_masks = []

    scores = torch.full((2, 1, 600, 256), -10.0)
    scores[0, 0, 2, 0] = 10.0
    scores[0, 0, 3, 1] = 10.0
    scores[1, 0, 0, 0] = 10.0
    scores[1, 0, 1, 1] = 10.0
    first_boxes = torch.tensor(
        [.5, .5, .1, .1, 0.0]).repeat(600, 1)
    first_boxes[2] = torch.tensor([.2, .2, .1, .1, 0.0])
    first_boxes[3] = torch.tensor([.7, .7, .1, .1, 0.0])
    final_boxes = torch.tensor(
        [.5, .5, .1, .1, 0.0]).repeat(600, 1)
    final_boxes[0] = torch.tensor([.2, .2, .1, .1, 0.0])
    final_boxes[1] = torch.tensor([.7, .7, .1, .1, 0.0])
    boxes = torch.stack([first_boxes, final_boxes], dim=0).unsqueeze(1)
    gt = InstanceData()
    gt.bboxes = RotatedBoxes(torch.tensor([
        [20.0, 20.0, 10.0, 10.0, 0.0],
        [70.0, 70.0, 10.0, 10.0, 0.0],
    ]))
    gt.labels = torch.tensor([0, 1])
    gt.positive_maps = torch.zeros(2, 256)
    gt.positive_maps[0, 0] = 1.0
    gt.positive_maps[1, 1] = 1.0
    gt.text_token_mask = torch.zeros(2, 256, dtype=torch.bool)
    gt.text_token_mask[:, :2] = True

    head._matching_loss_by_feat(
        scores, boxes, [gt], [dict(img_shape=(100, 100))])

    assert len(head._loss_matching_group_masks) == 1
    assert torch.equal(
        head._loss_matching_group_masks[0],
        torch.cat([
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 598, dtype=torch.bool),
        ], dim=1))
    assert head._capture_matching_targets is False


def test_balanced_replay_cannot_overwrite_actual_capture(monkeypatch):
    head = _construct_head(
        monkeypatch,
        existence_loss_weight=1.0,
        matching_query_groups=3,
        balanced_cfg={'enabled': True})
    head.text_masks = torch.ones(1, 1, dtype=torch.bool)
    head.max_text_len = 1
    head.loss_cls = object()
    calls = []

    actual = torch.zeros(600, dtype=torch.bool)
    actual[0] = True
    replay = ~actual

    def parent_targets(self, *args, **kwargs):
        mask = actual if len(calls) % 2 == 0 else replay
        calls.append(mask.clone())
        return _target_tuple([mask])

    def parent_single(self, cls_scores, bbox_preds, gt_instances, img_metas):
        self.get_targets(
            [cls_scores[0]], [bbox_preds[0]], gt_instances, img_metas)
        zero = cls_scores.sum() * 0.0
        return zero, zero, zero

    def parent_matching(self, cls_scores, bbox_preds, gt_instances,
                        img_metas, ignored=None):
        self.loss_by_feat_single(
            cls_scores[-1], bbox_preds[-1], gt_instances, img_metas)
        return {'loss_cls': cls_scores.sum() * 0.0}

    monkeypatch.setattr(
        RotatedGroundingDINOHead, 'get_targets', parent_targets)
    monkeypatch.setattr(
        RotatedGroundingDINOHead, 'loss_by_feat_single', parent_single)
    monkeypatch.setattr(DETRHead, 'loss_by_feat', parent_matching)
    monkeypatch.setattr(
        head_module,
        'balanced_group_classification_loss',
        lambda *args, **kwargs: (
            kwargs['cls_scores'].sum() * 0.0, {}))
    head._loss_matching_group_masks = []

    head._matching_loss_by_feat(
        torch.zeros(1, 1, 600, 1), torch.zeros(1, 1, 600, 5),
        [InstanceData(labels=torch.zeros(1, dtype=torch.long))],
        [dict(img_shape=(32, 32))])

    assert len(calls) == 2
    assert torch.equal(head._loss_matching_group_masks[0][0], actual)
    assert not torch.equal(head._loss_matching_group_masks[0][0], replay)


def test_assignment_capture_is_disabled_in_finally(monkeypatch):
    head = _construct_head(
        monkeypatch, existence_loss_weight=1.0, matching_query_groups=3)

    def exploding_parent(self, *args, **kwargs):
        raise RuntimeError('parent failure')

    monkeypatch.setattr(DETRHead, 'loss_by_feat', exploding_parent)
    head._loss_matching_group_masks = []
    with pytest.raises(RuntimeError, match='parent failure'):
        head._matching_loss_by_feat(
            torch.zeros(1, 1, 600, 1),
            torch.zeros(1, 1, 600, 5), [], [])
    assert head._capture_matching_targets is False
    assert head._loss_matching_group_masks == []


def test_dn_prefix_cannot_change_training_logits_or_targets(monkeypatch):
    head = _construct_head(
        monkeypatch, existence_loss_weight=1.0, matching_query_groups=3)
    hidden, cls_scores, bbox_preds, masks, dn_meta = _training_inputs()
    _install_forward_stub(head, cls_scores, bbox_preds)
    _install_loss_stub(head, masks)
    first = _call_loss(head, hidden, dn_meta)
    logits_first = head.last_existence_logits.clone()
    masks_first = head.last_matching_group_masks.clone()

    changed = hidden.detach().clone()
    changed[-1, :, :2, :] = -987654.0
    changed.requires_grad_()
    second = _call_loss(head, changed, dn_meta)

    assert torch.equal(head.last_existence_logits, logits_first)
    assert torch.equal(head.last_matching_group_masks, masks_first)
    assert torch.equal(first['loss_existence'], second['loss_existence'])


def test_control_loss_stays_positive_zero_when_raw_loss_is_infinite(
        monkeypatch):
    def wrong_sign_loss(coefficient):
        head = _construct_head(
            monkeypatch,
            existence_loss_weight=coefficient,
            matching_query_groups=3)
        hidden, cls_scores, bbox_preds, masks, dn_meta = _training_inputs()
        masks.zero_()
        with torch.no_grad():
            head.existence_residual.weight.zero_()
            head.existence_residual.weight[0, 0] = 1.0
            head.existence_residual.bias.zero_()
            hidden[-1, :, -1800:, 0] = float('inf')
        _install_forward_stub(head, cls_scores, bbox_preds)
        _install_loss_stub(head, masks)
        loss = _call_loss(head, hidden, dn_meta)['loss_existence']
        assert torch.isinf(head.last_existence_logits).all()
        return head, loss

    control, control_loss = wrong_sign_loss(0.0)
    assert control_loss.requires_grad
    assert torch.isfinite(control_loss)
    assert torch.equal(control_loss, torch.zeros_like(control_loss))
    assert not torch.signbit(control_loss)
    control_loss.backward()
    for parameter in control.existence_residual.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.equal(parameter.grad, torch.zeros_like(parameter))

    _, candidate_loss = wrong_sign_loss(1.0)
    assert torch.isinf(candidate_loss)
    assert not torch.isnan(candidate_loss)


@pytest.mark.parametrize('coefficient', [0.0, 1.0])
def test_loss_gradient_isolated_to_head_and_control_is_exact_zero(
        monkeypatch, coefficient):
    head = _construct_head(
        monkeypatch,
        existence_loss_weight=coefficient,
        matching_query_groups=3)
    hidden, cls_scores, bbox_preds, masks, dn_meta = _training_inputs()
    masks.zero_()
    masks[:, 0, 0] = True
    _install_forward_stub(head, cls_scores, bbox_preds)
    _install_loss_stub(head, masks)

    loss = _call_loss(head, hidden, dn_meta)['loss_existence']
    loss.backward()

    assert hidden.grad is None
    assert head.existence_residual.weight.grad is not None
    assert head.existence_residual.bias.grad is not None
    if coefficient == 0.0:
        assert torch.equal(
            head.existence_residual.weight.grad,
            torch.zeros_like(head.existence_residual.weight))
        assert torch.equal(
            head.existence_residual.bias.grad,
            torch.zeros_like(head.existence_residual.bias))
    else:
        assert torch.count_nonzero(head.existence_residual.weight.grad) > 0


def test_training_telemetry_is_detached_and_complete(monkeypatch):
    head = _construct_head(
        monkeypatch, existence_loss_weight=1.0, matching_query_groups=3)
    hidden, cls_scores, bbox_preds, masks, dn_meta = _training_inputs()
    cls_scores.fill_(-100.0)
    _install_forward_stub(head, cls_scores, bbox_preds)
    _install_loss_stub(head, masks)
    _call_loss(head, hidden, dn_meta)

    tensors = [
        head.last_existence_logits,
        head.last_existence_residual,
        head.last_matching_group_masks,
        head.last_matching_group_matched_counts,
        head.last_existence_logit_quantiles,
        head.last_existence_residual_quantiles,
        head.last_existence_clamp_hit_counts,
    ]
    assert head.last_matching_group_matched_counts.shape == (1, 3)
    assert head.last_existence_logit_quantiles.shape == (3, 3)
    assert head.last_existence_residual_quantiles.shape == (3, 3)
    assert head.last_existence_clamp_hit_counts.shape == (2, )
    assert head.last_existence_clamp_hit_counts[0] > 0
    assert all(isinstance(value, torch.Tensor) for value in tensors)
    assert all(not value.requires_grad and value.grad_fn is None
               for value in tensors)


def test_inference_keeps_all_rows_and_changes_scores_only(monkeypatch):
    head = _construct_head(monkeypatch, existence_loss_weight=1.0)
    head.angle_factor = math.pi
    queries = 600
    cls_scores = torch.empty(1, queries, 2)
    cls_scores[..., 0] = torch.linspace(-3.0, 1.0, queries)
    cls_scores[..., 1] = torch.linspace(1.0, -3.0, queries)
    bbox_preds = torch.zeros(1, queries, 5)
    bbox_preds[..., 0] = torch.linspace(0.1, 0.9, queries)
    bbox_preds[..., 1:4] = 0.5
    _install_forward_stub(head, cls_scores.unsqueeze(0), bbox_preds.unsqueeze(0))
    hidden = torch.zeros(2, 1, queries + 3, 256)
    hidden[-1, 0, -queries:, 0] = torch.linspace(-2.0, 2.0, queries)
    sample = types.SimpleNamespace(
        metainfo=dict(
            img_shape=(100, 200), scale_factor=(1.0, 1.0)),
        token_positive_map={1: [0], 2: [1]})

    zero = head.predict(
        hidden, None, None, torch.ones(1, 2, dtype=torch.bool), [sample])
    legacy = head._predict_by_feat_single(
        cls_scores[0], bbox_preds[0], sample.token_positive_map,
        sample.metainfo)
    assert zero[0].scores.shape == (600, )
    assert torch.equal(zero[0].scores, legacy.scores)
    assert torch.equal(zero[0].labels, legacy.labels)
    assert torch.equal(zero[0].bboxes, legacy.bboxes)

    with torch.no_grad():
        head.existence_residual.weight[0, 0] = 1.0
    adapted = head.predict(
        hidden, None, None, torch.ones(1, 2, dtype=torch.bool), [sample])

    assert adapted[0].scores.shape == (600, )
    assert not torch.equal(adapted[0].scores, zero[0].scores)
    assert torch.equal(adapted[0].labels, zero[0].labels)
    assert torch.equal(adapted[0].bboxes, zero[0].bboxes)
    assert torch.equal(adapted[0].bboxes[:, 0], zero[0].bboxes[:, 0])
    assert torch.isfinite(head.last_existence_clamp_hit_counts).all()
    assert not head.last_existence_clamp_hit_counts.requires_grad


@pytest.mark.parametrize('queries', [599, 601])
def test_inference_rejects_non_q600_rows(monkeypatch, queries):
    head = _construct_head(monkeypatch, existence_loss_weight=1.0)
    cls_scores = torch.zeros(1, 1, queries, 2)
    bbox_preds = torch.zeros(1, 1, queries, 5)
    _install_forward_stub(head, cls_scores, bbox_preds)
    hidden = torch.zeros(2, 1, queries, 256)
    sample = types.SimpleNamespace(
        metainfo=dict(
            img_shape=(100, 200), scale_factor=(1.0, 1.0)),
        token_positive_map={1: [0], 2: [1]})

    with pytest.raises(ValueError, match='exactly 600 rows'):
        head.predict(
            hidden, None, None, torch.ones(1, 2, dtype=torch.bool),
            [sample])
