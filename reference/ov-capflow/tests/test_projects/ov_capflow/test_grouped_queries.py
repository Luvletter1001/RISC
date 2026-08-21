import pytest
import torch


def _api():
    from projects.OVCapFlow.ov_capflow.grouped_queries import (
        average_matching_loss_dicts,
        expand_dn_attention_mask,
        grouped_matching_losses,
        repeat_matching_queries,
        split_matching_groups,
    )
    return {
        'average': average_matching_loss_dicts,
        'expand_mask': expand_dn_attention_mask,
        'grouped_losses': grouped_matching_losses,
        'repeat': repeat_matching_queries,
        'split': split_matching_groups,
    }


def test_one_group_is_identity_and_gradient_exact():
    api = _api()
    query = torch.randn(2, 4, 3, requires_grad=True)
    references = torch.rand(2, 4, 5, requires_grad=True)
    repeated_query, repeated_references = api['repeat'](
        query, references, groups=1)
    assert repeated_query is query
    assert repeated_references is references

    loss = repeated_query.square().sum() + repeated_references.square().sum()
    loss.backward()
    assert torch.equal(query.grad, 2 * query.detach())
    assert torch.equal(references.grad, 2 * references.detach())


def test_three_groups_repeat_shared_queries_and_accumulate_gradients():
    api = _api()
    query = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    references = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    query.requires_grad_()
    references.requires_grad_()

    repeated_query, repeated_references = api['repeat'](
        query, references, groups=3)

    assert repeated_query.shape == (1, 12, 3)
    assert repeated_references.shape == (1, 12, 5)
    assert torch.equal(repeated_query, torch.cat([query, query, query], dim=1))
    assert torch.equal(
        repeated_references,
        torch.cat([references, references, references], dim=1))
    (repeated_query.sum() + repeated_references.sum()).backward()
    assert torch.equal(query.grad, torch.full_like(query, 3.0))
    assert torch.equal(references.grad, torch.full_like(references, 3.0))


def test_expanded_dn_mask_copies_dn_rules_and_isolates_matching_groups():
    api = _api()
    num_dn = 2
    queries = 3
    original = torch.zeros(num_dn + queries, num_dn + queries,
                           dtype=torch.bool)
    original[:num_dn, :num_dn] = torch.tensor(
        [[False, True], [True, False]])
    original[:num_dn, num_dn:] = torch.tensor(
        [[True, False, True], [False, True, False]])
    original[num_dn:, :num_dn] = torch.tensor(
        [[True, False], [False, True], [True, True]])
    original[num_dn:, num_dn:] = torch.tensor(
        [[False, True, False], [True, False, True], [False, True, False]])

    expanded = api['expand_mask'](
        original, num_dn=num_dn, queries_per_group=queries, groups=3)

    assert expanded.shape == (11, 11)
    assert torch.equal(expanded[:num_dn, :num_dn],
                       original[:num_dn, :num_dn])
    for group in range(3):
        start = num_dn + group * queries
        end = start + queries
        assert torch.equal(expanded[:num_dn, start:end],
                           original[:num_dn, num_dn:])
        assert torch.equal(expanded[start:end, :num_dn],
                           original[num_dn:, :num_dn])
        assert torch.equal(expanded[start:end, start:end],
                           original[num_dn:, num_dn:])
        for other in range(3):
            if other == group:
                continue
            other_start = num_dn + other * queries
            other_end = other_start + queries
            assert expanded[start:end, other_start:other_end].all()


def test_split_groups_validates_query_axis():
    api = _api()
    tensor = torch.arange(2 * 1 * 12 * 2).reshape(2, 1, 12, 2)
    groups = api['split'](tensor, queries_per_group=4, groups=3)
    assert len(groups) == 3
    assert all(group.shape == (2, 1, 4, 2) for group in groups)
    assert torch.equal(torch.cat(groups, dim=-2), tensor)
    with pytest.raises(ValueError, match='matching query axis'):
        api['split'](tensor, queries_per_group=5, groups=3)


def test_grouped_matching_invokes_one_loss_per_group_then_averages():
    api = _api()
    cls_scores = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12, 1)
    box_preds = torch.arange(60, dtype=torch.float32).reshape(1, 1, 12, 5)
    seen = []

    def loss_fn(group_cls, group_box):
        seen.append((group_cls.clone(), group_box.clone()))
        return {
            'loss_cls': group_cls.mean(),
            'loss_bbox': group_box.mean(),
        }

    losses = api['grouped_losses'](
        cls_scores, box_preds, queries_per_group=4, groups=3,
        loss_fn=loss_fn)

    assert len(seen) == 3
    assert [item[0][0, 0, 0, 0].item() for item in seen] == [0, 4, 8]
    assert losses['loss_cls'].item() == pytest.approx(5.5)
    assert losses['loss_bbox'].item() == pytest.approx(29.5)


def test_average_requires_identical_keys_and_preserves_gradient():
    api = _api()
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(3.0, requires_grad=True)
    averaged = api['average']([
        {'loss_cls': first, 'loss_bbox': first * 2},
        {'loss_cls': second, 'loss_bbox': second * 2},
    ])
    assert averaged['loss_cls'].item() == 2.0
    assert averaged['loss_bbox'].item() == 4.0
    averaged['loss_cls'].backward()
    assert first.grad.item() == 0.5
    assert second.grad.item() == 0.5
    with pytest.raises(ValueError, match='identical keys'):
        api['average']([{'loss_cls': first}, {'loss_bbox': second}])


class _DummyInitializer:

    def __call__(self, batch_size):
        query = torch.arange(
            batch_size * 2 * 4, dtype=torch.float32).reshape(
                batch_size, 2, 4)
        references = torch.full((batch_size, 2, 5), 0.5)
        return query, references


class _DummyDNGenerator:

    def __call__(self, batch_data_samples):
        batch_size = len(batch_data_samples)
        labels = torch.full((batch_size, 2, 4), -1.0)
        boxes = torch.zeros(batch_size, 2, 5)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[:2, 2:] = True
        mask[2:, :2] = True
        meta = dict(num_denoising_queries=2, num_denoising_groups=1)
        return labels, boxes, mask, meta


class _DummyDetector:

    num_queries = 2
    train_query_groups = 3
    query_initializer = _DummyInitializer()
    dn_query_generator = _DummyDNGenerator()


def _pre_decoder(training):
    from projects.OVCapFlow.ov_capflow.ov_capflow import OVCapFlow
    detector = _DummyDetector()
    detector.training = training
    return OVCapFlow.pre_decoder(
        detector,
        memory=torch.zeros(2, 7, 4),
        memory_mask=torch.zeros(2, 7, dtype=torch.bool),
        spatial_shapes=torch.tensor([[7, 1]]),
        memory_text=torch.zeros(2, 6, 4),
        text_token_mask=torch.ones(2, 6, dtype=torch.bool),
        batch_data_samples=[object(), object()])


def test_pre_decoder_groups_only_during_training():
    decoder_inputs, head_inputs = _pre_decoder(training=True)
    assert decoder_inputs['query'].shape == (2, 8, 4)
    assert decoder_inputs['reference_points'].shape == (2, 8, 5)
    assert decoder_inputs['dn_mask'].shape == (8, 8)
    assert decoder_inputs['matching_query_count'] == 6
    assert torch.equal(
        decoder_inputs['native_query'], decoder_inputs['query'])
    assert head_inputs['dn_meta']['num_matching_query_groups'] == 3
    assert head_inputs['dn_meta']['num_matching_queries_per_group'] == 2
    assert head_inputs['enc_outputs_class'] is None
    assert head_inputs['enc_outputs_coord'] is None

    decoder_inputs, head_inputs = _pre_decoder(training=False)
    assert decoder_inputs['query'].shape == (2, 2, 4)
    assert decoder_inputs['reference_points'].shape == (2, 2, 5)
    assert decoder_inputs['dn_mask'] is None
    assert decoder_inputs['matching_query_count'] == 2
    assert 'dn_meta' not in head_inputs


def test_head_grouped_loss_matches_each_group_and_computes_dn_once(
        monkeypatch):
    from projects.OVCapFlow.ov_capflow.ov_capflow_head import OVCapFlowHead

    head = OVCapFlowHead.__new__(OVCapFlowHead)
    head.matching_query_groups = 3
    matching_calls = []
    dn_calls = []

    def matching_loss(self, cls_scores, box_preds, batch_gt_instances,
                      batch_img_metas, batch_gt_instances_ignore=None):
        matching_calls.append((cls_scores.clone(), box_preds.clone()))
        return {
            'loss_cls': cls_scores.mean(),
            'loss_bbox': box_preds.mean(),
        }

    def dn_loss(self, cls_scores, box_preds, batch_gt_instances,
                batch_img_metas, dn_meta):
        dn_calls.append((cls_scores.clone(), box_preds.clone(), dn_meta))
        return {'dn_loss_cls': cls_scores.new_tensor(99.0)}

    monkeypatch.setattr(
        OVCapFlowHead, '_matching_loss_by_feat', matching_loss, raising=False)
    monkeypatch.setattr(
        OVCapFlowHead, '_denoising_loss_dict', dn_loss, raising=False)
    cls_scores = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
    box_preds = torch.arange(40, dtype=torch.float32).reshape(1, 1, 8, 5)
    losses = OVCapFlowHead.loss_by_feat(
        head,
        cls_scores,
        box_preds,
        enc_cls_scores=None,
        enc_bbox_preds=None,
        batch_gt_instances=[],
        batch_img_metas=[],
        dn_meta=dict(
            num_denoising_queries=2,
            num_denoising_groups=1,
            num_matching_query_groups=3,
            num_matching_queries_per_group=2))

    assert len(matching_calls) == 3
    assert len(dn_calls) == 1
    assert [call[0][0, 0, 0, 0].item() for call in matching_calls] == [2, 4, 6]
    assert losses['loss_cls'].item() == pytest.approx(4.5)
    assert losses['loss_bbox'].item() == pytest.approx(24.5)
    assert losses['dn_loss_cls'].item() == 99.0


def test_head_one_group_uses_parent_loss_path_exactly(monkeypatch):
    from projects.OVCapFlow.ov_capflow.ov_capflow_head import OVCapFlowHead
    from projects.GroundingDINO.groundingdino.grounding_dino_head import (
        RotatedGroundingDINOHead)

    sentinel = {'loss_cls': torch.tensor(7.0)}
    monkeypatch.setattr(
        RotatedGroundingDINOHead,
        'loss_by_feat',
        lambda self, *args, **kwargs: sentinel)
    head = OVCapFlowHead.__new__(OVCapFlowHead)
    head.matching_query_groups = 1
    result = OVCapFlowHead.loss_by_feat(
        head,
        torch.zeros(1, 1, 3, 1),
        torch.zeros(1, 1, 3, 5),
        None,
        None,
        [],
        [],
        dict(num_denoising_queries=1, num_denoising_groups=1))
    assert result is sentinel
