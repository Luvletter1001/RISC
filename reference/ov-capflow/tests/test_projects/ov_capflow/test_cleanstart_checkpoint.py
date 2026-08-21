import json
from pathlib import Path

import pytest
import torch


def _api():
    from projects.OVCapFlow.tools.prepare_cleanstart_checkpoint import (
        convert_groundingdino_state_dict,
        filter_compatible_state_dict,
        reject_forbidden_namespaces,
        reject_forbidden_source,
        require_prefix_coverage,
        sha256_file,
        transport_first_content_queries,
        unwrap_model_state,
        write_provenance,
    )
    return {
        'convert': convert_groundingdino_state_dict,
        'filter': filter_compatible_state_dict,
        'reject_namespaces': reject_forbidden_namespaces,
        'reject_source': reject_forbidden_source,
        'require_prefixes': require_prefix_coverage,
        'sha256': sha256_file,
        'transport_queries': transport_first_content_queries,
        'unwrap': unwrap_model_state,
        'write_provenance': write_provenance,
    }


def test_filter_keeps_only_exact_name_and_shape_matches():
    api = _api()
    source = {
        'backbone.ok': torch.ones(2, 2),
        'bbox_head.reg_branches.0.2.weight': torch.ones(4, 8),
        'query_embedding.weight': torch.ones(900, 8),
        'encoder.unused': torch.ones(1),
    }
    target = {
        'backbone.ok': torch.zeros(2, 2),
        'bbox_head.reg_branches.0.2.weight': torch.zeros(5, 8),
        'query_initializer.query_embedding.weight': torch.zeros(600, 8),
    }

    kept, report = api['filter'](source, target)

    assert tuple(kept) == ('backbone.ok',)
    assert report['shape_mismatches'] == [{
        'key': 'bbox_head.reg_branches.0.2.weight',
        'source_shape': [4, 8],
        'target_shape': [5, 8],
    }]
    assert report['unexpected_keys'] == [
        'encoder.unused', 'query_embedding.weight']
    assert report['missing_keys'] == [
        'bbox_head.reg_branches.0.2.weight',
        'query_initializer.query_embedding.weight',
    ]
    assert report['matched_numel'] == 4
    assert report['target_numel'] == 4 + 40 + 4800
    assert report['coverage_ratio'] == pytest.approx(4 / 4844)


def test_transport_first_content_queries_injects_exact_first_rows():
    api = _api()
    source_queries = torch.arange(
        9 * 4, dtype=torch.float32).reshape(9, 4)
    compatible = {'backbone.ok': torch.ones(2, 2)}
    target = {
        'backbone.ok': torch.zeros(2, 2),
        'query_initializer.query_embedding.weight': torch.zeros(6, 4),
    }

    transported, report = api['transport_queries'](
        {'query_embedding.weight': source_queries},
        compatible,
        target,
        count=6)

    assert tuple(transported) == (
        'backbone.ok',
        'query_initializer.query_embedding.weight',
    )
    assert torch.equal(
        transported['query_initializer.query_embedding.weight'],
        source_queries[:6])
    assert transported[
        'query_initializer.query_embedding.weight'].data_ptr() != (
            source_queries.data_ptr())
    assert report == {
        'enabled': True,
        'source_key': 'query_embedding.weight',
        'target_key': 'query_initializer.query_embedding.weight',
        'source_shape': [9, 4],
        'transported_shape': [6, 4],
        'selection': 'first_contiguous_rows',
        'start_row_inclusive': 0,
        'end_row_exclusive': 6,
    }


@pytest.mark.parametrize(
    'source,target,compatible,count,error',
    [
        ({}, {'query_initializer.query_embedding.weight': torch.zeros(6, 4)},
         {}, 6, 'source query key'),
        ({'query_embedding.weight': torch.zeros(5, 4)},
         {'query_initializer.query_embedding.weight': torch.zeros(6, 4)},
         {}, 6, 'source query shape'),
        ({'query_embedding.weight': torch.zeros(9, 5)},
         {'query_initializer.query_embedding.weight': torch.zeros(6, 4)},
         {}, 6, 'source query shape'),
        ({'query_embedding.weight': torch.zeros(9, 4)},
         {'query_initializer.query_embedding.weight': torch.zeros(7, 4)},
         {}, 6, 'target query shape'),
        ({'query_embedding.weight': torch.zeros(9, 4)},
         {'query_initializer.query_embedding.weight': torch.zeros(6, 4)},
         {'query_initializer.query_embedding.weight': torch.zeros(6, 4)},
         6, 'already present'),
    ])
def test_transport_first_content_queries_fails_closed(
        source, target, compatible, count, error):
    with pytest.raises((KeyError, ValueError), match=error):
        _api()['transport_queries'](
            source, compatible, target, count=count)


@pytest.mark.parametrize('key', [
    'teacher.backbone.weight',
    'student.distill_head.weight',
    'pseudo_bank.items',
    'dense_head.cls.weight',
    'rpn_head.conv.weight',
    'roi_head.fc.weight',
])
def test_forbidden_namespace_is_rejected(key):
    with pytest.raises(ValueError, match='forbidden checkpoint namespace'):
        _api()['reject_namespaces']({key: torch.ones(1)})


@pytest.mark.parametrize('name', [
    'dota_epoch_12.pth',
    'OpenRSD_best.pth',
    'hrsc_parent.pth',
    'FAIR1M_model.pth',
    'dior-r_model.pth',
    'p126c.pth',
    'p121.pth',
    'teacher.pth',
    'distill.pth',
    'pseudo_labels.pth',
])
def test_forbidden_source_name_is_rejected(tmp_path, name):
    with pytest.raises(ValueError, match='forbidden checkpoint source'):
        _api()['reject_source'](tmp_path / name)


def test_generic_groundingdino_source_name_is_allowed(tmp_path):
    _api()['reject_source'](tmp_path / 'groundingdino_swint_ogc.pth')


def test_unwrap_accepts_model_only_and_rejects_training_state():
    api = _api()
    tensor = torch.ones(1)
    assert api['unwrap']({'model': {'module.backbone.x': tensor}}) == {
        'backbone.x': tensor}

    for forbidden in ('optimizer', 'optim_wrapper', 'param_schedulers',
                      'ema_state_dict'):
        with pytest.raises(ValueError, match='training state'):
            api['unwrap']({
                'model': {'backbone.x': tensor},
                forbidden: {'state': tensor},
            })


def test_converter_matches_representative_official_rules():
    api = _api()
    converted = api['convert']({
        'backbone.0.patch_embed.proj.weight': torch.ones(1),
        'bert.embeddings.word_embeddings.weight': torch.ones(1),
        'feat_map.weight': torch.ones(1),
        'transformer.encoder.layers.0.norm1.weight': torch.ones(1),
        'transformer.decoder.layers.0.self_attn.in_proj_weight':
            torch.ones(1),
        'transformer.decoder.bbox_embed.0.layers.2.weight': torch.ones(1),
        'transformer.tgt_embed.weight': torch.ones(1),
    })

    assert tuple(converted) == (
        'backbone.patch_embed.projection.weight',
        'bbox_head.reg_branches.0.4.weight',
        'decoder.layers.0.self_attn.attn.in_proj_weight',
        'encoder.layers.0.norms.0.weight',
        'language_model.language_backbone.body.model.embeddings.'
        'word_embeddings.weight',
        'query_embedding.weight',
        'text_feat_map.weight',
    )


def test_converter_collapses_identical_official_bbox_aliases():
    convert = _api()['convert']
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    converted = convert({
        'transformer.decoder.bbox_embed.0.layers.0.weight': weight,
        'bbox_embed.0.layers.0.weight': weight.clone(),
    })
    assert tuple(converted) == ('bbox_head.reg_branches.0.0.weight',)
    assert torch.equal(
        converted['bbox_head.reg_branches.0.0.weight'], weight)


def test_converter_rejects_conflicting_official_bbox_aliases():
    convert = _api()['convert']
    with pytest.raises(ValueError, match='conflicting converted checkpoint'):
        convert({
            'transformer.decoder.bbox_embed.0.layers.0.weight':
                torch.zeros(2, 4),
            'bbox_embed.0.layers.0.weight': torch.ones(2, 4),
        })


def test_required_prefix_coverage_fails_loudly():
    require = _api()['require_prefixes']
    state = {
        'backbone.x': torch.ones(1),
        'encoder.x': torch.ones(1),
    }
    with pytest.raises(ValueError, match='decoder'):
        require(state, ('backbone.', 'encoder.', 'decoder.'))


def test_sha_and_provenance_are_deterministic(tmp_path):
    api = _api()
    payload = tmp_path / 'payload.bin'
    payload.write_bytes(b'clean-start')
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    report = {
        'source_sha256': api['sha256'](payload),
        'forbidden_hits': [],
        'optimizer_state_present': False,
        'scheduler_state_present': False,
        'ema_state_present': False,
    }

    api['write_provenance'](first, report)
    api['write_provenance'](second, report)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b'\n')
    assert json.loads(first.read_text(encoding='utf-8')) == report
    assert report['source_sha256'] == (
        'b436dd7bee448114beef2313742c1602fe4f6f5d5461fe07feabe304ca65ff84')
