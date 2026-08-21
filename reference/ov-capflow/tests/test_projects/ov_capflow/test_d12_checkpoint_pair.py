import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch


def _api():
    from projects.OVCapFlow.tools.prepare_d12_checkpoint_pair import (
        AUTHORIZED_POSITION_COUNT,
        allowed_transport_keys,
        build_d12_pair_states,
        build_manifest,
        publish_artifacts_no_clobber,
        validate_pair_delta,
    )
    return {
        'authorized_count': AUTHORIZED_POSITION_COUNT,
        'allowed_keys': allowed_transport_keys,
        'build_pair': build_d12_pair_states,
        'build_manifest': build_manifest,
        'publish': publish_artifacts_no_clobber,
        'validate': validate_pair_delta,
    }


def _weight_key(branch):
    return f'bbox_head.reg_branches.{branch}.4.weight'


def _bias_key(branch):
    return f'bbox_head.reg_branches.{branch}.4.bias'


def _states():
    target = OrderedDict()
    target['backbone.shared'] = torch.zeros(2, 2)
    target['query_initializer.query_embedding.weight'] = torch.randn(6, 4)
    target['query_initializer.reference_embedding.weight'] = torch.randn(6, 5)
    target['dn_query_generator.label_embedding.weight'] = torch.randn(3, 4)
    target['bbox_head.cls_branches.0.bias'] = torch.randn(1)
    for branch in range(7):
        target[_weight_key(branch)] = torch.zeros(5, 256)
        target[_bias_key(branch)] = torch.zeros(5)

    source = OrderedDict()
    source['backbone.shared'] = torch.full((2, 2), 7.0)
    source['query_embedding.weight'] = torch.full((9, 4), 99.0)
    for branch in range(6):
        base = 1.0 + branch * 2048
        source[_weight_key(branch)] = (
            torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256)
            + base)
        source[_bias_key(branch)] = (
            torch.arange(4, dtype=torch.float32) + base)
    return target, source


def _clone_state(state):
    return OrderedDict((key, value.clone()) for key, value in state.items())


def test_allowed_terminal_keys_are_exactly_decoder_branches_zero_to_five():
    api = _api()

    assert api['allowed_keys']() == tuple(
        f'bbox_head.reg_branches.{branch}.4.{suffix}'
        for branch in range(6)
        for suffix in ('weight', 'bias'))
    assert api['authorized_count'] == 6168


def test_transport_replaces_only_xywh_and_preserves_angle_and_branch_six():
    api = _api()
    target, source = _states()
    original_target = _clone_state(target)
    original_source = _clone_state(source)

    control, candidate, summary = api['build_pair'](target, source)

    assert tuple(control) == tuple(sorted(target))
    assert tuple(candidate) == tuple(sorted(target))
    assert summary['transported_keys'] == list(api['allowed_keys']())
    assert summary['authorized_position_count'] == 6168
    assert summary['actual_unequal_position_count'] == 6168
    for branch in range(6):
        weight_key = _weight_key(branch)
        bias_key = _bias_key(branch)
        assert torch.equal(candidate[weight_key][:4], source[weight_key])
        assert torch.equal(candidate[bias_key][:4], source[bias_key])
        assert torch.equal(candidate[weight_key][4:], control[weight_key][4:])
        assert torch.equal(candidate[bias_key][4:], control[bias_key][4:])
    for key in (_weight_key(6), _bias_key(6)):
        assert torch.equal(candidate[key], control[key])
        assert torch.count_nonzero(candidate[key]) == 0
    assert torch.equal(
        candidate['query_initializer.query_embedding.weight'],
        control['query_initializer.query_embedding.weight'])
    assert torch.equal(
        candidate['query_initializer.reference_embedding.weight'],
        control['query_initializer.reference_embedding.weight'])
    assert torch.equal(
        candidate['dn_query_generator.label_embedding.weight'],
        control['dn_query_generator.label_embedding.weight'])
    assert torch.equal(
        candidate['bbox_head.cls_branches.0.bias'],
        control['bbox_head.cls_branches.0.bias'])
    assert torch.equal(control['backbone.shared'], source['backbone.shared'])
    assert control['backbone.shared'].data_ptr() != source[
        'backbone.shared'].data_ptr()
    assert all(torch.equal(target[key], original_target[key]) for key in target)
    assert all(torch.equal(source[key], original_source[key]) for key in source)


@pytest.mark.parametrize(
    'mutation,error',
    [
        ('missing_source', 'missing source terminal'),
        ('wrong_source_weight_shape', 'source terminal shape'),
        ('wrong_source_bias_shape', 'source terminal shape'),
        ('wrong_target_weight_shape', 'target terminal shape'),
        ('wrong_target_bias_shape', 'target terminal shape'),
        ('dtype_mismatch', 'terminal dtype mismatch'),
        ('nonfinite_source', 'non-finite source terminal'),
        ('nonzero_branch6', 'branch 6 must be all-zero'),
        ('bad_target_weight_init', 'target terminal weight'),
        ('bad_target_bias_init', 'target terminal bias'),
    ])
def test_transport_fails_closed_on_invalid_terminal_contract(mutation, error):
    target, source = _states()
    if mutation == 'missing_source':
        source.pop(_weight_key(3))
    elif mutation == 'wrong_source_weight_shape':
        source[_weight_key(3)] = torch.zeros(5, 256)
    elif mutation == 'wrong_source_bias_shape':
        source[_bias_key(3)] = torch.zeros(5)
    elif mutation == 'wrong_target_weight_shape':
        target[_weight_key(3)] = torch.zeros(4, 256)
    elif mutation == 'wrong_target_bias_shape':
        target[_bias_key(3)] = torch.zeros(4)
    elif mutation == 'dtype_mismatch':
        source[_weight_key(3)] = source[_weight_key(3)].double()
    elif mutation == 'nonfinite_source':
        source[_weight_key(3)][0, 0] = float('nan')
    elif mutation == 'nonzero_branch6':
        target[_weight_key(6)][0, 0] = 1
    elif mutation == 'bad_target_weight_init':
        target[_weight_key(1)][0, 0] = 1
    elif mutation == 'bad_target_bias_init':
        target[_bias_key(0)][2:] = -2

    with pytest.raises((KeyError, ValueError), match=error):
        _api()['build_pair'](target, source)


def test_validate_pair_rejects_changes_outside_allowed_rows():
    api = _api()
    target, source = _states()
    control, candidate, _ = api['build_pair'](target, source)

    candidate['backbone.shared'][0, 0] += 1
    with pytest.raises(ValueError, match='outside allowed transport'):
        api['validate'](control, candidate, source)

    control, candidate, _ = api['build_pair'](target, source)
    candidate[_weight_key(2)][4, 0] += 1
    with pytest.raises(ValueError, match='angle row'):
        api['validate'](control, candidate, source)

    control, candidate, _ = api['build_pair'](target, source)
    candidate[_weight_key(6)][0, 0] += 1
    with pytest.raises(ValueError, match='branch 6'):
        api['validate'](control, candidate, source)

    control, candidate, _ = api['build_pair'](target, source)
    candidate[_bias_key(1)][0] += 1
    with pytest.raises(ValueError, match='source XYWH'):
        api['validate'](control, candidate, source)


def test_actual_unequal_count_is_computed_not_assumed():
    api = _api()
    target, source = _states()
    source[_weight_key(4)].zero_()
    source[_bias_key(4)].zero_()

    _, _, summary = api['build_pair'](target, source)

    assert summary['authorized_position_count'] == 6168
    assert summary['actual_unequal_position_count'] == 6168 - (4 * 256 + 4)


def test_manifest_has_auditable_transport_and_safety_schema(tmp_path):
    api = _api()
    target, source = _states()
    control, candidate, summary = api['build_pair'](target, source)
    source_path = tmp_path / 'groundingdino_swint_ogc.pth'
    config_path = tmp_path / 'control.py'
    control_path = tmp_path / 'control.pth'
    candidate_path = tmp_path / 'candidate.pth'
    source_path.write_bytes(b'generic')
    config_path.write_text('model = dict(type="OVCapFlow")\n')
    control_path.write_bytes(b'control')
    candidate_path.write_bytes(b'candidate')

    manifest = api['build_manifest'](
        source_path=source_path,
        config_path=config_path,
        control_path=control_path,
        candidate_path=candidate_path,
        control=control,
        candidate=candidate,
        summary=summary,
        git_commit='abc123')

    assert manifest['schema_version'] == 1
    assert manifest['git_commit'] == 'abc123'
    assert manifest['checkpoint_payload_keys'] == ['state_dict']
    assert manifest['transport']['transported_keys'] == list(
        api['allowed_keys']())
    assert manifest['transport']['authorized_position_count'] == 6168
    assert manifest['transport']['actual_unequal_position_count'] == 6168
    assert manifest['transport']['query_transport'] == {'enabled': False}
    assert manifest['transport']['reference_transport'] == {'enabled': False}
    assert manifest['transport']['dn_transport'] == {'enabled': False}
    assert manifest['transport']['classification_transport'] == {
        'enabled': False}
    assert all(manifest['checks'].values())
    assert manifest['optimizer_state_present'] is False
    assert manifest['scheduler_state_present'] is False
    assert manifest['ema_state_present'] is False
    assert manifest['teacher_state_present'] is False
    assert manifest['forbidden_hits'] == []
    assert manifest['included_keys'] == sorted(control)
    assert manifest['included_shapes'][_weight_key(0)] == [5, 256]
    assert manifest['outputs']['control']['size_bytes'] == len(b'control')
    assert manifest['outputs']['candidate']['size_bytes'] == len(b'candidate')


def test_publish_artifacts_is_atomic_no_clobber_and_deterministic(tmp_path):
    publish = _api()['publish']
    first = tmp_path / 'first.bin'
    second = tmp_path / 'second.json'
    payloads = [
        (first, b'checkpoint'),
        (second, json.dumps({'pass': True}, sort_keys=True).encode() + b'\n'),
    ]

    publish(payloads)

    assert first.read_bytes() == b'checkpoint'
    assert json.loads(second.read_text()) == {'pass': True}
    assert not list(tmp_path.glob('*.pending.*'))
    with pytest.raises(FileExistsError, match='output collision'):
        publish(payloads)
    assert first.read_bytes() == b'checkpoint'


def test_publish_refuses_any_existing_pending_path_before_writing(tmp_path):
    publish = _api()['publish']
    first = tmp_path / 'first.bin'
    second = tmp_path / 'second.bin'
    pending = tmp_path / 'second.bin.pending.999'
    pending.write_bytes(b'evidence')

    with pytest.raises(FileExistsError, match='output collision'):
        publish([(first, b'first'), (second, b'second')])

    assert not first.exists()
    assert not second.exists()
    assert pending.read_bytes() == b'evidence'
