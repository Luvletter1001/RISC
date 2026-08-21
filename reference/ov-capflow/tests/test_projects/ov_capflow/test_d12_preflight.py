import json
from collections import OrderedDict

import pytest
import torch


def _api():
    from projects.OVCapFlow.tools.audit_d12_preflight import (
        assert_deterministic_tensors,
        compare_complete_states,
        enforce_loss_gradient_pair,
        enforce_prediction_pair,
        publish_json_no_clobber,
        summarize_normalized_cxywh,
    )
    return {
        'compare': compare_complete_states,
        'deterministic': assert_deterministic_tensors,
        'loss_gate': enforce_loss_gradient_pair,
        'prediction_gate': enforce_prediction_pair,
        'publish': publish_json_no_clobber,
        'summarize': summarize_normalized_cxywh,
    }


def _weight_key(branch):
    return f'bbox_head.reg_branches.{branch}.4.weight'


def _bias_key(branch):
    return f'bbox_head.reg_branches.{branch}.4.bias'


def _complete_pair():
    control = OrderedDict()
    control['query_initializer.query_embedding.weight'] = torch.randn(6, 4)
    control['query_initializer.reference_embedding.weight'] = torch.randn(6, 5)
    control['dn_query_generator.label_embedding.weight'] = torch.randn(3, 4)
    control['bbox_head.cls_branches.0.bias'] = torch.randn(1)
    for branch in range(7):
        control[_weight_key(branch)] = torch.zeros(5, 256)
        control[_bias_key(branch)] = torch.zeros(5)
    candidate = OrderedDict(
        (key, value.clone()) for key, value in control.items())
    source = OrderedDict()
    for branch in range(6):
        source[_weight_key(branch)] = torch.full((4, 256), branch + 1.0)
        source[_bias_key(branch)] = torch.full((4,), branch + 1.0)
        candidate[_weight_key(branch)][:4].copy_(
            source[_weight_key(branch)])
        candidate[_bias_key(branch)][:4].copy_(source[_bias_key(branch)])
    return control, candidate, source


def test_compare_complete_states_accepts_only_exact_d12_delta():
    control, candidate, source = _complete_pair()

    report = _api()['compare'](control, candidate, source)

    assert report['pass'] is True
    assert report['authorized_position_count'] == 6168
    assert report['actual_unequal_position_count'] == 6168
    assert report['query_state_equal'] is True
    assert report['reference_state_equal'] is True
    assert report['dn_state_equal'] is True
    assert report['classification_bias_equal'] is True


@pytest.mark.parametrize(
    'key,mutate,error',
    [
        ('query_initializer.query_embedding.weight',
         lambda value: value.add_(1), 'outside allowed transport'),
        ('query_initializer.reference_embedding.weight',
         lambda value: value.add_(1), 'outside allowed transport'),
        ('dn_query_generator.label_embedding.weight',
         lambda value: value.add_(1), 'outside allowed transport'),
        ('bbox_head.cls_branches.0.bias',
         lambda value: value.add_(1), 'outside allowed transport'),
        (_weight_key(0), lambda value: value[4:].add_(1), 'angle row'),
        (_weight_key(6), lambda value: value.add_(1), 'branch 6'),
    ])
def test_compare_complete_states_rejects_boundary_violations(
        key, mutate, error):
    control, candidate, source = _complete_pair()
    mutate(candidate[key])

    with pytest.raises(ValueError, match=error):
        _api()['compare'](control, candidate, source)


def test_summarize_normalized_boxes_reports_saturation_and_area():
    boxes = torch.tensor([
        [0.50, 0.50, 0.20, 0.40, 0.10],
        [0.25, 0.75, 0.10, 0.20, 0.90],
    ])

    summary = _api()['summarize'](boxes)

    assert summary['prediction_count'] == 2
    assert summary['coordinate_count'] == 8
    assert summary['saturation_fraction'] == 0
    assert summary['median_area'] == pytest.approx(0.05)
    assert summary['p99_area'] == pytest.approx(0.0794)
    assert summary['min_cxywh'] == pytest.approx(0.1)
    assert summary['max_cxywh'] == pytest.approx(0.75)


@pytest.mark.parametrize(
    'boxes,error',
    [
        (torch.tensor([[float('nan'), 0.5, 0.2, 0.2, 0.0]]), 'finite'),
        (torch.tensor([[-0.1, 0.5, 0.2, 0.2, 0.0]]), r'\[0, 1\]'),
        (torch.tensor([[1.1, 0.5, 0.2, 0.2, 0.0]]), r'\[0, 1\]'),
        (torch.tensor([[0.5, 0.5, 0.0, 0.2, 0.0]]), 'strictly positive'),
        (torch.zeros(2, 4), 'last dimension'),
    ])
def test_summarize_normalized_boxes_fails_closed(boxes, error):
    with pytest.raises(ValueError, match=error):
        _api()['summarize'](boxes)


def _prediction_summary(saturation, median_area, p99_area=0.5):
    return {
        'prediction_count': 600,
        'per_image_counts': [600],
        'saturation_fraction': saturation,
        'median_area': median_area,
        'p99_area': p99_area,
        'all_finite': True,
        'all_cxywh_in_unit_interval': True,
        'all_width_height_positive': True,
    }


def test_prediction_gate_accepts_inclusive_frozen_boundaries():
    control = _prediction_summary(0.0, 1.0)
    candidate = _prediction_summary(0.05, 1 / 16, p99_area=1.0)

    report = _api()['prediction_gate'](candidate, control)

    assert report['pass'] is True
    assert report['median_area_ratio'] == pytest.approx(1 / 16)
    assert report['saturation_increase'] == pytest.approx(0.05)


@pytest.mark.parametrize(
    'candidate,control,error',
    [
        (_prediction_summary(0.050001, 1.0),
         _prediction_summary(0.0, 1.0), 'saturation fraction'),
        (_prediction_summary(0.05, 1.0),
         _prediction_summary(-0.001, 1.0), 'saturation increase'),
        (_prediction_summary(0.0, 0.062499),
         _prediction_summary(0.0, 1.0), 'median area ratio'),
        (_prediction_summary(0.0, 16.001),
         _prediction_summary(0.0, 1.0), 'median area ratio'),
        (_prediction_summary(0.0, 1.0, p99_area=1.00001),
         _prediction_summary(0.0, 1.0), 'p99 area'),
        ({**_prediction_summary(0.0, 1.0), 'per_image_counts': [599]},
         _prediction_summary(0.0, 1.0), 'Q600'),
    ])
def test_prediction_gate_rejects_each_threshold(
        candidate, control, error):
    with pytest.raises(ValueError, match=error):
        _api()['prediction_gate'](candidate, control)


def _loss_stats(total=2.0, grad=3.0):
    return {
        'individual_losses': {'loss_cls': 1.0, 'loss_bbox': 1.0},
        'total_loss': total,
        'gradient_norm': grad,
        'gradients_finite': True,
        'parameters_unchanged': True,
        'prediction_succeeds_after_backward': True,
        'max_estimated_query_area': 49_000_000,
    }


def test_loss_gradient_gate_accepts_inclusive_frozen_boundaries():
    control = _loss_stats(total=2.0, grad=3.0)
    candidate = _loss_stats(total=40.0, grad=300.0)

    report = _api()['loss_gate'](candidate, control)

    assert report['pass'] is True
    assert report['total_loss_ratio'] == pytest.approx(20)
    assert report['gradient_norm_ratio'] == pytest.approx(100)


@pytest.mark.parametrize(
    'field,value,error',
    [
        ('total_loss', 40.001, 'total loss ratio'),
        ('gradient_norm', 300.001, 'gradient norm ratio'),
        ('gradients_finite', False, 'finite gradients'),
        ('parameters_unchanged', False, 'parameter changed'),
        ('prediction_succeeds_after_backward', False, 'prediction after'),
        ('max_estimated_query_area', 50_000_001, 'query area'),
    ])
def test_loss_gradient_gate_rejects_each_threshold(field, value, error):
    candidate = _loss_stats()
    candidate[field] = value
    with pytest.raises(ValueError, match=error):
        _api()['loss_gate'](candidate, _loss_stats())


def test_loss_gradient_gate_rejects_nonfinite_or_nonpositive_control():
    candidate = _loss_stats()
    candidate['individual_losses']['loss_bbox'] = float('nan')
    with pytest.raises(ValueError, match='finite losses'):
        _api()['loss_gate'](candidate, _loss_stats())

    with pytest.raises(ValueError, match='positive control total loss'):
        _api()['loss_gate'](_loss_stats(), _loss_stats(total=0))

    with pytest.raises(ValueError, match='positive control gradient norm'):
        _api()['loss_gate'](_loss_stats(), _loss_stats(grad=0))


def test_determinism_requires_bitwise_identical_tensor_sequence():
    check = _api()['deterministic']
    first = [torch.tensor([[1.0, 2.0]]), torch.tensor([3])]
    second = [value.clone() for value in first]

    assert check(first, second) is True
    second[0][0, 0] += 1e-7
    with pytest.raises(ValueError, match='not deterministic'):
        check(first, second)


def test_preflight_report_publication_is_no_clobber(tmp_path):
    output = tmp_path / 'preflight.json'

    _api()['publish']({'pass': True, 'ratio': 2.0}, output)

    assert json.loads(output.read_text()) == {'pass': True, 'ratio': 2.0}
    assert output.read_bytes().endswith(b'\n')
    with pytest.raises(FileExistsError, match='output collision'):
        _api()['publish']({'pass': False}, output)


def test_preflight_report_rejects_existing_pending_file(tmp_path):
    output = tmp_path / 'preflight.json'
    pending = tmp_path / 'preflight.json.pending.77'
    pending.write_text('evidence')

    with pytest.raises(FileExistsError, match='output collision'):
        _api()['publish']({'pass': True}, output)

    assert pending.read_text() == 'evidence'
    assert not output.exists()
