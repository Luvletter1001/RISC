import json

import torch

from projects.OVCapFlow.tools.audit_parent_equivalence import tensor_report


def test_tensor_report_marks_identical_values_exact():
    value = torch.tensor([[1.0, 2.0]])

    report = tensor_report(value, value.clone())

    assert report == {
        'parent_shape': [1, 2],
        'candidate_shape': [1, 2],
        'exact': True,
        'max_abs_diff': 0.0,
    }


def test_tensor_report_serializes_identical_negative_infinity_as_zero_diff():
    value = torch.tensor([float('-inf'), 1.0])

    report = tensor_report(value, value.clone())

    assert report['exact']
    assert report['max_abs_diff'] == 0.0
    json.dumps(report, allow_nan=False)


def test_tensor_report_measures_changed_values():
    parent = torch.tensor([[1.0, 2.0]])
    candidate = torch.tensor([[1.0, 2.25]])

    report = tensor_report(parent, candidate)

    assert not report['exact']
    assert report['max_abs_diff'] == 0.25


def test_tensor_report_ignores_shared_infinity_when_finite_value_changes():
    parent = torch.tensor([float('-inf'), 1.0])
    candidate = torch.tensor([float('-inf'), 1.25])

    report = tensor_report(parent, candidate)

    assert not report['exact']
    assert report['max_abs_diff'] == 0.25


def test_tensor_report_rejects_shape_mismatch_without_subtraction():
    report = tensor_report(torch.zeros(1, 2), torch.zeros(2, 1))

    assert not report['exact']
    assert report['max_abs_diff'] is None
