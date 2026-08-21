import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
               'tools' / 'validate_dotav2_q600_dump.py')


def _load_module():
    spec = importlib.util.spec_from_file_location(
        'validate_dotav2_q600_dump', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(image_id, queries=2):
    return {
        'img_id': image_id,
        'pred_instances': {
            'bboxes': torch.zeros((queries, 5)),
            'scores': torch.linspace(0.1, 0.9, queries),
            'labels': torch.arange(queries) % 3,
        },
        'gt_instances': {
            'bboxes': torch.zeros((1, 5)),
            'labels': torch.zeros((1,), dtype=torch.long),
        },
    }


def test_dump_validation_error_preserves_both_public_error_contracts():
    validator = _load_module()
    error = validator.DumpValidationError('invalid dump')

    assert isinstance(error, ValueError)
    assert isinstance(error, AssertionError)


def test_chained_helper_failure_uses_dump_validation_error():
    validator = _load_module()

    with pytest.raises(validator.DumpValidationError) as captured:
        validator.as_tensor(object(), 'prediction boxes')

    assert isinstance(captured.value, ValueError)
    assert isinstance(captured.value, AssertionError)
    assert isinstance(captured.value.__cause__, AttributeError)


def test_cpu_unpickler_and_validator_accept_valid_dump(tmp_path):
    validator = _load_module()
    dump_path = tmp_path / 'predictions.pkl'
    with dump_path.open('wb') as stream:
        pickle.dump([_record('a'), _record('b')], stream)

    records = validator.load_cpu(dump_path)
    summary = validator.validate_records(
        records,
        expected_records=2,
        queries_per_image=2,
        num_classes=3,
    )

    assert summary == {
        'records': 2,
        'unique_image_ids': 2,
        'prediction_rows': 4,
        'all_cpu_finite': True,
    }


def test_validator_rejects_duplicate_image_ids():
    validator = _load_module()

    with pytest.raises(validator.DumpValidationError,
                       match='duplicate image id'):
        validator.validate_records(
            [_record('same'), _record('same')],
            expected_records=2,
            queries_per_image=2,
            num_classes=3,
        )


def test_validator_rejects_nonfinite_boxes():
    validator = _load_module()
    record = _record('bad')
    record['pred_instances']['bboxes'][0, 0] = float('nan')

    with pytest.raises(validator.DumpValidationError,
                       match='non-finite prediction boxes'):
        validator.validate_records(
            [record],
            expected_records=1,
            queries_per_image=2,
            num_classes=3,
        )


@pytest.mark.parametrize(
    'argument,value',
    [('expected_records', 0), ('queries_per_image', -1), ('num_classes', 0)],
)
def test_validator_rejects_nonpositive_contract_arguments(argument, value):
    validator = _load_module()
    arguments = {
        'expected_records': 1,
        'queries_per_image': 2,
        'num_classes': 3,
    }
    arguments[argument] = value

    with pytest.raises(validator.DumpValidationError, match=argument):
        validator.validate_records([_record('a')], **arguments)


def test_validator_rejects_record_count_and_nonsequence_records():
    validator = _load_module()

    with pytest.raises(validator.DumpValidationError,
                       match='records must be a sequence'):
        validator.validate_records(None, expected_records=1)
    with pytest.raises(validator.DumpValidationError, match='record count'):
        validator.validate_records([], expected_records=1)


@pytest.mark.parametrize(
    'records,match',
    [
        ([None], 'record at index 0 must be a mapping'),
        ([{'img_id': 'a'}], 'missing record keys'),
        ([{
            'img_id': 'a',
            'pred_instances': None,
            'gt_instances': _record('a')['gt_instances'],
        }], 'pred_instances at index 0 must be a mapping'),
        ([{
            'img_id': 'a',
            'pred_instances': _record('a')['pred_instances'],
            'gt_instances': None,
        }], 'gt_instances at index 0 must be a mapping'),
    ],
)
def test_validator_rejects_invalid_record_structure(records, match):
    validator = _load_module()

    with pytest.raises(validator.DumpValidationError, match=match):
        validator.validate_records(
            records, expected_records=1, queries_per_image=2, num_classes=3)


@pytest.mark.parametrize('missing_key', ['bboxes', 'scores', 'labels'])
def test_validator_rejects_missing_prediction_keys(missing_key):
    validator = _load_module()
    record = _record('a')
    del record['pred_instances'][missing_key]

    with pytest.raises(validator.DumpValidationError,
                       match='missing prediction keys'):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


@pytest.mark.parametrize('missing_key', ['bboxes', 'labels'])
def test_validator_rejects_missing_gt_keys(missing_key):
    validator = _load_module()
    record = _record('a')
    del record['gt_instances'][missing_key]

    with pytest.raises(validator.DumpValidationError,
                       match='missing GT keys'):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


@pytest.mark.parametrize(
    'field,value,match',
    [
        ('bboxes', torch.zeros((2, 4)), 'prediction box shape'),
        ('scores', torch.zeros((2, 1)), 'prediction score shape'),
        ('labels', torch.zeros((2, 1), dtype=torch.long),
         'prediction label shape'),
    ],
)
def test_validator_rejects_prediction_shapes(field, value, match):
    validator = _load_module()
    record = _record('a')
    record['pred_instances'][field] = value

    with pytest.raises(validator.DumpValidationError, match=match):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


@pytest.mark.parametrize(
    'field,value,match',
    [
        ('bboxes', torch.tensor([[float('inf'), 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]]),
         'non-finite prediction boxes'),
        ('scores', torch.tensor([0.1, float('nan')]),
         'non-finite prediction scores'),
    ],
)
def test_validator_rejects_nonfinite_prediction_tensors(field, value, match):
    validator = _load_module()
    record = _record('a')
    record['pred_instances'][field] = value

    with pytest.raises(validator.DumpValidationError, match=match):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


@pytest.mark.parametrize(
    'labels,match',
    [
        (torch.tensor([0.0, 1.0]), 'prediction labels must have integer dtype'),
        (torch.tensor([False, True]), 'prediction labels must have integer dtype'),
        (torch.tensor([0, 3]), 'prediction label out of range'),
        (torch.tensor([-1, 0]), 'prediction label out of range'),
    ],
)
def test_validator_rejects_prediction_label_dtype_and_range(labels, match):
    validator = _load_module()
    record = _record('a')
    record['pred_instances']['labels'] = labels

    with pytest.raises(validator.DumpValidationError, match=match):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


@pytest.mark.parametrize(
    'boxes,labels,match',
    [
        (torch.zeros((1, 4)), torch.zeros((1,), dtype=torch.long),
         'GT box shape'),
        (torch.zeros((1, 5)), torch.zeros((1, 1), dtype=torch.long),
         'GT label shape'),
        (torch.zeros((2, 5)), torch.zeros((1,), dtype=torch.long),
         'GT box/label length mismatch'),
        (torch.tensor([[float('nan'), 0, 0, 0, 0]]),
         torch.zeros((1,), dtype=torch.long), 'non-finite GT boxes'),
        (torch.zeros((1, 5)), torch.tensor([0.0]),
         'GT labels must have integer dtype'),
        (torch.zeros((1, 5)), torch.tensor([True]),
         'GT labels must have integer dtype'),
        (torch.zeros((1, 5)), torch.tensor([3]), 'GT label out of range'),
        (torch.zeros((1, 5)), torch.tensor([-1]), 'GT label out of range'),
    ],
)
def test_validator_rejects_invalid_gt(boxes, labels, match):
    validator = _load_module()
    record = _record('a')
    record['gt_instances']['bboxes'] = boxes
    record['gt_instances']['labels'] = labels

    with pytest.raises(validator.DumpValidationError, match=match):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


def test_validator_accepts_empty_gt():
    validator = _load_module()
    record = _record('empty')
    record['gt_instances']['bboxes'] = torch.zeros((0, 5))
    record['gt_instances']['labels'] = torch.zeros((0,), dtype=torch.long)

    summary = validator.validate_records(
        [record], expected_records=1, queries_per_image=2, num_classes=3)

    assert summary['prediction_rows'] == 2


@pytest.mark.parametrize('section,field', [
    ('pred_instances', 'bboxes'),
    ('pred_instances', 'scores'),
    ('pred_instances', 'labels'),
    ('gt_instances', 'bboxes'),
    ('gt_instances', 'labels'),
])
def test_validator_rejects_non_tensor_values(section, field):
    validator = _load_module()
    record = _record('a')
    record[section][field] = object()

    with pytest.raises(validator.DumpValidationError,
                       match='must be a torch.Tensor'):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


@pytest.mark.parametrize('section,field', [
    ('pred_instances', 'bboxes'),
    ('pred_instances', 'scores'),
    ('pred_instances', 'labels'),
    ('gt_instances', 'bboxes'),
    ('gt_instances', 'labels'),
])
def test_validator_rejects_non_cpu_tensors_with_meta_fixture(section, field):
    validator = _load_module()
    record = _record('a')
    original = record[section][field]
    record[section][field] = torch.empty(
        original.shape, dtype=original.dtype, device='meta')

    with pytest.raises(validator.DumpValidationError, match='must be on CPU'):
        validator.validate_records(
            [record], expected_records=1, queries_per_image=2, num_classes=3)


def _run_cli(tmp_path, records, optimized=False):
    dump_path = tmp_path / ('optimized.pkl' if optimized else 'normal.pkl')
    with dump_path.open('wb') as stream:
        pickle.dump(records, stream)
    command = [sys.executable]
    if optimized:
        command.append('-O')
    command.extend([
        str(SCRIPT_PATH), str(dump_path), '--expected-records', '2',
        '--queries-per-image', '2', '--num-classes', '3',
    ])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_cli_accepts_valid_dump(tmp_path):
    result = _run_cli(tmp_path, [_record('a'), _record('b')])

    assert result.returncode == 0, result.stderr
    assert '"prediction_rows": 4' in result.stdout


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize(
    'corruption,expected_error',
    [
        ('duplicate', 'duplicate image id'),
        ('prediction_nan', 'non-finite prediction boxes'),
        ('prediction_label', 'prediction label out of range'),
        ('gt_nan', 'non-finite GT boxes'),
    ],
)
def test_cli_rejects_invalid_dump_even_with_python_optimized(
        tmp_path, optimized, corruption, expected_error):
    records = [_record('a'), _record('b')]
    if corruption == 'duplicate':
        records[1]['img_id'] = 'a'
    elif corruption == 'prediction_nan':
        records[0]['pred_instances']['bboxes'][0, 0] = float('nan')
    elif corruption == 'prediction_label':
        records[0]['pred_instances']['labels'][0] = 99
    elif corruption == 'gt_nan':
        records[0]['gt_instances']['bboxes'][0, 0] = float('nan')

    result = _run_cli(tmp_path, records, optimized=optimized)

    assert result.returncode != 0
    assert expected_error in result.stderr
