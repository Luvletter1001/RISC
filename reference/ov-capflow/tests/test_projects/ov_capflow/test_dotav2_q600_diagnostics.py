import copy
import csv
import importlib.util
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from mmrotate.evaluation.functional.mean_ap import tpfp_default


SCRIPT_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
               'tools' / 'dotav2_q600_diagnostics.py')
ANALYZER_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
                 'tools' / 'analyze_dotav2_q600_dump.py')


class _CustomReal(float):
    pass


def _load_module():
    module_name = 'dotav2_q600_diagnostics'
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_analyzer():
    module_name = 'analyze_dotav2_q600_dump'
    spec = importlib.util.spec_from_file_location(module_name, ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _record(img_id, pred_boxes, scores, labels, gt_boxes=(), gt_labels=(),
            ignored_boxes=(), ignored_labels=()):
    return {
        'img_id': img_id,
        'pred_instances': {
            'bboxes': torch.tensor(
                pred_boxes, dtype=torch.float32).reshape(-1, 5),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
        },
        'gt_instances': {
            'bboxes': torch.tensor(
                gt_boxes, dtype=torch.float32).reshape(-1, 5),
            'labels': torch.tensor(gt_labels, dtype=torch.long),
        },
        'ignored_instances': {
            'bboxes': torch.tensor(
                ignored_boxes, dtype=torch.float32).reshape(-1, 5),
            'labels': torch.tensor(ignored_labels, dtype=torch.long),
        },
    }


def test_protocol_and_metric_selection_are_exact(tmp_path):
    diagnostics = _load_module()
    config = tmp_path / 'protocol.py'
    config.write_text(
        "classes = ('base-a', 'novel-b')\n"
        "base_classes = ('base-a',)\n"
        "novel_classes = ('novel-b',)\n",
        encoding='utf-8')
    metrics = tmp_path / 'metrics.json'
    rows = (
        {'step': 23, 'dota/mAP': 0.4, 'dota/AP50': 0.4},
        {'step': 24, 'dota/mAP': 0.5123456789, 'dota/AP50': 0.512},
    )
    metrics.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    protocol = diagnostics.load_class_protocol(config, canonical=False)
    metric = diagnostics.read_metric_record(metrics, step=24)

    assert protocol.classes == ('base-a', 'novel-b')
    assert protocol.base_classes == ('base-a',)
    assert protocol.novel_classes == ('novel-b',)
    assert protocol.canonical is False
    assert metric.mAP == 0.5123456789
    assert metric.ap50 == 0.512
    assert metric.step == 24


@pytest.mark.parametrize(
    ('base_classes', 'novel_classes'),
    [
        (('a', 'a'), ('b',)),
        (('a',), ('b', 'b')),
    ],
)
def test_protocol_rejects_duplicate_partition_members(
        tmp_path, base_classes, novel_classes):
    diagnostics = _load_module()
    config = tmp_path / 'duplicate_protocol.py'
    config.write_text(
        'classes = {!r}\nbase_classes = {!r}\nnovel_classes = {!r}\n'.format(
            ('a', 'b'), base_classes, novel_classes),
        encoding='utf-8')

    with pytest.raises(diagnostics.DiagnosticError, match='unique|duplicate'):
        diagnostics.load_class_protocol(config, canonical=False)


def test_metric_directory_requires_one_applicable_row(tmp_path):
    diagnostics = _load_module()
    for index in range(2):
        path = tmp_path / 'metrics_{}.json'.format(index)
        path.write_text(
            json.dumps({'dota/mAP': 0.5, 'dota/AP50': 0.5}) + '\n',
            encoding='utf-8')

    with pytest.raises(diagnostics.DiagnosticError, match='exactly one'):
        diagnostics.read_metric_record(tmp_path)


@pytest.mark.parametrize('invalid_step', ['24', True, [24]])
def test_metric_record_rejects_noninteger_step(tmp_path, invalid_step):
    diagnostics = _load_module()
    metrics = tmp_path / 'invalid_step.json'
    metrics.write_text(
        json.dumps({
            'step': invalid_step,
            'dota/mAP': 0.5,
            'dota/AP50': 0.5,
        }) + '\n',
        encoding='utf-8')

    with pytest.raises(diagnostics.DiagnosticError, match='step'):
        diagnostics.read_metric_record(metrics)


@pytest.mark.parametrize(
    ('field', 'invalid_value'),
    [
        ('dota/mAP', float('nan')),
        ('dota/AP50', float('inf')),
        ('dota/mAP', 'NaN'),
        ('dota/AP50', True),
    ],
)
def test_metric_record_rejects_nonfinite_and_nonreal_values(
        tmp_path, field, invalid_value):
    diagnostics = _load_module()
    row = {'step': 24, 'dota/mAP': 0.5, 'dota/AP50': 0.5}
    row[field] = invalid_value
    metrics = tmp_path / 'invalid_metric.json'
    metrics.write_text(json.dumps(row) + '\n', encoding='utf-8')

    with pytest.raises(
            diagnostics.DiagnosticError, match='numeric|finite'):
        diagnostics.read_metric_record(metrics)


def test_reference_suffix_extraction_and_ambiguity():
    diagnostics = _load_module()
    suffix = diagnostics.REFERENCE_SUFFIX
    state_dict = {
        'model.{}'.format(suffix): torch.zeros((2, 5)),
    }

    references = diagnostics.extract_reference_points(
        {'state_dict': state_dict}, expected_queries=2, canonical=False)

    assert references.shape == (2, 5)
    assert references.dtype == np.float64
    np.testing.assert_array_equal(references, np.full((2, 5), 0.5))

    state_dict['ema.{}'.format(suffix)] = torch.zeros((2, 5))
    with pytest.raises(diagnostics.DiagnosticError, match='exactly one'):
        diagnostics.extract_reference_points(
            {'state_dict': state_dict}, expected_queries=2, canonical=False)

    with pytest.raises(diagnostics.DiagnosticError, match='exactly one'):
        diagnostics.extract_reference_points(
            {'state_dict': {'model.other': torch.zeros((2, 5))}},
            expected_queries=2,
            canonical=False)


@pytest.mark.parametrize('invalid_kind', ['nan', 'inf', 'integer', 'complex'])
def test_reference_points_reject_nonfinite_and_nonfloating_weights(
        invalid_kind):
    diagnostics = _load_module()
    if invalid_kind == 'integer':
        weight = torch.zeros((2, 5), dtype=torch.long)
    elif invalid_kind == 'complex':
        weight = torch.zeros((2, 5), dtype=torch.complex64)
    else:
        weight = torch.zeros((2, 5), dtype=torch.float32)
        weight[0, 0] = float(invalid_kind)
    checkpoint = {
        'state_dict': {
            'model.{}'.format(diagnostics.REFERENCE_SUFFIX): weight,
        },
    }

    with pytest.raises(
            diagnostics.DiagnosticError, match='finite|floating'):
        diagnostics.extract_reference_points(
            checkpoint, expected_queries=2, canonical=False)


def test_metric_parity_rejects_exact_rounded_and_training_mismatches():
    diagnostics = _load_module()
    official = diagnostics.MetricRecord(0.5, 0.5, 'official.json', 24)

    with pytest.raises(
            diagnostics.DiagnosticError,
            match='same-dump evaluator parity failed'):
        diagnostics.validate_metric_parity(0.500001, official, None)

    rounded = diagnostics.MetricRecord(0.5, 0.499, 'rounded.json', 24)
    with pytest.raises(
            diagnostics.DiagnosticError,
            match='same-dump evaluator parity failed'):
        diagnostics.validate_metric_parity(0.5, rounded, None)

    training = diagnostics.MetricRecord(0.501, 0.501, 'train.json', 24)
    with pytest.raises(
            diagnostics.DiagnosticError,
            match='training replay parity failed'):
        diagnostics.validate_metric_parity(0.5, official, training)


@pytest.mark.parametrize(
    'invalid_operand',
    [
        'reconstructed_nan',
        'reconstructed_inf',
        'official_map',
        'official_ap50',
        'training_map',
        'training_ap50',
    ],
)
def test_metric_parity_rejects_nonfinite_operands(invalid_operand):
    diagnostics = _load_module()
    reconstructed = 0.5
    official = diagnostics.MetricRecord(0.5, 0.5, 'official.json', 24)
    training = diagnostics.MetricRecord(0.5, 0.5, 'training.json', 24)
    if invalid_operand == 'reconstructed_nan':
        reconstructed = float('nan')
    elif invalid_operand == 'reconstructed_inf':
        reconstructed = float('inf')
    elif invalid_operand == 'official_map':
        official = diagnostics.MetricRecord(
            float('nan'), 0.5, 'official.json', 24)
    elif invalid_operand == 'official_ap50':
        official = diagnostics.MetricRecord(
            0.5, float('inf'), 'official.json', 24)
    elif invalid_operand == 'training_map':
        training = diagnostics.MetricRecord(
            float('nan'), 0.5, 'training.json', 24)
    elif invalid_operand == 'training_ap50':
        training = diagnostics.MetricRecord(
            0.5, float('inf'), 'training.json', 24)

    with pytest.raises(diagnostics.DiagnosticError, match='finite'):
        diagnostics.validate_metric_parity(reconstructed, official, training)


def test_prepare_records_preserves_query_rows_and_rejects_bad_gt():
    diagnostics = _load_module()
    expected_boxes = np.array(
        [[0.0, 0.0, 2.0, 3.0, 0.0],
         [1.0, 1.0, 4.0, 5.0, 0.1]],
        dtype=np.float32)
    expected_scores = np.array([0.1, 0.9], dtype=np.float32)
    expected_labels = np.array([0, 1], dtype=np.int64)
    record = _record(
        'x',
        expected_boxes,
        expected_scores,
        expected_labels,
        [[0.0, 0.0, 2.0, 3.0, 0.0]],
        [0],
    )
    box_storage = torch.empty((2, 10), dtype=torch.float32)
    noncontiguous_boxes = box_storage[:, ::2]
    noncontiguous_boxes.copy_(torch.from_numpy(expected_boxes))
    assert noncontiguous_boxes.shape == (2, 5)
    assert noncontiguous_boxes.is_contiguous() is False
    record['pred_instances']['bboxes'] = noncontiguous_boxes

    prepared = diagnostics.prepare_records(
        [record], queries_per_image=2, num_classes=2)

    assert len(prepared) == 1
    assert prepared[0].img_id == 'x'
    assert prepared[0].boxes.shape == (2, 5)
    assert prepared[0].scores.shape == (2,)
    assert prepared[0].labels.shape == (2,)
    assert prepared[0].gt_boxes.shape == (1, 5)
    assert prepared[0].gt_labels.shape == (1,)
    assert prepared[0].ignored_boxes.shape == (0, 5)
    assert prepared[0].ignored_labels.shape == (0,)
    np.testing.assert_array_equal(prepared[0].boxes, expected_boxes)
    np.testing.assert_array_equal(prepared[0].scores, expected_scores)
    np.testing.assert_array_equal(prepared[0].labels, expected_labels)
    arrays = (
        prepared[0].boxes,
        prepared[0].scores,
        prepared[0].labels,
        prepared[0].gt_boxes,
        prepared[0].gt_labels,
        prepared[0].ignored_boxes,
        prepared[0].ignored_labels,
    )
    assert all(array.flags.c_contiguous for array in arrays)

    record['gt_instances']['bboxes'][0, 2] = 0
    with pytest.raises(diagnostics.DiagnosticError, match='positive'):
        diagnostics.prepare_records(
            [record], queries_per_image=2, num_classes=2)


def test_prepare_records_rejects_duplicate_image_ids():
    diagnostics = _load_module()
    record = _record(
        'same',
        [[0, 0, 2, 2, 0], [1, 1, 2, 2, 0]],
        [0.9, 0.4],
        [0, 1],
    )

    with pytest.raises(diagnostics.DiagnosticError, match='duplicate'):
        diagnostics.prepare_records(
            [record, record], queries_per_image=2, num_classes=2)


@pytest.mark.parametrize(
    ('section', 'field', 'index'),
    [
        ('pred_instances', 'bboxes', (0, 0)),
        ('pred_instances', 'scores', (0,)),
        ('gt_instances', 'bboxes', (0, 0)),
        ('ignored_instances', 'bboxes', (0, 0)),
    ],
)
def test_prepare_records_rejects_nonfinite_data(
        section, field, index):
    diagnostics = _load_module()
    record = _record(
        'nonfinite',
        [[0, 0, 2, 2, 0], [1, 1, 2, 2, 0]],
        [0.9, 0.4],
        [0, 1],
        [[0, 0, 2, 2, 0]],
        [0],
        [[2, 2, 2, 2, 0]],
        [1],
    )
    record[section][field][index] = float('nan')

    with pytest.raises(diagnostics.DiagnosticError, match='finite'):
        diagnostics.prepare_records(
            [record], queries_per_image=2, num_classes=2)


@pytest.mark.parametrize(
    'invalid_scores',
    [
        torch.tensor([True, False], dtype=torch.bool),
        torch.tensor([.9 + 0j, .2 + 0j], dtype=torch.complex64),
        np.array([object(), object()], dtype=object),
    ],
)
@pytest.mark.parametrize(
    'prepare_name',
    ['prepare_records', 'prepare_records_variable_queries'],
)
def test_prepare_record_variants_reject_nonreal_score_dtypes(
        invalid_scores, prepare_name):
    diagnostics = _load_module()
    record = _record(
        'invalid-scores',
        [[10, 10, 4, 4, 0], [20, 20, 4, 4, 0]],
        [.9, .2],
        [0, 0],
    )
    record['pred_instances']['scores'] = invalid_scores
    kwargs = {'num_classes': 1}
    if prepare_name == 'prepare_records':
        kwargs['queries_per_image'] = 2

    with pytest.raises(diagnostics.DiagnosticError):
        getattr(diagnostics, prepare_name)([record], **kwargs)


@pytest.mark.parametrize(
    'dtype', [torch.float16, torch.float32, torch.float64])
def test_prepare_records_accepts_real_floating_score_dtypes(dtype):
    diagnostics = _load_module()
    record = _record(
        'floating-scores',
        [[10, 10, 4, 4, 0], [20, 20, 4, 4, 0]],
        [.9, .2],
        [0, 0],
    )
    record['pred_instances']['scores'] = torch.tensor(
        [.9, .2], dtype=dtype)

    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=1)

    assert prepared[0].scores.dtype == torch.empty(
        (), dtype=dtype).numpy().dtype


@pytest.mark.parametrize('section', [
    'pred_instances', 'gt_instances', 'ignored_instances'])
def test_prepare_records_rejects_invalid_labels(section):
    diagnostics = _load_module()
    record = _record(
        'invalid-label',
        [[0, 0, 2, 2, 0], [1, 1, 2, 2, 0]],
        [0.9, 0.4],
        [0, 1],
        [[0, 0, 2, 2, 0]],
        [0],
        [[2, 2, 2, 2, 0]],
        [1],
    )
    if section == 'pred_instances':
        record[section]['labels'] = torch.tensor([0.5, 1.0])
    elif section == 'gt_instances':
        record[section]['labels'] = torch.tensor([2])
    else:
        record[section]['labels'] = torch.tensor([-1])

    with pytest.raises(
            diagnostics.DiagnosticError, match=r'integer|\[0'):
        diagnostics.prepare_records(
            [record], queries_per_image=2, num_classes=2)


def test_prepare_records_rejects_wrong_query_count():
    diagnostics = _load_module()
    record = _record(
        'short', [[0, 0, 2, 2, 0]], [0.9], [0])

    with pytest.raises(diagnostics.DiagnosticError, match='count'):
        diagnostics.prepare_records(
            [record], queries_per_image=2, num_classes=2)


def test_evaluator_preserves_query_ids_and_classifies_all_six_outcomes():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    far = [40, 40, 4, 4, 0]
    records = [
        _record(
            'filled',
            [square, square, far, square],
            [0.9, 0.8, 0.7, 0.6],
            [0, 0, 0, 1],
            [square],
            [0],
        ),
        _record('empty', [square], [0.95], [0]),
        _record(
            'ignored',
            [square],
            [0.99],
            [0],
            ignored_boxes=[square],
            ignored_labels=[0],
        ),
    ]

    prepared = diagnostics.prepare_records_variable_queries(
        records, num_classes=2)
    evaluation = diagnostics.evaluate_records(
        prepared, num_classes=2, iou_threshold=0.5)

    assert evaluation.mean_ap == pytest.approx(0.5)
    class_zero = evaluation.class_results[0]
    assert class_zero.last_tp_end == 3
    assert class_zero.ap_support_end == 3
    assert class_zero.oracle_ap == pytest.approx(1.0)
    np.testing.assert_array_equal(
        class_zero.query_ids,
        np.array([0, 0, 0, 1, 2]),
    )
    np.testing.assert_allclose(
        class_zero.scores,
        np.array([0.99, 0.95, 0.9, 0.8, 0.7]),
    )
    np.testing.assert_array_equal(
        np.concatenate(evaluation.row_outcomes),
        np.array([
            'tp',
            'duplicate_fp',
            'localization_background_fp',
            'semantic_fp',
            'empty_tile_fp',
            'ignored_prediction',
        ]),
    )
    assert sum(row.sum() for row in evaluation.row_tp) == 1
    assert sum(row.sum() for row in evaluation.row_fp) == 4
    np.testing.assert_allclose(
        np.concatenate(evaluation.row_same_label_iou),
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.concatenate(evaluation.row_assigned_iou),
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.concatenate(evaluation.row_matched_gt),
        [0, 0, -1, -1, -1, -1],
    )


def test_voc07_ap_wrapper_uses_mmdet_average_precision():
    diagnostics = _load_module()
    recall = np.array([0.0, 0.5, 0.5, 1.0])
    precision = np.array([0.0, 0.5, 1.0 / 3.0, 0.5])
    expected = diagnostics.average_precision(
        recall, precision, mode='11points')

    assert diagnostics._voc07_ap(recall, precision) == pytest.approx(expected)


def test_voc07_support_ranks_choose_earliest_precision_tie():
    diagnostics = _load_module()
    recall = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
    precision = np.array([0.0, 0.6, 0.6, 0.4, 0.4])

    support_ranks = diagnostics.voc07_support_ranks(recall, precision)

    assert len(support_ranks) == 11
    assert support_ranks[:6] == [1, 1, 1, 1, 1, 1]
    assert support_ranks[6:] == [3, 3, 3, 3, 3]
    assert max(rank for rank in support_ranks if rank is not None) + 1 == 4

    partial_support = diagnostics.voc07_support_ranks(
        np.array([0.0, 0.5]), np.array([0.8, 0.6]))
    assert partial_support[:6] == [0, 1, 1, 1, 1, 1]
    assert partial_support[6:] == [None] * 5


@pytest.mark.parametrize(
    ('recall', 'precision'),
    [
        (np.array([0.0, 0.5], dtype=np.complex64), [0.8, 0.6]),
        ([0.0, 0.5], np.array([0.8, 0.6], dtype=np.complex64)),
        ([-0.1, 0.5], [0.8, 0.6]),
        ([0.0, 1.1], [0.8, 0.6]),
        ([0.0, 0.5], [-0.1, 0.6]),
        ([0.0, 0.5], [0.8, 1.1]),
        ([0.5, 0.4], [0.8, 0.6]),
    ],
)
def test_voc07_support_rejects_nonreal_unbounded_or_decreasing_curves(
        recall, precision):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.voc07_support_ranks(recall, precision)


def test_perfect_ranking_oracle_keeps_maximum_recall():
    diagnostics = _load_module()
    tp = np.array([0, 1, 0, 0], dtype=np.float32)
    fp = np.array([1, 0, 1, 1], dtype=np.float32)

    oracle_ap = diagnostics.perfect_ranking_ap(tp, fp, num_gts=2)

    assert oracle_ap == pytest.approx(6.0 / 11.0)
    assert tp.sum() / 2 == pytest.approx(0.5)

    ignored_tp = np.array([0, 1, 0, 0, 0], dtype=np.float32)
    ignored_fp = np.array([1, 0, 0, 1, 1], dtype=np.float32)
    original_tp = ignored_tp.copy()
    original_fp = ignored_fp.copy()
    ignored_oracle_ap = diagnostics.perfect_ranking_ap(
        ignored_tp, ignored_fp, num_gts=2)

    assert ignored_oracle_ap == pytest.approx(6.0 / 11.0)
    assert ignored_tp.sum() / 2 == pytest.approx(0.5)
    np.testing.assert_array_equal(ignored_tp, original_tp)
    np.testing.assert_array_equal(ignored_fp, original_fp)


def test_perfect_ranking_rows_preserve_ignored_pairs():
    diagnostics = _load_module()
    tp = np.array([0, 1, 0, 0, 0], dtype=np.float32)
    fp = np.array([1, 0, 0, 1, 0], dtype=np.float32)

    ranked_tp, ranked_fp = diagnostics._perfect_ranking_rows(tp, fp)

    np.testing.assert_array_equal(ranked_tp, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(ranked_fp, [0, 0, 0, 1, 1])
    assert list(zip(ranked_tp, ranked_fp)).count((0, 0)) == 2


@pytest.mark.parametrize(
    ('tp', 'fp'),
    [
        (np.array([0, 1], dtype=np.complex64), np.array([1, 0])),
        (np.array([0, 1]), np.array([1, 0], dtype=np.complex64)),
    ],
)
def test_perfect_ranking_rejects_complex_rows(tp, fp):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.perfect_ranking_ap(tp, fp, num_gts=1)


def test_gt_decomposition_has_geometry_semantic_ownership_and_reachable():
    diagnostics = _load_module()
    gt_boxes = [
        [10, 10, 4, 4, 0],
        [30, 30, 4, 4, 0],
        [50, 50, 4, 4, 0],
        [52, 50, 4, 4, 0],
    ]
    record = _record(
        'states',
        [[90, 90, 4, 4, 0],
         [30, 30, 4, 4, 0],
         [51, 50, 6, 4, 0]],
        [0.2, 0.8, 0.9],
        [0, 1, 0],
        gt_boxes,
        [0, 0, 0, 0],
    )
    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=2)

    evidence = diagnostics.decompose_ground_truth(
        prepared, iou_threshold=0.5)

    assert [row.state for row in evidence] == [
        'geometry_miss',
        'semantic_miss',
        'evaluator_reachable',
        'ownership_miss',
    ]
    assert [row.gt_index for row in evidence] == [0, 1, 2, 3]
    assert all(row.image_gt_count == 4 for row in evidence)
    assert evidence[2].witness_query == 2
    assert evidence[2].witness_score == pytest.approx(0.9)
    assert evidence[3].candidate_count == 1
    assert evidence[3].witness_query is None


def test_gt_ownership_ignores_better_overlap_with_different_class():
    diagnostics = _load_module()
    target = [10, 10, 4, 4, 0]
    other_class = [11, 10, 4, 4, 0]
    record = _record(
        'class-scoped-ownership',
        [other_class],
        [0.9],
        [0],
        [target, other_class],
        [0, 1],
    )
    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=2)

    evidence = diagnostics.decompose_ground_truth(
        prepared, iou_threshold=0.5)

    assert evidence[0].state == 'evaluator_reachable'
    assert evidence[0].candidate_count == 1
    assert evidence[0].witness_query == 0
    assert evidence[1].state == 'semantic_miss'


def test_gt_decomposition_empty_predictions_is_geometry_miss():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    record = _record(
        'empty-predictions', [], [], [], [square], [0])
    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=1)

    evidence = diagnostics.decompose_ground_truth(
        prepared, iou_threshold=0.5)

    assert len(evidence) == 1
    assert evidence[0].state == 'geometry_miss'
    assert evidence[0].best_any_query is None
    assert evidence[0].best_any_iou == pytest.approx(0.0)


def test_gt_decomposition_equal_score_owners_choose_smallest_query():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    record = _record(
        'owner-score-tie',
        [square, square],
        [0.9, 0.9],
        [0, 0],
        [square],
        [0],
    )
    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=1)

    evidence = diagnostics.decompose_ground_truth(
        prepared, iou_threshold=0.5)

    assert evidence[0].state == 'evaluator_reachable'
    assert evidence[0].candidate_count == 2
    assert evidence[0].witness_query == 0


def test_summarize_gt_evidence_reports_candidate_excess_and_witnesses():
    diagnostics = _load_module()
    states = [
        'geometry_miss',
        'semantic_miss',
        'evaluator_reachable',
        'ownership_miss',
    ]
    candidate_counts = [0, 0, 3, 1]
    witness_queries = [None, None, 5, None]
    evidence = [
        diagnostics.GtEvidence(
            img_id='summary',
            gt_index=index,
            class_id=0,
            state=state,
            image_gt_count=4,
            gt_box=(float(index), 0.0, 4.0, 4.0, 0.0),
            best_any_query=None,
            best_any_iou=0.0,
            best_same_query=None,
            best_same_iou=0.0,
            candidate_count=candidate_count,
            witness_query=witness_query,
            witness_score=None if witness_query is None else 0.9,
            witness_iou=None if witness_query is None else 1.0,
        )
        for index, (state, candidate_count, witness_query) in enumerate(
            zip(states, candidate_counts, witness_queries))
    ]
    evidence.append(diagnostics.GtEvidence(
        img_id='summary-2',
        gt_index=0,
        class_id=1,
        state='evaluator_reachable',
        image_gt_count=1,
        gt_box=(0.0, 0.0, 4.0, 4.0, 0.0),
        best_any_query=2,
        best_any_iou=1.0,
        best_same_query=2,
        best_same_iou=1.0,
        candidate_count=1,
        witness_query=2,
        witness_score=0.8,
        witness_iou=1.0,
    ))

    summary = diagnostics.summarize_gt_evidence(evidence)

    assert summary == {
        'total_gt': 5,
        'geometry_miss': 1,
        'semantic_miss': 1,
        'evaluator_reachable': 2,
        'ownership_miss': 1,
        'candidate_excess': 2,
        'witness_query_ids': [2, 5],
        'witness_query_count': 2,
    }


@pytest.mark.parametrize(
    'changes',
    [
        {
            'state': 'geometry_miss',
            'candidate_count': 1,
            'witness_query': None,
            'witness_score': None,
            'witness_iou': None,
        },
        {
            'state': 'semantic_miss',
            'candidate_count': 0,
            'witness_query': 0,
            'witness_score': None,
            'witness_iou': None,
        },
        {
            'state': 'ownership_miss',
            'candidate_count': 0,
            'witness_query': None,
            'witness_score': None,
            'witness_iou': None,
        },
        {
            'state': 'ownership_miss',
            'candidate_count': 1,
            'witness_query': None,
            'witness_score': 0.5,
            'witness_iou': None,
        },
        {'candidate_count': 0},
        {'witness_query': None},
        {'witness_score': None},
        {'witness_iou': None},
        {'witness_score': float('nan')},
        {'witness_iou': float('inf')},
        {'witness_query': -1},
        {'witness_query': True},
        {'witness_query': 1.5},
        {'witness_score': True},
        {'witness_score': 0.5 + 0.0j},
        {'witness_iou': True},
        {'witness_iou': 0.5 + 0.0j},
        {'witness_iou': -0.1},
        {'witness_iou': 1.1},
    ],
)
def test_summarize_gt_evidence_rejects_impossible_state_fields(changes):
    diagnostics = _load_module()
    valid = diagnostics.GtEvidence(
        img_id='valid',
        gt_index=0,
        class_id=0,
        state='evaluator_reachable',
        image_gt_count=1,
        gt_box=(0.0, 0.0, 4.0, 4.0, 0.0),
        best_any_query=0,
        best_any_iou=1.0,
        best_same_query=0,
        best_same_iou=1.0,
        candidate_count=1,
        witness_query=0,
        witness_score=0.5,
        witness_iou=1.0,
    )

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.summarize_gt_evidence([replace(valid, **changes)])


def test_capacity_separates_lower_bound_from_observed_misses():
    diagnostics = _load_module()

    summary = diagnostics.summarize_capacity(
        image_gt_counts=[1, 5],
        class_image_gt_counts={0: [1, 4], 1: [0, 1]},
        query_count=2,
    )

    assert summary['query_count'] == 2
    assert summary['total_gt'] == 6
    assert summary['capacity_excess'] == 3
    assert summary['optimistic_recall_ceiling'] == pytest.approx(0.5)
    assert list(summary['per_class']) == [0, 1]
    assert summary['per_class'][0]['total_gt'] == 5
    assert summary['per_class'][0]['capacity_excess'] == 2
    assert summary['per_class'][0][
        'optimistic_recall_ceiling'] == pytest.approx(0.6)
    assert summary['per_class'][1]['total_gt'] == 1
    assert summary['per_class'][1]['capacity_excess'] == 0
    assert summary['per_class'][1][
        'optimistic_recall_ceiling'] == pytest.approx(1.0)

    empty_summary = diagnostics.summarize_capacity(
        image_gt_counts=[0, 0],
        class_image_gt_counts={},
        query_count=2,
    )
    assert empty_summary['total_gt'] == 0
    assert empty_summary['optimistic_recall_ceiling'] is None
    assert empty_summary['per_class'] == {}


def test_capacity_rejects_invalid_counts_and_query_count():
    diagnostics = _load_module()
    invalid_inputs = [
        ([1], {0: [1]}, True),
        ([1], {0: [1]}, 0),
        ([1.5], {0: [1]}, 2),
        ([1], {0: [-1]}, 2),
        ([1, 2], {0: [1]}, 2),
    ]

    for image_counts, class_counts, query_count in invalid_inputs:
        with pytest.raises(diagnostics.DiagnosticError):
            diagnostics.summarize_capacity(
                image_gt_counts=image_counts,
                class_image_gt_counts=class_counts,
                query_count=query_count,
            )


@pytest.mark.parametrize(
    ('image_gt_counts', 'class_image_gt_counts'),
    [
        ([1], {0: [2]}),
        ([2, 1], {0: [1, 1]}),
    ],
)
def test_capacity_rejects_inconsistent_class_partition(
        image_gt_counts, class_image_gt_counts):
    diagnostics = _load_module()

    with pytest.raises(
            diagnostics.DiagnosticError, match='partition|total'):
        diagnostics.summarize_capacity(
            image_gt_counts=image_gt_counts,
            class_image_gt_counts=class_image_gt_counts,
            query_count=2,
        )


def test_all_empty_evaluation_uses_numpy_argsort_tie_policy():
    diagnostics = _load_module()
    scores = np.full(20, 0.5, dtype=np.float32)
    boxes = np.column_stack((
        np.arange(20),
        np.zeros(20),
        np.full(20, 2),
        np.full(20, 2),
        np.zeros(20),
    ))
    record = _record('empty', boxes, scores, np.zeros(20, dtype=np.int64))

    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=2)
    evaluation = diagnostics.evaluate_records(
        prepared, num_classes=2, iou_threshold=0.5)

    expected_order = np.argsort(-scores)
    assert not np.array_equal(expected_order, np.arange(len(scores)))
    np.testing.assert_array_equal(
        evaluation.class_results[0].query_ids, expected_order)
    zero_gt_with_rows = evaluation.class_results[0]
    zero_gt_without_rows = evaluation.class_results[1]
    assert zero_gt_with_rows.ap == pytest.approx(0.0)
    assert zero_gt_with_rows.oracle_ap == pytest.approx(0.0)
    assert zero_gt_with_rows.ap_support_end == 1
    assert zero_gt_without_rows.ap == pytest.approx(0.0)
    assert zero_gt_without_rows.oracle_ap == pytest.approx(0.0)
    assert zero_gt_without_rows.ap_support_end == 0
    assert evaluation.mean_ap == pytest.approx(0.0)


def test_image_local_greedy_matching_uses_numpy_argsort_tie_policy():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    scores = np.array([0.1] + [0.5] * 20, dtype=np.float32)
    default_order = np.argsort(-scores)
    stable_order = np.argsort(-scores, kind='stable')
    default_first = int(default_order[0])
    stable_first = int(stable_order[0])
    assert default_first != stable_first
    record = _record(
        'tied',
        [square] * len(scores),
        scores,
        np.zeros(len(scores), dtype=np.int64),
        [square],
        [0],
    )

    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=1)
    evaluation = diagnostics.evaluate_records(
        prepared, num_classes=1, iou_threshold=0.5)

    assert evaluation.row_tp[0][default_first] == 1
    assert evaluation.row_tp[0][stable_first] == 0
    other_max_score_rows = np.flatnonzero(scores == scores.max())
    other_max_score_rows = other_max_score_rows[
        other_max_score_rows != default_first]
    assert all(
        evaluation.row_outcomes[0][query_id] == 'duplicate_fp'
        for query_id in other_max_score_rows)


def test_mixed_ordinary_and_ignored_gt_matches_official_assignment():
    diagnostics = _load_module()
    ordinary = [10, 10, 4, 4, 0]
    ignored = [11, 10, 4, 4, 0]
    pred_boxes = [ignored, ordinary, ignored]
    scores = [0.9, 0.8, 0.7]
    record = _record(
        'mixed-gt',
        pred_boxes,
        scores,
        [0, 0, 0],
        [ordinary],
        [0],
        [ignored],
        [0],
    )

    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=1)
    evaluation = diagnostics.evaluate_records(
        prepared, num_classes=1, iou_threshold=0.5)
    official_tp, official_fp = tpfp_default(
        np.column_stack((
            np.asarray(pred_boxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
        )),
        np.asarray([ordinary], dtype=np.float32),
        np.asarray([ignored], dtype=np.float32),
        iou_thr=0.5,
    )

    np.testing.assert_array_equal(
        evaluation.row_outcomes[0],
        np.array([
            'ignored_prediction',
            'tp',
            'ignored_prediction',
        ]),
    )
    np.testing.assert_array_equal(evaluation.row_tp[0], official_tp[0])
    np.testing.assert_array_equal(evaluation.row_fp[0], official_fp[0])
    np.testing.assert_array_equal(evaluation.row_tp[0], [0, 1, 0])
    np.testing.assert_array_equal(evaluation.row_fp[0], [0, 0, 0])
    np.testing.assert_allclose(
        evaluation.row_same_label_iou[0], [0.6, 1.0, 0.6], atol=1e-6)
    np.testing.assert_allclose(
        evaluation.row_assigned_iou[0], [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(
        evaluation.row_matched_gt[0], [-1, 0, -1])


def test_ordinary_gt_wins_equal_iou_tie_against_ignored_gt():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    record = _record(
        'gt-tie',
        [square],
        [0.9],
        [0],
        [square],
        [0],
        [square],
        [0],
    )

    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=1)
    evaluation = diagnostics.evaluate_records(
        prepared, num_classes=1, iou_threshold=0.5)

    assert evaluation.row_outcomes[0].tolist() == ['tp']
    np.testing.assert_array_equal(evaluation.row_tp[0], [1])
    np.testing.assert_array_equal(evaluation.row_fp[0], [0])
    np.testing.assert_allclose(evaluation.row_same_label_iou[0], [1.0])
    np.testing.assert_allclose(evaluation.row_assigned_iou[0], [1.0])
    np.testing.assert_array_equal(evaluation.row_matched_gt[0], [0])


def test_matched_gt_keeps_original_index_across_class_splitting():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    far = [40, 40, 4, 4, 0]
    record = _record(
        'interleaved-gt',
        [square],
        [0.9],
        [0],
        [far, square],
        [1, 0],
    )

    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=2)
    evaluation = diagnostics.evaluate_records(
        prepared, num_classes=2, iou_threshold=0.5)

    assert evaluation.row_outcomes[0].tolist() == ['tp']
    np.testing.assert_allclose(evaluation.row_same_label_iou[0], [1.0])
    np.testing.assert_allclose(evaluation.row_assigned_iou[0], [1.0])
    np.testing.assert_array_equal(evaluation.row_matched_gt[0], [1])


def test_evaluation_preserves_tuple_image_ids_as_scalar_objects():
    diagnostics = _load_module()
    image_id = ('tile', 'crop')
    square = [10, 10, 4, 4, 0]
    far = [40, 40, 4, 4, 0]
    record = _record(
        image_id,
        [square, far],
        [0.9, 0.8],
        [0, 1],
        [square, far],
        [0, 1],
    )

    prepared = diagnostics.prepare_records_variable_queries(
        [record], num_classes=2)
    evaluation = diagnostics.evaluate_records(
        prepared, num_classes=2, iou_threshold=0.5)

    for class_result in evaluation.class_results:
        assert class_result.image_ids.shape == (1,)
        assert isinstance(class_result.image_ids[0], tuple)
        assert class_result.image_ids[0] == image_id


def test_stable_sample_keys_uses_order_independent_sha256_ranking():
    diagnostics = _load_module()
    keys = [('b', 1), ('a', 0), ('c', 2), ('d', 3)]

    expected = sorted(
        keys,
        key=lambda key: (
            hashlib.sha256(
                '{}\0{}\0{}'.format(20260722, key[0], key[1]).encode(
                    'utf-8')).hexdigest(),
            str(key[0]),
            key[1],
        ),
    )
    forward = diagnostics.stable_sample_keys(
        (key for key in keys), sample_size=2, seed=20260722)
    backward = diagnostics.stable_sample_keys(
        (key for key in reversed(keys)), sample_size=2, seed=20260722)

    assert forward == expected[:2]
    assert backward == expected[:2]
    assert diagnostics.stable_sample_keys(
        iter(keys), sample_size=0, seed=20260722) == []
    assert diagnostics.stable_sample_keys(
        (key for key in reversed(keys)), sample_size=10,
        seed=20260722) == expected


def test_stable_sample_keys_breaks_same_text_ties_by_typed_id():
    diagnostics = _load_module()
    integer_and_string = [(1, 0), ('1', 0)]
    tuple_and_string = [
        (('tile', 1), 0),
        ("('tile', 1)", 0),
    ]

    assert diagnostics.stable_sample_keys(
        integer_and_string, sample_size=1, seed=7) == [(1, 0)]
    assert diagnostics.stable_sample_keys(
        reversed(integer_and_string), sample_size=1, seed=7) == [(1, 0)]
    assert diagnostics.stable_sample_keys(
        tuple_and_string, sample_size=1, seed=7) == [("('tile', 1)", 0)]
    assert diagnostics.stable_sample_keys(
        reversed(tuple_and_string), sample_size=1,
        seed=7) == [("('tile', 1)", 0)]


def test_stable_sample_keys_rejects_unstable_custom_image_ids():
    diagnostics = _load_module()

    class CustomImageId:
        def __str__(self):
            return 'tile'

    with pytest.raises(diagnostics.DiagnosticError, match='image ID|supported'):
        diagnostics.stable_sample_keys(
            [(CustomImageId(), 0)], sample_size=1, seed=7)


def test_safe_spearman_uses_average_tied_ranks_and_handles_constants():
    diagnostics = _load_module()

    assert diagnostics.safe_spearman([1, 1], [2, 3]) is None
    assert diagnostics.safe_spearman(
        [1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    expected = np.corrcoef(
        np.array([1.0, 2.5, 2.5, 4.0]),
        np.array([4.0, 1.0, 2.0, 3.0]),
    )[0, 1]
    assert diagnostics.safe_spearman(
        [1, 2, 2, 4], [4, 1, 2, 3]) == pytest.approx(expected)


@pytest.mark.parametrize(
    ('lhs', 'rhs'),
    [
        ([1, 2], [1]),
        ([[1, 2]], [[1, 2]]),
        ([1, float('nan')], [1, 2]),
        ([1, float('inf')], [1, 2]),
        (np.array([1, 2], dtype=np.complex64), [1, 2]),
        ([1, 2], np.array([1, 2], dtype=np.complex64)),
        (['one', 'two'], [1, 2]),
    ],
)
def test_safe_spearman_rejects_invalid_inputs(lhs, rhs):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.safe_spearman(lhs, rhs)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (8, '<=8'),
        (8.0001, '(8,16]'),
        (16, '(8,16]'),
        (32, '(16,32]'),
        (64, '(32,64]'),
        (64.1, '>64'),
    ],
)
def test_size_bin_boundaries(value, expected):
    diagnostics = _load_module()
    assert diagnostics.size_bin(value) == expected


@pytest.mark.parametrize(
    ('count', 'expected'),
    [
        (1, '1-10'),
        (10, '1-10'),
        (11, '11-50'),
        (50, '11-50'),
        (51, '51-100'),
        (101, '101-200'),
        (201, '201-400'),
        (401, '401-600'),
        (601, '>600'),
    ],
)
def test_density_bin_boundaries(count, expected):
    diagnostics = _load_module()
    assert diagnostics.density_bin(count) == expected


@pytest.mark.parametrize(
    ('ratio', 'expected'),
    [
        (1.5, '<=1.5'),
        (1.5001, '(1.5,3]'),
        (3, '(1.5,3]'),
        (5, '(3,5]'),
        (5.1, '>5'),
    ],
)
def test_aspect_bin_boundaries(ratio, expected):
    diagnostics = _load_module()
    assert diagnostics.aspect_bin(ratio) == expected


@pytest.mark.parametrize(
    ('radians', 'expected'),
    [
        (0, '[0,15)'),
        (np.deg2rad(14.999), '[0,15)'),
        (np.deg2rad(15), '[15,30)'),
        (np.deg2rad(45), '[45,60)'),
        (np.deg2rad(75), '[75,90]'),
        (np.deg2rad(90), '[75,90]'),
        (-np.deg2rad(14.999), '[0,15)'),
        (np.pi + np.deg2rad(45), '[45,60)'),
    ],
)
def test_angle_bin_boundaries_and_periodic_normalization(radians, expected):
    diagnostics = _load_module()
    assert diagnostics.angle_bin(radians) == expected


@pytest.mark.parametrize(
    ('degrees', 'expected'),
    [
        (0, '[0,15)'),
        (15, '[15,30)'),
        (30, '[30,45)'),
        (45, '[45,60)'),
        (60, '[60,75)'),
        (75, '[75,90]'),
        (90, '[75,90]'),
    ],
)
def test_angle_bin_exact_boundaries_are_periodic(degrees, expected):
    diagnostics = _load_module()
    theta = np.deg2rad(degrees)
    equivalent_angles = (
        theta,
        -theta,
        2 * np.pi + theta,
        2 * np.pi - theta,
        np.pi + theta,
    )

    for angle in equivalent_angles:
        assert diagnostics.angle_bin(angle) == expected


@pytest.mark.parametrize(
    ('score', 'expected'),
    [
        (0, '[0,.05)'),
        (.0499, '[0,.05)'),
        (.05, '[.05,.10)'),
        (.10, '[.10,.25)'),
        (.25, '[.25,.50)'),
        (.50, '[.50,.75)'),
        (.75, '[.75,1]'),
        (1., '[.75,1]'),
    ],
)
def test_score_bin_boundaries(score, expected):
    diagnostics = _load_module()
    assert diagnostics.score_bin(score) == expected


@pytest.mark.parametrize(
    ('function_name', 'value'),
    [
        ('size_bin', 0),
        ('size_bin', float('nan')),
        ('density_bin', 0),
        ('density_bin', 1.5),
        ('aspect_bin', .9),
        ('aspect_bin', complex(2, 0)),
        ('angle_bin', float('inf')),
        ('angle_bin', complex(0, 0)),
        ('score_bin', -.01),
        ('score_bin', 1.01),
        ('score_bin', True),
    ],
)
def test_fixed_bins_reject_invalid_types_and_bounds(function_name, value):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        getattr(diagnostics, function_name)(value)


def test_effective_query_count_handles_support_and_zero_mass():
    diagnostics = _load_module()

    assert diagnostics.effective_query_count(
        np.array([1.0, 1.0, 0.0])) == pytest.approx(2.0)
    assert diagnostics.effective_query_count(np.zeros(4)) is None


@pytest.mark.parametrize(
    'counts',
    [
        np.array([1.0, -1.0]),
        np.array([1.0, float('nan')]),
        np.array([1.0, float('inf')]),
        np.array([1.0, 2.0], dtype=np.complex64),
        np.array([[1.0, 2.0]]),
        [True, False],
    ],
)
def test_effective_query_count_rejects_invalid_contributions(counts):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.effective_query_count(counts)


def test_refinement_values_use_symmetric_box_changes():
    diagnostics = _load_module()
    prior = np.array([.5, .5, .25, .25, 0.0])
    final = np.array([520.0, 512.0, 512.0, 256.0, np.pi - .1])

    values = diagnostics.refinement_values(prior, final, patch_size=1024)

    assert values == pytest.approx({
        'center_distance': 8.0,
        'scale_change': np.sqrt(2.0),
        'aspect_change': 2.0,
        'angle_change': .1,
    })


def _aligned_refinement_fixture(diagnostics):
    final = [20.0, 20.0, 4.0, 4.0, 0.0]
    prepared = diagnostics.prepare_records_variable_queries(
        [_record(
            'aligned', [final], [.9], [0],
            gt_boxes=[final], gt_labels=[0])],
        num_classes=2,
    )
    record = prepared[0]
    evidence = diagnostics.GtEvidence(
        img_id=record.img_id, gt_index=0, class_id=0,
        state='evaluator_reachable', image_gt_count=1,
        gt_box=tuple(float(value) for value in record.gt_boxes[0]),
        best_any_query=0, best_any_iou=1.0,
        best_same_query=0, best_same_iou=1.0, candidate_count=1,
        witness_query=0, witness_score=float(record.scores[0]),
        witness_iou=1.0)
    references = np.array([[.5, .5, .25, .25, 0.0]])
    return record, evidence, references


@pytest.mark.parametrize(
    'prior',
    [
        [-.1, .5, .25, .25, 0.0],
        [.5, .5, .25, .25, 1.1],
    ],
)
def test_refinement_values_rejects_out_of_range_normalized_prior(prior):
    diagnostics = _load_module()

    with pytest.raises(
            diagnostics.DiagnosticError, match=r'normalized|\[0, 1\]'):
        diagnostics.refinement_values(
            prior, [20.0, 20.0, 4.0, 4.0, 0.0])


@pytest.mark.parametrize(
    ('component', 'value'),
    [
        (0, -.1),
        (4, 1.1),
    ],
)
def test_attach_refinement_rejects_out_of_range_reference_matrix(
        component, value):
    diagnostics = _load_module()
    record, evidence, references = _aligned_refinement_fixture(diagnostics)
    references[0, component] = value

    with pytest.raises(
            diagnostics.DiagnosticError, match=r'normalized|\[0, 1\]'):
        diagnostics.attach_refinement([evidence], [record], references)


@pytest.mark.parametrize(
    ('changes', 'match'),
    [
        ({'gt_index': -1}, 'GT index'),
        ({'gt_index': 1}, 'GT index'),
        ({'image_gt_count': 2}, 'GT count'),
        ({'class_id': 1}, 'class'),
        ({'gt_box': (21.0, 20.0, 4.0, 4.0, 0.0)}, 'GT box'),
        ({'witness_score': .1}, 'witness score'),
    ],
)
def test_attach_refinement_rejects_misaligned_evidence(changes, match):
    diagnostics = _load_module()
    record, evidence, references = _aligned_refinement_fixture(diagnostics)

    with pytest.raises(diagnostics.DiagnosticError, match=match):
        diagnostics.attach_refinement(
            [replace(evidence, **changes)], [record], references)


def test_attach_refinement_rejects_witness_label_mismatch():
    diagnostics = _load_module()
    record, evidence, references = _aligned_refinement_fixture(diagnostics)
    mismatched_record = replace(record, labels=np.array([1], dtype=np.int64))

    with pytest.raises(diagnostics.DiagnosticError, match='witness.*class|label'):
        diagnostics.attach_refinement(
            [evidence], [mismatched_record], references)


def test_attach_refinement_updates_only_reachable_witnesses():
    diagnostics = _load_module()
    final = [520.0, 512.0, 512.0, 256.0, np.pi - .1]
    prepared = diagnostics.prepare_records_variable_queries(
        [_record(
            'tile',
            [[20.0, 20.0, 4.0, 4.0, 0.0], final],
            [.2, .9],
            [0, 0],
            gt_boxes=[[20.0, 20.0, 4.0, 4.0, 0.0], final],
            gt_labels=[0, 0],
        )],
        num_classes=1,
    )
    witness = diagnostics.GtEvidence(
        img_id='tile', gt_index=1, class_id=0,
        state='evaluator_reachable', image_gt_count=2,
        gt_box=tuple(final), best_any_query=1, best_any_iou=1.0,
        best_same_query=1, best_same_iou=1.0, candidate_count=1,
        witness_query=1, witness_score=.9, witness_iou=1.0)
    non_witness = diagnostics.GtEvidence(
        img_id='tile', gt_index=0, class_id=0,
        state='geometry_miss', image_gt_count=2,
        gt_box=(20.0, 20.0, 4.0, 4.0, 0.0),
        best_any_query=0, best_any_iou=.1,
        best_same_query=0, best_same_iou=.1, candidate_count=0,
        witness_query=None, witness_score=None, witness_iou=None)
    references = np.array([
        [.1, .1, .1, .1, .5],
        [.5, .5, .25, .25, 0.0],
    ])

    attached = diagnostics.attach_refinement(
        [witness, non_witness], prepared, references, patch_size=1024)

    assert [row.gt_index for row in attached] == [1, 0]
    assert attached[0] is not witness
    assert attached[0].prior_center_distance == pytest.approx(8.0)
    assert attached[0].prior_scale_change == pytest.approx(np.sqrt(2.0))
    assert attached[0].prior_aspect_change == pytest.approx(2.0)
    assert attached[0].prior_angle_change == pytest.approx(.1)
    assert attached[1] is non_witness
    assert attached[1].prior_center_distance is None
    assert attached[1].prior_scale_change is None
    assert attached[1].prior_aspect_change is None
    assert attached[1].prior_angle_change is None


def test_attach_refinement_rejects_missing_rows_and_invalid_dimensions():
    diagnostics = _load_module()
    prepared = diagnostics.prepare_records_variable_queries(
        [_record(
            'tile',
            [[20.0, 20.0, 4.0, 4.0, 0.0]],
            [.9],
            [0],
            gt_boxes=[[20.0, 20.0, 4.0, 4.0, 0.0]],
            gt_labels=[0],
        )],
        num_classes=1,
    )
    witness = diagnostics.GtEvidence(
        img_id='tile', gt_index=0, class_id=0,
        state='evaluator_reachable', image_gt_count=1,
        gt_box=(20.0, 20.0, 4.0, 4.0, 0.0),
        best_any_query=0, best_any_iou=1.0,
        best_same_query=0, best_same_iou=1.0, candidate_count=1,
        witness_query=0, witness_score=.9, witness_iou=1.0)
    references = np.array([[.5, .5, .25, .25, 0.0]])

    with pytest.raises(diagnostics.DiagnosticError, match='record|image'):
        diagnostics.attach_refinement(
            [replace(witness, img_id='missing')], prepared, references)
    with pytest.raises(diagnostics.DiagnosticError, match='record|query'):
        diagnostics.attach_refinement(
            [replace(witness, witness_query=1)], prepared,
            np.vstack([references, references]))
    with pytest.raises(diagnostics.DiagnosticError, match='reference|query'):
        diagnostics.attach_refinement(
            [witness], prepared, np.empty((0, 5)))

    bad_boxes = prepared[0].boxes.copy()
    bad_boxes[0, 2] = 0.0
    bad_record = replace(prepared[0], boxes=bad_boxes)
    with pytest.raises(diagnostics.DiagnosticError, match='positive'):
        diagnostics.attach_refinement(
            [witness], [bad_record], references)


@pytest.mark.parametrize(
    ('prior', 'final', 'patch_size'),
    [
        ([.5, .5, 0.0, .25, 0.0], [1, 1, 1, 1, 0], 1024),
        ([.5, .5, .25, .25, 0.0], [1, 1, -1, 1, 0], 1024),
        ([.5, .5, .25, .25, 0.0], [1, 1, 1, 1, 0], 0),
        ([.5, .5, .25, .25], [1, 1, 1, 1, 0], 1024),
        ([.5, .5, .25, .25, complex(0, 0)], [1, 1, 1, 1, 0], 1024),
    ],
)
def test_refinement_values_reject_invalid_boxes(prior, final, patch_size):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.refinement_values(prior, final, patch_size=patch_size)


def _outcome_evaluation(diagnostics, reverse=False):
    square = [10, 10, 4, 4, 0]
    far = [40, 40, 4, 4, 0]
    records = [
        _record(
            'filled',
            [square, square, far, square],
            [0.9, 0.8, 0.7, 0.6],
            [0, 0, 0, 1],
            [square],
            [0],
        ),
        _record('empty', [square], [0.95], [0]),
        _record(
            'ignored',
            [square],
            [0.99],
            [0],
            ignored_boxes=[square],
            ignored_labels=[0],
        ),
    ]
    if reverse:
        records.reverse()
    prepared = diagnostics.prepare_records_variable_queries(
        records, num_classes=2)
    return diagnostics.evaluate_records(
        prepared, num_classes=2, iou_threshold=0.5)


def _three_tp_evaluation(diagnostics, reverse=False):
    target = [10, 10, 4, 4, 0]
    records = [
        _record('iou-1.0', [target], [.7], [0], [target], [0]),
        _record(
            'iou-two-thirds', [[10, 10, 6, 4, 0]], [.8], [0],
            [target], [0]),
        _record(
            'iou-one-half', [[10, 10, 8, 4, 0]], [.9], [0],
            [target], [0]),
    ]
    if reverse:
        records.reverse()
    prepared = diagnostics.prepare_records_variable_queries(
        records, num_classes=2)
    return diagnostics.evaluate_records(
        prepared, num_classes=2, iou_threshold=0.5)


def test_calibration_sampling_is_stable_and_emits_fixed_score_bins():
    diagnostics = _load_module()
    bundle = _outcome_evaluation(diagnostics)
    reversed_bundle = _outcome_evaluation(diagnostics, reverse=True)
    keys = [
        (record.img_id, query_id)
        for record in bundle.records
        for query_id in range(len(record.scores))
    ]
    expected_keys = sorted(
        keys,
        key=lambda key: (
            hashlib.sha256(
                '{}\0{}\0{}'.format(20260722, key[0], key[1]).encode(
                    'utf-8')).hexdigest(),
            str(key[0]),
            key[1],
        ),
    )

    summary = diagnostics.summarize_calibration(
        bundle, sample_size=20, seed=20260722)
    reversed_summary = diagnostics.summarize_calibration(
        reversed_bundle, sample_size=20, seed=20260722)

    assert summary['sample_size_requested'] == 20
    assert summary['sample_size_used'] == 6
    assert summary['sample_keys'] == expected_keys
    assert reversed_summary['sample_keys'] == expected_keys
    assert [row['score_bin'] for row in summary['score_bins']] == list(
        diagnostics.SCORE_BINS)
    high = summary['score_bins'][-1]
    assert high['row_count'] == 4
    assert high['evaluator_row_count'] == 3
    assert high['ignored_count'] == 1
    assert high['tp_count'] == 1
    assert high['tp_precision'] == pytest.approx(1.0 / 3.0)
    assert high['same_label_hit_count'] == 2
    assert high['same_label_hit_rate'] == pytest.approx(.5)
    assert high['mean_same_label_iou'] == pytest.approx(.5, abs=1e-6)
    assert high['mean_tp_assigned_iou'] == pytest.approx(1.0, abs=1e-6)
    for empty in summary['score_bins'][:4]:
        assert empty['row_count'] == 0
        assert empty['evaluator_row_count'] == 0
        assert empty['ignored_count'] == 0
        assert empty['tp_count'] == 0
        assert empty['tp_precision'] is None
        assert empty['same_label_hit_rate'] is None
        assert empty['mean_same_label_iou'] is None
        assert empty['mean_tp_assigned_iou'] is None


def test_calibration_tp_spearman_uses_all_tps_not_the_sample():
    diagnostics = _load_module()
    bundle = _three_tp_evaluation(diagnostics)

    summary = diagnostics.summarize_calibration(
        bundle, sample_size=1, seed=7)

    assert summary['sample_size_used'] == 1
    assert summary['tp_score_iou_spearman'] == pytest.approx(-1.0)
    assert summary['per_class'] == [
        {
            'class_id': 0,
            'tp_count': 3,
            'tp_score_iou_spearman': pytest.approx(-1.0),
        },
        {
            'class_id': 1,
            'tp_count': 0,
            'tp_score_iou_spearman': None,
        },
    ]


def test_calibration_rejects_misaligned_bundles_and_sample_arguments():
    diagnostics = _load_module()
    bundle = _outcome_evaluation(diagnostics)
    bad_bundle = replace(
        bundle,
        row_tp=(np.zeros(1),) + bundle.row_tp[1:],
    )

    with pytest.raises(diagnostics.DiagnosticError, match='align|row'):
        diagnostics.summarize_calibration(
            bad_bundle, sample_size=1, seed=7)
    with pytest.raises(diagnostics.DiagnosticError, match='sample'):
        diagnostics.summarize_calibration(
            bundle, sample_size=-1, seed=7)
    with pytest.raises(diagnostics.DiagnosticError, match='seed'):
        diagnostics.summarize_calibration(
            bundle, sample_size=1, seed=True)
    with pytest.raises(diagnostics.DiagnosticError, match='threshold'):
        diagnostics.summarize_calibration(
            bundle, sample_size=1, seed=7, iou_threshold=1.1)


def test_query_summary_emits_all_queries_and_support_prefix_counts():
    diagnostics = _load_module()
    bundle = _outcome_evaluation(diagnostics)
    evidence = diagnostics.decompose_ground_truth(
        bundle.records, iou_threshold=.5)
    original_labels = [record.labels.copy() for record in bundle.records]
    original_scores = [record.scores.copy() for record in bundle.records]
    original_outcomes = [row.copy() for row in bundle.row_outcomes]

    summary = diagnostics.summarize_queries(
        bundle, evidence, query_count=6, num_classes=2)

    assert [row['query_id'] for row in summary['rows']] == list(range(6))
    query_zero = summary['rows'][0]
    assert query_zero['total_rows'] == 3
    assert query_zero['per_label_counts'] == [3, 0]
    assert query_zero['tp'] == 1
    assert query_zero['duplicate_fp'] == 0
    assert query_zero['semantic_fp'] == 0
    assert query_zero['localization_background_fp'] == 0
    assert query_zero['empty_tile_fp'] == 1
    assert query_zero['ignored_prediction'] == 1
    assert query_zero['reachable_witnesses'] == 1
    assert query_zero['ap_support_appearances'] == 3
    assert query_zero['score_min'] == pytest.approx(.9)
    assert query_zero['score_q25'] == pytest.approx(.925)
    assert query_zero['score_median'] == pytest.approx(.95)
    assert query_zero['score_q75'] == pytest.approx(.97)
    assert query_zero['score_max'] == pytest.approx(.99)
    assert summary['rows'][1]['duplicate_fp'] == 1
    assert summary['rows'][2]['localization_background_fp'] == 1
    assert summary['rows'][3]['semantic_fp'] == 1
    assert summary['rows'][3]['ap_support_appearances'] == 1
    for unused in summary['rows'][4:]:
        assert unused['total_rows'] == 0
        assert unused['per_label_counts'] == [0, 0]
        assert unused['score_min'] is None
        assert unused['score_q25'] is None
        assert unused['score_median'] is None
        assert unused['score_q75'] is None
        assert unused['score_max'] is None

    assert summary['all_row_label_allocation'] == [5, 1]
    assert summary['ap_support_label_allocation'] == [3, 1]
    assert summary['effective_query_count']['tp'] == pytest.approx(1.0)
    assert summary['effective_query_count'][
        'reachable_witness'] == pytest.approx(1.0)
    assert summary['effective_query_count']['ap_support'] == pytest.approx(
        diagnostics.effective_query_count([3, 0, 0, 1, 0, 0]))
    for record, labels, scores in zip(
            bundle.records, original_labels, original_scores):
        np.testing.assert_array_equal(record.labels, labels)
        np.testing.assert_array_equal(record.scores, scores)
    for actual, expected in zip(bundle.row_outcomes, original_outcomes):
        np.testing.assert_array_equal(actual, expected)


def test_query_summary_rejects_out_of_range_query_and_class_ids():
    diagnostics = _load_module()
    bundle = _outcome_evaluation(diagnostics)
    evidence = diagnostics.decompose_ground_truth(
        bundle.records, iou_threshold=.5)

    with pytest.raises(diagnostics.DiagnosticError, match='query'):
        diagnostics.summarize_queries(
            bundle, evidence, query_count=3, num_classes=2)
    invalid_class_evidence = [replace(evidence[0], class_id=2)]
    with pytest.raises(diagnostics.DiagnosticError, match='class'):
        diagnostics.summarize_queries(
            bundle, invalid_class_evidence, query_count=6, num_classes=2)
    bad_class_zero = replace(
        bundle.class_results[0],
        query_ids=np.full(
            len(bundle.class_results[0].query_ids), 99, dtype=np.int64),
    )
    bad_bundle = replace(
        bundle,
        class_results=(bad_class_zero,) + bundle.class_results[1:],
    )
    with pytest.raises(diagnostics.DiagnosticError, match='query'):
        diagnostics.summarize_queries(
            bad_bundle, evidence, query_count=6, num_classes=2)


def test_query_summary_rejects_witness_without_original_record_query():
    diagnostics = _load_module()
    bundle = _outcome_evaluation(diagnostics)
    evidence = diagnostics.decompose_ground_truth(
        bundle.records, iou_threshold=.5)
    witness = evidence[0]

    with pytest.raises(diagnostics.DiagnosticError, match='record|image'):
        diagnostics.summarize_queries(
            bundle, [replace(witness, img_id='missing')],
            query_count=6, num_classes=2)
    with pytest.raises(diagnostics.DiagnosticError, match='original|query'):
        diagnostics.summarize_queries(
            bundle, [replace(witness, witness_query=5)],
            query_count=6, num_classes=2)


def _strata_protocol(diagnostics):
    return diagnostics.ClassProtocol(
        classes=('small-vehicle', 'airport'),
        base_classes=('small-vehicle',),
        novel_classes=('airport',),
        canonical=False,
    )


def _strata_evidence(diagnostics, state, gt_index=0):
    reachable = state == 'evaluator_reachable'
    candidate_count = int(state in ('ownership_miss', 'evaluator_reachable'))
    return diagnostics.GtEvidence(
        img_id='dense', gt_index=gt_index, class_id=0,
        state=state, image_gt_count=11,
        gt_box=(10.0 + gt_index, 10.0, 8.0, 8.0, 0.0),
        best_any_query=0, best_any_iou=.1,
        best_same_query=0, best_same_iou=.1,
        candidate_count=candidate_count,
        witness_query=0 if reachable else None,
        witness_score=.9 if reachable else None,
        witness_iou=1.0 if reachable else None,
    )


def test_strata_emit_fixed_rows_including_empty_small_vehicle_crosses():
    diagnostics = _load_module()
    protocol = _strata_protocol(diagnostics)
    evidence = [_strata_evidence(diagnostics, 'geometry_miss')]

    rows = diagnostics.summarize_strata(
        evidence, protocol=protocol, small_vehicle_class_id=0)
    keyed = {(row['stratum_type'], row['stratum']): row for row in rows}

    assert len(rows) == 2 + 2 + 5 + 7 + 4 + 6 + 5 * 7
    assert [
        (row['stratum_type'], row['stratum']) for row in rows[:4]
    ] == [
        ('class', 'small-vehicle'),
        ('class', 'airport'),
        ('class_group', 'base'),
        ('class_group', 'novel'),
    ]
    assert rows[0]['class_id'] == 0
    assert rows[1]['class_id'] == 1
    assert keyed[('class_group', 'base')]['gt_count'] == 1
    assert keyed[('class_group', 'novel')]['gt_count'] == 0
    assert keyed[('small_vehicle_size_density', '<=8|11-50')][
        'gt_count'] == 1
    empty = keyed[('small_vehicle_size_density', '>64|>600')]
    assert empty['gt_count'] == 0
    assert empty['geometry_miss_rate'] is None
    assert empty['semantic_miss_rate'] is None
    assert empty['ownership_miss_rate'] is None
    assert empty['evaluator_reachable_rate'] is None
    assert empty['rate'] is None


def test_strata_rates_cover_all_four_mutually_exclusive_states():
    diagnostics = _load_module()
    protocol = _strata_protocol(diagnostics)
    states = (
        'geometry_miss',
        'semantic_miss',
        'ownership_miss',
        'evaluator_reachable',
    )
    evidence = [
        _strata_evidence(diagnostics, state, gt_index=index)
        for index, state in enumerate(states)
    ]

    rows = diagnostics.summarize_strata(
        evidence, protocol=protocol, small_vehicle_class_id=0)
    class_row = next(
        row for row in rows
        if row['stratum_type'] == 'class' and row['class_id'] == 0)

    assert class_row['gt_count'] == 4
    for state in states:
        assert class_row[state] == 1
        assert class_row['{}_rate'.format(state)] == pytest.approx(.25)
    assert class_row['rate'] == pytest.approx(.25)


@pytest.mark.parametrize(
    ('change', 'match'),
    [
        ({'class_id': 2}, 'class'),
        ({'image_gt_count': 0}, 'density|count'),
        ({'gt_box': (10.0, 10.0, 0.0, 8.0, 0.0)}, 'positive'),
        ({'state': 'unknown'}, 'state'),
    ],
)
def test_strata_reject_invalid_evidence(change, match):
    diagnostics = _load_module()
    evidence = replace(
        _strata_evidence(diagnostics, 'geometry_miss'), **change)

    with pytest.raises(diagnostics.DiagnosticError, match=match):
        diagnostics.summarize_strata(
            [evidence], _strata_protocol(diagnostics),
            small_vehicle_class_id=0)


def test_strata_validate_partition_and_small_vehicle_identity():
    diagnostics = _load_module()
    evidence = [_strata_evidence(diagnostics, 'geometry_miss')]
    bad_protocol = diagnostics.ClassProtocol(
        classes=('small-vehicle', 'airport'),
        base_classes=('small-vehicle',),
        novel_classes=(),
        canonical=False,
    )

    with pytest.raises(diagnostics.DiagnosticError, match='partition'):
        diagnostics.summarize_strata(
            evidence, bad_protocol, small_vehicle_class_id=0)
    with pytest.raises(diagnostics.DiagnosticError, match='small-vehicle'):
        diagnostics.summarize_strata(
            evidence, _strata_protocol(diagnostics),
            small_vehicle_class_id=1)


def test_case_manifest_loads_only_the_exact_contract(tmp_path):
    diagnostics = _load_module()
    manifest = {
        'context_false_sv': ['P0148__1024__651___0'],
        'true_sv_safety': ['P0682__1024__553___0'],
    }
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')

    loaded = diagnostics.load_case_manifest(path)

    assert loaded == diagnostics.EXPECTED_CASE_MANIFEST
    assert loaded == manifest


@pytest.mark.parametrize(
    'invalid_manifest',
    [
        {'context_false_sv': ['P0148__1024__651___0']},
        {
            'context_false_sv': ['P0148__1024__651___0'],
            'true_sv_safety': ['P0682__1024__553___0'],
            'threshold': 0.5,
        },
        {
            'context_false_sv': ['P0148__1024__651___0', 'extra'],
            'true_sv_safety': ['P0682__1024__553___0'],
        },
        ['P0148__1024__651___0', 'P0682__1024__553___0'],
    ],
)
def test_case_manifest_rejects_every_nonexact_shape(
        tmp_path, invalid_manifest):
    diagnostics = _load_module()
    path = tmp_path / 'invalid-cases.json'
    path.write_text(json.dumps(invalid_manifest), encoding='utf-8')

    with pytest.raises(diagnostics.DiagnosticError, match='exact'):
        diagnostics.load_case_manifest(path)


@pytest.mark.parametrize('contents', ['{', 'null'])
def test_case_manifest_wraps_read_and_json_failures(tmp_path, contents):
    diagnostics = _load_module()
    path = tmp_path / 'bad-cases.json'
    path.write_text(contents, encoding='utf-8')

    with pytest.raises(diagnostics.DiagnosticError, match='exact'):
        diagnostics.load_case_manifest(path)

    with pytest.raises(diagnostics.DiagnosticError, match='exact'):
        diagnostics.load_case_manifest(tmp_path / 'missing.json')


def test_case_manifest_rejects_duplicate_json_keys_even_when_last_is_correct(
        tmp_path):
    diagnostics = _load_module()
    path = tmp_path / 'duplicate-key.json'
    path.write_text(
        '{'
        '"context_false_sv":["wrong"],'
        '"context_false_sv":["P0148__1024__651___0"],'
        '"true_sv_safety":["P0682__1024__553___0"]'
        '}',
        encoding='utf-8')

    with pytest.raises(diagnostics.DiagnosticError, match='exact|duplicate'):
        diagnostics.load_case_manifest(path)


def test_case_manifest_validation_uses_immutable_private_oracle():
    diagnostics = _load_module()
    correct = _case_manifest()
    diagnostics.EXPECTED_CASE_MANIFEST.clear()
    diagnostics.EXPECTED_CASE_MANIFEST['attacker'] = ['accepted']

    validated = diagnostics._validate_case_manifest(correct)

    assert type(validated) is dict
    assert validated == correct
    assert validated is not correct
    assert all(type(value) is list for value in validated.values())
    assert all(
        value is not correct[key]
        for key, value in validated.items())
    validated['context_false_sv'].append('mutation')
    assert diagnostics._validate_case_manifest(correct) == correct
    with pytest.raises(diagnostics.DiagnosticError, match='exact'):
        diagnostics._validate_case_manifest({
            **correct,
            'outcome': 'attacker-controlled',
        })


def test_case_manifest_rejects_list_subclass_with_malicious_comparison():
    diagnostics = _load_module()

    class AlwaysEqualList(list):

        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    manifest = _case_manifest()
    manifest['context_false_sv'] = AlwaysEqualList(['wrong'])

    with pytest.raises(diagnostics.DiagnosticError, match='exact'):
        diagnostics._validate_case_manifest(manifest)


@pytest.mark.parametrize('error_type', [KeyError, ValueError])
def test_case_manifest_wraps_mapping_access_failures(error_type):
    diagnostics = _load_module()

    class BrokenAccessManifest(dict):

        def __getitem__(self, key):
            raise error_type('blocked')

    manifest = BrokenAccessManifest(_case_manifest())

    with pytest.raises(diagnostics.DiagnosticError, match='exact'):
        diagnostics._validate_case_manifest(manifest)


def test_case_summary_not_run_and_paired_input_gate_are_exact():
    diagnostics = _load_module()
    expected = [{
        'case_gate': 'not_run',
        'image_id': None,
        'case_group': None,
    }]

    assert diagnostics.summarize_cases(None, None, 0) == expected
    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.summarize_cases([], None, 0)
    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.summarize_cases(
            None, diagnostics.EXPECTED_CASE_MANIFEST, 0)
    with pytest.raises(diagnostics.DiagnosticError, match='exact'):
        diagnostics.summarize_cases([], {'context_false_sv': []}, 0)


def _case_manifest():
    return {
        'context_false_sv': ['P0148__1024__651___0'],
        'true_sv_safety': ['P0682__1024__553___0'],
    }


def test_case_summary_requires_fixed_ids_and_emits_manifest_order_without_mutation():
    diagnostics = _load_module()
    records = [
        _record('P0682__1024__553___0', [], [], []),
        _record('P0148__1024__651___0', [], [], []),
    ]
    original = copy.deepcopy(records)
    manifest = _case_manifest()
    original_manifest = copy.deepcopy(manifest)

    rows = diagnostics.summarize_cases(records, manifest, 0)

    assert [(row['image_id'], row['case_group']) for row in rows] == [
        ('P0148__1024__651___0', 'context_false_sv'),
        ('P0682__1024__553___0', 'true_sv_safety'),
    ]
    assert [row['case_gate'] for row in rows] == ['complete', 'complete']
    for record, before in zip(records, original):
        assert record['img_id'] == before['img_id']
        for section in (
                'pred_instances', 'gt_instances', 'ignored_instances'):
            for field, tensor in record[section].items():
                assert torch.equal(tensor, before[section][field])
    assert manifest == original_manifest


def test_case_summary_equal_score_ap_support_is_input_order_independent():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    records = [
        _record(
            'P0148__1024__651___0',
            [[40, 40, 4, 4, 0]],
            [.9],
            [0],
        ),
        _record(
            'P0682__1024__553___0',
            [square],
            [.9],
            [0],
            [square],
            [0],
        ),
    ]

    forward = diagnostics.summarize_cases(records, _case_manifest(), 0)
    reversed_rows = diagnostics.summarize_cases(
        list(reversed(records)), _case_manifest(), 0)

    assert reversed_rows == forward
    assert [
        row['small_vehicle_ap_support_count'] for row in forward
    ] == [1, 1]
    assert forward[0]['small_vehicle_empty_tile_fp_count'] == 1
    assert forward[1]['small_vehicle_tp_count'] == 1


@pytest.mark.parametrize(
    'image_ids',
    [
        ['P0148__1024__651___0'],
        ['P0148__1024__651___0', 'P0148__1024__651___0'],
        ['P0148__1024__651___0', 'unexpected'],
        [
            'P0148__1024__651___0',
            'P0682__1024__553___0',
            'unexpected',
        ],
    ],
)
def test_case_summary_rejects_missing_duplicate_and_extra_ids(image_ids):
    diagnostics = _load_module()
    records = [_record(image_id, [], [], []) for image_id in image_ids]

    with pytest.raises(diagnostics.DiagnosticError, match='exactly once'):
        diagnostics.summarize_cases(records, _case_manifest(), 0)


@pytest.mark.parametrize(
    'invalid_image_id',
    [
        np.array(['P0148__1024__651___0', 'other']),
        ['P0148__1024__651___0'],
        {'image': 'P0148__1024__651___0'},
        True,
    ],
)
def test_case_summary_rejects_nonstable_image_ids_as_diagnostic_error(
        invalid_image_id):
    diagnostics = _load_module()
    records = [
        _record(invalid_image_id, [], [], []),
        _record('P0682__1024__553___0', [], [], []),
    ]

    with pytest.raises(
            diagnostics.DiagnosticError, match='image ID|exactly once'):
        diagnostics.summarize_cases(records, _case_manifest(), 0)


def _outcome_case_records():
    shared_owner = [51, 50, 6, 4, 0]
    semantic_box = [70, 70, 4, 4, 0]
    ignored_box = [130, 130, 4, 4, 0]
    return [
        _record(
            'P0148__1024__651___0',
            [[200, 200, 4, 4, 0]],
            [.95],
            [0],
        ),
        _record(
            'P0682__1024__553___0',
            [
                [30, 30, 4, 4, 0],
                shared_owner,
                shared_owner,
                semantic_box,
                [110, 110, 4, 4, 0],
                ignored_box,
            ],
            [.99, .90, .80, .70, .60, .50],
            [1, 0, 0, 0, 0, 0],
            [
                [10, 10, 4, 4, 0],
                [30, 30, 4, 4, 0],
                [50, 50, 4, 4, 0],
                [52, 50, 4, 4, 0],
                semantic_box,
            ],
            [0, 0, 0, 0, 1],
            [ignored_box],
            [0],
        ),
    ]


def test_case_summary_reports_evaluator_outcomes_support_and_gt_states():
    diagnostics = _load_module()

    rows = diagnostics.summarize_cases(
        _outcome_case_records(), _case_manifest(), 0)
    p0148, p0682 = rows

    assert p0148['gt_count'] == 0
    assert p0148['small_vehicle_gt_count'] == 0
    assert p0148['predicted_small_vehicle_rows'] == 1
    assert p0148['top1_small_vehicle_scores'] == pytest.approx([.95])
    assert p0148['top5_small_vehicle_scores'] == pytest.approx([.95])
    assert p0148['top20_small_vehicle_scores'] == pytest.approx([.95])
    assert p0148['small_vehicle_empty_tile_fp_count'] == 1
    assert p0148['small_vehicle_ap_support_count'] == 1
    assert p0148['small_vehicle_gt_witnesses'] == []
    assert not any(
        'map' in key.lower() or 'contribution' in key.lower()
        for row in rows for key in row)

    assert p0682['gt_count'] == 5
    assert p0682['small_vehicle_gt_count'] == 4
    assert p0682['predicted_small_vehicle_rows'] == 5
    assert p0682['top1_small_vehicle_scores'] == pytest.approx([.90])
    assert p0682['top5_small_vehicle_scores'] == pytest.approx(
        [.90, .80, .70, .60, .50])
    assert p0682['top20_small_vehicle_scores'] == pytest.approx(
        [.90, .80, .70, .60, .50])
    for outcome, expected in {
            'tp': 1,
            'duplicate_fp': 1,
            'semantic_fp': 1,
            'localization_background_fp': 1,
            'empty_tile_fp': 0,
            'ignored_prediction': 1,
    }.items():
        assert p0682['small_vehicle_{}_count'.format(outcome)] == expected
    assert p0682['small_vehicle_ap_support_count'] == 1

    witnesses = p0682['small_vehicle_gt_witnesses']
    assert [row['gt_index'] for row in witnesses] == [0, 1, 2, 3]
    assert [row['state'] for row in witnesses] == [
        'geometry_miss',
        'semantic_miss',
        'evaluator_reachable',
        'ownership_miss',
    ]
    assert all('best_same_query' in row for row in witnesses)
    assert all('best_same_iou' in row for row in witnesses)
    for state in (
            'geometry_miss', 'semantic_miss', 'ownership_miss',
            'evaluator_reachable'):
        assert p0682['small_vehicle_{}_count'.format(state)] == 1
    assert sum(
        p0682['small_vehicle_{}_count'.format(state)]
        for state in (
            'geometry_miss', 'semantic_miss', 'ownership_miss',
            'evaluator_reachable')) == p0682['small_vehicle_gt_count']


def test_case_summary_selects_highest_false_sv_and_nearest_ordinary_gt():
    diagnostics = _load_module()

    p0148, p0682 = diagnostics.summarize_cases(
        _outcome_case_records(), _case_manifest(), 0)

    empty_witness = p0148['highest_score_false_small_vehicle_witness']
    assert empty_witness == {
        'query_id': 0,
        'score': pytest.approx(.95),
        'outcome': 'empty_tile_fp',
        'nearest_gt_index': None,
        'nearest_gt_label': None,
        'nearest_gt_iou': pytest.approx(0.0),
    }

    false_witness = p0682['highest_score_false_small_vehicle_witness']
    assert false_witness == {
        'query_id': 2,
        'score': pytest.approx(.80),
        'outcome': 'duplicate_fp',
        'nearest_gt_index': 2,
        'nearest_gt_label': 0,
        'nearest_gt_iou': pytest.approx(2.0 / 3.0, abs=1e-6),
    }


def test_case_summary_false_sv_nearest_gt_can_be_non_small_vehicle():
    diagnostics = _load_module()
    small_vehicle_gt = [10, 10, 4, 4, 0]
    non_small_vehicle_gt = [50, 50, 4, 4, 0]
    records = [
        _record('P0148__1024__651___0', [], [], []),
        _record(
            'P0682__1024__553___0',
            [non_small_vehicle_gt],
            [.9],
            [0],
            [small_vehicle_gt, non_small_vehicle_gt],
            [0, 1],
        ),
    ]

    rows = diagnostics.summarize_cases(records, _case_manifest(), 0)

    witness = rows[1]['highest_score_false_small_vehicle_witness']
    assert witness['outcome'] == 'semantic_fp'
    assert witness['nearest_gt_index'] == 1
    assert witness['nearest_gt_label'] == 1
    assert witness['nearest_gt_iou'] == pytest.approx(1.0)


def test_case_summary_nearest_gt_iou_tie_uses_lowest_original_index():
    diagnostics = _load_module()
    shared_box = [50, 50, 4, 4, 0]
    records = [
        _record('P0148__1024__651___0', [], [], []),
        _record(
            'P0682__1024__553___0',
            [shared_box],
            [.9],
            [0],
            [shared_box, shared_box],
            [2, 1],
        ),
    ]

    rows = diagnostics.summarize_cases(records, _case_manifest(), 0)

    witness = rows[1]['highest_score_false_small_vehicle_witness']
    assert witness['nearest_gt_index'] == 0
    assert witness['nearest_gt_label'] == 2
    assert witness['nearest_gt_iou'] == pytest.approx(1.0)


def test_case_summary_has_no_false_witness_for_only_tp_and_ignored_rows():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    ignored = [30, 30, 4, 4, 0]
    records = [
        _record('P0148__1024__651___0', [], [], []),
        _record(
            'P0682__1024__553___0',
            [square, ignored],
            [.9, .8],
            [0, 0],
            [square],
            [0],
            [ignored],
            [0],
        ),
    ]

    rows = diagnostics.summarize_cases(records, _case_manifest(), 0)

    assert [
        row['highest_score_false_small_vehicle_witness'] for row in rows
    ] == [None, None]
    assert rows[0]['top1_small_vehicle_scores'] == []
    assert rows[0]['top5_small_vehicle_scores'] == []
    assert rows[0]['top20_small_vehicle_scores'] == []


def test_case_summary_false_witness_score_tie_keeps_original_query_order():
    diagnostics = _load_module()
    records = [
        _record(
            'P0148__1024__651___0',
            [[10, 10, 4, 4, 0], [20, 20, 4, 4, 0]],
            [.9, .9],
            [0, 0],
        ),
        _record('P0682__1024__553___0', [], [], []),
    ]

    rows = diagnostics.summarize_cases(records, _case_manifest(), 0)

    witness = rows[0]['highest_score_false_small_vehicle_witness']
    assert witness['query_id'] == 0
    assert witness['outcome'] == 'empty_tile_fp'


def test_case_summary_top_scores_sort_and_truncate_more_than_twenty_rows():
    diagnostics = _load_module()
    scores = [
        .11, .92, .33, .77, .05, .88, .44, .66, .22, .99, .55, .13,
        .81, .37, .73, .02, .95, .48, .69, .27, .84, .16, .99,
    ]
    boxes = [
        [10 + 10 * query_id, 10, 4, 4, 0]
        for query_id in range(len(scores))
    ]
    records = [
        _record(
            'P0148__1024__651___0',
            boxes,
            scores,
            [0] * len(scores),
        ),
        _record('P0682__1024__553___0', [], [], []),
    ]

    rows = diagnostics.summarize_cases(records, _case_manifest(), 0)

    expected = sorted(scores, reverse=True)
    row = rows[0]
    assert row['predicted_small_vehicle_rows'] == 23
    assert row['top1_small_vehicle_scores'] == pytest.approx(expected[:1])
    assert row['top5_small_vehicle_scores'] == pytest.approx(expected[:5])
    assert row['top20_small_vehicle_scores'] == pytest.approx(expected[:20])
    assert len(row['top20_small_vehicle_scores']) == 20
    assert all(
        lhs >= rhs
        for lhs, rhs in zip(
            row['top20_small_vehicle_scores'],
            row['top20_small_vehicle_scores'][1:]))
    assert row['highest_score_false_small_vehicle_witness']['query_id'] == 9


@pytest.mark.parametrize('invalid_class_id', [True, -1, 0.5, '0'])
def test_case_summary_rejects_invalid_small_vehicle_class_id(
        invalid_class_id):
    diagnostics = _load_module()
    records = [
        _record('P0148__1024__651___0', [], [], []),
        _record('P0682__1024__553___0', [], [], []),
    ]

    with pytest.raises(
            diagnostics.DiagnosticError, match='small_vehicle_class_id'):
        diagnostics.summarize_cases(
            records, _case_manifest(), invalid_class_id)


@pytest.mark.parametrize(
    ('section', 'labels'),
    [
        ('pred_instances', torch.tensor([-1], dtype=torch.long)),
        ('gt_instances', torch.tensor([0.5], dtype=torch.float32)),
        ('ignored_instances', torch.tensor([True], dtype=torch.bool)),
    ],
)
def test_case_summary_rejects_out_of_range_and_invalid_labels(
        section, labels):
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    record = _record(
        'P0682__1024__553___0',
        [square],
        [.9],
        [0],
        [square],
        [0],
        [square],
        [0],
    )
    record[section]['labels'] = labels
    records = [
        _record('P0148__1024__651___0', [], [], []),
        record,
    ]

    with pytest.raises(diagnostics.DiagnosticError, match='label'):
        diagnostics.summarize_cases(records, _case_manifest(), 0)


def test_case_summary_infers_classes_from_prediction_gt_and_ignored_labels():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    records = [
        _record(
            'P0148__1024__651___0',
            [square],
            [.9],
            [3],
        ),
        _record(
            'P0682__1024__553___0',
            [],
            [],
            [],
            [square],
            [4],
            [square],
            [5],
        ),
    ]

    rows = diagnostics.summarize_cases(records, _case_manifest(), 0)

    assert [row['predicted_small_vehicle_rows'] for row in rows] == [0, 0]
    assert [row['small_vehicle_gt_count'] for row in rows] == [0, 0]


def test_case_summary_infers_classes_when_small_vehicle_id_is_highest():
    diagnostics = _load_module()
    square = [10, 10, 4, 4, 0]
    records = [
        _record(
            'P0148__1024__651___0',
            [square],
            [.9],
            [0],
        ),
        _record(
            'P0682__1024__553___0',
            [square],
            [.8],
            [1],
            [square],
            [2],
            [square],
            [3],
        ),
    ]

    rows = diagnostics.summarize_cases(
        records, _case_manifest(), small_vehicle_class_id=5)

    assert [row['predicted_small_vehicle_rows'] for row in rows] == [0, 0]
    assert [row['small_vehicle_gt_count'] for row in rows] == [0, 0]
    assert [row['top20_small_vehicle_scores'] for row in rows] == [[], []]
    assert [
        row['highest_score_false_small_vehicle_witness'] for row in rows
    ] == [None, None]


_DEFAULT_DIAGNOSTIC_ARGUMENT = object()


def _diagnostic_summary(
        diagnostics,
        reconstructed_map=np.float64(.6),
        oracle_map=np.float32(.65),
        case_gate=_DEFAULT_DIAGNOSTIC_ARGUMENT,
        warnings=_DEFAULT_DIAGNOSTIC_ARGUMENT):
    if case_gate is _DEFAULT_DIAGNOSTIC_ARGUMENT:
        case_gate = {'status': 'not_run'}
    if warnings is _DEFAULT_DIAGNOSTIC_ARGUMENT:
        warnings = ('warning-b', 'warning-a')
    return diagnostics.build_diagnostics(
        canonical=True,
        integrity={'dump_sha256': 'abc'},
        parity={'same_dump': True},
        reconstructed_map=reconstructed_map,
        oracle_map=oracle_map,
        gt_decomposition={'geometry_miss': 2},
        capacity={'reachable': 3},
        fp_all={'semantic_fp': 5},
        fp_last_tp={'semantic_fp': 4},
        fp_ap_support={'semantic_fp': 3},
        calibration={'bins': [1, 2]},
        query_summary={'effective': {'tp': 7}},
        base_summary={'ap': .55},
        novel_summary={'ap': .45},
        case_gate=case_gate,
        warnings=warnings,
    )


def test_build_diagnostics_has_exact_math_and_structure():
    diagnostics = _load_module()

    summary = _diagnostic_summary(diagnostics)

    assert summary == {
        'schema_version': 1,
        'canonical': True,
        'integrity': {'dump_sha256': 'abc'},
        'parity': {'same_dump': True},
        'metrics': {
            'map': .6,
            'ap50': .6,
            'ap70_gap': pytest.approx(.1),
            'oracle_map': pytest.approx(.65),
            'oracle_headroom': pytest.approx(.05),
            'headroom_fraction_to_ap70': pytest.approx(2.0),
        },
        'gt_decomposition': {'geometry_miss': 2},
        'capacity': {'reachable': 3},
        'fp_regions': {
            'all': {'semantic_fp': 5},
            'through_last_tp': {'semantic_fp': 4},
            'ap_support': {'semantic_fp': 3},
        },
        'calibration': {'bins': [1, 2]},
        'queries': {'effective': {'tp': 7}},
        'groups': {'base': {'ap': .55}, 'novel': {'ap': .45}},
        'case_gate': {'status': 'not_run'},
        'warnings': ['warning-b', 'warning-a'],
    }


def test_build_diagnostics_detaches_all_mutable_inputs():
    diagnostics = _load_module()
    integrity = {'nested': [{'value': np.int64(1)}]}
    warnings = [{'message': 'before'}]

    summary = diagnostics.build_diagnostics(
        True, integrity, {}, .7, .7, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {'status': 'not_run'}, warnings)
    integrity['nested'][0]['value'] = 99
    warnings[0]['message'] = 'after'

    assert summary['integrity'] == {'nested': [{'value': 1}]}
    assert summary['warnings'] == [{'message': 'before'}]
    assert type(summary['integrity']['nested'][0]['value']) is int
    assert summary['metrics']['headroom_fraction_to_ap70'] is None


def test_build_diagnostics_rejects_oracle_below_reconstructed_map():
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError, match='oracle'):
        _diagnostic_summary(
            diagnostics, reconstructed_map=.6, oracle_map=.59)


def test_build_diagnostics_clamps_near_equal_oracle_within_tolerance():
    diagnostics = _load_module()

    summary = _diagnostic_summary(
        diagnostics,
        reconstructed_map=.6,
        oracle_map=.6 - 5e-13,
    )

    assert summary['metrics']['oracle_map'] == .6
    assert summary['metrics']['oracle_headroom'] == 0.0
    assert summary['metrics']['headroom_fraction_to_ap70'] is None


@pytest.mark.parametrize(
    ('field', 'invalid'),
    [
        ('canonical', 1),
        ('canonical', np.bool_(True)),
        ('reconstructed_map', True),
        ('oracle_map', float('inf')),
        ('warnings', 'warning'),
        ('integrity', {'bad': float('nan')}),
        ('calibration', {'bad': np.array([1.0])}),
    ],
)
def test_build_diagnostics_rejects_invalid_and_nonfinite_values(
        field, invalid):
    diagnostics = _load_module()
    kwargs = {
        'canonical': True,
        'integrity': {},
        'parity': {},
        'reconstructed_map': .6,
        'oracle_map': .65,
        'gt_decomposition': {},
        'capacity': {},
        'fp_all': {},
        'fp_last_tp': {},
        'fp_ap_support': {},
        'calibration': {},
        'query_summary': {},
        'base_summary': {},
        'novel_summary': {},
        'case_gate': {'status': 'not_run'},
        'warnings': [],
    }
    kwargs[field] = invalid

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.build_diagnostics(**kwargs)


@pytest.mark.parametrize(
    'invalid_case_gate',
    [
        {},
        [],
        {'status': 'unknown'},
        [{}],
        [{'case_gate': 'not_run', 'image_id': None,
          'case_group': 'context_false_sv'}],
        [{'case_gate': 'complete',
          'image_id': 'P0148__1024__651___0',
          'case_group': 'context_false_sv'}],
        [
            {'case_gate': 'complete',
             'image_id': 'P0148__1024__651___0',
             'case_group': 'context_false_sv'},
            {'case_gate': 'complete',
             'image_id': 'P0682__1024__553___0',
             'case_group': 'context_false_sv'},
        ],
    ],
    ids=[
        'empty-mapping',
        'empty-list',
        'unknown-mapping-status',
        'empty-row',
        'damaged-not-run-row',
        'incomplete-complete-rows',
        'wrong-complete-pair',
    ],
)
@pytest.mark.parametrize('entrypoint', ['build', 'render'])
def test_diagnostics_rejects_case_gate_outside_closed_states(
        entrypoint, invalid_case_gate):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError, match='case_gate'):
        if entrypoint == 'build':
            _diagnostic_summary(
                diagnostics, case_gate=copy.deepcopy(invalid_case_gate))
        else:
            summary = _diagnostic_summary(diagnostics)
            summary['case_gate'] = copy.deepcopy(invalid_case_gate)
            diagnostics.render_payloads(summary, _render_tables())


def test_build_diagnostics_accepts_task5_case_rows_and_normalizes_order():
    diagnostics = _load_module()
    not_run = diagnostics.summarize_cases(None, None, 0)
    complete = [
        {'case_gate': 'complete',
         'image_id': 'P0682__1024__553___0',
         'case_group': 'true_sv_safety',
         'small_vehicle_gt_count': 3},
        {'case_gate': 'complete',
         'image_id': 'P0148__1024__651___0',
         'case_group': 'context_false_sv',
         'small_vehicle_gt_count': 0},
    ]

    assert _diagnostic_summary(
        diagnostics, case_gate=not_run)['case_gate'] == not_run
    summary = _diagnostic_summary(diagnostics, case_gate=complete)
    assert [row['image_id'] for row in summary['case_gate']] == [
        'P0148__1024__651___0',
        'P0682__1024__553___0',
    ]


def test_build_diagnostics_wraps_warning_iteration_failure_with_cause():
    diagnostics = _load_module()
    failure = RuntimeError('warnings exploded')

    def warnings():
        yield 'before'
        raise failure

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        _diagnostic_summary(diagnostics, warnings=warnings())

    assert caught.value.__cause__ is failure


def _error_example(outcome, img_id, query_id, score, **extra):
    return {
        'outcome': outcome,
        'img_id': img_id,
        'query_id': query_id,
        'score': score,
        **extra,
    }


def test_cap_error_examples_caps_each_type_and_uses_frozen_order():
    diagnostics = _load_module()
    examples = [
        _error_example('semantic_fp', 'z', 4, .9),
        _error_example('empty_tile_fp', 'd', 3, .4),
        _error_example('semantic_fp', 'b', 2, .8),
        _error_example('duplicate_fp', 'c', 2, .7),
        _error_example('semantic_fp', 'a', 8, .9),
        _error_example('localization_background_fp', 'e', 1, .6),
        _error_example('duplicate_fp', 'a', 9, .7),
    ]
    original = copy.deepcopy(examples)

    capped = diagnostics.cap_error_examples(examples, max_per_type=2)

    assert [(row['outcome'], row['img_id'], row['query_id'])
            for row in capped] == [
        ('duplicate_fp', 'a', 9),
        ('duplicate_fp', 'c', 2),
        ('semantic_fp', 'a', 8),
        ('semantic_fp', 'z', 4),
        ('localization_background_fp', 'e', 1),
        ('empty_tile_fp', 'd', 3),
    ]
    assert examples == original


def test_cap_error_examples_handles_zero_and_detaches_rows():
    diagnostics = _load_module()
    row = _error_example(
        'semantic_fp', 'a', np.int64(1), np.float32(.5),
        details={'values': [np.int64(2)]})

    capped = diagnostics.cap_error_examples([row], max_per_type=np.int64(1))
    row['details']['values'][0] = 9

    assert capped == [{
        'outcome': 'semantic_fp',
        'img_id': 'a',
        'query_id': 1,
        'score': pytest.approx(.5),
        'details': {'values': [2]},
    }]
    assert diagnostics.cap_error_examples([row], max_per_type=0) == []


@pytest.mark.parametrize('vary_evidence', [False, True])
def test_cap_error_examples_rejects_duplicate_fixed_sort_keys(vary_evidence):
    diagnostics = _load_module()
    first = _error_example(
        'semantic_fp', 'same-image', 7, .9, class_id=1)
    second = copy.deepcopy(first)
    if vary_evidence:
        second['class_id'] = 2

    with pytest.raises(diagnostics.DiagnosticError, match='duplicate'):
        diagnostics.cap_error_examples([first, second], max_per_type=2)


def test_cap_error_examples_nonduplicate_rows_remain_order_independent():
    diagnostics = _load_module()
    rows = [
        _error_example('semantic_fp', 'same-image', 8, .9, class_id=2),
        _error_example('semantic_fp', 'same-image', 7, .9, class_id=1),
    ]

    assert diagnostics.cap_error_examples(
        rows, max_per_type=2) == diagnostics.cap_error_examples(
            list(reversed(rows)), max_per_type=2)


def test_cap_error_examples_streams_large_generator_into_bounded_results():
    diagnostics = _load_module()

    class ObservedRow(dict):
        was_read = False

        def __getitem__(self, key):
            self.was_read = True
            return super().__getitem__(key)

    outcomes = (
        'duplicate_fp', 'semantic_fp',
        'localization_background_fp', 'empty_tile_fp')

    def rows():
        previous = None
        for index in range(240):
            if previous is not None:
                assert previous.was_read, 'input was materialized before use'
            previous = ObservedRow(_error_example(
                outcomes[index % len(outcomes)],
                'image-{:03d}'.format(index),
                index,
                index / 1000.0,
                class_id=index % 3,
            ))
            yield previous
        assert previous is not None and previous.was_read

    capped = diagnostics.cap_error_examples(rows(), max_per_type=3)

    assert len(capped) == 12
    assert [row['outcome'] for row in capped] == [
        outcome for outcome in outcomes for _ in range(3)]
    for offset in range(0, len(capped), 3):
        scores = [row['score'] for row in capped[offset:offset + 3]]
        assert scores == sorted(scores, reverse=True)


def test_cap_error_examples_ignores_duplicate_keys_below_retained_cap():
    diagnostics = _load_module()
    rows = [
        _error_example('semantic_fp', 'retained', 0, .9),
        _error_example('semantic_fp', 'not-retained', 1, .1, class_id=1),
        _error_example('semantic_fp', 'not-retained', 1, .1, class_id=2),
    ]

    assert diagnostics.cap_error_examples(rows, max_per_type=1) == [
        _error_example('semantic_fp', 'retained', 0, .9)]


def test_cap_error_examples_wraps_mid_iteration_failure_with_cause():
    diagnostics = _load_module()

    def rows():
        yield _error_example('semantic_fp', 'a', 0, .9)
        raise RuntimeError('iterator exploded')

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.cap_error_examples(rows(), max_per_type=2)

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_cap_error_examples_preserves_iterator_diagnostic_error_and_cause():
    diagnostics = _load_module()
    root_cause = RuntimeError('root cause')
    diagnostic = diagnostics.DiagnosticError('already classified')
    diagnostic.__cause__ = root_cause

    def rows():
        yield _error_example('semantic_fp', 'a', 0, .9)
        raise diagnostic

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.cap_error_examples(rows(), max_per_type=2)

    assert caught.value is diagnostic
    assert caught.value.__cause__ is root_cause


@pytest.mark.parametrize(
    'row',
    [
        [],
        _error_example('tp', 'a', 0, .5),
        _error_example('ignored_prediction', 'a', 0, .5),
        _error_example('semantic_fp', 1, 0, .5),
        _error_example('semantic_fp', 'a', True, .5),
        _error_example('semantic_fp', 'a', -1, .5),
        _error_example('semantic_fp', 'a', 0, True),
        _error_example('semantic_fp', 'a', 0, float('nan')),
    ],
)
def test_cap_error_examples_rejects_invalid_rows(row):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.cap_error_examples([row], max_per_type=2)


def test_cap_error_examples_wraps_nonstring_outcome_as_diagnostic_error():
    diagnostics = _load_module()
    row = _error_example(
        np.array(['semantic_fp', 'duplicate_fp']), 'a', 0, .5)

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.cap_error_examples([row], max_per_type=2)


@pytest.mark.parametrize('limit', [True, -1, 1.5, '2'])
def test_cap_error_examples_rejects_invalid_limit(limit):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.cap_error_examples([], max_per_type=limit)


_EXPECTED_CSV_SCHEMAS = {
    'per_class.csv': (
        'class_id', 'class_name', 'group', 'num_gts', 'prediction_rows',
        'tp_count', 'fp_count', 'recall', 'ap', 'oracle_ap',
        'oracle_headroom', 'last_tp_end', 'ap_support_end',
        'geometry_miss', 'semantic_miss', 'ownership_miss',
        'evaluator_reachable', 'all_tp', 'all_duplicate_fp',
        'all_semantic_fp', 'all_localization_background_fp',
        'all_empty_tile_fp', 'all_ignored_prediction',
        'through_last_tp_tp', 'through_last_tp_duplicate_fp',
        'through_last_tp_semantic_fp',
        'through_last_tp_localization_background_fp',
        'through_last_tp_empty_tile_fp',
        'through_last_tp_ignored_prediction', 'ap_support_tp',
        'ap_support_duplicate_fp', 'ap_support_semantic_fp',
        'ap_support_localization_background_fp',
        'ap_support_empty_tile_fp', 'ap_support_ignored_prediction',
        'tp_score_iou_spearman',
    ),
    'strata.csv': (
        'stratum_type', 'stratum', 'class_id', 'gt_count',
        'geometry_miss', 'semantic_miss', 'ownership_miss',
        'evaluator_reachable', 'geometry_miss_rate',
        'semantic_miss_rate', 'ownership_miss_rate',
        'evaluator_reachable_rate', 'rate',
    ),
    'per_query.csv': (
        'query_id', 'total_rows', 'per_label_counts', 'tp',
        'duplicate_fp', 'semantic_fp', 'localization_background_fp',
        'empty_tile_fp', 'ignored_prediction', 'reachable_witnesses',
        'ap_support_appearances', 'score_min', 'score_q25',
        'score_median', 'score_q75', 'score_max',
    ),
    'case_studies.csv': (
        'case_gate', 'image_id', 'case_group', 'gt_count',
        'small_vehicle_gt_count', 'predicted_small_vehicle_rows',
        'top1_small_vehicle_scores', 'top5_small_vehicle_scores',
        'top20_small_vehicle_scores', 'small_vehicle_ap_support_count',
        'small_vehicle_tp_count', 'small_vehicle_duplicate_fp_count',
        'small_vehicle_semantic_fp_count',
        'small_vehicle_localization_background_fp_count',
        'small_vehicle_empty_tile_fp_count',
        'small_vehicle_ignored_prediction_count',
        'small_vehicle_geometry_miss_count',
        'small_vehicle_semantic_miss_count',
        'small_vehicle_ownership_miss_count',
        'small_vehicle_evaluator_reachable_count',
        'small_vehicle_gt_witnesses',
        'highest_score_false_small_vehicle_witness',
    ),
    'error_examples.csv': (
        'outcome', 'img_id', 'query_id', 'score', 'class_id',
        'matched_gt', 'assigned_iou', 'same_label_iou',
        'nearest_gt_index', 'nearest_gt_label', 'nearest_gt_iou',
    ),
}


def _render_tables():
    return {
        'per_class.csv': [
            {'class_id': 2, 'class_name': 'two', 'ap': None},
            {'class_id': 1, 'class_name': 'comma, newline\nand "quote"',
             'ap': .4},
        ],
        'strata.csv': [
            {'stratum_type': 'small_vehicle_size_density',
             'stratum': '(8,16]|11-50', 'gt_count': 2},
            {'stratum_type': 'size', 'stratum': '(8,16]', 'gt_count': 3},
            {'stratum_type': 'class_group', 'stratum': 'novel',
             'gt_count': 4},
            {'stratum_type': 'class', 'stratum': 'one', 'class_id': 1,
             'gt_count': 5},
            {'stratum_type': 'small_vehicle_size_density',
             'stratum': '<=8|401-600', 'gt_count': 6},
            {'stratum_type': 'size', 'stratum': '<=8', 'gt_count': 7},
            {'stratum_type': 'class_group', 'stratum': 'base',
             'gt_count': 8},
            {'stratum_type': 'class', 'stratum': 'zero', 'class_id': 0,
             'gt_count': 9},
        ],
        'per_query.csv': [
            {'query_id': 2, 'total_rows': 4, 'per_label_counts': [1, 3]},
            {'query_id': 1, 'total_rows': 5,
             'per_label_counts': {'b': 2, 'a': 3}},
        ],
        'case_studies.csv': [
            {'case_gate': 'complete', 'image_id': 'P0682__1024__553___0',
             'case_group': 'true_sv_safety',
             'small_vehicle_gt_witnesses': [{'state': 'semantic_miss'}]},
            {'case_gate': 'complete', 'image_id': 'P0148__1024__651___0',
             'case_group': 'context_false_sv',
             'top5_small_vehicle_scores': [.9, .2]},
        ],
        'error_examples.csv': [
            _error_example('semantic_fp', 'z', 2, .9, class_id=1),
            _error_example('duplicate_fp', 'b', 4, .7, class_id=0),
            _error_example('semantic_fp', 'a', 3, .9, class_id=1),
            _error_example(
                'localization_background_fp', 'c', 1, .8, class_id=2),
            _error_example('empty_tile_fp', 'd', 0, .6, class_id=2),
        ],
    }


def _csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode('utf-8'))))


def test_render_payloads_has_exact_files_schemas_and_stable_sorted_bytes():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    tables = _render_tables()
    original = copy.deepcopy(tables)

    payloads = diagnostics.render_payloads(summary, tables)
    reversed_payloads = diagnostics.render_payloads(
        summary, {name: list(reversed(rows))
                  for name, rows in tables.items()})

    assert tuple(diagnostics.CSV_SCHEMAS) == tuple(_EXPECTED_CSV_SCHEMAS)
    assert diagnostics.CSV_SCHEMAS == _EXPECTED_CSV_SCHEMAS
    assert set(payloads) == {
        'diagnostics.json', 'per_class.csv', 'strata.csv',
        'per_query.csv', 'case_studies.csv', 'error_examples.csv',
        'report.md',
    }
    assert all(type(payload) is bytes for payload in payloads.values())
    assert payloads == reversed_payloads
    assert diagnostics.render_payloads(summary, tables) == payloads
    assert tables == original
    assert json.loads(payloads['diagnostics.json']) == summary
    assert payloads['diagnostics.json'].endswith(b'\n')
    assert not payloads['diagnostics.json'].endswith(b'\n\n')
    assert b'NaN' not in b''.join(payloads.values())
    assert b'Infinity' not in b''.join(payloads.values())

    for name, fields in _EXPECTED_CSV_SCHEMAS.items():
        header = payloads[name].splitlines()[0].decode('utf-8')
        assert next(csv.reader([header])) == list(fields)
        assert payloads[name].endswith(b'\n')
    per_class = _csv_rows(payloads['per_class.csv'])
    assert [row['class_id'] for row in per_class] == ['1', '2']
    assert per_class[0]['class_name'] == 'comma, newline\nand "quote"'
    assert per_class[1]['ap'] == ''
    assert [row['query_id'] for row in _csv_rows(
        payloads['per_query.csv'])] == ['1', '2']
    assert _csv_rows(payloads['per_query.csv'])[0][
        'per_label_counts'] == '{"a":3,"b":2}'
    assert [row['image_id'] for row in _csv_rows(
        payloads['case_studies.csv'])] == [
        'P0148__1024__651___0', 'P0682__1024__553___0']
    assert [row['outcome'] for row in _csv_rows(
        payloads['error_examples.csv'])] == [
        'duplicate_fp', 'semantic_fp', 'semantic_fp',
        'localization_background_fp', 'empty_tile_fp']
    assert [row['img_id'] for row in _csv_rows(
        payloads['error_examples.csv'])[1:3]] == ['a', 'z']


def test_render_report_keeps_aggregates_without_copying_large_row_details():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    calibration_detail = 'CALIBRATION_SAMPLE_DETAIL_SENTINEL_' + 'x' * 128
    query_detail = 'QUERY_ROW_DETAIL_SENTINEL_' + 'y' * 128
    summary['calibration'] = {
        'sample_size_requested': 500000,
        'sample_size_used': 500000,
        'sample_keys': [
            [calibration_detail, query_id] for query_id in range(6000)
        ],
        'tp_score_iou_spearman': .277,
        'per_class': [{
            'class_id': 0,
            'tp_count': 17,
            'tp_score_iou_spearman': .123,
        }],
        'score_bins': [{
            'score_bin': '[0.9,1.0]',
            'row_count': 23,
        }],
    }
    summary['queries'] = {
        'rows': [
            {'query_id': query_id, 'detail': query_detail}
            for query_id in range(6000)
        ],
        'effective_query_count': {
            'tp': 7,
            'reachable_witness': 11,
            'ap_support': 13,
        },
        'all_row_label_allocation': [101, 202],
        'ap_support_label_allocation': [31, 41],
    }
    original = copy.deepcopy(summary)

    payloads = diagnostics.render_payloads(summary, _render_tables())
    report = payloads['report.md']

    assert len(payloads['diagnostics.json']) > 1024 * 1024
    assert len(report) < 1024 * 1024
    assert calibration_detail.encode() in payloads['diagnostics.json']
    assert query_detail.encode() in payloads['diagnostics.json']
    assert calibration_detail.encode() not in report
    assert query_detail.encode() not in report
    assert b'sample_keys' not in report
    for aggregate in (
            b'per_class', b'score_bins', b'sample_size_requested',
            b'sample_size_used', b'tp_score_iou_spearman',
            b'all_row_label_allocation', b'ap_support_label_allocation',
            b'effective_query_count'):
        assert aggregate in report
    assert summary == original


def test_render_payloads_strata_order_matches_fixed_protocol():
    diagnostics = _load_module()

    rows = _csv_rows(diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), _render_tables())['strata.csv'])

    assert [(row['stratum_type'], row['stratum']) for row in rows] == [
        ('class', 'zero'),
        ('class', 'one'),
        ('class_group', 'base'),
        ('class_group', 'novel'),
        ('size', '<=8'),
        ('size', '(8,16]'),
        ('small_vehicle_size_density', '<=8|401-600'),
        ('small_vehicle_size_density', '(8,16]|11-50'),
    ]


def test_render_payloads_accepts_single_not_run_case_row():
    diagnostics = _load_module()
    tables = _render_tables()
    tables['case_studies.csv'] = [{
        'case_gate': 'not_run', 'image_id': None, 'case_group': None}]

    rows = _csv_rows(diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), tables)['case_studies.csv'])

    assert rows == [{field: (
        'not_run' if field == 'case_gate' else '')
        for field in _EXPECTED_CSV_SCHEMAS['case_studies.csv']}]


@pytest.mark.parametrize('mutation', ['missing', 'extra'])
def test_render_payloads_requires_exact_table_names(mutation):
    diagnostics = _load_module()
    tables = _render_tables()
    if mutation == 'missing':
        del tables['per_query.csv']
    else:
        tables['other.csv'] = []

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.render_payloads(_diagnostic_summary(diagnostics), tables)


@pytest.mark.parametrize(
    ('table_name', 'row'),
    [
        ('per_class.csv', {'class_id': 1, 'unexpected': 'evidence'}),
        ('per_query.csv', {'query_id': 1, 'score_max': float('nan')}),
        ('error_examples.csv', _error_example(
            'semantic_fp', 'a', 1, .5, unexpected='evidence')),
        ('strata.csv', {
            'stratum_type': 'size', 'stratum': 'unknown', 'gt_count': 1}),
    ],
)
def test_render_payloads_rejects_extra_nonfinite_and_unknown_rows(
        table_name, row):
    diagnostics = _load_module()
    tables = _render_tables()
    tables[table_name] = [row]

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.render_payloads(_diagnostic_summary(diagnostics), tables)


@pytest.mark.parametrize(
    ('table_name', 'rows'),
    [
        ('per_class.csv', [{'class_id': 1}, {'class_id': 1}]),
        ('per_query.csv', [{'query_id': 1}, {'query_id': 1}]),
        ('strata.csv', [
            {'stratum_type': 'size', 'stratum': '<=8'},
            {'stratum_type': 'size', 'stratum': '<=8'},
        ]),
        ('case_studies.csv', [
            {'case_gate': 'complete',
             'image_id': 'P0148__1024__651___0',
             'case_group': 'context_false_sv'},
            {'case_gate': 'complete',
             'image_id': 'P0148__1024__651___0',
             'case_group': 'context_false_sv'},
        ]),
    ],
)
def test_render_payloads_rejects_duplicate_sort_keys(table_name, rows):
    diagnostics = _load_module()
    tables = _render_tables()
    tables[table_name] = rows

    with pytest.raises(diagnostics.DiagnosticError, match='duplicate'):
        diagnostics.render_payloads(_diagnostic_summary(diagnostics), tables)


def test_render_payloads_rejects_error_rows_with_duplicate_fixed_sort_key():
    diagnostics = _load_module()
    tables = _render_tables()
    tables['error_examples.csv'] = [
        _error_example(
            'semantic_fp', 'same-image', 7, .9, class_id=1),
        _error_example(
            'semantic_fp', 'same-image', 7, .9, class_id=2),
    ]

    with pytest.raises(diagnostics.DiagnosticError, match='duplicate'):
        diagnostics.render_payloads(_diagnostic_summary(diagnostics), tables)


@pytest.mark.parametrize(
    'summary',
    [
        {},
        {'schema_version': True},
        {'schema_version': 2},
        {'schema_version': 1, 'bad': float('nan')},
    ],
)
def test_render_payloads_rejects_invalid_diagnostics_summary(summary):
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.render_payloads(summary, _render_tables())


def test_render_payloads_rejects_schema_one_summary_missing_report_facts():
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.render_payloads(
            {'schema_version': 1}, _render_tables())


@pytest.mark.parametrize(
    'invalid_case',
    [
        'canonical_string',
        'metrics_scalar',
        'metrics_missing',
        'metrics_bool',
        'metrics_nonfinite',
        'metrics_fraction_bool',
        'integrity_scalar',
        'parity_scalar',
        'gt_decomposition_scalar',
        'capacity_scalar',
        'calibration_scalar',
        'queries_scalar',
        'fp_regions_scalar',
        'fp_regions_missing',
        'fp_region_child_scalar',
        'groups_scalar',
        'groups_missing',
        'group_child_scalar',
        'case_gate_scalar',
        'case_gate_empty_list',
        'case_gate_nonmapping_row',
        'warnings_bool',
        'warnings_tuple',
    ],
)
def test_render_payloads_rejects_wrong_summary_structure(invalid_case):
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    if invalid_case == 'canonical_string':
        summary['canonical'] = 'yes'
    elif invalid_case == 'metrics_scalar':
        summary['metrics'] = 'bad'
    elif invalid_case == 'metrics_missing':
        del summary['metrics']['map']
    elif invalid_case == 'metrics_bool':
        summary['metrics']['map'] = True
    elif invalid_case == 'metrics_nonfinite':
        summary['metrics']['oracle_map'] = float('inf')
    elif invalid_case == 'metrics_fraction_bool':
        summary['metrics']['headroom_fraction_to_ap70'] = False
    elif invalid_case.endswith('_scalar') and invalid_case.split('_scalar')[
            0] in {
                'integrity', 'parity', 'gt_decomposition', 'capacity',
                'calibration', 'queries'}:
        summary[invalid_case[:-len('_scalar')]] = 7
    elif invalid_case == 'fp_regions_scalar':
        summary['fp_regions'] = 'bad'
    elif invalid_case == 'fp_regions_missing':
        del summary['fp_regions']['ap_support']
    elif invalid_case == 'fp_region_child_scalar':
        summary['fp_regions']['all'] = 7
    elif invalid_case == 'groups_scalar':
        summary['groups'] = 'bad'
    elif invalid_case == 'groups_missing':
        del summary['groups']['novel']
    elif invalid_case == 'group_child_scalar':
        summary['groups']['base'] = 7
    elif invalid_case == 'case_gate_scalar':
        summary['case_gate'] = 42
    elif invalid_case == 'case_gate_empty_list':
        summary['case_gate'] = []
    elif invalid_case == 'case_gate_nonmapping_row':
        summary['case_gate'] = [{'case_gate': 'complete'}, 7]
    elif invalid_case == 'warnings_bool':
        summary['warnings'] = False
    elif invalid_case == 'warnings_tuple':
        summary['warnings'] = ('warning',)

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.render_payloads(summary, _render_tables())


@pytest.mark.parametrize(
    ('field', 'invalid_value'),
    [
        ('ap50', .123),
        ('ap70_gap', .2),
        ('oracle_headroom', .4),
        ('headroom_fraction_to_ap70', .25),
    ],
)
def test_render_payloads_rejects_inconsistent_metric_math(
        field, invalid_value):
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    summary['metrics'][field] = invalid_value

    with pytest.raises(diagnostics.DiagnosticError, match='metric'):
        diagnostics.render_payloads(summary, _render_tables())


def test_render_payloads_accepts_complete_case_gate_rows_in_any_input_order():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    summary['case_gate'] = [
        {'case_gate': 'complete',
         'image_id': 'P0682__1024__553___0',
         'case_group': 'true_sv_safety'},
        {'case_gate': 'complete',
         'image_id': 'P0148__1024__651___0',
         'case_group': 'context_false_sv'},
    ]

    rendered = json.loads(diagnostics.render_payloads(
        summary, _render_tables())['diagnostics.json'])

    assert [row['image_id'] for row in rendered['case_gate']] == [
        'P0148__1024__651___0',
        'P0682__1024__553___0',
    ]


def test_render_payloads_clamps_tiny_negative_headroom_within_tolerance():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    summary['metrics'].update({
        'map': .6,
        'ap50': .6,
        'ap70_gap': .1,
        'oracle_map': .6,
        'oracle_headroom': -5e-13,
        'headroom_fraction_to_ap70': None,
    })

    rendered = json.loads(diagnostics.render_payloads(
        summary, _render_tables())['diagnostics.json'])

    assert rendered['metrics']['oracle_headroom'] == 0.0
    assert rendered['metrics']['headroom_fraction_to_ap70'] is None


def test_build_diagnostics_reuses_summary_structure_validation():
    diagnostics = _load_module()

    with pytest.raises(diagnostics.DiagnosticError, match='integrity'):
        diagnostics.build_diagnostics(
            True, 7, {}, .6, .65, {}, {}, {}, {}, {}, {}, {}, {}, {},
            {'status': 'not_run'}, [])


def test_report_has_six_traceable_sections_and_no_forbidden_claims():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    summary['metrics']['map'] = .75
    summary['metrics']['ap50'] = .75
    summary['metrics']['ap70_gap'] = -.05
    summary['metrics']['oracle_map'] = .8
    summary['metrics']['oracle_headroom'] = .05
    summary['metrics']['headroom_fraction_to_ap70'] = -1.0
    summary['warnings'] = ['decision-warning-unique']

    report = diagnostics.render_payloads(
        summary, _render_tables())['report.md'].decode('utf-8')

    assert report.endswith('\n') and not report.endswith('\n\n')
    headings = [
        '## Integrity and parity',
        '## Raw metrics/AP70 gap',
        '## Macro AP-support/ranking',
        '## Micro reachability/capacity',
        '## Base/novel and optional cases',
        '## Decision inputs',
    ]
    assert all(report.count(heading) == 1 for heading in headings)
    assert [report.index(heading) for heading in headings] == sorted(
        report.index(heading) for heading in headings)
    for traceable_value in (
            '"dump_sha256": "abc"',
            '"same_dump": true',
            '"map": 0.75',
            '"ap70_gap": -0.05',
            '"ap_support"',
            '"geometry_miss": 2',
            '"reachable": 3',
            '"base"',
            '"novel"',
            '"status": "not_run"',
            'decision-warning-unique'):
        assert traceable_value in report
    lowered = report.lower()
    assert 'causal' not in lowered
    assert '因果' not in report
    assert 'e25+' not in lowered
    assert 'achiev' not in lowered
    assert 'recommend' not in lowered
    assert 'new run' not in lowered


def test_report_depends_only_on_normalized_summary_not_tables():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    tables_a = _render_tables()
    tables_b = _render_tables()
    tables_b['per_class.csv'][0]['class_name'] = 'different table evidence'
    tables_b['error_examples.csv'][0]['class_id'] = 99

    report_a = diagnostics.render_payloads(summary, tables_a)['report.md']
    report_b = diagnostics.render_payloads(summary, tables_b)['report.md']

    assert report_a == report_b


def test_build_manifest_hashes_exact_payloads_without_hashing_itself():
    diagnostics = _load_module()
    payloads = diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), _render_tables())
    provenance = {'checkpoint': {'step': np.int64(24)}}
    command = ['python', 'diagnostics.py', '--quoted=a b']
    environment = {'CUDA_VISIBLE_DEVICES': '', 'workers': np.int64(1)}
    original_payloads = dict(payloads)
    original_provenance = copy.deepcopy(provenance)

    manifest_bytes = diagnostics.build_manifest(
        payloads, provenance, command, environment)
    repeated = diagnostics.build_manifest(
        dict(reversed(list(payloads.items()))),
        provenance, command, environment)
    manifest = json.loads(manifest_bytes)
    provenance['checkpoint']['step'] = 99

    assert manifest_bytes == repeated
    assert manifest_bytes.endswith(b'\n')
    assert not manifest_bytes.endswith(b'\n\n')
    assert manifest == {
        'schema_version': 1,
        'files': {
            name: {
                'sha256': hashlib.sha256(payload).hexdigest(),
                'bytes': len(payload),
            }
            for name, payload in sorted(payloads.items())
        },
        'provenance': original_provenance,
        'command': command,
        'environment': {'CUDA_VISIBLE_DEVICES': '', 'workers': 1},
    }
    assert list(manifest['files']) == sorted(payloads)
    assert 'manifest.json' not in manifest['files']
    for name, payload in payloads.items():
        assert manifest['files'][name]['sha256'] == hashlib.sha256(
            payload).hexdigest()
        assert manifest['files'][name]['bytes'] == len(payload)
    assert payloads == original_payloads


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'self'])
def test_build_manifest_rejects_nonexact_payload_names(mutation):
    diagnostics = _load_module()
    payloads = diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), _render_tables())
    if mutation == 'missing':
        del payloads['report.md']
    elif mutation == 'extra':
        payloads['extra.txt'] = b'extra\n'
    else:
        payloads['manifest.json'] = b'{}\n'

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.build_manifest(payloads, {}, [], {})


@pytest.mark.parametrize('invalid_payload', [bytearray(b'x'), 'x', None])
def test_build_manifest_rejects_nonbytes_payload(invalid_payload):
    diagnostics = _load_module()
    payloads = diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), _render_tables())
    payloads['report.md'] = invalid_payload

    with pytest.raises(diagnostics.DiagnosticError, match='bytes'):
        diagnostics.build_manifest(payloads, {}, [], {})


@pytest.mark.parametrize(
    ('field', 'invalid'),
    [
        ('provenance', {'bad': float('inf')}),
        ('provenance', {'1': 'string', 1: 'integer'}),
        ('command', object()),
        ('environment', {'bad': np.array([1])}),
    ],
)
def test_build_manifest_rejects_non_json_safe_metadata(field, invalid):
    diagnostics = _load_module()
    payloads = diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), _render_tables())
    kwargs = {'provenance': {}, 'command': [], 'environment': {}}
    kwargs[field] = invalid

    with pytest.raises(diagnostics.DiagnosticError):
        diagnostics.build_manifest(payloads, **kwargs)


@pytest.mark.parametrize(
    'invalid_numeric',
    [Fraction(1, 2), _CustomReal(.5), Decimal('.5'), 1 + 2j],
)
@pytest.mark.parametrize('entrypoint', ['summary', 'table', 'provenance'])
def test_recursive_normalization_rejects_non_native_numeric_types(
        entrypoint, invalid_numeric):
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    tables = _render_tables()

    with pytest.raises(diagnostics.DiagnosticError):
        if entrypoint == 'summary':
            summary['integrity']['invalid_numeric'] = invalid_numeric
            diagnostics.render_payloads(summary, tables)
        elif entrypoint == 'table':
            tables['per_class.csv'][0]['ap'] = invalid_numeric
            diagnostics.render_payloads(summary, tables)
        else:
            payloads = diagnostics.render_payloads(summary, tables)
            diagnostics.build_manifest(
                payloads,
                provenance={'invalid_numeric': invalid_numeric},
                command=[],
                environment={},
            )


def test_recursive_normalization_keeps_supported_numpy_scalars():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    summary['integrity']['numpy_scalars'] = [
        np.int32(3),
        np.uint64(4),
        np.float32(.5),
        np.bool_(True),
        np.str_('text'),
    ]

    rendered = json.loads(diagnostics.render_payloads(
        summary, _render_tables())['diagnostics.json'])

    assert rendered['integrity']['numpy_scalars'] == [
        3, 4, pytest.approx(.5), True, 'text']


@pytest.mark.parametrize('cycle_kind', ['list', 'mapping'])
def test_recursive_normalization_rejects_summary_cycles(cycle_kind):
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    if cycle_kind == 'list':
        cycle = []
        cycle.append(cycle)
    else:
        cycle = {}
        cycle['self'] = cycle
    summary['integrity']['cycle'] = cycle

    with pytest.raises(diagnostics.DiagnosticError, match='cycl|depth'):
        diagnostics.render_payloads(summary, _render_tables())


def test_recursive_normalization_rejects_manifest_command_cycle():
    diagnostics = _load_module()
    payloads = diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), _render_tables())
    command = []
    command.append(command)

    with pytest.raises(diagnostics.DiagnosticError, match='cycl|depth'):
        diagnostics.build_manifest(payloads, {}, command, {})


def test_recursive_normalization_rejects_excessive_depth():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    nested = 'leaf'
    for _ in range(70):
        nested = [nested]
    summary['integrity']['nested'] = nested

    with pytest.raises(diagnostics.DiagnosticError, match='depth'):
        diagnostics.render_payloads(summary, _render_tables())


def test_recursive_normalization_allows_shared_noncyclic_alias():
    diagnostics = _load_module()
    summary = _diagnostic_summary(diagnostics)
    shared = {'value': [1, 2]}
    summary['integrity']['left'] = shared
    summary['integrity']['right'] = shared

    rendered = json.loads(diagnostics.render_payloads(
        summary, _render_tables())['diagnostics.json'])

    assert rendered['integrity']['left'] == {'value': [1, 2]}
    assert rendered['integrity']['right'] == {'value': [1, 2]}


def test_recursive_normalization_wraps_mapping_items_failure():
    diagnostics = _load_module()

    class ExplodingItems(dict):

        def items(self):
            yield 'before', 1
            raise RuntimeError('items exploded')

    summary = _diagnostic_summary(diagnostics)
    summary['integrity']['bad_mapping'] = ExplodingItems()

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.render_payloads(summary, _render_tables())

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_cap_error_examples_wraps_mapping_access_failure_with_cause():
    diagnostics = _load_module()

    class ExplodingRow(dict):

        def __getitem__(self, key):
            raise RuntimeError('row access exploded')

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.cap_error_examples(
            [ExplodingRow(_error_example('semantic_fp', 'a', 0, .9))],
            max_per_type=1,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_render_payloads_wraps_summary_mapping_access_failure_with_cause():
    diagnostics = _load_module()

    class ExplodingSummary(dict):

        def __getitem__(self, key):
            raise RuntimeError('summary access exploded')

    summary = ExplodingSummary(_diagnostic_summary(diagnostics))

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.render_payloads(summary, _render_tables())

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_render_payloads_wraps_table_mapping_access_failure_with_cause():
    diagnostics = _load_module()

    class ExplodingTables(dict):

        def __getitem__(self, key):
            raise RuntimeError('table access exploded')

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.render_payloads(
            _diagnostic_summary(diagnostics),
            ExplodingTables(_render_tables()),
        )

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_render_payloads_wraps_table_row_iteration_failure_with_cause():
    diagnostics = _load_module()
    failure = RuntimeError('table rows exploded')

    def rows():
        yield {'class_id': 0}
        raise failure

    tables = _render_tables()
    tables['per_class.csv'] = rows()

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.render_payloads(_diagnostic_summary(diagnostics), tables)

    assert caught.value.__cause__ is failure


def test_build_manifest_wraps_payload_mapping_access_failure_with_cause():
    diagnostics = _load_module()

    class ExplodingPayloads(dict):

        def __getitem__(self, key):
            raise RuntimeError('payload access exploded')

    payloads = diagnostics.render_payloads(
        _diagnostic_summary(diagnostics), _render_tables())

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        diagnostics.build_manifest(
            ExplodingPayloads(payloads), {}, [], {})

    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize('entrypoint', ['tables', 'payloads'])
@pytest.mark.parametrize('failure_kind', ['diagnostic', 'runtime'])
def test_mapping_key_iteration_preserves_or_wraps_exceptions(
        entrypoint, failure_kind):
    diagnostics = _load_module()
    failure = (
        diagnostics.DiagnosticError('keys already classified')
        if failure_kind == 'diagnostic'
        else RuntimeError('keys exploded')
    )

    class ExplodingKeys(Mapping):

        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            return self.values[key]

        def __iter__(self):
            yield next(iter(self.values))
            raise failure

        def __len__(self):
            return len(self.values)

    if entrypoint == 'tables':
        invoke = lambda: diagnostics.render_payloads(  # noqa: E731
            _diagnostic_summary(diagnostics), ExplodingKeys(_render_tables()))
    else:
        payloads = diagnostics.render_payloads(
            _diagnostic_summary(diagnostics), _render_tables())
        invoke = lambda: diagnostics.build_manifest(  # noqa: E731
            ExplodingKeys(payloads), {}, [], {})

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        invoke()

    if failure_kind == 'diagnostic':
        assert caught.value is failure
    else:
        assert caught.value.__cause__ is failure


@pytest.mark.parametrize('entrypoint', ['tables', 'payloads'])
def test_mapping_getter_preserves_existing_diagnostic_error(entrypoint):
    diagnostics = _load_module()
    failure = diagnostics.DiagnosticError('getter already classified')

    class ExplodingGetter(Mapping):

        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            raise failure

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

    if entrypoint == 'tables':
        invoke = lambda: diagnostics.render_payloads(  # noqa: E731
            _diagnostic_summary(diagnostics),
            ExplodingGetter(_render_tables()),
        )
    else:
        payloads = diagnostics.render_payloads(
            _diagnostic_summary(diagnostics), _render_tables())
        invoke = lambda: diagnostics.build_manifest(  # noqa: E731
            ExplodingGetter(payloads), {}, [], {})

    with pytest.raises(diagnostics.DiagnosticError) as caught:
        invoke()

    assert caught.value is failure


def _parse_analyzer_args(analyzer, tmp_path, *extra):
    required = [
        str(tmp_path / 'dump.pkl'),
        '--config', str(tmp_path / 'config.py'),
        '--checkpoint', str(tmp_path / 'checkpoint.pth'),
        '--official-metrics-json', str(tmp_path / 'official.json'),
        '--output-dir', str(tmp_path / 'output'),
    ]
    return analyzer.build_parser().parse_args(required + list(extra))


def test_analyzer_parser_defaults_and_canonical_mode(tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    args = _parse_analyzer_args(
        analyzer, tmp_path,
        '--training-metrics-json', str(tmp_path / 'training.json'))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)

    assert args.training_step == 24
    assert args.iou_threshold == pytest.approx(.5)
    assert args.sample_size == 500000
    assert args.sample_seed == 20260722
    assert args.max_error_examples == 100
    assert (args.expected_records, args.queries_per_image,
            args.num_classes) == (13833, 600, 18)
    assert analyzer.validate_mode(args) is True


def test_analyzer_mode_requires_explicit_noncanonical_escape_hatch(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    for extra in (
            ('--expected-records', '1'),
            ('--queries-per-image', '1'),
            ('--num-classes', '1'),
            ('--iou-threshold', '.6')):
        args = _parse_analyzer_args(
            analyzer, tmp_path,
            '--training-metrics-json', str(tmp_path / 'training.json'),
            *extra)
        with pytest.raises(
                analyzer.DiagnosticError, match='allow-noncanonical'):
            analyzer.validate_mode(args)

    escaped = _parse_analyzer_args(
        analyzer, tmp_path, '--allow-noncanonical',
        '--expected-records', '1', '--queries-per-image', '1',
        '--num-classes', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    assert analyzer.validate_mode(escaped) is False


@pytest.mark.parametrize(
    ('extra', 'match'),
    [
        ((), 'training'),
        (('--training-metrics-json', 'training.json', '--training-step', '23'),
         '24'),
    ],
)
def test_analyzer_canonical_training_gate(tmp_path, monkeypatch, extra, match):
    analyzer = _load_analyzer()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    args = _parse_analyzer_args(analyzer, tmp_path, *extra)

    with pytest.raises(analyzer.DiagnosticError, match=match):
        analyzer.validate_mode(args)


def test_analyzer_canonical_rejects_visible_cuda(tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    args = _parse_analyzer_args(
        analyzer, tmp_path,
        '--training-metrics-json', str(tmp_path / 'training.json'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

    with pytest.raises(analyzer.DiagnosticError, match='CUDA_VISIBLE_DEVICES'):
        analyzer.validate_mode(args)


def test_analyzer_case_inputs_must_be_paired(tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    for option in ('--case-dump', '--case-manifest'):
        args = _parse_analyzer_args(
            analyzer, tmp_path, '--allow-noncanonical', option,
            str(tmp_path / 'case'))
        with pytest.raises(analyzer.DiagnosticError, match='together|pair'):
            analyzer.validate_mode(args)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('expected_records', 0),
        ('queries_per_image', -1),
        ('num_classes', True),
        ('iou_threshold', -0.1),
        ('iou_threshold', 1.1),
        ('sample_size', True),
        ('sample_size', -1),
        ('sample_seed', True),
        ('sample_seed', 1.5),
        ('max_error_examples', True),
        ('max_error_examples', -1),
    ],
)
def test_analyzer_mode_numeric_gates_fail_closed(
        tmp_path, monkeypatch, field, value):
    analyzer = _load_analyzer()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    args = _parse_analyzer_args(analyzer, tmp_path, '--allow-noncanonical')
    setattr(args, field, value)

    with pytest.raises(analyzer.DiagnosticError):
        analyzer.validate_mode(args)


@pytest.mark.parametrize('missing_field', [
    'dump', 'config', 'checkpoint', 'official_metrics_json', 'output_dir',
    'training_metrics_json', 'training_step', 'iou_threshold', 'sample_size',
    'sample_seed', 'max_error_examples', 'case_dump', 'case_manifest',
    'allow_noncanonical', 'expected_records', 'queries_per_image',
    'num_classes',
])
def test_analyzer_mode_missing_namespace_field_is_diagnostic(
        tmp_path, monkeypatch, missing_field):
    analyzer = _load_analyzer()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    args = _parse_analyzer_args(
        analyzer, tmp_path, '--allow-noncanonical')
    delattr(args, missing_field)

    with pytest.raises(analyzer.DiagnosticError) as caught:
        analyzer.validate_mode(args)

    message = str(caught.value).lower()
    assert 'missing' in message
    assert missing_field in message


def test_atomic_output_directory_publishes_by_rename_and_fsyncs(tmp_path,
                                                                monkeypatch):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'
    fsynced = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        fsynced.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(analyzer.os, 'fsync', recording_fsync)
    with analyzer.atomic_output_directory(output) as temporary:
        assert temporary.parent == output.parent
        assert temporary.name.startswith(output.name + '.')
        (temporary / 'partial.txt').write_text('complete', encoding='utf-8')

    assert (output / 'partial.txt').read_text(encoding='utf-8') == 'complete'
    assert not temporary.exists()
    assert len(fsynced) >= 1


def test_atomic_output_directory_refuses_existing_destination(tmp_path):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'
    output.mkdir()
    marker = output / 'owned.txt'
    marker.write_text('preserve', encoding='utf-8')

    with pytest.raises(analyzer.DiagnosticError, match='already exists'):
        with analyzer.atomic_output_directory(output):
            pytest.fail('existing destination must not yield')

    assert marker.read_text(encoding='utf-8') == 'preserve'
    assert not list(tmp_path.glob('bundle.*.failed'))


def test_atomic_output_directory_refuses_dangling_symlink_destination(
        tmp_path):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'
    output.symlink_to(tmp_path / 'missing-target', target_is_directory=True)

    with pytest.raises(analyzer.DiagnosticError, match='already exists'):
        with analyzer.atomic_output_directory(output):
            pytest.fail('occupied destination must not yield')

    assert output.is_symlink()
    assert os.readlink(output) == str(tmp_path / 'missing-target')
    assert not list(tmp_path.glob('bundle.*.failed'))


def test_atomic_output_directory_preserves_partial_on_base_exception(tmp_path):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'

    with pytest.raises(RuntimeError, match='explode'):
        with analyzer.atomic_output_directory(output) as temporary:
            (temporary / 'partial.txt').write_text(
                'preserved', encoding='utf-8')
            raise RuntimeError('explode')

    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert (failed[0] / 'partial.txt').read_text(
        encoding='utf-8') == 'preserved'
    assert not output.exists()


def test_atomic_publication_race_never_overwrites_destination(tmp_path):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'

    with pytest.raises(analyzer.DiagnosticError, match='already exists'):
        with analyzer.atomic_output_directory(output) as temporary:
            (temporary / 'ours.txt').write_text('ours', encoding='utf-8')
            output.mkdir()
            (output / 'racer.txt').write_text('theirs', encoding='utf-8')

    assert (output / 'racer.txt').read_text(encoding='utf-8') == 'theirs'
    assert not (output / 'ours.txt').exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert (failed[0] / 'ours.txt').read_text(encoding='utf-8') == 'ours'


def test_atomic_post_rename_parent_fsync_failure_rolls_back_eight_files(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'
    original_error = OSError('post-rename parent fsync failed')
    real_fsync_directory = analyzer._fsync_directory
    observed = []
    expected_names = set(analyzer.NON_MANIFEST_PAYLOADS) | {'manifest.json'}

    def fail_first_parent_fsync(path):
        failed = list(tmp_path.glob('bundle.*.failed'))
        observed.append({
            'path': Path(path),
            'output': output.exists(),
            'failed_count': len(failed),
        })
        if len(observed) == 1:
            assert Path(path) == tmp_path
            assert output.exists()
            assert not failed
            raise original_error
        return real_fsync_directory(path)

    monkeypatch.setattr(
        analyzer, '_fsync_directory', fail_first_parent_fsync)

    with pytest.raises(OSError) as caught:
        with analyzer.atomic_output_directory(output) as temporary:
            for name in expected_names:
                (temporary / name).write_bytes(name.encode('utf-8'))

    assert caught.value is original_error
    assert not output.exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert {path.name for path in failed[0].iterdir()} == expected_names
    assert not (failed[0] / 'failure.json').exists()
    assert len(observed) == 2
    assert observed[1]['output'] is False
    assert observed[1]['failed_count'] == 1


def test_atomic_rollback_fsync_failure_keeps_failed_and_chains_original(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'
    publication_error = OSError('publication fsync failed')
    rollback_error = OSError('rollback fsync failed')
    calls = 0

    def fail_both_parent_fsync(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert output.exists()
            raise publication_error
        assert not output.exists()
        assert len(list(tmp_path.glob('bundle.*.failed'))) == 1
        raise rollback_error

    monkeypatch.setattr(
        analyzer, '_fsync_directory', fail_both_parent_fsync)

    with pytest.raises(OSError) as caught:
        with analyzer.atomic_output_directory(output) as temporary:
            (temporary / 'partial.txt').write_text(
                'preserved', encoding='utf-8')

    assert caught.value is rollback_error
    assert caught.value.__cause__ is publication_error
    assert not output.exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert (failed[0] / 'partial.txt').read_text(
        encoding='utf-8') == 'preserved'


def test_atomic_expected_late_body_failure_replaces_partial_with_failure_json(
        tmp_path):
    analyzer = _load_analyzer()
    output = tmp_path / 'bundle'
    command = ['/absolute/analyzer.py', '--example']
    original_error = analyzer.DiagnosticError('late expected failure')

    with pytest.raises(analyzer.DiagnosticError) as caught:
        with analyzer.atomic_output_directory(
                output, failure_command=command) as temporary:
            (temporary / 'diagnostics.json').write_text(
                '{}\n', encoding='utf-8')
            (temporary / 'manifest.json').write_text(
                '{}\n', encoding='utf-8')
            raise original_error

    assert caught.value is original_error
    assert not output.exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert {path.name for path in failed[0].iterdir()} == {'failure.json'}
    failure = json.loads((failed[0] / 'failure.json').read_text(
        encoding='utf-8'))
    assert failure == {
        'schema_version': 1,
        'canonical': False,
        'error': 'late expected failure',
        'command': command,
    }


def _synthetic_analyzer_inputs(tmp_path, official_map=1.0):
    config = tmp_path / 'config.py'
    config.write_text(
        "classes = ('small-vehicle',)\n"
        "base_classes = ('small-vehicle',)\n"
        "novel_classes = ()\n",
        encoding='utf-8')
    checkpoint = tmp_path / 'checkpoint.pth'
    torch.save({
        'state_dict': {
            'model.query_initializer.reference_embedding.weight':
                torch.zeros((1, 5), dtype=torch.float32),
        },
    }, checkpoint)
    dump = tmp_path / 'dump.pkl'
    square = [512, 512, 512, 512, 0]
    records = [_record(
        'tile', [square], [.9], [0], gt_boxes=[square], gt_labels=[0])]
    with dump.open('wb') as stream:
        pickle.dump(records, stream)
    official = tmp_path / 'official.json'
    official.write_text(json.dumps({
        'dota/mAP': official_map,
        'dota/AP50': round(official_map, 3),
    }) + '\n', encoding='utf-8')
    return config, checkpoint, dump, official, records


def _canonical_optimized_inputs(tmp_path):
    classes = ('small-vehicle',) + tuple(
        'class-{}'.format(index) for index in range(1, 18))
    config = tmp_path / 'canonical.py'
    config.write_text(
        'classes = {!r}\nbase_classes = {!r}\nnovel_classes = {!r}\n'.
        format(classes, classes[:14], classes[14:]),
        encoding='utf-8')
    checkpoint = tmp_path / 'canonical.pth'
    torch.save({
        'state_dict': {
            'model.query_initializer.reference_embedding.weight':
                torch.zeros((600, 5), dtype=torch.float32),
        },
    }, checkpoint)
    dump = tmp_path / 'canonical.pkl'
    square = [512, 512, 512, 512, 0]
    record = _record(
        'only-one-tile', [square] * 600,
        np.linspace(.99, .01, 600), [0] * 600,
        gt_boxes=[square], gt_labels=[0])
    with dump.open('wb') as stream:
        pickle.dump([record], stream)
    official = tmp_path / 'official.json'
    official.write_text(json.dumps({
        'dota/mAP': 1.0, 'dota/AP50': 1.0,
    }) + '\n', encoding='utf-8')
    training = tmp_path / 'training.json'
    training.write_text(json.dumps({
        'step': 24, 'dota/mAP': 1.0, 'dota/AP50': 1.0,
    }) + '\n', encoding='utf-8')
    return config, checkpoint, dump, official, training


def _analyzer_argv(config, checkpoint, dump, official, output, *extra):
    return [
        str(dump), '--config', str(config), '--checkpoint', str(checkpoint),
        '--official-metrics-json', str(official), '--output-dir', str(output),
        '--allow-noncanonical', '--expected-records', '1',
        '--queries-per-image', '1', '--num-classes', '1', *extra,
    ]


def _assert_finite_json(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_json(item)
    elif isinstance(value, float):
        assert np.isfinite(value)


def test_analyzer_noncanonical_synthetic_e2e_publishes_exact_bundle(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    argv = _analyzer_argv(config, checkpoint, dump, official, output)
    exit_code = analyzer.main(argv)

    assert exit_code == 0
    expected_names = set(analyzer.NON_MANIFEST_PAYLOADS) | {'manifest.json'}
    assert {path.name for path in output.iterdir()} == expected_names
    diagnostics = json.loads((output / 'diagnostics.json').read_text(
        encoding='utf-8'))
    assert diagnostics['canonical'] is False
    assert diagnostics['integrity']['records'] == 1
    assert diagnostics['integrity']['unique_image_ids'] == 1
    assert diagnostics['integrity']['prediction_rows'] == 1
    assert diagnostics['parity']['exact_error'] <= 1e-7
    assert diagnostics['case_gate'] == [{
        'case_gate': 'not_run', 'image_id': None, 'case_group': None}]
    _assert_finite_json(diagnostics)

    manifest = json.loads((output / 'manifest.json').read_text(
        encoding='utf-8'))
    assert manifest['command'] == [
        sys.executable, str(ANALYZER_PATH.resolve()), *argv]
    assert manifest['environment']['cwd'] == str(Path.cwd().resolve())
    assert set(manifest['files']) == set(analyzer.NON_MANIFEST_PAYLOADS)
    assert 'manifest.json' not in manifest['files']
    for name, metadata in manifest['files'].items():
        payload = (output / name).read_bytes()
        assert metadata == {
            'sha256': hashlib.sha256(payload).hexdigest(),
            'bytes': len(payload),
        }
    _assert_finite_json(manifest)


def test_analyzer_script_help_exits_zero():
    completed = subprocess.run(
        [sys.executable, str(ANALYZER_PATH), '--help'],
        check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 0
    assert '--official-metrics-json' in completed.stdout


def test_analyzer_python_optimized_canonical_count_gate_is_explicit(tmp_path):
    config, checkpoint, dump, official, training = (
        _canonical_optimized_inputs(tmp_path))
    output = tmp_path / 'optimized-bundle'
    environment = os.environ.copy()
    environment.update({
        'PYTHONNOUSERSITE': '1',
        'CUDA_VISIBLE_DEVICES': '',
        'OMP_NUM_THREADS': '1',
        'PYTHONPATH': str(Path(__file__).parents[3]),
    })

    completed = subprocess.run([
        sys.executable, '-O', str(ANALYZER_PATH), str(dump),
        '--config', str(config), '--checkpoint', str(checkpoint),
        '--official-metrics-json', str(official),
        '--training-metrics-json', str(training),
        '--output-dir', str(output),
    ], check=False, capture_output=True, text=True, env=environment)

    assert completed.returncode == 2
    assert not output.exists()
    failed = list(tmp_path.glob('optimized-bundle.*.failed'))
    assert len(failed) == 1
    assert {path.name for path in failed[0].iterdir()} == {'failure.json'}
    failure = json.loads((failed[0] / 'failure.json').read_text(
        encoding='utf-8'))
    assert failure['canonical'] is False
    assert 'record' in failure['error'].lower()
    assert '13833' in failure['error']


def test_analyzer_explicit_dump_gate_rejects_inconsistent_integrity(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    _, _, dump, _, records = _synthetic_analyzer_inputs(tmp_path)
    monkeypatch.setattr(analyzer, 'load_cpu', lambda path: records)
    monkeypatch.setattr(analyzer, 'validate_records', lambda *args, **kwargs: {
        'records': 999,
        'unique_image_ids': 999,
        'prediction_rows': 999,
        'all_cpu_finite': True,
    })

    with pytest.raises(analyzer.DiagnosticError, match='integrity|record'):
        analyzer._load_validated_dump(dump, 1, 1, 1)


def test_analyzer_canonical_case_gate_is_explicit_when_asserts_are_absent(
        monkeypatch):
    analyzer = _load_analyzer()
    square = [10, 10, 4, 4, 0]
    case_raw = [_record('one-case', [square], [.9], [0])]
    monkeypatch.setattr(
        analyzer, 'validate_records', lambda *args, **kwargs: {})

    with pytest.raises(analyzer.DiagnosticError, match='case.*record|2'):
        analyzer._validate_canonical_case_records(case_raw, 1, 1)


@pytest.mark.parametrize('occupied_kind', ['directory', 'dangling_symlink'])
def test_analyzer_existing_output_refusal_has_zero_input_reads(
        tmp_path, monkeypatch, capsys, occupied_kind):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    output = tmp_path / 'bundle'
    marker = None
    if occupied_kind == 'directory':
        output.mkdir()
        marker = output / 'owned.txt'
        marker.write_text('preserve', encoding='utf-8')
    else:
        output.symlink_to(
            tmp_path / 'missing-competitor', target_is_directory=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    calls = {
        name: 0 for name in (
            'load_class_protocol', 'read_metric_record', '_file_provenance',
            '_load_checkpoint', '_load_validated_dump',
            'load_case_manifest', 'load_cpu')
    }

    def forbidden_read(name):
        def fail(*args, **kwargs):
            calls[name] += 1
            pytest.fail('input read before existing-output refusal: {}'.format(
                name))
        return fail

    for name in calls:
        monkeypatch.setattr(analyzer, name, forbidden_read(name))

    exit_code = analyzer.main(_analyzer_argv(
        config, checkpoint, dump, official, output))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert len(captured.err.splitlines()) == 1
    assert calls == {name: 0 for name in calls}
    if occupied_kind == 'directory':
        assert marker.read_text(encoding='utf-8') == 'preserve'
        assert not (output / 'diagnostics.json').exists()
    else:
        assert output.is_symlink()
        assert os.readlink(output) == str(tmp_path / 'missing-competitor')
    assert not list(tmp_path.glob('bundle.*.failed'))


def test_analyzer_publication_eexist_keeps_competitor_and_only_failure_json(
        tmp_path, monkeypatch, capsys):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    real_rename_no_replace = analyzer._rename_no_replace

    def race_with_competitor(source, destination):
        destination.mkdir()
        (destination / 'competitor.txt').write_text(
            'theirs', encoding='utf-8')
        return real_rename_no_replace(source, destination)

    monkeypatch.setattr(
        analyzer, '_rename_no_replace', race_with_competitor)

    exit_code = analyzer.main(_analyzer_argv(
        config, checkpoint, dump, official, output))

    assert exit_code == 2
    assert len(capsys.readouterr().err.splitlines()) == 1
    assert {path.name for path in output.iterdir()} == {'competitor.txt'}
    assert (output / 'competitor.txt').read_text(
        encoding='utf-8') == 'theirs'
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert {path.name for path in failed[0].iterdir()} == {'failure.json'}
    failure = json.loads((failed[0] / 'failure.json').read_text(
        encoding='utf-8'))
    assert failure['schema_version'] == 1
    assert failure['canonical'] is False
    assert 'already exists' in failure['error']
    assert failure['command'][:2] == [
        sys.executable, str(ANALYZER_PATH.resolve())]
    assert not (failed[0] / 'diagnostics.json').exists()


@pytest.mark.parametrize('failure_kind', ['parity', 'invalid_dump'])
def test_analyzer_expected_failure_publishes_only_failed_failure_json(
        tmp_path, monkeypatch, capsys, failure_kind):
    analyzer = _load_analyzer()
    official_map = .5 if failure_kind == 'parity' else 1.0
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path, official_map=official_map)
    if failure_kind == 'invalid_dump':
        invalid = [_record('tile', [], [], [], [], [])]
        with dump.open('wb') as stream:
            pickle.dump(invalid, stream)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    exit_code = analyzer.main(_analyzer_argv(
        config, checkpoint, dump, official, output))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert len(captured.err.splitlines()) == 1
    assert not output.exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert {path.name for path in failed[0].iterdir()} == {'failure.json'}
    failure = json.loads((failed[0] / 'failure.json').read_text(
        encoding='utf-8'))
    assert failure['schema_version'] == 1
    assert failure['canonical'] is False
    assert '\n' not in failure['error'] and '\r' not in failure['error']
    assert failure['command'][:2] == [
        sys.executable, str(ANALYZER_PATH.resolve())]
    assert not (failed[0] / 'diagnostics.json').exists()


def test_analyzer_corrupt_dump_is_expected_failed_publication(
        tmp_path, monkeypatch, capsys):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    dump.write_bytes(b'not a pickle')
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    exit_code = analyzer.main(_analyzer_argv(
        config, checkpoint, dump, official, output))

    assert exit_code == 2
    assert len(capsys.readouterr().err.splitlines()) == 1
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert {path.name for path in failed[0].iterdir()} == {'failure.json'}


@pytest.mark.parametrize('loader_kind', ['checkpoint', 'dump'])
def test_analyzer_primary_loaders_propagate_memory_error(
        tmp_path, monkeypatch, loader_kind):
    analyzer = _load_analyzer()
    memory_error = MemoryError('{} exhausted'.format(loader_kind))
    if loader_kind == 'checkpoint':
        monkeypatch.setattr(
            analyzer.torch, 'load', lambda *args, **kwargs: (_ for _ in ()).
            throw(memory_error))
        invoke = lambda: analyzer._load_checkpoint(  # noqa: E731
            tmp_path / 'checkpoint.pth', 1, False)
    else:
        monkeypatch.setattr(
            analyzer, 'load_cpu', lambda *args, **kwargs: (_ for _ in ()).
            throw(memory_error))
        invoke = lambda: analyzer._load_validated_dump(  # noqa: E731
            tmp_path / 'dump.pkl', 1, 1, 1)

    with pytest.raises(MemoryError) as caught:
        invoke()

    assert caught.value is memory_error


def test_analyzer_optional_case_loader_propagates_memory_error(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    case_dump, case_manifest = _case_analyzer_inputs(tmp_path)
    output = tmp_path / 'bundle'
    memory_error = MemoryError('optional case exhausted')
    real_load_cpu = analyzer.load_cpu
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    def load_or_exhaust(path):
        if Path(path).resolve() == case_dump.resolve():
            raise memory_error
        return real_load_cpu(path)

    monkeypatch.setattr(analyzer, 'load_cpu', load_or_exhaust)

    with pytest.raises(MemoryError) as caught:
        analyzer.main(_analyzer_argv(
            config, checkpoint, dump, official, output,
            '--case-dump', str(case_dump),
            '--case-manifest', str(case_manifest)))

    assert caught.value is memory_error
    assert not output.exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert not (failed[0] / 'failure.json').exists()
    assert not (failed[0] / 'diagnostics.json').exists()


def test_analyzer_unexpected_failure_propagates_and_preserves_partial(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    def explode(*args, **kwargs):
        raise RuntimeError('unexpected explosion')

    monkeypatch.setattr(analyzer, 'load_class_protocol', explode)
    with pytest.raises(RuntimeError, match='unexpected explosion'):
        analyzer.main(_analyzer_argv(
            config, checkpoint, dump, official, output))

    assert not output.exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert not (failed[0] / 'failure.json').exists()
    assert not (failed[0] / 'diagnostics.json').exists()


def _case_analyzer_inputs(tmp_path):
    square = [512, 512, 512, 512, 0]
    case_dump = tmp_path / 'cases.pkl'
    case_records = [
        _record('P0148__1024__651___0', [square], [.8], [0]),
        _record(
            'P0682__1024__553___0', [square], [.9], [0],
            gt_boxes=[square], gt_labels=[0]),
    ]
    with case_dump.open('wb') as stream:
        pickle.dump(case_records, stream)
    case_manifest = tmp_path / 'cases.json'
    case_manifest.write_text(json.dumps({
        'context_false_sv': ['P0148__1024__651___0'],
        'true_sv_safety': ['P0682__1024__553___0'],
    }) + '\n', encoding='utf-8')
    return case_dump, case_manifest


def test_analyzer_optional_cases_are_isolated_from_canonical_bundle(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    case_dump, case_manifest = _case_analyzer_inputs(tmp_path)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    exit_code = analyzer.main(_analyzer_argv(
        config, checkpoint, dump, official, output,
        '--case-dump', str(case_dump),
        '--case-manifest', str(case_manifest)))

    assert exit_code == 0
    diagnostics = json.loads((output / 'diagnostics.json').read_text(
        encoding='utf-8'))
    assert diagnostics['integrity']['records'] == 1
    assert diagnostics['integrity']['prediction_rows'] == 1
    assert diagnostics['metrics']['map'] == pytest.approx(1.0)
    assert [row['case_gate'] for row in diagnostics['case_gate']] == [
        'complete', 'complete']
    case_rows = list(csv.DictReader((output / 'case_studies.csv').open(
        encoding='utf-8', newline='')))
    assert len(case_rows) == 2


def test_analyzer_resolves_metric_file_and_hashes_all_inputs_before_load(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    metrics_dir = tmp_path / 'metrics'
    metrics_dir.mkdir()
    selected_metric = metrics_dir / 'selected.json'
    official.rename(selected_metric)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    events = []
    real_provenance = analyzer._file_provenance
    real_checkpoint = analyzer._load_checkpoint
    real_dump = analyzer._load_validated_dump

    def record_hash(path):
        result = real_provenance(path)
        events.append(('hash', Path(result['path'])))
        return result

    def record_checkpoint(*args, **kwargs):
        assert {config.resolve(), checkpoint.resolve(), dump.resolve(),
                selected_metric.resolve()} <= {
                    path for event, path in events if event == 'hash'}
        events.append(('load', checkpoint.resolve()))
        return real_checkpoint(*args, **kwargs)

    def record_dump(*args, **kwargs):
        assert ('load', checkpoint.resolve()) in events
        events.append(('load', dump.resolve()))
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(analyzer, '_file_provenance', record_hash)
    monkeypatch.setattr(analyzer, '_load_checkpoint', record_checkpoint)
    monkeypatch.setattr(analyzer, '_load_validated_dump', record_dump)

    assert analyzer.main(_analyzer_argv(
        config, checkpoint, dump, metrics_dir, output)) == 0

    manifest = json.loads((output / 'manifest.json').read_text(
        encoding='utf-8'))
    official_provenance = manifest['provenance']['official_metrics']
    assert official_provenance['path'] == str(selected_metric.resolve())
    assert official_provenance['sha256'] == hashlib.sha256(
        selected_metric.read_bytes()).hexdigest()
    assert official_provenance['bytes'] == selected_metric.stat().st_size
    assert manifest['environment']['torch_num_threads'] == 1
    assert manifest['environment']['torch_num_interop_threads'] == 1
    assert manifest['environment']['cwd'] == str(Path.cwd().resolve())


def test_analyzer_direct_run_reconstructs_complete_replay_command(
        tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    training = tmp_path / 'training.json'
    training.write_text(json.dumps({
        'step': 24, 'dota/mAP': 1.0, 'dota/AP50': 1.0,
    }) + '\n', encoding='utf-8')
    case_dump, case_manifest = _case_analyzer_inputs(tmp_path)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    args = analyzer.build_parser().parse_args(_analyzer_argv(
        config, checkpoint, dump, official, output,
        '--training-metrics-json', str(training),
        '--case-dump', str(case_dump),
        '--case-manifest', str(case_manifest)))
    assert not hasattr(args, '_command')

    analyzer.run(args)

    manifest = json.loads((output / 'manifest.json').read_text(
        encoding='utf-8'))
    assert manifest['command'] == [
        sys.executable, str(ANALYZER_PATH.resolve()),
        str(dump),
        '--config', str(config),
        '--checkpoint', str(checkpoint),
        '--official-metrics-json', str(official),
        '--output-dir', str(output),
        '--training-metrics-json', str(training),
        '--training-step', '24',
        '--iou-threshold', '0.5',
        '--sample-size', '500000',
        '--sample-seed', '20260722',
        '--max-error-examples', '100',
        '--case-dump', str(case_dump),
        '--case-manifest', str(case_manifest),
        '--allow-noncanonical',
        '--expected-records', '1',
        '--queries-per-image', '1',
        '--num-classes', '1',
    ]


@pytest.mark.parametrize('target', [
    'config', 'checkpoint', 'dump', 'official', 'training',
    'case_manifest', 'case_dump',
])
def test_analyzer_rejects_input_changed_between_hash_and_load(
        tmp_path, monkeypatch, capsys, target):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, _ = _synthetic_analyzer_inputs(
        tmp_path)
    training = tmp_path / 'training.json'
    training.write_text(json.dumps({
        'step': 24, 'dota/mAP': 1.0, 'dota/AP50': 1.0,
    }) + '\n', encoding='utf-8')
    case_dump, case_manifest = _case_analyzer_inputs(tmp_path)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    mutated = False

    def mutate_once():
        nonlocal mutated
        if mutated:
            return
        mutated = True
        if target == 'config':
            config.write_text(
                config.read_text(encoding='utf-8') + '# changed\n',
                encoding='utf-8')
        elif target == 'checkpoint':
            replacement = tmp_path / 'replacement.pth'
            torch.save({
                'state_dict': {
                    'model.query_initializer.reference_embedding.weight':
                        torch.ones((1, 5), dtype=torch.float32),
                },
            }, replacement)
            replacement.replace(checkpoint)
        elif target == 'dump':
            dump.write_bytes(dump.read_bytes() + b'changed-after-hash')
        elif target == 'official':
            official.write_text(json.dumps({
                'dota/mAP': 1.0, 'dota/AP50': 1.0,
                'changed': True,
            }) + '\n', encoding='utf-8')
        elif target == 'training':
            training.write_text(json.dumps({
                'step': 24, 'dota/mAP': 1.0, 'dota/AP50': 1.0,
                'changed': True,
            }) + '\n', encoding='utf-8')
        elif target == 'case_manifest':
            case_manifest.write_text(
                case_manifest.read_text(encoding='utf-8') + ' \n',
                encoding='utf-8')
        else:
            case_dump.write_bytes(
                case_dump.read_bytes() + b'changed-after-hash')

    if target == 'config':
        real = analyzer.load_class_protocol

        def wrapped(*args, **kwargs):
            mutate_once()
            return real(*args, **kwargs)

        monkeypatch.setattr(analyzer, 'load_class_protocol', wrapped)
    elif target == 'checkpoint':
        real = analyzer._load_checkpoint

        def wrapped(*args, **kwargs):
            mutate_once()
            return real(*args, **kwargs)

        monkeypatch.setattr(analyzer, '_load_checkpoint', wrapped)
    elif target == 'dump':
        real = analyzer._load_validated_dump

        def wrapped(*args, **kwargs):
            mutate_once()
            return real(*args, **kwargs)

        monkeypatch.setattr(analyzer, '_load_validated_dump', wrapped)
    elif target in ('official', 'training'):
        real = analyzer.read_metric_record
        calls = {'official': 0, 'training': 0}

        def wrapped(path, *args, **kwargs):
            kind = ('training' if Path(path).resolve() == training.resolve()
                    else 'official')
            calls[kind] += 1
            if kind == target and calls[kind] == 2:
                mutate_once()
            return real(path, *args, **kwargs)

        monkeypatch.setattr(analyzer, 'read_metric_record', wrapped)
    elif target == 'case_manifest':
        real = analyzer.load_case_manifest

        def wrapped(*args, **kwargs):
            mutate_once()
            return real(*args, **kwargs)

        monkeypatch.setattr(analyzer, 'load_case_manifest', wrapped)
    else:
        real = analyzer.load_cpu

        def wrapped(path):
            if Path(path).resolve() == case_dump.resolve():
                mutate_once()
            return real(path)

        monkeypatch.setattr(analyzer, 'load_cpu', wrapped)

    exit_code = analyzer.main(_analyzer_argv(
        config, checkpoint, dump, official, output,
        '--training-metrics-json', str(training),
        '--case-dump', str(case_dump),
        '--case-manifest', str(case_manifest)))

    assert mutated is True
    assert exit_code == 2
    assert len(capsys.readouterr().err.splitlines()) == 1
    assert not output.exists()
    failed = list(tmp_path.glob('bundle.*.failed'))
    assert len(failed) == 1
    assert {path.name for path in failed[0].iterdir()} == {'failure.json'}
    failure = json.loads((failed[0] / 'failure.json').read_text(
        encoding='utf-8'))
    assert ('changed' in failure['error'].lower() or
            'provenance' in failure['error'].lower())


def test_analyzer_does_not_mutate_loaded_dump_records(tmp_path, monkeypatch):
    analyzer = _load_analyzer()
    config, checkpoint, dump, official, records = _synthetic_analyzer_inputs(
        tmp_path)
    original = copy.deepcopy(records)
    output = tmp_path / 'bundle'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(analyzer, 'load_cpu', lambda path: records)

    assert analyzer.main(_analyzer_argv(
        config, checkpoint, dump, official, output)) == 0

    assert records[0]['img_id'] == original[0]['img_id']
    for section in ('pred_instances', 'gt_instances', 'ignored_instances'):
        for field in records[0][section]:
            assert torch.equal(
                records[0][section][field], original[0][section][field])


def test_analyzer_source_has_no_model_runner_dataloader_or_cuda_builds():
    source = ANALYZER_PATH.read_text(encoding='utf-8')

    forbidden = (
        'MODELS.build', 'Runner(', 'runner.build', 'build_dataloader',
        'DataLoader(', '.cuda(', "device='cuda'", 'device="cuda"',
    )
    assert all(token not in source for token in forbidden)


def test_analyzer_fp_regions_use_each_class_global_ap_support_endpoint():
    analyzer = _load_analyzer()
    core = _load_module()
    bundle = _outcome_evaluation(core)

    all_rows = analyzer._region_summary(bundle.class_results, None)
    last_tp = analyzer._region_summary(
        bundle.class_results, 'last_tp_end')
    ap_support = analyzer._region_summary(
        bundle.class_results, 'ap_support_end')

    assert (all_rows['row_count'], all_rows['tp_count'],
            all_rows['fp_count'], all_rows['ignored_prediction_count']) == (
                6, 1, 4, 1)
    assert (last_tp['row_count'], last_tp['tp_count'],
            last_tp['fp_count'], last_tp['ignored_prediction_count']) == (
                3, 1, 1, 1)
    assert (ap_support['row_count'], ap_support['tp_count'],
            ap_support['fp_count'],
            ap_support['ignored_prediction_count']) == (4, 1, 2, 1)
    assert ap_support['shares'] == {
        'duplicate_fp': 0.0,
        'semantic_fp': .5,
        'localization_background_fp': 0.0,
        'empty_tile_fp': .5,
    }
    assert {outcome: ap_support[outcome] for outcome in analyzer._OUTCOMES} == {
        'tp': 1,
        'duplicate_fp': 0,
        'semantic_fp': 1,
        'localization_background_fp': 0,
        'empty_tile_fp': 1,
        'ignored_prediction': 1,
    }


def test_analyzer_group_summary_and_per_class_rows_follow_core_schema():
    analyzer = _load_analyzer()
    core = _load_module()
    bundle = _outcome_evaluation(core)
    evidence = core.decompose_ground_truth(bundle.records, iou_threshold=.5)
    protocol = core.ClassProtocol(
        classes=('small-vehicle', 'airport'),
        base_classes=('small-vehicle',),
        novel_classes=('airport',),
        canonical=False,
    )
    calibration = core.summarize_calibration(
        bundle, sample_size=20, seed=7)

    base = analyzer._group_summary(protocol, bundle, protocol.base_classes)
    novel = analyzer._group_summary(
        protocol, bundle, protocol.novel_classes)
    rows = analyzer._per_class_rows(
        protocol, bundle, evidence, calibration)

    assert base['class_ids'] == [0]
    assert base['class_names'] == ['small-vehicle']
    assert base['num_gts'] == 1
    assert base['map'] == pytest.approx(bundle.class_results[0].ap)
    assert base['oracle_map'] == pytest.approx(
        bundle.class_results[0].oracle_ap)
    assert base['headroom'] == pytest.approx(
        base['oracle_map'] - base['map'])
    assert novel == {
        'class_ids': [1],
        'class_names': ['airport'],
        'num_gts': 0,
        'map': None,
        'oracle_map': None,
        'headroom': None,
    }
    assert len(rows) == 2
    assert set(rows[0]) == set(core.CSV_SCHEMAS['per_class.csv'])
    assert rows[0]['prediction_rows'] == 5
    assert rows[0]['tp_count'] == 1
    assert rows[0]['fp_count'] == 3
    assert rows[0]['ap_support_empty_tile_fp'] == 1
    assert rows[0]['through_last_tp_ignored_prediction'] == 1
    assert rows[0]['evaluator_reachable'] == 1
    assert rows[0]['tp_score_iou_spearman'] is None


def test_analyzer_error_generator_caps_only_four_false_outcomes():
    analyzer = _load_analyzer()
    core = _load_module()
    square = [10, 10, 4, 4, 0]
    far = [30, 30, 4, 4, 0]
    bundle = core.evaluate_records(
        core.prepare_records_variable_queries([
            _record(
                'errors', [square, square, square, square],
                [.9, .8, .7, .6], [0, 0, 0, 0]),
            _record(
                'tp-and-ignored', [square, far], [.95, .99], [0, 0],
                [square], [0], [far], [0]),
        ], num_classes=1),
        num_classes=1, iou_threshold=.5)

    rows = core.cap_error_examples(
        analyzer._error_examples(bundle), max_per_type=1)

    assert [row['outcome'] for row in rows] == ['empty_tile_fp']
    assert rows[0]['score'] == pytest.approx(.9)
    assert set(rows[0]) == {
        'outcome', 'img_id', 'query_id', 'score', 'class_id',
        'matched_gt', 'assigned_iou', 'same_label_iou',
    }
    assert all(row['outcome'] not in ('tp', 'ignored_prediction')
               for row in rows)
