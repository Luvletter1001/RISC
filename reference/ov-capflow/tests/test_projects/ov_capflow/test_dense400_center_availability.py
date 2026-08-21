import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
               'tools' / 'analyze_dense400_center_availability.py')


def _load_module():
    name = 'analyze_dense400_center_availability'
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _record(img_id, pred_boxes, pred_labels, gt_boxes, gt_labels):
    query_count = len(pred_boxes)
    return {
        'img_id': img_id,
        'pred_instances': {
            'bboxes': torch.tensor(pred_boxes, dtype=torch.float32).reshape(
                query_count, 5),
            'scores': torch.linspace(0.1, 0.9, query_count),
            'labels': torch.tensor(pred_labels, dtype=torch.long),
        },
        'gt_instances': {
            'bboxes': torch.tensor(gt_boxes, dtype=torch.float32).reshape(
                -1, 5),
            'labels': torch.tensor(gt_labels, dtype=torch.long),
        },
    }


def test_normalized_distance_uses_gt_rotated_local_coordinates():
    availability = _load_module()
    theta = np.pi / 4
    cosine = np.cos(theta)
    sine = np.sin(theta)
    gt_boxes = np.asarray([[0.0, 0.0, 4.0, 2.0, theta]])
    query_centers = np.asarray([
        [1.9 * cosine, 1.9 * sine],
        [-0.9 * sine, 0.9 * cosine],
    ])

    distances = availability.normalized_center_distances(
        query_centers, gt_boxes)

    assert distances.shape == (1, 2)
    assert distances[0].tolist() == pytest.approx([0.95, 0.9])


def test_geometry_miss_availability_separates_label_and_matching():
    availability = _load_module()
    records = availability.prepare_records([
        _record(
            'two-gt',
            pred_boxes=[
                [0.0, 0.0, 0.2, 0.2, 0.0],
                [10.0, 0.0, 0.2, 0.2, 0.0],
            ],
            pred_labels=[0, 0],
            gt_boxes=[
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [10.0, 0.0, 2.0, 2.0, 0.0],
            ],
            gt_labels=[0, 1],
        )
    ], queries_per_image=2, num_classes=2)

    result = availability.analyze_records(records, iou_threshold=0.5)

    assert result['anchors'] == {'geometry_miss': 2}
    assert result['availability']['any_label']['available'] == 2
    assert result['availability']['any_label']['rate'] == 1.0
    assert result['availability']['same_label']['available'] == 1
    assert result['availability']['same_label']['rate'] == 0.5
    assert result['matching']['all_label'] == 2
    assert result['matching']['same_label'] == 1
    any_distance = result['min_normalized_chebyshev']['any_label']
    same_distance = result['min_normalized_chebyshev']['same_label']
    assert any_distance['finite'] == 2
    assert any_distance['missing_candidates'] == 0
    assert any_distance['quantiles']['p50'] == 0.0
    assert same_distance['finite'] == 1
    assert same_distance['missing_candidates'] == 1
    assert same_distance['quantiles']['p50'] == 0.0


def test_matching_enforces_one_query_per_geometry_miss_gt():
    availability = _load_module()
    records = availability.prepare_records([
        _record(
            'conflict',
            pred_boxes=[
                [0.0, 0.0, 0.2, 0.2, 0.0],
                [10.0, 10.0, 0.2, 0.2, 0.0],
            ],
            pred_labels=[0, 0],
            gt_boxes=[
                [-0.25, 0.0, 2.0, 2.0, 0.0],
                [0.25, 0.0, 2.0, 2.0, 0.0],
            ],
            gt_labels=[0, 0],
        )
    ], queries_per_image=2, num_classes=1)

    result = availability.analyze_records(records, iou_threshold=0.5)

    assert result['availability']['any_label']['available'] == 2
    assert result['matching']['all_label'] == 1
    assert result['matching']['same_label'] == 1


def test_anchor_failure_does_not_publish_output(tmp_path, monkeypatch):
    availability = _load_module()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    dump = tmp_path / 'predictions.pkl'
    manifest = tmp_path / 'manifest.json'
    output = tmp_path / 'availability.json'
    far_boxes = [[100.0 + index, 100.0, 0.2, 0.2, 0.0]
                 for index in range(600)]
    raw = [_record(
        'tiny', far_boxes, [0] * 600,
        [[0.0, 0.0, 2.0, 2.0, 0.0]], [0])]
    with dump.open('wb') as stream:
        pickle.dump(raw, stream)
    manifest.write_text(json.dumps({
        'selected_total': 1,
        'selected_gt_total': 1,
        'selection_content_sha256': 'fixture',
        'records': [{
            'dataset_index': 0,
            'img_id': 'tiny',
            'gt_count': 1,
        }],
    }), encoding='utf-8')
    args = SimpleNamespace(
        dump=dump,
        manifest=manifest,
        output=output,
        iou_threshold=0.5,
    )

    with pytest.raises(availability.ControlError, match='anchor mismatch'):
        availability.run(args)

    assert not output.exists()
    assert not list(tmp_path.glob('.availability.json.*.tmp'))


def test_visible_cuda_fails_before_loading_dump(monkeypatch):
    availability = _load_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    called = []
    monkeypatch.setattr(
        availability, 'load_dense400',
        lambda *args, **kwargs: called.append(True))

    with pytest.raises(availability.ControlError, match='must be empty'):
        availability.run(SimpleNamespace())

    assert called == []
