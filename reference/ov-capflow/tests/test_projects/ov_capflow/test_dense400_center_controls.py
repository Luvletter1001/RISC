import importlib.util
import json
import pickle
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
               'tools' / 'analyze_dense400_center_controls.py')
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepared_record(controls):
    record = {
        'img_id': 'dense-two',
        'pred_instances': {
            'bboxes': torch.tensor([
                [5.0, 0.0, 2.0, 2.0, 0.0],
                [15.0, 0.0, 2.0, 2.0, 0.0],
            ]),
            'scores': torch.tensor([0.9, 0.8]),
            'labels': torch.tensor([0, 0]),
        },
        'gt_instances': {
            'bboxes': torch.tensor([
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [10.0, 0.0, 2.0, 2.0, 0.0],
            ]),
            'labels': torch.tensor([0, 0]),
        },
    }
    return controls.prepare_records(
        [record], queries_per_image=2, num_classes=1)[0]


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


def test_nested_query_subsets_are_seeded_nested_and_full():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')

    subsets = controls.nested_query_subsets(6, (2, 4, 6), seed=7)

    assert len(subsets[2]) == 2
    assert set(subsets[2]) < set(subsets[4])
    assert set(subsets[4]) < set(subsets[6])
    assert set(subsets[6]) == set(range(6))
    assert np.array_equal(
        subsets[4],
        controls.nested_query_subsets(6, (2, 4, 6), seed=7)[4])


def test_same_class_cyclic_donors_are_deranged_and_mark_singletons():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    labels = np.asarray([0, 0, 0, 1, 2, 2])

    donors = controls.cyclic_same_class_donors(labels, seed=11)

    assert donors[3] == -1
    supported = np.flatnonzero(donors >= 0)
    assert np.all(donors[supported] != supported)
    assert np.array_equal(labels[donors[supported]], labels[supported])


def test_center_controls_use_real_iou_matching_and_shuffled_placebo():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')

    result = controls.analyze_records(
        [_prepared_record(controls)],
        query_counts=(1, 2),
        local_candidate_counts=(1,),
        seeds=(17,),
        iou_threshold=0.5,
    )

    assert result['anchors'] == {
        'gt': 2,
        'geometry_miss': 2,
        'local_center_rescue': 2,
        'baseline_matching': 0,
        'center_matching': 2,
    }
    assert result['candidate_matching']['1']['runs'][0] == {
        'seed': 17,
        'baseline_matching': 0,
        'center_matching': 1,
    }
    assert result['candidate_matching']['2']['runs'][0][
        'center_matching'] == 2
    assert result['local_candidate_rescue']['1']['runs'][0][
        'rescued'] == 2
    assert result['placebo']['eligible_geometry_miss'] == 2
    assert result['placebo']['true_center_local_rescue'] == 2
    assert result['placebo']['runs'][0]['local_rescue'] == 0
    assert result['placebo']['runs'][0]['matching'] == 0


def test_canonical_anchor_gate_fails_closed():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')

    with pytest.raises(controls.ControlError, match='anchor mismatch'):
        controls.require_anchors(
            {'geometry_miss': 1, 'local_center_rescue': 1,
             'baseline_matching': 1, 'center_matching': 1},
            {'geometry_miss': 2, 'local_center_rescue': 1,
             'baseline_matching': 1, 'center_matching': 1},
        )


def test_geometry_miss_uses_all_labels_before_same_label_edges():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    prepared = controls.prepare_records([
        _record(
            'semantic-only',
            pred_boxes=[
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [10.0, 0.0, 2.0, 2.0, 0.0],
            ],
            pred_labels=[1, 0],
            gt_boxes=[[0.0, 0.0, 2.0, 2.0, 0.0]],
            gt_labels=[0],
        )
    ], queries_per_image=2, num_classes=2)[0]

    edges = controls._record_edges(prepared, iou_threshold=0.5)

    assert edges['geometry_miss'].tolist() == [False]
    assert not edges['baseline'].any()
    assert not edges['center'].any()


def test_matching_edges_require_the_prediction_label_to_match_gt():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    prepared = controls.prepare_records([
        _record(
            'label-edge',
            pred_boxes=[
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 0.0],
            ],
            pred_labels=[1, 0],
            gt_boxes=[[0.0, 0.0, 2.0, 2.0, 0.0]],
            gt_labels=[0],
        )
    ], queries_per_image=2, num_classes=2)[0]

    edges = controls._record_edges(prepared, iou_threshold=0.5)

    assert edges['baseline'].tolist() == [[False, True]]
    assert controls._matching_count(edges['baseline']) == 1


def test_matching_curve_uses_one_global_query_subset_across_images():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    seed = 23
    selected = int(controls.nested_query_subsets(2, (1,), seed)[1][0])
    center = np.zeros((1, 2), dtype=np.bool_)
    center[0, selected] = True
    images = [
        {'baseline': np.zeros((1, 2), dtype=np.bool_), 'center': center},
        {'baseline': np.zeros((1, 2), dtype=np.bool_), 'center': center},
    ]

    curve = controls._matching_curve(images, (1,), (seed,), query_count=2)

    assert curve['1']['runs'][0]['center_matching'] == 2


def test_placebo_true_center_reference_excludes_singleton_classes():
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    prepared = controls.prepare_records([
        _record(
            'eligible-only',
            pred_boxes=[
                [5.0, 0.0, 2.0, 2.0, 0.0],
                [15.0, 0.0, 2.0, 2.0, 0.0],
                [25.0, 0.0, 2.0, 2.0, 0.0],
            ],
            pred_labels=[0, 0, 1],
            gt_boxes=[
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [10.0, 0.0, 2.0, 2.0, 0.0],
                [20.0, 0.0, 2.0, 2.0, 0.0],
            ],
            gt_labels=[0, 0, 1],
        )
    ], queries_per_image=3, num_classes=2)[0]
    image = controls._record_edges(prepared, iou_threshold=0.5)

    placebo = controls._placebo_control(
        [image], seeds=(31,), iou_threshold=0.5)

    assert placebo['eligible_geometry_miss'] == 2
    assert placebo['true_center_local_rescue'] == 2
    assert placebo['true_center_matching'] == 2
    assert placebo['runs'][0] == {
        'seed': 31,
        'local_rescue': 0,
        'matching': 0,
    }


def test_canonical_anchor_failure_does_not_publish_output(
        tmp_path, monkeypatch):
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    dump = tmp_path / 'predictions.pkl'
    manifest = tmp_path / 'manifest.json'
    output = tmp_path / 'controls.json'
    far_boxes = [[100.0 + index, 100.0, 2.0, 2.0, 0.0]
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
        query_counts=(600,),
        local_candidate_counts=(1,),
        seed_base=7,
        num_seeds=1,
        iou_threshold=0.5,
    )

    with pytest.raises(controls.ControlError, match='anchor mismatch'):
        controls.run(args)

    assert not output.exists()
    assert not list(tmp_path.glob('.controls.json.*.tmp'))


def test_json_publication_does_not_clobber_and_cleans_temporary_file(
        tmp_path):
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    output = tmp_path / 'result.json'

    controls._publish_json(output, {'value': 1})

    assert json.loads(output.read_text()) == {'value': 1}
    assert not list(tmp_path.glob('.result.json.*.tmp'))
    with pytest.raises(controls.ControlError, match='already exists'):
        controls._publish_json(output, {'value': 2})
    assert json.loads(output.read_text()) == {'value': 1}
    assert not list(tmp_path.glob('.result.json.*.tmp'))


def test_visible_cuda_fails_before_loading_dump(monkeypatch):
    controls = _load(SCRIPT_PATH, 'analyze_dense400_center_controls')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    called = []
    monkeypatch.setattr(
        controls, 'load_dense400', lambda *args, **kwargs: called.append(True))
    args = SimpleNamespace(num_seeds=1)

    with pytest.raises(controls.ControlError, match='must be empty'):
        controls.run(args)

    assert called == []
