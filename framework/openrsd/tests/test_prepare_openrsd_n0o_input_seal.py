import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT_PATH = (
    Path(__file__).parents[1] / 'tools' / 'risc_n0o'
    / 'prepare_openrsd_n0o_input_seal.py')


def load_builder():
    module_name = 'prepare_openrsd_n0o_input_seal'
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_scene_plan(scenes_per_fold=2):
    fold_specs = (
        ('c4_a', 4, [0, 90, 180, 270]),
        ('c4_b', 4, [0, 90, 180, 270]),
        ('c8_a', 8, [0, 45, 90, 135, 180, 225, 270, 315]),
        ('c8_b', 8, [0, 45, 90, 135, 180, 225, 270, 315]),
    )
    records = []
    scene_number = 1
    for fold_index, (fold_id, group_order, angles) in enumerate(fold_specs):
        for index in range(scenes_per_fold):
            scene_id = 'P{:04d}'.format(scene_number)
            records.append({
                'angles_deg': list(angles),
                'annotation_path': '/annotations/{}.txt'.format(scene_id),
                'annotation_sha256': '{:064x}'.format(scene_number),
                'fold_id': fold_id,
                'fold_index': fold_index,
                'group_order': group_order,
                'image_path': '/images/{}.png'.format(scene_id),
                'image_sha256': '{:064x}'.format(scene_number + 1000),
                'scene_id': scene_id,
                'scene_rank': index + 1,
                'tile_name': '{}.png'.format(scene_id),
            })
            scene_number += 1
    return {
        'dataset_mouth': {
            'known_openrsd_filtered_tile_count': 6605,
            'raw_tile_count': 13833,
            'selection': 'nonempty_difficulty0_then_scene_hash',
            'split': 'DOTA2_1024_500/ss_val',
        },
        'excluded_scenes': ['P0148'],
        'fold_specs': [
            {'fold_id': fold_id, 'group_order': order,
             'scenes': scenes_per_fold}
            for fold_id, order, _ in fold_specs
        ],
        'nonempty_tiles_observed': 6605,
        'prompt': ['class-a', 'class-b'],
        'records': records,
    }


def test_canonical_json_is_sorted_compact_utf8_with_one_lf():
    builder = load_builder()
    value = {'z': 1, 'a': '场景'}

    encoded = builder.canonical_json_bytes(value)

    assert encoded == json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8') + b'\n'


def test_canonical_jsonl_preserves_row_order_and_rejects_nan():
    builder = load_builder()
    rows = [{'rank': 2}, {'rank': 1}]

    encoded = builder.canonical_jsonl_bytes(rows)

    assert encoded == b'{"rank":2}\n{"rank":1}\n'
    with pytest.raises(ValueError, match='JSON'):
        builder.canonical_json_bytes({'invalid': float('nan')})


def test_scene_plan_requires_four_disjoint_folds_and_excludes_p0148():
    builder = load_builder()
    plan = synthetic_scene_plan()

    summary = builder.validate_scene_plan(plan, scenes_per_fold=2)

    assert summary == {
        'scene_count': 8,
        'fold_counts': {'c4_a': 2, 'c4_b': 2, 'c8_a': 2, 'c8_b': 2},
        'group_counts': {'4': 4, '8': 4},
    }
    plan['records'][1]['scene_id'] = plan['records'][0]['scene_id']
    with pytest.raises(builder.SealError, match='unique scene'):
        builder.validate_scene_plan(plan, scenes_per_fold=2)


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (lambda plan: plan['records'][0].update(scene_id='P0148'), 'P0148'),
        (lambda plan: plan['records'][0].update(angles_deg=[0, 180]),
         'angles'),
        (lambda plan: plan['records'][0].update(image_sha256='ABC'),
         'image_sha256'),
        (lambda plan: plan['records'][0].update(
            image_path=plan['records'][1]['image_path']), 'unique image'),
        (lambda plan: plan['fold_specs'][0].update(scenes=3), 'fold spec'),
        (lambda plan: plan.update(nonempty_tiles_observed=6604), '6605'),
    ],
)
def test_scene_plan_rejects_protocol_drift(mutate, message):
    builder = load_builder()
    plan = synthetic_scene_plan()
    mutate(plan)

    with pytest.raises(builder.SealError, match=message):
        builder.validate_scene_plan(plan, scenes_per_fold=2)


def synthetic_support_data(classes=('class-a', 'class-b'), prompts=10,
                           width=4):
    support = {}
    for class_index, class_name in enumerate(classes):
        values = np.arange(prompts * width, dtype=np.float32)
        values = values.reshape(prompts, width) + class_index * 100
        support[class_name] = {
            'text_embeds': values,
            'visual_embeds': np.zeros((prompts, 6), dtype=np.float32),
        }
    return support


def write_tiny_checkpoint(path, *, include_raw=True, raw_nonfinite=False):
    state = {
        'text_support_mapping.0.weight': torch.arange(
            20, dtype=torch.float32).reshape(5, 4) / 20,
        'text_support_mapping.0.bias': torch.arange(
            5, dtype=torch.float32) / 10,
        'text_support_mapping.2.weight': torch.arange(
            15, dtype=torch.float32).reshape(3, 5) / 15,
        'text_support_mapping.2.bias': torch.arange(
            3, dtype=torch.float32) / 20,
    }
    if raw_nonfinite:
        state['text_support_mapping.0.weight'][0, 0] = float('nan')
    checkpoint = {
        'ema_state_dict': {
            key: torch.full_like(value, 99) for key, value in state.items()
        }
    }
    if include_raw:
        checkpoint['state_dict'] = state
    torch.save(checkpoint, path)
    return state


def test_support_indices_are_scene_specific_and_repeatable():
    builder = load_builder()
    classes = ('class-a', 'class-b')
    support = synthetic_support_data(classes=classes)

    first = builder.select_support_indices(
        'P0001', support, classes=classes, shot=3)
    second = builder.select_support_indices(
        'P0001', support, classes=classes, shot=3)
    other = builder.select_support_indices(
        'P0002', support, classes=classes, shot=3)

    assert first == second
    assert first != other
    assert list(first) == list(classes)
    assert all(len(set(indices)) == 3 for indices in first.values())


def test_support_selection_rejects_missing_short_or_malformed_classes():
    builder = load_builder()
    support = synthetic_support_data()

    with pytest.raises(builder.SealError, match='missing class'):
        builder.select_support_indices(
            'P0001', support, classes=('class-a', 'missing'), shot=3)
    support['class-a']['text_embeds'] = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(builder.SealError, match='at least 3'):
        builder.select_support_indices(
            'P0001', support, classes=('class-a', 'class-b'), shot=3)
    support['class-a']['text_embeds'] = np.zeros((3, 4), dtype=np.float64)
    with pytest.raises(builder.SealError, match='float16 or float32'):
        builder.select_support_indices(
            'P0001', support, classes=('class-a', 'class-b'), shot=3)


def test_support_selection_accepts_frozen_float16_source_and_maps_float32(
        tmp_path):
    builder = load_builder()
    support = synthetic_support_data()
    for class_data in support.values():
        class_data['text_embeds'] = class_data['text_embeds'].astype(
            np.float16)
    checkpoint_path = tmp_path / 'checkpoint.pth'
    write_tiny_checkpoint(checkpoint_path)
    mapping = builder.load_text_mapping(checkpoint_path)

    row, mapped = builder.build_support_row(
        synthetic_scene_plan()['records'][0],
        support,
        mapping,
        classes=('class-a', 'class-b'),
        shot=2)

    assert mapped.dtype == np.dtype('<f4')
    assert row['source_tensor_dtype'] == '<f4'


def test_load_text_mapping_uses_raw_state_and_rejects_invalid(tmp_path):
    builder = load_builder()
    checkpoint_path = tmp_path / 'checkpoint.pth'
    raw_state = write_tiny_checkpoint(checkpoint_path)

    mapping = builder.load_text_mapping(checkpoint_path)

    np.testing.assert_array_equal(
        mapping.first_weight,
        raw_state['text_support_mapping.0.weight'].numpy())
    assert not np.all(mapping.first_weight == 99)

    ema_only = tmp_path / 'ema_only.pth'
    write_tiny_checkpoint(ema_only, include_raw=False)
    with pytest.raises(builder.SealError, match='raw state_dict'):
        builder.load_text_mapping(ema_only)

    nonfinite = tmp_path / 'nonfinite.pth'
    write_tiny_checkpoint(nonfinite, raw_nonfinite=True)
    with pytest.raises(builder.SealError, match='finite'):
        builder.load_text_mapping(nonfinite)


def test_mapping_emits_float32_hash_schema(tmp_path):
    builder = load_builder()
    checkpoint_path = tmp_path / 'checkpoint.pth'
    write_tiny_checkpoint(checkpoint_path)
    mapping = builder.load_text_mapping(checkpoint_path)
    classes = ('class-a', 'class-b')
    support = synthetic_support_data(classes=classes)
    scene_record = synthetic_scene_plan()['records'][0]

    row, mapped = builder.build_support_row(
        scene_record,
        support,
        mapping,
        classes=classes,
        shot=2)

    assert mapped.shape == (2, 2, 3)
    assert mapped.dtype == np.dtype('<f4')
    assert np.isfinite(mapped).all()
    assert row['mapped_tensor_shape'] == [2, 2, 3]
    assert row['mapped_tensor_dtype'] == '<f4'
    assert row['mapped_tensor_byte_count'] == mapped.nbytes
    assert row['mapped_tensor_sha256'] == builder.sha256_bytes(
        mapped.tobytes(order='C'))
    assert row['source_tensor_shape'] == [2, 2, 4]
    assert [item['class_name'] for item in row['selections']] == list(classes)


def test_builder_support_path_never_calls_cuda(tmp_path, monkeypatch):
    builder = load_builder()
    checkpoint_path = tmp_path / 'checkpoint.pth'
    write_tiny_checkpoint(checkpoint_path)
    monkeypatch.setattr(
        torch.cuda,
        'is_available',
        lambda: (_ for _ in ()).throw(AssertionError('CUDA called')))

    mapping = builder.load_text_mapping(checkpoint_path)
    builder.build_support_row(
        synthetic_scene_plan()['records'][0],
        synthetic_support_data(),
        mapping,
        classes=('class-a', 'class-b'),
        shot=2)


def make_synthetic_authority(builder, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_dir = tmp_path / 'images'
    annotation_dir = tmp_path / 'annotations'
    image_dir.mkdir()
    annotation_dir.mkdir()
    plan = synthetic_scene_plan()
    for index, row in enumerate(plan['records']):
        image = image_dir / row['tile_name']
        annotation = annotation_dir / '{}.txt'.format(row['scene_id'])
        image.write_bytes('image-{}'.format(index).encode('ascii'))
        annotation.write_bytes('annotation-{}'.format(index).encode('ascii'))
        row['image_path'] = str(image)
        row['annotation_path'] = str(annotation)
        row['image_sha256'] = builder.sha256_file(image)
        row['annotation_sha256'] = builder.sha256_file(annotation)
    scene_plan = tmp_path / 'scene_plan.json'
    scene_plan.write_bytes(builder.canonical_json_bytes(plan))

    support_pickle = tmp_path / 'support.pkl'
    with support_pickle.open('wb') as stream:
        pickle.dump(synthetic_support_data(), stream)
    checkpoint = tmp_path / 'checkpoint.pth'
    write_tiny_checkpoint(checkpoint)

    def asset(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        return builder.Asset(path=path, sha256=builder.sha256_file(path))

    return builder.SealAuthority(
        checkpoint=builder.Asset(
            path=checkpoint, sha256=builder.sha256_file(checkpoint)),
        scene_plan=builder.Asset(
            path=scene_plan, sha256=builder.sha256_file(scene_plan)),
        support_pickle=builder.Asset(
            path=support_pickle, sha256=builder.sha256_file(support_pickle)),
        historical_assets={'config': asset('config.py', b'config')},
        supporting_assets={'metadata': asset('metadata.pkl', b'metadata')},
        s0_assets={'adapter': asset('adapter.py', b'adapter')},
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        image_count=8,
        annotation_count=8,
        nonempty_annotation_count=8,
        empty_annotation_count=0,
        classes=('class-a', 'class-b'),
        scenes_per_fold=2,
        support_shot=2,
        support_source_dtype='<f4',
        text_width=4,
        hidden_width=5,
        mapped_width=3,
        s0_base_commit='a' * 40,
    )


def test_full_builder_is_deterministic_hash_chained_and_no_replace(tmp_path):
    builder = load_builder()
    authority = make_synthetic_authority(builder, tmp_path)
    first = tmp_path / 'seal-a'
    second = tmp_path / 'seal-b'

    first_summary = builder.build_input_seal(first, authority=authority)
    second_summary = builder.build_input_seal(second, authority=authority)

    assert first_summary == second_summary
    for filename in (
            'scene_plan_40.json', 'support_ledger.jsonl',
            'input_manifest.json'):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    manifest = json.loads((first / 'input_manifest.json').read_text())
    ledger_lines = (first / 'support_ledger.jsonl').read_bytes().splitlines()
    assert manifest['status'] == 'SEALED_INPUTS_GPU_NOT_AUTHORIZED'
    assert manifest['scene_plan']['row_count'] == 8
    assert manifest['support']['ledger_row_count'] == 8
    assert manifest['support']['class_count'] == 2
    assert manifest['support']['shot'] == 2
    assert manifest['artifacts']['support_ledger.jsonl']['sha256'] == (
        builder.sha256_file(first / 'support_ledger.jsonl'))
    assert len(ledger_lines) == 8
    assert all(len(json.loads(line)['selections']) == 2
               for line in ledger_lines)
    with pytest.raises(builder.SealError, match='already exists'):
        builder.build_input_seal(first, authority=authority)


def test_builder_rejects_source_hash_and_selected_file_drift(tmp_path):
    builder = load_builder()
    authority = make_synthetic_authority(builder, tmp_path)
    authority.historical_assets['config'].path.write_bytes(b'drift')

    with pytest.raises(builder.SealError, match='hash mismatch'):
        builder.build_input_seal(tmp_path / 'bad-asset', authority=authority)

    authority = make_synthetic_authority(builder, tmp_path / 'second')
    first_image = Path(json.loads(
        authority.scene_plan.path.read_text())['records'][0]['image_path'])
    first_image.write_bytes(b'drift')
    with pytest.raises(builder.SealError, match='selected image hash'):
        builder.build_input_seal(tmp_path / 'bad-image', authority=authority)
