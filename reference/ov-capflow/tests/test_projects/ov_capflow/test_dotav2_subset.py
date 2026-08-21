import hashlib
import json
from pathlib import Path

import pytest


CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
NOVEL = ('airport', 'container-crane', 'helipad', 'helicopter')


def _api():
    from projects.OVCapFlow.tools.build_dotav2_cleanstart_subset import (
        DEFAULT_S1_QUOTAS,
        build_subsets,
        parse_dota_annotation,
    )
    return DEFAULT_S1_QUOTAS, build_subsets, parse_dota_annotation


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_line(label):
    return f'0 0 8 0 8 8 0 8 {label} 0\n'


def _write_record(root, stem, count, class_offset):
    image = root / 'ss_train' / 'images' / f'{stem}.png'
    ann = root / 'ss_train' / 'annfiles' / f'{stem}.txt'
    image.write_bytes(('image-' + stem).encode('ascii'))
    lines = ['imagesource:GoogleEarth\n', 'gsd:0.5\n']
    for index in range(count):
        lines.append(_object_line(CLASSES[(class_offset + index) % 18]))
    ann.write_text(''.join(lines), encoding='utf-8')


@pytest.fixture()
def miniature_dota(tmp_path):
    source = tmp_path / 'source'
    (source / 'ss_train' / 'images').mkdir(parents=True)
    (source / 'ss_train' / 'annfiles').mkdir(parents=True)
    bins = {
        'empty': 0,
        'sparse': 2,
        'medium': 12,
        'dense': 51,
        'ultra': 201,
    }
    for bin_index, (name, count) in enumerate(bins.items()):
        for record_index in range(5):
            _write_record(
                source,
                f'{name}_{record_index}',
                count,
                class_offset=(bin_index * 5 + record_index) % 18)
    return source


def test_default_s1_protocol_is_locked():
    quotas, _, _ = _api()
    assert quotas == {
        'empty': 600,
        'sparse_1_10': 900,
        'medium_11_50': 300,
        'dense_51_200': 150,
        'ultra_gt_200': 50,
    }
    assert sum(quotas.values()) == 2000


def test_parser_accepts_only_native_dota_object_rows(tmp_path):
    _, _, parse = _api()
    annotation = tmp_path / 'tile.txt'
    annotation.write_text(
        'imagesource:GoogleEarth\n'
        'gsd:0.5\n'
        '0 0 8 0 8 8 0 8 airport 0\n',
        encoding='utf-8')
    objects = parse(annotation, CLASSES)
    assert len(objects) == 1
    assert objects[0].label == 'airport'
    assert objects[0].difficulty == 0
    assert objects[0].line == '0 0 8 0 8 8 0 8 airport 0\n'

    annotation.write_text(
        '0 0 8 0 8 8 0 8 airport 0 0.99\n', encoding='utf-8')
    with pytest.raises(ValueError, match='native DOTA row'):
        parse(annotation, CLASSES)


def test_builder_is_deterministic_and_preserves_real_gt(miniature_dota,
                                                         tmp_path):
    _, build, _ = _api()
    output_a = tmp_path / 'subset-a'
    output_b = tmp_path / 'subset-b'
    quotas = {
        'empty': 5,
        'sparse_1_10': 5,
        'medium_11_50': 5,
        'dense_51_200': 5,
        'ultra_gt_200': 5,
    }
    source_hashes = {
        path.relative_to(miniature_dota).as_posix(): _sha(path)
        for path in miniature_dota.rglob('*') if path.is_file()
    }

    first = build(
        source_root=miniature_dota,
        output_root=output_a,
        seed=20260715,
        classes=CLASSES,
        novel_classes=NOVEL,
        s1_quotas=quotas,
        s0_nonempty=8,
        s0_empty=2,
        s0_dense_min=2)
    second = build(
        source_root=miniature_dota,
        output_root=output_b,
        seed=20260715,
        classes=CLASSES,
        novel_classes=NOVEL,
        s1_quotas=quotas,
        s0_nonempty=8,
        s0_empty=2,
        s0_dense_min=2)

    assert first == second
    assert (output_a / 'manifest.json').read_bytes() == (
        output_b / 'manifest.json').read_bytes()
    assert first['seed'] == 20260715
    assert first['uses_pseudo_labels'] is False
    assert first['novel_classes'] == list(NOVEL)
    assert first['s0']['nonempty_count'] == 8
    assert first['s0']['empty_count'] == 2
    assert first['s0']['dense_gt_200_count'] >= 2
    assert first['s0']['covered_classes'] == list(CLASSES)
    assert first['s1']['quota'] == quotas
    assert first['s1']['train_count'] == 20
    assert first['s1']['val_count'] == 5
    assert first['s1']['total_count'] == 25
    assert first['s1']['covered_classes'] == list(CLASSES)
    assert first['s1']['train_val_overlap'] == []

    for item in first['records']:
        image_link = output_a / item['views'][0] / 'images' / (
            item['stem'] + '.png')
        assert image_link.is_symlink()
        assert image_link.resolve() == Path(item['source_image'])
        native_ann = output_a / item['views'][0] / 'annfiles' / (
            item['stem'] + '.txt')
        assert native_ann.read_bytes() == Path(item['source_annotation']).read_bytes()
        assert item['source_image_sha256'] == _sha(Path(item['source_image']))
        assert item['source_annotation_sha256'] == _sha(
            Path(item['source_annotation']))

    for base_ann in (output_a / 's1_train_base14' / 'annfiles').glob('*.txt'):
        labels = [obj.label for obj in _api()[2](base_ann, CLASSES)]
        assert not set(labels) & set(NOVEL)
        source_ann = miniature_dota / 'ss_train' / 'annfiles' / base_ann.name
        source_lines = set(source_ann.read_text(encoding='utf-8').splitlines())
        assert set(base_ann.read_text(encoding='utf-8').splitlines()) <= source_lines

    assert source_hashes == {
        path.relative_to(miniature_dota).as_posix(): _sha(path)
        for path in miniature_dota.rglob('*') if path.is_file()
    }
    manifest_bytes = (output_a / 'manifest.json').read_bytes()
    assert json.loads(manifest_bytes) == first
    assert (output_a / 'manifest.sha256').read_text(
        encoding='ascii').strip() == hashlib.sha256(manifest_bytes).hexdigest()


def test_builder_fails_when_quota_or_coverage_is_impossible(miniature_dota,
                                                             tmp_path):
    _, build, _ = _api()
    with pytest.raises(ValueError, match='quota'):
        build(
            source_root=miniature_dota,
            output_root=tmp_path / 'too-many',
            seed=1,
            classes=CLASSES,
            novel_classes=NOVEL,
            s1_quotas={
                'empty': 6,
                'sparse_1_10': 5,
                'medium_11_50': 5,
                'dense_51_200': 5,
                'ultra_gt_200': 5,
            },
            s0_nonempty=8,
            s0_empty=2,
            s0_dense_min=2)

    with pytest.raises(ValueError, match='dense'):
        build(
            source_root=miniature_dota,
            output_root=tmp_path / 'too-dense',
            seed=1,
            classes=CLASSES,
            novel_classes=NOVEL,
            s1_quotas={
                'empty': 5,
                'sparse_1_10': 5,
                'medium_11_50': 5,
                'dense_51_200': 5,
                'ultra_gt_200': 5,
            },
            s0_nonempty=8,
            s0_empty=2,
            s0_dense_min=8)
