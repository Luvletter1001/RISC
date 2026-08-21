from pathlib import Path

import pytest

from projects.OVCapFlow.tools.build_rare_repeat_subset import (
    build_rare_repeat_subset,
)


def _sample(root: Path, stem: str, labels):
    ann_dir = root / 'annfiles'
    image_dir = root / 'images'
    ann_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    lines = [f'0 0 1 0 1 1 0 1 {label} 0' for label in labels]
    (ann_dir / f'{stem}.txt').write_text('\n'.join(lines))
    (image_dir / f'{stem}.png').write_bytes(b'image')


def test_rare_repeat_replaces_empty_slots_at_fixed_size(tmp_path):
    source = tmp_path / 'source'
    _sample(source, 'rare_a', ['airport'])
    _sample(source, 'rare_b', ['helicopter', 'helicopter'])
    _sample(source, 'base', ['ship'])
    for index in range(6):
        _sample(source, f'empty_{index}', [])

    output = tmp_path / 'output'
    manifest = build_rare_repeat_subset(
        source, output, repeat_factor=3, seed=17)

    assert manifest['source_images'] == 9
    assert manifest['output_images'] == 9
    assert manifest['rare_unique_images'] == 2
    assert manifest['rare_output_exposures'] == 6
    assert manifest['source_empty_images'] == 6
    assert manifest['output_empty_images'] == 2
    assert len(list((output / 'annfiles').glob('*.txt'))) == 9
    assert len(list((output / 'images').glob('*.png'))) == 9
    assert len(list((output / 'annfiles').glob('*__rare_repeat_*.txt'))) == 4


def test_rare_repeat_is_deterministic_for_seed(tmp_path):
    source = tmp_path / 'source'
    _sample(source, 'rare', ['container-crane'])
    for index in range(4):
        _sample(source, f'empty_{index}', [])

    first = build_rare_repeat_subset(
        source, tmp_path / 'first', repeat_factor=3, seed=23)
    second = build_rare_repeat_subset(
        source, tmp_path / 'second', repeat_factor=3, seed=23)

    assert first['removed_empty_stems'] == second['removed_empty_stems']


def test_rare_repeat_rejects_insufficient_empty_slots(tmp_path):
    source = tmp_path / 'source'
    _sample(source, 'rare_a', ['airport'])
    _sample(source, 'rare_b', ['helicopter'])
    _sample(source, 'empty', [])

    with pytest.raises(ValueError, match='empty replacements'):
        build_rare_repeat_subset(
            source, tmp_path / 'output', repeat_factor=3)
