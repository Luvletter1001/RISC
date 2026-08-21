from pathlib import Path

import pytest

from projects.OVCapFlow.tools.audit_dotav2_mouth import (
    audit_mouth,
    validate_contract,
)


def _write_split(root: Path, records):
    image_dir = root / 'images'
    annotation_dir = root / 'annfiles'
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    for stem, annotation in records:
        (image_dir / f'{stem}.png').write_bytes(b'fixture')
        if annotation is not None:
            (annotation_dir / f'{stem}.txt').write_text(
                annotation, encoding='utf-8')


def _fixture_mouth(root: Path):
    train_root = root / 'train'
    val_root = root / 'val'
    _write_split(train_root, [
        ('train_a', '0 0 1 0 1 1 0 1 plane 0\n'),
        ('train_b', ''),
    ])
    _write_split(val_root, [
        ('val_a', '0 0 1 0 1 1 0 1 ship 0\n'),
        ('val_b', ''),
    ])
    return train_root, val_root


def test_audit_mouth_reports_pairs_empty_tiles_and_classes(tmp_path):
    train_root, val_root = _fixture_mouth(tmp_path)

    report = audit_mouth(train_root, val_root)

    assert report['train_image_count'] == 2
    assert report['train_annotation_count'] == 2
    assert report['val_image_count'] == 2
    assert report['val_annotation_count'] == 2
    assert report['train_empty_count'] == 1
    assert report['val_empty_count'] == 1
    assert report['classes'] == ['plane', 'ship']
    assert report['missing_train_images'] == []
    assert report['missing_train_annotations'] == []
    assert report['missing_val_images'] == []
    assert report['missing_val_annotations'] == []


def test_audit_mouth_reports_missing_stems(tmp_path):
    train_root, val_root = _fixture_mouth(tmp_path)
    (train_root / 'images' / 'image_only.png').write_bytes(b'fixture')
    (val_root / 'annfiles' / 'annotation_only.txt').write_text(
        '', encoding='utf-8')

    report = audit_mouth(train_root, val_root)

    assert report['missing_train_annotations'] == ['image_only']
    assert report['missing_val_images'] == ['annotation_only']


def test_validate_contract_rejects_wrong_count(tmp_path):
    train_root, val_root = _fixture_mouth(tmp_path)
    report = audit_mouth(train_root, val_root)

    with pytest.raises(ValueError, match='train image count'):
        validate_contract(
            report,
            expected_train=3,
            expected_val=2,
            expected_classes=('plane', 'ship'))


def test_validate_contract_rejects_wrong_class_token(tmp_path):
    train_root, val_root = _fixture_mouth(tmp_path)
    report = audit_mouth(train_root, val_root)

    with pytest.raises(ValueError, match='class tokens'):
        validate_contract(
            report,
            expected_train=2,
            expected_val=2,
            expected_classes=('bridge', 'ship'))


def test_validate_contract_rejects_unpaired_files(tmp_path):
    train_root, val_root = _fixture_mouth(tmp_path)
    (val_root / 'images' / 'image_only.png').write_bytes(b'fixture')
    report = audit_mouth(train_root, val_root)

    with pytest.raises(ValueError, match='unpaired files'):
        validate_contract(
            report,
            expected_train=2,
            expected_val=3,
            expected_classes=('plane', 'ship'))
