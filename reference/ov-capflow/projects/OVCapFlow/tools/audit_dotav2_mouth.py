import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Set, Tuple


IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}


def _stems(directory: Path, suffixes: Set[str]) -> Set[str]:
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _annotation_summary(annotation_dir: Path) -> Tuple[Set[str], int, Set[str]]:
    stems: Set[str] = set()
    classes: Set[str] = set()
    empty_count = 0
    for path in sorted(annotation_dir.glob('*.txt')):
        stems.add(path.stem)
        object_count = 0
        for line_number, raw_line in enumerate(
                path.read_text(encoding='utf-8').splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.lower().startswith(('imagesource:', 'gsd:')):
                continue
            tokens = line.split()
            if len(tokens) < 9:
                raise ValueError(
                    f'invalid annotation line {path}:{line_number}: {line!r}')
            classes.add(tokens[8])
            object_count += 1
        if object_count == 0:
            empty_count += 1
    return stems, empty_count, classes


def _audit_split(root: Path) -> Dict[str, object]:
    image_dir = root / 'images'
    annotation_dir = root / 'annfiles'
    if not image_dir.is_dir():
        raise ValueError(f'missing image directory: {image_dir}')
    if not annotation_dir.is_dir():
        raise ValueError(f'missing annotation directory: {annotation_dir}')

    image_stems = _stems(image_dir, IMAGE_SUFFIXES)
    annotation_stems, empty_count, classes = _annotation_summary(
        annotation_dir)
    return {
        'image_count': len(image_stems),
        'annotation_count': len(annotation_stems),
        'empty_count': empty_count,
        'classes': classes,
        'missing_images': sorted(annotation_stems - image_stems),
        'missing_annotations': sorted(image_stems - annotation_stems),
    }


def audit_mouth(train_root: Path, val_root: Path) -> Dict[str, object]:
    train_root = Path(train_root)
    val_root = Path(val_root)
    train = _audit_split(train_root)
    val = _audit_split(val_root)
    return {
        'train_root': str(train_root.absolute()),
        'train_root_resolved': str(train_root.resolve()),
        'val_root': str(val_root.absolute()),
        'val_root_resolved': str(val_root.resolve()),
        'train_image_count': train['image_count'],
        'train_annotation_count': train['annotation_count'],
        'val_image_count': val['image_count'],
        'val_annotation_count': val['annotation_count'],
        'train_empty_count': train['empty_count'],
        'val_empty_count': val['empty_count'],
        'classes': sorted(train['classes'] | val['classes']),
        'missing_train_images': train['missing_images'],
        'missing_train_annotations': train['missing_annotations'],
        'missing_val_images': val['missing_images'],
        'missing_val_annotations': val['missing_annotations'],
    }


def validate_contract(
        report: Dict[str, object], expected_train: int, expected_val: int,
        expected_classes: Sequence[str]) -> None:
    if report['train_image_count'] != expected_train:
        raise ValueError(
            'train image count mismatch: '
            f"{report['train_image_count']} != {expected_train}")
    if report['val_image_count'] != expected_val:
        raise ValueError(
            'val image count mismatch: '
            f"{report['val_image_count']} != {expected_val}")

    missing_fields = (
        'missing_train_images', 'missing_train_annotations',
        'missing_val_images', 'missing_val_annotations')
    if any(report[field] for field in missing_fields):
        raise ValueError('unpaired files detected in DOTA2 mouth')

    if report['train_annotation_count'] != expected_train:
        raise ValueError(
            'train annotation count mismatch: '
            f"{report['train_annotation_count']} != {expected_train}")
    if report['val_annotation_count'] != expected_val:
        raise ValueError(
            'val annotation count mismatch: '
            f"{report['val_annotation_count']} != {expected_val}")

    expected_tokens = sorted(set(expected_classes))
    if report['classes'] != expected_tokens:
        raise ValueError(
            f"class tokens mismatch: {report['classes']} != {expected_tokens}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit the immutable DOTA-v2.0 train/validation mouth.')
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--val-root', type=Path, required=True)
    parser.add_argument('--expected-train', type=int, default=47294)
    parser.add_argument('--expected-val', type=int, default=13833)
    parser.add_argument(
        '--expected-classes', nargs='+', required=True,
        help='Expected class tokens. Order is recorded separately by config.')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def _write_report(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    args = parse_args()
    report = audit_mouth(args.train_root, args.val_root)
    report['expected_class_order'] = list(args.expected_classes)
    try:
        validate_contract(
            report,
            expected_train=args.expected_train,
            expected_val=args.expected_val,
            expected_classes=args.expected_classes)
    except ValueError as error:
        report['valid'] = False
        report['validation_error'] = str(error)
        _write_report(args.output, report)
        raise SystemExit(str(error))
    report['valid'] = True
    _write_report(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
