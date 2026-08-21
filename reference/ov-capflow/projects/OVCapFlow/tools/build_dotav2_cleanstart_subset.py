"""Build deterministic real-GT DOTA-v2 S0 and S1 subsets."""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


DOTA2_CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
NOVEL_CLASSES = ('airport', 'container-crane', 'helipad', 'helicopter')
DEFAULT_S1_QUOTAS = {
    'empty': 600,
    'sparse_1_10': 900,
    'medium_11_50': 300,
    'dense_51_200': 150,
    'ultra_gt_200': 50,
}
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


@dataclass(frozen=True)
class DOTAObject:
    coordinates: Tuple[float, ...]
    label: str
    difficulty: int
    line: str


@dataclass(frozen=True)
class DOTARecord:
    stem: str
    image: Path
    annotation: Path
    objects: Tuple[DOTAObject, ...]

    @property
    def gt_count(self) -> int:
        return len(self.objects)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(sorted({obj.label for obj in self.objects}))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dota_annotation(path: Path,
                          classes: Sequence[str]) -> Tuple[DOTAObject, ...]:
    allowed = set(classes)
    objects = []
    for line_number, line in enumerate(
            Path(path).read_text(encoding='utf-8').splitlines(keepends=True),
            start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith('imagesource:') or \
                stripped.startswith('gsd:'):
            continue
        fields = stripped.split()
        if len(fields) != 10:
            raise ValueError(
                f'{path}:{line_number} is not a native DOTA row')
        try:
            coordinates = tuple(float(value) for value in fields[:8])
            difficulty = int(fields[9])
        except ValueError as error:
            raise ValueError(
                f'{path}:{line_number} is not a native DOTA row') from error
        label = fields[8]
        if label not in allowed:
            raise ValueError(
                f'{path}:{line_number} has unknown DOTA class {label}')
        objects.append(DOTAObject(
            coordinates=coordinates,
            label=label,
            difficulty=difficulty,
            line=line))
    return tuple(objects)


def _find_image(images_dir: Path, stem: str) -> Path:
    matches = [images_dir / (stem + suffix) for suffix in IMAGE_SUFFIXES
               if (images_dir / (stem + suffix)).is_file()]
    if len(matches) != 1:
        raise ValueError(
            f'expected one source image for {stem}, found {len(matches)}')
    return matches[0]


def _scan_source(source_root: Path,
                 classes: Sequence[str]) -> Tuple[DOTARecord, ...]:
    annotation_dir = source_root / 'ss_train' / 'annfiles'
    images_dir = source_root / 'ss_train' / 'images'
    if not annotation_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(
            'source root must contain ss_train/annfiles and ss_train/images')
    records = []
    for annotation in sorted(annotation_dir.glob('*.txt')):
        records.append(DOTARecord(
            stem=annotation.stem,
            image=_find_image(images_dir, annotation.stem).resolve(),
            annotation=annotation.resolve(),
            objects=parse_dota_annotation(annotation, classes)))
    if not records:
        raise ValueError('source DOTA training split is empty')
    return tuple(records)


def _density_bin(gt_count: int) -> str:
    if gt_count == 0:
        return 'empty'
    if gt_count <= 10:
        return 'sparse_1_10'
    if gt_count <= 50:
        return 'medium_11_50'
    if gt_count <= 200:
        return 'dense_51_200'
    return 'ultra_gt_200'


def _select_s1(records: Sequence[DOTARecord], quotas: Mapping[str, int],
               seed: int) -> Tuple[Tuple[DOTARecord, ...],
                                   Tuple[DOTARecord, ...]]:
    expected_bins = tuple(DEFAULT_S1_QUOTAS)
    if tuple(quotas) != expected_bins:
        raise ValueError(
            'S1 quota keys must follow the fixed density-bin order')
    pools = defaultdict(list)
    for record in records:
        pools[_density_bin(record.gt_count)].append(record)

    rng = random.Random(seed)
    train_records = []
    val_records = []
    for bin_name in expected_bins:
        pool = sorted(pools[bin_name], key=lambda item: item.stem)
        rng.shuffle(pool)
        requested = int(quotas[bin_name])
        if requested < 0 or len(pool) < requested:
            raise ValueError(
                f'S1 quota for {bin_name} requests {requested}, '
                f'but only {len(pool)} records exist')
        selected = pool[:requested]
        train_count = requested * 4 // 5
        if requested % 5:
            raise ValueError(
                f'S1 quota for {bin_name} must be divisible by 5')
        train_records.extend(selected[:train_count])
        val_records.extend(selected[train_count:])
    return (tuple(sorted(train_records, key=lambda item: item.stem)),
            tuple(sorted(val_records, key=lambda item: item.stem)))


def _select_s0(records: Sequence[DOTARecord], classes: Sequence[str],
               seed: int, nonempty_count: int, empty_count: int,
               dense_min: int) -> Tuple[DOTARecord, ...]:
    nonempty = [record for record in records if record.gt_count > 0]
    empty = [record for record in records if record.gt_count == 0]
    dense = [record for record in nonempty if record.gt_count > 200]
    if len(dense) < dense_min:
        raise ValueError(
            f'S0 dense requirement is {dense_min}, only {len(dense)} exist')
    if len(nonempty) < nonempty_count:
        raise ValueError('S0 nonempty quota cannot be satisfied')
    if len(empty) < empty_count:
        raise ValueError('S0 empty quota cannot be satisfied')

    rng = random.Random(seed + 1)
    dense = sorted(dense, key=lambda item: item.stem)
    nonempty = sorted(nonempty, key=lambda item: item.stem)
    empty = sorted(empty, key=lambda item: item.stem)
    rng.shuffle(dense)
    rng.shuffle(nonempty)
    rng.shuffle(empty)
    selected = list(dense[:dense_min])
    selected_stems = {record.stem for record in selected}
    missing = set(classes) - {
        label for record in selected for label in record.labels}
    rank = {record.stem: index for index, record in enumerate(nonempty)}
    while missing:
        candidates = [
            record for record in nonempty
            if record.stem not in selected_stems and
            missing.intersection(record.labels)
        ]
        if not candidates or len(selected) >= nonempty_count:
            raise ValueError(
                'S0 class coverage cannot be satisfied by nonempty quota')
        chosen = min(
            candidates,
            key=lambda record: (
                -len(missing.intersection(record.labels)),
                rank[record.stem]))
        selected.append(chosen)
        selected_stems.add(chosen.stem)
        missing.difference_update(chosen.labels)

    for record in nonempty:
        if len(selected) >= nonempty_count:
            break
        if record.stem not in selected_stems:
            selected.append(record)
            selected_stems.add(record.stem)
    selected.extend(empty[:empty_count])
    return tuple(sorted(selected, key=lambda item: item.stem))


def _prepare_view(output_root: Path, name: str) -> Tuple[Path, Path]:
    images = output_root / name / 'images'
    annotations = output_root / name / 'annfiles'
    images.mkdir(parents=True, exist_ok=False)
    annotations.mkdir(parents=True, exist_ok=False)
    return images, annotations


def _link_image(record: DOTARecord, image_dir: Path) -> None:
    destination = image_dir / record.image.name
    destination.symlink_to(record.image)


def _copy_native_annotation(record: DOTARecord, annotation_dir: Path) -> None:
    shutil.copyfile(record.annotation, annotation_dir / record.annotation.name)


def _write_base_annotation(record: DOTARecord, annotation_dir: Path,
                           novel_classes: Sequence[str]) -> int:
    novel = set(novel_classes)
    retained = []
    removed = 0
    for line in record.annotation.read_text(
            encoding='utf-8').splitlines(keepends=True):
        stripped = line.strip()
        fields = stripped.split()
        if not stripped or stripped.startswith('imagesource:') or \
                stripped.startswith('gsd:'):
            retained.append(line)
        elif len(fields) == 10 and fields[8] not in novel:
            retained.append(line)
        else:
            removed += 1
    (annotation_dir / record.annotation.name).write_text(
        ''.join(retained), encoding='utf-8')
    return removed


def _covered_classes(records: Sequence[DOTARecord],
                     classes: Sequence[str]) -> list:
    present = {label for record in records for label in record.labels}
    return [label for label in classes if label in present]


def _class_counts(records: Sequence[DOTARecord],
                  classes: Sequence[str]) -> Dict[str, int]:
    counts = Counter(
        obj.label for record in records for obj in record.objects)
    return {label: counts[label] for label in classes}


def build_subsets(
        source_root: Path,
        output_root: Path,
        seed: int,
        classes: Sequence[str] = DOTA2_CLASSES,
        novel_classes: Sequence[str] = NOVEL_CLASSES,
        s1_quotas: Mapping[str, int] = DEFAULT_S1_QUOTAS,
        s0_nonempty: int = 64,
        s0_empty: int = 32,
        s0_dense_min: int = 8) -> dict:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f'output root already exists and is immutable: {output_root}')
    if len(classes) != len(set(classes)):
        raise ValueError('classes must be unique')
    if not set(novel_classes) <= set(classes):
        raise ValueError('novel classes must be drawn from classes')

    records = _scan_source(source_root, classes)
    s0_records = _select_s0(
        records, classes, seed, s0_nonempty, s0_empty, s0_dense_min)
    s1_train, s1_val = _select_s1(records, s1_quotas, seed)
    s1_all = tuple(sorted(s1_train + s1_val, key=lambda item: item.stem))
    if _covered_classes(s0_records, classes) != list(classes):
        raise ValueError('S0 does not cover every requested class')
    if _covered_classes(s1_all, classes) != list(classes):
        raise ValueError('S1 does not cover every requested class')
    if not any(
            label in set(novel_classes)
            for record in s1_all for label in record.labels):
        raise ValueError('S1 contains no novel-class instances')

    output_root.mkdir(parents=True)
    view_dirs = {
        name: _prepare_view(output_root, name)
        for name in ('s0', 's1_train_all18', 's1_val_all18',
                     's1_train_base14')
    }
    membership = defaultdict(list)
    removed_novel_rows = Counter()

    for record in s0_records:
        _link_image(record, view_dirs['s0'][0])
        _copy_native_annotation(record, view_dirs['s0'][1])
        membership[record.stem].append('s0')
    for record in s1_train:
        for view in ('s1_train_all18', 's1_train_base14'):
            _link_image(record, view_dirs[view][0])
        _copy_native_annotation(record, view_dirs['s1_train_all18'][1])
        removed_novel_rows[record.stem] = _write_base_annotation(
            record, view_dirs['s1_train_base14'][1], novel_classes)
        membership[record.stem].extend(
            ('s1_train_all18', 's1_train_base14'))
    for record in s1_val:
        _link_image(record, view_dirs['s1_val_all18'][0])
        _copy_native_annotation(record, view_dirs['s1_val_all18'][1])
        membership[record.stem].append('s1_val_all18')

    selected_by_stem = {
        record.stem: record for record in s0_records + s1_train + s1_val}
    train_stems = {record.stem for record in s1_train}
    val_stems = {record.stem for record in s1_val}
    manifest_records = []
    for stem in sorted(selected_by_stem):
        record = selected_by_stem[stem]
        manifest_records.append({
            'stem': stem,
            'source_image': str(record.image),
            'source_annotation': str(record.annotation),
            'source_image_sha256': _sha256(record.image),
            'source_annotation_sha256': _sha256(record.annotation),
            'gt_count': record.gt_count,
            'labels': list(record.labels),
            'density_bin': _density_bin(record.gt_count),
            's0_member': record in s0_records,
            's1_split': (
                'train' if stem in train_stems else
                'val' if stem in val_stems else None),
            'views': membership[stem],
            'removed_novel_rows_in_base14': removed_novel_rows[stem],
        })

    manifest = {
        'schema_version': 1,
        'source_root': str(source_root),
        'seed': int(seed),
        'classes': list(classes),
        'novel_classes': list(novel_classes),
        'base_classes': [
            label for label in classes if label not in set(novel_classes)],
        'uses_pseudo_labels': False,
        's0': {
            'total_count': len(s0_records),
            'nonempty_count': sum(
                record.gt_count > 0 for record in s0_records),
            'empty_count': sum(
                record.gt_count == 0 for record in s0_records),
            'dense_gt_200_count': sum(
                record.gt_count > 200 for record in s0_records),
            'covered_classes': _covered_classes(s0_records, classes),
            'class_object_counts': _class_counts(s0_records, classes),
        },
        's1': {
            'quota': dict(s1_quotas),
            'train_count': len(s1_train),
            'val_count': len(s1_val),
            'total_count': len(s1_all),
            'covered_classes': _covered_classes(s1_all, classes),
            'class_object_counts': _class_counts(s1_all, classes),
            'train_stems': sorted(train_stems),
            'val_stems': sorted(val_stems),
            'train_val_overlap': sorted(train_stems.intersection(val_stems)),
        },
        'records': manifest_records,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    (output_root / 'manifest.json').write_bytes(manifest_bytes)
    (output_root / 'manifest.sha256').write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + '\n', encoding='ascii')
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_subsets(
        source_root=args.source_root,
        output_root=args.output_root,
        seed=args.seed)
    print(json.dumps({
        'manifest': str(args.output_root / 'manifest.json'),
        's0': manifest['s0'],
        's1': manifest['s1'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
