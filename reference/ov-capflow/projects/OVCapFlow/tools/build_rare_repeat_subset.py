"""Build a fixed-size rare-positive replacement view of an S1 train set."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple


DEFAULT_RARE_CLASSES = (
    'airport', 'container-crane', 'helipad', 'helicopter')


def _labels(path: Path) -> Tuple[str, ...]:
    labels = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 9:
            labels.append(fields[8])
    return tuple(labels)


def _stable_order(stems: Iterable[str], seed: int):
    return sorted(
        stems,
        key=lambda stem: (
            hashlib.sha256(f'{seed}:{stem}'.encode()).hexdigest(), stem))


def _link(source: Path, target: Path) -> None:
    target.symlink_to(source.resolve())


def build_rare_repeat_subset(
        source_root: Path,
        output_root: Path,
        repeat_factor: int = 4,
        seed: int = 20260718,
        rare_classes: Sequence[str] = DEFAULT_RARE_CLASSES) -> Dict:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if repeat_factor < 2:
        raise ValueError('repeat_factor must be at least 2')
    if output_root.exists():
        raise FileExistsError(f'output already exists: {output_root}')

    ann_dir = source_root / 'annfiles'
    image_dir = source_root / 'images'
    annotations = sorted(ann_dir.glob('*.txt'))
    images = {path.stem: path for path in image_dir.iterdir()
              if path.is_file()}
    if not annotations:
        raise ValueError(f'no annotations found in {ann_dir}')
    stems = [path.stem for path in annotations]
    missing_images = sorted(set(stems) - set(images))
    if missing_images:
        raise ValueError('missing images: ' + ', '.join(missing_images[:5]))

    labels_by_stem = {path.stem: _labels(path) for path in annotations}
    rare_set = set(rare_classes)
    rare_stems = sorted(
        stem for stem, labels in labels_by_stem.items()
        if rare_set.intersection(labels))
    empty_stems = sorted(
        stem for stem, labels in labels_by_stem.items() if not labels)
    if not rare_stems:
        raise ValueError('source set contains no rare-positive images')

    extra_count = len(rare_stems) * (repeat_factor - 1)
    if extra_count > len(empty_stems):
        raise ValueError(
            f'need {extra_count} empty replacements, found {len(empty_stems)}')
    removed_empty = set(_stable_order(empty_stems, seed)[:extra_count])
    retained_stems = [stem for stem in stems if stem not in removed_empty]

    out_ann = output_root / 'annfiles'
    out_images = output_root / 'images'
    out_ann.mkdir(parents=True)
    out_images.mkdir(parents=True)
    annotation_map = {path.stem: path for path in annotations}
    for stem in retained_stems:
        _link(annotation_map[stem], out_ann / annotation_map[stem].name)
        _link(images[stem], out_images / images[stem].name)
    for stem in rare_stems:
        for repeat_index in range(1, repeat_factor):
            alias = f'{stem}__rare_repeat_{repeat_index:02d}'
            _link(annotation_map[stem], out_ann / f'{alias}.txt')
            _link(images[stem], out_images / f'{alias}{images[stem].suffix}')

    instance_counts = Counter()
    rare_image_counts = Counter()
    for stem in rare_stems:
        labels = labels_by_stem[stem]
        instance_counts.update(label for label in labels if label in rare_set)
        rare_image_counts.update(set(labels).intersection(rare_set))
    output_files = len(retained_stems) + extra_count
    if output_files != len(stems):
        raise AssertionError('fixed-size replacement invariant failed')

    manifest = dict(
        source_root=str(source_root),
        output_root=str(output_root),
        seed=seed,
        repeat_factor=repeat_factor,
        rare_classes=list(rare_classes),
        source_images=len(stems),
        output_images=output_files,
        source_empty_images=len(empty_stems),
        output_empty_images=len(empty_stems) - extra_count,
        rare_unique_images=len(rare_stems),
        rare_output_exposures=len(rare_stems) * repeat_factor,
        rare_image_counts=dict(sorted(rare_image_counts.items())),
        rare_instance_counts=dict(sorted(instance_counts.items())),
        removed_empty_stems=sorted(removed_empty),
        rare_source_stems=rare_stems)
    encoded = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False).encode()
    manifest_path = output_root / 'manifest.json'
    manifest_path.write_bytes(encoded + b'\n')
    manifest_sha256 = hashlib.sha256(encoded + b'\n').hexdigest()
    (output_root / 'manifest.sha256').write_text(
        f'{manifest_sha256}  manifest.json\n')
    manifest['manifest_sha256'] = manifest_sha256
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--repeat-factor', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260718)
    parser.add_argument(
        '--rare-classes', nargs='+', default=DEFAULT_RARE_CLASSES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_rare_repeat_subset(
        args.source_root,
        args.output_root,
        repeat_factor=args.repeat_factor,
        seed=args.seed,
        rare_classes=args.rare_classes), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
