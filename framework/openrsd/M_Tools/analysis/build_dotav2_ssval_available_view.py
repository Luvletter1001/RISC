#!/usr/bin/env python
"""Build a symlinked DOTAv2 ss_val view from locally available split files."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument(
        '--roots',
        nargs='+',
        required=True,
        help='Dataset split roots containing images/ and annfiles/.')
    parser.add_argument('--split', default='ss_val')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def iter_existing_dirs(root):
    root = Path(root)
    candidates = [
        (root / 'images', root / 'annfiles'),
        (root / 'ss_val' / 'images', root / 'ss_val' / 'annfiles'),
        (root / 'val' / 'images', root / 'val' / 'annfiles'),
        (root / 'train' / 'images', root / 'train' / 'annfiles'),
    ]
    for img_dir, ann_dir in candidates:
        if img_dir.is_dir() and ann_dir.is_dir():
            yield img_dir, ann_dir


def build_file_index(roots):
    image_index = {}
    ann_index = {}
    source_counts = Counter()
    duplicate_counts = Counter()

    for root in roots:
        for img_dir, ann_dir in iter_existing_dirs(root):
            source = str(img_dir.parent)
            for path in sorted(img_dir.iterdir()):
                if not path.is_file():
                    continue
                if path.name in image_index:
                    duplicate_counts['images'] += 1
                    continue
                image_index[path.name] = path
                source_counts[f'images:{source}'] += 1
            for path in sorted(ann_dir.iterdir()):
                if not path.is_file() or path.suffix != '.txt':
                    continue
                if path.name in ann_index:
                    duplicate_counts['annfiles'] += 1
                    continue
                ann_index[path.name] = path
                source_counts[f'annfiles:{source}'] += 1
    return image_index, ann_index, source_counts, duplicate_counts


def safe_symlink(src, dst, overwrite=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.symlink_to(src.resolve(strict=True))


def main():
    args = parse_args()
    manifest = Path(args.manifest)
    out_root = Path(args.out_root)
    out_img_dir = out_root / args.split / 'images'
    out_ann_dir = out_root / args.split / 'annfiles'
    out_root.mkdir(parents=True, exist_ok=True)

    image_index, ann_index, source_counts, duplicate_counts = build_file_index(
        args.roots)

    total = 0
    linked = 0
    missing_rows = []
    linked_sources = Counter()

    with manifest.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') != args.split:
                continue
            total += 1
            image_file = row['image_file']
            ann_file = row['ann_file']
            image_src = image_index.get(image_file)
            ann_src = ann_index.get(ann_file)
            if image_src is None or ann_src is None:
                missing_rows.append({
                    'image_file': image_file,
                    'ann_file': ann_file,
                    'source_id': row.get('source_id', ''),
                    'missing_image': image_src is None,
                    'missing_ann': ann_src is None,
                })
                continue

            safe_symlink(image_src, out_img_dir / image_file, args.overwrite)
            safe_symlink(ann_src, out_ann_dir / ann_file, args.overwrite)
            linked += 1
            linked_sources[str(image_src.parent.parent)] += 1

    missing_csv = out_root / f'{args.split}_missing.csv'
    with missing_csv.open('w', newline='') as f:
        fieldnames = [
            'image_file', 'ann_file', 'source_id', 'missing_image',
            'missing_ann'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)

    summary = {
        'manifest': str(manifest),
        'split': args.split,
        'out_root': str(out_root),
        'roots': args.roots,
        'manifest_rows': total,
        'linked_rows': linked,
        'missing_rows': len(missing_rows),
        'coverage': linked / total if total else 0.0,
        'indexed_images': len(image_index),
        'indexed_annfiles': len(ann_index),
        'source_counts': dict(source_counts),
        'duplicate_counts': dict(duplicate_counts),
        'linked_sources': dict(linked_sources),
        'missing_csv': str(missing_csv),
    }
    (out_root / 'coverage.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
    (out_root / 'README.md').write_text(
        f'# DOTAv2 {args.split} available view\n\n'
        f'- manifest rows: {total}\n'
        f'- linked rows: {linked}\n'
        f'- missing rows: {len(missing_rows)}\n'
        f'- coverage: {summary["coverage"]:.6f}\n'
        f'- missing csv: {missing_csv.name}\n')

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
