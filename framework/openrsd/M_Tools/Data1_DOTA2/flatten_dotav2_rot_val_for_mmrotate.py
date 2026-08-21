#!/usr/bin/env python
"""Flatten rotated DOTA-v2 validation splits into one mmrotate DOTA split.

Input layout:
  rot_val_standard/realistic/angle_030/images/*.png
  rot_val_standard/realistic/angle_030/annfiles/*.txt

Output layout:
  rot_val_standard/realistic/mmrotate_dota/images/angle_030__*.png
  rot_val_standard/realistic/mmrotate_dota/annfiles/angle_030__*.txt

mmrotate's DOTADataset expects a single image directory and a single annotation
directory. Prefixing the angle keeps file ids unique across rotations.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src-root',
        default='data/DOTA2_1024_500/rot_val_standard/realistic',
        help='Root containing angle_xxx directories.')
    parser.add_argument(
        '--out-root',
        default='data/DOTA2_1024_500/rot_val_standard/realistic/mmrotate_dota',
        help='Output DOTA-style split root.')
    parser.add_argument(
        '--image-ext',
        default='.png',
        help='Image extension to include.')
    parser.add_argument(
        '--link-mode',
        choices=['symlink', 'hardlink', 'copy'],
        default='symlink',
        help='How to materialize files in the flattened split.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace existing files in the output split.')
    parser.add_argument(
        '--angles',
        nargs='*',
        default=None,
        help='Optional angle directory names or suffixes, e.g. angle_030 060.')
    return parser.parse_args()


def normalize_angle_name(name):
    return name if name.startswith('angle_') else f'angle_{name}'


def link_or_copy(src, dst, mode, overwrite):
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return False
        dst.unlink()

    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    else:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)
    return True


def collect_angle_dirs(src_root, angles):
    if angles:
        angle_dirs = [src_root / normalize_angle_name(angle) for angle in angles]
    else:
        angle_dirs = sorted(p for p in src_root.glob('angle_*') if p.is_dir())

    missing = [
        str(p) for p in angle_dirs
        if not (p / 'images').is_dir() or not (p / 'annfiles').is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            'These angle dirs do not contain images/ and annfiles/: '
            + ', '.join(missing))
    return angle_dirs


def flatten_angle(angle_dir, out_img_dir, out_ann_dir, image_ext, link_mode,
                  overwrite):
    image_dir = angle_dir / 'images'
    ann_dir = angle_dir / 'annfiles'
    prefix = angle_dir.name
    image_paths = sorted(image_dir.glob(f'*{image_ext}'))

    linked_images = 0
    linked_annfiles = 0
    missing_annfiles = []

    for image_path in image_paths:
        out_stem = f'{prefix}__{image_path.stem}'
        out_image = out_img_dir / f'{out_stem}{image_ext}'
        if link_or_copy(image_path, out_image, link_mode, overwrite):
            linked_images += 1

        ann_path = ann_dir / f'{image_path.stem}.txt'
        if not ann_path.exists():
            missing_annfiles.append(str(ann_path))
            continue

        out_ann = out_ann_dir / f'{out_stem}.txt'
        if link_or_copy(ann_path, out_ann, link_mode, overwrite):
            linked_annfiles += 1

    return {
        'angle_dir': str(angle_dir),
        'images': len(image_paths),
        'linked_images': linked_images,
        'linked_annfiles': linked_annfiles,
        'missing_annfiles': missing_annfiles,
    }


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_img_dir = out_root / 'images'
    out_ann_dir = out_root / 'annfiles'

    angle_dirs = collect_angle_dirs(src_root, args.angles)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for angle_dir in angle_dirs:
        summary = flatten_angle(angle_dir, out_img_dir, out_ann_dir,
                                args.image_ext, args.link_mode,
                                args.overwrite)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    missing_total = sum(len(item['missing_annfiles']) for item in summaries)
    manifest = {
        'src_root': str(src_root),
        'out_root': str(out_root),
        'image_ext': args.image_ext,
        'link_mode': args.link_mode,
        'angles': summaries,
        'total_images': sum(item['images'] for item in summaries),
        'total_missing_annfiles': missing_total,
        'mmrotate_dataset': {
            'data_root': str(out_root),
            'img_dir': 'images',
            'ann_dir': 'annfiles',
        },
    }
    manifest_path = out_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                             + '\n')
    print(f'Wrote {manifest_path}')
    if missing_total:
        raise RuntimeError(f'{missing_total} images are missing annfiles.')


if __name__ == '__main__':
    main()
