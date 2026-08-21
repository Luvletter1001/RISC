#!/usr/bin/env python
"""Prepare per-angle DOTA2 validation splits for rotation response curves.

This script avoids duplicating existing 30-degree rotated data. Angles that
already exist in the flattened ``rot_val_standard`` split are materialized as
hardlink/symlink/copy views. Missing angles, such as 15/45/.../345, are built
from ``ss_val`` with the same fixed-canvas rotation code used by
``build_dotav2_rot_val.py``.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from build_dotav2_rot_val import build_angle, normalize_angle  # noqa: E402


DEFAULT_ANGLES = tuple(range(0, 360, 15))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        default='/data1/zcy/OpenRSD/data/DOTA2_1024_500',
        help='DOTA2 root containing ss_val/ and rot_val_standard/.')
    parser.add_argument(
        '--out-root',
        default='/data1/zcy/OpenRSD/data/DOTA2_1024_500/angle_sweep_val',
        help='Output root. Splits are written under <out-root>/<mode>/angle_xxx.')
    parser.add_argument(
        '--angles',
        nargs='+',
        type=float,
        default=list(DEFAULT_ANGLES),
        help='Angles to prepare. Default: 0 15 ... 345.')
    parser.add_argument('--mode', default='realistic', choices=['realistic'])
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--link-mode',
        choices=['hardlink', 'symlink', 'copy'],
        default='hardlink',
        help='How to materialize existing 30-degree split views.')
    parser.add_argument('--image-ext', default='.png')
    return parser.parse_args()


def link_or_copy(src, dst, mode, overwrite):
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return False
        dst.unlink()

    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'symlink':
        dst.symlink_to(os.path.relpath(src, dst.parent))
    else:
        os.link(src, dst)
    return True


def count_files(path, pattern):
    return sum(1 for _ in path.glob(pattern))


def materialize_existing_angle(data_root, out_root, angle, link_mode, overwrite,
                               image_ext):
    angle_name = normalize_angle(angle)
    src_img_dir = data_root / 'rot_val_standard' / 'images'
    src_ann_dir = data_root / 'rot_val_standard' / 'annfiles'
    out_dir = out_root / 'realistic' / f'angle_{angle_name}'
    out_img_dir = out_dir / 'images'
    out_ann_dir = out_dir / 'annfiles'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    prefix = f'angle_{angle_name}__'
    image_paths = sorted(src_img_dir.glob(f'{prefix}*{image_ext}'))
    if not image_paths:
        raise FileNotFoundError(
            f'No existing rotated images found for {prefix} in {src_img_dir}')

    linked_images = 0
    linked_annfiles = 0
    missing_annfiles = []
    for image_path in image_paths:
        ann_path = src_ann_dir / f'{image_path.stem}.txt'
        if not ann_path.exists():
            missing_annfiles.append(str(ann_path))
            continue
        if link_or_copy(image_path, out_img_dir / image_path.name, link_mode,
                        overwrite):
            linked_images += 1
        if link_or_copy(ann_path, out_ann_dir / ann_path.name, link_mode,
                        overwrite):
            linked_annfiles += 1

    if missing_annfiles:
        raise RuntimeError(
            f'{len(missing_annfiles)} missing annfiles for angle {angle_name}')

    return {
        'angle': angle,
        'angle_name': angle_name,
        'source': 'rot_val_standard_view',
        'images': len(image_paths),
        'linked_images': linked_images,
        'linked_annfiles': linked_annfiles,
        'images_dir': str(out_img_dir),
        'annfiles_dir': str(out_ann_dir),
    }


def split_ready(out_root, angle, image_ext, expected_images):
    angle_name = normalize_angle(angle)
    out_dir = out_root / 'realistic' / f'angle_{angle_name}'
    img_dir = out_dir / 'images'
    ann_dir = out_dir / 'annfiles'
    return (img_dir.is_dir() and ann_dir.is_dir()
            and count_files(img_dir, f'*{image_ext}') == expected_images
            and count_files(ann_dir, '*.txt') == expected_images)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    ss_val = data_root / 'ss_val'
    expected_images = count_files(ss_val / 'images', f'*{args.image_ext}')
    if expected_images == 0:
        raise FileNotFoundError(f'No validation images found in {ss_val}')

    summaries = []
    for angle in args.angles:
        angle_name = normalize_angle(angle)
        if split_ready(out_root, angle, args.image_ext,
                       expected_images) and not args.overwrite:
            out_dir = out_root / 'realistic' / f'angle_{angle_name}'
            summary = {
                'angle': angle,
                'angle_name': angle_name,
                'source': 'existing',
                'images': expected_images,
                'images_dir': str(out_dir / 'images'),
                'annfiles_dir': str(out_dir / 'annfiles'),
            }
        elif int(round(angle)) % 30 == 0 and abs(angle - round(angle)) < 1e-6:
            summary = materialize_existing_angle(data_root, out_root, angle,
                                                 args.link_mode,
                                                 args.overwrite,
                                                 args.image_ext)
        else:
            summary = build_angle(
                ss_val, out_root, angle, args.mode, [114, 114, 114],
                args.nproc, args.overwrite, 0, args.image_ext)
            summary['source'] = 'generated_from_ss_val'

        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    manifest = {
        'data_root': str(data_root),
        'out_root': str(out_root),
        'mode': args.mode,
        'expected_images_per_angle': expected_images,
        'angles': summaries,
    }
    manifest_path = out_root / args.mode / 'manifest_angle_sweep.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
