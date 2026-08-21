#!/usr/bin/env python
"""Build rotated DOTA-v2.0 validation splits.

The script keeps train/test untouched and writes one validation split per
rotation angle:

  out_root/<mode>/angle_030/images
  out_root/<mode>/angle_030/annfiles

Each annotation polygon is transformed by the same affine matrix used for the
image. Fully invisible boxes are dropped for fixed-canvas rotations.
"""

import argparse
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


STANDARD_ANGLES = tuple(range(0, 360, 30))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src-root',
        default='data/DOTA2_1024_500/ss_val',
        help='Source validation split with images/ and annfiles/.')
    parser.add_argument(
        '--out-root',
        default='data/DOTA2_1024_500/rot_val_standard',
        help='Output root for rotated validation splits.')
    parser.add_argument(
        '--angles',
        nargs='+',
        type=float,
        default=list(STANDARD_ANGLES),
        help='Rotation angles in degrees. Default: 0 30 ... 330.')
    parser.add_argument(
        '--mode',
        choices=['realistic', 'expanded'],
        default='realistic',
        help='realistic keeps the original canvas size; expanded keeps the '
        'full rotated image to avoid object cropping.')
    parser.add_argument(
        '--border-value',
        nargs=3,
        type=int,
        default=[114, 114, 114],
        help='BGR border fill value used by cv2.warpAffine.')
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument(
        '--image-ext',
        default='.png',
        help='Image extension used by the split.')
    return parser.parse_args()


def normalize_angle(angle):
    angle = angle % 360
    if abs(angle - round(angle)) < 1e-6:
        return f'{int(round(angle)):03d}'
    return f'{angle:06.2f}'.replace('.', 'p')


def rotation_matrix(width, height, angle, mode):
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    if mode == 'realistic':
        return matrix, width, height

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    out_w = int(math.ceil(height * sin + width * cos))
    out_h = int(math.ceil(height * cos + width * sin))
    matrix[0, 2] += (out_w - width) / 2.0
    matrix[1, 2] += (out_h - height) / 2.0
    return matrix, out_w, out_h


def transform_points(points, matrix):
    hom = np.concatenate(
        [points.astype(np.float32),
         np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1)
    return hom @ matrix.T


def intersects_canvas(points, width, height):
    xs = points[:, 0]
    ys = points[:, 1]
    return xs.max() >= 0 and xs.min() <= width and ys.max() >= 0 and ys.min(
    ) <= height


def safe_link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rotate_one(task):
    img_path, ann_path, out_img_path, out_ann_path, angle, mode, border_value = task
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to read image: {img_path}')

    height, width = img.shape[:2]
    matrix, out_w, out_h = rotation_matrix(width, height, angle, mode)

    if abs(angle % 360) < 1e-6 and mode == 'realistic':
        safe_link_or_copy(img_path, out_img_path)
        safe_link_or_copy(ann_path, out_ann_path)
        return 1, 0, 0

    rotated = cv2.warpAffine(
        img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(border_value))
    if not cv2.imwrite(str(out_img_path), rotated):
        raise RuntimeError(f'Failed to write image: {out_img_path}')

    kept = 0
    dropped = 0
    written = []
    if ann_path.exists():
        for raw in ann_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 10:
                dropped += 1
                continue
            points = np.array([float(x) for x in parts[:8]],
                              dtype=np.float32).reshape(4, 2)
            points = transform_points(points, matrix)
            if not intersects_canvas(points, out_w, out_h):
                dropped += 1
                continue
            coords = ' '.join(f'{x:.1f}' for x in points.reshape(-1))
            written.append(f'{coords} {parts[8]} {parts[9]}')
            kept += 1

    out_ann_path.write_text('\n'.join(written) + ('\n' if written else ''))
    return 1, kept, dropped


def build_angle(src_root, out_root, angle, mode, border_value, nproc, overwrite,
                limit, image_ext):
    angle_name = normalize_angle(angle)
    out_dir = out_root / mode / f'angle_{angle_name}'
    out_img_dir = out_dir / 'images'
    out_ann_dir = out_dir / 'annfiles'
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f'{out_dir} already exists. Use --overwrite to rebuild it.')
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    img_dir = src_root / 'images'
    ann_dir = src_root / 'annfiles'
    img_paths = sorted(img_dir.glob(f'*{image_ext}'))
    if limit > 0:
        img_paths = img_paths[:limit]

    tasks = []
    for img_path in img_paths:
        stem = img_path.stem
        tasks.append((img_path, ann_dir / f'{stem}.txt',
                      out_img_dir / img_path.name, out_ann_dir / f'{stem}.txt',
                      float(angle), mode, border_value))

    done = kept = dropped = 0
    if nproc <= 1:
        for task in tasks:
            d, k, r = rotate_one(task)
            done += d
            kept += k
            dropped += r
    else:
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(rotate_one, task) for task in tasks]
            for future in as_completed(futures):
                d, k, r = future.result()
                done += d
                kept += k
                dropped += r

    return {
        'angle': angle,
        'angle_name': angle_name,
        'mode': mode,
        'images': done,
        'kept_instances': kept,
        'dropped_instances': dropped,
        'images_dir': str(out_img_dir),
        'annfiles_dir': str(out_ann_dir),
    }


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    if not (src_root / 'images').is_dir() or not (src_root /
                                                  'annfiles').is_dir():
        raise FileNotFoundError(
            f'{src_root} must contain images/ and annfiles/.')

    summaries = []
    for angle in args.angles:
        summary = build_angle(src_root, out_root, angle, args.mode,
                              args.border_value, args.nproc, args.overwrite,
                              args.limit, args.image_ext)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    manifest_path = out_root / args.mode / 'manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                'src_root': str(src_root),
                'out_root': str(out_root),
                'mode': args.mode,
                'angles': summaries,
            },
            ensure_ascii=False,
            indent=2) + '\n')
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
