#!/usr/bin/env python
"""Build HRSC validation rotation splits in DOTA-style layout.

Input copied HRSC layout:
  data/HRSC_unzip/dota/images/val/*.png
  data/HRSC_unzip/dota/labels/val/*.txt

The HRSC `dota/labels` files in this workspace use normalized polygon labels:
  class_id x1 y1 x2 y2 x3 y3 x4 y4

Outputs:
  data/HRSC_unzip/dota/ss_val/{images,annfiles}
  data/HRSC_unzip/dota/rot_val_standard/{images,annfiles}
  data/HRSC_unzip/dota/rot_val_standard_0_90_180_270/{images,annfiles}
  data/HRSC_unzip/dota/rot_val_standard_30_60_120_150_210_240_300_330/{images,annfiles}

All outputs are materialized as regular files, not symlinks.
"""

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


SPLITS = {
    'rot_val_standard': tuple(range(0, 360, 30)),
    'rot_val_standard_0_90_180_270': (0, 90, 180, 270),
    'rot_val_standard_30_60_120_150_210_240_300_330':
    (30, 60, 120, 150, 210, 240, 300, 330),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        default='/data1/zcy/OpenRSD/data/HRSC_unzip',
        help='Copied HRSC root.')
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--border-value', nargs=3, type=int, default=[114, 114, 114])
    return parser.parse_args()


def angle_name(angle):
    return f'{int(angle) % 360:03d}'


def rotation_matrix(width, height, angle):
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    return cv2.getRotationMatrix2D(center, float(angle), 1.0)


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


def read_normalized_ann(label_path, width, height):
    anns = []
    dropped = 0
    if not label_path.exists():
        return anns, 1

    for raw in label_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 9:
            dropped += 1
            continue
        try:
            coords = [float(x) for x in parts[1:9]]
        except ValueError:
            dropped += 1
            continue
        points = np.array(coords, dtype=np.float32).reshape(4, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        anns.append(points)
    return anns, dropped


def write_dota_ann(ann_path, polys):
    lines = []
    for points in polys:
        coords = ' '.join(f'{x:.1f}' for x in points.reshape(-1))
        lines.append(f'{coords} ship 0')
    ann_path.write_text('\n'.join(lines) + ('\n' if lines else ''))


def prepare_base_one(task):
    img_path, label_path, out_img_path, out_ann_path = task
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to read image: {img_path}')
    height, width = img.shape[:2]
    anns, dropped = read_normalized_ann(label_path, width, height)
    shutil.copy2(img_path, out_img_path)
    write_dota_ann(out_ann_path, anns)
    return {'images': 1, 'instances': len(anns), 'dropped_labels': dropped}


def rotate_one(task):
    img_path, ann_path, out_img_path, out_ann_path, angle, border_value = task
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to read image: {img_path}')
    height, width = img.shape[:2]
    matrix = rotation_matrix(width, height, angle)

    if int(angle) % 360 == 0:
        shutil.copy2(img_path, out_img_path)
        shutil.copy2(ann_path, out_ann_path)
        kept = sum(1 for line in ann_path.read_text().splitlines() if line.strip())
        return {'images': 1, 'kept_instances': kept, 'dropped_instances': 0}

    rotated = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(border_value))
    if not cv2.imwrite(str(out_img_path), rotated):
        raise RuntimeError(f'Failed to write image: {out_img_path}')

    kept_polys = []
    dropped = 0
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
        if not intersects_canvas(points, width, height):
            dropped += 1
            continue
        kept_polys.append(points)

    write_dota_ann(out_ann_path, kept_polys)
    return {
        'images': 1,
        'kept_instances': len(kept_polys),
        'dropped_instances': dropped,
    }


def run_tasks(tasks, fn, nproc):
    total = {}
    if nproc <= 1:
        iterator = (fn(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(fn, task) for task in tasks]
            iterator = (future.result() for future in as_completed(futures))

            for result in iterator:
                for key, value in result.items():
                    total[key] = total.get(key, 0) + value
            return total

    for result in iterator:
        for key, value in result.items():
            total[key] = total.get(key, 0) + value
    return total


def ensure_clean_dir(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f'{path} exists. Use --overwrite to rebuild.')
        shutil.rmtree(path)
    (path / 'images').mkdir(parents=True)
    (path / 'annfiles').mkdir(parents=True)


def build_base(dota_root, nproc, overwrite):
    src_img_dir = dota_root / 'images' / 'val'
    src_label_dir = dota_root / 'labels' / 'val'
    out_root = dota_root / 'ss_val'
    ensure_clean_dir(out_root, overwrite)

    image_paths = sorted(src_img_dir.glob('*.png'))
    tasks = []
    for img_path in image_paths:
        stem = img_path.stem
        tasks.append((img_path, src_label_dir / f'{stem}.txt',
                      out_root / 'images' / img_path.name,
                      out_root / 'annfiles' / f'{stem}.txt'))

    summary = run_tasks(tasks, prepare_base_one, nproc)
    summary.update({
        'name': 'ss_val',
        'root': str(out_root),
        'source_images': str(src_img_dir),
        'source_labels': str(src_label_dir),
    })
    (out_root / 'manifest.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def build_rot_split(dota_root, split_name, angles, nproc, overwrite,
                    border_value):
    src_root = dota_root / 'ss_val'
    src_img_dir = src_root / 'images'
    src_ann_dir = src_root / 'annfiles'
    out_root = dota_root / split_name
    ensure_clean_dir(out_root, overwrite)

    tasks = []
    image_paths = sorted(src_img_dir.glob('*.png'))
    for angle in angles:
        prefix = f'angle_{angle_name(angle)}'
        for img_path in image_paths:
            stem = img_path.stem
            out_stem = f'{prefix}__{stem}'
            tasks.append((img_path, src_ann_dir / f'{stem}.txt',
                          out_root / 'images' / f'{out_stem}.png',
                          out_root / 'annfiles' / f'{out_stem}.txt', angle,
                          border_value))

    summary = run_tasks(tasks, rotate_one, nproc)
    summary.update({
        'name': split_name,
        'root': str(out_root),
        'source_root': str(src_root),
        'angles': list(angles),
        'images_per_angle': len(image_paths),
        'expected_images': len(image_paths) * len(angles),
    })
    (out_root / 'manifest.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    args = parse_args()
    root = Path(args.root)
    dota_root = root / 'dota'
    if not (dota_root / 'images' / 'val').is_dir():
        raise FileNotFoundError(f'Missing {(dota_root / "images" / "val")}')
    if not (dota_root / 'labels' / 'val').is_dir():
        raise FileNotFoundError(f'Missing {(dota_root / "labels" / "val")}')

    summaries = [build_base(dota_root, args.nproc, args.overwrite)]
    for split_name, angles in SPLITS.items():
        summaries.append(
            build_rot_split(dota_root, split_name, angles, args.nproc,
                            args.overwrite, args.border_value))
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    manifest = {
        'root': str(root),
        'dota_root': str(dota_root),
        'outputs': summaries,
        'format': 'DOTA-style images/ + annfiles/, class name ship',
        'materialization': 'regular files, no symlinks',
    }
    manifest_path = dota_root / 'rotation_splits_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                             + '\n')
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
