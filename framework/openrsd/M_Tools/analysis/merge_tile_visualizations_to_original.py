#!/usr/bin/env python
"""Merge visualized 1024 DOTA tiles back into original-scale images."""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
TILE_RE = re.compile(r'^(?P<orig>.+)__(?P<size>\d+)__(?P<x>\d+)___(?P<y>\d+)$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-summary-csv', required=True)
    parser.add_argument('--tile-img-dir', required=True)
    parser.add_argument('--vis-subdir', default='vis_wrong_class_iou_gt0p7')
    parser.add_argument('--output-subdir',
                        default='vis_wrong_class_iou_gt0p7_merged_original')
    parser.add_argument('--original-img-dir', default='')
    parser.add_argument('--diff-thr', type=int, default=18)
    parser.add_argument('--dilate-iter', type=int, default=1)
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--max-originals-per-model', type=int, default=0)
    return parser.parse_args()


def parse_tile_stem(stem):
    match = TILE_RE.match(stem)
    if not match:
        raise ValueError(f'not a DOTA split tile name: {stem}')
    return {
        'orig_id': match.group('orig'),
        'size': int(match.group('size')),
        'x': int(match.group('x')),
        'y': int(match.group('y')),
    }


def image_files(root):
    root = Path(root)
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def find_image(root, stem):
    if not root:
        return None
    root = Path(root)
    for suffix in IMAGE_SUFFIXES:
        path = root / f'{stem}{suffix}'
        if path.exists():
            return path
    return None


def resolve_path(path, cwd):
    path = Path(path)
    if path.is_absolute():
        return path
    return cwd / path


def index_base_tiles(tile_img_dir):
    groups = defaultdict(list)
    for path in image_files(tile_img_dir):
        try:
            info = parse_tile_stem(path.stem)
        except ValueError:
            continue
        groups[info['orig_id']].append({
            'path': path,
            'x': info['x'],
            'y': info['y'],
            'size': info['size'],
        })
    for tiles in groups.values():
        tiles.sort(key=lambda item: (item['y'], item['x'], str(item['path'])))
    return groups


def read_manifest(manifest_path, cwd):
    rows_by_orig = defaultdict(list)
    with Path(manifest_path).open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = parse_tile_stem(row['img_id'])
            row.update(info)
            row['vis_path'] = resolve_path(row['vis_path'], cwd)
            row['image_path'] = resolve_path(row['image_path'], cwd)
            for key in ('wrong_boxes', 'correct_boxes', 'ignored_boxes',
                        'localized_boxes'):
                row[key] = int(row.get(key) or 0)
            rows_by_orig[info['orig_id']].append(row)
    for rows in rows_by_orig.values():
        rows.sort(key=lambda item: (item['y'], item['x'], item['img_id']))
    return rows_by_orig


def paste_image(canvas, img, x, y):
    h, w = img.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + w)
    y1 = min(canvas_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src_x0 = x0 - x
    src_y0 = y0 - y
    canvas[y0:y1, x0:x1] = img[src_y0:src_y0 + (y1 - y0),
                               src_x0:src_x0 + (x1 - x0)]


def paste_diff(canvas, vis_img, base_img, x, y, diff_thr, dilate_iter):
    h = min(vis_img.shape[0], base_img.shape[0])
    w = min(vis_img.shape[1], base_img.shape[1])
    vis = vis_img[:h, :w]
    base = base_img[:h, :w]
    diff = np.max(np.abs(vis.astype(np.int16) - base.astype(np.int16)), axis=2)
    mask = diff > diff_thr
    if dilate_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel,
                          iterations=dilate_iter).astype(bool)

    canvas_h, canvas_w = canvas.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + w)
    y1 = min(canvas_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0
    src_x0 = x0 - x
    src_y0 = y0 - y
    src_mask = mask[src_y0:src_y0 + (y1 - y0), src_x0:src_x0 + (x1 - x0)]
    src_vis = vis[src_y0:src_y0 + (y1 - y0), src_x0:src_x0 + (x1 - x0)]
    region = canvas[y0:y1, x0:x1]
    region[src_mask] = src_vis[src_mask]
    return int(src_mask.sum())


def build_canvas(orig_id, base_tiles, original_img_dir):
    original_path = find_image(original_img_dir, orig_id)
    if original_path:
        canvas = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if canvas is not None:
            return canvas, str(original_path), 'original_image'

    max_w = 0
    max_h = 0
    loaded_tiles = []
    for tile in base_tiles:
        img = cv2.imread(str(tile['path']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        max_w = max(max_w, tile['x'] + w)
        max_h = max(max_h, tile['y'] + h)
        loaded_tiles.append((tile, img))
    if max_w == 0 or max_h == 0:
        return None, '', 'missing'

    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    for tile, img in loaded_tiles:
        paste_image(canvas, img, tile['x'], tile['y'])
    return canvas, '', 'tile_mosaic'


def render_model(row, base_groups, args, cwd):
    model = row['model']
    output_dir = Path(row['output_dir'])
    if not output_dir.is_absolute():
        output_dir = cwd / output_dir
    vis_dir = output_dir / args.vis_subdir
    manifest_path = vis_dir / 'manifest.csv'
    merged_dir = output_dir / args.output_subdir
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_manifest = merged_dir / 'manifest.csv'
    summary_path = merged_dir / 'summary.json'

    rows_by_orig = read_manifest(manifest_path, cwd)
    manifest_rows = []
    stats = Counter()

    for orig_id in sorted(rows_by_orig):
        if args.max_originals_per_model > 0:
            if stats['originals_rendered'] >= args.max_originals_per_model:
                break

        vis_rows = rows_by_orig[orig_id]
        base_tiles = base_groups.get(orig_id, [])
        canvas, source_path, source_kind = build_canvas(
            orig_id, base_tiles, args.original_img_dir)
        if canvas is None:
            stats['missing_base_originals'] += 1
            continue

        overlay_pixels = 0
        missing_vis_tiles = 0
        missing_base_tiles = 0
        for vis_row in vis_rows:
            vis_img = cv2.imread(str(vis_row['vis_path']), cv2.IMREAD_COLOR)
            if vis_img is None:
                missing_vis_tiles += 1
                continue
            base_img = cv2.imread(str(vis_row['image_path']), cv2.IMREAD_COLOR)
            if base_img is None:
                missing_base_tiles += 1
                paste_image(canvas, vis_img, vis_row['x'], vis_row['y'])
                continue
            overlay_pixels += paste_diff(canvas, vis_img, base_img,
                                         vis_row['x'], vis_row['y'],
                                         args.diff_thr, args.dilate_iter)

        out_path = merged_dir / f'{orig_id}.jpg'
        ok = cv2.imwrite(str(out_path), canvas,
                         [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok:
            stats['write_failures'] += 1
            continue

        wrong_boxes = sum(item['wrong_boxes'] for item in vis_rows)
        correct_boxes = sum(item['correct_boxes'] for item in vis_rows)
        ignored_boxes = sum(item['ignored_boxes'] for item in vis_rows)
        localized_boxes = sum(item['localized_boxes'] for item in vis_rows)
        stats['originals_rendered'] += 1
        stats['vis_tiles_merged'] += len(vis_rows)
        stats['base_tiles_used'] += len(base_tiles)
        stats['wrong_boxes'] += wrong_boxes
        stats['correct_boxes'] += correct_boxes
        stats['ignored_boxes'] += ignored_boxes
        stats['localized_boxes'] += localized_boxes
        stats['overlay_pixels'] += overlay_pixels
        stats['missing_vis_tiles'] += missing_vis_tiles
        stats['missing_base_tiles'] += missing_base_tiles

        manifest_rows.append({
            'model': model,
            'orig_id': orig_id,
            'merged_path': str(out_path.relative_to(cwd)),
            'source_kind': source_kind,
            'source_path': source_path,
            'canvas_width': canvas.shape[1],
            'canvas_height': canvas.shape[0],
            'base_tiles': len(base_tiles),
            'vis_tiles': len(vis_rows),
            'wrong_boxes': wrong_boxes,
            'correct_boxes': correct_boxes,
            'ignored_boxes': ignored_boxes,
            'localized_boxes': localized_boxes,
            'overlay_pixels': overlay_pixels,
            'missing_vis_tiles': missing_vis_tiles,
            'missing_base_tiles': missing_base_tiles,
        })

    with merged_manifest.open('w', newline='') as f:
        fieldnames = [
            'model', 'orig_id', 'merged_path', 'source_kind', 'source_path',
            'canvas_width', 'canvas_height', 'base_tiles', 'vis_tiles',
            'wrong_boxes', 'correct_boxes', 'ignored_boxes',
            'localized_boxes', 'overlay_pixels', 'missing_vis_tiles',
            'missing_base_tiles'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        'model': model,
        'vis_dir': str(vis_dir.relative_to(cwd)),
        'merged_dir': str(merged_dir.relative_to(cwd)),
        'manifest_csv': str(merged_manifest.relative_to(cwd)),
        'source_rule': (
            'original image when --original-img-dir matches; otherwise '
            'reconstructed from all available split tiles'),
        'overlay_rule': (
            f'diff mask from visualized tile vs base tile, diff_thr={args.diff_thr}, '
            f'dilate_iter={args.dilate_iter}'),
        'stats': dict(stats),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    args = parse_args()
    cwd = Path.cwd()
    tile_img_dir = Path(args.tile_img_dir)
    base_groups = index_base_tiles(tile_img_dir)
    print(json.dumps({
        'tile_img_dir': str(tile_img_dir),
        'base_originals': len(base_groups),
        'base_tiles': sum(len(v) for v in base_groups.values()),
    }, ensure_ascii=False), flush=True)

    summaries = []
    with Path(args.run_summary_csv).open(newline='') as f:
        for row in csv.DictReader(f):
            summaries.append(render_model(row, base_groups, args, cwd))

    global_summary = (
        Path(args.run_summary_csv).with_name(
            Path(args.run_summary_csv).stem + '_merged_original_summary.json'))
    global_summary.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + os.linesep)
    print(f'wrote {global_summary}', flush=True)


if __name__ == '__main__':
    main()
