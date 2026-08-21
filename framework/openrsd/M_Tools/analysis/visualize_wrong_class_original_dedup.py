#!/usr/bin/env python
"""Render original-scale wrong-class visualizations with per-GT dedup."""

import argparse
import csv
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from merge_tile_visualizations_to_original import build_canvas, index_base_tiles, parse_tile_stem
from visualize_wrong_class_correct_location import (
    draw_poly,
    get_img_id,
    get_pred_arrays,
    is_ignored_confusion,
    label_name,
    load_class_names,
    qboxes_to_rboxes,
    read_gt,
    rboxes_to_qboxes,
)

from mmcv.ops import box_iou_rotated  # noqa: E402


RED = (0, 0, 255)
GRAY = (150, 150, 150)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-summary-csv', required=True)
    parser.add_argument('--tile-img-dir', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--output-subdir',
                        default='vis_wrong_class_iou_gt0p7_merged_original_dedup')
    parser.add_argument('--original-img-dir', default='')
    parser.add_argument('--iou-thr', type=float, default=0.7)
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--diff-thr', type=int, default=100)
    parser.add_argument('--max-dets-per-img', type=int, default=0)
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--keep-by', choices=('iou_score', 'score_iou'),
                        default='iou_score')
    parser.add_argument('--only-originals', default='')
    parser.add_argument('--max-originals-per-model', type=int, default=0)
    return parser.parse_args()


def translate_qbox(qbox, x, y):
    qbox = np.asarray(qbox, dtype=np.float32).reshape(4, 2).copy()
    qbox[:, 0] += float(x)
    qbox[:, 1] += float(y)
    return qbox.reshape(8)


def gt_identity(gt_class, gt_qbox_orig):
    coords = tuple(int(round(v)) for v in np.asarray(gt_qbox_orig).reshape(-1))
    return (gt_class, coords)


def better_record(new, old, keep_by):
    if old is None:
        return True
    if keep_by == 'score_iou':
        new_key = (new['pred_score'], new['iou'])
        old_key = (old['pred_score'], old['iou'])
    else:
        new_key = (new['iou'], new['pred_score'])
        old_key = (old['iou'], old['pred_score'])
    return new_key > old_key


def localized_original_records(sample, ann_dir, class_names, class_to_label,
                               args):
    img_id = get_img_id(sample)
    tile = parse_tile_stem(img_id)
    ann_file = Path(ann_dir) / f'{img_id}.txt'
    gt_rows = read_gt(ann_file, class_to_label, args.diff_thr)
    pred_boxes, pred_labels, pred_scores = get_pred_arrays(
        sample, args.score_thr, args.max_dets_per_img)
    if len(pred_scores) == 0 or len(gt_rows) == 0:
        return tile['orig_id'], []

    gt_boxes = np.asarray([row['rbox'] for row in gt_rows], dtype=np.float32)
    ious = box_iou_rotated(
        torch.from_numpy(pred_boxes.astype(np.float32)),
        torch.from_numpy(gt_boxes)).cpu().numpy()
    best_gt = ious.argmax(axis=1)
    best_iou = ious[np.arange(ious.shape[0]), best_gt]
    pred_qboxes = rboxes_to_qboxes(pred_boxes)

    records = []
    for det_idx, gt_idx in enumerate(best_gt):
        iou = float(best_iou[det_idx])
        if iou <= args.iou_thr:
            continue
        gt = gt_rows[int(gt_idx)]
        pred_label = int(pred_labels[det_idx])
        pred_class = label_name(class_names, pred_label)
        gt_class = gt['class_name']
        correct = pred_label == int(gt['label'])
        ignored = (not correct) and is_ignored_confusion(pred_class, gt_class)
        gt_qbox_orig = translate_qbox(gt['qbox'], tile['x'], tile['y'])
        pred_qbox_orig = translate_qbox(pred_qboxes[det_idx], tile['x'], tile['y'])
        records.append({
            'orig_id': tile['orig_id'],
            'tile_img_id': img_id,
            'tile_x': tile['x'],
            'tile_y': tile['y'],
            'det_index': int(det_idx),
            'pred_label': pred_label,
            'pred_class': pred_class,
            'pred_score': float(pred_scores[det_idx]),
            'pred_qbox': pred_qbox_orig,
            'gt_label': int(gt['label']),
            'gt_class': gt_class,
            'gt_qbox': gt_qbox_orig,
            'gt_key': gt_identity(gt_class, gt_qbox_orig),
            'iou': iou,
            'is_wrong': (not correct) and (not ignored),
            'is_correct': correct,
            'is_ignored': ignored,
        })
    return tile['orig_id'], records


def collect_model_records(row, ann_dir, class_names, class_to_label, args):
    predictions_path = Path(row['output_dir']) / 'predictions.pkl'
    with predictions_path.open('rb') as f:
        predictions = pickle.load(f)

    only = {item for item in args.only_originals.split(',') if item}
    by_orig = defaultdict(list)
    stats = Counter()
    for sample in predictions:
        stats['tiles_scanned'] += 1
        img_id = get_img_id(sample)
        tile = parse_tile_stem(img_id)
        if only and tile['orig_id'] not in only:
            continue
        orig_id, records = localized_original_records(
            sample, ann_dir, class_names, class_to_label, args)
        by_orig[orig_id].extend(records)
        stats['localized_records_raw'] += len(records)
    return by_orig, stats


def select_records_for_original(records, keep_by):
    wrong_by_gt = {}
    correct_by_gt = {}
    ignored = 0
    for rec in records:
        if rec['is_ignored']:
            ignored += 1
            continue
        if rec['is_wrong']:
            if better_record(rec, wrong_by_gt.get(rec['gt_key']), keep_by):
                wrong_by_gt[rec['gt_key']] = rec
            continue
        if rec['is_correct']:
            if better_record(rec, correct_by_gt.get(rec['gt_key']), keep_by):
                correct_by_gt[rec['gt_key']] = rec

    selected_wrong = list(wrong_by_gt.values())
    selected_wrong_keys = set(wrong_by_gt)
    selected_correct = [
        rec for key, rec in correct_by_gt.items()
        if key not in selected_wrong_keys
    ]
    return selected_wrong, selected_correct, ignored


def draw_records(canvas, selected_wrong, selected_correct):
    for rec in selected_correct:
        label = f'{rec["pred_class"]} {rec["pred_score"]:.2f}/{rec["iou"]:.2f}'
        draw_poly(canvas, rec['pred_qbox'], GRAY, label=label, thickness=1)
    for rec in selected_wrong:
        label = (
            f'{rec["pred_class"]}->{rec["gt_class"]} '
            f'{rec["pred_score"]:.2f}/{rec["iou"]:.2f}')
        draw_poly(canvas, rec['pred_qbox'], RED, label=label, thickness=2)


def write_box_manifest(path, model, orig_id, selected_wrong, selected_correct):
    rows = []
    for kind, records in (('wrong', selected_wrong), ('correct', selected_correct)):
        for rec in records:
            rows.append({
                'model': model,
                'orig_id': rec.get('orig_id', orig_id),
                'kind': kind,
                'tile_img_id': rec['tile_img_id'],
                'pred_class': rec['pred_class'],
                'gt_class': rec['gt_class'],
                'score': f'{rec["pred_score"]:.6f}',
                'iou': f'{rec["iou"]:.6f}',
                'pred_qbox_original': ' '.join(f'{v:.2f}' for v in rec['pred_qbox']),
                'gt_qbox_original': ' '.join(f'{v:.2f}' for v in rec['gt_qbox']),
            })
    with path.open('w', newline='') as f:
        fieldnames = [
            'model', 'orig_id', 'kind', 'tile_img_id', 'pred_class',
            'gt_class', 'score', 'iou', 'pred_qbox_original',
            'gt_qbox_original'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_model(row, base_groups, ann_dir, args, cwd):
    model = row['model']
    output_dir = Path(row['output_dir'])
    if not output_dir.is_absolute():
        output_dir = cwd / output_dir
    out_dir = output_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'manifest.csv'
    box_manifest_path = out_dir / 'box_manifest.csv'
    summary_path = out_dir / 'summary.json'

    class_names = load_class_names(row['config'])
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    by_orig, stats = collect_model_records(
        row, ann_dir, class_names, class_to_label, args)

    manifest_rows = []
    all_wrong = []
    all_correct = []
    rendered = 0
    for orig_id in sorted(by_orig):
        if args.max_originals_per_model > 0 and rendered >= args.max_originals_per_model:
            break
        selected_wrong, selected_correct, ignored = select_records_for_original(
            by_orig[orig_id], args.keep_by)
        raw_wrong = sum(1 for rec in by_orig[orig_id] if rec['is_wrong'])
        raw_correct = sum(1 for rec in by_orig[orig_id] if rec['is_correct'])
        if not selected_wrong:
            stats['originals_skipped_no_wrong_after_dedup'] += 1
            continue

        canvas, source_path, source_kind = build_canvas(
            orig_id, base_groups.get(orig_id, []), args.original_img_dir)
        if canvas is None:
            stats['missing_base_originals'] += 1
            continue
        draw_records(canvas, selected_wrong, selected_correct)

        out_path = out_dir / f'{orig_id}.jpg'
        ok = cv2.imwrite(str(out_path), canvas,
                         [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok:
            stats['write_failures'] += 1
            continue

        rendered += 1
        stats['originals_rendered'] += 1
        stats['raw_wrong_records'] += raw_wrong
        stats['raw_correct_records'] += raw_correct
        stats['selected_wrong_boxes'] += len(selected_wrong)
        stats['selected_correct_boxes'] += len(selected_correct)
        stats['ignored_small_large_vehicle_confusions'] += ignored
        stats['dedup_removed_wrong_boxes'] += raw_wrong - len(selected_wrong)
        stats['dedup_removed_correct_boxes'] += raw_correct - len(selected_correct)
        all_wrong.extend((orig_id, rec) for rec in selected_wrong)
        all_correct.extend((orig_id, rec) for rec in selected_correct)
        manifest_rows.append({
            'model': model,
            'orig_id': orig_id,
            'merged_path': str(out_path.relative_to(cwd)),
            'source_kind': source_kind,
            'source_path': source_path,
            'canvas_width': canvas.shape[1],
            'canvas_height': canvas.shape[0],
            'raw_wrong_boxes': raw_wrong,
            'selected_wrong_boxes': len(selected_wrong),
            'raw_correct_boxes': raw_correct,
            'selected_correct_boxes': len(selected_correct),
            'ignored_boxes': ignored,
            'dedup_removed_wrong_boxes': raw_wrong - len(selected_wrong),
            'dedup_removed_correct_boxes': raw_correct - len(selected_correct),
        })

    with manifest_path.open('w', newline='') as f:
        fieldnames = [
            'model', 'orig_id', 'merged_path', 'source_kind', 'source_path',
            'canvas_width', 'canvas_height', 'raw_wrong_boxes',
            'selected_wrong_boxes', 'raw_correct_boxes',
            'selected_correct_boxes', 'ignored_boxes',
            'dedup_removed_wrong_boxes', 'dedup_removed_correct_boxes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_box_manifest(box_manifest_path, model, '*',
                       [rec for _, rec in all_wrong],
                       [rec for _, rec in all_correct])

    summary = {
        'model': model,
        'output_dir': str(out_dir.relative_to(cwd)),
        'manifest_csv': str(manifest_path.relative_to(cwd)),
        'box_manifest_csv': str(box_manifest_path.relative_to(cwd)),
        'dedup_rule': (
            'convert tile boxes to original coordinates; group by original GT '
            f'identity; keep one wrong-class detection per GT by {args.keep_by}; '
            'if a GT has a selected wrong box, its correct box is not drawn'),
        'iou_rule': f'max tile IoU > {args.iou_thr}',
        'stats': dict(stats),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    args = parse_args()
    cwd = Path.cwd()
    base_groups = index_base_tiles(Path(args.tile_img_dir))
    ann_dir = Path(args.ann_dir)
    print(json.dumps({
        'tile_img_dir': args.tile_img_dir,
        'ann_dir': args.ann_dir,
        'base_originals': len(base_groups),
        'base_tiles': sum(len(v) for v in base_groups.values()),
        'output_subdir': args.output_subdir,
        'keep_by': args.keep_by,
    }, ensure_ascii=False), flush=True)

    summaries = []
    with Path(args.run_summary_csv).open(newline='') as f:
        for row in csv.DictReader(f):
            summaries.append(render_model(row, base_groups, ann_dir, args, cwd))

    global_summary = (
        Path(args.run_summary_csv).with_name(
            Path(args.run_summary_csv).stem + '_merged_original_dedup_summary.json'))
    global_summary.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + os.linesep)
    print(f'wrote {global_summary}', flush=True)


if __name__ == '__main__':
    main()
