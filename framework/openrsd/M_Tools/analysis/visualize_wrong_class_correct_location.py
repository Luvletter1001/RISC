#!/usr/bin/env python
"""Visualize localized detections, coloring correct classes gray and wrong red."""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

from mmcv.ops import box_iou_rotated  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmrotate.structures.bbox import qbox2rbox, rbox2qbox  # noqa: E402


DEFAULT_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport',
    'helipad')
RED = (0, 0, 255)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
IGNORED_CONFUSION_PAIRS = {
    ('small-vehicle', 'large-vehicle'),
    ('large-vehicle', 'small-vehicle'),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-summary-csv', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--iou-thr', type=float, default=0.7)
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--diff-thr', type=int, default=100)
    parser.add_argument('--max-dets-per-img', type=int, default=0)
    parser.add_argument('--max-images-per-model', type=int, default=0)
    parser.add_argument('--output-subdir', default='vis_wrong_class_iou_gt0p7')
    parser.add_argument('--jpeg-quality', type=int, default=90)
    return parser.parse_args()


def tensor_to_numpy(value):
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def field(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def optional_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def canonical_img_id(img_id):
    img_id = str(img_id)
    if img_id.startswith('angle_') and '__' in img_id:
        return img_id.split('__', 1)[1]
    return img_id


def load_class_names(config_path):
    cfg = Config.fromfile(config_path)
    dataset_cfg = cfg.get('test_dataloader', {}).get('dataset', {})
    metainfo = dataset_cfg.get('metainfo') or cfg.get('metainfo') or {}
    classes = metainfo.get('classes') or cfg.get('class_name')
    return tuple(classes or DEFAULT_CLASSES)


def qboxes_to_rboxes(qboxes):
    if len(qboxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return qbox2rbox(torch.from_numpy(np.asarray(qboxes, dtype=np.float32))).numpy().astype(np.float32)


def boxes_to_rboxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    boxes = boxes.reshape(boxes.shape[0], -1)
    if boxes.shape[1] == 5:
        return boxes.astype(np.float32)
    if boxes.shape[1] == 8:
        return qboxes_to_rboxes(boxes)
    raise ValueError(f'Unsupported bbox shape: {boxes.shape}')


def rboxes_to_qboxes(rboxes):
    if len(rboxes) == 0:
        return np.zeros((0, 8), dtype=np.float32)
    return rbox2qbox(torch.from_numpy(np.asarray(rboxes, dtype=np.float32))).numpy().astype(np.float32)


def read_gt(ann_file, class_to_label, diff_thr):
    rows = []
    if not ann_file.exists():
        return rows
    for gt_index, raw in enumerate(ann_file.read_text().splitlines()):
        parts = raw.split()
        if len(parts) < 9:
            continue
        class_name = parts[8]
        if class_name not in class_to_label:
            continue
        difficulty = int(parts[9]) if len(parts) > 9 else 0
        if difficulty > diff_thr:
            continue
        qbox = np.asarray([float(x) for x in parts[:8]], dtype=np.float32)
        rows.append({
            'gt_index': gt_index,
            'label': class_to_label[class_name],
            'class_name': class_name,
            'qbox': qbox,
            'rbox': qboxes_to_rboxes(qbox.reshape(1, 8))[0],
        })
    return rows


def get_img_id(sample):
    img_id = optional_field(sample, 'img_id')
    if img_id is not None:
        return str(img_id)
    metainfo = optional_field(sample, 'metainfo', {}) or {}
    return str(metainfo.get('img_id'))


def get_pred_arrays(sample, score_thr, max_dets_per_img):
    pred_instances = field(sample, 'pred_instances')
    boxes = boxes_to_rboxes(tensor_to_numpy(field(pred_instances, 'bboxes')))
    labels = tensor_to_numpy(field(pred_instances, 'labels')).astype(np.int64)
    scores = tensor_to_numpy(field(pred_instances, 'scores')).astype(np.float32)
    keep = scores >= score_thr
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    if max_dets_per_img > 0 and len(scores) > max_dets_per_img:
        order = np.argsort(-scores)[:max_dets_per_img]
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]
    return boxes, labels, scores


def label_name(class_names, label):
    label = int(label)
    if 0 <= label < len(class_names):
        return class_names[label]
    return f'label_{label}'


def is_ignored_confusion(pred_class, gt_class):
    return (pred_class, gt_class) in IGNORED_CONFUSION_PAIRS


def draw_poly(img, qbox, color, label='', thickness=1):
    pts = np.asarray(qbox, dtype=np.float32).reshape(-1, 2)
    h, w = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.round(pts).astype(np.int32)
    cv2.polylines(img, [pts_i], True, color, thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    x = int(pts_i[:, 0].min())
    y = int(pts_i[:, 1].min())
    y = max(16, y - 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.34
    (tw, th), bl = cv2.getTextSize(label, font, scale, 1)
    bg_x2 = min(w - 1, x + tw + 6)
    bg_y1 = max(0, y - th - bl - 4)
    cv2.rectangle(img, (x, bg_y1), (bg_x2, y + bl), color, -1)
    cv2.putText(img, label, (x + 3, y - 3), font, scale, WHITE, 1, cv2.LINE_AA)


def image_path_for(img_dir, img_id):
    for suffix in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        path = img_dir / f'{img_id}{suffix}'
        if path.exists():
            return path
    return img_dir / f'{img_id}.png'


def localized_records(sample, ann_dir, class_names, class_to_label, args):
    img_id = get_img_id(sample)
    ann_file = ann_dir / f'{img_id}.txt'
    if not ann_file.exists():
        ann_file = ann_dir / f'{canonical_img_id(img_id)}.txt'
    gt_rows = read_gt(ann_file, class_to_label, args.diff_thr)
    pred_boxes, pred_labels, pred_scores = get_pred_arrays(
        sample, args.score_thr, args.max_dets_per_img)
    if len(pred_scores) == 0 or len(gt_rows) == 0:
        return img_id, []

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
        records.append({
            'det_index': int(det_idx),
            'pred_label': pred_label,
            'pred_class': pred_class,
            'pred_score': float(pred_scores[det_idx]),
            'pred_qbox': pred_qboxes[det_idx],
            'gt_index': int(gt['gt_index']),
            'gt_label': int(gt['label']),
            'gt_class': gt_class,
            'iou': iou,
            'is_wrong': (not correct) and (not ignored),
            'is_ignored': ignored,
        })
    return img_id, records


def render_model(row, img_dir, ann_dir, args):
    output_dir = Path(row['output_dir'])
    vis_dir = output_dir / args.output_subdir
    vis_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = vis_dir / 'manifest.csv'
    summary_path = vis_dir / 'summary.json'
    predictions_path = Path(row['output_dir']) / 'predictions.pkl'
    class_names = load_class_names(row['config'])
    class_to_label = {name: idx for idx, name in enumerate(class_names)}

    with predictions_path.open('rb') as f:
        predictions = pickle.load(f)

    manifest_rows = []
    stats = Counter()
    wrong_pair_counts = Counter()
    wrong_images_by_count = []
    for sample in predictions:
        stats['images_scanned'] += 1
        img_id, records = localized_records(sample, ann_dir, class_names,
                                            class_to_label, args)
        wrong_records = [rec for rec in records if rec['is_wrong']]
        correct_records = [
            rec for rec in records
            if (not rec['is_wrong']) and (not rec.get('is_ignored', False))
        ]
        ignored_records = [rec for rec in records if rec.get('is_ignored', False)]
        stats['localized_correct_boxes'] += len(correct_records)
        stats['localized_wrong_boxes'] += len(wrong_records)
        stats['ignored_small_large_vehicle_confusions'] += len(ignored_records)
        if not wrong_records:
            continue
        if args.max_images_per_model > 0 and stats['images_rendered'] >= args.max_images_per_model:
            continue

        img_path = image_path_for(img_dir, img_id)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            stats['missing_images'] += 1
            continue

        for rec in correct_records:
            label = f'{rec["pred_class"]} {rec["pred_score"]:.2f}/{rec["iou"]:.2f}'
            draw_poly(img, rec['pred_qbox'], GRAY, label=label, thickness=1)
        for rec in wrong_records:
            label = (
                f'{rec["pred_class"]}->{rec["gt_class"]} '
                f'{rec["pred_score"]:.2f}/{rec["iou"]:.2f}')
            draw_poly(img, rec['pred_qbox'], RED, label=label, thickness=2)
            wrong_pair_counts[(rec['pred_class'], rec['gt_class'])] += 1

        out_file = vis_dir / f'{img_id}.jpg'
        ok = cv2.imwrite(str(out_file), img,
                         [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok:
            stats['write_failures'] += 1
            continue
        stats['images_rendered'] += 1
        wrong_images_by_count.append((len(wrong_records), img_id, str(out_file)))
        manifest_rows.append({
            'model': row['model'],
            'img_id': img_id,
            'image_path': str(img_path),
            'vis_path': str(out_file),
            'wrong_boxes': len(wrong_records),
            'correct_boxes': len(correct_records),
            'ignored_boxes': len(ignored_records),
            'localized_boxes': len(records),
        })

    with manifest_path.open('w', newline='') as f:
        fieldnames = [
            'model', 'img_id', 'image_path', 'vis_path', 'wrong_boxes',
            'correct_boxes', 'ignored_boxes', 'localized_boxes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        'model': row['model'],
        'config': row['config'],
        'checkpoint': row['checkpoint'],
        'predictions': str(predictions_path),
        'img_dir': str(img_dir),
        'ann_dir': str(ann_dir),
        'vis_dir': str(vis_dir),
        'iou_rule': (
            f'max IoU > {args.iou_thr}; gray=correct class, red=wrong class; '
            'small-vehicle<->large-vehicle confusions are ignored and not drawn'
        ),
        'score_thr': args.score_thr,
        'diff_thr': args.diff_thr,
        'stats': dict(stats),
        'manifest_csv': str(manifest_path),
        'top_wrong_pairs': [{
            'pred_class': pred,
            'gt_class': gt,
            'count': count
        } for (pred, gt), count in wrong_pair_counts.most_common(50)],
        'top_images_by_wrong_boxes': [{
            'wrong_boxes': count,
            'img_id': img_id,
            'vis_path': path,
        } for count, img_id, path in sorted(wrong_images_by_count, reverse=True)[:50]],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    with Path(args.run_summary_csv).open(newline='') as f:
        rows = list(csv.DictReader(f))

    summaries = [render_model(row, img_dir, ann_dir, args) for row in rows]
    all_summary_path = Path(args.run_summary_csv).with_name(
        Path(args.run_summary_csv).stem + '_visualization_summary.json')
    all_summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + os.linesep)
    print(f'wrote {all_summary_path}')


if __name__ == '__main__':
    main()
