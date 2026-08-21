#!/usr/bin/env python3
import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

from mmcv.ops import box_iou_rotated
from mmrotate.structures.bbox import qbox2rbox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--classes', required=True,
                        help='Comma-separated class names in label order.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--near-iou-thr', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=500)
    parser.add_argument('--max-dets-per-img', type=int, default=300)
    parser.add_argument('--append', action='store_true')
    return parser.parse_args()


def as_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def image_id(pred):
    img_id = str(pred.get('img_id', ''))
    if img_id:
        return img_id
    metainfo = pred.get('metainfo', {})
    if 'img_path' in metainfo:
        return Path(metainfo['img_path']).stem
    return ''


def read_gt(ann_file, class_to_label):
    boxes, labels = [], []
    if not ann_file.exists():
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    for line in ann_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        label = class_to_label.get(parts[8])
        if label is None or int(parts[9]) > 100:
            continue
        qbox = torch.tensor([[float(x) for x in parts[:8]]], dtype=torch.float32)
        boxes.append(qbox2rbox(qbox).numpy()[0])
        labels.append(label)
    if not boxes:
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    return np.asarray(boxes, np.float32), np.asarray(labels, np.int64)


def summarize_image(pred, ann_dir, class_to_label, args, global_stats, class_stats):
    img_id = image_id(pred)
    if not img_id:
        return
    ann_file = ann_dir / f'{img_id}.txt'
    if not ann_file.exists() and '__' in img_id:
        ann_file = ann_dir / f'{img_id.split("__", 1)[1]}.txt'
    gt_boxes, gt_labels = read_gt(ann_file, class_to_label)
    inst = pred['pred_instances']
    pred_boxes = as_numpy(inst['bboxes']).astype(np.float32)
    pred_labels = as_numpy(inst['labels']).astype(np.int64)
    pred_scores = as_numpy(inst['scores']).astype(np.float32)
    keep = pred_scores >= args.score_thr
    pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]
    if args.max_dets_per_img > 0 and len(pred_scores) > args.max_dets_per_img:
        order = np.argsort(-pred_scores)[:args.max_dets_per_img]
        pred_boxes, pred_labels, pred_scores = pred_boxes[order], pred_labels[order], pred_scores[order]

    global_stats['images'] += 1
    global_stats['gts'] += len(gt_labels)
    global_stats['dets'] += len(pred_labels)
    for label in gt_labels:
        class_stats[int(label)]['gts'] += 1
    for label in pred_labels:
        if 0 <= int(label) < len(class_to_label):
            class_stats[int(label)]['dets'] += 1

    if len(gt_labels) == 0:
        global_stats['false_positive'] += len(pred_labels)
        return
    if len(pred_labels) == 0:
        global_stats['missed'] += len(gt_labels)
        for label in gt_labels:
            class_stats[int(label)]['missed'] += 1
        return

    ious = box_iou_rotated(torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)).cpu().numpy()
    candidates = []
    for d in range(ious.shape[0]):
        for g in range(ious.shape[1]):
            if ious[d, g] >= args.iou_thr and pred_labels[d] == gt_labels[g]:
                candidates.append((ious[d, g], d, g))
    candidates.sort(reverse=True)
    used_det, used_gt = set(), set()
    for _, d, g in candidates:
        if d in used_det or g in used_gt:
            continue
        used_det.add(d)
        used_gt.add(g)
        global_stats['correct'] += 1
        class_stats[int(gt_labels[g])]['correct'] += 1

    for g, gt_label in enumerate(gt_labels):
        if g in used_gt:
            continue
        best_d = int(ious[:, g].argmax())
        best_iou = float(ious[best_d, g])
        if best_iou >= args.iou_thr and pred_labels[best_d] != gt_label:
            global_stats['wrong_class_given_matched'] += 1
            class_stats[int(gt_label)]['wrong_class_given_matched'] += 1
        elif best_iou >= args.near_iou_thr:
            global_stats['localization_fail'] += 1
            class_stats[int(gt_label)]['localization_fail'] += 1
        else:
            global_stats['missed'] += 1
            class_stats[int(gt_label)]['missed'] += 1

    false_positive = len(pred_labels) - len(used_det)
    global_stats['false_positive'] += false_positive


def row(args, scope, stats):
    gts = stats.get('gts', 0)
    dets = stats.get('dets', 0)
    correct = stats.get('correct', 0)
    return {
        'dataset': args.dataset,
        'model': args.model,
        'split': args.split,
        'scope': scope,
        'images': stats.get('images', ''),
        'gts': gts,
        'dets': dets,
        'dets_per_image': f'{dets / stats["images"]:.6f}' if stats.get('images') else 'NA',
        'correct': correct,
        'recall_at_iou50': f'{correct / gts:.6f}' if gts else 'NA',
        'wrong_class_given_matched': stats.get('wrong_class_given_matched', 0),
        'localization_fail': stats.get('localization_fail', 0),
        'missed': stats.get('missed', 0),
        'false_positive': stats.get('false_positive', 0),
    }


def main():
    args = parse_args()
    classes = [c for c in args.classes.split(',') if c]
    class_to_label = {name: idx for idx, name in enumerate(classes)}
    with open(args.predictions, 'rb') as f:
        predictions = pickle.load(f)
    if args.max_images > 0:
        predictions = predictions[:args.max_images]
    global_stats = defaultdict(int)
    class_stats = defaultdict(lambda: defaultdict(int))
    ann_dir = Path(args.ann_dir)
    for pred in predictions:
        summarize_image(pred, ann_dir, class_to_label, args, global_stats, class_stats)
    fields = [
        'dataset', 'model', 'split', 'scope', 'images', 'gts', 'dets',
        'dets_per_image', 'correct', 'recall_at_iou50',
        'wrong_class_given_matched', 'localization_fail', 'missed',
        'false_positive'
    ]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if args.append and out.exists() else 'w'
    with out.open(mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w':
            writer.writeheader()
        writer.writerow(row(args, 'all', global_stats))
        for idx, name in enumerate(classes):
            if class_stats[idx]:
                writer.writerow(row(args, name, class_stats[idx]))


if __name__ == '__main__':
    main()
