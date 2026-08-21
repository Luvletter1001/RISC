#!/usr/bin/env python
"""Decompose rotation-induced detection failures from saved predictions."""

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

from mmcv.ops import box_iou_rotated

from mmrotate.structures.bbox import qbox2rbox


DOTA2_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport',
    'helipad')
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(DOTA2_CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--angle', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out-tsv', required=True)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--near-iou-thr', type=float, default=0.1)
    parser.add_argument('--max-dets-per-img', type=int, default=300)
    parser.add_argument('--append', action='store_true')
    return parser.parse_args()


def tensor_to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def canonical_img_id(img_id):
    if isinstance(img_id, (int, np.integer)):
        img_id = str(img_id)
    if img_id.startswith('angle_') and '__' in img_id:
        return img_id.split('__', 1)[1]
    return img_id


def read_gt(ann_file):
    boxes = []
    labels = []
    ignored = []
    if not ann_file.exists():
        return (np.zeros((0, 5), dtype=np.float32),
                np.zeros((0, ), dtype=np.int64), np.zeros((0, ), dtype=bool))
    for raw in ann_file.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 10:
            continue
        label = CLASS_TO_LABEL.get(parts[8])
        if label is None:
            continue
        qbox = torch.tensor([[float(x) for x in parts[:8]]],
                            dtype=torch.float32)
        rbox = qbox2rbox(qbox).numpy()[0]
        boxes.append(rbox)
        labels.append(label)
        ignored.append(int(parts[9]) > 100)
    if not boxes:
        return (np.zeros((0, 5), dtype=np.float32),
                np.zeros((0, ), dtype=np.int64), np.zeros((0, ), dtype=bool))
    return (np.asarray(boxes, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
            np.asarray(ignored, dtype=bool))


def greedy_true_positives(ious, pred_labels, gt_labels, iou_thr):
    candidates = []
    for det_idx in range(ious.shape[0]):
        for gt_idx in range(ious.shape[1]):
            if ious[det_idx, gt_idx] >= iou_thr and pred_labels[
                    det_idx] == gt_labels[gt_idx]:
                candidates.append((ious[det_idx, gt_idx], det_idx, gt_idx))
    candidates.sort(reverse=True)

    used_det = set()
    used_gt = set()
    matches = []
    for iou, det_idx, gt_idx in candidates:
        if det_idx in used_det or gt_idx in used_gt:
            continue
        used_det.add(det_idx)
        used_gt.add(gt_idx)
        matches.append((det_idx, gt_idx, float(iou)))
    return matches, used_det, used_gt


def update_class_stats(stats, class_name, values):
    for key, value in values.items():
        stats[class_name][key] += value


def analyze_sample(pred, ann_dir, score_thr, iou_thr, near_iou_thr,
                   max_dets_per_img, global_stats, class_stats):
    raw_img_id = str(pred['img_id'])
    ann_file = ann_dir / f'{raw_img_id}.txt'
    if not ann_file.exists():
        ann_file = ann_dir / f'{canonical_img_id(raw_img_id)}.txt'
    gt_boxes, gt_labels, ignored = read_gt(ann_file)
    valid = ~ignored
    gt_boxes = gt_boxes[valid]
    gt_labels = gt_labels[valid]

    pred_instances = pred['pred_instances']
    pred_boxes = tensor_to_numpy(pred_instances['bboxes']).astype(np.float32)
    pred_labels = tensor_to_numpy(pred_instances['labels']).astype(np.int64)
    pred_scores = tensor_to_numpy(pred_instances['scores']).astype(np.float32)
    keep = pred_scores >= score_thr
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    if max_dets_per_img > 0 and len(pred_scores) > max_dets_per_img:
        order = np.argsort(-pred_scores)[:max_dets_per_img]
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

    gts = int(len(gt_labels))
    dets = int(len(pred_labels))
    global_stats['gts'] += gts
    global_stats['dets'] += dets

    for label in gt_labels:
        update_class_stats(class_stats, DOTA2_CLASSES[int(label)], {'gts': 1})
    for label in pred_labels:
        update_class_stats(class_stats, DOTA2_CLASSES[int(label)], {'dets': 1})

    if gts == 0:
        global_stats['false_positive'] += dets
        for label in pred_labels:
            update_class_stats(class_stats, DOTA2_CLASSES[int(label)],
                               {'false_positive': 1})
        return
    if dets == 0:
        global_stats['missed'] += gts
        for label in gt_labels:
            update_class_stats(class_stats, DOTA2_CLASSES[int(label)],
                               {'missed': 1})
        return

    ious = box_iou_rotated(
        torch.from_numpy(pred_boxes),
        torch.from_numpy(gt_boxes)).cpu().numpy()
    matches, used_det, used_gt = greedy_true_positives(
        ious, pred_labels, gt_labels, iou_thr)
    global_stats['true_positive'] += len(matches)
    global_stats['tp_score_sum'] += sum(float(pred_scores[d]) for d, _, _ in matches)

    for det_idx, gt_idx, _ in matches:
        cls_name = DOTA2_CLASSES[int(gt_labels[gt_idx])]
        update_class_stats(
            class_stats, cls_name, {
                'true_positive': 1,
                'tp_score_sum': float(pred_scores[det_idx])
            })

    for gt_idx, gt_label in enumerate(gt_labels):
        if gt_idx in used_gt:
            continue
        cls_name = DOTA2_CLASSES[int(gt_label)]
        overlaps = ious[:, gt_idx]
        best_det = int(overlaps.argmax())
        best_iou = float(overlaps[best_det])
        if best_iou >= iou_thr and pred_labels[best_det] != gt_label:
            global_stats['class_wrong'] += 1
            update_class_stats(class_stats, cls_name, {'class_wrong': 1})
        elif best_iou >= near_iou_thr:
            global_stats['localization_fail'] += 1
            update_class_stats(class_stats, cls_name, {'localization_fail': 1})
        else:
            global_stats['missed'] += 1
            update_class_stats(class_stats, cls_name, {'missed': 1})

    false_positive = dets - len(used_det)
    global_stats['false_positive'] += false_positive
    for det_idx, label in enumerate(pred_labels):
        if det_idx not in used_det:
            update_class_stats(class_stats, DOTA2_CLASSES[int(label)],
                               {'false_positive': 1})


def summarize_row(model, angle, scope, stats):
    gts = stats.get('gts', 0)
    dets = stats.get('dets', 0)
    tp = stats.get('true_positive', 0)
    class_wrong = stats.get('class_wrong', 0)
    loc_fail = stats.get('localization_fail', 0)
    missed = stats.get('missed', 0)
    fp = stats.get('false_positive', 0)
    score_sum = stats.get('tp_score_sum', 0.0)
    return {
        'model': model,
        'angle': angle,
        'scope': scope,
        'gts': gts,
        'dets': dets,
        'true_positive': tp,
        'recall_at_iou50': f'{tp / gts:.6f}' if gts else 'NA',
        'mean_tp_score': f'{score_sum / tp:.6f}' if tp else 'NA',
        'class_wrong': class_wrong,
        'class_wrong_rate': f'{class_wrong / gts:.6f}' if gts else 'NA',
        'localization_fail': loc_fail,
        'localization_fail_rate': f'{loc_fail / gts:.6f}' if gts else 'NA',
        'missed': missed,
        'missed_rate': f'{missed / gts:.6f}' if gts else 'NA',
        'false_positive': fp,
        'fp_per_image': 'NA',
    }


def main():
    args = parse_args()
    with open(args.predictions, 'rb') as f:
        predictions = pickle.load(f)

    global_stats = defaultdict(float)
    class_stats = defaultdict(lambda: defaultdict(float))
    ann_dir = Path(args.ann_dir)
    for pred in predictions:
        analyze_sample(pred, ann_dir, args.score_thr, args.iou_thr,
                       args.near_iou_thr, args.max_dets_per_img,
                       global_stats, class_stats)
    global_row = summarize_row(args.model, args.angle, 'all', global_stats)
    global_row['fp_per_image'] = f'{global_stats["false_positive"] / len(predictions):.6f}'

    rows = [global_row]
    for class_name in DOTA2_CLASSES:
        if class_stats[class_name].get('gts', 0) == 0 and class_stats[
                class_name].get('dets', 0) == 0:
            continue
        row = summarize_row(args.model, args.angle, class_name,
                            class_stats[class_name])
        row['fp_per_image'] = f'{class_stats[class_name]["false_positive"] / len(predictions):.6f}'
        rows.append(row)

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not args.append or not out_path.exists()
    with out_path.open('a' if args.append else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(global_row, ensure_ascii=False))


if __name__ == '__main__':
    main()
