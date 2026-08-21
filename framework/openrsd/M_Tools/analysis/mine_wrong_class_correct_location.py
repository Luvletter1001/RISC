#!/usr/bin/env python
"""Mine detections that localize a GT object but predict the wrong class."""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

from mmcv.ops import box_iou_rotated  # noqa: E402
from mmrotate.structures.bbox import qbox2rbox  # noqa: E402


DEFAULT_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport',
    'helipad')
CLASS_NAMES = DEFAULT_CLASSES
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--out-jsonl', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--summary-json', required=True)
    parser.add_argument(
        '--config',
        help='Optional model config. Uses its dataset metainfo class order.')
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--iou-thr', type=float, default=0.7)
    parser.add_argument('--diff-thr', type=int, default=100)
    parser.add_argument('--max-dets-per-img', type=int, default=0)
    return parser.parse_args()


def field(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def optional_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def tensor_to_numpy(value):
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def label_name(label):
    label = int(label)
    if 0 <= label < len(CLASS_NAMES):
        return CLASS_NAMES[label]
    return f'label_{label}'


def set_class_names(class_names):
    global CLASS_NAMES, CLASS_TO_LABEL
    CLASS_NAMES = tuple(class_names)
    CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_class_names(config_path):
    if not config_path:
        return DEFAULT_CLASSES
    from mmengine.config import Config
    cfg = Config.fromfile(config_path)
    dataset_cfg = cfg.get('test_dataloader', {}).get('dataset', {})
    metainfo = dataset_cfg.get('metainfo') or cfg.get('metainfo') or {}
    classes = metainfo.get('classes') or cfg.get('class_name')
    if not classes:
        return DEFAULT_CLASSES
    return tuple(classes)


def canonical_img_id(img_id):
    img_id = str(img_id)
    if img_id.startswith('angle_') and '__' in img_id:
        return img_id.split('__', 1)[1]
    return img_id


def qboxes_to_rboxes(qboxes):
    if len(qboxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    qbox_tensor = torch.from_numpy(np.asarray(qboxes, dtype=np.float32))
    return qbox2rbox(qbox_tensor).numpy().astype(np.float32)


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


def read_gt(ann_file, diff_thr):
    rows = []
    if not ann_file.exists():
        return rows
    for gt_index, raw in enumerate(ann_file.read_text().splitlines()):
        parts = raw.split()
        if len(parts) < 9:
            continue
        class_name = parts[8]
        if class_name not in CLASS_TO_LABEL:
            continue
        difficulty = int(parts[9]) if len(parts) > 9 else 0
        if difficulty > diff_thr:
            continue
        qbox = np.asarray([float(x) for x in parts[:8]], dtype=np.float32)
        rbox = qboxes_to_rboxes(qbox.reshape(1, 8))[0]
        rows.append({
            'gt_index': gt_index,
            'label': CLASS_TO_LABEL[class_name],
            'class_name': class_name,
            'difficulty': difficulty,
            'qbox': qbox,
            'rbox': rbox,
            'raw_line': raw,
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


def record_for_detection(img_id, ann_file, det_idx, pred_box, pred_label,
                         pred_score, gt, iou):
    return {
        'img_id': img_id,
        'ann_file': str(ann_file),
        'det_index': int(det_idx),
        'pred_label': int(pred_label),
        'pred_class': label_name(pred_label),
        'pred_score': float(pred_score),
        'pred_bbox_rbox': [float(x) for x in pred_box],
        'gt_index': int(gt['gt_index']),
        'gt_label': int(gt['label']),
        'gt_class': gt['class_name'],
        'gt_difficulty': int(gt['difficulty']),
        'gt_bbox_qbox': [float(x) for x in gt['qbox']],
        'gt_bbox_rbox': [float(x) for x in gt['rbox']],
        'iou': float(iou),
    }


def main():
    args = parse_args()
    set_class_names(load_class_names(args.config))
    pred_path = Path(args.predictions)
    ann_dir = Path(args.ann_dir)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    summary_json = Path(args.summary_json)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with pred_path.open('rb') as f:
        predictions = pickle.load(f)

    selected = []
    stats = Counter()
    wrong_by_pair = Counter()
    wrong_by_pred = Counter()
    wrong_by_gt = Counter()

    for sample in predictions:
        stats['images_scanned'] += 1
        img_id = get_img_id(sample)
        ann_file = ann_dir / f'{img_id}.txt'
        if not ann_file.exists():
            ann_file = ann_dir / f'{canonical_img_id(img_id)}.txt'
        gt_rows = read_gt(ann_file, args.diff_thr)
        if not ann_file.exists():
            stats['images_missing_ann'] += 1

        pred_boxes, pred_labels, pred_scores = get_pred_arrays(
            sample, args.score_thr, args.max_dets_per_img)
        stats['detections_checked'] += len(pred_scores)
        stats['gt_objects_checked'] += len(gt_rows)
        if len(pred_scores) == 0 or len(gt_rows) == 0:
            continue

        gt_boxes = np.asarray([row['rbox'] for row in gt_rows],
                              dtype=np.float32)
        ious = box_iou_rotated(
            torch.from_numpy(pred_boxes.astype(np.float32)),
            torch.from_numpy(gt_boxes)).cpu().numpy()
        best_gt = ious.argmax(axis=1)
        best_iou = ious[np.arange(ious.shape[0]), best_gt]
        for det_idx, gt_idx in enumerate(best_gt):
            gt = gt_rows[int(gt_idx)]
            if best_iou[det_idx] <= args.iou_thr:
                continue
            if int(pred_labels[det_idx]) == int(gt['label']):
                stats['localized_correct_class'] += 1
                continue
            record = record_for_detection(
                img_id=img_id,
                ann_file=ann_file,
                det_idx=det_idx,
                pred_box=pred_boxes[det_idx],
                pred_label=pred_labels[det_idx],
                pred_score=pred_scores[det_idx],
                gt=gt,
                iou=best_iou[det_idx])
            selected.append(record)
            stats['localized_wrong_class'] += 1
            wrong_by_pair[(record['pred_class'], record['gt_class'])] += 1
            wrong_by_pred[record['pred_class']] += 1
            wrong_by_gt[record['gt_class']] += 1

    with out_jsonl.open('w') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + os.linesep)

    fieldnames = [
        'img_id', 'det_index', 'pred_label', 'pred_class', 'pred_score',
        'gt_index', 'gt_label', 'gt_class', 'gt_difficulty', 'iou',
        'pred_bbox_rbox', 'gt_bbox_qbox', 'gt_bbox_rbox', 'ann_file'
    ]
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            csv_row = dict(row)
            csv_row['pred_bbox_rbox'] = json.dumps(row['pred_bbox_rbox'])
            csv_row['gt_bbox_qbox'] = json.dumps(row['gt_bbox_qbox'])
            csv_row['gt_bbox_rbox'] = json.dumps(row['gt_bbox_rbox'])
            writer.writerow(csv_row)

    summary = {
        'predictions': str(pred_path),
        'ann_dir': str(ann_dir),
        'score_thr': args.score_thr,
        'iou_rule': f'max IoU > {args.iou_thr} with any valid GT and class differs',
        'diff_thr': args.diff_thr,
        'class_names': list(CLASS_NAMES),
        'stats': dict(stats),
        'selected_count': len(selected),
        'wrong_by_pred_class': dict(wrong_by_pred),
        'wrong_by_gt_class': dict(wrong_by_gt),
        'top_wrong_pairs': [{
            'pred_class': pred,
            'gt_class': gt,
            'count': count
        } for (pred, gt), count in wrong_by_pair.most_common(50)],
        'out_jsonl': str(out_jsonl),
        'out_csv': str(out_csv),
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
