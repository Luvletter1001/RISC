#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch

from openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

from mmcv.ops import box_iou_rotated
from mmrotate.structures.bbox import qbox2rbox


FAIR1M_CLASSES = [
    'a220', 'a321', 'a330', 'a350', 'arj21', 'baseball_field',
    'basketball_court', 'boeing737', 'boeing747', 'boeing777', 'boeing787',
    'bridge', 'bus', 'c919', 'cargo_truck', 'dry_cargo_ship', 'dump_truck',
    'engineering_ship', 'excavator', 'fishing_boat', 'football_field',
    'intersection', 'liquid_cargo_ship', 'motorboat', 'other-airplane',
    'other-ship', 'other-vehicle', 'passenger_ship', 'roundabout',
    'small_car', 'tennis_court', 'tractor', 'trailer', 'truck_tractor',
    'tugboat', 'van', 'warship'
]

COARSE_CLASSES = ['airplane', 'ship', 'vehicle', 'court', 'road_bridge']

COARSE_BY_FINE = {
    **{name: 'airplane' for name in [
        'a220', 'a321', 'a330', 'a350', 'arj21', 'boeing737', 'boeing747',
        'boeing777', 'boeing787', 'c919', 'other-airplane'
    ]},
    **{name: 'ship' for name in [
        'dry_cargo_ship', 'engineering_ship', 'fishing_boat',
        'liquid_cargo_ship', 'motorboat', 'other-ship', 'passenger_ship',
        'tugboat', 'warship'
    ]},
    **{name: 'vehicle' for name in [
        'bus', 'cargo_truck', 'dump_truck', 'excavator', 'other-vehicle',
        'small_car', 'tractor', 'trailer', 'truck_tractor', 'van'
    ]},
    **{name: 'court' for name in [
        'baseball_field', 'basketball_court', 'football_field',
        'tennis_court'
    ]},
    **{name: 'road_bridge' for name in ['bridge', 'intersection', 'roundabout']},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--max-dets-per-img', type=int, default=2000)
    parser.add_argument('--max-images', type=int, default=0)
    return parser.parse_args()


def as_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def read_predictions(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_gt(ann_file):
    boxes, labels = [], []
    if not ann_file.exists():
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    fine_to_coarse = {
        i: COARSE_CLASSES.index(COARSE_BY_FINE[name])
        for i, name in enumerate(FAIR1M_CLASSES)
    }
    fine_name_to_idx = {name: i for i, name in enumerate(FAIR1M_CLASSES)}
    for line in ann_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        fine_idx = fine_name_to_idx.get(parts[8])
        if fine_idx is None or int(parts[9]) > 100:
            continue
        qbox = torch.tensor([[float(x) for x in parts[:8]]], dtype=torch.float32)
        boxes.append(qbox2rbox(qbox).numpy()[0])
        labels.append(fine_to_coarse[fine_idx])
    if not boxes:
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    return np.asarray(boxes, np.float32), np.asarray(labels, np.int64)


def image_id(pred):
    img_id = str(pred.get('img_id', ''))
    if img_id:
        return img_id
    metainfo = pred.get('metainfo', {})
    if 'img_path' in metainfo:
        return Path(metainfo['img_path']).stem
    return ''


def remap_pred_instances(pred):
    fine_to_coarse = np.array([
        COARSE_CLASSES.index(COARSE_BY_FINE[name]) for name in FAIR1M_CLASSES
    ], dtype=np.int64)
    inst = pred['pred_instances']
    boxes = as_numpy(inst['bboxes']).astype(np.float32)
    labels = as_numpy(inst['labels']).astype(np.int64)
    scores = as_numpy(inst['scores']).astype(np.float32)
    keep = (labels >= 0) & (labels < len(fine_to_coarse))
    return boxes[keep], fine_to_coarse[labels[keep]], scores[keep]


def voc_ap(recalls, precisions):
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        vals = precisions[recalls >= thr]
        ap += vals.max() if vals.size else 0.0
    return ap / 11.0


def eval_one_class(preds, gts, cls, iou_thr):
    total_gt = sum(int((gt_labels == cls).sum()) for _, gt_labels in gts.values())
    dets = []
    matched = {img_id: np.zeros((gt_labels == cls).sum(), dtype=bool)
               for img_id, (_, gt_labels) in gts.items()}
    gt_boxes_by_img = {
        img_id: gt_boxes[gt_labels == cls] for img_id, (gt_boxes, gt_labels) in gts.items()
    }
    for img_id, boxes, labels, scores in preds:
        keep = labels == cls
        for box, score in zip(boxes[keep], scores[keep]):
            dets.append((float(score), img_id, box))
    dets.sort(reverse=True, key=lambda x: x[0])
    if total_gt == 0:
        return None
    tp = np.zeros(len(dets), dtype=np.float32)
    fp = np.zeros(len(dets), dtype=np.float32)
    for idx, (_, img_id, box) in enumerate(dets):
        gt_boxes = gt_boxes_by_img.get(img_id, np.zeros((0, 5), np.float32))
        if len(gt_boxes) == 0:
            fp[idx] = 1
            continue
        ious = box_iou_rotated(
            torch.from_numpy(box[None].astype(np.float32)),
            torch.from_numpy(gt_boxes.astype(np.float32))).cpu().numpy()[0]
        best = int(ious.argmax())
        if ious[best] >= iou_thr and not matched[img_id][best]:
            tp[idx] = 1
            matched[img_id][best] = True
        else:
            fp[idx] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return voc_ap(recalls, precisions)


def main():
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    predictions = read_predictions(args.predictions)
    if args.max_images > 0:
        predictions = predictions[:args.max_images]
    preds, gts = [], {}
    for pred in predictions:
        img_id = image_id(pred)
        if not img_id:
            continue
        ann_file = ann_dir / f'{img_id}.txt'
        gts[img_id] = read_gt(ann_file)
        boxes, labels, scores = remap_pred_instances(pred)
        keep = scores >= args.score_thr
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        if args.max_dets_per_img > 0 and len(scores) > args.max_dets_per_img:
            order = np.argsort(-scores)[:args.max_dets_per_img]
            boxes, labels, scores = boxes[order], labels[order], scores[order]
        preds.append((img_id, boxes, labels, scores))
    rows = []
    aps = []
    for cls, name in enumerate(COARSE_CLASSES):
        ap = eval_one_class(preds, gts, cls, args.iou_thr)
        rows.append({'split': args.split, 'granularity': 'FAIR1M-5',
                     'class': name, 'AP50': 'NA' if ap is None else f'{ap:.6f}'})
        if ap is not None:
            aps.append(ap)
    rows.append({'split': args.split, 'granularity': 'FAIR1M-5',
                 'class': 'mAP', 'AP50': f'{np.mean(aps):.6f}' if aps else 'NA'})
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'granularity', 'class', 'AP50'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
