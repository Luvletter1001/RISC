#!/usr/bin/env python3
"""Evaluate DOTA-style AP50 by GT geometry bins."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]
AR_BINS = (
    ('AR 1-3', 1.0, 3.0),
    ('AR 3-6', 3.0, 6.0),
    ('AR 6-9', 6.0, 9.0),
    ('AR 9-12', 9.0, 12.0),
    ('AR 12-30', 12.0, 30.0),
    ('AR > 30', 30.0, float('inf')),
)


def setup_repo(repo_root: Path) -> None:
    for path in (repo_root, repo_root / 'tools'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from openrsd_env import preload_installed_mmengine
        preload_installed_mmengine()
    except Exception:
        pass


def load_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value, 'cpu'):
        return value.cpu().numpy()
    return np.asarray(value)


def normalize_img_id(img_id: Any) -> str:
    text = str(img_id)
    if text.startswith('angle_') and '__' in text:
        return text.split('__', 1)[1]
    return text


def polygon_to_rbox(coords: Sequence[float]) -> tuple[float, float, float, float, float]:
    xs = [float(coords[i]) for i in range(0, 8, 2)]
    ys = [float(coords[i]) for i in range(1, 8, 2)]
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    edge0 = (xs[1] - xs[0], ys[1] - ys[0])
    edge1 = (xs[2] - xs[1], ys[2] - ys[1])
    w = math.hypot(*edge0)
    h = math.hypot(*edge1)
    angle = math.atan2(edge0[1], edge0[0])
    if h > w:
        w, h = h, w
        angle += math.pi / 2.0
    while angle < -math.pi / 2.0:
        angle += math.pi
    while angle >= math.pi / 2.0:
        angle -= math.pi
    return cx, cy, max(w, 1e-6), max(h, 1e-6), angle


def ap_11point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    ap = 0.0
    for thr in np.arange(0.0, 1.01, 0.1):
        inds = np.where(recalls >= thr)[0]
        ap += (precisions[inds].max() if inds.size else 0.0) / 11.0
    return float(ap)


def area_thresholds(gt_by_img: dict[str, list[dict[str, Any]]]) -> tuple[float, float]:
    areas = [gt['area'] for items in gt_by_img.values() for gt in items]
    if len(areas) < 3:
        return 0.0, float('inf')
    q1, q2 = np.quantile(np.asarray(areas, dtype=np.float64), [1 / 3, 2 / 3])
    return float(q1), float(q2)


def load_gt(repo_root: Path, angle: str, classes: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    ann_dir = repo_root / 'data/DOTA1_1024_500/angle_sweep_val/realistic' / f'angle_{angle}' / 'annfiles'
    gt_by_img: dict[str, list[dict[str, Any]]] = {}
    for ann_file in sorted(ann_dir.glob('*.txt')):
        image_id = normalize_img_id(ann_file.stem)
        rows = []
        for line in ann_file.read_text(encoding='utf-8', errors='replace').splitlines():
            parts = line.strip().split()
            if len(parts) < 9 or parts[8] not in class_to_id:
                continue
            try:
                box = polygon_to_rbox([float(x) for x in parts[:8]])
            except ValueError:
                continue
            w, h = box[2], box[3]
            ar = max(w, h) / max(min(w, h), 1e-6)
            ar_bin = next(name for name, lo, hi in AR_BINS if ar >= lo and ar < hi)
            rows.append({
                'bbox': np.asarray(box, dtype=np.float32),
                'label': class_to_id[parts[8]],
                'class': parts[8],
                'ar': ar,
                'area': float(w * h),
                'ar_bin': ar_bin,
            })
        gt_by_img[image_id] = rows
    small_thr, large_thr = area_thresholds(gt_by_img)
    for rows in gt_by_img.values():
        for gt in rows:
            if gt['area'] < small_thr:
                gt['area_bin'] = 'area_small'
            elif gt['area'] < large_thr:
                gt['area_bin'] = 'area_medium'
            else:
                gt['area_bin'] = 'area_large'
    return gt_by_img


def collect_predictions(pkl_path: Path, classes: Sequence[str]) -> dict[int, list[dict[str, Any]]]:
    preds: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(classes))}
    for sample in load_pickle(pkl_path):
        image_id = normalize_img_id(sample.get('img_id'))
        inst = sample.get('pred_instances', {})
        labels = to_numpy(inst.get('labels', [])).astype(int)
        scores = to_numpy(inst.get('scores', [])).astype(float)
        bboxes = to_numpy(inst.get('bboxes', [])).astype(np.float32)
        for label, score, box in zip(labels, scores, bboxes):
            if 0 <= label < len(classes):
                preds[label].append({'image_id': image_id, 'score': float(score), 'bbox': box.astype(np.float32)})
    for label in preds:
        preds[label].sort(key=lambda item: item['score'], reverse=True)
    return preds


def eval_one_bin(gt_by_img: dict[str, list[dict[str, Any]]],
                 preds_by_class: dict[int, list[dict[str, Any]]],
                 classes: Sequence[str],
                 bin_field: str,
                 bin_name: str,
                 iou_thr: float = 0.5) -> dict[str, Any]:
    import torch
    from mmcv.ops import box_iou_rotated

    gt_subset: dict[int, dict[str, list[dict[str, Any]]]] = {idx: {} for idx in range(len(classes))}
    for image_id, gts in gt_by_img.items():
        for gt in gts:
            if gt.get(bin_field) != bin_name:
                continue
            gt_subset[gt['label']].setdefault(image_id, []).append({**gt, 'matched': False})
    total_gt = sum(len(items) for by_img in gt_subset.values() for items in by_img.values())
    rows = []
    all_scores = []
    all_tp = []
    all_fp = []
    pred_count = 0
    for label, preds in preds_by_class.items():
        by_img = gt_subset[label]
        class_scores = []
        class_tp = []
        class_fp = []
        for pred in preds:
            pred_count += 1
            class_scores.append(pred['score'])
            candidates = by_img.get(pred['image_id'], [])
            if not candidates:
                class_tp.append(0.0)
                class_fp.append(1.0)
                continue
            gt_boxes = np.stack([gt['bbox'] for gt in candidates]).astype(np.float32)
            ious = box_iou_rotated(
                torch.from_numpy(pred['bbox'][None].astype(np.float32)),
                torch.from_numpy(gt_boxes)).cpu().numpy()[0]
            best_idx = int(np.argmax(ious)) if ious.size else -1
            if best_idx >= 0 and float(ious[best_idx]) >= iou_thr and not candidates[best_idx]['matched']:
                candidates[best_idx]['matched'] = True
                class_tp.append(1.0)
                class_fp.append(0.0)
            else:
                class_tp.append(0.0)
                class_fp.append(1.0)
        num_gts = sum(len(items) for items in by_img.values())
        if class_scores:
            tp = np.cumsum(np.asarray(class_tp, dtype=np.float32))
            fp = np.cumsum(np.asarray(class_fp, dtype=np.float32))
            recalls = tp / max(num_gts, np.finfo(np.float32).eps)
            precisions = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
            ap = ap_11point(recalls, precisions) if num_gts else None
        else:
            tp = np.asarray([], dtype=np.float32)
            fp = np.asarray([], dtype=np.float32)
            ap = 0.0 if num_gts else None
        if ap is not None:
            rows.append({
                'class': classes[label],
                'ap50': ap,
                'gt_count': num_gts,
                'prediction_count': len(class_scores),
                'tp': float(tp[-1]) if tp.size else 0.0,
                'fp': float(fp[-1]) if fp.size else 0.0,
                'fn': max(num_gts - (float(tp[-1]) if tp.size else 0.0), 0.0),
                'mean_score': float(np.mean(class_scores)) if class_scores else 0.0,
            })
        all_scores.extend(class_scores)
        all_tp.extend(class_tp)
        all_fp.extend(class_fp)
    valid_aps = [row['ap50'] for row in rows if row['gt_count'] > 0]
    tp_total = float(sum(all_tp))
    fp_total = float(sum(all_fp))
    return {
        'bin_type': bin_field,
        'bin': bin_name,
        'ap50': float(np.mean(valid_aps)) if valid_aps else 0.0,
        'gt_count': total_gt,
        'prediction_count': pred_count,
        'tp': tp_total,
        'fp': fp_total,
        'fn': max(total_gt - tp_total, 0.0),
        'mean_score': float(np.mean(all_scores)) if all_scores else 0.0,
        'class_rows': rows,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    setup_repo(args.repo_root)
    classes = args.classes.split(',') if args.classes else DOTA_CLASSES
    gt_by_img = load_gt(args.repo_root, args.angle, classes)
    preds_by_class = collect_predictions(args.predictions, classes)
    small_thr, large_thr = area_thresholds(gt_by_img)
    bin_defs = [('ar_bin', name) for name, _, _ in AR_BINS] + [
        ('area_bin', 'area_small'), ('area_bin', 'area_medium'), ('area_bin', 'area_large')
    ]
    rows = [
        eval_one_bin(gt_by_img, preds_by_class, classes, field, name)
        for field, name in bin_defs
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open('w', newline='', encoding='utf-8') as f:
        fields = ['model', 'angle', 'bin_type', 'bin', 'ap50', 'gt_count', 'prediction_count', 'tp', 'fp', 'fn', 'mean_score']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({'model': args.model, 'angle': args.angle, **{k: row[k] for k in fields if k in row}})
    payload = {
        'status': 'DONE',
        'model': args.model,
        'angle': args.angle,
        'predictions': str(args.predictions),
        'area_thresholds': {'small_lt': small_thr, 'medium_lt': large_thr, 'large_gte': large_thr},
        'rows': rows,
        'csv': str(args.out_csv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--angle', required=True)
    parser.add_argument('--model', default='unknown')
    parser.add_argument('--classes', default='')
    parser.add_argument('--out-csv', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == '__main__':
    main()
