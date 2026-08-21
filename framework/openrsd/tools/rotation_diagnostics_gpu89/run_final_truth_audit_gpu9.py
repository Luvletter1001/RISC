#!/usr/bin/env python
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from commonlibs.common_tools import pklload


SMALL = 'small-vehicle'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-path', required=True)
    p.add_argument('--ann-path', required=True)
    p.add_argument('--detections-json', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--max-crops', type=int, default=300)
    return p.parse_args()


def poly_to_xyxy(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]


def rbox_to_xyxy(box):
    x, y, w, h = map(float, box[:4])
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (aa + bb - inter + 1e-12))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    crop_dir = out_dir / 'rot000_small_vehicle_crops'
    crop_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(args.image_path)
    ann = pklload(args.ann_path)
    gt_rows = []
    gt_boxes = []
    for idx, (text, poly) in enumerate(zip(ann['texts'], ann['polys'])):
        box = poly_to_xyxy(poly)
        gt_boxes.append((text, box))
        gt_rows.append(dict(gt_id=idx, class_name=text, xyxy=json.dumps(box)))
    with open(args.detections_json) as f:
        det = json.load(f)['detections']
    rows = []
    small_indices = [i for i, n in enumerate(det['class_names']) if n == SMALL]
    for rank, i in enumerate(small_indices[:args.max_crops]):
        box = rbox_to_xyxy(det['bboxes'][i])
        score = float(det['scores'][i])
        best_cls = ''
        best_iou = 0.0
        for cls, gt_box in gt_boxes:
            v = iou(box, gt_box)
            if v > best_iou:
                best_iou = v
                best_cls = cls
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = image.shape[:2]
        pad = 16
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop_path = crop_dir / f'sv_{rank:04d}_score_{score:.3f}_iou_{best_iou:.2f}.jpg'
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size:
            cv2.imwrite(str(crop_path), crop)
        rows.append(dict(
            det_rank=rank,
            score=score,
            bbox_xyxy=json.dumps(box),
            crop_path=str(crop_path),
            best_gt_class=best_cls,
            best_gt_iou=best_iou,
            auto_truth_label=('true_vehicle' if best_cls == SMALL and best_iou >= 0.1
                              else 'needs_human_review'),
            human_label='',
            false_positive_type='',
            notes=''))
    gt_csv = out_dir / 'ftable_rot000_gt_objects.csv'
    audit_csv = out_dir / 'ftable_rot000_small_vehicle_detection_audit_template.csv'
    write_csv(gt_csv, gt_rows, ['gt_id', 'class_name', 'xyxy'])
    write_csv(audit_csv, rows, list(rows[0].keys()))
    true_proxy = sum(1 for r in rows if r['auto_truth_label'] == 'true_vehicle')
    report = report_dir / 'fres_detection_truth_audit.md'
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    with open(report, 'w') as f:
        f.write('# P0148 Detection Truth Audit\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{cuda_visible}`\n')
        f.write(f'- image: `{args.image_path}`\n')
        f.write(f'- GT ann: `{args.ann_path}`\n')
        f.write(f'- detection json: `{args.detections_json}`\n')
        f.write(f'- GT table: `{gt_csv}`\n')
        f.write(f'- audit template: `{audit_csv}`\n')
        f.write(f'- crop dir: `{crop_dir}`\n\n')
        f.write('## Auto Proxy Summary\n\n')
        f.write(f'- sampled/exported small-vehicle detections: `{len(rows)}`\n')
        f.write(f'- matched to GT small-vehicle at IoU>=0.1: `{true_proxy}`\n')
        f.write(f'- proxy true vehicle ratio: `{(true_proxy / len(rows)) if rows else 0:.6f}`\n\n')
        f.write('Human labels should use one of: true_vehicle, road_texture, roof_edge, shadow, sports_line, other.\n')
    with open(out_dir / 'gpu9_truth_audit_summary.json', 'w') as f:
        json.dump(dict(report=str(report), audit_csv=str(audit_csv),
                       gt_csv=str(gt_csv), crop_dir=str(crop_dir),
                       exported=len(rows), proxy_true_vehicle_ratio=(true_proxy / len(rows)) if rows else 0), f, indent=2)
    print(report)


if __name__ == '__main__':
    main()
