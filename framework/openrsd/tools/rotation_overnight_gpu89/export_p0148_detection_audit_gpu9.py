#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from commonlibs.common_tools import pklload
from tools.rotation_overnight_gpu89 import common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-path', default='vis/P0148__1024__651___0/dataset/images/P0148__1024__651___0_rot000.jpg')
    p.add_argument('--ann-path', default='vis/P0148__1024__651___0/dataset/annfiles/P0148__1024__651___0_rot000.pkl')
    p.add_argument('--detections-json', default='work_dirs/rotation_stage_probe_P0148_full_physgpu8_9/detections/P0148__1024__651___0_rot000.json')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--max-crops', type=int, default=300)
    return p.parse_args()


def rbox_to_xyxy(box):
    x, y, w, h = map(float, box[:4])
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def poly_to_xyxy(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (aa + bb - inter + 1e-12))


def likely_vehicle_heuristic(box, score, best_cls, best_iou):
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if best_cls == 'small-vehicle' and best_iou >= 0.1:
        return 'gt_small_vehicle_overlap'
    if score >= 0.45 and 20 <= area <= 1600:
        return 'high_score_vehicle_sized'
    return 'unknown'


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    crop_dir = out_dir / 'p0148_rot000_small_vehicle_crops'
    overlay_dir = out_dir / 'p0148_rot000_overlays'
    crop_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(args.image_path)
    gt_rows, gt_boxes = [], []
    gt_found = Path(args.ann_path).exists()
    if gt_found:
        ann = pklload(args.ann_path)
        for idx, (text, poly) in enumerate(zip(ann['texts'], ann['polys'])):
            box = poly_to_xyxy(poly)
            gt_boxes.append((text, box))
            gt_rows.append(dict(gt_id=idx, class_name=text, xyxy=json.dumps(box)))
    with open(args.detections_json) as f:
        det = json.load(f)['detections']
    overlay = image.copy()
    rows = []
    small_indices = [i for i, n in enumerate(det['class_names']) if n == 'small-vehicle']
    for det_id, src_idx in enumerate(small_indices[:args.max_crops]):
        box = rbox_to_xyxy(det['bboxes'][src_idx])
        score = float(det['scores'][src_idx])
        best_cls, best_iou = '', 0.0
        for cls_name, gt_box in gt_boxes:
            v = iou(box, gt_box)
            if v > best_iou:
                best_cls, best_iou = cls_name, v
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = image.shape[:2]
        x1c, y1c = max(0, x1 - 16), max(0, y1 - 16)
        x2c, y2c = min(w, x2 + 16), min(h, y2 + 16)
        crop_path = crop_dir / f'sv_{det_id:04d}_score_{score:.3f}_iou_{best_iou:.2f}.jpg'
        crop = image[y1c:y2c, x1c:x2c]
        if crop.size:
            cv2.imwrite(str(crop_path), crop)
        cv2.rectangle(overlay, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), (255, 0, 255), 1)
        cv2.putText(overlay, str(det_id), (max(0, x1), max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        bw, bh = max(0, box[2] - box[0]), max(0, box[3] - box[1])
        rows.append(dict(
            det_id=det_id,
            source_detection_index=src_idx,
            score=score,
            bbox=json.dumps(box),
            box_area=float(bw * bh),
            aspect_ratio=float(bw / (bh + 1e-12)),
            crop_path=str(crop_path),
            likely_true_vehicle_auto=likely_vehicle_heuristic(box, score, best_cls, best_iou),
            best_gt_class=best_cls,
            best_gt_iou=best_iou,
            human_label_true_vehicle='',
            false_positive_type='',
            note=''))
    overlay_path = overlay_dir / 'p0148_rot000_small_vehicle_numbered_overlay.jpg'
    cv2.imwrite(str(overlay_path), overlay)
    C.write_csv(out_dir / 'ftable_p0148_rot000_gt_objects.csv',
                gt_rows, ['gt_id', 'class_name', 'xyxy'])
    fields = ['det_id', 'source_detection_index', 'score', 'bbox', 'box_area',
              'aspect_ratio', 'crop_path', 'likely_true_vehicle_auto',
              'best_gt_class', 'best_gt_iou', 'human_label_true_vehicle',
              'false_positive_type', 'note']
    audit_csv = out_dir / 'ftable_p0148_rot000_manual_audit_template.csv'
    C.write_csv(audit_csv, rows, fields)
    scores = [r['score'] for r in rows]
    areas = [r['box_area'] for r in rows]
    proxy_true = [r for r in rows if r['best_gt_class'] == 'small-vehicle' and float(r['best_gt_iou']) >= 0.1]
    report = report_dir / 'fres_p0148_detection_audit.md'
    with open(report, 'w') as f:
        f.write('# P0148 Rot000 Small-Vehicle Detection Audit\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{C.cuda_visible()}`\n')
        f.write(f'- GT found: `{gt_found}`\n')
        f.write(f'- audit CSV: `{audit_csv}`\n')
        f.write(f'- crop dir: `{crop_dir}`\n')
        f.write(f'- numbered overlay: `{overlay_path}`\n\n')
        f.write('## Distribution\n\n')
        f.write(f'- exported detections: `{len(rows)}`\n')
        f.write(f'- score mean: `{np.mean(scores) if scores else 0:.6f}`; p95 `{np.quantile(scores, 0.95) if scores else 0:.6f}`\n')
        f.write(f'- box area mean: `{np.mean(areas) if areas else 0:.6f}`; p95 `{np.quantile(areas, 0.95) if areas else 0:.6f}`\n')
        f.write(f'- GT small-vehicle IoU>=0.1 proxy matches: `{len(proxy_true)}`\n\n')
        f.write('Manual false_positive_type enum: true_vehicle, road_texture, roof_edge, shadow, sports_field_line, tree_or_background, border_or_padding, unknown.\n')
        f.write('For a 95% CI of about +/-7%, manually label roughly 200 samples; all exported rows are available if stricter precision is needed.\n')
    with open(out_dir / 'gpu9_audit_summary.json', 'w') as f:
        json.dump(dict(report=str(report), audit_csv=str(audit_csv),
                       crop_dir=str(crop_dir), overlay=str(overlay_path),
                       exported=len(rows), gt_found=gt_found,
                       proxy_true_vehicle_ratio=(len(proxy_true) / len(rows)) if rows else 0), f, indent=2)
    print(report)


if __name__ == '__main__':
    main()
