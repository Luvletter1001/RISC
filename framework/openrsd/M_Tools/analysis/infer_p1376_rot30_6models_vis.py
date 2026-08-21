#!/usr/bin/env python
"""Run 30-degree rotation inference visualization for P1376 tiles."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmrotate.structures.bbox import rbox2qbox, qbox2rbox  # noqa: E402


DEFAULT_ANGLES = tuple(range(0, 360, 30))
DEFAULT_OUT_DIR = 'work_dirs/p1376_rot30_6models_vis_20260616'
PADDING_VALUE = (104, 116, 124)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-summary-csv',
                        default='work_dirs/dotav2_current_wrongcls_gpu67_20260616_all_summary.csv')
    parser.add_argument('--source-img-dir',
                        default='/data/zcy/dataset/trainval_ms_full/images')
    parser.add_argument('--tile-glob', default='P1376__1024__*.png')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--model-names', default='',
                        help='Comma-separated model names. Empty means all.')
    parser.add_argument('--angles', default=','.join(str(a) for a in DEFAULT_ANGLES))
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--jpeg-quality', type=int, default=90)
    parser.add_argument('--save-rotated', action='store_true')
    return parser.parse_args()


def read_model_rows(summary_csv, model_names):
    wanted = {name for name in model_names.split(',') if name}
    rows = []
    with Path(summary_csv).open(newline='') as f:
        for row in csv.DictReader(f):
            if wanted and row['model'] not in wanted:
                continue
            rows.append(row)
    if wanted:
        found = {row['model'] for row in rows}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f'missing model names in summary csv: {missing}')
    return rows


def parse_angles(raw):
    angles = []
    for item in raw.split(','):
        item = item.strip()
        if item:
            angles.append(float(item))
    return angles


def rotate_image_same_canvas(img, angle):
    h, w = img.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PADDING_VALUE)


def tensor_to_numpy(value):
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def boxes_to_qboxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 8), dtype=np.float32)
    boxes = boxes.reshape(boxes.shape[0], -1)
    if boxes.shape[1] == 8:
        return boxes
    if boxes.shape[1] == 5:
        return rbox2qbox(torch.from_numpy(boxes)).numpy().astype(np.float32)
    if boxes.shape[1] == 4:
        x1, y1, x2, y2 = boxes.T
        return np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1).astype(np.float32)
    raise ValueError(f'Unsupported bbox shape: {boxes.shape}')


def get_result_arrays(result):
    pred = result.pred_instances
    qboxes = boxes_to_qboxes(tensor_to_numpy(pred.bboxes))
    labels = tensor_to_numpy(pred.labels).astype(np.int64)
    scores = tensor_to_numpy(pred.scores).astype(np.float32)
    return qboxes, labels, scores


def class_name(model, label):
    classes = getattr(model.dataset_meta, 'classes', None)
    if classes is None and isinstance(model.dataset_meta, dict):
        classes = model.dataset_meta.get('classes')
    if classes and 0 <= int(label) < len(classes):
        return classes[int(label)]
    return f'class_{int(label)}'


def color_for_label(label):
    label = int(label)
    return (
        int((37 * label + 53) % 255),
        int((97 * label + 149) % 255),
        int((173 * label + 211) % 255),
    )


def draw_poly(img, qbox, color, label='', thickness=2):
    pts = np.asarray(qbox, dtype=np.float32).reshape(4, 2)
    h, w = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.round(pts).astype(np.int32)
    cv2.polylines(img, [pts_i], True, color, thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    x = int(pts_i[:, 0].min())
    y = max(16, int(pts_i[:, 1].min()) - 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.34
    (tw, th), bl = cv2.getTextSize(label, font, scale, 1)
    bg_x2 = min(w - 1, x + tw + 6)
    bg_y1 = max(0, y - th - bl - 4)
    cv2.rectangle(img, (x, bg_y1), (bg_x2, y + bl), color, -1)
    cv2.putText(img, label, (x + 3, y - 3), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def visualize_predictions(model, img, result, score_thr):
    canvas = img.copy()
    qboxes, labels, scores = get_result_arrays(result)
    keep = np.where(scores >= score_thr)[0]
    # Low-score visualizations can be dense. Draw high scores last so they stay visible.
    keep = keep[np.argsort(scores[keep])]
    for idx in keep:
        label = int(labels[idx])
        score = float(scores[idx])
        name = class_name(model, label)
        text = f'{name} {score:.2f}'
        draw_poly(canvas, qboxes[idx], color_for_label(label), text, thickness=2)
    return canvas, len(keep), int(len(scores))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / 'visualizations'
    rotated_dir = out_dir / 'rotated_images'
    manifest_dir = out_dir / 'manifests'
    vis_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rotated:
        rotated_dir.mkdir(parents=True, exist_ok=True)

    angles = parse_angles(args.angles)
    model_rows = read_model_rows(args.run_summary_csv, args.model_names)
    image_paths = sorted(Path(args.source_img_dir).glob(args.tile_glob))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise FileNotFoundError(
            f'no images matched {Path(args.source_img_dir) / args.tile_glob}')

    run_info = {
        'source_img_dir': str(args.source_img_dir),
        'tile_glob': args.tile_glob,
        'num_images': len(image_paths),
        'angles': angles,
        'models': [row['model'] for row in model_rows],
        'score_thr': args.score_thr,
        'device': args.device,
        'visualizations_dir': str(vis_dir),
    }
    (out_dir / f'run_info_{os.getpid()}.json').write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + os.linesep)
    print(json.dumps(run_info, ensure_ascii=False), flush=True)

    for row in model_rows:
        model_name = row['model']
        manifest_path = manifest_dir / f'{model_name}.csv'
        summary_path = manifest_dir / f'{model_name}_summary.json'
        print(json.dumps({
            'event': 'load_model',
            'model': model_name,
            'config': row['config'],
            'checkpoint': row['checkpoint'],
            'device': args.device,
        }, ensure_ascii=False), flush=True)
        model = init_detector(row['config'], row['checkpoint'], device=args.device)
        rows = []
        total_kept = 0
        total_raw = 0
        for image_path in image_paths:
            src = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if src is None:
                rows.append({
                    'model': model_name,
                    'angle': '',
                    'img_id': image_path.stem,
                    'source_path': str(image_path),
                    'vis_path': '',
                    'kept_boxes': 0,
                    'raw_boxes': 0,
                    'error': 'failed_to_read_image',
                })
                continue
            for angle in angles:
                rotated = rotate_image_same_canvas(src, angle)
                angle_i = int(round(angle))
                if args.save_rotated:
                    angle_dir = rotated_dir / f'angle_{angle_i:03d}'
                    angle_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(angle_dir / image_path.name), rotated)
                result = inference_detector(model, rotated)
                vis, kept, raw = visualize_predictions(
                    model, rotated, result, args.score_thr)
                out_name = f'{model_name}__angle_{angle_i:03d}__{image_path.stem}.jpg'
                out_path = vis_dir / out_name
                ok = cv2.imwrite(str(out_path), vis,
                                 [int(cv2.IMWRITE_JPEG_QUALITY),
                                  args.jpeg_quality])
                rows.append({
                    'model': model_name,
                    'angle': angle_i,
                    'img_id': image_path.stem,
                    'source_path': str(image_path),
                    'vis_path': str(out_path),
                    'kept_boxes': kept,
                    'raw_boxes': raw,
                    'error': '' if ok else 'failed_to_write_visualization',
                })
                total_kept += kept
                total_raw += raw
        with manifest_path.open('w', newline='') as f:
            fieldnames = [
                'model', 'angle', 'img_id', 'source_path', 'vis_path',
                'kept_boxes', 'raw_boxes', 'error'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            'model': model_name,
            'images': len(image_paths),
            'angles': angles,
            'expected_visualizations': len(image_paths) * len(angles),
            'written_visualizations': sum(1 for r in rows if not r['error']),
            'score_thr': args.score_thr,
            'total_kept_boxes': total_kept,
            'total_raw_boxes': total_raw,
            'manifest_csv': str(manifest_path),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
