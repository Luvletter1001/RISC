#!/usr/bin/env python
"""Scale counterfactual probe for impossible DOTAV2 class confusions.

The primary target is a localized but semantically suspicious error such as
``small-vehicle -> tennis-court`` or a high-frequency top confusion pair.  The
script takes high-IoU wrong-class cases, crops the GT object, pastes it onto a
neutral canvas at the original scale and at a configured class-prior scale,
then reruns the selected detector.

This is intentionally a diagnostic script, not an evaluator.  It records what
class the detector assigns to the transformed target object after NMS.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

from mmcv.ops import box_iou_rotated  # noqa: E402
from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmrotate.structures.bbox import qbox2rbox, rbox2qbox  # noqa: E402


DEFAULT_RUN_SUMMARY = (
    'work_dirs/dotav2_current_wrongcls_gpu67_20260616_all_summary.csv')
DEFAULT_HARD_CASES = (
    'work_dirs/semantic_ambiguity_study_20260617/hard_cases_target_pairs.csv')
DEFAULT_SOURCE_IMG_DIR = '/data/zcy/dataset/trainval_ms_full/images'
DEFAULT_OUT_DIR = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'e9_scale_counterfactual_probe')
DEFAULT_TARGET_PAIR = 'small-vehicle->tennis-court'
DEFAULT_CLASS_AREA_PRIORS = (
    'work_dirs/semantic_ambiguity_study_20260617/class_area_priors.csv')
PADDING_VALUE = (104, 116, 124)
TILE_RE = re.compile(
    r'^(?P<orig>.+)__(?P<size>\d+)__(?P<x>\d+)___(?P<y>\d+)$')
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
EP2_VARIANT_STEPS = {
    'original_tile': '00',
    'neutral_same_scale': '01',
    'neutral_small_vehicle_scale': '02',
    'neutral_pred_class_scale': '02',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-summary-csv', default=DEFAULT_RUN_SUMMARY)
    parser.add_argument('--hard-cases-csv', default=DEFAULT_HARD_CASES)
    parser.add_argument('--use-full-manifest', action='store_true',
                        help='Read all target-pair cases from each model dedup box_manifest.csv.')
    parser.add_argument('--dedup-subdir',
                        default='vis_wrong_class_iou_gt0p7_merged_original_dedup')
    parser.add_argument('--source-img-dir', default=DEFAULT_SOURCE_IMG_DIR)
    parser.add_argument('--extra-source-img-dirs', default='',
                        help='Comma-separated fallback image directories.')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--model-names', default='faahead_lsknet')
    parser.add_argument('--override-model-name', default='',
                        help='Actual model label when running a config/checkpoint '
                        'not listed in run-summary-csv.')
    parser.add_argument('--override-config', default='',
                        help='Actual detector config override.')
    parser.add_argument('--override-checkpoint', default='',
                        help='Actual detector checkpoint override.')
    parser.add_argument('--case-model-name', default='',
                        help='Source model name used to select hard cases.')
    parser.add_argument('--target-pair', default=DEFAULT_TARGET_PAIR)
    parser.add_argument('--max-cases', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.01)
    parser.add_argument('--match-iou-thr', type=float, default=0.10)
    parser.add_argument('--small-area', type=float, default=582.0)
    parser.add_argument('--scale-target-mode',
                        choices=['small_vehicle', 'pred_class'],
                        default='small_vehicle')
    parser.add_argument('--class-area-priors-csv',
                        default=DEFAULT_CLASS_AREA_PRIORS)
    parser.add_argument('--min-small-scale', type=float, default=0.18)
    parser.add_argument('--max-small-scale', type=float, default=0.55)
    parser.add_argument('--min-pred-class-scale', type=float, default=0.10)
    parser.add_argument('--max-pred-class-scale', type=float, default=2.50)
    parser.add_argument('--canvas-size', type=int, default=1024)
    parser.add_argument('--crop-padding', type=int, default=10)
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--dry-run', action='store_true',
                        help='Build counterfactual images without loading a detector.')
    parser.add_argument('--save-variant-images-dir', default='',
                        help='Optional directory for raw E-P2 variant images.')
    parser.add_argument('--infer-from-saved-variant-path', action='store_true',
                        help='Run inference on saved variant image paths so '
                        'pre-NMS dump hooks can parse E-P2 metadata from img_path.')
    return parser.parse_args()


def read_csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def load_class_area_priors(path):
    if not path or not Path(path).exists():
        return {}
    priors = {}
    for row in read_csv_rows(path):
        cls = row.get('class', '')
        if not cls:
            continue
        try:
            priors[cls] = float(row.get('median_area') or 0.0)
        except ValueError:
            continue
    return priors


def parse_model_names(raw):
    return [item.strip() for item in raw.split(',') if item.strip()]


def safe_token(value, preserve_underscore=True):
    value = str(value)
    pattern = r'[^0-9A-Za-z_]+' if preserve_underscore else r'[^0-9A-Za-z]+'
    token = re.sub(pattern, '-', value).strip('-')
    token = re.sub(r'-+', '-', token)
    return token or 'na'


def ep2_variant_filename(model_name, case_index, tile_img_id, variant_name):
    tile_token = safe_token(tile_img_id, preserve_underscore=False)
    model_token = safe_token(model_name, preserve_underscore=True)
    variant_token = safe_token(variant_name, preserve_underscore=True)
    step = EP2_VARIANT_STEPS.get(str(variant_name), '99')
    return (
        f'ep2case-{model_token}-{int(case_index):04d}-{tile_token}'
        f'__ep2ctx-{tile_token}'
        f'__ep2obj-{model_token}-{int(case_index):04d}'
        f'__ep2ctrl-{variant_token}'
        f'__ep2step-{step}.jpg')


def resolve_variant_image_dir(args, out_dir):
    raw = args.save_variant_images_dir
    if not raw and not args.infer_from_saved_variant_path:
        return None
    path = Path(raw) if raw else Path(out_dir) / 'ep2_variant_images'
    if not path.is_absolute():
        path = Path(out_dir) / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_model_rows(summary_csv, model_names):
    wanted = set(model_names)
    rows = []
    for row in read_csv_rows(summary_csv):
        if wanted and row['model'] not in wanted:
            continue
        rows.append(row)
    found = {row['model'] for row in rows}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f'missing model names in summary csv: {missing}')
    return rows


def build_override_model_rows(override_model_name, override_config,
                              override_checkpoint, case_model_name=''):
    if not override_config and not override_checkpoint:
        return []
    if not override_config or not override_checkpoint:
        raise ValueError(
            '--override-config and --override-checkpoint must be set together')
    model_name = override_model_name or Path(override_config).stem
    return [{
        'model': model_name,
        'config': override_config,
        'checkpoint': override_checkpoint,
        'case_model': case_model_name or model_name,
    }]


def parse_tile_stem(stem):
    match = TILE_RE.match(stem)
    if not match:
        raise ValueError(f'not a DOTA split tile name: {stem}')
    return {
        'orig_id': match.group('orig'),
        'size': int(match.group('size')),
        'x': int(match.group('x')),
        'y': int(match.group('y')),
    }


def parse_qbox(raw):
    return np.asarray(
        [float(item) for item in raw.replace(',', ' ').split()],
        dtype=np.float32,
    ).reshape(4, 2)


def qbox_area(qbox):
    pts = np.asarray(qbox, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1))) / 2.0)


def qbox_to_text(qbox):
    pts = np.asarray(qbox, dtype=np.float32).reshape(-1)
    return ' '.join(f'{float(v):.2f}' for v in pts)


def parse_source_img_dirs(args):
    dirs = [args.source_img_dir]
    dirs.extend(parse_model_names(args.extra_source_img_dirs))
    return [Path(item) for item in dirs if item]


def find_image(root, stem):
    root = Path(root)
    for suffix in IMAGE_SUFFIXES:
        path = root / f'{stem}{suffix}'
        if path.exists():
            return path
    return None


def find_image_in_dirs(roots, stem):
    for root in roots:
        path = find_image(root, stem)
        if path is not None:
            return path
    return None


def qbox_original_to_tile(qbox, tile):
    offset = np.asarray([tile['x'], tile['y']], dtype=np.float32)
    return np.asarray(qbox, dtype=np.float32).reshape(4, 2) - offset


def crop_bounds(qbox, shape, padding):
    h, w = shape[:2]
    pts = np.asarray(qbox, dtype=np.float32).reshape(4, 2)
    x0 = max(0, int(math.floor(float(pts[:, 0].min()))) - padding)
    y0 = max(0, int(math.floor(float(pts[:, 1].min()))) - padding)
    x1 = min(w, int(math.ceil(float(pts[:, 0].max()))) + padding)
    y1 = min(h, int(math.ceil(float(pts[:, 1].max()))) + padding)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('empty crop bounds')
    return x0, y0, x1, y1


def paste_center_on_neutral(img, qbox, scale, canvas_size, crop_padding):
    x0, y0, x1, y1 = crop_bounds(qbox, img.shape, crop_padding)
    crop = img[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    new_w = max(2, int(round(crop_w * scale)))
    new_h = max(2, int(round(crop_h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.empty((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(PADDING_VALUE, dtype=np.uint8)
    dst_x0 = max(0, (canvas_size - new_w) // 2)
    dst_y0 = max(0, (canvas_size - new_h) // 2)
    dst_x1 = min(canvas_size, dst_x0 + new_w)
    dst_y1 = min(canvas_size, dst_y0 + new_h)
    src_w = dst_x1 - dst_x0
    src_h = dst_y1 - dst_y0
    if src_w <= 0 or src_h <= 0:
        raise ValueError('resized crop does not fit neutral canvas')
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[:src_h, :src_w]

    qbox_local = (np.asarray(qbox, dtype=np.float32).reshape(4, 2) -
                  np.asarray([x0, y0], dtype=np.float32))
    qbox_new = qbox_local * float(scale) + np.asarray(
        [dst_x0, dst_y0], dtype=np.float32)
    return canvas, qbox_new, {
        'crop_x0': x0,
        'crop_y0': y0,
        'crop_w': crop_w,
        'crop_h': crop_h,
        'scale': float(scale),
        'dst_x0': dst_x0,
        'dst_y0': dst_y0,
        'dst_w': src_w,
        'dst_h': src_h,
    }


def resolve_scale_target_area(args, fallback_class='small-vehicle'):
    if getattr(args, 'scale_target_mode', 'small_vehicle') == 'pred_class':
        priors = getattr(args, 'pred_class_area_priors', {}) or {}
        area = float(priors.get(fallback_class, 0.0) or 0.0)
        if area > 0:
            return area, fallback_class
    return float(args.small_area), 'small-vehicle'


def build_variants(img, qbox_tile, args, scale_class='small-vehicle'):
    area = qbox_area(qbox_tile)
    if area <= 0:
        raise ValueError('non-positive target area')
    scale_area, scale_label = resolve_scale_target_area(
        args, fallback_class=scale_class)
    scale_factor = math.sqrt(scale_area / area)
    if getattr(args, 'scale_target_mode', 'small_vehicle') == 'pred_class':
        min_scale = args.min_pred_class_scale
        max_scale = args.max_pred_class_scale
        scale_variant_name = 'neutral_pred_class_scale'
    else:
        min_scale = args.min_small_scale
        max_scale = args.max_small_scale
        scale_variant_name = 'neutral_small_vehicle_scale'
    scale_factor = max(min_scale, min(max_scale, scale_factor))

    variants = [{
        'variant': 'original_tile',
        'image': img.copy(),
        'target_qbox': qbox_tile.copy(),
        'scale': 1.0,
        'target_area': area,
        'note': 'full original tile context',
    }]
    same_img, same_qbox, same_meta = paste_center_on_neutral(
        img, qbox_tile, 1.0, args.canvas_size, args.crop_padding)
    variants.append({
        'variant': 'neutral_same_scale',
        'image': same_img,
        'target_qbox': same_qbox,
        'scale': 1.0,
        'target_area': qbox_area(same_qbox),
        'note': json.dumps(same_meta, ensure_ascii=False),
    })
    small_img, small_qbox, small_meta = paste_center_on_neutral(
        img, qbox_tile, scale_factor, args.canvas_size, args.crop_padding)
    small_meta['scale_target_class'] = scale_label
    small_meta['scale_target_area'] = float(scale_area)
    small_meta['scale_target_mode'] = getattr(
        args, 'scale_target_mode', 'small_vehicle')
    variants.append({
        'variant': scale_variant_name,
        'image': small_img,
        'target_qbox': small_qbox,
        'scale': scale_factor,
        'target_area': qbox_area(small_qbox),
        'note': json.dumps(small_meta, ensure_ascii=False),
    })
    return variants


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
        return np.stack(
            [x1, y1, x2, y1, x2, y2, x1, y2], axis=1).astype(np.float32)
    raise ValueError(f'unsupported bbox shape: {boxes.shape}')


def get_result_arrays(result):
    pred = result.pred_instances
    qboxes = boxes_to_qboxes(tensor_to_numpy(pred.bboxes))
    labels = tensor_to_numpy(pred.labels).astype(np.int64)
    scores = tensor_to_numpy(pred.scores).astype(np.float32)
    return qboxes, labels, scores


def get_classes(model):
    meta = getattr(model, 'dataset_meta', None)
    if isinstance(meta, dict):
        classes = meta.get('classes')
    else:
        classes = getattr(meta, 'classes', None)
    return list(classes or [])


def class_name(classes, label):
    label = int(label)
    if 0 <= label < len(classes):
        return classes[label]
    return f'class_{label}'


def qboxes_to_rboxes(qboxes):
    qboxes = np.asarray(qboxes, dtype=np.float32).reshape(-1, 8)
    if len(qboxes) == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    return qbox2rbox(torch.from_numpy(qboxes)).float()


def match_target(result, target_qbox, classes, score_thr):
    qboxes, labels, scores = get_result_arrays(result)
    keep = np.where(scores >= score_thr)[0]
    top_idx = int(np.argmax(scores)) if len(scores) else -1
    top = {
        'top_class': '',
        'top_score': '',
        'top_qbox': '',
    }
    if top_idx >= 0:
        top = {
            'top_class': class_name(classes, labels[top_idx]),
            'top_score': f'{float(scores[top_idx]):.6f}',
            'top_qbox': qbox_to_text(qboxes[top_idx]),
        }
    if len(keep) == 0:
        return {
            **top,
            'raw_detections': int(len(scores)),
            'kept_detections': 0,
            'matched': False,
            'matched_class': '',
            'matched_score': '',
            'matched_iou': '',
            'matched_qbox': '',
        }

    pred_qboxes = qboxes[keep]
    pred_rboxes = qboxes_to_rboxes(pred_qboxes)
    target_rbox = qboxes_to_rboxes(
        np.asarray(target_qbox, dtype=np.float32).reshape(1, 8))
    ious = box_iou_rotated(pred_rboxes, target_rbox).squeeze(1)
    best_local = int(torch.argmax(ious).item())
    best_idx = int(keep[best_local])
    best_iou = float(ious[best_local].item())
    return {
        **top,
        'raw_detections': int(len(scores)),
        'kept_detections': int(len(keep)),
        'matched': True,
        'matched_class': class_name(classes, labels[best_idx]),
        'matched_score': f'{float(scores[best_idx]):.6f}',
        'matched_iou': f'{best_iou:.6f}',
        'matched_qbox': qbox_to_text(qboxes[best_idx]),
    }


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
    y = max(18, int(pts_i[:, 1].min()) - 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.36
    (tw, th), bl = cv2.getTextSize(label, font, scale, 1)
    bg_x2 = min(w - 1, x + tw + 8)
    bg_y1 = max(0, y - th - bl - 4)
    cv2.rectangle(img, (x, bg_y1), (bg_x2, y + bl), color, -1)
    cv2.putText(img, label, (x + 4, y - 3), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def parse_qbox_field(raw):
    if not raw:
        return None
    return parse_qbox(raw)


def visualize_variant(img, target_qbox, match, target_class):
    canvas = img.copy()
    draw_poly(canvas, target_qbox, (0, 220, 255), f'GT {target_class}', 2)
    matched_qbox = parse_qbox_field(match.get('matched_qbox', ''))
    if matched_qbox is None:
        return canvas
    cls = match.get('matched_class') or 'none'
    score = match.get('matched_score') or 'nan'
    iou = match.get('matched_iou') or 'nan'
    color = (0, 0, 255)
    if cls == target_class:
        color = (0, 170, 0)
    elif cls == 'small-vehicle':
        color = (255, 0, 0)
    draw_poly(canvas, matched_qbox, color, f'{cls} s={score} iou={iou}', 2)
    return canvas


def filter_hard_cases(rows, model_name, target_pair, max_cases):
    pred_class, gt_class = target_pair.split('->', 1)
    selected = [
        row for row in rows
        if row['model'] == model_name and row['pred_class'] == pred_class
        and row['gt_class'] == gt_class
    ]
    selected.sort(key=lambda row: (
        -float(row.get('score') or 0),
        -float(row.get('iou') or 0),
        row.get('orig_id', ''),
        row.get('tile_img_id', ''),
    ))
    if max_cases > 0:
        selected = selected[:max_cases]
    return selected


def load_full_manifest_cases(model_rows, dedup_subdir, target_pair):
    pred_class, gt_class = target_pair.split('->', 1)
    cases = []
    for model_row in model_rows:
        output_dir = Path(model_row['output_dir'])
        manifest_path = output_dir / dedup_subdir / 'box_manifest.csv'
        if not manifest_path.exists():
            raise FileNotFoundError(f'missing box manifest: {manifest_path}')
        for row in read_csv_rows(manifest_path):
            if row.get('kind') != 'wrong':
                continue
            if row.get('pred_class') != pred_class:
                continue
            if row.get('gt_class') != gt_class:
                continue
            row = dict(row)
            row['model'] = model_row['model']
            row['gt_area'] = f'{qbox_area(parse_qbox(row["gt_qbox_original"])):.2f}'
            row['pred_area'] = f'{qbox_area(parse_qbox(row["pred_qbox_original"])):.2f}'
            row['source_manifest'] = str(manifest_path)
            cases.append(row)
    return cases


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def summarize_rows(rows, target_class, impossible_pred_class):
    by_model_variant = defaultdict(list)
    for row in rows:
        by_model_variant[(row['model'], row['variant'])].append(row)

    summary_rows = []
    for (model, variant), items in sorted(by_model_variant.items()):
        classes = Counter(row['matched_class'] or 'no_match' for row in items)
        valid = [
            row for row in items
            if row.get('matched_iou') and
            float(row['matched_iou']) >= float(row['match_iou_thr'])
        ]
        valid_classes = Counter(row['matched_class'] for row in valid)
        summary_rows.append({
            'model': model,
            'variant': variant,
            'cases': len(items),
            'target_matched_cases': len(valid),
            'target_class_matches': valid_classes.get(target_class, 0),
            'impossible_pred_matches': valid_classes.get(
                impossible_pred_class, 0),
            'matched_class_hist': ';'.join(
                f'{k}:{v}' for k, v in classes.most_common()),
        })
    return summary_rows


def write_markdown(path, run_info, summary_rows, rows):
    pred_class, gt_class = run_info['target_pair'].split('->', 1)
    scale_mode = run_info.get('scale_target_mode', 'small_vehicle')
    if scale_mode == 'pred_class':
        scale_variant = 'neutral_pred_class_scale'
        scale_purpose = (
            f'将同一个 `{gt_class}` 裁剪块缩放到 `{pred_class}` 的类别面积'
            '先验，测试错误类别的尺度吸引是否会改变分类。')
    else:
        scale_variant = 'neutral_small_vehicle_scale'
        scale_purpose = (
            '将同一个目标裁剪块缩放到 small-vehicle 典型面积，测试小目标'
            '尺度先验是否被分类头使用。')
    lines = [
        '# E9 Scale Counterfactual Probe',
        '',
        '这个 probe 针对定位正确但分类错误的 top confusion pair。它不报告 '
        'mAP，而是追踪同一个 GT 目标在不同反事实图像中的目标匹配检测类别。',
        '',
        '## Run Info',
        '',
        f'- `target_pair`: `{run_info["target_pair"]}`',
        f'- `device`: `{run_info["device"]}`',
        f'- `score_thr`: `{run_info["score_thr"]}`',
        f'- `match_iou_thr`: `{run_info["match_iou_thr"]}`',
        f'- `small_area`: `{run_info["small_area"]}`',
        f'- `scale_target_mode`: `{scale_mode}`',
        f'- `class_area_priors_csv`: `{run_info.get("class_area_priors_csv", "")}`',
        f'- `class_area_priors_loaded`: `{run_info.get("class_area_priors_loaded", 0)}`',
        f'- `models`: `{", ".join(run_info["models"])}`',
        f'- `source_img_dirs`: `{", ".join(run_info.get("source_img_dirs", []))}`',
        '',
        '## Variant Definition',
        '',
        '| variant | purpose |',
        '| --- | --- |',
        '| `original_tile` | 保留完整 1024 tile 上下文，复现原始错误。 |',
        f'| `neutral_same_scale` | 只保留 `{gt_class}` 裁剪块并放到中性背景，测试上下文是否导致错误。 |',
        f'| `{scale_variant}` | {scale_purpose} |',
        '',
        '## Summary',
        '',
        '| model | variant | cases | target_matched_cases | target_class_matches | impossible_pred_matches | matched_class_hist |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in summary_rows:
        lines.append(
            f'| {row["model"]} | {row["variant"]} | {row["cases"]} | '
            f'{row["target_matched_cases"]} | {row["target_class_matches"]} | '
            f'{row["impossible_pred_matches"]} | '
            f'{row["matched_class_hist"]} |')
    lines.extend([
        '',
        '## Interpretation Rule',
        '',
        f'- 如果 `neutral_same_scale` 仍大量输出 `{pred_class}`，说明错误不是由复杂上下文单独触发，而是目标外观或分类头自身存在强吸引子。',
        f'- 如果 `neutral_same_scale` 回到 `{gt_class}`，而 `original_tile` 错，说明上下文、邻域目标或局部纹理诱导更强。',
        f'- 如果 `{scale_variant}` 更容易输出 `{pred_class}`，说明分类器对该类别尺度先验敏感；如果同尺度大目标也输出 `{pred_class}`，则说明模型没有使用应有的尺度约束。',
        '',
        '## Output Files',
        '',
        f'- CSV: `{run_info["result_csv"]}`',
        f'- visualizations: `{run_info["visualization_dir"]}`',
        '',
    ])
    if rows:
        lines.extend([
            '## First Cases',
            '',
            '| model | orig_id | tile_img_id | variant | matched_class | matched_score | matched_iou | vis_path |',
            '| --- | --- | --- | --- | --- | ---: | ---: | --- |',
        ])
        for row in rows[:20]:
            lines.append(
                f'| {row["model"]} | {row["orig_id"]} | '
                f'{row["tile_img_id"]} | {row["variant"]} | '
                f'{row["matched_class"]} | {row["matched_score"]} | '
                f'{row["matched_iou"]} | `{row["vis_path"]}` |')
    Path(path).write_text('\n'.join(lines) + '\n')


def run_model(row, hard_rows, args, out_dir):
    model_name = row['model']
    if args.dry_run:
        print(json.dumps({
            'event': 'dry_run_skip_model_load',
            'model': model_name,
        }, ensure_ascii=False), flush=True)
        model = None
        classes = []
    else:
        print(json.dumps({
            'event': 'load_model',
            'model': model_name,
            'config': row['config'],
            'checkpoint': row['checkpoint'],
            'device': args.device,
        }, ensure_ascii=False), flush=True)
        model = init_detector(row['config'], row['checkpoint'],
                              device=args.device)
        classes = get_classes(model)
    pred_class, gt_class = args.target_pair.split('->', 1)
    source_dirs = parse_source_img_dirs(args)
    vis_dir = out_dir / 'visualizations' / model_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    variant_image_dir = resolve_variant_image_dir(args, out_dir)

    output_rows = []
    for case_idx, case in enumerate(hard_rows):
        image_path = find_image_in_dirs(source_dirs, case['tile_img_id'])
        if image_path is None:
            output_rows.append({
                'model': model_name,
                'case_index': case_idx,
                'orig_id': case['orig_id'],
                'tile_img_id': case['tile_img_id'],
                'variant': '',
                'error': 'missing_source_image',
            })
            continue
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            output_rows.append({
                'model': model_name,
                'case_index': case_idx,
                'orig_id': case['orig_id'],
                'tile_img_id': case['tile_img_id'],
                'variant': '',
                'error': 'failed_to_read_source_image',
            })
            continue

        tile = parse_tile_stem(case['tile_img_id'])
        gt_qbox_tile = qbox_original_to_tile(parse_qbox(case['gt_qbox_original']),
                                             tile)
        try:
            variants = build_variants(
                img, gt_qbox_tile, args, scale_class=pred_class)
        except Exception as exc:  # noqa: BLE001 - diagnostic row
            output_rows.append({
                'model': model_name,
                'case_index': case_idx,
                'orig_id': case['orig_id'],
                'tile_img_id': case['tile_img_id'],
                'variant': '',
                'error': f'failed_to_build_variants:{exc}',
            })
            continue

        for variant in variants:
            variant_image_path = ''
            if variant_image_dir is not None:
                variant_image_path = str(
                    variant_image_dir / ep2_variant_filename(
                        model_name=model_name,
                        case_index=case_idx,
                        tile_img_id=case['tile_img_id'],
                        variant_name=variant['variant']))
                cv2.imwrite(
                    variant_image_path,
                    variant['image'],
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if args.dry_run:
                match = {
                    'raw_detections': 0,
                    'kept_detections': 0,
                    'matched_class': '',
                    'matched_score': '',
                    'matched_iou': '',
                    'matched_qbox': '',
                    'top_class': '',
                    'top_score': '',
                }
            else:
                infer_input = (
                    variant_image_path
                    if args.infer_from_saved_variant_path
                    and variant_image_path else variant['image'])
                result = inference_detector(model, infer_input)
                match = match_target(
                    result, variant['target_qbox'], classes, args.score_thr)
            target_matched = False
            if match.get('matched_iou'):
                target_matched = (
                    float(match['matched_iou']) >= args.match_iou_thr)
            vis = visualize_variant(
                variant['image'], variant['target_qbox'], match, gt_class)
            vis_name = (
                f'{case_idx:03d}__{case["orig_id"]}__'
                f'{case["tile_img_id"]}__{variant["variant"]}.jpg')
            vis_path = vis_dir / vis_name
            cv2.imwrite(str(vis_path), vis,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            output_rows.append({
                'model': model_name,
                'case_index': case_idx,
                'orig_id': case['orig_id'],
                'tile_img_id': case['tile_img_id'],
                'source_path': str(image_path),
                'variant': variant['variant'],
                'target_pair': args.target_pair,
                'original_wrong_score': case['score'],
                'original_wrong_iou': case['iou'],
                'target_gt_class': gt_class,
                'impossible_pred_class': pred_class,
                'original_gt_area': case['gt_area'],
                'variant_target_area': f'{variant["target_area"]:.2f}',
                'variant_scale': f'{variant["scale"]:.6f}',
                'target_qbox': qbox_to_text(variant['target_qbox']),
                'score_thr': f'{args.score_thr:.4f}',
                'match_iou_thr': f'{args.match_iou_thr:.4f}',
                'target_matched': int(target_matched),
                'dry_run': int(args.dry_run),
                'matched_class': match.get('matched_class', ''),
                'matched_score': match.get('matched_score', ''),
                'matched_iou': match.get('matched_iou', ''),
                'matched_qbox': match.get('matched_qbox', ''),
                'top_class': match.get('top_class', ''),
                'top_score': match.get('top_score', ''),
                'raw_detections': match.get('raw_detections', 0),
                'kept_detections': match.get('kept_detections', 0),
                'variant_image_path': variant_image_path,
                'vis_path': str(vis_path),
                'variant_note': variant['note'],
                'error': '',
            })
            print(json.dumps({
                'event': 'case_variant_built' if args.dry_run else
                'case_variant_done',
                'model': model_name,
                'case_index': case_idx,
                'tile_img_id': case['tile_img_id'],
                'variant': variant['variant'],
                'matched_class': match.get('matched_class', ''),
                'matched_score': match.get('matched_score', ''),
                'matched_iou': match.get('matched_iou', ''),
            }, ensure_ascii=False), flush=True)
    return output_rows


def main():
    args = parse_args()
    args.pred_class_area_priors = load_class_area_priors(
        args.class_area_priors_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = parse_model_names(args.model_names)
    override_rows = build_override_model_rows(
        args.override_model_name,
        args.override_config,
        args.override_checkpoint,
        args.case_model_name or (model_names[0] if model_names else ''))
    model_rows = (
        override_rows if override_rows
        else read_model_rows(args.run_summary_csv, model_names))
    if args.use_full_manifest:
        hard_case_rows = load_full_manifest_cases(
            model_rows, args.dedup_subdir, args.target_pair)
        hard_cases_source = 'full_manifest'
    else:
        hard_case_rows = read_csv_rows(args.hard_cases_csv)
        hard_cases_source = args.hard_cases_csv

    run_info = {
        'target_pair': args.target_pair,
        'models': [row['model'] for row in model_rows],
        'device': args.device,
        'score_thr': args.score_thr,
        'match_iou_thr': args.match_iou_thr,
        'small_area': args.small_area,
        'scale_target_mode': args.scale_target_mode,
        'class_area_priors_csv': args.class_area_priors_csv,
        'class_area_priors_loaded': len(args.pred_class_area_priors),
        'min_small_scale': args.min_small_scale,
        'max_small_scale': args.max_small_scale,
        'min_pred_class_scale': args.min_pred_class_scale,
        'max_pred_class_scale': args.max_pred_class_scale,
        'source_img_dir': args.source_img_dir,
        'extra_source_img_dirs': args.extra_source_img_dirs,
        'source_img_dirs': [str(path) for path in parse_source_img_dirs(args)],
        'hard_cases_csv': args.hard_cases_csv,
        'use_full_manifest': args.use_full_manifest,
        'dedup_subdir': args.dedup_subdir,
        'hard_cases_source': hard_cases_source,
        'run_summary_csv': args.run_summary_csv,
        'override_model_name': args.override_model_name,
        'override_config': args.override_config,
        'override_checkpoint': args.override_checkpoint,
        'case_model_name': args.case_model_name,
    }
    (out_dir / f'run_info_{os.getpid()}.json').write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + os.linesep)
    print(json.dumps({'event': 'run_info', **run_info},
                     ensure_ascii=False), flush=True)

    rows = []
    for model_row in model_rows:
        case_model = model_row.get('case_model', model_row['model'])
        cases = filter_hard_cases(
            hard_case_rows, case_model, args.target_pair,
            args.max_cases)
        print(json.dumps({
            'event': 'selected_cases',
            'model': model_row['model'],
            'case_model': case_model,
            'cases': len(cases),
        }, ensure_ascii=False), flush=True)
        rows.extend(run_model(model_row, cases, args, out_dir))

    result_csv = out_dir / 'e9_counterfactual_probe.csv'
    write_csv(result_csv, rows)
    summary_rows = summarize_rows(
        [row for row in rows if not row.get('error')],
        target_class=args.target_pair.split('->', 1)[1],
        impossible_pred_class=args.target_pair.split('->', 1)[0],
    )
    summary_csv = out_dir / 'e9_counterfactual_summary.csv'
    write_csv(summary_csv, summary_rows)
    run_info.update({
        'result_csv': str(result_csv),
        'summary_csv': str(summary_csv),
        'visualization_dir': str(out_dir / 'visualizations'),
    })
    write_markdown(out_dir / 'e9_counterfactual_probe.md',
                   run_info, summary_rows,
                   [row for row in rows if not row.get('error')])
    print(json.dumps({
        'event': 'done',
        'result_csv': str(result_csv),
        'summary_csv': str(summary_csv),
        'markdown': str(out_dir / 'e9_counterfactual_probe.md'),
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
