import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pickle
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctlib.rbox import obb2poly


CLASS_NAMES = [
    'airport',
    'baseball-diamond',
    'basketball-court',
    'bridge',
    'container-crane',
    'ground-track-field',
    'harbor',
    'helicopter',
    'helipad',
    'large-vehicle',
    'plane',
    'roundabout',
    'ship',
    'small-vehicle',
    'soccer-ball-field',
    'storage-tank',
    'swimming-pool',
    'tennis-court',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize DumpDetResults predictions only.')
    parser.add_argument('--results', required=True, help='DumpDetResults pkl.')
    parser.add_argument('--out-dir', required=True, help='Output vis folder.')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Score threshold.')
    return parser.parse_args()


def class_color(class_name):
    palette = [
        (56, 56, 255), (80, 220, 60), (255, 140, 40),
        (230, 80, 180), (40, 190, 240), (180, 160, 40),
        (255, 90, 90), (90, 220, 220), (170, 90, 255),
        (70, 170, 120), (220, 120, 40), (120, 120, 255),
    ]
    idx = sum(ord(ch) for ch in str(class_name)) % len(palette)
    return palette[idx]


def sample_get(sample, key, default=None):
    if hasattr(sample, key):
        return getattr(sample, key)
    if isinstance(sample, dict):
        return sample.get(key, default)
    try:
        return sample[key]
    except Exception:
        return default


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, 'tensor'):
        x = x.tensor
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        return x.cpu().numpy()
    return np.asarray(x)


def draw_predictions(img_pth, out_pth, boxes, labels, scores, score_thr):
    img = cv2.imread(img_pth)
    if img is None:
        print(f'WARNING: failed to read image: {img_pth}')
        return False

    boxes = to_numpy(boxes)
    labels = to_numpy(labels)
    scores = to_numpy(scores)
    if boxes is None or labels is None or scores is None:
        cv2.imwrite(out_pth, img)
        return True

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    keep = scores >= score_thr
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if boxes.size == 0:
        cv2.imwrite(out_pth, img)
        return True

    polys = obb2poly(boxes)
    h, w = img.shape[:2]
    for poly, label, score in zip(polys, labels, scores):
        class_name = CLASS_NAMES[int(label)]
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
        pts = np.round(pts).astype(np.int32)

        color = class_color(class_name)
        cv2.polylines(img, [pts], True, color, thickness=2,
                      lineType=cv2.LINE_AA)

        text = f'{class_name} {float(score):.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(np.min(pts[:, 0]))
        y = int(np.min(pts[:, 1]))
        x = min(max(x, 0), max(w - tw - 6, 0))
        y = max(y - 4, th + baseline + 4)
        cv2.rectangle(
            img,
            (x, y - th - baseline - 4),
            (min(x + tw + 6, w - 1), min(y + baseline, h - 1)),
            color,
            thickness=-1)
        cv2.putText(
            img,
            text,
            (x + 3, y - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_pth), exist_ok=True)
    return cv2.imwrite(out_pth, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    args = parse_args()
    with open(args.results, 'rb') as f:
        results = pickle.load(f)

    total = len(results)
    saved = 0
    for idx, sample in enumerate(results, start=1):
        img_pth = sample_get(sample, 'img_path')
        pred_instances = sample_get(sample, 'pred_instances')
        if img_pth is None or pred_instances is None:
            continue

        img_name = Path(img_pth).stem
        out_pth = os.path.join(args.out_dir, f'{img_name}.jpg')
        boxes = sample_get(pred_instances, 'bboxes')
        labels = sample_get(pred_instances, 'labels')
        scores = sample_get(pred_instances, 'scores')
        if draw_predictions(img_pth, out_pth, boxes, labels, scores,
                            args.score_thr):
            saved += 1

        if idx % 50 == 0 or idx == total:
            print(f'{idx} / {total}, saved {saved}')

    print(f'Saved visualizations to: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
