import argparse
import os
import pickle

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize OpenRSD SimpleRun results.pkl.')
    parser.add_argument(
        '--results',
        default='./SimpleRun/results/results.pkl',
        help='Path to the SimpleRun results pkl.')
    parser.add_argument(
        '--image-dir',
        default='/data/zcy/dataset/test_ss/images',
        help='Directory containing source images.')
    parser.add_argument(
        '--out-dir',
        default='./SimpleRun/results/vis',
        help='Directory to save visualized images.')
    parser.add_argument(
        '--ext',
        default='.jpg',
        help='Output image extension, for example .jpg or .png.')
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


def find_image(image_dir, img_name):
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
        img_pth = os.path.join(image_dir, f'{img_name}{ext}')
        if os.path.exists(img_pth):
            return img_pth
    return None


def draw_result(img_pth, out_pth, result):
    img = cv2.imread(img_pth)
    if img is None:
        print(f'WARNING: failed to read image: {img_pth}')
        return False

    polys = result.get('polys', [])
    texts = result.get('texts', [])
    scores = result.get('scores', [])
    h, w = img.shape[:2]

    for poly, text, score in zip(polys, texts, scores):
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
        pts = np.round(pts).astype(np.int32)

        color = class_color(text)
        cv2.polylines(img, [pts], True, color, thickness=2,
                      lineType=cv2.LINE_AA)

        label = f'{text} {float(score):.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale,
                                             thickness)
        x = int(np.min(pts[:, 0]))
        y = int(np.min(pts[:, 1]))
        x = min(max(x, 0), max(w - tw - 6, 0))
        y = max(y - 4, th + baseline + 4)
        cv2.rectangle(img, (x, y - th - baseline - 4),
                      (min(x + tw + 6, w - 1), min(y + baseline, h - 1)),
                      color, thickness=-1)
        cv2.putText(img, label, (x + 3, y - 3), font, font_scale,
                    (255, 255, 255), thickness=thickness,
                    lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_pth), exist_ok=True)
    ext = os.path.splitext(out_pth)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return cv2.imwrite(out_pth, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.imwrite(out_pth, img)


def main():
    args = parse_args()
    with open(args.results, 'rb') as f:
        results = pickle.load(f)

    out_ext = args.ext if args.ext.startswith('.') else f'.{args.ext}'
    total = len(results)
    saved = 0
    missing = 0

    for idx, (img_name, result) in enumerate(results.items(), start=1):
        img_pth = find_image(args.image_dir, img_name)
        if img_pth is None:
            missing += 1
            print(f'WARNING: missing image for {img_name}')
            continue

        out_pth = os.path.join(args.out_dir, f'{img_name}{out_ext}')
        if draw_result(img_pth, out_pth, result):
            saved += 1

        if idx % 100 == 0 or idx == total:
            print(f'{idx} / {total}, saved {saved}, missing {missing}')

    print(f'Saved visualizations to: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
