#!/usr/bin/env python
"""Extract non-GT-overlapping background patches from high-risk tiles."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U
from tools.sv_attractor_repair_gpu89.common import parse_dota_txt


def load_highrisk_tiles(result_mech: Path) -> list:
    p = result_mech / 'ftable_01_attractor_highrisk_tiles.csv'
    if p.exists():
        rows = U.read_csv(p)
        return [r['tile_id'] for r in rows[:6]]
    summ = result_mech / 'ftable_01_attractor_atlas_tile_summary.csv'
    if summ.exists():
        rows = sorted(U.read_csv(summ), key=lambda r: -float(r.get('baseline_final_sv', 0)))
        return [r['tile_id'] for r in rows[:6]]
    return U.HIGHRISK_TILES[:6]


def gt_hboxes(tile_id: str) -> np.ndarray:
    ann = REPO / 'vis' / tile_id / 'dataset' / 'annfiles'
    boxes = []
    for txt in ann.glob('*.txt'):
        for b in parse_dota_txt(txt):
            poly = np.array(b['poly'], dtype=np.float32).reshape(-1, 2)
            x1, y1 = poly.min(0)
            x2, y2 = poly.max(0)
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6
    return inter / ua


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', type=Path, default=REPO)
    ap.add_argument('--result-dir', type=Path, default=U.RESULT_MD)
    ap.add_argument('--patches-per-tile', type=int, default=30)
    ap.add_argument('--sizes', type=str, default='96,128,192')
    args = ap.parse_args()
    U.ensure_dirs()
    tiles = load_highrisk_tiles(U.MECH_RESULT)
    sizes = [int(s) for s in args.sizes.split(',')]
    rng = random.Random(20260520)
    rows = []
    for tile in tiles:
        img_dir = REPO / 'vis' / tile / 'dataset' / 'images'
        imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        if not imgs:
            continue
        img_path = imgs[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = gt_hboxes(tile)
        sv = 0.0
        summ = U.read_csv(U.MECH_RESULT / 'ftable_01_attractor_atlas_tile_summary.csv')
        for r in summ:
            if r['tile_id'] == tile:
                sv = float(r.get('baseline_final_sv', 0))
                break
        n_ok = 0
        for _ in range(args.patches_per_tile * 20):
            if n_ok >= args.patches_per_tile:
                break
            ps = rng.choice(sizes)
            if ps >= min(h, w):
                continue
            x0, y0 = rng.randint(0, w - ps), rng.randint(0, h - ps)
            box = np.array([x0, y0, x0 + ps, y0 + ps], dtype=np.float32)
            max_iou = max((iou(box, g) for g in gt), default=0.0)
            if max_iou > 0.05:
                continue
            patch = img[y0:y0 + ps, x0:x0 + ps]
            out = U.PATCH_BANK / f'{tile}_s{ps}_{n_ok:03d}.png'
            cv2.imwrite(str(out), patch)
            rows.append(dict(
                source_tile=tile,
                patch_path=str(out),
                bbox_xyxy=f'{x0},{y0},{x0+ps},{y0+ps}',
                max_iou_to_gt=f'{max_iou:.4f}',
                source_final_sv=f'{sv:.4f}',
                extraction_rule='random_non_overlap_bg',
            ))
            n_ok += 1
    out_csv = args.result_dir / 'ftable_hard_negative_patch_bank.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'source_tile', 'patch_path', 'bbox_xyxy', 'max_iou_to_gt',
            'source_final_sv', 'extraction_rule'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} patches -> {out_csv}')


if __name__ == '__main__':
    main()
