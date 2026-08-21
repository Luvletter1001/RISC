#!/usr/bin/env python
"""GPU8: background / object counterfactual inputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U
from tools.sv_attractor_repair_gpu89.common import parse_dota_txt, poly_to_xyxy

CSV = U.RESULT_MD / 'ftable_03_background_counterfactual.csv'
INDEX_JSON = U.RESULT_MD / 'fmeta_03_counterfactual_images.json'
SCHEMA = [
    'tile_id', 'risk_group', 'angle', 'condition', 'image_path',
    'dense_top1_sv_ratio', 'dense_sv_margin_vs_runnerup', 'final_sv_ratio',
    'det_count', 'objectness_mean', 'background_to_sv_cosine_mean', 'notes',
]

CONDITIONS = [
    'A_original', 'B_gt_object_masked', 'C_background_only', 'D_object_only',
    'E_highrisk_bg_paste', 'F_lowrisk_bg_paste', 'G_random_patch_shuffle',
]


def load_image(tile_id: str, angle: int, repo: Path) -> tuple:
    _, src = U.resolve_tile_source(repo, tile_id, angle)
    for ext in ('.jpg', '.png'):
        p = U.vis_tile_dir(repo, tile_id)
        if p:
            ip = p / f'{tile_id}_rot{angle:03d}{ext}'
            if ip.exists():
                img = cv2.imread(str(ip))
                if img is not None:
                    return img, ip
    asv = U.angle_sweep_dir(repo, angle) / 'images'
    for ext in ('.png', '.jpg'):
        ip = asv / f'{tile_id}{ext}'
        if ip.exists():
            img = cv2.imread(str(ip))
            if img is not None:
                return img, ip
    raise FileNotFoundError(tile_id)


def gt_boxes(repo: Path, tile_id: str, angle: int) -> list:
    boxes = []
    ann_pkl = repo / 'vis' / tile_id / 'dataset' / 'annfiles' / f'{tile_id}_rot{angle:03d}.pkl'
    if ann_pkl.exists():
        from commonlibs.common_tools import pklload
        ann = pklload(str(ann_pkl))
        for poly in ann.get('polys', []):
            boxes.append(poly_to_xyxy(poly))
        return boxes
    ann_txt = U.angle_sweep_dir(repo, angle) / 'annfiles' / f'{tile_id}.txt'
    if ann_txt.exists():
        for b in parse_dota_txt(ann_txt):
            boxes.append(poly_to_xyxy(b['poly']))
    return boxes


def mask_boxes(img: np.ndarray, boxes: list, mode: str) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    if not boxes:
        if mode == 'background_only':
            return np.zeros_like(img)
        return img
    mask_obj = np.zeros((h, w), np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(mask_obj, (x1, y1), (x2, y2), 255, -1)
    bg_mean = out[mask_obj == 0].mean(axis=0) if (mask_obj == 0).any() else out.mean(axis=(0, 1))
    if mode == 'B_gt_object_masked':
        out[mask_obj > 0] = bg_mean
    elif mode == 'C_background_only':
        out[mask_obj > 0] = bg_mean
        out = out  # keep bg
    elif mode == 'D_object_only':
        out[mask_obj == 0] = bg_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def patch_shuffle(img: np.ndarray, patch: int = 128) -> np.ndarray:
    h, w = img.shape[:2]
    patches, coords = [], []
    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            patches.append(img[y:y + patch, x:x + patch].copy())
            coords.append((y, x))
    rng = np.random.default_rng(U.MechContext.seed)
    order = rng.permutation(len(patches))
    out = img.copy()
    for (y, x), idx in zip(coords, order):
        out[y:y + patch, x:x + patch] = patches[idx]
    return out


def save_cf_image(img: np.ndarray, tile_id: str, angle: int, condition: str, repo: Path) -> Path:
    """Save under temp dataset layout so probe dataloader finds the stem."""
    import shutil
    root = U.COUNTERFACTUAL_ROOT / tile_id / 'dataset'
    img_dir = root / 'images'
    ann_dir = root / 'annfiles'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{tile_id}_rot{angle:03d}_{condition}'
    p = img_dir / f'{stem}.png'
    cv2.imwrite(str(p), img)
    for src in [
        repo / 'vis' / tile_id / 'dataset' / 'annfiles' / f'{tile_id}_rot{angle:03d}.pkl',
        U.angle_sweep_dir(repo, angle) / 'annfiles' / f'{tile_id}.pkl',
    ]:
        if src.exists():
            shutil.copy2(src, ann_dir / f'{stem}.pkl')
            break
    return p


def paste_background(dst: np.ndarray, src_bg: np.ndarray, boxes_dst: list) -> np.ndarray:
    """Paste src background into non-GT regions of dst."""
    out = dst.copy()
    h, w = out.shape[:2]
    mask_keep = np.ones((h, w), np.uint8) * 255
    for box in boxes_dst:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(mask_keep, (x1, y1), (x2, y2), 0, -1)
    src_r = cv2.resize(src_bg, (w, h))
    out[mask_keep > 0] = src_r[mask_keep > 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=8)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    ctx = U.MechContext(gpu=args.gpu)
    ctx.work_dir = U.WORK_ROOT / 'gpu8_bg_cf'
    U.setup_logging(ctx.work_dir, 'bg_cf')

    hi_csv = U.read_csv(U.RESULT_MD / 'ftable_01_attractor_highrisk_tiles.csv')
    summ = U.read_csv(U.RESULT_MD / 'ftable_01_attractor_atlas_tile_summary.csv')
    if hi_csv:
        hi = [r['tile_id'] for r in hi_csv[:5]]
        lo = [r['tile_id'] for r in sorted(summ, key=lambda x: float(x.get('baseline_final_sv', 0)))[:5]]
    else:
        hi, lo = [U.DEFAULT_TILE], []

    tiles = [U.DEFAULT_TILE] + hi + lo
    tiles = list(dict.fromkeys(tiles))

    bargs, model, device, det_support, name2id, _, _ = U.build_model(ctx)
    index = {}

    ref_hi_img = None
    ref_lo_img = None
    try:
        ref_hi_img, _ = load_image(hi[0] if hi else U.DEFAULT_TILE, 0, ctx.repo_root)
        ref_lo_img, _ = load_image(lo[0] if lo else U.DEFAULT_TILE, 0, ctx.repo_root)
    except Exception:
        pass

    for tid in tiles:
        grp = 'high' if tid in hi else ('low' if tid in lo else 'medium')
        for angle in U.DECOMP_ANGLES:
            try:
                img, _ = load_image(tid, angle, ctx.repo_root)
            except FileNotFoundError as exc:
                U.log.warning('%s', exc)
                continue
            boxes = gt_boxes(ctx.repo_root, tid, angle)
            variants = {
                'A_original': img,
                'B_gt_object_masked': mask_boxes(img, boxes, 'B_gt_object_masked'),
                'C_background_only': mask_boxes(img, boxes, 'C_background_only'),
                'D_object_only': mask_boxes(img, boxes, 'D_object_only'),
                'G_random_patch_shuffle': patch_shuffle(img),
            }
            if ref_hi_img is not None:
                variants['E_highrisk_bg_paste'] = paste_background(img, ref_hi_img, boxes)
            if ref_lo_img is not None:
                variants['F_lowrisk_bg_paste'] = paste_background(img, ref_lo_img, boxes)

            for cond, im in variants.items():
                key = f'{tid}|{angle}|{cond}'
                if U.already_done(CSV, tid, angle, cond, extra=cond):
                    continue
                ip = save_cf_image(im, tid, angle, cond, ctx.repo_root)
                index[key] = str(ip)
                row = U.eval_one(
                    ctx, model, bargs, device, det_support, name2id,
                    tid, angle, 'baseline',
                    image_override=U.COUNTERFACTUAL_ROOT / tid / 'dataset' / 'images' / ip.name,
                    extra_tag=cond)
                row['condition'] = cond
                row['risk_group'] = grp
                row['image_path'] = str(ip)
                U.append_csv(CSV, row, SCHEMA)

    INDEX_JSON.write_text(json.dumps(index, indent=2), encoding='utf-8')
    write_bg_report(U.read_csv(CSV))
    print('done', CSV)


def write_bg_report(rows: list):
    import statistics
    lines = ['# Background Counterfactual (Exp3)', '']

    def mean_cond(sub, cond, key):
        vals = []
        for r in sub:
            if r.get('condition') != cond:
                continue
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                pass
        return statistics.mean(vals) if vals else float('nan')

    for grp in ['high', 'low']:
        sub = [r for r in rows if r.get('risk_group') == grp]
        if not sub:
            continue
        lines.append(f'## {grp}-risk')
        lines.append('| condition | final_sv | dense_sv |')
        lines.append('|---|---:|---:|')
        for cond in CONDITIONS:
            fv, dv = mean_cond(sub, cond, 'final_sv_ratio'), mean_cond(sub, cond, 'dense_top1_sv_ratio')
            if fv == fv:
                lines.append(f"| {cond} | {fv:.3f} | {dv:.3f} |")
    c = mean_cond(rows, 'C_background_only', 'final_sv_ratio')
    a = mean_cond(rows, 'A_original', 'final_sv_ratio')
    lines += ['', '## H2 判定', '']
    if c > 0.3 and c > 0.5 * a:
        lines.append('- **H2 PARTIALLY_SUPPORTED**：去除 object 后背景仍保留显著 SV 偏置。')
    elif c < 0.2:
        lines.append('- **H2 NOT_SUPPORTED**：background-only 偏置消失。')
    else:
        lines.append('- **H2 INCONCLUSIVE**')
    (U.RESULT_MD / 'fres_03_background_counterfactual.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
