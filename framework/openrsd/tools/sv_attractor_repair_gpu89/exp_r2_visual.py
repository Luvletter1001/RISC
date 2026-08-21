#!/usr/bin/env python
"""R2B: visual/dense negative prototypes for full mode (outside-GT hard bg mining)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.eval_ap import forward_with_repair
from tools.sv_attractor_repair_gpu89.repair_forward import build_support_tensors
from tools.sv_attractor_repair_gpu89.repair_modules import NegativePrototypeBank, RepairStack


def point_in_poly(px: float, py: float, poly8) -> bool:
    pts = np.asarray(poly8, dtype=np.float32).reshape(-1, 2)
    return cv2.pointPolygonTest(pts, (px, py), False) >= 0


def max_iou_point_gt(px: float, py: float, polys) -> float:
    best = 0.0
    for poly in polys:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        ix1, iy1 = max(x1, px - 4), max(y1, py - 4)
        ix2, iy2 = min(x2, px + 4), min(y2, py + 4)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        best = max(best, inter / area)
    return best


def grid_centers(h: int, w: int, gh: int, gw: int):
    for r in range(gh):
        for c in range(gw):
            yield r, c, (c + 0.5) / gw * w, (r + 0.5) / gh * h


def mine_outside_gt_features(
        ctx: C.RepairContext,
        model,
        device,
        det_support,
        name2id,
        max_stems: int = 64,
        topk_per_image: int = 128,
) -> tuple:
    """Collect hard background pred_embeds outside all GT polygons."""
    splits = C.build_splits(ctx)
    stems = splits['train_calib'][:max_stems]
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    rep = RepairStack()
    feats_all = []
    meta_rows = []

    mean = torch.tensor([103.53, 116.28, 123.675], device=device).view(1, 3, 1, 1)
    std = torch.tensor([57.375, 57.12, 58.395], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for stem in stems:
            img_p = C.image_path_for_stem(stem)
            ann_p = C.SS_TRAIN_ROOT / 'annfiles' / f'{stem}.txt'
            if not img_p or not ann_p.exists():
                continue
            polys = []
            for line in ann_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 8:
                    polys.append([float(x) for x in parts[:8]])
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            img = cv2.resize(img, (1024, 1024))
            t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            t = (t - mean) / std
            x = model.prompt_extract_feats(t)
            outs_all = model.bbox_head(
                x, sf, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
                num_in_classes=C.NUM_CLASSES, align_style='labelled',
                support_type='visual', text_cls_scale=0.0)
            if len(outs_all) < 4:
                continue
            pred_embeds = outs_all[3]
            cls_scores = outs_all[0]
            pe = pred_embeds[0]
            cs = cls_scores[0]
            _, d, gh, gw = pe.shape
            scores = cs.sigmoid().permute(0, 2, 3, 1).reshape(-1, C.NUM_CLASSES)
            emb = pe.permute(0, 2, 3, 1).reshape(-1, d)
            sv = scores[:, C.SMALL]
            outside = []
            for r, c, px, py in grid_centers(1024, 1024, gh, gw):
                idx = r * gw + c
                if idx >= emb.shape[0]:
                    break
                if any(point_in_poly(px, py, p) for p in polys):
                    continue
                if polys and max_iou_point_gt(px, py, polys) >= 0.1:
                    continue
                outside.append(idx)
            if not outside:
                continue
            outside = np.asarray(outside, dtype=np.int64)
            sv_out = sv[outside]
            order = sv_out.argsort(descending=True)[:topk_per_image]
            pick = outside[order.cpu().numpy()]
            feats_all.append(emb[pick])
            for rank, idx in enumerate(pick[:16]):
                meta_rows.append(dict(
                    stem=stem, grid_idx=int(idx),
                    sv_score=float(sv[idx].item()),
                    rank=rank,
                    region='outside_gt',
                ))
    if not feats_all:
        return torch.zeros(0, 256, device=device), meta_rows
    return torch.cat(feats_all, dim=0), meta_rows


def train_visual_bank(ctx: C.RepairContext, k: int, features: torch.Tensor,
                      iters: int = 200, lr: float = 1e-2):
    device = features.device
    dim = features.shape[1] if features.numel() else 256
    bank = NegativePrototypeBank(dim, k).to(device)
    if features.shape[0] < k:
        return bank, None
    opt = torch.optim.AdamW(bank.parameters(), lr=lr)
    for step in range(iters):
        opt.zero_grad()
        idx = torch.randint(0, features.shape[0], (min(256, features.shape[0]),), device=device)
        batch = features[idx]
        neg = bank.neg_logits(batch)
        loss = torch.relu(neg - 0.2).mean() + 0.1 * bank.diversity_loss()
        loss.backward()
        opt.step()
    ckpt = ctx.ckpt_dir() / f'R2B_visual_proto_K{k}_best.pth'
    if ctx.mode == 'full':
        ckpt = ctx.ckpt_dir() / f'R2B_visual_proto_K{k}_best.pth'
    torch.save(bank.state_dict(), ckpt)
    return bank, ckpt


def save_activation_examples(ctx, bank, features, out_csv: Path, n: int = 32):
    if features.shape[0] == 0:
        C.write_csv(out_csv, [dict(note='no features')])
        return
    with torch.no_grad():
        proto = F.normalize(bank.prototypes, dim=-1)
        feat = F.normalize(features[:n], dim=-1)
        act = (feat @ proto.t()).max(dim=-1).values
    rows = [dict(example_id=i, max_proto_activation=float(act[i].item()))
            for i in range(min(n, feat.shape[0]))]
    C.write_csv(out_csv, rows)


def run_full_visual_proto(ctx: C.RepairContext, k_list=None) -> dict:
    """Train R2B visual prototypes (full mode only)."""
    k_list = k_list or [4, 8, 16]
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    max_stems = min(len(C.build_splits(ctx)['train_calib']), 2000) if ctx.mode == 'full' else 32
    feats, meta = mine_outside_gt_features(
        ctx, model, device, det_support, name2id, max_stems=max_stems)
    C.write_csv(ctx.tables_dir() / 'ftable_r2b_mining_meta.csv', meta)
    results = {}
    for k in k_list:
        bank, ckpt = train_visual_bank(ctx, k, feats, iters=ctx.iters, lr=ctx.lr)
        if ckpt:
            results[f'K{k}'] = str(ckpt)
            save_activation_examples(
                ctx, bank, feats,
                ctx.tables_dir() / f'ftable_r2b_activation_K{k}.csv')
    ctx.progress['r2_visual_ckpts'] = results
    ctx.save_progress()
    return dict(status='OK', ckpts=results, n_features=int(feats.shape[0]))


def is_implemented() -> bool:
    return True
