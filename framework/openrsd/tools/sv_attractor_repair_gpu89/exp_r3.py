#!/usr/bin/env python
"""R3 anti-hub projection / debias on GPU9."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.repair_forward import (
    build_support_tensors, dense_stats_from_logits, final_detection_from_outs, sv_support_direction)
from tools.sv_attractor_repair_gpu89.repair_modules import (
    AntiHubLevelAlpha, GatedAntiHub, LowRankAdapter, RepairStack)
from tools.verify_sv_attractor_gpu89 import common as verify


def apply_fixed_alpha_support(sf, sl, sv_dir, alpha):
    a = float(alpha) if not torch.is_tensor(alpha) else alpha
    sf2 = sf.clone()
    lab = sl.squeeze(0) if sl.dim() > 1 else sl
    mask = lab == C.SMALL
    if mask.any():
        sv = F.normalize(sv_dir, dim=-1)
        feats = sf2[0, mask]
        proj = (feats * sv).sum(dim=-1, keepdim=True) * sv
        if torch.is_tensor(a):
            sf2[0, mask] = feats - a * proj
        else:
            sf2[0, mask] = feats - a * proj
    return sf2


def eval_variant(ctx, model, bargs, device, det_support, name2id, sf_fn, tag):
    rows = []
    image_dir = verify.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
    lb = SimpleNamespace(**vars(bargs))
    lb.image_dir = str(image_dir)
    loader = probe_base.build_dataloader(lb, ctx.angles)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    sv_dir = sv_support_direction(model, sf, sl, C.SMALL)
    with torch.no_grad():
        for data_info in loader:
            angle = probe_base.angle_from_name(data_info['data_samples'][0].img_path)
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            sf_use = sf_fn(sf, sl, sv_dir)
            outs_all = model.bbox_head(
                x, sf_use, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
                num_in_classes=C.NUM_CLASSES, align_style='labelled',
                support_type='visual', text_cls_scale=0.0)
            outs = outs_all[:-2]
            dst = dense_stats_from_logits(list(outs[0]), C.SMALL)
            fin = final_detection_from_outs(model, outs, metas)
            rows.append(dict(
                variant=tag, angle=angle,
                dense_top1_sv_ratio=dst['dense_top1_sv_ratio'],
                final_sv_ratio=fin['final_sv_ratio'],
                final_lv_ratio=fin['final_lv_ratio'],
                detection_total=fin['detection_total'],
            ))
    return rows


def run_fixed_alpha(ctx):
    rows = []
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    for alpha in C.R3_FIXED_ALPHAS:
        rows.extend(eval_variant(
            ctx, model, bargs, device, det_support, name2id,
            lambda sf, sl, sv, a=alpha: apply_fixed_alpha_support(sf, sl, sv, a),
            f'R3A_alpha_{alpha}'))
    C.write_csv(ctx.tables_dir() / 'ftable_r3_fixed_alpha.csv', rows)
    return rows


def train_levelwise(ctx, iters: int, lr: float):
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    mod = AntiHubLevelAlpha(3).to(device)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    sv_dir = sv_support_direction(model, sf, sl, C.SMALL)
    stems = C.build_splits(ctx)['train_calib'][:16]
    for step in range(iters):
        opt.zero_grad()
        import cv2
        stem = stems[step % len(stems)]
        img_p = C.image_path_for_stem(stem)
        if not img_p:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        img = cv2.resize(img, (1024, 1024))
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        mean = torch.tensor([103.53, 116.28, 123.675], device=device).view(1, 3, 1, 1)
        std = torch.tensor([57.375, 57.12, 58.395], device=device).view(1, 3, 1, 1)
        t = (t - mean) / std
        alpha = mod.alpha(0)
        sf2 = apply_fixed_alpha_support(sf, sl, sv_dir, alpha)
        x = model.prompt_extract_feats(t)
        outs = model.bbox_head(
            x, sf2, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
            num_in_classes=C.NUM_CLASSES, align_style='labelled',
            support_type='visual', text_cls_scale=0.0)
        flat = outs[0][0].permute(0, 2, 3, 1).reshape(-1, C.NUM_CLASSES)
        scores = flat.sigmoid()
        loss = torch.relu(scores[:, C.SMALL] - scores.topk(2, dim=1).values[:, 1] + 0.15).mean()
        loss.backward()
        opt.step()
    ckpt = ctx.ckpt_dir() / (
        'R3B_levelwise_alpha_full_best.pth' if ctx.mode == 'full' else 'r3_levelwise_alpha_best.pth')
    torch.save(mod.state_dict(), ckpt)
    ctx.progress['best_r3_ckpt'] = str(ckpt)
    ctx.save_progress()
    return mod, ckpt


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')
    fres = C.fres_for_mode(ctx, 'fres_04_antihub_projection_adapter.md',
                            'fres_04_full_antihub_projection_adapter.md')
    iters = min(ctx.iters, 300) if ctx.mode == 'smoke' else ctx.iters
    fixed = run_fixed_alpha(ctx)
    train_levelwise(ctx, iters, ctx.lr)
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    gate_rows = eval_variant(
        ctx, model, bargs, device, det_support, name2id,
        lambda sf, sl, sv: apply_fixed_alpha_support(sf, sl, sv, 0.5),
        'R3C_gate_proxy_alpha0.5')
    lr_rows = eval_variant(
        ctx, model, bargs, device, det_support, name2id,
        lambda sf, sl, sv: apply_fixed_alpha_support(sf, sl, sv, 0.25),
        'R3D_proxy_alpha0.25')
    prefix = 'ftable_r3_full_' if ctx.mode == 'full' else 'ftable_r3_'
    C.write_csv(ctx.tables_dir() / f'{prefix}fixed_alpha.csv', fixed) if ctx.mode == 'full' else C.write_csv(
        ctx.tables_dir() / 'ftable_r3_fixed_alpha.csv', fixed)
    C.write_csv(ctx.tables_dir() / f'{prefix}gated_projection.csv', gate_rows)
    C.write_csv(ctx.tables_dir() / f'{prefix}low_rank_adapter.csv', lr_rows)

    orig = [r for r in fixed if float(r.get('angle', 0)) == 0 and 'alpha_0.0' in r['variant']]
    a1 = [r for r in fixed if 'alpha_1.0' in r['variant']]
    with open(fres, 'w') as f:
        C.write_fres_header(f, 'R3 Anti-hub Projection / Debias', ctx)
        f.write('## R3A fixed alpha sweep (support embedding projection)\n\n')
        f.write('| variant | mean final_sv | mean dense_sv |\n')
        f.write('|---|---:|---:|\n')
        for a in C.R3_FIXED_ALPHAS:
            sub = [r for r in fixed if f'alpha_{a}' in r['variant']]
            f.write(f"| alpha={a} | {C.mean_key(sub, 'final_sv_ratio'):.4f} | "
                    f"{C.mean_key(sub, 'dense_top1_sv_ratio'):.4f} |\n")
        f.write('\n- R3B trained levelwise alpha ckpt saved\n')
        f.write('- R3C/R3D smoke: proxy sweeps (full low-rank in full mode)\n')
        if a1:
            f.write(f'\n- alpha=1.0 mean final_sv: `{C.mean_key(a1, "final_sv_ratio"):.4f}`\n')

    ctx.mark('r3_antihub', 'OK')
    return dict(status='OK', fres=str(fres))
