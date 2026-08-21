#!/usr/bin/env python
"""R2 negative background prototypes on GPU8."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.repair_forward import build_support_tensors, dense_stats_from_logits, final_detection_from_outs
from tools.sv_attractor_repair_gpu89.repair_modules import NegativePrototypeBank, RepairStack
from tools.verify_sv_attractor_gpu89 import common as verify


def check_online_text_encoder(ctx) -> bool:
    """OpenRSD uses offline support pkl; no online text encoder in inference path."""
    return False


def train_prototypes(ctx, k: int, iters: int, lr: float):
    """Train negative prototypes in **logit space** (15-d) — offline text/visual 256-d mining deferred."""
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    dim = C.NUM_CLASSES
    bank = NegativePrototypeBank(dim, k).to(device)
    opt = torch.optim.AdamW(bank.parameters(), lr=lr)
    splits = C.build_splits(ctx)
    stems = splits['train_calib'][:24 if ctx.mode == 'smoke' else 256]
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)

    for step in range(iters):
        opt.zero_grad()
        stem = stems[step % len(stems)]
        img_p = C.image_path_for_stem(stem)
        if not img_p:
            continue
        import cv2
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        img = cv2.resize(img, (1024, 1024))
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        mean = torch.tensor([103.53, 116.28, 123.675], device=device).view(1, 3, 1, 1)
        std = torch.tensor([57.375, 57.12, 58.395], device=device).view(1, 3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            x = model.prompt_extract_feats(t)
            outs = model.bbox_head(
                x, sf, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
                num_in_classes=C.NUM_CLASSES, align_style='labelled',
                support_type='visual', text_cls_scale=0.0)
        flat = outs[0][0].permute(0, 2, 3, 1).reshape(-1, C.NUM_CLASSES)
        scores = flat.sigmoid()
        sv = scores[:, C.SMALL]
        hard = sv > scores.quantile(0.9)
        if hard.sum() < 10:
            continue
        vis_proxy = flat[hard]
        neg_logit = bank.neg_logits(vis_proxy)
        loss_bg = torch.relu(scores[hard, C.SMALL] - neg_logit + 0.3).mean()
        loss_div = bank.diversity_loss()
        loss = loss_bg + 0.1 * loss_div
        loss.backward()
        opt.step()

    ckpt = ctx.ckpt_dir() / f'r2_negative_prototypes_K{k}_best.pth'
    torch.save(bank.state_dict(), ckpt)
    return bank, ckpt


def eval_bank(ctx, model, bargs, device, det_support, name2id, bank, tag):
    rows = []
    rep = RepairStack()
    rep.neg_bank = bank
    rep.sv_idx = C.SMALL
    image_dir = verify.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
    lb = SimpleNamespace(**vars(bargs))
    lb.image_dir = str(image_dir)
    loader = probe_base.build_dataloader(lb, ctx.angles)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    with torch.no_grad():
        for data_info in loader:
            angle = probe_base.angle_from_name(data_info['data_samples'][0].img_path)
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            outs_all = model.bbox_head(
                x, sf, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
                num_in_classes=C.NUM_CLASSES, align_style='labelled',
                support_type='visual', text_cls_scale=0.0)
            cls_adj = []
            for cs in outs_all[0]:
                b, c, h, w = cs.shape
                flat = cs.permute(0, 2, 3, 1).reshape(-1, c)
                flat = rep.apply_logits(flat, flat)
                cls_adj.append(flat.reshape(b, h, w, c).permute(0, 3, 1, 2))
            outs = (tuple(cls_adj), outs_all[1], outs_all[2])
            dst = dense_stats_from_logits(cls_adj, C.SMALL)
            fin = final_detection_from_outs(model, outs, metas)
            rows.append(dict(
                variant=tag, angle=angle,
                dense_top1_sv_ratio=dst['dense_top1_sv_ratio'],
                final_sv_ratio=fin['final_sv_ratio'],
                detection_total=fin['detection_total'],
                lambda_neg=float(bank.lambda_neg.item()),
            ))
    return rows


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')
    fres = C.fres_for_mode(ctx, 'fres_03_negative_background_prototypes.md',
                            'fres_03_full_negative_background_visual_prototypes.md')
    iters = min(ctx.iters, 300) if ctx.mode == 'smoke' else ctx.iters
    text_ok = check_online_text_encoder(ctx)
    all_rows = []
    with open(fres, 'w') as f:
        C.write_fres_header(f, 'R2 Negative Background Prototypes', ctx)
        f.write(f'- R2A online text encoder available: `{text_ok}`\n')
        if not text_ok:
            f.write('- R2A skipped: offline support pkl only (per spec).\n\n')
        f.write('## R2B train background prototypes\n\n')

    logit_ks = [8] if ctx.mode == 'smoke' else []
    for k in logit_ks:
        bank, ckpt = train_prototypes(ctx, k, iters, ctx.lr)
        bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
        rows = eval_bank(ctx, model, bargs, device, det_support, name2id, bank, f'R2_logit_K{k}')
        all_rows.extend(rows)
        ctx.progress['best_r2_ckpt'] = str(ckpt)
        ctx.save_progress()

    visual_note = 'smoke: logit-space only; full runs R2B_visual_proto'
    if ctx.mode == 'full':
        from tools.sv_attractor_repair_gpu89 import exp_r2_visual
        vis_res = exp_r2_visual.run_full_visual_proto(ctx, [4, 8, 16, 32])
        visual_note = str(vis_res)
        if vis_res.get('ckpts'):
            first = list(vis_res['ckpts'].values())[0]
            ctx.progress['best_r2_visual_ckpt'] = first
            ctx.save_progress()

    C.write_csv(ctx.tables_dir() / 'ftable_r2_negative_prototypes_train.csv', all_rows)
    C.write_csv(ctx.tables_dir() / 'ftable_r2_negative_text_prompts.csv',
                [dict(note='skipped', reason='no_online_text_encoder')])

    with open(fres, 'a') as f:
        f.write(f'- mean final_sv (R2 logit-space smoke): `{C.mean_key(all_rows, "final_sv_ratio"):.4f}`\n')
        f.write(f'- R2B visual (full only): `{visual_note}`\n')
        f.write(f'- csv: `{ctx.tables_dir() / "ftable_r2_negative_prototypes_train.csv"}`\n')

    ctx.mark('r2_negative', 'OK')
    return dict(status='OK', fres=str(fres))
