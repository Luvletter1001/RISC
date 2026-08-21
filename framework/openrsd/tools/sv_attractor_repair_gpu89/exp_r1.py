#!/usr/bin/env python
"""R1 class-wise calibration on GPU8."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.full_metrics import full_method_report
from tools.sv_attractor_repair_gpu89.repair_forward import (
    build_support_tensors, dense_stats_from_logits, final_detection_from_outs)
from tools.sv_attractor_repair_gpu89.repair_modules import ClassWiseCalibration, RepairStack
from tools.verify_sv_attractor_gpu89 import common as verify


def _eval_repair(ctx, model, bargs, device, det_support, name2id, repair, tag, angles):
    rows = []
    image_dir = verify.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
    lb = SimpleNamespace(**vars(bargs))
    lb.image_dir = str(image_dir)
    lb.out_dir = str(ctx.work_dir / '_tmp')
    loader = probe_base.build_dataloader(lb, angles)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', bargs.support_shot)
    repair.sv_idx = C.SMALL
    with torch.no_grad():
        for data_info in loader:
            angle = probe_base.angle_from_name(data_info['data_samples'][0].img_path)
            if angle not in angles:
                continue
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in data['data_samples']]
            outs_all = model.bbox_head(
                x, sf, sl, sl, support_shot=bargs.support_shot,
                num_classes=C.NUM_CLASSES, num_in_classes=C.NUM_CLASSES,
                align_style='labelled', support_type='visual', text_cls_scale=0.0)
            cls_adj = []
            for cs in outs_all[0]:
                b, c, h, w = cs.shape
                flat = cs.permute(0, 2, 3, 1).reshape(-1, c)
                flat = repair.apply_logits(flat)
                cls_adj.append(flat.reshape(b, h, w, c).permute(0, 3, 1, 2))
            outs = (tuple(cls_adj), outs_all[1], outs_all[2])
            dst = dense_stats_from_logits(cls_adj, C.SMALL)
            fin = final_detection_from_outs(model, outs, metas)
            hist = C.parse_class_histogram(fin['class_histogram'])
            rows.append(dict(
                variant=tag, angle=angle,
                dense_top1_sv_ratio=dst['dense_top1_sv_ratio'],
                final_sv_ratio=fin['final_sv_ratio'],
                final_lv_ratio=fin['final_lv_ratio'],
                final_ship_ratio=float(hist.get('ship', 0) / max(fin['detection_total'], 1)),
                detection_total=fin['detection_total'],
                mean_score=fin['mean_score'],
                sv_bias=float(repair.calibration.bias[C.SMALL].item()) if repair.calibration else repair.grid_sv_bias,
            ))
    return rows


def run_grid(ctx):
    rows = []
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    biases = C.R1_GRID_BIAS_FULL if ctx.mode == 'full' else C.R1_GRID_BIAS
    for bias in biases:
        rep = RepairStack()
        rep.sv_idx = C.SMALL
        rep.grid_sv_bias = bias
        rows.extend(_eval_repair(ctx, model, bargs, device, det_support, name2id, rep,
                               f'R1A_bias_{bias:+.1f}', ctx.angles))
    name = 'ftable_r1_full_grid_bias.csv' if ctx.mode == 'full' else 'ftable_r1_grid_bias.csv'
    C.write_csv(ctx.tables_dir() / name, rows)
    return rows


def train_bias(ctx, train_temperature: bool, tag: str, iters: int, lr: float):
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    calib = ClassWiseCalibration(C.NUM_CLASSES, train_temperature).to(device)
    opt = torch.optim.AdamW(calib.parameters(), lr=lr, weight_decay=1e-4)
    splits = C.build_splits(ctx)
    n_stems = 512 if ctx.mode == 'full' else 32
    stems = splits['train_calib'][:n_stems]
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', bargs.support_shot)

    for step in range(iters):
        opt.zero_grad()
        stem = stems[step % len(stems)]
        img_p = C.image_path_for_stem(stem)
        if img_p is None:
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
            outs_all = model.bbox_head(
                x, sf, sl, sl, support_shot=8, num_classes=C.NUM_CLASSES,
                num_in_classes=C.NUM_CLASSES, align_style='labelled',
                support_type='visual', text_cls_scale=0.0)
        cls_scores = outs_all[0][0]
        b, c, h, w = cls_scores.shape
        flat = cls_scores.permute(0, 2, 3, 1).reshape(-1, c)
        flat = calib(flat)
        scores = flat.sigmoid()
        sv = scores[:, C.SMALL]
        loss_bg = torch.relu(sv - scores.topk(2, dim=1).values[:, 1] + 0.2).mean()
        loss_reg = (calib.bias.pow(2).sum() + ((calib.tau() - 1) ** 2).sum())
        loss = loss_bg + 0.01 * loss_reg
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f'{tag} step={step} loss={loss.item():.4f}')

    rep = RepairStack()
    rep.calibration = calib
    rep.sv_idx = C.SMALL
    rows = _eval_repair(ctx, model, bargs, device, det_support, name2id, rep, tag, ctx.angles)
    ckpt_name = f'{tag}_full_best.pth' if ctx.mode == 'full' else f'{tag}_best.pth'
    ckpt = ctx.ckpt_dir() / ckpt_name
    torch.save(calib.state_dict(), ckpt)
    return rows, str(ckpt)


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')
    fres = C.fres_for_mode(ctx, 'fres_02_classwise_calibration.md',
                            'fres_02_full_classwise_calibration.md')
    iters = ctx.iters if ctx.mode == 'full' else min(ctx.iters, 300)
    grid_rows = run_grid(ctx)
    tag_b = 'R1B_train_class_bias_full' if ctx.mode == 'full' else 'R1B_train_class_bias'
    tag_c = 'R1C_train_bias_temperature_full' if ctx.mode == 'full' else 'R1C_train_bias_temperature'
    r1b, ckpt_b = train_bias(ctx, False, tag_b, iters, ctx.lr)
    r1c, ckpt_c = train_bias(ctx, True, tag_c, iters, ctx.lr * 0.3)
    tb = 'ftable_r1_full_train_bias.csv' if ctx.mode == 'full' else 'ftable_r1_train_bias.csv'
    tc = 'ftable_r1_full_bias_temperature.csv' if ctx.mode == 'full' else 'ftable_r1_train_bias_temperature.csv'
    C.write_csv(ctx.tables_dir() / tb, r1b)
    C.write_csv(ctx.tables_dir() / tc, r1c)

    base_sv = C.mean_key(grid_rows, 'final_sv_ratio')
    best = min(grid_rows, key=lambda r: float(r['final_sv_ratio']))
    ap_rows = []
    if ctx.mode == 'full':
        bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
        ctx.progress['best_r1_ckpt'] = ckpt_c
        ctx.save_progress()
        for label in ['best_R1_grid', 'best_R1B', 'best_R1C']:
            rep = RepairStack()
            rep.sv_idx = C.SMALL
            if label == 'best_R1_grid':
                rep.grid_sv_bias = float(best.get('sv_bias', -1.0))
            else:
                calib = ClassWiseCalibration(C.NUM_CLASSES, label.endswith('C')).to(device)
                p = ckpt_c if label.endswith('C') else ckpt_b
                calib.load_state_dict(torch.load(p, map_location=device))
                rep.calibration = calib
            ctx.progress['best_r1_ckpt'] = ckpt_c if label.endswith('C') else ckpt_b
            ap_rows.append(full_method_report(
                ctx, model, bargs, device, det_support, name2id, 'best_R1'))
        C.write_csv(ctx.tables_dir() / 'ftable_r1_full_ap_summary.csv', ap_rows)

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'R1 Class-wise Calibration (full)' if ctx.mode == 'full' else 'R1', ctx)
        f.write(f'- grid mean final_sv: `{base_sv:.4f}`\n')
        f.write(f'- best grid bias: `{best.get("sv_bias")}` final_sv `{float(best["final_sv_ratio"]):.4f}`\n')
        f.write(f'- R1B ckpt: `{ckpt_b}`\n')
        f.write(f'- R1C ckpt: `{ckpt_c}`\n')
        f.write(f'- R1B mean final_sv: `{C.mean_key(r1b, "final_sv_ratio"):.4f}`\n')
        f.write(f'- R1C mean final_sv: `{C.mean_key(r1c, "final_sv_ratio"):.4f}`\n')
        if ap_rows:
            f.write('\n## Heldout AP summary\n\n')
            f.write('| label | AP50 | sv_AP50 | P0148_final_sv |\n|---|---:|---:|---:|\n')
            for r in ap_rows:
                f.write(f"| {r.get('method')} | {r.get('ap50_overall', float('nan')):.4f} | "
                        f"{r.get('sv_ap50', float('nan')):.4f} | {r.get('P0148_final_sv', float('nan')):.4f} |\n")

    ctx.mark('r1_calib', 'OK', best_bias=best.get('sv_bias'))
    ctx.progress['best_r1_ckpt'] = ckpt_c
    ctx.save_progress()
    return dict(status='OK', fres=str(fres))
