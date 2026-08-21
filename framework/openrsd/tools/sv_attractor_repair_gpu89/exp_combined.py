#!/usr/bin/env python
"""Combined repair ablation."""
from __future__ import annotations

import torch
from types import SimpleNamespace

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.repair_logits import apply_repair_logits
from tools.sv_attractor_repair_gpu89.repair_forward import build_support_tensors, dense_stats_from_logits, final_detection_from_outs
from tools.sv_attractor_repair_gpu89.repair_modules import ClassWiseCalibration, NegativePrototypeBank, RepairStack
from tools.sv_attractor_repair_gpu89.exp_r3 import apply_fixed_alpha_support
from tools.sv_attractor_repair_gpu89.repair_forward import sv_support_direction
from tools.verify_sv_attractor_gpu89 import common as verify


def load_repair_stack(ctx, device, tag: str) -> RepairStack:
    rep = RepairStack()
    rep.sv_idx = C.SMALL
    from pathlib import Path
    if 'R1' in tag or 'C1' in tag or 'C4' in tag or 'C5' in tag or 'C7' in tag:
        p = ctx.progress.get('best_r1_ckpt') or str(ctx.ckpt_dir() / 'R1C_train_bias_temperature_best.pth')
        if Path(p).exists():
            calib = ClassWiseCalibration(C.NUM_CLASSES, True).to(device)
            calib.load_state_dict(torch.load(p, map_location=device))
            rep.calibration = calib
    if ('R2' in tag and 'R2B' not in tag and 'failed' not in tag.lower()) or tag == 'C2_best_R2':
        p = ctx.progress.get('best_r2_ckpt')
        if p and Path(p).exists():
            sd = torch.load(p, map_location=device)
            dim = int(sd['prototypes'].shape[1])
            k = int(sd['prototypes'].shape[0])
            bank = NegativePrototypeBank(dim, k).to(device)
            bank.load_state_dict(sd)
            rep.neg_bank = bank
    if 'R2B' in tag or (ctx.progress.get('best_r2_visual_ckpt') and 'R2' in tag):
        vp = ctx.progress.get('best_r2_visual_ckpt')
        if vp and Path(vp).exists():
            sd = torch.load(vp, map_location=device)
            dim = int(sd['prototypes'].shape[1])
            k = int(sd['prototypes'].shape[0])
            bank = NegativePrototypeBank(dim, k).to(device)
            bank.load_state_dict(sd)
            rep.neg_bank = bank
    return rep


def eval_method(ctx, model, bargs, device, det_support, name2id, method, sf_mod=None):
    rows = []
    image_dir = verify.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
    lb = SimpleNamespace(**vars(bargs))
    lb.image_dir = str(image_dir)
    loader = probe_base.build_dataloader(lb, ctx.angles)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', 8)
    if sf_mod:
        sv = sv_support_direction(model, sf, sl, C.SMALL)
        sf = sf_mod(sf, sl, sv)
    rep = RepairStack()
    if method != 'C0_baseline':
        rep = load_repair_stack(ctx, device, method)
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
            cls_adj = apply_repair_logits(outs_all, rep)
            outs = (tuple(cls_adj), outs_all[1], outs_all[2])
            dst = dense_stats_from_logits(cls_adj, C.SMALL)
            fin = final_detection_from_outs(model, outs, metas)
            rows.append(dict(
                method=method, angle=angle,
                P0148_final_sv=fin['final_sv_ratio'],
                dense_sv=dst['dense_top1_sv_ratio'],
                detection_total=fin['detection_total'],
                final_lv=fin['final_lv_ratio'],
            ))
    return rows


def run(ctx: C.RepairContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED')
    fres = C.fres_for_mode(ctx, 'fres_05_combined_repair_ablation.md',
                            'fres_05_full_combined_repair_ablation.md')
    bargs, model, device, det_support, name2id, _, _, _ = C.build_model_ctx(ctx)
    r2b_ok = bool(ctx.progress.get('best_r2_visual_ckpt'))
    r2_tag = 'C2_best_R2B_visual' if r2b_ok else 'C2_R2_failed'
    methods = [
        'C0_baseline', 'C1_best_R1', r2_tag, 'C3_best_R3',
        'C4_R1+R2B' if r2b_ok else 'C4_R1+R2_failed',
        'C5_R1+R3',
        'C6_R2B+R3' if r2b_ok else 'C6_R2+R3',
        'C7_R1+R2B+R3' if r2b_ok else 'C7_R1+R2+R3',
    ] if ctx.mode == 'full' else [
        'C0_baseline', 'C1_best_R1', 'C2_best_R2', 'C3_best_R3',
        'C4_R1+R2', 'C5_R1+R3', 'C6_R2+R3', 'C7_R1+R2+R3',
    ]
    all_rows = []
    sf_r3 = lambda sf, sl, sv: apply_fixed_alpha_support(sf, sl, sv, 0.5)
    for m in methods:
        sf_mod = sf_r3 if 'R3' in m and m != 'C1_best_R1' and m != 'C2_best_R2' else None
        if m == 'C3_best_R3':
            sf_mod = sf_r3
        all_rows.extend(eval_method(ctx, model, bargs, device, det_support, name2id, m, sf_mod))

    csv_name = 'ftable_combined_repair_full.csv' if ctx.mode == 'full' else 'ftable_combined_repair_ablation.csv'
    C.write_csv(ctx.tables_dir() / csv_name, all_rows)
    base = C.mean_key([r for r in all_rows if r['method'] == 'C0_baseline'], 'P0148_final_sv')
    best = min(all_rows, key=lambda r: float(r['P0148_final_sv'])) if all_rows else {}

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Combined Repair Ablation', ctx)
        f.write(f'- baseline mean P0148 final_sv: `{base:.4f}`\n')
        f.write(f'- best method: `{best.get("method")}` sv=`{float(best.get("P0148_final_sv", 0)):.4f}`\n\n')
        f.write('| method | mean P0148_final_sv | mean dense_sv | det_total |\n')
        f.write('|---|---:|---:|---:|\n')
        for m in methods:
            sub = [r for r in all_rows if r['method'] == m]
            f.write(f"| {m} | {C.mean_key(sub, 'P0148_final_sv'):.4f} | "
                    f"{C.mean_key(sub, 'dense_sv'):.4f} | {C.mean_key(sub, 'detection_total'):.0f} |\n")
        ctx.progress['best_combined_method'] = best.get('method', '')
        ctx.save_progress()

    ctx.mark('combined', 'OK')
    return dict(status='OK', fres=str(fres))
