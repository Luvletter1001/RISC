#!/usr/bin/env python
import json

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89.repair_forward import count_trainable, freeze_openrsd
from tools.sv_attractor_repair_gpu89.repair_modules import (
    AntiHubLevelAlpha, ClassWiseCalibration, GatedAntiHub, LowRankAdapter, NegativePrototypeBank)


def run(ctx: C.RepairContext) -> dict:
    fres = ctx.fres('fres_00_repair_preflight.md')
    status = 'OK'
    err = ''
    verify_ok = {}
    vdir = ctx.verify_work_dir
    for name, sub in [
        ('dense', 'exp_02_dense/ftable_dense_by_angle.csv'),
        ('intervention', 'exp_03_intervention/ftable_embedding_intervention_final.csv'),
        ('postprocess', 'exp_05_postprocess/ftable_postprocess_ablation.csv'),
        ('bg_gt', 'exp_04_bg_gt/ftable_gt_match_summary.csv'),
        ('tile_angle', 'exp_06_tile_angle/ftable_tile_angle_visual.csv'),
    ]:
        p = vdir / sub
        verify_ok[name] = p.exists()

    splits = C.build_splits(ctx)
    param_counts = {
        'ClassWiseCalibration_bias_only': ClassWiseCalibration(C.NUM_CLASSES, False),
        'ClassWiseCalibration_bias_tau': ClassWiseCalibration(C.NUM_CLASSES, True),
        'NegativePrototypeBank_K8': NegativePrototypeBank(256, 8),
        'AntiHubLevelAlpha': AntiHubLevelAlpha(3),
        'GatedAntiHub': GatedAntiHub(),
        'LowRankAdapter_r4': LowRankAdapter(256, 4),
    }
    trainable = {k: count_trainable(m) for k, m in param_counts.items()}

    model_info = {}
    try:
        bargs, model, device, det_support, name2id, id2name, cfg, frozen = C.build_model_ctx(ctx)
        head = model.bbox_head
        model_info['bbox_head'] = head.__class__.__name__
        model_info['frozen_params'] = frozen
        model_info['rtm_cls_heads'] = [
            f'level{i}:{m.__class__.__name__}' for i, m in enumerate(head.rtm_cls_heads)]
        model_info['total_params'] = sum(p.numel() for p in model.parameters())
        model_info['trainable_after_freeze'] = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as exc:
        status = 'FAILED'
        err = str(exc)
        ctx.set_blocked(err)

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Repair Preflight', ctx, status)
        f.write(f'- git: `{C.git_commit(ctx.repo_root)}`\n')
        for k, v in C.package_versions().items():
            f.write(f'- {k}: `{v}`\n')
        f.write(f'\n- config: `{ctx.config}`\n')
        f.write(f'- checkpoint: `{ctx.checkpoint}`\n')
        f.write(f'- support: `{ctx.support_pkl}`\n')
        f.write(f'- verify_work_dir exists: `{vdir.exists()}`\n\n')
        f.write('## Verify artifacts\n\n')
        for k, v in verify_ok.items():
            f.write(f'- {k}: `{v}`\n')
        f.write('\n## Splits\n\n')
        f.write(f'- train_calib: {splits["n_train"]} images\n')
        f.write(f'- heldout_calib: {splits["n_heldout"]} images\n')
        f.write(f'- P0148 excluded: `{splits["p0148_excluded"]}`\n')
        f.write(f'- cross_tiles: {len(splits["cross_tiles"])}\n')
        f.write('\n## Trainable module param counts\n\n')
        for k, v in trainable.items():
            f.write(f'- {k}: `{v}`\n')
        f.write('\n## Frozen\n\n')
        f.write('- backbone, neck, rtm_reg, rtm_ang, original support embeddings: **frozen**\n')
        if model_info:
            f.write(f'\n- total model params: `{model_info.get("total_params")}`\n')
            f.write(f'- rtm_cls_heads: `{model_info.get("rtm_cls_heads")}`\n')
        if err:
            f.write(f'\n**ERROR**: {err}\n')

    ctx.mark('preflight', status, err=err)
    return dict(status=status, fres=str(fres))
