#!/usr/bin/env python
"""Verify dehub loss gradients and frozen modules."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U
from M_AD.models.losses.sv_dehub_loss import compute_background_sv_dehub_loss
from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-dir', type=Path, default=U.RESULT_MD)
    args = ap.parse_args()
    U.ensure_dirs()
    sv_idx = probe.DOTA1_CLASSES.index('small-vehicle')
    rows = []

    # synthetic gradient check
    logits = torch.randn(64, 18, requires_grad=True)
    labels = torch.full((64,), 17)
    loss, st = compute_background_sv_dehub_loss(logits, labels, sv_idx, margin=0.05, bg_class_ind=17)
    loss.backward()
    rows.append(dict(
        module_name='synthetic_cls_logits', requires_grad='yes',
        grad_norm=f'{float(logits.grad.norm()):.6f}',
        param_norm='NA', updated='yes', comment=f'n_neg={st["n_neg"]}',
    ))

    # load model if possible
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        cfg = Config.fromfile(str(U.DEBUB_CONFIG))
        cfg.work_dir = str(U.WORK_ROOT / 'grad_audit')
        runner = Runner.from_cfg(cfg)
        model = runner.model
        head = model.bbox_head if hasattr(model, 'bbox_head') else None
        for name, p in model.named_parameters():
            if not any(k in name for k in ('bbox_head', 'backbone', 'neck')):
                continue
            tag = 'head' if 'bbox_head' in name else ('backbone' if 'backbone' in name else 'neck')
            rows.append(dict(
                module_name=name[:80], requires_grad=str(p.requires_grad),
                grad_norm='pending_train', param_norm=f'{float(p.data.norm()):.4f}',
                updated='NA', comment=tag,
            ))
        if head is not None:
            rows.append(dict(
                module_name='bbox_head.use_sv_dehub_loss',
                requires_grad='NA', grad_norm='NA', param_norm='NA',
                updated=str(getattr(head, 'use_sv_dehub_loss', False)),
                comment=f'margin={getattr(head, "sv_dehub_margin", 0)} weight={getattr(head, "sv_dehub_loss_weight", 0)}',
            ))
    except Exception as exc:
        rows.append(dict(
            module_name='model_load', requires_grad='NA', grad_norm='NA',
            param_norm='NA', updated='no', comment=str(exc)[:200],
        ))

    out = args.result_dir / 'ftable_gradient_audit.csv'
    fields = ['module_name', 'requires_grad', 'grad_norm', 'param_norm', 'updated', 'comment']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = args.result_dir / 'fres_01_gradient_and_loss_audit.md'
    md.write_text(
        '# Gradient and Loss Audit\n\n'
        f'- sv_idx (DOTA1): {sv_idx}\n'
        f'- synthetic dehub grad norm: {rows[0]["grad_norm"]}\n'
        f'- model load: see ftable_gradient_audit.csv\n',
        encoding='utf-8',
    )
    print('wrote', out, md)


if __name__ == '__main__':
    main()
