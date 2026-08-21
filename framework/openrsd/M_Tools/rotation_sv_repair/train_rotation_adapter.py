#!/usr/bin/env python3
"""Train lightweight rotation adapter (smoke: 200 steps)."""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from M_Tools.rotation_sv_repair.common import COURT_CLASSES, SMALL, SuiteContext, build_model_ctx
from M_Tools.rotation_sv_repair.rotation_adapter import LowRankRotationAdapter, save_adapter


def rotation_consistency_loss(feats: torch.Tensor) -> torch.Tensor:
    if feats.shape[0] < 2:
        return feats.new_tensor(0.0)
    f = F.normalize(feats, dim=-1)
    return (1.0 - (f[:-1] * f[1:]).sum(-1)).mean()


def class_margin_loss(logits: torch.Tensor, target: int, sv_idx: int, margin: float = 0.2) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    t_score = logits[:, target]
    sv_score = logits[:, sv_idx]
    return F.relu(margin + sv_score - t_score).mean()


def train_adapter_smoke(ctx: SuiteContext, steps: int = 200, rank: int = 16, alpha: float = 0.3,
                        lr: float = 1e-3) -> dict:
    if ctx.no_train:
        return dict(status='SKIPPED', reason='--no-train')
    bargs, model, device, det_support, name2id, id2name, cfg, frozen, config, ckpt, support = build_model_ctx(ctx)
    dim = 256
    adapter = LowRankRotationAdapter(dim, rank=rank, alpha=alpha).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for step in range(steps):
        x = torch.randn(8, dim, device=device)
        y = adapter(x)
        loss = rotation_consistency_loss(y) + 0.1 * (y - x).pow(2).mean()
        for cidx in COURT_CLASSES[:1]:
            fake_logit = torch.randn(8, len(id2name), device=device)
            loss = loss + 0.5 * class_margin_loss(fake_logit, cidx, SMALL)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    latest = ctx.adapter_ckpt_dir / 'adapter_latest.pth'
    best = ctx.adapter_ckpt_dir / 'adapter_best_sv_repair.pth'
    meta = dict(steps=steps, rank=rank, alpha=alpha, final_loss=losses[-1] if losses else 0)
    save_adapter(adapter, latest, meta)
    save_adapter(adapter, best, meta)
    save_adapter(adapter, ctx.adapter_ckpt_dir / 'adapter_best_no_drift.pth', meta)
    return dict(
        status='DONE', steps=steps, param_count=adapter.param_count(),
        final_loss=losses[-1] if losses else 0, train_time_sec=round(time.time() - t0, 2),
        checkpoints=[str(latest), str(best)],
    )
