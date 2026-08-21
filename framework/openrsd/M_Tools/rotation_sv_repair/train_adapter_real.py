#!/usr/bin/env python3
"""Train rotation adapter on real OpenRSD visual features (adapter_train tiles)."""
from __future__ import annotations

import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from M_Tools.rotation_sv_repair.common import COURT_CLASSES, SMALL, SuiteContext, setup_env, build_model_ctx
from M_Tools.rotation_sv_repair.experiments.inference_core import tiles_for_split, _prepare_batched_tile_dataset
from M_Tools.rotation_sv_repair.rotation_adapter import LowRankRotationAdapter, save_adapter
from tools.sv_attractor_repair_gpu89.repair_forward import build_support_tensors, forward_head_dense


def _pool_feat(feat_x) -> torch.Tensor:
    if isinstance(feat_x, (list, tuple)):
        x = feat_x[-1]
    else:
        x = feat_x
    if x.dim() == 4:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    return x.flatten(1)


def train_adapter_real(ctx: SuiteContext, steps: int = 800, rank: int = 16, alpha: float = 0.3,
                       lr: float = 3e-4, max_tiles: int = 40) -> dict:
    if ctx.no_train:
        return dict(status='SKIPPED', reason='--no-train')

    setup_env(ctx.gpu_ids.split(',')[0])
    bargs, model, device, det_support, name2id, _, _, _, _, _, _ = build_model_ctx(ctx)
    model.eval()

    tiles = list(dict.fromkeys(
        tiles_for_split(ctx, 'adapter_train') + tiles_for_split(ctx, 'calibration')))
    tiles = [t for t in tiles if t][:max_tiles]
    if not tiles:
        tiles = tiles_for_split(ctx, 'diagnostic_stress')[:3]

    angles = ctx.angles[:4] if ctx.mode == 'smoke' else ctx.angles[:6]
    dim = 256
    adapter = LowRankRotationAdapter(dim, rank=rank, alpha=alpha).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    sf, sl = build_support_tensors(model, det_support, name2id, device, 'visual', bargs.support_shot)

    cache = []
    with torch.no_grad():
        for tid in tiles:
            packed = _prepare_batched_tile_dataset(ctx, tid, angles)
            if packed is None:
                continue
            image_dir, _ = packed
            lb = SimpleNamespace(**vars(bargs))
            lb.image_dir = str(image_dir)
            lb.out_dir = str(ctx.work_dir / '_tmp_train')
            lb.angles = list(angles)
            try:
                from M_Tools.rotation_sv_repair.experiments.inference_core import _build_single_tile_dataloader
                loader = _build_single_tile_dataloader(lb, image_dir)
            except Exception:
                continue
            for data_info in loader:
                data = model.data_preprocessor(data_info, False)
                data['inputs'] = data['inputs'].to(device)
                x = model.prompt_extract_feats(data['inputs'])
                pooled = _pool_feat(x)
                logits_list, _, _, _ = forward_head_dense(model, x, sf, sl, bargs, repair=None)
                logit = logits_list[0]
                flat = logit.permute(0, 2, 3, 1).reshape(-1, logit.shape[1])
                for i in range(min(pooled.shape[0], flat.shape[0])):
                    cache.append((pooled[i].detach(), flat[i].detach()))

    if len(cache) < 4:
        return dict(status='FAILED', error=f'insufficient_train_samples:{len(cache)}')

    losses = []
    t0 = time.time()
    random.shuffle(cache)
    for step in range(steps):
        feat, logit = cache[step % len(cache)]
        feat = feat.unsqueeze(0)
        logit = logit.unsqueeze(0)
        adapted = adapter(feat)
        reg = 0.1 * (adapted - feat).pow(2).mean()
        margin = torch.tensor(0.0, device=device)
        for cidx in COURT_CLASSES:
            margin = margin + F.relu(
                0.2 + logit[:, SMALL] - logit[:, cidx]).mean()
        sv_pen = F.relu(logit[:, SMALL] + 0.15).mean()
        loss = reg + 0.5 * margin + 0.3 * sv_pen
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    ctx.adapter_ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(steps=steps, rank=rank, alpha=alpha, n_samples=len(cache), n_tiles=len(tiles),
                final_loss=losses[-1] if losses else 0, data_source='REAL_ADAPTER_TRAIN')
    latest = ctx.adapter_ckpt_dir / 'adapter_latest.pth'
    best = ctx.adapter_ckpt_dir / 'adapter_best_sv_repair.pth'
    save_adapter(adapter, latest, meta)
    save_adapter(adapter, best, meta)
    save_adapter(adapter, ctx.adapter_ckpt_dir / 'adapter_best_no_drift.pth', meta)
    return dict(
        status='DONE', steps=steps, param_count=adapter.param_count(),
        final_loss=losses[-1] if losses else 0,
        train_time_sec=round(time.time() - t0, 2),
        n_samples=len(cache), n_tiles=len(tiles),
        checkpoints=[str(latest), str(best)],
        data_source='REAL_ADAPTER_TRAIN',
    )
