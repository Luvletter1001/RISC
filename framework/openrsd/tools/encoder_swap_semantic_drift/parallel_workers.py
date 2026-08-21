#!/usr/bin/env python3
"""Multiprocessing workers for encoder-swap detector eval (spawn-safe)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def _ctx_from_dict(d: dict):
    from tools.encoder_swap_semantic_drift.common import EncoderSwapCtx
    return EncoderSwapCtx(**d)


def _ctx_to_dict(ctx) -> dict:
    return dict(
        repo_root=ctx.repo_root,
        work_dir=ctx.work_dir,
        result_md_dir=ctx.result_md_dir,
        pretrained_root=ctx.pretrained_root,
        python=ctx.python,
        mode=ctx.mode,
        force=ctx.force,
        smoke=ctx.smoke,
        batch_size=getattr(ctx, 'batch_size', 12),
        ap_batch_size=getattr(ctx, 'ap_batch_size', 8),
    )


def worker_exp03_shard(
        gpu: str,
        enc_fam_shard: List[Tuple[str, str]],
        models: List[str],
        tiles: List[str],
        angles: List[int],
        ctx_dict: dict,
) -> Tuple[List[dict], List[dict]]:
    from tools.encoder_swap_semantic_drift.prompts import FAMILY_SHORT
    from tools.encoder_swap_semantic_drift.support_builder import build_text_swapped_support
    from tools.encoder_swap_semantic_drift.detector_eval import (
        DetectorSession, aggregate_metrics, _tile_batch_size,
    )
    from tools.encoder_swap_semantic_drift.common import bind_gpu

    ctx = _ctx_from_dict(ctx_dict)
    encode_device = bind_gpu(gpu)
    bs = _tile_batch_size(ctx, len(angles))
    # Pool workers are daemonic — DataLoader num_workers>0 cannot spawn children.
    sessions = {mk: DetectorSession(ctx, gpu, mk, batch_size=bs, num_workers=0) for mk in models}
    summary_rows, all_raw = [], []
    try:
        for enc, fam in enc_fam_shard:
            try:
                pkl, meta = build_text_swapped_support(
                    ctx, enc, fam, 'orthogonal_procrustes', device=encode_device)
            except Exception as exc:
                summary_rows.append(dict(
                    text_encoder=enc, prompt_family=FAMILY_SHORT.get(fam, fam),
                    status='BLOCKED', error=str(exc)[:200], gpu=gpu))
                continue
            for mk in models:
                raw = sessions[mk].eval_tiles(
                    pkl, tiles, angles, support_type='text',
                    text_encoder=enc, prompt_family=FAMILY_SHORT.get(fam, fam),
                    projection_method=meta.get('projection_used', ''),
                )
                all_raw.extend(raw)
                agg = aggregate_metrics(raw, tiles[0] if tiles else 'P0148__1024__651___0')
                row = dict(
                    model_key=mk, text_encoder=enc, prompt_family=FAMILY_SHORT.get(fam, fam),
                    swap_scope='text_prompt_only', gpu=gpu,
                )
                row.update(agg)
                summary_rows.append(row)
    finally:
        for s in sessions.values():
            s.close()
    return summary_rows, all_raw


def worker_exp05_shard(
        gpu: str,
        pair_shard: List[Tuple[str, str]],
        angles: List[int],
        ctx_dict: dict,
) -> List[dict]:
    from tools.encoder_swap_semantic_drift.common import bind_gpu, OPENRSD_CLASSES, P0148
    from tools.encoder_swap_semantic_drift.support_builder import (
        build_text_swapped_support, build_visual_swapped_support,
    )
    from tools.encoder_swap_semantic_drift.crop_sampler import sample_class_crops
    from tools.encoder_swap_semantic_drift.detector_eval import (
        DetectorSession, aggregate_metrics, _tile_batch_size,
    )
    from commonlibs.common_tools import pklload, pklsave

    ctx = _ctx_from_dict(ctx_dict)
    encode_device = bind_gpu(gpu)
    crops = sample_class_crops(ctx.repo_root, smoke=ctx.smoke)
    bs = _tile_batch_size(ctx, len(angles))
    session = DetectorSession(ctx, gpu, 'baseline', batch_size=bs, num_workers=0)
    rows = []
    try:
        for te, ie in pair_shard:
            try:
                tp, _ = build_text_swapped_support(
                    ctx, te, 'F2_remote_sensing', 'orthogonal_procrustes', device=encode_device)
                vp, _ = build_visual_swapped_support(
                    ctx, ie, 'orthogonal_procrustes', crops, device=encode_device)
                merged = pklload(str(vp))
                text_part = pklload(str(tp))
                for c in OPENRSD_CLASSES:
                    if c in text_part:
                        merged[c]['text_embeds'] = text_part[c]['text_embeds']
                        merged[c]['texts'] = text_part[c].get('texts', [])
                combo_path = ctx.support_swap_dir / f'support_pair_{te}_{ie}.pkl'
                pklsave(merged, str(combo_path))
                raw = session.eval_tiles(
                    combo_path, [P0148], angles,
                    support_type='visual', text_encoder=te, image_encoder=ie)
                agg = aggregate_metrics(raw, P0148)
                row = dict(text_encoder=te, image_encoder=ie, gpu=gpu)
                row.update(agg)
                rows.append(row)
            except Exception as exc:
                rows.append(dict(
                    text_encoder=te, image_encoder=ie, status='BLOCKED',
                    error=str(exc)[:120], gpu=gpu))
    finally:
        session.close()
    return rows


def worker_exp07_shard(
        gpu: str,
        encoder_shard: List[str],
        max_images: int,
        ctx_dict: dict,
) -> List[dict]:
    from tools.encoder_swap_semantic_drift.common import bind_gpu
    from tools.encoder_swap_semantic_drift.support_builder import build_visual_swapped_support
    from tools.encoder_swap_semantic_drift.crop_sampler import sample_class_crops
    from tools.encoder_swap_semantic_drift.detector_eval import HeldoutApSession

    ctx = _ctx_from_dict(ctx_dict)
    encode_device = bind_gpu(gpu)
    crops = sample_class_crops(ctx.repo_root, smoke=ctx.smoke)
    orig = ctx.repo_root / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'
    session = HeldoutApSession(ctx, gpu, 'baseline', num_workers=0)
    rows = []
    try:
        for enc in encoder_shard:
            try:
                if enc == 'original':
                    pkl = orig
                else:
                    pkl, _ = build_visual_swapped_support(
                        ctx, enc, 'orthogonal_procrustes', crops, device=encode_device)
                out = session.eval(Path(pkl), image_encoder=enc, max_images=max_images)
                rows.append(out)
            except Exception as exc:
                rows.append(dict(
                    model_key='baseline', image_encoder=enc, status='FAILED',
                    error=str(exc)[:200], data_source='REAL_OPENRSD_EVAL', gpu=gpu))
    finally:
        session.close()
    return rows


def _parallel_shard_target(result_q, worker_fn, gpu, shard, extra_args, ctx_dict):
    try:
        result_q.put(('ok', worker_fn(gpu, shard, *extra_args, ctx_dict)))
    except Exception as exc:
        result_q.put(('err', repr(exc)))


def run_parallel_shards(worker_fn, gpus: List[str], shard_items: list, ctx, *extra_args):
    """Run worker_fn across GPUs via spawn Process (daemon=False for nested loaders)."""
    import multiprocessing as mp

    ctx_dict = _ctx_to_dict(ctx)
    n = min(len(gpus), max(1, len(shard_items)))
    if n <= 1:
        return [worker_fn(gpus[0], shard_items, *extra_args, ctx_dict)]

    shards = [shard_items[i::n] for i in range(n)]
    active = [(gpus[i], shards[i]) for i in range(n) if shards[i]]
    mp_ctx = mp.get_context('spawn')

    procs, queues = [], []
    for gpu, shard in active:
        q = mp_ctx.Queue()
        p = mp_ctx.Process(
            target=_parallel_shard_target,
            args=(q, worker_fn, gpu, shard, extra_args, ctx_dict),
            daemon=False,
        )
        p.start()
        procs.append(p)
        queues.append(q)
    for p in procs:
        p.join()

    results = []
    for q in queues:
        status, payload = q.get()
        if status == 'err':
            raise RuntimeError(payload)
        results.append(payload)
    return results
