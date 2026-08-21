#!/usr/bin/env python3
"""Run OpenRSD detector with swapped support PKL; measure dense_sv / final_sv."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from tools.encoder_swap_semantic_drift.common import (
    MODEL_SPECS, OPENRSD_CLASSES, SMALL, P0148,
)

DEFAULT_TILE_BATCH = 12
DEFAULT_AP_BATCH = 8


def _tile_batch_size(ctx, n_angles: int) -> int:
    bs = getattr(ctx, 'batch_size', DEFAULT_TILE_BATCH) or DEFAULT_TILE_BATCH
    return max(1, min(int(bs), n_angles))


def _ap_batch_size(ctx) -> int:
    return max(1, int(getattr(ctx, 'ap_batch_size', DEFAULT_AP_BATCH) or DEFAULT_AP_BATCH))


class DetectorSession:
    """Load detector weights once per model_key; swap support PKL only."""

    def __init__(self, ctx, gpu: str, model_key: str, batch_size: int = DEFAULT_TILE_BATCH,
                 num_workers: int = 2):
        self.ctx = ctx
        self.gpu = str(gpu)
        self.model_key = model_key
        self.batch_size = max(1, int(batch_size))
        self.num_workers = num_workers
        self.model = None
        self.bargs = None
        self.device = None
        self.det_support = None
        self.name2id = None
        self.id2name = None
        self.wctx = None
        self._support_pkl: Optional[Path] = None
        self._load_model()

    def _load_model(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        os.environ['ROTATION_SV_REPAIR_NO_REEXEC'] = '1'

        from M_Tools.rotation_sv_repair.common import SuiteContext, build_model_ctx, setup_env
        import M_Tools.rotation_sv_repair.common as crc

        setup_env('0')
        spec = MODEL_SPECS[self.model_key]
        self.wctx = SuiteContext(
            repo_root=self.ctx.repo_root,
            work_dir=self.ctx.work_dir / 'detector_runs',
            result_dir=self.ctx.result_md_dir,
            gpu_ids=self.gpu,
            mode='smoke' if self.ctx.smoke else 'full',
            exp='encoder_swap',
            teacher='na', student='na', pseudo_label_mode='na', vocab_mode='na', teacher_vlm='na',
            only_config=str(spec['config']),
            only_checkpoint=str(spec['checkpoint']),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            force=True,
        )
        self._orig_disc = crc.discover_config_checkpoint
        self.bargs, self.model, self.device, self.det_support, self.name2id, self.id2name, *_ = (
            build_model_ctx(self.wctx)
        )
        self._support_pkl = Path(self.bargs.support_feat)

    def set_support(self, support_pkl: Path, support_type: str = 'visual'):
        support_pkl = Path(support_pkl).resolve()
        if self._support_pkl == support_pkl:
            return
        from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base

        self.bargs.support_feat = str(support_pkl)
        self.bargs.support_type = support_type
        _, _, self.det_support, self.name2id, self.id2name = probe_base.prepare_support(
            self.bargs, self.device)
        self._support_pkl = support_pkl

    def eval_tiles(
            self,
            support_pkl: Path,
            tile_ids: List[str],
            angles: List[int],
            support_type: str = 'visual',
            text_encoder: str = 'original',
            image_encoder: str = 'original',
            prompt_family: str = 'raw',
            projection_method: str = 'none',
            extra_tag: str = '',
    ) -> List[dict]:
        from M_Tools.rotation_sv_repair.experiments.inference_core import eval_tile
        from tools.sv_attractor_repair_gpu89.repair_modules import RepairStack

        self.set_support(support_pkl, support_type)
        self.wctx.batch_size = _tile_batch_size(self.ctx, len(angles))
        self.bargs.batch_size = self.wctx.batch_size
        repair = RepairStack()
        method = extra_tag or f'{text_encoder}_{image_encoder}'
        rows = []
        for tid in tile_ids:
            try:
                part = eval_tile(
                    self.wctx, self.model, self.bargs, self.device, self.det_support, self.name2id,
                    tid, angles, 'diagnostic_stress' if tid == P0148 else 'highrisk',
                    method=method, repair_stack=repair,
                )
                for r in part:
                    r.update(dict(
                        model_key=self.model_key,
                        text_encoder=text_encoder,
                        image_encoder=image_encoder,
                        prompt_family=prompt_family,
                        projection_method=projection_method,
                        support_pkl=str(support_pkl),
                        support_type=support_type,
                        swap_scope='prompt_encoder_only_not_backbone',
                        gpu=self.gpu,
                    ))
                    rows.append(r)
            except Exception as exc:
                for ang in angles:
                    rows.append(dict(
                        tile_id=tid, angle=f'{ang:03d}', status=f'FAILED:{exc}',
                        model_key=self.model_key, text_encoder=text_encoder,
                        image_encoder=image_encoder, gpu=self.gpu,
                    ))
        return rows

    def close(self):
        if self.device is not None and self.device.type == 'cuda':
            import torch
            del self.model
            torch.cuda.empty_cache()
        self.model = None


def run_detector_eval(
        ctx,
        gpu: str,
        model_key: str,
        support_pkl: Path,
        tile_ids: List[str],
        angles: List[int],
        support_type: str = 'visual',
        text_encoder: str = 'original',
        image_encoder: str = 'original',
        prompt_family: str = 'raw',
        projection_method: str = 'none',
        extra_tag: str = '',
        session: Optional[DetectorSession] = None,
) -> List[dict]:
    """REAL forward via rotation_sv_repair inference_core (optionally cached session)."""
    owned = session is None
    if owned:
        session = DetectorSession(
            ctx, gpu, model_key,
            batch_size=_tile_batch_size(ctx, len(angles)),
            num_workers=2 if not ctx.smoke else 0,
        )
    try:
        return session.eval_tiles(
            support_pkl, tile_ids, angles, support_type=support_type,
            text_encoder=text_encoder, image_encoder=image_encoder,
            prompt_family=prompt_family, projection_method=projection_method,
            extra_tag=extra_tag,
        )
    finally:
        if owned:
            session.close()


def prepare_heldout_subset(ctx, max_images: int) -> Path:
    """Symlink held-out tile images/anns for AP eval (final_test / calib pool)."""
    stems: List[str] = []
    calib_json = ctx.repo_root / 'work_dirs/sv_attractor_repair_gpu89_20260519_161938/splits/calib_split.json'
    if calib_json.exists():
        data = json.loads(calib_json.read_text(encoding='utf-8'))
        stems = list(data.get('heldout_calib', []))[:max_images]
    if not stems:
        split_csv = ctx.repo_root / 'resultmd/exp_rotation_sv_repair_20260524/tables/ftable_001_tile_splits.csv'
        if split_csv.exists():
            import csv as _csv
            with open(split_csv, newline='', encoding='utf-8') as f:
                for r in _csv.DictReader(f):
                    if r.get('split') in ('final_test', 'calibration', 'validation'):
                        stems.append(r['tile_id'])
            stems = list(dict.fromkeys(stems))[:max_images]
    root = ctx.work_dir / 'eval_subsets' / 'heldout_ap'
    img_dir = root / 'images'
    ann_dir = root / 'annfiles'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for p in list(img_dir.iterdir()):
        p.unlink(missing_ok=True)
    for p in list(ann_dir.iterdir()):
        p.unlink(missing_ok=True)
    ss_train = ctx.repo_root / 'data/DOTA1_1024_500/ss_train'
    ss_val = ctx.repo_root / 'data/DOTA1_1024_500/ss_val'
    linked = 0
    for stem in stems:
        for base in (ss_train, ss_val):
            ann = base / 'annfiles' / f'{stem}.txt'
            if not ann.exists():
                continue
            for ext in ('.png', '.jpg', '.jpeg'):
                ip = base / 'images' / f'{stem}{ext}'
                if ip.exists():
                    (img_dir / ip.name).symlink_to(ip.resolve())
                    (ann_dir / ann.name).symlink_to(ann.resolve())
                    linked += 1
                    break
            if (ann_dir / ann.name).exists():
                break
    root.joinpath('meta.json').write_text(
        json.dumps(dict(stems=stems[:linked], n=linked), indent=2), encoding='utf-8')
    return root


class HeldoutApSession:
    """Cached model + batched held-out AP inference."""

    def __init__(self, ctx, gpu: str, model_key: str = 'baseline', num_workers: int = 0):
        self.ctx = ctx
        self.gpu = str(gpu)
        self.model_key = model_key
        self.ap_batch = _ap_batch_size(ctx)
        self.num_workers = int(num_workers)
        self._detector = DetectorSession(
            ctx, gpu, model_key, batch_size=self.ap_batch, num_workers=self.num_workers)

    def eval(self, support_pkl: Path, image_encoder: str = 'original',
             max_images: int = 60, angles: Optional[List[int]] = None) -> dict:
        import torch
        from mmengine.dataset import pseudo_collate
        from torch.utils.data import DataLoader
        from tools.sv_attractor_repair_gpu89 import common as C
        from tools.sv_attractor_repair_gpu89.eval_ap import (
            _as_rbox_tensor, build_eval_cfg, classwise_ap50, forward_with_repair,
            rotate_image_tensor,
        )
        from tools.sv_attractor_repair_gpu89.repair_forward import build_support_tensors
        from tools.sv_attractor_repair_gpu89.repair_modules import RepairStack

        eval_angles = angles if angles is not None else ([0] if self.ctx.smoke else [0])
        self._detector.set_support(support_pkl, 'visual')
        model = self._detector.model
        device = self._detector.device
        bargs = self._detector.bargs
        det_support = self._detector.det_support
        name2id = self._detector.name2id
        rep = RepairStack().to(device)
        sf_base, sl = build_support_tensors(model, det_support, name2id, device, 'visual', bargs.support_shot)
        spec = MODEL_SPECS[self.model_key]
        rctx = C.RepairContext(
            repo_root=self.ctx.repo_root,
            config=Path(spec['config']),
            checkpoint=Path(spec['checkpoint']),
            support_pkl=Path(support_pkl),
            verify_work_dir=self.ctx.work_dir,
            work_dir=self.ctx.work_dir / 'heldout_ap_runs',
            result_md_dir=self.ctx.result_md_dir,
            gpu=int(self.gpu),
            mode='smoke' if self.ctx.smoke else 'full',
            angles=eval_angles,
            tiles=[], train_max_images=32,
            heldout_max_images=max_images, eval_max_images=max_images,
            iters=0, batch_size=self.ap_batch, lr=0.0, force=True,
        )
        subset = prepare_heldout_subset(self.ctx, max_images)
        cfg = build_eval_cfg(rctx, subset)
        dataset = __import__('mmdet.registry', fromlist=['DATASETS']).DATASETS.build(cfg.val_dataloader.dataset)
        loader = DataLoader(
            dataset, batch_size=self.ap_batch, shuffle=False,
            num_workers=self.num_workers, collate_fn=pseudo_collate,
            pin_memory=(self.num_workers == 0))
        ap_samples = []
        ratio_rows = []
        n_seen = 0
        with torch.no_grad():
            for batch in loader:
                if n_seen >= max_images:
                    break
                samples = batch['data_samples']
                bs = len(samples)
                if n_seen + bs > max_images:
                    samples = samples[: max_images - n_seen]
                    bs = len(samples)
                    batch = dict(
                        inputs=batch['inputs'][:bs],
                        data_samples=samples,
                    )
                n_seen += bs
                for angle in eval_angles:
                    data = model.data_preprocessor(batch, False)
                    data['inputs'] = data['inputs'].to(device)
                    if angle:
                        data['inputs'] = rotate_image_tensor(data['inputs'], angle)
                    x = model.prompt_extract_feats(data['inputs'])
                    metas = [s.metainfo for s in data['data_samples']]
                    outs = forward_with_repair(model, x, sf_base, sl, rep, sf_mod=None)
                    preds = model.bbox_head.predict_by_feat(
                        *outs, batch_img_metas=metas, rescale=True, with_nms=True)
                    for i, pred in enumerate(preds):
                        stem = Path(data['data_samples'][i].img_path).stem
                        keep = pred.scores >= 0.01
                        ratio_rows.append(dict(
                            stem=stem, angle=angle,
                            final_sv_ratio=float(
                                (pred.labels[keep] == C.SMALL).float().mean().item()
                                if keep.any() else 0.0),
                        ))
                        if angle != 0:
                            continue
                        gt = data['data_samples'][i].gt_instances
                        ap_samples.append(dict(
                            img_id=stem,
                            gt_instances=dict(labels=gt.labels.cpu(), bboxes=_as_rbox_tensor(gt.bboxes)),
                            ignored_instances=dict(
                                labels=torch.zeros(0, dtype=torch.long),
                                bboxes=torch.zeros((0, 5), dtype=torch.float32)),
                            pred_instances=dict(
                                bboxes=_as_rbox_tensor(pred.bboxes[keep]),
                                scores=pred.scores[keep].detach().cpu(),
                                labels=pred.labels[keep].detach().cpu(),
                            ),
                        ))
        cls_rows, map_ap = classwise_ap50(ap_samples, tuple(C.CLASSES))
        per = {r['class_name']: r['ap50'] for r in cls_rows}
        return dict(
            status='OK' if ap_samples else 'FAILED',
            model_key=self.model_key,
            image_encoder=image_encoder,
            support_pkl=str(support_pkl),
            n_heldout=n_seen,
            n_ap_samples=len(ap_samples),
            mAP50=round(float(map_ap), 4) if ap_samples else '',
            small_vehicle_AP50=round(float(per.get('small-vehicle', float('nan'))), 4) if ap_samples else '',
            tennis_court_AP50=round(float(per.get('tennis-court', float('nan'))), 4) if ap_samples else '',
            final_sv_mean=round(float(np.mean([r['final_sv_ratio'] for r in ratio_rows])), 4) if ratio_rows else '',
            data_source='REAL_OPENRSD_EVAL',
            gpu=self.gpu,
        )

    def close(self):
        self._detector.close()


def run_heldout_ap_eval(
        ctx,
        gpu: str,
        model_key: str,
        support_pkl: Path,
        image_encoder: str = 'original',
        max_images: int = 60,
        angles: Optional[List[int]] = None,
        session: Optional[HeldoutApSession] = None,
) -> dict:
    """REAL held-out AP50 with swapped support PKL (batched, angle-0 predictions)."""
    owned = session is None
    if owned:
        session = HeldoutApSession(ctx, gpu, model_key)
    try:
        return session.eval(support_pkl, image_encoder=image_encoder,
                            max_images=max_images, angles=angles)
    finally:
        if owned:
            session.close()


def aggregate_metrics(rows: List[dict], tile_id: str = P0148) -> dict:
    sub = [r for r in rows if r.get('tile_id') == tile_id and r.get('status') == 'OK']
    if not sub:
        return dict(status='FAILED', n=0)
    final_sv = [float(r['final_sv_ratio']) for r in sub if r.get('final_sv_ratio') != '']
    dense_sv = [float(r['dense_top1_sv_ratio']) for r in sub if r.get('dense_top1_sv_ratio') != '']
    det = [int(r['final_total_det']) for r in sub if r.get('final_total_det') != '']
    return dict(
        status='OK',
        n=len(sub),
        final_sv_mean=round(float(np.mean(final_sv)), 4) if final_sv else '',
        dense_sv_mean=round(float(np.mean(dense_sv)), 4) if dense_sv else '',
        det_mean=round(float(np.mean(det)), 1) if det else '',
    )


def shard_list(items: list, n: int) -> List[list]:
    n = max(1, n)
    return [items[i::n] for i in range(n)]


def merge_exp03_shard_results(shards: List[Tuple[List[dict], List[dict]]]) -> Tuple[List[dict], List[dict]]:
    summary, raw = [], []
    for s, r in shards:
        summary.extend(s)
        raw.extend(r)
    return summary, raw
