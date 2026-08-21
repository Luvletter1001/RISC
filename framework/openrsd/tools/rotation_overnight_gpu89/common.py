#!/usr/bin/env python
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rotation_diagnostics import probe_rotated_stage_outputs as base


CONFIG = 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
CHECKPOINT = 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
CLASSES = base.DOTA1_CLASSES
SMALL = CLASSES.index('small-vehicle')
LARGE = CLASSES.index('large-vehicle')


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def mean_float(rows, key):
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float('nan')


def make_base_args(args, support_type='visual'):
    return SimpleNamespace(
        config=CONFIG,
        checkpoint=CHECKPOINT,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        angles=args.angles,
        angle_step=None,
        score_thr=0.3,
        iou_thr=0.5,
        support_shot=getattr(args, 'support_shot', 8),
        support_type=support_type,
        support_feat='',
        normalized_class_dict='data/normalized_class_dict.pkl',
        batch_size=1,
        num_workers=0,
        seed=20260518,
        device='cuda:0',
        max_spatial_size=32,
        result_md_dir=args.report_dir)


def build_model_and_data(args, support_type='visual'):
    bargs = make_base_args(args, support_type=support_type)
    base.setup_reproducibility(20260518)
    support_path, _ = base.resolve_support_path('')
    _, model, _ = base.build_runner_model(bargs, support_path)
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, _, det_support_data, name2id, id2name = base.prepare_support(bargs, device)
    loader = base.build_dataloader(bargs, args.angles)
    return bargs, model, device, det_support_data, name2id, id2name, loader


def build_support(model, base_args, det_support_data, name2id, device, intervention):
    embed_name = 'visual_embeds' if base_args.support_type == 'visual' else 'text_embeds'
    mapping = model.visual_support_mapping if base_args.support_type == 'visual' else model.text_support_mapping
    feats = []
    labels = []
    for cls_name, info in det_support_data.items():
        cls_id = name2id[cls_name]
        arr = np.asarray(info[embed_name])[:base_args.support_shot]
        feats.append(arr)
        labels.append(np.ones(len(arr)) * cls_id)
    raw_feats = torch.tensor(np.concatenate(feats), device=device).float()
    raw_labels = torch.tensor(np.concatenate(labels), device=device).long()
    order = torch.argsort(raw_labels)
    support_feats = mapping(raw_feats[order])
    support_labels = raw_labels[order]
    before = embedding_summary(support_feats, support_labels, intervention, 'before')

    if intervention == 'zero_sv':
        support_feats[support_labels == SMALL] = 0
    elif intervention == 'swap_sv_lv':
        small_mask = support_labels == SMALL
        large_mask = support_labels == LARGE
        tmp = support_feats[small_mask].clone()
        support_feats[small_mask] = support_feats[large_mask][:tmp.shape[0]]
        support_feats[large_mask] = tmp[:support_feats[large_mask].shape[0]]
    elif intervention == 'norm_sv_mean':
        norms = support_feats.norm(dim=1)
        mean_norm = norms.mean().clamp_min(1e-12)
        small_mask = support_labels == SMALL
        support_feats[small_mask] = (
            support_feats[small_mask] /
            support_feats[small_mask].norm(dim=1, keepdim=True).clamp_min(1e-12) *
            mean_norm)
    elif intervention == 'normalize_all':
        mean_norm = support_feats.norm(dim=1).mean().clamp_min(1e-12)
        support_feats = support_feats / support_feats.norm(dim=1, keepdim=True).clamp_min(1e-12) * mean_norm
    elif intervention == 'random_sv':
        gen = torch.Generator(device=device)
        gen.manual_seed(20260518)
        small_mask = support_labels == SMALL
        replacement = torch.randn(
            support_feats[small_mask].shape,
            generator=gen,
            device=device,
            dtype=support_feats.dtype)
        target_norm = support_feats[small_mask].norm(dim=1, keepdim=True).mean().clamp_min(1e-12)
        replacement = replacement / replacement.norm(dim=1, keepdim=True).clamp_min(1e-12) * target_norm
        support_feats[small_mask] = replacement
    after = embedding_summary(support_feats, support_labels, intervention, 'after')
    return support_feats.unsqueeze(0), support_labels.unsqueeze(0), before + after


def embedding_summary(feats, labels, intervention, phase):
    rows = []
    sv = feats[labels == SMALL]
    lv = feats[labels == LARGE]
    all_norm = feats.norm(dim=1)
    sv_mean = sv.mean(dim=0, keepdim=True)
    lv_mean = lv.mean(dim=0, keepdim=True)
    cos = torch.nn.functional.cosine_similarity(sv_mean, lv_mean).item()
    rows.append(dict(
        intervention=intervention,
        phase=phase,
        class_name='all',
        mean_norm=float(all_norm.mean().item()),
        min_norm=float(all_norm.min().item()),
        max_norm=float(all_norm.max().item()),
        sv_lv_mean_cosine=float(cos)))
    for cls_id, cls_name in enumerate(CLASSES):
        mask = labels == cls_id
        if not mask.any():
            continue
        norms = feats[mask].norm(dim=1)
        rows.append(dict(
            intervention=intervention,
            phase=phase,
            class_name=cls_name,
            mean_norm=float(norms.mean().item()),
            min_norm=float(norms.min().item()),
            max_norm=float(norms.max().item()),
            sv_lv_mean_cosine=float(cos)))
    return rows


def decode_dense(head, cls_scores, bbox_preds, angle_preds, img_shape):
    featmap_sizes = [x.shape[-2:] for x in cls_scores]
    priors = head.prior_generator.grid_priors(
        featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
    levels = []
    for lvl, (cs, bp, ap, pr) in enumerate(zip(cls_scores, bbox_preds, angle_preds, priors)):
        logits = cs[0].permute(1, 2, 0).reshape(-1, len(CLASSES))
        scores = logits.sigmoid()
        bbox = bp[0].permute(1, 2, 0).reshape(-1, 4)
        angle = ap[0].permute(1, 2, 0).reshape(-1, head.angle_coder.encode_size)
        dec_angle = head.angle_coder.decode(angle, keepdim=True)
        decoded = head.bbox_coder.decode(
            pr, torch.cat([bbox, dec_angle], dim=-1), max_shape=img_shape)
        boxes = decoded.tensor if hasattr(decoded, 'tensor') else decoded
        levels.append((lvl, logits.detach(), scores.detach(), boxes.detach()))
    return levels


def model_forward_dense(model, x, support_feats, support_labels, base_args):
    outs_all = model.bbox_head(
        x, support_feats, support_labels, support_labels,
        support_shot=base_args.support_shot,
        num_classes=len(CLASSES),
        num_in_classes=len(CLASSES),
        align_style='labelled',
        support_type=base_args.support_type,
        text_cls_scale=0.0)
    return outs_all[:-2]


def post_summary(angle, intervention, head, outs, metas, cfg=None):
    pred = head.predict_by_feat(*outs, batch_img_metas=metas, cfg=cfg, rescale=True, with_nms=True)[0]
    labels = pred.labels.detach().cpu().numpy().astype(int).tolist()
    scores = pred.scores.detach().cpu().numpy().tolist()
    hist = {CLASSES[i]: labels.count(i) for i in sorted(set(labels))}
    total = len(labels)
    small = sum(1 for x in labels if x == SMALL)
    large = sum(1 for x in labels if x == LARGE)
    top = max(hist.items(), key=lambda kv: kv[1])[0] if hist else ''
    return dict(
        angle=angle,
        intervention=intervention,
        detection_total=total,
        small_vehicle_count=small,
        large_vehicle_count=large,
        small_vehicle_ratio=float(small / total) if total else 0.0,
        large_vehicle_ratio=float(large / total) if total else 0.0,
        top1_class=top,
        mean_score=float(np.mean(scores)) if scores else 0.0,
        max_score=float(np.max(scores)) if scores else 0.0,
        class_histogram=json.dumps(hist, ensure_ascii=True))


def top_location_arrays(scores, topk_classes=5):
    probs = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = (-(probs * probs.log())).sum(dim=1)
    topk = scores.topk(min(topk_classes, scores.shape[1]), dim=1)
    order = torch.argsort(scores, dim=1, descending=True)
    sv_rank = (order == SMALL).nonzero()[:, 1] + 1
    return dict(
        top_classes=topk.indices.cpu().numpy().astype(np.int16),
        top_scores=topk.values.cpu().numpy().astype(np.float16),
        entropy=entropy.cpu().numpy().astype(np.float16),
        small_vehicle_score=scores[:, SMALL].cpu().numpy().astype(np.float16),
        small_vehicle_rank=sv_rank.cpu().numpy().astype(np.int16))


def cuda_visible():
    return os.environ.get('CUDA_VISIBLE_DEVICES', '')
