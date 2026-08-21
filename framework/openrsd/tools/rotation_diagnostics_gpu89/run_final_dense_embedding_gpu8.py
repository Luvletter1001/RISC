#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmcv.ops import nms_rotated

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rotation_diagnostics import probe_rotated_stage_outputs as base


CONFIG = 'M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
CHECKPOINT = 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'
CLASSES = base.DOTA1_CLASSES
SMALL = CLASSES.index('small-vehicle')
LARGE = CLASSES.index('large-vehicle')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='/data1/zcy/OpenRSD')
    p.add_argument('--image-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--angles', nargs='+', type=int, required=True)
    p.add_argument('--support-type', default='visual', choices=['visual', 'text'])
    p.add_argument('--topk', type=int, default=1000)
    return p.parse_args()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_base_args(args):
    return SimpleNamespace(
        config=CONFIG,
        checkpoint=CHECKPOINT,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        angles=args.angles,
        angle_step=None,
        score_thr=0.3,
        iou_thr=0.5,
        support_shot=8,
        support_type=args.support_type,
        support_feat='',
        normalized_class_dict='data/normalized_class_dict.pkl',
        batch_size=1,
        num_workers=0,
        seed=2024,
        device='cuda:0',
        max_spatial_size=32,
        result_md_dir=args.report_dir)


def build_support(base_args, det_support_data, name2id, device, intervention):
    embed_name = 'visual_embeds' if base_args.support_type == 'visual' else 'text_embeds'
    mapping = MODEL.visual_support_mapping if base_args.support_type == 'visual' else MODEL.text_support_mapping
    feats = []
    labels = []
    for cls_name, info in det_support_data.items():
        cls_id = name2id[cls_name]
        arr = np.asarray(info[embed_name])[:base_args.support_shot]
        feats.append(arr)
        labels.append(np.ones(len(arr)) * cls_id)
    support_feats = torch.tensor(np.concatenate(feats), device=device).float()
    support_labels = torch.tensor(np.concatenate(labels), device=device).long()
    order = torch.argsort(support_labels)
    support_feats = mapping(support_feats[order])
    support_labels = support_labels[order]

    if intervention == 'zero_small_vehicle':
        support_feats[support_labels == SMALL] = 0
    elif intervention == 'swap_small_large':
        small_mask = support_labels == SMALL
        large_mask = support_labels == LARGE
        tmp = support_feats[small_mask].clone()
        support_feats[small_mask] = support_feats[large_mask][:tmp.shape[0]]
        support_feats[large_mask] = tmp[:support_feats[large_mask].shape[0]]
    elif intervention == 'small_norm_to_class_mean':
        norms = support_feats.norm(dim=1)
        mean_norm = norms.mean().clamp_min(1e-12)
        small_mask = support_labels == SMALL
        support_feats[small_mask] = (
            support_feats[small_mask] /
            support_feats[small_mask].norm(dim=1, keepdim=True).clamp_min(1e-12) *
            mean_norm)
    return support_feats.unsqueeze(0), support_labels.unsqueeze(0)


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
        bbox = torch.cat([bbox, dec_angle], dim=-1)
        decoded = head.bbox_coder.decode(pr, bbox, max_shape=img_shape)
        boxes = decoded.tensor if hasattr(decoded, 'tensor') else decoded
        levels.append((lvl, logits.detach(), scores.detach(), boxes.detach()))
    return levels


def dense_stats(angle, intervention, levels, out_npz, topk):
    rows = []
    npz = {}
    all_scores = []
    all_boxes = []
    all_levels = []
    for lvl, logits, scores, boxes in levels:
        top1 = scores.argmax(dim=1)
        max_scores = scores.max(dim=1).values
        small_scores = scores[:, SMALL]
        rows.append(dict(
            angle=angle,
            intervention=intervention,
            level=lvl,
            num_locations=int(scores.shape[0]),
            dense_top1_small_vehicle_ratio=float((top1 == SMALL).float().mean().item()),
            small_vehicle_score_mean=float(small_scores.mean().item()),
            small_vehicle_score_p95=float(torch.quantile(small_scores, 0.95).item()),
            small_vehicle_score_p99=float(torch.quantile(small_scores, 0.99).item()),
            mean_entropy=float((-(scores / scores.sum(1, keepdim=True).clamp_min(1e-12)) *
                                (scores / scores.sum(1, keepdim=True).clamp_min(1e-12)).log()).sum(1).mean().item()),
            mean_top1_margin=float((scores.topk(2, dim=1).values[:, 0] -
                                    scores.topk(2, dim=1).values[:, 1]).mean().item()),
        ))
        npz[f'level{lvl}_logits'] = logits.cpu().numpy().astype(np.float16)
        npz[f'level{lvl}_scores'] = scores.cpu().numpy().astype(np.float16)
        npz[f'level{lvl}_boxes'] = boxes.cpu().numpy().astype(np.float32)
        all_scores.append(scores)
        all_boxes.append(boxes)
        all_levels.append(torch.full((scores.shape[0],), lvl, device=scores.device))
    torch.save({}, out_npz.with_suffix('.pt.touch'))
    np.savez_compressed(out_npz, **npz)
    return rows, torch.cat(all_scores), torch.cat(all_boxes), torch.cat(all_levels)


def summarize_candidates(angle, intervention, scores, boxes, out_dir, topk):
    rows = []
    flat_scores = scores.reshape(-1)
    labels = torch.arange(scores.shape[1], device=scores.device).repeat(scores.shape[0])
    point_idx = torch.arange(scores.shape[0], device=scores.device).repeat_interleave(scores.shape[1])
    for thr in [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        keep = flat_scores >= thr
        labs = labels[keep]
        total = int(keep.sum().item())
        small = int((labs == SMALL).sum().item())
        rows.append(dict(
            angle=angle, intervention=intervention, score_thr=thr,
            candidate_total=total, small_vehicle_count=small,
            small_vehicle_ratio=float(small / total) if total else 0.0))
    k = min(topk, flat_scores.numel())
    vals, idx = torch.topk(flat_scores, k)
    labs = labels[idx]
    pts = point_idx[idx]
    top_boxes = boxes[pts]
    small_ratio = float((labs == SMALL).float().mean().item())
    top_rows = [dict(angle=angle, intervention=intervention, mode='no_nms_global_topk',
                     topk=k, total=k, small_vehicle_count=int((labs == SMALL).sum().item()),
                     small_vehicle_ratio=small_ratio)]
    # per-class top-k
    per_total = 0
    per_small = 0
    for c in range(scores.shape[1]):
        kk = min(100, scores.shape[0])
        per_total += kk
        if c == SMALL:
            per_small += kk
    top_rows.append(dict(angle=angle, intervention=intervention, mode='per_class_top100',
                         topk=100, total=per_total, small_vehicle_count=per_small,
                         small_vehicle_ratio=float(per_small / per_total)))
    # NMS over top-k candidates.
    nms_rows = []
    for mode in ['class_aware_nms', 'class_agnostic_nms']:
        kept_scores = []
        kept_labels = []
        if mode == 'class_aware_nms':
            for c in range(scores.shape[1]):
                mask = labs == c
                if mask.any():
                    _, keep = nms_rotated(top_boxes[mask], vals[mask], iou_threshold=0.5)
                    kept_scores.append(vals[mask][keep])
                    kept_labels.append(labs[mask][keep])
        else:
            _, keep = nms_rotated(top_boxes, vals, iou_threshold=0.5)
            kept_scores.append(vals[keep])
            kept_labels.append(labs[keep])
        if kept_labels:
            kl = torch.cat(kept_labels)
            total = int(kl.numel())
            small = int((kl == SMALL).sum().item())
        else:
            total = small = 0
        nms_rows.append(dict(angle=angle, intervention=intervention, mode=mode,
                             input_topk=k, detection_total=total,
                             small_vehicle_count=small,
                             small_vehicle_ratio=float(small / total) if total else 0.0))
    return rows, top_rows, nms_rows


def post_summary(angle, intervention, head, outs, metas):
    pred = head.predict_by_feat(*outs, batch_img_metas=metas, rescale=True, with_nms=True)[0]
    labels = pred.labels.detach().cpu().numpy().astype(int).tolist()
    scores = pred.scores.detach().cpu().numpy().tolist()
    total = len(labels)
    small = sum(1 for x in labels if x == SMALL)
    return dict(angle=angle, intervention=intervention, detection_total=total,
                small_vehicle_count=small,
                small_vehicle_ratio=float(small / total) if total else 0.0,
                mean_score=float(np.mean(scores)) if scores else 0.0,
                max_score=float(np.max(scores)) if scores else 0.0)


def main():
    global MODEL
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    bargs = make_base_args(args)
    base.setup_reproducibility(2024)
    support_path, _ = base.resolve_support_path('')
    runner, MODEL, cfg = base.build_runner_model(bargs, support_path)
    device = torch.device('cuda:0')
    MODEL.to(device).eval()
    _, _, det_support_data, name2id, id2name = base.prepare_support(bargs, device)
    loader = base.build_dataloader(bargs, args.angles)
    interventions = ['original', 'zero_small_vehicle', 'swap_small_large',
                     'small_norm_to_class_mean']
    dense_rows, cand_rows, topk_rows, nms_rows, post_rows = [], [], [], [], []
    with torch.no_grad():
        for data_info in loader:
            img_path = data_info['data_samples'][0].img_path
            angle = base.angle_from_name(img_path)
            if angle not in args.angles:
                continue
            data = MODEL.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            samples = data['data_samples']
            x = MODEL.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in samples]
            for intervention in interventions:
                sf, sl = build_support(bargs, det_support_data, name2id, device, intervention)
                outs_all = MODEL.bbox_head(x, sf, sl, sl, support_shot=bargs.support_shot,
                                           num_classes=len(CLASSES),
                                           num_in_classes=len(CLASSES),
                                           align_style='labelled',
                                           support_type=args.support_type,
                                           text_cls_scale=0.0)
                outs = outs_all[:-2]
                cls_scores, bbox_preds, angle_preds = outs
                levels = decode_dense(MODEL.bbox_head, cls_scores, bbox_preds, angle_preds,
                                      metas[0]['img_shape'])
                npz_path = out_dir / 'dense_npz' / f'rot{angle:03d}_{intervention}.npz'
                npz_path.parent.mkdir(exist_ok=True)
                dr, scores, boxes, _ = dense_stats(angle, intervention, levels, npz_path, args.topk)
                dense_rows.extend(dr)
                cr, tr, nr = summarize_candidates(angle, intervention, scores, boxes, out_dir, args.topk)
                cand_rows.extend(cr)
                topk_rows.extend(tr)
                nms_rows.extend(nr)
                post_rows.append(post_summary(angle, intervention, MODEL.bbox_head, outs, metas))
                print(f'DONE angle={angle:03d} intervention={intervention}')
    write_csv(out_dir / 'ftable_dense_level_stats.csv', dense_rows, list(dense_rows[0].keys()))
    write_csv(out_dir / 'ftable_pre_nms_candidates.csv', cand_rows, list(cand_rows[0].keys()))
    write_csv(out_dir / 'ftable_no_nms_topk.csv', topk_rows, list(topk_rows[0].keys()))
    write_csv(out_dir / 'ftable_nms_mode_compare.csv', nms_rows, list(nms_rows[0].keys()))
    write_csv(out_dir / 'ftable_official_post_nms.csv', post_rows, list(post_rows[0].keys()))
    orig = [r for r in dense_rows if r['intervention'] == 'original']
    original_dense_mean = float(np.mean([r['dense_top1_small_vehicle_ratio'] for r in orig]))
    post_orig = [r for r in post_rows if r['intervention'] == 'original']
    post_mean = float(np.mean([r['small_vehicle_ratio'] for r in post_orig]))
    zero_post = [r for r in post_rows if r['intervention'] == 'zero_small_vehicle']
    zero_mean = float(np.mean([r['small_vehicle_ratio'] for r in zero_post]))
    report = report_dir / 'fres_dense_prenms_embedding_intervention.md'
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    with open(report, 'w') as f:
        f.write('# Dense Pre-NMS And Embedding Intervention\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{cuda_visible}`\n')
        f.write(f'- out_dir: `{out_dir}`\n')
        f.write(f'- dense original mean top1 small_vehicle_ratio: `{original_dense_mean:.6f}`\n')
        f.write(f'- official post-NMS original mean small_vehicle_ratio: `{post_mean:.6f}`\n')
        f.write(f'- zero-small-vehicle post-NMS mean small_vehicle_ratio: `{zero_mean:.6f}`\n')
        f.write('\n## Tables\n\n')
        for name in ['ftable_dense_level_stats.csv', 'ftable_pre_nms_candidates.csv',
                     'ftable_no_nms_topk.csv', 'ftable_nms_mode_compare.csv',
                     'ftable_official_post_nms.csv']:
            f.write(f'- `{out_dir / name}`\n')
    with open(out_dir / 'gpu8_dense_summary.json', 'w') as f:
        json.dump(dict(original_dense_mean=original_dense_mean,
                       post_mean=post_mean, zero_post_mean=zero_mean,
                       report=str(report)), f, indent=2)
    print(report)


if __name__ == '__main__':
    main()
