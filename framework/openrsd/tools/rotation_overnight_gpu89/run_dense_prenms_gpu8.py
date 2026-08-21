#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mmcv.ops import nms_rotated

from tools.rotation_overnight_gpu89 import common as C
from tools.rotation_diagnostics import probe_rotated_stage_outputs as base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='/data1/zcy/OpenRSD')
    p.add_argument('--image-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--angles', nargs='+', type=int, required=True)
    p.add_argument('--topk', type=int, default=1000)
    p.add_argument('--save-top-locations', type=int, default=4096)
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def summarize_level(angle, lvl, logits, scores, out_dir, save_top_locations):
    top2 = scores.topk(2, dim=1)
    top1 = top2.indices[:, 0]
    probs = scores / scores.sum(1, keepdim=True).clamp_min(1e-12)
    entropy = (-(probs * probs.log())).sum(1)
    sv_rank = (torch.argsort(scores, dim=1, descending=True) == C.SMALL).nonzero()[:, 1] + 1
    rows = [dict(
        angle=angle,
        level=lvl,
        num_locations=int(scores.shape[0]),
        dense_top1_small_vehicle_ratio=float((top1 == C.SMALL).float().mean().item()),
        top1_score_mean=float(top2.values[:, 0].mean().item()),
        top2_score_mean=float(top2.values[:, 1].mean().item()),
        top1_top2_margin_mean=float((top2.values[:, 0] - top2.values[:, 1]).mean().item()),
        entropy_mean=float(entropy.mean().item()),
        small_vehicle_score_mean=float(scores[:, C.SMALL].mean().item()),
        small_vehicle_score_p95=float(torch.quantile(scores[:, C.SMALL], 0.95).item()),
        small_vehicle_score_p99=float(torch.quantile(scores[:, C.SMALL], 0.99).item()),
        small_vehicle_rank_mean=float(sv_rank.float().mean().item()))]
    k = min(save_top_locations, scores.shape[0])
    vals, idx = torch.topk(scores.max(1).values, k)
    arrays = C.top_location_arrays(scores[idx])
    arrays['location_index'] = idx.cpu().numpy().astype(np.int32)
    arrays['top1_logits'] = logits[idx, top1[idx]].cpu().numpy().astype(np.float16)
    npz = out_dir / 'dense_top_locations' / f'rot{angle:03d}_level{lvl}.npz'
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    return rows


def flatten_scores(levels):
    scores = torch.cat([s for _, _, s, _ in levels])
    boxes = torch.cat([b for _, _, _, b in levels])
    levels_flat = torch.cat([
        torch.full((s.shape[0],), lvl, device=s.device)
        for lvl, _, s, _ in levels])
    return scores, boxes, levels_flat


def score_thr_rows(angle, scores):
    flat_scores = scores.reshape(-1)
    labels = torch.arange(scores.shape[1], device=scores.device).repeat(scores.shape[0])
    rows = []
    for thr in [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        keep = flat_scores >= thr
        labs = labels[keep]
        total = int(keep.sum().item())
        small = int((labs == C.SMALL).sum().item())
        rows.append(dict(
            angle=angle,
            score_thr=thr,
            candidate_total=total,
            small_vehicle_count=small,
            small_vehicle_ratio=float(small / total) if total else 0.0))
    return rows


def nms_and_topk_rows(angle, scores, boxes, topk, out_dir):
    flat_scores = scores.reshape(-1)
    labels = torch.arange(scores.shape[1], device=scores.device).repeat(scores.shape[0])
    point_idx = torch.arange(scores.shape[0], device=scores.device).repeat_interleave(scores.shape[1])
    k = min(topk, flat_scores.numel())
    vals, idx = torch.topk(flat_scores, k)
    labs = labels[idx]
    pts = point_idx[idx]
    top_boxes = boxes[pts]
    np.savez_compressed(
        out_dir / 'top_candidates' / f'rot{angle:03d}_top{k}.npz',
        scores=vals.detach().cpu().numpy().astype(np.float16),
        labels=labs.detach().cpu().numpy().astype(np.int16),
        boxes=top_boxes.detach().cpu().numpy().astype(np.float32))
    top_rows = [dict(
        angle=angle,
        mode='no_nms_global_topk',
        nms_iou='',
        topk=k,
        detection_total=k,
        small_vehicle_count=int((labs == C.SMALL).sum().item()),
        small_vehicle_ratio=float((labs == C.SMALL).float().mean().item()))]
    per_total = 0
    per_small = 0
    for cls_id in range(scores.shape[1]):
        per_total += min(100, scores.shape[0])
        if cls_id == C.SMALL:
            per_small += min(100, scores.shape[0])
    top_rows.append(dict(
        angle=angle,
        mode='per_class_top100',
        nms_iou='',
        topk=100,
        detection_total=per_total,
        small_vehicle_count=per_small,
        small_vehicle_ratio=float(per_small / per_total)))

    nms_rows = []
    for iou in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for mode in ['class_aware_nms', 'class_agnostic_nms']:
            kept_labels = []
            if mode == 'class_aware_nms':
                for cls_id in range(scores.shape[1]):
                    mask = labs == cls_id
                    if mask.any():
                        _, keep = nms_rotated(top_boxes[mask], vals[mask], iou_threshold=iou)
                        kept_labels.append(labs[mask][keep])
            else:
                _, keep = nms_rotated(top_boxes, vals, iou_threshold=iou)
                kept_labels.append(labs[keep])
            kl = torch.cat(kept_labels) if kept_labels else torch.empty(0, device=scores.device, dtype=torch.long)
            total = int(kl.numel())
            small = int((kl == C.SMALL).sum().item())
            row = dict(
                angle=angle,
                mode=mode,
                nms_iou=iou,
                topk=k,
                detection_total=total,
                small_vehicle_count=small,
                small_vehicle_ratio=float(small / total) if total else 0.0)
            nms_rows.append(row)
    return top_rows, nms_rows


def write_report(report_dir, out_dir, dense_rows, cand_rows, top_rows, nms_rows, post_rows):
    report_dir.mkdir(parents=True, exist_ok=True)
    dense_mean = np.mean([r['dense_top1_small_vehicle_ratio'] for r in dense_rows]) if dense_rows else np.nan
    cand_03 = [r for r in cand_rows if float(r['score_thr']) == 0.3]
    cand_mean = np.mean([r['small_vehicle_ratio'] for r in cand_03]) if cand_03 else np.nan
    no_nms = [r for r in top_rows if r['mode'] == 'no_nms_global_topk']
    no_nms_mean = np.mean([r['small_vehicle_ratio'] for r in no_nms]) if no_nms else np.nan
    per_class = [r for r in top_rows if r['mode'] == 'per_class_top100']
    per_class_mean = np.mean([r['small_vehicle_ratio'] for r in per_class]) if per_class else np.nan
    aware = [r for r in nms_rows if r['mode'] == 'class_aware_nms' and float(r['nms_iou']) == 0.5]
    agnostic = [r for r in nms_rows if r['mode'] == 'class_agnostic_nms' and float(r['nms_iou']) == 0.5]
    aware_mean = np.mean([r['small_vehicle_ratio'] for r in aware]) if aware else np.nan
    agnostic_mean = np.mean([r['small_vehicle_ratio'] for r in agnostic]) if agnostic else np.nan
    post_mean = np.mean([r['small_vehicle_ratio'] for r in post_rows]) if post_rows else np.nan
    report = report_dir / 'fres_dense_prenms_nms_audit.md'
    with open(report, 'w') as f:
        f.write('# Dense Pre-NMS / No-NMS Audit\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{C.cuda_visible()}`\n')
        f.write(f'- out_dir: `{out_dir}`\n\n')
        f.write('## Key Numbers\n\n')
        f.write(f'- dense top1 small_vehicle_ratio mean: `{dense_mean:.6f}`\n')
        f.write(f'- pre-NMS candidate ratio at score_thr=0.3 mean: `{cand_mean:.6f}`\n')
        f.write(f'- no-NMS global top-k ratio mean: `{no_nms_mean:.6f}`\n')
        f.write(f'- per-class top100 ratio mean: `{per_class_mean:.6f}`\n')
        f.write(f'- class-aware NMS iou=0.5 ratio mean: `{aware_mean:.6f}`\n')
        f.write(f'- class-agnostic NMS iou=0.5 ratio mean: `{agnostic_mean:.6f}`\n')
        f.write(f'- official post-NMS ratio mean: `{post_mean:.6f}`\n\n')
        f.write('## Interpretation\n\n')
        f.write('- H3 is supported if dense/no-NMS ratios are high before NMS.\n')
        f.write('- H4 is supported as an amplifier if NMS ratios exceed no-NMS while dense is already biased.\n')
    with open(out_dir / 'gpu8_dense_summary.json', 'w') as f:
        json.dump(dict(
            report=str(report),
            dense_mean=float(dense_mean),
            pre_nms_score03_mean=float(cand_mean),
            no_nms_mean=float(no_nms_mean),
            per_class_mean=float(per_class_mean),
            class_aware_iou05_mean=float(aware_mean),
            class_agnostic_iou05_mean=float(agnostic_mean),
            post_mean=float(post_mean)), f, indent=2)
    return report


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'top_candidates').mkdir(parents=True, exist_ok=True)
    bargs, model, device, support_data, name2id, _, loader = C.build_model_and_data(args, 'visual')
    sf, sl, _ = C.build_support(model, bargs, support_data, name2id, device, 'original')
    dense_rows, cand_rows, top_rows, nms_rows, post_rows = [], [], [], [], []
    with torch.no_grad():
        for data_info in loader:
            img_path = data_info['data_samples'][0].img_path
            angle = base.angle_from_name(img_path)
            if angle not in args.angles:
                continue
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            samples = data['data_samples']
            x = model.prompt_extract_feats(data['inputs'])
            metas = [s.metainfo for s in samples]
            outs = C.model_forward_dense(model, x, sf, sl, bargs)
            cls_scores, bbox_preds, angle_preds = outs
            levels = C.decode_dense(model.bbox_head, cls_scores, bbox_preds, angle_preds, metas[0]['img_shape'])
            for lvl, logits, scores, _ in levels:
                dense_rows.extend(summarize_level(angle, lvl, logits, scores, out_dir, args.save_top_locations))
            scores, boxes, _ = flatten_scores(levels)
            cand_rows.extend(score_thr_rows(angle, scores))
            tr, nr = nms_and_topk_rows(angle, scores, boxes, args.topk, out_dir)
            top_rows.extend(tr)
            nms_rows.extend(nr)
            post_rows.append(C.post_summary(angle, 'original', model.bbox_head, outs, metas))
            print(f'DONE dense angle={angle:03d}')
    C.write_csv(out_dir / 'ftable_dense_top1_by_level_angle.csv', dense_rows, list(dense_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_prenms_candidate_sweep.csv', cand_rows, list(cand_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_score_thr_sweep.csv', cand_rows, list(cand_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_nms_mode_compare.csv', nms_rows, list(nms_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_nms_iou_sweep.csv', nms_rows, list(nms_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_no_nms_and_per_class_topk.csv', top_rows, list(top_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_official_post_nms.csv', post_rows, list(post_rows[0].keys()))
    print(write_report(report_dir, out_dir, dense_rows, cand_rows, top_rows, nms_rows, post_rows))


if __name__ == '__main__':
    main()
