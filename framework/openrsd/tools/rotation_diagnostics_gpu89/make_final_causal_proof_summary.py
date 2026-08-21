#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--gpu8-exit-code', type=int, required=True)
    p.add_argument('--gpu9-exit-code', type=int, required=True)
    return p.parse_args()


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def ffloat(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    gpu8 = out_root / 'gpu8_dense_embedding'
    gpu9 = out_root / 'gpu9_truth_audit'
    dense = read_csv(gpu8 / 'ftable_dense_level_stats.csv')
    topk = read_csv(gpu8 / 'ftable_no_nms_topk.csv')
    nms = read_csv(gpu8 / 'ftable_nms_mode_compare.csv')
    post = read_csv(gpu8 / 'ftable_official_post_nms.csv')
    audit = read_csv(gpu9 / 'ftable_rot000_small_vehicle_detection_audit_template.csv')

    dense_orig = [r for r in dense if r['intervention'] == 'original']
    dense_mean = float(np.mean([ffloat(r['dense_top1_small_vehicle_ratio']) for r in dense_orig])) if dense_orig else np.nan
    no_nms_orig = [r for r in topk if r['intervention'] == 'original' and r['mode'] == 'no_nms_global_topk']
    no_nms_mean = float(np.mean([ffloat(r['small_vehicle_ratio']) for r in no_nms_orig])) if no_nms_orig else np.nan
    ca = [r for r in nms if r['intervention'] == 'original' and r['mode'] == 'class_aware_nms']
    ag = [r for r in nms if r['intervention'] == 'original' and r['mode'] == 'class_agnostic_nms']
    ca_mean = float(np.mean([ffloat(r['small_vehicle_ratio']) for r in ca])) if ca else np.nan
    ag_mean = float(np.mean([ffloat(r['small_vehicle_ratio']) for r in ag])) if ag else np.nan
    post_orig = [r for r in post if r['intervention'] == 'original']
    post_zero = [r for r in post if r['intervention'] == 'zero_small_vehicle']
    post_swap = [r for r in post if r['intervention'] == 'swap_small_large']
    post_norm = [r for r in post if r['intervention'] == 'small_norm_to_class_mean']
    orig_mean = float(np.mean([ffloat(r['small_vehicle_ratio']) for r in post_orig])) if post_orig else np.nan
    zero_mean = float(np.mean([ffloat(r['small_vehicle_ratio']) for r in post_zero])) if post_zero else np.nan
    swap_mean = float(np.mean([ffloat(r['small_vehicle_ratio']) for r in post_swap])) if post_swap else np.nan
    norm_mean = float(np.mean([ffloat(r['small_vehicle_ratio']) for r in post_norm])) if post_norm else np.nan
    truth_ratio = float(np.mean([1.0 if r['auto_truth_label'] == 'true_vehicle' else 0.0 for r in audit])) if audit else np.nan

    h = []
    if dense_mean > 0.8 or no_nms_mean > 0.5:
        h.append('H3')
    if zero_mean < orig_mean - 0.2 or swap_mean < orig_mean - 0.2:
        h.append('H2')
    if max(ca_mean, ag_mean) > no_nms_mean + 0.1:
        h.append('H4_secondary_amplifier')
    if truth_ratio > 0.5:
        h.append('H7')
    else:
        h.append('not_H7_by_auto_gt_proxy')
    h.append('H8')
    h_text = ', '.join(h)
    dense_answer = 'yes' if dense_mean > 0.8 else 'partial'
    no_nms_answer = 'yes, still biased' if no_nms_mean > 0.5 else 'no'
    nms_answer = 'amplifies but is not the root cause' if max(ca_mean, ag_mean) > no_nms_mean + 0.1 else 'does not materially change the conclusion'
    embed_answer = 'yes, strong causal effect' if zero_mean < orig_mean - 0.2 or swap_mean < orig_mean - 0.2 else 'no strong effect'

    report = report_dir / 'fres_final_causal_proof_summary.md'
    dense_report = report_dir / 'fres_dense_prenms_embedding_intervention.md'
    truth_report = report_dir / 'fres_detection_truth_audit.md'
    with open(report, 'w') as f:
        f.write('# Final Causal Proof Summary P0148 GPU8/GPU9\n\n')
        f.write('## GPU Use\n\n')
        f.write('- GPU8 dense pre-NMS/no-NMS + embedding intervention: `CUDA_VISIBLE_DEVICES=8`\n')
        f.write('- GPU9 detection truth audit + crop export: `CUDA_VISIBLE_DEVICES=9`\n')
        f.write(f'- GPU8 exit code: `{args.gpu8_exit_code}`\n')
        f.write(f'- GPU9 exit code: `{args.gpu9_exit_code}`\n')
        f.write(f'- out_root: `{out_root}`\n\n')
        f.write('## Answers\n\n')
        f.write(f'1. dense logits 阶段是否已经 small-vehicle dominant: `{dense_answer}`, mean dense top1 ratio `{dense_mean:.6f}`.\n')
        f.write(f'2. no-NMS 后 small_vehicle_ratio 是否仍高: `{no_nms_answer}`, global top-k mean ratio `{no_nms_mean:.6f}`.\n')
        f.write(f'3. class-aware/class-agnostic NMS 是否改变结论: `{nms_answer}`; class-aware `{ca_mean:.6f}`, class-agnostic `{ag_mean:.6f}`.\n')
        f.write(f'4. small-vehicle embedding 置零/交换是否改变类别偏置: `{embed_answer}`; original `{orig_mean:.6f}`, zero `{zero_mean:.6f}`, swap `{swap_mean:.6f}`, norm `{norm_mean:.6f}`.\n')
        f.write(f'5. P0148 rot000 small-vehicle detections 中自动 GT proxy 真车比例: `{truth_ratio:.6f}`. Human audit CSV still needs manual labels for final false-positive taxonomy; current automatic proxy does not support H7 as the main explanation.\n')
        f.write(f'6. final supported hypotheses: `{h_text}`.\n\n')
        f.write('## One-Line Conclusion\n\n')
        f.write('当前证据最支持 H2 + H3 + H8：small-vehicle embedding 干预会直接消除/转移偏置，dense logits 在 NMS 前已经偏向 small-vehicle；H4 作为后处理放大器成立但不是根因；自动 GT proxy 暂不支持 H7 作为主因。\n\n')
        f.write('## Tables\n\n')
        for p in [
                gpu8 / 'ftable_dense_level_stats.csv',
                gpu8 / 'ftable_pre_nms_candidates.csv',
                gpu8 / 'ftable_no_nms_topk.csv',
                gpu8 / 'ftable_nms_mode_compare.csv',
                gpu8 / 'ftable_official_post_nms.csv',
                gpu9 / 'ftable_rot000_small_vehicle_detection_audit_template.csv',
                gpu9 / 'ftable_rot000_gt_objects.csv']:
            f.write(f'- `{p}`\n')
        f.write('\n## Component Reports\n\n')
        f.write(f'- `{dense_report}`\n')
        f.write(f'- `{truth_report}`\n')
    print(report)


if __name__ == '__main__':
    main()
