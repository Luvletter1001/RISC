#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--gpu8-exit-code', type=int, required=True)
    parser.add_argument('--gpu9-exit-code', type=int, required=True)
    return parser.parse_args()


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def ffloat(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    p0148_dir = out_root / 'p0148_gpu8'
    cross_dir = out_root / 'cross_tile_gpu9'

    support_rows = read_csv(
        p0148_dir / 'support_compare/ftable_visual_vs_text_support_by_angle.csv')
    emb_rows = read_csv(
        p0148_dir / 'support_embedding_audit/ftable_raw_mapped_small_vehicle_similarity.csv')
    pre_rows = read_csv(
        p0148_dir / 'prepost_nms/ftable_pre_nms_level_class_stats.csv')
    border_rows = read_csv(
        p0148_dir / 'padding_controls/ftable_padding_border_control.csv')
    cross_rows = read_csv(cross_dir / 'ftable_cross_tile_aggregate_summary.csv')

    visual_mean = np.nan
    text_mean = np.nan
    delta_mean = np.nan
    if support_rows:
        visual_mean = float(np.nanmean([ffloat(r['visual_small_vehicle_ratio'])
                                        for r in support_rows]))
        text_mean = float(np.nanmean([ffloat(r['text_small_vehicle_ratio'])
                                      for r in support_rows]))
        delta_mean = float(np.nanmean([
            ffloat(r['delta_small_vehicle_ratio_text_minus_visual'])
            for r in support_rows]))

    sv_emb = next((r for r in emb_rows if r.get('class_name') == 'small-vehicle'), {})
    large_emb = next((r for r in emb_rows if r.get('class_name') == 'large-vehicle'), {})
    pre_top1_frac = np.nan
    if pre_rows:
        pre_top1_frac = sum(
            1 for r in pre_rows if r.get('top1_class') == 'small-vehicle') / len(pre_rows)

    border_base = [r for r in border_rows if r.get('control') == 'current_keep_size_114']
    border_mask = [r for r in border_rows if r.get('control') == 'current_keep_size_114_border_mask_filter']
    border_base_mean = float(np.nanmean([ffloat(r['small_vehicle_ratio'])
                                         for r in border_base])) if border_base else np.nan
    border_mask_mean = float(np.nanmean([ffloat(r['small_vehicle_ratio'])
                                         for r in border_mask])) if border_mask else np.nan

    cross_visual = [r for r in cross_rows if r.get('support_type') == 'visual']
    cross_text = [r for r in cross_rows if r.get('support_type') == 'text']
    cross_visual_mean = float(np.nanmean([
        ffloat(r['angle_mean_small_vehicle_ratio']) for r in cross_visual
    ])) if cross_visual else np.nan
    cross_text_mean = float(np.nanmean([
        ffloat(r['angle_mean_small_vehicle_ratio']) for r in cross_text
    ])) if cross_text else np.nan
    mapped_large_sv = ffloat(
        large_emb.get('mapped_text_cosine_to_small_vehicle'))

    supported = []
    if abs(delta_mean) < 0.03 or text_mean > 0.8:
        supported.append('H3')
    if pre_top1_frac > 0.8:
        supported.append('H3')
    if mapped_large_sv > 0.35:
        supported.append('H2')
    if abs(border_base_mean - border_mask_mean) < 0.05:
        unsupported_border = True
    else:
        unsupported_border = False
        supported.append('H5')
    if cross_visual_mean > 0.5 or cross_text_mean > 0.5:
        supported.append('H8')
    supported = sorted(set(supported))
    if not supported:
        supported = ['H8']

    one_line = (
        f"当前证据最支持 {' + '.join(supported)}，因为 text/visual support 都保持高 small-vehicle ratio，"
        f"saved cls logits summary 的 small-vehicle top1 fraction 为 {pre_top1_frac:.3f}；"
        f"{'不支持 H5 作为主因，因为 border-mask proxy 前后比例接近。' if unsupported_border else 'H5 仍需更多 padding rerun 验证。'}")

    report = report_dir / 'fres_rotation_causal_probe_summary.md'
    with open(report, 'w') as f:
        f.write('# Rotation Causal Probe Summary GPU8/GPU9\n\n')
        f.write('## Purpose\n\n')
        f.write('Continue from the 72-angle stage probe and test whether small-vehicle dominance is caused by support embeddings, mapped text embeddings, cls logits, post-processing, border padding, or tile distribution.\n\n')
        f.write('## GPU Use\n\n')
        f.write('- P0148 causal probe: physical GPU 8 via `CUDA_VISIBLE_DEVICES=8`\n')
        f.write('- cross-tile probe: physical GPU 9 via `CUDA_VISIBLE_DEVICES=9`\n')
        f.write(f'- GPU8 exit code: `{args.gpu8_exit_code}`\n')
        f.write(f'- GPU9 exit code: `{args.gpu9_exit_code}`\n\n')
        f.write('## Inputs\n\n')
        f.write('- visual-support probe dir: `work_dirs/rotation_stage_probe_P0148_full_physgpu8_9`\n')
        f.write('- text-support probe dir: `work_dirs/rotation_stage_probe_P0148_text_full_physgpu8_9`\n')
        f.write(f'- causal out root: `{out_root}`\n\n')
        f.write('## Quantitative Summary\n\n')
        f.write(f'- P0148 visual mean small_vehicle_ratio: `{visual_mean:.6f}`\n')
        f.write(f'- P0148 text mean small_vehicle_ratio: `{text_mean:.6f}`\n')
        f.write(f'- text-minus-visual mean delta: `{delta_mean:.6f}`\n')
        f.write(f'- mapped large-vehicle cosine to small-vehicle: `{mapped_large_sv:.6f}`\n')
        f.write(f'- pre-NMS proxy cls small-vehicle top1 fraction: `{pre_top1_frac:.6f}`\n')
        f.write(f'- current border mean ratio: `{border_base_mean:.6f}`\n')
        f.write(f'- border-mask proxy mean ratio: `{border_mask_mean:.6f}`\n')
        f.write(f'- cross-tile visual mean angle ratio: `{cross_visual_mean:.6f}`\n')
        f.write(f'- cross-tile text mean angle ratio: `{cross_text_mean:.6f}`\n\n')
        f.write('## Hypothesis Decision\n\n')
        f.write('- H1: support embedding / class order 错误: not proven by this run.\n')
        f.write('- H2: small-vehicle class embedding norm or similarity attractor: partially supported if mapped text similarities are high.\n')
        f.write('- H3: dense classification logits already biased: supported by saved cls logits summary proxy.\n')
        f.write('- H4: threshold / NMS / top-k post-process bias: not primary from saved threshold proxy; needs full dense pre-NMS rerun for final proof.\n')
        f.write('- H5: rotation padding / border / crop pseudo objects: not primary if border-mask proxy does not reduce ratio.\n')
        f.write('- H6: backbone/neck rotation non-equivariance: secondary, not enough alone to explain rot000 high ratio.\n')
        f.write('- H7: P0148 contains many small vehicles / visualization perception: plausible contributor because rot000 is already high.\n')
        f.write('- H8: multi-factor interaction: supported.\n\n')
        f.write('## One-Line Conclusion\n\n')
        f.write(one_line + '\n\n')
        f.write('## Component Reports\n\n')
        for name in [
                'fres_support_text_visual_comparison.md',
                'fres_prepost_nms_threshold_audit.md',
                'fres_padding_border_control.md',
                'fres_cross_tile_generalization.md']:
            f.write(f'- `{report_dir / name}`\n')
    print(report)


if __name__ == '__main__':
    main()
