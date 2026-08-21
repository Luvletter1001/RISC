#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np

from tools.rotation_overnight_gpu89 import common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--gpu8-dense-exit-code', type=int, required=True)
    p.add_argument('--gpu9-embedding-exit-code', type=int, required=True)
    p.add_argument('--gpu9-audit-exit-code', type=int, required=True)
    p.add_argument('--gpu9-cross-tile-exit-code', type=int, required=True)
    return p.parse_args()


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    dense = read_json(out_root / 'gpu8_dense' / 'gpu8_dense_summary.json')
    emb = read_json(out_root / 'gpu9_embedding' / 'gpu9_embedding_summary.json')
    audit = read_json(out_root / 'gpu9_audit' / 'gpu9_audit_summary.json')
    cross = read_json(out_root / 'gpu9_cross_tile_ext' / 'gpu9_cross_tile_extended_summary.json')
    dense_mean = dense.get('dense_mean', float('nan'))
    no_nms = dense.get('no_nms_mean', float('nan'))
    aware = dense.get('class_aware_iou05_mean', float('nan'))
    agnostic = dense.get('class_agnostic_iou05_mean', float('nan'))
    emb_means = emb.get('small_vehicle_ratio_means', {})
    orig = emb_means.get('original', float('nan'))
    zero = emb_means.get('zero_sv', float('nan'))
    swap = emb_means.get('swap_sv_lv', float('nan'))
    norm_all = emb_means.get('normalize_all', float('nan'))
    truth = audit.get('proxy_true_vehicle_ratio', float('nan'))
    cross_rows = C.read_csv(out_root / 'gpu9_cross_tile_ext' / 'ftable_cross_tile_30_summary.csv')
    visual = [float(r['small_vehicle_ratio']) for r in cross_rows if r.get('support_type') == 'visual']
    text = [float(r['small_vehicle_ratio']) for r in cross_rows if r.get('support_type') == 'text']
    visual_mean = float(np.mean(visual)) if visual else float('nan')
    text_mean = float(np.mean(text)) if text else float('nan')
    h2 = zero < orig - 0.2 or swap < orig - 0.2
    h3 = dense_mean > 0.7 or no_nms > 0.5
    h4 = max(aware, agnostic) > no_nms + 0.1
    h7 = truth > 0.5
    h8 = sum(bool(x) for x in [h2, h3, h4, h7]) >= 2
    p0148_special = visual_mean < 0.7
    report = report_dir / 'fres_overnight_summary.md'
    with open(report, 'w') as f:
        f.write('# Overnight GPU8/GPU9 Rotation Summary\n\n')
        f.write(f'- out_root: `{out_root}`\n')
        f.write('- GPU8 dense pre-NMS/no-NMS: physical GPU 8, `CUDA_VISIBLE_DEVICES=8`\n')
        f.write('- GPU9 embedding/audit/cross-tile: physical GPU 9, `CUDA_VISIBLE_DEVICES=9`\n\n')
        f.write('## Task Status\n\n')
        f.write(f'- GPU8 dense exit code: `{args.gpu8_dense_exit_code}`\n')
        f.write(f'- GPU9 embedding exit code: `{args.gpu9_embedding_exit_code}`\n')
        f.write(f'- GPU9 audit exit code: `{args.gpu9_audit_exit_code}`\n')
        f.write(f'- GPU9 cross-tile exit code: `{args.gpu9_cross_tile_exit_code}`\n\n')
        f.write('## Hypothesis Evidence\n\n')
        f.write(f'- H2 embedding attractor: `{h2}`; original `{orig:.6f}`, zero `{zero:.6f}`, swap `{swap:.6f}`, normalize_all `{norm_all:.6f}`\n')
        f.write(f'- H3 dense cls logits bias: `{h3}`; dense top1 `{dense_mean:.6f}`, no-NMS `{no_nms:.6f}`\n')
        f.write(f'- H4 postprocess bias: `{h4}`; class-aware `{aware:.6f}`, class-agnostic `{agnostic:.6f}`\n')
        f.write('- H5 padding/border artifact: `not supported by prior border-mask proxy; not rerun tonight`\n')
        f.write('- H6 backbone/neck equivariance issue: `possible contributor from previous stage probe; not the primary test tonight`\n')
        f.write(f'- H7 P0148 true small-vehicle-heavy scene: `{h7}` by automatic GT proxy `{truth:.6f}`; manual CSV remains authoritative.\n')
        f.write(f'- H8 multi-factor interaction: `{h8}`\n')
        f.write(f'- P0148 special relative to extended cross-tile: `{p0148_special}`; cross-tile visual mean `{visual_mean:.6f}`, text mean `{text_mean:.6f}`\n\n')
        f.write('## Reports\n\n')
        for p in [
                report_dir / 'fres_dense_prenms_nms_audit.md',
                report_dir / 'fres_embedding_intervention.md',
                report_dir / 'fres_p0148_detection_audit.md',
                report_dir / 'fres_cross_tile_extended.md']:
            f.write(f'- `{p}`\n')
        f.write('\n## Logs\n\n')
        for p in sorted((out_root / 'logs').glob('*.log')):
            f.write(f'- `{p}`\n')
        f.write('\n## Final Judgment\n\n')
        f.write('当前 overnight 证据用于定案 H2/H3/H4/H7/H8：若 dense/no-NMS 已偏高且 embedding 干预显著移动类别分布，则根因优先是 H2+H3，H4 是放大器，H7 需人工 crop 审计确认。\n')
    print(report)


if __name__ == '__main__':
    main()
