#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tools.rotation_overnight_gpu89 import common as C
from tools.rotation_diagnostics import probe_rotated_stage_outputs as base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='/data1/zcy/OpenRSD')
    p.add_argument('--image-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--angles', nargs='+', type=int, required=True)
    p.add_argument('--interventions', nargs='+', default=[
        'original', 'zero_sv', 'swap_sv_lv', 'norm_sv_mean',
        'normalize_all', 'random_sv'])
    return p.parse_args()


def report(report_dir, out_dir, rows):
    report_dir.mkdir(parents=True, exist_ok=True)
    by_int = {}
    for row in rows:
        by_int.setdefault(row['intervention'], []).append(row)
    means = {
        k: float(np.mean([r['small_vehicle_ratio'] for r in v]))
        for k, v in by_int.items()
    }
    large_means = {
        k: float(np.mean([r['large_vehicle_ratio'] for r in v]))
        for k, v in by_int.items()
    }
    path = report_dir / 'fres_embedding_intervention.md'
    with open(path, 'w') as f:
        f.write('# Small-Vehicle Embedding Intervention\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{C.cuda_visible()}`\n')
        f.write(f'- out_dir: `{out_dir}`\n\n')
        f.write('## Mean Detection Ratios\n\n')
        for name in sorted(means):
            f.write(f'- {name}: small_vehicle_ratio `{means[name]:.6f}`, large_vehicle_ratio `{large_means[name]:.6f}`\n')
        f.write('\n## Interpretation\n\n')
        f.write('- H2 is upgraded to causal evidence if zero/swap/random interventions move the dominant class distribution.\n')
        f.write('- If swap_sv_lv raises large-vehicle and lowers small-vehicle, the class embedding acts as an attractor.\n')
    with open(out_dir / 'gpu9_embedding_summary.json', 'w') as f:
        json.dump(dict(report=str(path), small_vehicle_ratio_means=means,
                       large_vehicle_ratio_means=large_means), f, indent=2)
    return path


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bargs, model, device, support_data, name2id, _, loader = C.build_model_and_data(args, 'visual')
    det_rows, norm_rows, hist_rows = [], [], []
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
            for intervention in args.interventions:
                sf, sl, nr = C.build_support(model, bargs, support_data, name2id, device, intervention)
                norm_rows.extend(nr)
                outs = C.model_forward_dense(model, x, sf, sl, bargs)
                row = C.post_summary(angle, intervention, model.bbox_head, outs, metas)
                det_rows.append(row)
                hist = json.loads(row['class_histogram'])
                for class_name, count in sorted(hist.items()):
                    hist_rows.append(dict(
                        angle=angle,
                        intervention=intervention,
                        class_name=class_name,
                        count=count,
                        detection_total=row['detection_total'],
                        ratio=float(count / row['detection_total']) if row['detection_total'] else 0.0))
                print(f'DONE embedding angle={angle:03d} intervention={intervention}')
    C.write_csv(out_dir / 'ftable_embedding_intervention_detection_summary.csv',
                det_rows, list(det_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_embedding_norms_before_after.csv',
                norm_rows, list(norm_rows[0].keys()))
    C.write_csv(out_dir / 'ftable_embedding_intervention_class_hist.csv',
                hist_rows, list(hist_rows[0].keys()) if hist_rows else [
                    'angle', 'intervention', 'class_name', 'count',
                    'detection_total', 'ratio'])
    print(report(report_dir, out_dir, det_rows))


if __name__ == '__main__':
    main()
