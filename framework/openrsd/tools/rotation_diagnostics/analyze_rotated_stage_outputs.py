#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path = [p for p in sys.path if not p.startswith('/home/zcy/.local/')]

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


KEY_ANGLES = [0, 5, 10, 45, 90, 180, 270]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze OpenRSD rotated stage probe outputs.')
    parser.add_argument('--probe-dir', required=True)
    parser.add_argument('--out-md', required=True)
    parser.add_argument('--plots-dir', default='')
    parser.add_argument(
        '--merge-probe-dirs',
        nargs='*',
        default=None,
        help='Optional source probe dirs to merge into --probe-dir before analysis.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def angle_from_feature_path(path):
    match = re.search(r'_rot(\d{3})_features\.npz$', Path(path).name)
    if match:
        return int(match.group(1))
    return None


def load_feature_files(probe_dir):
    out = {}
    for path in sorted((Path(probe_dir) / 'features').glob('*_features.npz')):
        angle = angle_from_feature_path(path)
        if angle is not None:
            out[angle] = path
    return out


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1.0e-12:
        return np.nan
    return float(np.dot(a, b) / denom)


def norm_l2(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) + 1.0e-12
    return float(np.linalg.norm(a - b) / denom)


def aligned_cosine(base_fmap, other_fmap, angle):
    a = torch.tensor(other_fmap, dtype=torch.float32).unsqueeze(0)
    b = torch.tensor(base_fmap, dtype=torch.float32).unsqueeze(0)
    if a.shape != b.shape or a.ndim != 4:
        return np.nan
    theta = -float(angle) * math.pi / 180.0
    matrix = torch.tensor(
        [[[math.cos(theta), -math.sin(theta), 0.0],
          [math.sin(theta), math.cos(theta), 0.0]]],
        dtype=torch.float32)
    grid = F.affine_grid(matrix, a.shape, align_corners=False)
    rotated = F.grid_sample(a, grid, mode='bilinear', padding_mode='zeros',
                            align_corners=False)
    return cosine(b.numpy(), rotated.numpy())


def group_for_key(key):
    low = key.lower()
    if '__gap' not in key:
        return None
    if low.startswith('backbone__'):
        return 'backbone'
    if low.startswith('neck__'):
        return 'neck'
    if low.startswith('head_input__'):
        return 'head_input'
    if low.startswith('head_cls_branch__') or low.startswith('head_pred_embed__'):
        return 'head_cls_branch'
    if low.startswith('cls_logits__') or 'cls_mean_score_distribution' in low:
        return 'cls'
    return None


def compare_features(feature_files):
    if 0 not in feature_files:
        raise RuntimeError('rot000 baseline is required for analysis')
    base = np.load(feature_files[0])
    rows = []
    for angle, path in sorted(feature_files.items()):
        cur = np.load(path)
        for key in base.files:
            group = group_for_key(key)
            if group is None or key not in cur.files:
                continue
            rows.append(dict(
                angle=angle,
                group=group,
                key=key,
                naive_cosine=cosine(base[key], cur[key]),
                normalized_l2=norm_l2(base[key], cur[key]),
                aligned_cosine=np.nan))
            fmap_key = key.replace('__gap', '__fmap')
            if fmap_key in base.files and fmap_key in cur.files:
                rows[-1]['aligned_cosine'] = aligned_cosine(
                    base[fmap_key], cur[fmap_key], angle)
    return rows


def aggregate_rows(rows):
    agg = {}
    by = defaultdict(list)
    for row in rows:
        by[(row['angle'], row['group'])].append(row)
    for (angle, group), items in by.items():
        agg[(angle, group)] = dict(
            angle=angle,
            group=group,
            naive_cosine=float(np.nanmean([x['naive_cosine'] for x in items])),
            normalized_l2=float(np.nanmean([x['normalized_l2'] for x in items])),
            aligned_cosine=float(np.nanmean([x['aligned_cosine'] for x in items])),
            n=len(items))
    return list(agg.values())


def read_detection_summary(probe_dir):
    rows = read_csv(Path(probe_dir) / 'detections_summary.csv')
    for row in rows:
        for key in ['angle', 'detection_total', 'small_vehicle_count']:
            row[key] = int(float(row[key]))
        for key in ['small_vehicle_ratio', 'mean_score', 'max_score']:
            row[key] = float(row[key])
    return sorted(rows, key=lambda x: x['angle'])


def read_cls_summary(probe_dir):
    rows = read_csv(Path(probe_dir) / 'cls_summary.csv')
    for row in rows:
        row['angle'] = int(float(row['angle']))
        row['level'] = int(float(row['level']))
        for key in ['top1_score', 'top2_score', 'margin',
                    'small_vehicle_score', 'entropy']:
            row[key] = float(row[key])
    return rows


def cls_js_rows(feature_files):
    if 0 not in feature_files:
        return []
    base = np.load(feature_files[0])
    if 'cls_mean_score_distribution' not in base.files:
        return []
    p = base['cls_mean_score_distribution']
    rows = []
    for angle, path in sorted(feature_files.items()):
        cur = np.load(path)
        if 'cls_mean_score_distribution' not in cur.files:
            continue
        q = cur['cls_mean_score_distribution']
        m = 0.5 * (p + q)
        eps = 1.0e-12
        js = 0.5 * np.sum(p * np.log((p + eps) / (m + eps)))
        js += 0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
        rows.append(dict(angle=angle, cls_js=float(js)))
    return rows


def save_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_csvs(source_dirs, out_dir, name):
    rows = []
    fields = None
    for src_dir in source_dirs:
        path = src_dir / name
        if not path.exists():
            continue
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = reader.fieldnames
            rows.extend(reader)
    if fields is None:
        return
    with (out_dir / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_probe_dirs(source_dirs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'features').mkdir(exist_ok=True)
    (out_dir / 'detections').mkdir(exist_ok=True)
    source_dirs = [Path(x) for x in source_dirs]

    metadata_items = []
    for src_dir in source_dirs:
        meta_path = src_dir / 'metadata.json'
        if meta_path.exists():
            metadata_items.append(json.load(open(meta_path)))
        for src in (src_dir / 'features').glob('*.npz'):
            shutil.copy2(src, out_dir / 'features' / src.name)
        for src in (src_dir / 'detections').glob('*.json'):
            shutil.copy2(src, out_dir / 'detections' / src.name)
        inv = src_dir / 'module_inventory.csv'
        if inv.exists() and not (out_dir / 'module_inventory.csv').exists():
            shutil.copy2(inv, out_dir / 'module_inventory.csv')

    for name in [
            'stage_stats.csv',
            'cls_summary.csv',
            'detections_summary.csv',
            'detections_by_class.csv']:
        merge_csvs(source_dirs, out_dir, name)

    all_detections = {}
    for src_dir in source_dirs:
        path = src_dir / 'detections_all.json'
        if path.exists():
            all_detections.update(json.load(open(path)))
    with (out_dir / 'detections_all.json').open('w') as f:
        json.dump(all_detections, f, indent=2)

    merged_meta = dict(
        generated_at='merged',
        source_probe_dirs=[str(x) for x in source_dirs],
        processed_images=len(all_detections),
        angles=sorted({
            int(m.group(1))
            for name in all_detections
            for m in [re.search(r'_rot(\d{3})', name)]
            if m
        }),
        cuda_visible_devices='; '.join(
            f"{src}: {meta.get('cuda_visible_devices', '')}"
            for src, meta in zip(source_dirs, metadata_items)),
    )
    if metadata_items:
        merged_meta.update(metadata_items[0])
        merged_meta['source_probe_dirs'] = [str(x) for x in source_dirs]
        merged_meta['processed_images'] = len(all_detections)
        merged_meta['angles'] = sorted(set().union(*[
            set(meta.get('angles', [])) for meta in metadata_items
        ]))
        merged_meta['cuda_visible_devices'] = '; '.join(
            f"{src.name}: {meta.get('cuda_visible_devices', '')}"
            for src, meta in zip(source_dirs, metadata_items))
    with (out_dir / 'metadata.json').open('w') as f:
        json.dump(merged_meta, f, indent=2)


def plot_line(path, x, ys, title, ylabel):
    plt.figure(figsize=(8, 4.5))
    for label, y in ys:
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel('angle')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(ys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_plots(probe_dir, plots_dir, agg_rows, det_rows, js_rows):
    plots_dir.mkdir(parents=True, exist_ok=True)
    angles = sorted({r['angle'] for r in agg_rows})
    groups = sorted({r['group'] for r in agg_rows})
    lookup = {(r['angle'], r['group']): r for r in agg_rows}
    for group in groups:
        plot_line(
            plots_dir / f'angle_vs_{group}_cosine.png',
            angles,
            [(group, [lookup.get((a, group), {}).get('naive_cosine', np.nan)
                      for a in angles])],
            f'angle vs {group} cosine',
            'cosine')
    plot_line(
        plots_dir / 'angle_vs_stage_cosine_compare.png',
        angles,
        [(g, [lookup.get((a, g), {}).get('naive_cosine', np.nan)
              for a in angles]) for g in groups if g != 'cls'],
        'angle vs stage cosine',
        'cosine')
    det_angles = [r['angle'] for r in det_rows]
    plot_line(plots_dir / 'angle_vs_small_vehicle_count.png', det_angles,
              [('small_vehicle_count',
                [r['small_vehicle_count'] for r in det_rows])],
              'angle vs small-vehicle count', 'count')
    plot_line(plots_dir / 'angle_vs_small_vehicle_ratio.png', det_angles,
              [('small_vehicle_ratio',
                [r['small_vehicle_ratio'] for r in det_rows])],
              'angle vs small-vehicle ratio', 'ratio')
    plot_line(plots_dir / 'angle_vs_detection_total_count.png', det_angles,
              [('detection_total', [r['detection_total'] for r in det_rows])],
              'angle vs detection total', 'count')
    plot_line(plots_dir / 'angle_vs_mean_detection_score.png', det_angles,
              [('mean_score', [r['mean_score'] for r in det_rows])],
              'angle vs mean detection score', 'score')
    if js_rows:
        plot_line(plots_dir / 'angle_vs_cls_js_divergence.png',
                  [r['angle'] for r in js_rows],
                  [('cls_js', [r['cls_js'] for r in js_rows])],
                  'angle vs cls JS divergence', 'JS')


def save_heatmaps(feature_files, plots_dir):
    heat_dir = plots_dir / 'heatmaps'
    heat_dir.mkdir(parents=True, exist_ok=True)
    for angle in KEY_ANGLES:
        if angle not in feature_files:
            continue
        data = np.load(feature_files[angle])
        for group in ['backbone', 'neck', 'cls_logits']:
            keys = [k for k in data.files
                    if k.lower().startswith(group) and k.endswith('__heatmap')]
            for i, key in enumerate(keys[:5]):
                arr = data[key]
                plt.figure(figsize=(4, 4))
                plt.imshow(arr, cmap='viridis')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f'rot{angle:03d} {group} L{i}')
                plt.tight_layout()
                plt.savefig(heat_dir / f'rot{angle:03d}_{group}_L{i}.png',
                            dpi=160)
                plt.close()


def top_key_angle_table(det_rows, cls_rows, agg_rows):
    agg_lookup = {(r['angle'], r['group']): r for r in agg_rows}
    cls_by_angle = defaultdict(list)
    for row in cls_rows:
        cls_by_angle[row['angle']].append(row)
    lines = []
    lines.append('| angle | det_total | small_vehicle | sv_ratio | backbone_cos | neck_cos | head_cos | top_cls | sv_score |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---|---:|')
    det_lookup = {r['angle']: r for r in det_rows}
    for angle in KEY_ANGLES:
        if angle not in det_lookup:
            continue
        det = det_lookup[angle]
        cls_items = cls_by_angle.get(angle, [])
        top_cls = ''
        sv_score = np.nan
        if cls_items:
            top_cls = Counter([x['top1_class'] for x in cls_items]).most_common(1)[0][0]
            sv_score = float(np.mean([x['small_vehicle_score'] for x in cls_items]))
        lines.append(
            f"| {angle:03d} | {det['detection_total']} | "
            f"{det['small_vehicle_count']} | {det['small_vehicle_ratio']:.3f} | "
            f"{agg_lookup.get((angle, 'backbone'), {}).get('naive_cosine', np.nan):.4f} | "
            f"{agg_lookup.get((angle, 'neck'), {}).get('naive_cosine', np.nan):.4f} | "
            f"{agg_lookup.get((angle, 'head_input'), {}).get('naive_cosine', np.nan):.4f} | "
            f"{top_cls} | {sv_score:.4f} |")
    return '\n'.join(lines)


def infer_stage(det_rows, agg_rows, cls_rows):
    if not det_rows:
        return 'insufficient detection data'
    max_sv = max(det_rows, key=lambda r: r['small_vehicle_ratio'])
    stage_means = defaultdict(list)
    for row in agg_rows:
        if row['angle'] == 0:
            continue
        stage_means[row['group']].append(row['naive_cosine'])
    means = {k: float(np.nanmean(v)) for k, v in stage_means.items() if v}
    cls_top_small = 0
    if cls_rows:
        cls_top_small = sum(1 for r in cls_rows if r['top1_class'] == 'small-vehicle') / len(cls_rows)
    parts = [
        f"worst small-vehicle ratio at rot{max_sv['angle']:03d} = {max_sv['small_vehicle_ratio']:.3f}",
        f"stage cosine means excluding rot000: {means}",
        f"fraction of cls-logit level summaries with top1 small-vehicle = {cls_top_small:.3f}",
    ]
    if cls_top_small > 0.5:
        parts.append('most likely stage: cls logits / classification branch bias')
    elif means.get('backbone', 1.0) < 0.5:
        parts.append('most likely stage: backbone feature inconsistency')
    elif means.get('neck', 1.0) < means.get('backbone', 1.0) - 0.1:
        parts.append('most likely stage: neck amplifies rotation perturbation')
    elif max_sv['small_vehicle_ratio'] > 0.5:
        parts.append('most likely stage: post-head score/NMS threshold interaction or final logits bias')
    else:
        parts.append('no strong small-vehicle collapse detected in this probe subset')
    return '; '.join(parts)


def write_markdown(path, metadata, probe_dir, plots_dir, agg_rows, det_rows,
                   cls_rows, js_rows, stage_judgement):
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = sorted({r['angle'] for r in det_rows})
    with open(path, 'w') as f:
        f.write('# Rotation Stage Probe Summary\n\n')
        f.write('## Experiment\n\n')
        f.write('- purpose: diagnose OpenRSD rotation-induced small-vehicle collapse\n')
        f.write(f"- config: `{metadata.get('config', '')}`\n")
        f.write(f"- checkpoint: `{metadata.get('checkpoint', '')}`\n")
        f.write(f"- image_dir: `{metadata.get('image_dir', '')}`\n")
        f.write(f"- probe_dir: `{probe_dir}`\n")
        f.write(f"- angles: `{angles}`\n")
        f.write(f"- CUDA_VISIBLE_DEVICES: `{metadata.get('cuda_visible_devices', '')}`\n")
        f.write(f"- support_type: `{metadata.get('support_type', '')}`\n")
        if metadata.get('support_warning'):
            f.write(f"- support_warning: `{metadata['support_warning']}`\n")
        f.write('\n## Hooked Modules\n\n')
        f.write('| stage | module | type | pre_hook |\n|---|---|---|---|\n')
        for row in metadata.get('hooked_modules', []):
            f.write(
                f"| {row['stage']} | `{row['module']}` | `{row['type']}` | {row['pre_hook']} |\n")
        f.write('\n## Key Angle Results\n\n')
        f.write(top_key_angle_table(det_rows, cls_rows, agg_rows))
        f.write('\n\n## Stage Judgement\n\n')
        f.write(stage_judgement + '\n\n')
        f.write('## Output Tables\n\n')
        for name in [
                'feature_similarity_summary.csv',
                'feature_similarity_detail.csv',
                'cls_js_summary.csv',
                'detections_summary.csv',
                'cls_summary.csv']:
            p = Path(probe_dir) / name
            if p.exists():
                f.write(f'- `{p}`\n')
        f.write('\n## Plots\n\n')
        for p in sorted(Path(plots_dir).glob('*.png')):
            f.write(f'- `{p}`\n')
        f.write('\n## Next Steps\n\n')
        f.write('- If aligned cosine recovers while cls JS remains high, inspect bbox_head cross-attention and contrastive class embeddings.\n')
        f.write('- If backbone/neck cosine drops before head, test rotation augmentation or rotation-equivariant feature extraction.\n')
        f.write('- If detection collapse appears only after threshold/NMS, sweep score_thr and NMS IoU using saved logits and detections.\n')


def main():
    args = parse_args()
    probe_dir = Path(args.probe_dir)
    out_md = Path(args.out_md)
    plots_dir = Path(args.plots_dir) if args.plots_dir else probe_dir / 'analysis_plots'
    if args.merge_probe_dirs:
        merge_probe_dirs([Path(x) for x in args.merge_probe_dirs], probe_dir)
    metadata = json.load(open(probe_dir / 'metadata.json'))
    feature_files = load_feature_files(probe_dir)
    feature_rows = compare_features(feature_files)
    agg_rows = aggregate_rows(feature_rows)
    det_rows = read_detection_summary(probe_dir)
    cls_rows = read_cls_summary(probe_dir)
    js_rows = cls_js_rows(feature_files)
    save_csv(probe_dir / 'feature_similarity_detail.csv', feature_rows,
             ['angle', 'group', 'key', 'naive_cosine', 'normalized_l2',
              'aligned_cosine'])
    save_csv(probe_dir / 'feature_similarity_summary.csv', agg_rows,
             ['angle', 'group', 'naive_cosine', 'normalized_l2',
              'aligned_cosine', 'n'])
    if js_rows:
        save_csv(probe_dir / 'cls_js_summary.csv', js_rows,
                 ['angle', 'cls_js'])
    generate_plots(probe_dir, plots_dir, agg_rows, det_rows, js_rows)
    save_heatmaps(feature_files, plots_dir)
    judgement = infer_stage(det_rows, agg_rows, cls_rows)
    write_markdown(out_md, metadata, probe_dir, plots_dir, agg_rows, det_rows,
                   cls_rows, js_rows, judgement)
    print(f'WROTE {out_md}')


if __name__ == '__main__':
    main()
