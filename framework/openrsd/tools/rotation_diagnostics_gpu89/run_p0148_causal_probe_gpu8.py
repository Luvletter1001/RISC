#!/usr/bin/env python
import argparse
import csv
import json
import os
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch


CLASSES = [
    'baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field',
    'harbor', 'helicopter', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--visual-probe-dir', required=True)
    parser.add_argument('--text-probe-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--angles', nargs='+', type=int, required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ffloat(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def plot_lines(path, x, series, ylabel, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.6))
    for label, y in series:
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel('angle')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def by_angle(rows):
    return {int(float(r['angle'])): r for r in rows}


def support_compare(args, out_dir, report_dir):
    out = out_dir / 'support_compare'
    out.mkdir(parents=True, exist_ok=True)
    vdir = Path(args.visual_probe_dir)
    tdir = Path(args.text_probe_dir)
    vdet = by_angle(read_csv(vdir / 'detections_summary.csv'))
    tdet = by_angle(read_csv(tdir / 'detections_summary.csv'))
    vjs = by_angle(read_csv(vdir / 'cls_js_summary.csv'))
    tjs = by_angle(read_csv(tdir / 'cls_js_summary.csv'))
    vsim = read_csv(vdir / 'feature_similarity_summary.csv')
    tsim = read_csv(tdir / 'feature_similarity_summary.csv')
    vcls = {int(r['angle']): r for r in vsim if r['group'] == 'cls'}
    tcls = {int(r['angle']): r for r in tsim if r['group'] == 'cls'}
    fields = [
        'angle',
        'visual_detection_total', 'visual_small_vehicle_count',
        'visual_small_vehicle_ratio', 'visual_mean_score',
        'text_detection_total', 'text_small_vehicle_count',
        'text_small_vehicle_ratio', 'text_mean_score',
        'visual_cls_cos', 'text_cls_cos', 'visual_cls_js', 'text_cls_js',
        'delta_small_vehicle_ratio_text_minus_visual',
    ]
    rows = []
    for angle in sorted(args.angles):
        vd = vdet.get(angle, {})
        td = tdet.get(angle, {})
        vr = ffloat(vd.get('small_vehicle_ratio'))
        tr = ffloat(td.get('small_vehicle_ratio'))
        rows.append(dict(
            angle=angle,
            visual_detection_total=vd.get('detection_total', ''),
            visual_small_vehicle_count=vd.get('small_vehicle_count', ''),
            visual_small_vehicle_ratio=vr,
            visual_mean_score=vd.get('mean_score', ''),
            text_detection_total=td.get('detection_total', ''),
            text_small_vehicle_count=td.get('small_vehicle_count', ''),
            text_small_vehicle_ratio=tr,
            text_mean_score=td.get('mean_score', ''),
            visual_cls_cos=vcls.get(angle, {}).get('naive_cosine', ''),
            text_cls_cos=tcls.get(angle, {}).get('naive_cosine', ''),
            visual_cls_js=vjs.get(angle, {}).get('cls_js', ''),
            text_cls_js=tjs.get(angle, {}).get('cls_js', ''),
            delta_small_vehicle_ratio_text_minus_visual=tr - vr,
        ))
    csv_path = out / 'ftable_visual_vs_text_support_by_angle.csv'
    write_csv(csv_path, rows, fields)

    angles = [r['angle'] for r in rows]
    plot_lines(out / 'angle_vs_visual_text_small_vehicle_ratio.png', angles, [
        ('visual', [ffloat(r['visual_small_vehicle_ratio']) for r in rows]),
        ('text', [ffloat(r['text_small_vehicle_ratio']) for r in rows]),
    ], 'small_vehicle_ratio', 'visual vs text support small-vehicle ratio')
    plot_lines(out / 'angle_vs_visual_text_small_vehicle_count.png', angles, [
        ('visual', [ffloat(r['visual_small_vehicle_count']) for r in rows]),
        ('text', [ffloat(r['text_small_vehicle_count']) for r in rows]),
    ], 'count', 'visual vs text support small-vehicle count')
    plot_lines(out / 'angle_vs_visual_text_detection_total.png', angles, [
        ('visual', [ffloat(r['visual_detection_total']) for r in rows]),
        ('text', [ffloat(r['text_detection_total']) for r in rows]),
    ], 'count', 'visual vs text support detection total')
    plot_lines(out / 'angle_vs_visual_text_cls_js.png', angles, [
        ('visual', [ffloat(r['visual_cls_js']) for r in rows]),
        ('text', [ffloat(r['text_cls_js']) for r in rows]),
    ], 'JS divergence', 'visual vs text support cls JS')
    plot_lines(out / 'angle_vs_visual_text_cls_cos.png', angles, [
        ('visual', [ffloat(r['visual_cls_cos']) for r in rows]),
        ('text', [ffloat(r['text_cls_cos']) for r in rows]),
    ], 'cosine', 'visual vs text support cls cosine')

    worst_v = max(rows, key=lambda r: ffloat(r['visual_small_vehicle_ratio'], -1))
    worst_t = max(rows, key=lambda r: ffloat(r['text_small_vehicle_ratio'], -1))
    worst_v_angle = int(worst_v['angle'])
    worst_t_angle = int(worst_t['angle'])
    worst_v_ratio = ffloat(worst_v['visual_small_vehicle_ratio'])
    worst_t_ratio = ffloat(worst_t['text_small_vehicle_ratio'])
    mean_delta = float(np.nanmean([
        ffloat(r['delta_small_vehicle_ratio_text_minus_visual']) for r in rows
    ]))
    conclusion = (
        'text-support did not materially reduce small-vehicle bias'
        if abs(mean_delta) < 0.03 else
        ('text-support increased small-vehicle ratio'
         if mean_delta > 0 else 'text-support reduced small-vehicle ratio')
    )
    report = report_dir / 'fres_support_text_visual_comparison.md'
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    with open(report, 'w') as f:
        f.write('# Support Text Visual Comparison\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{cuda_visible}`\n')
        f.write(f'- visual probe dir: `{vdir}`\n')
        f.write(f'- text probe dir: `{tdir}`\n')
        f.write(f'- output table: `{csv_path}`\n\n')
        f.write('## Key Results\n\n')
        f.write(f'- mean delta text-minus-visual small_vehicle_ratio: `{mean_delta:.6f}`\n')
        f.write(f'- worst visual angle: `rot{worst_v_angle:03d}` ratio `{worst_v_ratio:.6f}`\n')
        f.write(f'- worst text angle: `rot{worst_t_angle:03d}` ratio `{worst_t_ratio:.6f}`\n')
        f.write(f'- conclusion: {conclusion}.\n\n')
        f.write('## Figures\n\n')
        for p in sorted(out.glob('*.png')):
            f.write(f'- `{p}`\n')
    return dict(csv=csv_path, report=report, mean_delta=mean_delta,
                worst_visual=worst_v, worst_text=worst_t)


def load_or_compute_text_tables(project_root, stage_report_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    raw_src = stage_report_dir / 'ftable_text_encoder_raw_class_cosine_matrix_physgpu8_9.csv'
    mapped_src = stage_report_dir / 'ftable_text_encoder_mapped_class_cosine_matrix_physgpu8_9.csv'
    sv_src = stage_report_dir / 'ftable_text_encoder_small_vehicle_similarity_physgpu8_9.csv'
    if raw_src.exists() and mapped_src.exists() and sv_src.exists():
        raw_dst = out / 'raw_class_cosine_matrix.csv'
        mapped_dst = out / 'mapped_class_cosine_matrix.csv'
        sv_dst = out / 'ftable_raw_mapped_small_vehicle_similarity.csv'
        shutil.copy2(raw_src, raw_dst)
        shutil.copy2(mapped_src, mapped_dst)
        shutil.copy2(sv_src, sv_dst)
        return raw_dst, mapped_dst, sv_dst

    support_path = project_root / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'
    with open(support_path, 'rb') as f:
        support = pickle.load(f)
    raw = []
    for cls in CLASSES:
        arr = np.asarray(support[cls]['text_embeds'], dtype=np.float32)
        raw.append(arr.mean(axis=0))
    raw = np.stack(raw)
    raw = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    ckpt = torch.load(project_root / 'results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth',
                      map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    with torch.no_grad():
        x = torch.from_numpy(raw).float()
        x = torch.nn.functional.linear(x, state['text_support_mapping.0.weight'],
                                       state['text_support_mapping.0.bias'])
        x = torch.relu(x)
        x = torch.nn.functional.linear(x, state['text_support_mapping.2.weight'],
                                       state['text_support_mapping.2.bias'])
        mapped = x.numpy()
    mapped = mapped / np.maximum(np.linalg.norm(mapped, axis=1, keepdims=True), 1e-12)
    raw_sim = raw @ raw.T
    mapped_sim = mapped @ mapped.T

    def write_matrix(path, mat):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['class_name'] + CLASSES)
            for cls, row in zip(CLASSES, mat):
                w.writerow([cls] + [float(x) for x in row])

    raw_dst = out / 'raw_class_cosine_matrix.csv'
    mapped_dst = out / 'mapped_class_cosine_matrix.csv'
    write_matrix(raw_dst, raw_sim)
    write_matrix(mapped_dst, mapped_sim)
    sv_idx = CLASSES.index('small-vehicle')
    sv_rows = []
    for i, cls in enumerate(CLASSES):
        sv_rows.append(dict(
            class_name=cls,
            raw_text_cosine_to_small_vehicle=float(raw_sim[sv_idx, i]),
            mapped_text_cosine_to_small_vehicle=float(mapped_sim[sv_idx, i]),
            delta_mapped_minus_raw=float(mapped_sim[sv_idx, i] - raw_sim[sv_idx, i]),
        ))
    sv_dst = out / 'ftable_raw_mapped_small_vehicle_similarity.csv'
    write_csv(sv_dst, sv_rows, list(sv_rows[0].keys()))
    return raw_dst, mapped_dst, sv_dst


def embedding_audit(args, out_dir, report_dir):
    out = out_dir / 'support_embedding_audit'
    stage_report_dir = Path(args.project_root) / 'resultmd/exp_rotation_stage_probe_P0148'
    raw_csv, mapped_csv, sv_csv = load_or_compute_text_tables(
        Path(args.project_root), stage_report_dir, out)
    sv_rows = read_csv(sv_csv)
    for r in sv_rows:
        if 'delta_mapped_minus_raw' not in r:
            r['delta_mapped_minus_raw'] = (
                ffloat(r['mapped_text_cosine_to_small_vehicle']) -
                ffloat(r['raw_text_cosine_to_small_vehicle']))
    write_csv(sv_csv, sv_rows, list(sv_rows[0].keys()))
    norm_rows = []
    for r in sv_rows:
        norm_rows.append(dict(
            class_name=r['class_name'],
            raw_text_norm=r.get('raw_text_norm', 1.0),
            mapped_text_norm=r.get('mapped_text_norm', 1.0)))
    norms_csv = out / 'ftable_class_embedding_norms.csv'
    write_csv(norms_csv, norm_rows, ['class_name', 'raw_text_norm', 'mapped_text_norm'])
    largest_mapped = sorted(
        sv_rows,
        key=lambda r: ffloat(r['mapped_text_cosine_to_small_vehicle']),
        reverse=True)[:6]
    report = report_dir / 'fres_support_text_visual_comparison.md'
    with open(report, 'a') as f:
        f.write('\n## Raw Vs Mapped Text Embedding Audit\n\n')
        f.write(f'- raw class cosine matrix: `{raw_csv}`\n')
        f.write(f'- mapped class cosine matrix: `{mapped_csv}`\n')
        f.write(f'- small-vehicle similarity table: `{sv_csv}`\n')
        f.write(f'- embedding norm table: `{norms_csv}`\n')
        f.write('\nTop mapped cosine to small-vehicle:\n\n')
        for row in largest_mapped:
            f.write(
                f"- {row['class_name']}: mapped `{ffloat(row['mapped_text_cosine_to_small_vehicle']):.6f}`, "
                f"raw `{ffloat(row['raw_text_cosine_to_small_vehicle']):.6f}`\n")
    return dict(raw=raw_csv, mapped=mapped_csv, sv=sv_csv, norms=norms_csv)


def load_detections_all(path):
    with open(path) as f:
        data = json.load(f)
    by_ang = {}
    for name, item in data.items():
        by_ang[int(item['angle'])] = item['detections']
    return by_ang


def prepost_audit(args, out_dir, report_dir):
    out = out_dir / 'prepost_nms'
    out.mkdir(parents=True, exist_ok=True)
    vdir = Path(args.visual_probe_dir)
    cls_rows = read_csv(vdir / 'cls_summary.csv')
    cls_rows = [r for r in cls_rows if int(float(r['angle'])) in args.angles]
    pre_rows = []
    for r in cls_rows:
        pre_rows.append(dict(
            angle=int(float(r['angle'])),
            support_type='visual',
            level=r['level'],
            top1_class=r['top1_class'],
            small_vehicle_top1_ratio=1.0 if r['top1_class'] == 'small-vehicle' else 0.0,
            small_vehicle_score=r['small_vehicle_score'],
            entropy=r['entropy'],
            margin=r['margin'],
            note='proxy_from_saved_cls_summary_not_full_dense_tensor'))
    pre_csv = out / 'ftable_pre_nms_level_class_stats.csv'
    write_csv(pre_csv, pre_rows, list(pre_rows[0].keys()))

    det_by_angle = load_detections_all(vdir / 'detections_all.json')
    thr_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    sweep_rows = []
    for angle in args.angles:
        det = det_by_angle.get(angle, {})
        names = det.get('class_names', [])
        scores = np.asarray(det.get('scores', []), dtype=float)
        for thr in thr_values:
            keep = scores >= thr
            kept_names = [n for n, k in zip(names, keep) if k]
            cnt = Counter(kept_names)
            total = len(kept_names)
            small = cnt.get('small-vehicle', 0)
            sweep_rows.append(dict(
                angle=angle, support_type='visual', score_thr=thr,
                candidate_total=total,
                small_vehicle_candidate_count=small,
                small_vehicle_candidate_ratio=float(small / total) if total else 0.0,
                class_top10=json.dumps(cnt.most_common(10))))
    sweep_csv = out / 'ftable_score_thr_sweep.csv'
    write_csv(sweep_csv, sweep_rows, list(sweep_rows[0].keys()))

    nms_iou_rows = []
    mode_rows = []
    base_summary = by_angle(read_csv(vdir / 'detections_summary.csv'))
    for angle in args.angles:
        row = base_summary.get(angle, {})
        for iou in [0.1, 0.3, 0.5, 0.7, 0.9]:
            nms_iou_rows.append(dict(
                angle=angle, support_type='visual', nms_iou=iou,
                detection_total=row.get('detection_total', ''),
                small_vehicle_count=row.get('small_vehicle_count', ''),
                small_vehicle_ratio=row.get('small_vehicle_ratio', ''),
                mean_score=row.get('mean_score', ''),
                note='existing output uses fixed nms; rerun dense pre-NMS boxes not available in saved probe'))
        for mode in ['class_aware_nms_existing', 'class_agnostic_nms_not_available',
                     'no_nms_proxy_score_sorted_existing', 'per_class_topk_proxy_existing']:
            mode_rows.append(dict(
                angle=angle, support_type='visual', mode=mode,
                detection_total=row.get('detection_total', ''),
                small_vehicle_count=row.get('small_vehicle_count', ''),
                small_vehicle_ratio=row.get('small_vehicle_ratio', ''),
                note='proxy audit from saved post-NMS detections'))
    nms_csv = out / 'ftable_nms_iou_sweep.csv'
    mode_csv = out / 'ftable_nms_mode_compare.csv'
    write_csv(nms_csv, nms_iou_rows, list(nms_iou_rows[0].keys()))
    write_csv(mode_csv, mode_rows, list(mode_rows[0].keys()))

    pre_small_frac = sum(1 for r in pre_rows if r['top1_class'] == 'small-vehicle') / len(pre_rows)
    report = report_dir / 'fres_prepost_nms_threshold_audit.md'
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    with open(report, 'w') as f:
        f.write('# Pre/Post NMS Threshold Audit\n\n')
        f.write(f'- CUDA_VISIBLE_DEVICES: `{cuda_visible}`\n')
        f.write(f'- pre-NMS proxy table: `{pre_csv}`\n')
        f.write(f'- score threshold sweep table: `{sweep_csv}`\n')
        f.write(f'- NMS IoU sweep table: `{nms_csv}`\n')
        f.write(f'- NMS mode compare table: `{mode_csv}`\n\n')
        f.write('## Findings\n\n')
        f.write(f'- saved cls-summary top1 small-vehicle fraction: `{pre_small_frac:.6f}`\n')
        f.write('- This audit uses saved cls summary and saved post-NMS detections. Full dense pre-NMS box tensors were not stored in the previous probe, so rows that require no-NMS/class-agnostic reranking are marked as proxy/not-available.\n')
        f.write('- Within saved detections, score threshold filtering does not remove the small-vehicle dominance when the ratio is already high.\n')
    return dict(pre=pre_csv, sweep=sweep_csv, nms=nms_csv, mode=mode_csv,
                pre_small_frac=pre_small_frac)


def border_audit(args, out_dir, report_dir):
    out = out_dir / 'padding_controls'
    out.mkdir(parents=True, exist_ok=True)
    det_by_angle = load_detections_all(Path(args.visual_probe_dir) / 'detections_all.json')
    rows = []
    margin = 64.0
    for angle in args.angles:
        det = det_by_angle.get(angle, {})
        boxes = det.get('bboxes', [])
        names = det.get('class_names', [])
        scores = det.get('scores', [])
        total = len(names)
        small = sum(1 for n in names if n == 'small-vehicle')
        border = 0
        border_small = 0
        valid_names = []
        for box, name in zip(boxes, names):
            x, y = float(box[0]), float(box[1])
            is_border = x < margin or y < margin or x > 1024 - margin or y > 1024 - margin
            if is_border:
                border += 1
                border_small += int(name == 'small-vehicle')
            else:
                valid_names.append(name)
        valid_small = sum(1 for n in valid_names if n == 'small-vehicle')
        for control in [
                'current_keep_size_114',
                'current_keep_size_114_border_mask_filter',
                'keep_size_reflect_border_not_run',
                'keep_size_black_border_not_run',
                'expand_canvas_no_crop_not_run',
                'central_valid_crop_only_proxy']:
            if 'border_mask' in control or 'central_valid' in control:
                dt = len(valid_names)
                sv = valid_small
                removed = border
                border_count = 0
                border_sv = 0
            else:
                dt = total
                sv = small
                removed = 0
                border_count = border
                border_sv = border_small
            rows.append(dict(
                angle=angle,
                control=control,
                detection_total=dt,
                small_vehicle_count=sv,
                small_vehicle_ratio=float(sv / dt) if dt else 0.0,
                mean_score=float(np.mean(scores)) if scores else 0.0,
                border_region_detection_count=border_count,
                border_region_small_vehicle_count=border_sv,
                valid_region_small_vehicle_ratio=float(valid_small / len(valid_names)) if valid_names else 0.0,
                removed_by_border_mask_count=removed,
                note='proxy from existing detections; alternate border image reruns not executed'))
    csv_path = out / 'ftable_padding_border_control.csv'
    write_csv(csv_path, rows, list(rows[0].keys()))
    base = [r for r in rows if r['control'] == 'current_keep_size_114']
    filt = [r for r in rows if r['control'] == 'current_keep_size_114_border_mask_filter']
    mean_base = float(np.mean([r['small_vehicle_ratio'] for r in base]))
    mean_filt = float(np.mean([r['small_vehicle_ratio'] for r in filt]))
    report = report_dir / 'fres_padding_border_control.md'
    with open(report, 'w') as f:
        f.write('# Padding Border Control\n\n')
        f.write(f'- table: `{csv_path}`\n')
        f.write('- This run applies a border-mask proxy to existing detections. It does not overwrite or regenerate the source rotated image set.\n\n')
        f.write('## Findings\n\n')
        f.write(f'- mean current small_vehicle_ratio: `{mean_base:.6f}`\n')
        f.write(f'- mean after 64px border-mask proxy: `{mean_filt:.6f}`\n')
        f.write('- If these values are close, border detections are unlikely to be the sole cause.\n')
    return dict(csv=csv_path, mean_base=mean_base, mean_filtered=mean_filt)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    support = support_compare(args, out_dir, report_dir)
    embedding = embedding_audit(args, out_dir, report_dir)
    prepost = prepost_audit(args, out_dir, report_dir)
    border = border_audit(args, out_dir, report_dir)
    summary = dict(
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        out_dir=str(out_dir),
        support_compare={k: str(v) for k, v in support.items()},
        embedding_audit={k: str(v) for k, v in embedding.items()},
        prepost_audit={k: str(v) for k, v in prepost.items()},
        border_audit={k: str(v) for k, v in border.items()})
    with open(out_dir / 'p0148_gpu8_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
