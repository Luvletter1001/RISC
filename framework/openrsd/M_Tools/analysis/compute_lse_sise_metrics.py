#!/usr/bin/env python
"""Summarize localized semantic errors and scale-impossible errors.

This script consumes the already generated DOTAV2 wrong-class artifacts. It
does not run model inference. The raw wrong-class summaries are kept separate
from the original-level deduplicated hard-case manifests because their
denominators are different.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RUN_SUMMARY = (
    'work_dirs/dotav2_current_wrongcls_gpu67_20260616_all_summary.csv')
DEFAULT_OUT_DIR = 'work_dirs/semantic_ambiguity_study_20260617'
DEFAULT_DEDUP_SUBDIR = 'vis_wrong_class_iou_gt0p7_merged_original_dedup'
DEFAULT_TARGET_PAIRS = (
    'small-vehicle->tennis-court,tennis-court->small-vehicle')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-summary-csv', default=DEFAULT_RUN_SUMMARY)
    parser.add_argument('--ann-dir', default='',
                        help='GT annfiles used to estimate class area priors.')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--dedup-subdir', default=DEFAULT_DEDUP_SUBDIR)
    parser.add_argument('--scale-ratio-thr', type=float, default=4.0)
    parser.add_argument('--score-thrs', default='0.3,0.5,0.7')
    parser.add_argument('--target-pairs', default=DEFAULT_TARGET_PAIRS)
    parser.add_argument('--top-k-hard-cases', type=int, default=200)
    return parser.parse_args()


def qbox_area(values):
    pts = [float(v) for v in values]
    xs = pts[0::2]
    ys = pts[1::2]
    total = 0.0
    for idx in range(4):
        j = (idx + 1) % 4
        total += xs[idx] * ys[j] - xs[j] * ys[idx]
    return abs(total) / 2.0


def parse_qbox(raw):
    return [float(item) for item in raw.replace(',', ' ').split()]


def percentile(values, pct):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def median(values):
    return percentile(values, 50)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def read_run_rows(path):
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f'empty run summary: {path}')
    return rows


def resolve_ann_dir(args, run_rows):
    if args.ann_dir:
        return Path(args.ann_dir)
    first_summary = Path(run_rows[0]['summary_json'])
    if not first_summary.exists():
        raise FileNotFoundError(
            f'cannot infer ann-dir; missing {first_summary}')
    data = json.loads(first_summary.read_text())
    ann_dir = data.get('ann_dir')
    if not ann_dir:
        raise ValueError(f'ann_dir missing in {first_summary}')
    return Path(ann_dir)


def compute_area_priors(ann_dir):
    by_class = defaultdict(list)
    for path in sorted(Path(ann_dir).glob('*.txt')):
        for raw in path.read_text().splitlines():
            parts = raw.split()
            if len(parts) < 9:
                continue
            diff = int(parts[9]) if len(parts) > 9 else 0
            if diff > 100:
                continue
            by_class[parts[8]].append(qbox_area(parts[:8]))
    priors = {}
    for class_name, areas in by_class.items():
        priors[class_name] = {
            'class': class_name,
            'count': len(areas),
            'min_area': min(areas),
            'p01_area': percentile(areas, 1),
            'p05_area': percentile(areas, 5),
            'p10_area': percentile(areas, 10),
            'median_area': median(areas),
            'p90_area': percentile(areas, 90),
            'p95_area': percentile(areas, 95),
            'p99_area': percentile(areas, 99),
            'max_area': max(areas),
        }
    return priors


def scale_ratio(area, class_name, priors):
    prior = priors.get(class_name)
    if not prior:
        return 0.0
    med = prior['median_area']
    if med <= 0 or area <= 0:
        return 0.0
    return max(area / med, med / area)


def scale_prior_violation(area, class_name, priors):
    prior = priors.get(class_name)
    if not prior:
        return 0.0
    med = prior['median_area']
    if med <= 0 or area <= 0:
        return 0.0
    return abs(math.log(area / med))


def class_prior_outlier(area, class_name, priors):
    prior = priors.get(class_name)
    if not prior:
        return False
    return area < prior['p01_area'] or area > prior['p99_area']


def load_dedup_summary(output_dir, dedup_subdir):
    path = Path(output_dir) / dedup_subdir / 'summary.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def box_manifest_path(output_dir, dedup_subdir):
    return Path(output_dir) / dedup_subdir / 'box_manifest.csv'


def pair_key(row):
    return f'{row["pred_class"]}->{row["gt_class"]}'


def top_pairs(counter, limit=8):
    return ';'.join(f'{pair}:{count}' for pair, count in counter.most_common(limit))


def summarize_model(row, priors, args, target_pairs, score_thrs):
    model = row['model']
    output_dir = row['output_dir']
    summary = load_dedup_summary(output_dir, args.dedup_subdir)
    stats = summary.get('stats', {})
    manifest = box_manifest_path(output_dir, args.dedup_subdir)
    if not manifest.exists():
        raise FileNotFoundError(f'missing box manifest: {manifest}')

    wrong_rows = []
    correct_rows = []
    pair_rows = defaultdict(list)
    hard_cases = []
    all_pair_counts = Counter()
    high_pair_counts = Counter()
    sise_pair_counts = Counter()

    for box in read_csv_rows(manifest):
        kind = box['kind']
        if kind == 'wrong':
            pred_area = qbox_area(parse_qbox(box['pred_qbox_original']))
            gt_area = qbox_area(parse_qbox(box['gt_qbox_original']))
            score = safe_float(box['score'])
            pred_ratio = scale_ratio(pred_area, box['pred_class'], priors)
            gt_ratio = scale_ratio(gt_area, box['gt_class'], priors)
            pred_spv = scale_prior_violation(
                pred_area, box['pred_class'], priors)
            box.update({
                'pred_area': pred_area,
                'gt_area': gt_area,
                'score_float': score,
                'pred_scale_ratio': pred_ratio,
                'gt_scale_ratio': gt_ratio,
                'scale_prior_violation': pred_spv,
                'is_sise': pred_ratio >= args.scale_ratio_thr,
                'is_prior_outlier_sise': class_prior_outlier(
                    pred_area, box['pred_class'], priors),
                'merged_vis_path': str(Path(output_dir) / args.dedup_subdir /
                                       f'{box["orig_id"]}.jpg'),
            })
            wrong_rows.append(box)
            pair_rows[pair_key(box)].append(box)
            all_pair_counts[pair_key(box)] += 1
            if score >= 0.5:
                high_pair_counts[pair_key(box)] += 1
            if box['is_sise']:
                sise_pair_counts[pair_key(box)] += 1
            if pair_key(box) in target_pairs:
                hard_cases.append(box)
        elif kind == 'correct':
            pred_area = qbox_area(parse_qbox(box['pred_qbox_original']))
            gt_area = qbox_area(parse_qbox(box['gt_qbox_original']))
            score = safe_float(box['score'])
            pred_ratio = scale_ratio(pred_area, box['pred_class'], priors)
            pred_spv = scale_prior_violation(
                pred_area, box['pred_class'], priors)
            box.update({
                'pred_area': pred_area,
                'gt_area': gt_area,
                'score_float': score,
                'pred_scale_ratio': pred_ratio,
                'scale_prior_violation': pred_spv,
                'is_scale_flagged': pred_ratio >= args.scale_ratio_thr,
                'is_prior_outlier_flagged': class_prior_outlier(
                    pred_area, box['pred_class'], priors),
            })
            correct_rows.append(box)

    pair_summary_rows = []
    for pair, rows in sorted(pair_rows.items()):
        pred_areas = [r['pred_area'] for r in rows]
        gt_areas = [r['gt_area'] for r in rows]
        scores = [r['score_float'] for r in rows]
        ratios = [r['pred_scale_ratio'] for r in rows]
        spv = [r['scale_prior_violation'] for r in rows]
        pair_summary_rows.append({
            'model': model,
            'pair': pair,
            'pred_class': pair.split('->', 1)[0],
            'gt_class': pair.split('->', 1)[1],
            'count': len(rows),
            'score_ge_0p3': sum(s >= 0.3 for s in scores),
            'score_ge_0p5': sum(s >= 0.5 for s in scores),
            'score_ge_0p7': sum(s >= 0.7 for s in scores),
            'sise_count': sum(r['is_sise'] for r in rows),
            'prior_outlier_sise_count': sum(
                r['is_prior_outlier_sise'] for r in rows),
            'median_score': median(scores),
            'median_pred_area': median(pred_areas),
            'median_gt_area': median(gt_areas),
            'median_pred_scale_ratio': median(ratios),
            'median_scale_prior_violation': median(spv),
        })

    hard_cases_sorted = sorted(
        hard_cases,
        key=lambda r: (r['score_float'], r['pred_scale_ratio'], r['iou']),
        reverse=True)[:args.top_k_hard_cases]

    high_conf = [r for r in wrong_rows if r['score_float'] >= 0.5]
    sise_rows = [r for r in wrong_rows if r['is_sise']]
    high_sise = [r for r in sise_rows if r['score_float'] >= 0.5]
    target = [r for r in wrong_rows if pair_key(r) in target_pairs]
    target_high = [r for r in target if r['score_float'] >= 0.5]
    target_sise = [r for r in target if r['is_sise']]
    prior_outlier_sise = [
        r for r in wrong_rows if r['is_prior_outlier_sise']]
    high_prior_outlier_sise = [
        r for r in prior_outlier_sise if r['score_float'] >= 0.5]
    target_prior_outlier = [
        r for r in target if r['is_prior_outlier_sise']]
    scale_flagged_correct = [
        r for r in correct_rows if r.get('is_scale_flagged')]
    high_scale_flagged_correct = [
        r for r in scale_flagged_correct if r['score_float'] >= 0.5]
    prior_outlier_correct = [
        r for r in correct_rows if r.get('is_prior_outlier_flagged')]
    high_prior_outlier_correct = [
        r for r in prior_outlier_correct if r['score_float'] >= 0.5]
    gate_flags = len(sise_rows) + len(scale_flagged_correct)
    gate_precision = len(sise_rows) / gate_flags if gate_flags else 0.0
    high_gate_flags = len(high_sise) + len(high_scale_flagged_correct)
    high_gate_precision = (
        len(high_sise) / high_gate_flags if high_gate_flags else 0.0)
    prior_gate_flags = len(prior_outlier_sise) + len(prior_outlier_correct)
    prior_gate_precision = (
        len(prior_outlier_sise) / prior_gate_flags
        if prior_gate_flags else 0.0)
    high_prior_gate_flags = (
        len(high_prior_outlier_sise) + len(high_prior_outlier_correct))
    high_prior_gate_precision = (
        len(high_prior_outlier_sise) / high_prior_gate_flags
        if high_prior_gate_flags else 0.0)

    raw_correct = int(row.get('localized_correct_class') or 0)
    raw_wrong = int(row.get('wrong_class_iou_gt0p7') or 0)
    raw_total = raw_correct + raw_wrong
    selected_wrong = int(stats.get('selected_wrong_boxes') or len(wrong_rows))
    selected_correct = int(stats.get('selected_correct_boxes') or len(correct_rows))
    selected_total = selected_wrong + selected_correct

    model_summary = {
        'model': model,
        'mAP': safe_float(row.get('mAP')),
        'AP50': safe_float(row.get('AP50')),
        'images_scanned': int(row.get('images_scanned') or 0),
        'gt_objects_checked': int(row.get('gt_objects_checked') or 0),
        'raw_localized_correct': raw_correct,
        'raw_localized_wrong_all_pairs': raw_wrong,
        'raw_wrong_share_all_pairs': raw_wrong / raw_total if raw_total else 0.0,
        'dedup_rendered_originals': int(stats.get('originals_rendered') or 0),
        'dedup_selected_wrong': selected_wrong,
        'dedup_selected_correct': selected_correct,
        'dedup_wrong_share_on_rendered_cases': (
            selected_wrong / selected_total if selected_total else 0.0),
        'high_conf_wrong_score_ge_0p5': len(high_conf),
        'sise_wrong_scale_ratio_ge_thr': len(sise_rows),
        'high_conf_sise_score_ge_0p5': len(high_sise),
        'correct_scale_ratio_ge_thr_on_rendered_cases': len(scale_flagged_correct),
        'high_conf_correct_scale_ratio_ge_thr_on_rendered_cases': (
            len(high_scale_flagged_correct)),
        'geometry_gate_flag_precision_on_rendered_cases': gate_precision,
        'geometry_gate_high_conf_flag_precision_on_rendered_cases': (
            high_gate_precision),
        'prior_outlier_sise_wrong': len(prior_outlier_sise),
        'high_conf_prior_outlier_sise_score_ge_0p5': (
            len(high_prior_outlier_sise)),
        'correct_prior_outlier_on_rendered_cases': len(prior_outlier_correct),
        'high_conf_correct_prior_outlier_on_rendered_cases': (
            len(high_prior_outlier_correct)),
        'prior_outlier_gate_precision_on_rendered_cases': prior_gate_precision,
        'prior_outlier_gate_high_conf_precision_on_rendered_cases': (
            high_prior_gate_precision),
        'target_pair_wrong': len(target),
        'target_pair_high_conf_score_ge_0p5': len(target_high),
        'target_pair_sise': len(target_sise),
        'target_pair_sise_recall': (
            len(target_sise) / len(target) if target else 0.0),
        'target_pair_prior_outlier_sise': len(target_prior_outlier),
        'target_pair_prior_outlier_recall': (
            len(target_prior_outlier) / len(target) if target else 0.0),
        'scale_ratio_thr': args.scale_ratio_thr,
        'top_wrong_pairs_dedup': top_pairs(all_pair_counts),
        'top_high_conf_pairs_dedup': top_pairs(high_pair_counts),
        'top_sise_pairs_dedup': top_pairs(sise_pair_counts),
        'box_manifest_csv': str(manifest),
    }

    score_counts = {}
    for thr in score_thrs:
        tag = str(thr).replace('.', 'p')
        score_counts[f'wrong_score_ge_{tag}'] = sum(
            r['score_float'] >= thr for r in wrong_rows)
        score_counts[f'sise_score_ge_{tag}'] = sum(
            r['is_sise'] and r['score_float'] >= thr for r in wrong_rows)
    model_summary.update(score_counts)

    return model_summary, pair_summary_rows, hard_cases_sorted


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def format_float(value, digits=4):
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return value


def markdown_table(rows, columns):
    lines = []
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, '')
            if isinstance(value, float):
                value = format_float(value)
            cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_markdown(out_dir, priors, model_rows, pair_rows, hard_rows, args):
    md = []
    md.append('# OpenRSD LSE/SISE Metrics')
    md.append('')
    md.append('生成时间：2026-06-17')
    md.append('')
    md.append('说明：`raw_localized_wrong_all_pairs` 来自原始 tile 级匹配统计，可能包含重复匹配和 small/large vehicle 互混；`dedup_*` 和 `SISE` 来自原图级去重后的 `box_manifest.csv`，更适合做 hard-case 证据。')
    md.append('')
    md.append(f'`scale_ratio_thr={args.scale_ratio_thr}`。若预测框面积相对预测类别 GT 中位面积的倍率超过该阈值，则记为 `SISE`。')
    md.append('')
    summary_cols = [
        'model', 'AP50', 'raw_localized_wrong_all_pairs',
        'dedup_selected_wrong', 'high_conf_wrong_score_ge_0p5',
        'sise_wrong_scale_ratio_ge_thr', 'high_conf_sise_score_ge_0p5',
        'prior_outlier_sise_wrong',
        'high_conf_prior_outlier_sise_score_ge_0p5',
        'correct_scale_ratio_ge_thr_on_rendered_cases',
        'geometry_gate_flag_precision_on_rendered_cases',
        'target_pair_wrong', 'target_pair_high_conf_score_ge_0p5',
        'target_pair_sise_recall', 'target_pair_prior_outlier_recall',
    ]
    md.append('## Model Summary')
    md.append('')
    md.append(markdown_table(model_rows, summary_cols))
    md.append('')
    md.append('## Geometry Gate Diagnostic')
    md.append('')
    md.append('该表只在原图级去重 hard-case 集上评估一个诊断性几何门控：若预测框面积相对预测类别 GT 中位面积倍率超过阈值，则 flag。它不是最终方法结果，但可以估计尺度先验是否有修复价值。')
    md.append('')
    gate_cols = [
        'model', 'sise_wrong_scale_ratio_ge_thr',
        'correct_scale_ratio_ge_thr_on_rendered_cases',
        'geometry_gate_flag_precision_on_rendered_cases',
        'high_conf_sise_score_ge_0p5',
        'high_conf_correct_scale_ratio_ge_thr_on_rendered_cases',
        'geometry_gate_high_conf_flag_precision_on_rendered_cases',
        'prior_outlier_sise_wrong',
        'correct_prior_outlier_on_rendered_cases',
        'prior_outlier_gate_precision_on_rendered_cases',
        'high_conf_prior_outlier_sise_score_ge_0p5',
        'high_conf_correct_prior_outlier_on_rendered_cases',
        'prior_outlier_gate_high_conf_precision_on_rendered_cases',
    ]
    md.append(markdown_table(model_rows, gate_cols))
    md.append('')
    md.append('## Class Area Priors')
    md.append('')
    prior_rows = sorted(priors.values(), key=lambda r: r['class'])
    prior_cols = ['class', 'count', 'median_area', 'p10_area', 'p90_area', 'p99_area']
    md.append(markdown_table(prior_rows, prior_cols))
    md.append('')
    md.append('## Top SISE Pairs')
    md.append('')
    top_sise = sorted(pair_rows, key=lambda r: r['sise_count'], reverse=True)[:30]
    pair_cols = [
        'model', 'pair', 'count', 'score_ge_0p5', 'sise_count',
        'prior_outlier_sise_count',
        'median_pred_area', 'median_gt_area', 'median_pred_scale_ratio',
    ]
    md.append(markdown_table(top_sise, pair_cols))
    md.append('')
    md.append('## Target Hard Cases')
    md.append('')
    hard_cols = [
        'model', 'orig_id', 'pred_class', 'gt_class', 'score', 'iou',
        'pred_area', 'gt_area', 'pred_scale_ratio', 'merged_vis_path'
    ]
    md.append(markdown_table(hard_rows[:30], hard_cols))
    md.append('')
    (out_dir / 'lse_sise_metrics.md').write_text(
        '\n'.join(md) + os.linesep)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = read_run_rows(args.run_summary_csv)
    ann_dir = resolve_ann_dir(args, run_rows)
    if not ann_dir.exists():
        raise FileNotFoundError(f'ann-dir does not exist: {ann_dir}')

    score_thrs = [float(item) for item in args.score_thrs.split(',') if item]
    target_pairs = {item for item in args.target_pairs.split(',') if item}
    priors = compute_area_priors(ann_dir)
    if not priors:
        raise ValueError(f'no GT priors loaded from {ann_dir}')

    model_summaries = []
    all_pair_rows = []
    all_hard_cases = []
    for row in run_rows:
        model_summary, pair_rows, hard_cases = summarize_model(
            row, priors, args, target_pairs, score_thrs)
        model_summaries.append(model_summary)
        all_pair_rows.extend(pair_rows)
        all_hard_cases.extend(hard_cases)

    hard_cases = sorted(
        all_hard_cases,
        key=lambda r: (r['score_float'], r['pred_scale_ratio'], r['iou']),
        reverse=True)[:args.top_k_hard_cases]
    hard_rows = []
    for row in hard_cases:
        hard_rows.append({
            'model': row['model'],
            'orig_id': row['orig_id'],
            'tile_img_id': row['tile_img_id'],
            'pred_class': row['pred_class'],
            'gt_class': row['gt_class'],
            'score': f'{row["score_float"]:.6f}',
            'iou': row['iou'],
            'pred_area': f'{row["pred_area"]:.2f}',
            'gt_area': f'{row["gt_area"]:.2f}',
            'pred_scale_ratio': f'{row["pred_scale_ratio"]:.4f}',
            'scale_prior_violation': f'{row["scale_prior_violation"]:.4f}',
            'merged_vis_path': row['merged_vis_path'],
            'pred_qbox_original': row['pred_qbox_original'],
            'gt_qbox_original': row['gt_qbox_original'],
        })

    prior_rows = sorted(priors.values(), key=lambda r: r['class'])
    write_csv(
        out_dir / 'class_area_priors.csv',
        prior_rows,
        ['class', 'count', 'min_area', 'p01_area', 'p05_area', 'p10_area',
         'median_area', 'p90_area', 'p95_area', 'p99_area', 'max_area'])
    write_csv(
        out_dir / 'model_lse_sise_summary.csv',
        model_summaries,
        list(model_summaries[0].keys()))
    gate_fields = [
        'model', 'scale_ratio_thr', 'sise_wrong_scale_ratio_ge_thr',
        'correct_scale_ratio_ge_thr_on_rendered_cases',
        'geometry_gate_flag_precision_on_rendered_cases',
        'high_conf_sise_score_ge_0p5',
        'high_conf_correct_scale_ratio_ge_thr_on_rendered_cases',
        'geometry_gate_high_conf_flag_precision_on_rendered_cases',
        'prior_outlier_sise_wrong',
        'correct_prior_outlier_on_rendered_cases',
        'prior_outlier_gate_precision_on_rendered_cases',
        'high_conf_prior_outlier_sise_score_ge_0p5',
        'high_conf_correct_prior_outlier_on_rendered_cases',
        'prior_outlier_gate_high_conf_precision_on_rendered_cases',
        'target_pair_wrong', 'target_pair_sise', 'target_pair_sise_recall',
        'target_pair_prior_outlier_sise',
        'target_pair_prior_outlier_recall',
        'target_pair_high_conf_score_ge_0p5',
    ]
    write_csv(
        out_dir / 'geometry_gate_diagnostic.csv',
        model_summaries,
        gate_fields)
    write_csv(
        out_dir / 'pair_lse_sise_summary.csv',
        all_pair_rows,
        ['model', 'pair', 'pred_class', 'gt_class', 'count', 'score_ge_0p3',
         'score_ge_0p5', 'score_ge_0p7', 'sise_count',
         'prior_outlier_sise_count', 'median_score', 'median_pred_area',
         'median_gt_area', 'median_pred_scale_ratio',
         'median_scale_prior_violation'])
    write_csv(
        out_dir / 'hard_cases_target_pairs.csv',
        hard_rows,
        ['model', 'orig_id', 'tile_img_id', 'pred_class', 'gt_class',
         'score', 'iou', 'pred_area', 'gt_area', 'pred_scale_ratio',
         'scale_prior_violation', 'merged_vis_path', 'pred_qbox_original',
         'gt_qbox_original'])

    summary = {
        'run_summary_csv': args.run_summary_csv,
        'ann_dir': str(ann_dir),
        'out_dir': str(out_dir),
        'dedup_subdir': args.dedup_subdir,
        'scale_ratio_thr': args.scale_ratio_thr,
        'target_pairs': sorted(target_pairs),
        'models': [row['model'] for row in run_rows],
        'outputs': {
            'class_area_priors_csv': str(out_dir / 'class_area_priors.csv'),
            'model_lse_sise_summary_csv': str(out_dir / 'model_lse_sise_summary.csv'),
            'pair_lse_sise_summary_csv': str(out_dir / 'pair_lse_sise_summary.csv'),
            'geometry_gate_diagnostic_csv': str(out_dir / 'geometry_gate_diagnostic.csv'),
            'hard_cases_target_pairs_csv': str(out_dir / 'hard_cases_target_pairs.csv'),
            'markdown': str(out_dir / 'lse_sise_metrics.md'),
        },
    }
    (out_dir / 'run_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
    write_markdown(out_dir, priors, model_summaries, all_pair_rows, hard_rows, args)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
