#!/usr/bin/env python
"""Analyze score saturation and scale-prior gate counterfactuals."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_MANIFEST = (
    'work_dirs/dotav2_p4_lowtext_lser_sise_full9772_gpu67_20260617_1600/'
    'vis_wrong_class_iou_gt0p7_merged_original_dedup/box_manifest.csv')
DEFAULT_PRIORS = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'p4_lowtext_lser_sise_eval/class_area_priors.csv')
DEFAULT_OUT_DIR = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'p4_score_calibration_scale_gate')
DEFAULT_SCALE_RATIO_THRS = '2,3,4,5,8,10,15,20'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-csv', default=DEFAULT_MANIFEST)
    parser.add_argument('--class-area-priors-csv', default=DEFAULT_PRIORS)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--score-thr', type=float, default=0.5)
    parser.add_argument('--near-one-thr', type=float, default=0.999)
    parser.add_argument('--scale-ratio-thrs', default=DEFAULT_SCALE_RATIO_THRS)
    return parser.parse_args()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def parse_qbox(raw):
    return [float(item) for item in str(raw).replace(',', ' ').split()]


def qbox_area(values):
    pts = parse_qbox(values) if isinstance(values, str) else list(values)
    xs = pts[0::2]
    ys = pts[1::2]
    total = 0.0
    for idx in range(4):
        j = (idx + 1) % 4
        total += xs[idx] * ys[j] - xs[j] * ys[idx]
    return abs(total) / 2.0


def percentile(values, pct):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def median(values):
    return percentile(values, 50)


def load_priors(path):
    priors = {}
    for row in read_csv_rows(path):
        cls = row.get('class', '')
        if not cls:
            continue
        priors[cls] = {
            'median_area': safe_float(row.get('median_area')),
            'p01_area': safe_float(row.get('p01_area')),
            'p99_area': safe_float(row.get('p99_area')),
        }
    return priors


def scale_ratio(area, cls, priors):
    prior = priors.get(cls)
    if not prior:
        return 0.0
    med = prior.get('median_area', 0.0)
    if area <= 0 or med <= 0:
        return 0.0
    return max(area / med, med / area)


def scale_prior_violation(area, cls, priors):
    prior = priors.get(cls)
    if not prior:
        return 0.0
    med = prior.get('median_area', 0.0)
    if area <= 0 or med <= 0:
        return 0.0
    return abs(math.log(area / med))


def prior_outlier(area, cls, priors):
    prior = priors.get(cls)
    if not prior:
        return False
    return area < prior.get('p01_area', 0.0) or area > prior.get('p99_area', 0.0)


def enrich_manifest_rows(rows, priors, scale_ratio_thr):
    enriched_rows = []
    for row in rows:
        if row.get('kind') not in {'wrong', 'correct'}:
            continue
        pred_area = qbox_area(row['pred_qbox_original'])
        ratio = scale_ratio(pred_area, row.get('pred_class', ''), priors)
        enriched = dict(row)
        enriched.update({
            'score_float': safe_float(row.get('score')),
            'iou_float': safe_float(row.get('iou')),
            'pred_area': pred_area,
            'scale_ratio': ratio,
            'scale_prior_violation': scale_prior_violation(
                pred_area, row.get('pred_class', ''), priors),
            'is_sise': ratio >= scale_ratio_thr,
            'is_prior_outlier': prior_outlier(
                pred_area, row.get('pred_class', ''), priors),
        })
        enriched_rows.append(enriched)
    return enriched_rows


def pair_key(row):
    return f'{row["pred_class"]}->{row["gt_class"]}'


def summarize_wrong_pairs(rows, near_one_thr=0.999):
    by_pair = defaultdict(list)
    for row in rows:
        if row.get('kind') == 'wrong':
            by_pair[pair_key(row)].append(row)

    summaries = []
    for pair, pair_rows in by_pair.items():
        scores = [row['score_float'] for row in pair_rows]
        ious = [row['iou_float'] for row in pair_rows]
        ratios = [row['scale_ratio'] for row in pair_rows]
        summaries.append({
            'pair': pair,
            'pred_class': pair.split('->', 1)[0],
            'gt_class': pair.split('->', 1)[1],
            'count': len(pair_rows),
            'score_ge_0p5': sum(score >= 0.5 for score in scores),
            'score_ge_0p9': sum(score >= 0.9 for score in scores),
            'score_ge_0p99': sum(score >= 0.99 for score in scores),
            'score_ge_near_one': sum(score >= near_one_thr for score in scores),
            'near_one_rate': (
                sum(score >= near_one_thr for score in scores) / len(scores)
                if scores else 0.0),
            'median_score': median(scores),
            'mean_score': sum(scores) / len(scores) if scores else 0.0,
            'median_iou': median(ious),
            'median_scale_ratio': median(ratios),
            'sise_count': sum(parse_bool(row['is_sise']) for row in pair_rows),
            'prior_outlier_count': sum(
                parse_bool(row['is_prior_outlier']) for row in pair_rows),
        })
    summaries.sort(key=lambda row: (-row['count'], row['pair']))
    return summaries


def evaluate_gate(rows, gate, score_thr=0.5):
    if gate not in {'scale_ratio', 'prior_outlier'}:
        raise ValueError(f'unsupported gate: {gate}')
    flag_key = 'is_sise' if gate == 'scale_ratio' else 'is_prior_outlier'
    wrong_rows = [
        row for row in rows
        if row.get('kind') == 'wrong' and row['score_float'] >= score_thr
    ]
    correct_rows = [
        row for row in rows
        if row.get('kind') == 'correct' and row['score_float'] >= score_thr
    ]
    wrong_rejected = sum(parse_bool(row[flag_key]) for row in wrong_rows)
    correct_rejected = sum(parse_bool(row[flag_key]) for row in correct_rows)
    gate_flags = wrong_rejected + correct_rejected
    return {
        'gate': gate,
        'score_thr': score_thr,
        'high_conf_wrong_before': len(wrong_rows),
        'wrong_rejected': wrong_rejected,
        'wrong_after': len(wrong_rows) - wrong_rejected,
        'wrong_reduction_rate': (
            wrong_rejected / len(wrong_rows) if wrong_rows else 0.0),
        'high_conf_correct_before': len(correct_rows),
        'correct_rejected': correct_rejected,
        'correct_after': len(correct_rows) - correct_rejected,
        'correct_false_reject_rate': (
            correct_rejected / len(correct_rows) if correct_rows else 0.0),
        'correct_retention_rate': (
            (len(correct_rows) - correct_rejected) / len(correct_rows)
            if correct_rows else 0.0),
        'gate_precision': wrong_rejected / gate_flags if gate_flags else 0.0,
    }


def evaluate_pair_gate(rows, gate, score_thr=0.5):
    flag_key = 'is_sise' if gate == 'scale_ratio' else 'is_prior_outlier'
    by_pair = defaultdict(list)
    for row in rows:
        if row.get('kind') == 'wrong' and row['score_float'] >= score_thr:
            by_pair[pair_key(row)].append(row)
    out = []
    for pair, pair_rows in by_pair.items():
        rejected = sum(parse_bool(row[flag_key]) for row in pair_rows)
        out.append({
            'gate': gate,
            'pair': pair,
            'high_conf_wrong_before': len(pair_rows),
            'wrong_rejected': rejected,
            'wrong_reduction_rate': (
                rejected / len(pair_rows) if pair_rows else 0.0),
        })
    out.sort(key=lambda row: (-row['wrong_rejected'], row['pair']))
    return out


def score_histogram(rows, bins):
    counts = defaultdict(int)
    for row in rows:
        score = row['score_float']
        for lo, hi in zip(bins[:-1], bins[1:]):
            if lo <= score < hi:
                counts[(row['kind'], lo, hi)] += 1
                break
    out = []
    for (kind, lo, hi), count in sorted(counts.items()):
        out.append({
            'kind': kind,
            'score_bin': f'[{lo},{hi})',
            'count': count,
        })
    return out


def top_rows(rows, key, limit=10):
    return sorted(rows, key=lambda row: row[key], reverse=True)[:limit]


def markdown_table(rows, columns):
    lines = []
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, '')
            if isinstance(value, float):
                value = f'{value:.4f}'
            cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_report(path, args, pair_rows, gate_rows, pair_gate_rows, hist_rows):
    near_one_wrong = sum(
        row['score_ge_near_one'] for row in pair_rows)
    wrong_total = sum(row['count'] for row in pair_rows)
    best_scale = next(
        row for row in gate_rows
        if row['gate'] == 'scale_ratio' and row.get('scale_ratio_thr') == 4.0)
    prior_gate = next(row for row in gate_rows if row['gate'] == 'prior_outlier')
    lines = [
        '# P4 Score Calibration and Scale-Prior Counterfactual',
        '',
        '## 结论',
        '',
        (
            f'在去重 hard-case manifest 上，P4 错框 score 明显饱和：'
            f'`score>={args.near_one_thr}` 的 wrong boxes 为 '
            f'`{near_one_wrong}/{wrong_total}`。这说明当前 post-NMS score '
            '基本不能区分“定位正确但类别语义不合理”的错误。'
        ),
        '',
        (
            '尺度先验门控有明显诊断价值，但它是后处理 counterfactual，'
            '不是完整 AP 评估。`scale_ratio>=4` 可拒绝 '
            f'`{best_scale["wrong_rejected"]}` 个高置信 wrong boxes，同时误杀 '
            f'`{best_scale["correct_rejected"]}` 个高置信 correct boxes；'
            f'gate precision 为 `{best_scale["gate_precision"]:.4f}`。'
        ),
        '',
        (
            '更严格的 p01/p99 prior-outlier gate 可拒绝 '
            f'`{prior_gate["wrong_rejected"]}` 个高置信 wrong boxes，同时误杀 '
            f'`{prior_gate["correct_rejected"]}` 个高置信 correct boxes；'
            f'gate precision 为 `{prior_gate["gate_precision"]:.4f}`。'
        ),
        '',
        '## 为什么会出现大量 score=1.0',
        '',
        '这里的 score 是已保存检测结果的后处理 confidence，不是原始 logit。'
        '从分布看，错误不是低置信尾部噪声，而是高置信饱和：'
        '`small-vehicle->plane`、`small-vehicle->ship` 等 pair 的 score '
        '接近 1，同时预测框面积远大于 small-vehicle 的 GT 面积分布。'
        '这意味着分类分数没有编码类别尺度先验；NMS 只处理框间重复，'
        '不会惩罚“一个飞机大小的框被判为 small-vehicle”这种语义-尺度矛盾。',
        '',
        '## Top Wrong Pairs by Count',
        '',
        markdown_table(top_rows(pair_rows, 'count', 12), [
            'pair', 'count', 'score_ge_0p9', 'score_ge_0p99',
            'score_ge_near_one', 'near_one_rate', 'median_score',
            'median_scale_ratio', 'sise_count', 'prior_outlier_count']),
        '',
        '## Gate Counterfactual',
        '',
        markdown_table(gate_rows, [
            'gate', 'scale_ratio_thr', 'high_conf_wrong_before',
            'wrong_rejected', 'wrong_reduction_rate',
            'high_conf_correct_before', 'correct_rejected',
            'correct_false_reject_rate', 'gate_precision']),
        '',
        '## Top Pair Reductions',
        '',
        markdown_table(top_rows(pair_gate_rows, 'wrong_rejected', 20), [
            'gate', 'scale_ratio_thr', 'pair', 'high_conf_wrong_before',
            'wrong_rejected', 'wrong_reduction_rate']),
        '',
        '## Caveat',
        '',
        '该 counterfactual 基于原图级去重 manifest，分母是 hard-case 可视化集合，'
        '不是完整检测结果重跑后的 AP/mAP。因此它回答的是“能拦截多少已定位的错类框、'
        '误杀多少已定位正确框”，不能直接等价为最终 detector AP 变化。',
    ]
    Path(path).write_text('\n'.join(lines) + os.linesep, encoding='utf-8')


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    priors = load_priors(args.class_area_priors_csv)
    raw_rows = read_csv_rows(args.manifest_csv)
    scale_ratio_thrs = [
        float(item) for item in args.scale_ratio_thrs.split(',') if item.strip()
    ]
    base_thr = 4.0 if 4.0 in scale_ratio_thrs else scale_ratio_thrs[0]
    rows = enrich_manifest_rows(raw_rows, priors, base_thr)

    pair_rows = summarize_wrong_pairs(rows, args.near_one_thr)
    hist_rows = score_histogram(rows, [0.0, 0.3, 0.5, 0.7, 0.9, 0.99,
                                       args.near_one_thr, 1.000001])

    gate_rows = []
    pair_gate_rows = []
    for thr in scale_ratio_thrs:
        thr_rows = enrich_manifest_rows(raw_rows, priors, thr)
        result = evaluate_gate(thr_rows, 'scale_ratio', args.score_thr)
        result['scale_ratio_thr'] = thr
        gate_rows.append(result)
        pair_gate_rows.extend({
            **row,
            'scale_ratio_thr': thr,
        } for row in evaluate_pair_gate(thr_rows, 'scale_ratio', args.score_thr))
    prior_result = evaluate_gate(rows, 'prior_outlier', args.score_thr)
    prior_result['scale_ratio_thr'] = ''
    gate_rows.append(prior_result)
    pair_gate_rows.extend({
        **row,
        'scale_ratio_thr': '',
    } for row in evaluate_pair_gate(rows, 'prior_outlier', args.score_thr))

    enriched_csv = out_dir / 'enriched_manifest.csv'
    pair_csv = out_dir / 'score_pair_summary.csv'
    hist_csv = out_dir / 'score_histogram.csv'
    gate_csv = out_dir / 'scale_gate_counterfactual_summary.csv'
    pair_gate_csv = out_dir / 'scale_gate_pair_reduction.csv'
    report_md = out_dir / 'p4_score_calibration_scale_gate_report.md'
    summary_json = out_dir / 'summary.json'

    write_csv(enriched_csv, rows)
    write_csv(pair_csv, pair_rows)
    write_csv(hist_csv, hist_rows)
    write_csv(gate_csv, gate_rows)
    write_csv(pair_gate_csv, pair_gate_rows)
    write_report(report_md, args, pair_rows, gate_rows, pair_gate_rows, hist_rows)

    summary = {
        'manifest_csv': args.manifest_csv,
        'class_area_priors_csv': args.class_area_priors_csv,
        'out_dir': str(out_dir),
        'wrong_count': sum(row['count'] for row in pair_rows),
        'near_one_wrong_count': sum(row['score_ge_near_one']
                                    for row in pair_rows),
        'outputs': {
            'enriched_manifest_csv': str(enriched_csv),
            'score_pair_summary_csv': str(pair_csv),
            'score_histogram_csv': str(hist_csv),
            'scale_gate_counterfactual_summary_csv': str(gate_csv),
            'scale_gate_pair_reduction_csv': str(pair_gate_csv),
            'report_md': str(report_md),
        },
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
