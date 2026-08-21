#!/usr/bin/env python
"""P2 geometry-aware score calibration baseline.

This is a post-processing proxy baseline.  It does not rerun detector NMS and
does not claim AP gains.  It asks whether a simple geometry prior can lower
high-confidence localized wrong classifications while retaining localized
correct detections on the same rendered-case manifests.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_RUN_SUMMARY = (
    'work_dirs/dotav2_current_wrongcls_gpu67_20260616_all_summary.csv')
DEFAULT_OUT_DIR = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'p2_geometry_calibration_baseline')
DEFAULT_DEDUP_SUBDIR = 'vis_wrong_class_iou_gt0p7_merged_original_dedup'
DEFAULT_LAMBDAS = '0,0.25,0.5,0.75,1.0,1.5,2.0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-summary-csv', default=DEFAULT_RUN_SUMMARY)
    parser.add_argument('--ann-dir', default='')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--dedup-subdir', default=DEFAULT_DEDUP_SUBDIR)
    parser.add_argument('--lambdas', default=DEFAULT_LAMBDAS)
    parser.add_argument('--score-thr', type=float, default=0.5)
    parser.add_argument('--scale-ratio-thr', type=float, default=4.0)
    return parser.parse_args()


def read_csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def parse_qbox(raw):
    return [float(item) for item in raw.replace(',', ' ').split()]


def qbox_area(values):
    xs = values[0::2]
    ys = values[1::2]
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
            by_class[parts[8]].append(qbox_area([float(v) for v in parts[:8]]))
    priors = {}
    for cls, areas in by_class.items():
        priors[cls] = {
            'class': cls,
            'count': len(areas),
            'p01_area': percentile(areas, 1),
            'median_area': percentile(areas, 50),
            'p99_area': percentile(areas, 99),
        }
    return priors


def resolve_ann_dir(args, run_rows):
    if args.ann_dir:
        return Path(args.ann_dir)
    first = Path(run_rows[0]['summary_json'])
    data = json.loads(first.read_text())
    return Path(data['ann_dir'])


def scale_prior_violation(area, cls, priors):
    prior = priors.get(cls)
    if not prior:
        return 0.0
    med = prior['median_area']
    if area <= 0 or med <= 0:
        return 0.0
    return abs(math.log(area / med))


def scale_ratio(area, cls, priors):
    prior = priors.get(cls)
    if not prior:
        return 0.0
    med = prior['median_area']
    if area <= 0 or med <= 0:
        return 0.0
    return max(area / med, med / area)


def prior_outlier(area, cls, priors):
    prior = priors.get(cls)
    if not prior:
        return False
    return area < prior['p01_area'] or area > prior['p99_area']


def adjusted_score(score, violation, flagged, lam):
    score = safe_float(score)
    violation = safe_float(violation)
    if not flagged:
        return score
    return score * math.exp(-float(lam) * violation)


def evaluate_calibration(wrong_rows, correct_rows, lam, score_thr,
                         wrong_flag_key, correct_flag_key):
    wrong_before = [
        row for row in wrong_rows if safe_float(row.get('score')) >= score_thr
    ]
    correct_before = [
        row for row in correct_rows if safe_float(row.get('score')) >= score_thr
    ]
    wrong_after = [
        row for row in wrong_rows
        if adjusted_score(
            row.get('score'), row.get('scale_prior_violation'),
            parse_bool(row.get(wrong_flag_key)), lam) >= score_thr
    ]
    correct_after = [
        row for row in correct_rows
        if adjusted_score(
            row.get('score'), row.get('scale_prior_violation'),
            parse_bool(row.get(correct_flag_key)), lam) >= score_thr
    ]
    wrong_reduction = len(wrong_before) - len(wrong_after)
    correct_drop = len(correct_before) - len(correct_after)
    return {
        'lambda': lam,
        'score_thr': score_thr,
        'wrong_total': len(wrong_rows),
        'correct_total': len(correct_rows),
        'high_conf_wrong_before': len(wrong_before),
        'high_conf_wrong_after': len(wrong_after),
        'high_conf_wrong_reduction': wrong_reduction,
        'high_conf_wrong_reduction_rate': (
            wrong_reduction / len(wrong_before) if wrong_before else 0.0),
        'high_conf_correct_before': len(correct_before),
        'high_conf_correct_after': len(correct_after),
        'high_conf_correct_drop': correct_drop,
        'high_conf_correct_retention': (
            len(correct_after) / len(correct_before) if correct_before else 0.0),
    }


def load_manifest_rows(output_dir, dedup_subdir, priors, scale_ratio_thr):
    manifest = Path(output_dir) / dedup_subdir / 'box_manifest.csv'
    wrong_rows = []
    correct_rows = []
    for row in read_csv_rows(manifest):
        if row.get('kind') not in {'wrong', 'correct'}:
            continue
        area = qbox_area(parse_qbox(row['pred_qbox_original']))
        cls = row['pred_class']
        enriched = dict(row)
        enriched['pred_area'] = area
        enriched['score'] = row.get('score', '')
        enriched['scale_ratio'] = scale_ratio(area, cls, priors)
        enriched['scale_prior_violation'] = scale_prior_violation(
            area, cls, priors)
        enriched['is_scale_ratio_flagged'] = (
            enriched['scale_ratio'] >= scale_ratio_thr)
        enriched['is_prior_outlier_flagged'] = prior_outlier(area, cls, priors)
        if row['kind'] == 'wrong':
            enriched['is_scale_ratio_sise'] = enriched['is_scale_ratio_flagged']
            enriched['is_prior_outlier_sise'] = enriched[
                'is_prior_outlier_flagged']
            wrong_rows.append(enriched)
        else:
            correct_rows.append(enriched)
    return wrong_rows, correct_rows


def write_markdown(path, rows, best_rows, summary_csv):
    lines = [
        '# P2 Geometry-Aware Score Calibration Baseline',
        '',
        '这是后处理 proxy baseline：只调整已保存检测框的 score，'
        '不重新运行 NMS，也不声称 AP/mAP 提升。',
        '',
        f'- summary_csv: `{summary_csv}`',
        '',
        '## Best Prior-Outlier Lambda Per Model',
        '',
        '| model | lambda | high_conf_wrong_before | high_conf_wrong_after | high_conf_wrong_reduction_rate | high_conf_correct_retention |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in best_rows:
        lines.append(
            f'| `{row["model"]}` | {row["lambda"]} | '
            f'{row["high_conf_wrong_before"]} | '
            f'{row["high_conf_wrong_after"]} | '
            f'{float(row["high_conf_wrong_reduction_rate"]):.4f} | '
            f'{float(row["high_conf_correct_retention"]):.4f} |')
    lines.extend([
        '',
        '## Caveat',
        '',
        '`correct_total` 只来自当前可视化/原图级 manifest 中的 localized correct boxes，'
        '不是完整 DOTAV2 validation AP denominator。因此本表只能证明几何校准'
        '是否有潜力降低 HighConf-LSE，不能替代完整 detector eval。',
    ])
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = read_csv_rows(args.run_summary_csv)
    priors = compute_area_priors(resolve_ann_dir(args, run_rows))
    lambdas = [float(item) for item in args.lambdas.split(',') if item.strip()]
    summary_rows = []
    for run in run_rows:
        wrong_rows, correct_rows = load_manifest_rows(
            run['output_dir'], args.dedup_subdir, priors,
            args.scale_ratio_thr)
        for gate, wrong_key, correct_key in [
                ('prior_outlier', 'is_prior_outlier_sise',
                 'is_prior_outlier_flagged'),
                ('scale_ratio', 'is_scale_ratio_sise',
                 'is_scale_ratio_flagged')]:
            for lam in lambdas:
                result = evaluate_calibration(
                    wrong_rows, correct_rows, lam, args.score_thr,
                    wrong_key, correct_key)
                summary_rows.append({
                    'model': run['model'],
                    'gate': gate,
                    **result,
                })
    summary_csv = out_dir / 'p2_geometry_calibration_summary.csv'
    write_csv(summary_csv, summary_rows)
    best_rows = []
    for model in sorted(set(row['model'] for row in summary_rows)):
        candidates = [
            row for row in summary_rows
            if row['model'] == model and row['gate'] == 'prior_outlier'
        ]
        candidates.sort(key=lambda row: (
            -float(row['high_conf_wrong_reduction_rate']),
            -float(row['high_conf_correct_retention']),
            float(row['lambda']),
        ))
        if candidates:
            best_rows.append(candidates[0])
    best_csv = out_dir / 'p2_geometry_calibration_best_prior_outlier.csv'
    report_md = out_dir / 'p2_geometry_calibration_report.md'
    write_csv(best_csv, best_rows)
    write_markdown(report_md, summary_rows, best_rows, summary_csv)
    print(f'summary_csv={summary_csv}')
    print(f'best_csv={best_csv}')
    print(f'report_md={report_md}')
    print(f'rows={len(summary_rows)}')


if __name__ == '__main__':
    main()
