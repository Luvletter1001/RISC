#!/usr/bin/env python
"""Summarize E9 scale-counterfactual probe outputs.

The probe itself runs per GPU/output directory.  This script merges completed
probe CSV files and converts the three variants of each hard case into a
mechanism diagnosis:

* original wrong + neutral same scale correct: context shortcut.
* neutral same scale still predicts the impossible class: cls-reg scale
  decoupling.
* neutral same scale correct + configured scale impossible: scale-prior
  sensitive.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_DIRS = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'e9_scale_counterfactual_gpu6,'
    'work_dirs/semantic_ambiguity_study_20260617/'
    'e9_scale_counterfactual_gpu7'
)
DEFAULT_OUT_DIR = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'e9_scale_counterfactual_combined')
VARIANTS = (
    'original_tile',
    'neutral_same_scale',
    'neutral_small_vehicle_scale',
    'neutral_pred_class_scale',
)
BASE_VARIANTS = (
    'original_tile',
    'neutral_same_scale',
)
SCALE_VARIANTS = (
    'neutral_small_vehicle_scale',
    'neutral_pred_class_scale',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dirs', default=DEFAULT_INPUT_DIRS,
                        help='Comma-separated E9 output directories.')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--include-dry-run', action='store_true')
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


def load_inputs(input_dirs, include_dry_run):
    rows = []
    statuses = []
    for raw_dir in input_dirs:
        out_dir = Path(raw_dir)
        csv_path = out_dir / 'e9_counterfactual_probe.csv'
        if not out_dir.exists():
            statuses.append({
                'out_dir': str(out_dir),
                'csv_path': str(csv_path),
                'status': 'pending_missing_dir',
                'rows': 0,
            })
            continue
        if not csv_path.exists():
            statuses.append({
                'out_dir': str(out_dir),
                'csv_path': str(csv_path),
                'status': 'pending_missing_csv',
                'rows': 0,
            })
            continue
        loaded = read_csv_rows(csv_path)
        if not include_dry_run:
            loaded = [row for row in loaded if row.get('dry_run') != '1']
        for row in loaded:
            row['source_out_dir'] = str(out_dir)
        rows.extend(loaded)
        statuses.append({
            'out_dir': str(out_dir),
            'csv_path': str(csv_path),
            'status': 'loaded',
            'rows': len(loaded),
        })
    return rows, statuses


def as_float(raw, default=0.0):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def is_target_matched(row):
    return str(row.get('target_matched', '')).strip() in {'1', 'true', 'True'}


def variant_class(case_rows, variant):
    row = case_rows.get(variant)
    if not row or not is_target_matched(row):
        return 'no_match'
    return row.get('matched_class') or 'no_class'


def variant_score(case_rows, variant):
    row = case_rows.get(variant)
    if not row or not is_target_matched(row):
        return ''
    return row.get('matched_score', '')


def scale_variant_name(case_rows):
    for variant in SCALE_VARIANTS:
        if variant in case_rows:
            return variant
    return 'neutral_small_vehicle_scale'


def diagnose_case(case_rows):
    sample = next(iter(case_rows.values()))
    target_class = sample.get('target_gt_class', '')
    impossible_class = sample.get('impossible_pred_class', '')
    original_cls = variant_class(case_rows, 'original_tile')
    same_cls = variant_class(case_rows, 'neutral_same_scale')
    scale_variant = scale_variant_name(case_rows)
    scale_cls = variant_class(case_rows, scale_variant)

    if same_cls == impossible_class:
        diagnosis = 'cls_reg_scale_decoupling'
        reason = 'neutral_same_scale still predicts the impossible class'
    elif same_cls == target_class and scale_cls == impossible_class:
        diagnosis = 'scale_prior_sensitive'
        reason = 'configured scale counterfactual flips toward the wrong class'
    elif original_cls == impossible_class and same_cls == target_class:
        diagnosis = 'context_shortcut'
        reason = 'removing context restores the target class at the same scale'
    elif original_cls == target_class and same_cls == target_class:
        diagnosis = 'not_reproduced_after_rerun'
        reason = 'the rerun does not reproduce the original impossible class'
    elif 'no_match' in {original_cls, same_cls, scale_cls}:
        diagnosis = 'insufficient_match'
        reason = 'at least one variant has no matched detection'
    else:
        diagnosis = 'other_or_mixed'
        reason = 'variant classes do not match a primary mechanism template'

    return {
        'target_class': target_class,
        'impossible_class': impossible_class,
        'original_cls': original_cls,
        'neutral_same_cls': same_cls,
        'scale_variant': scale_variant,
        'neutral_scale_cls': scale_cls,
        'neutral_small_cls': scale_cls,
        'original_score': variant_score(case_rows, 'original_tile'),
        'neutral_same_score': variant_score(case_rows, 'neutral_same_scale'),
        'neutral_scale_score': variant_score(case_rows, scale_variant),
        'neutral_small_score': variant_score(case_rows, scale_variant),
        'diagnosis': diagnosis,
        'diagnosis_reason': reason,
    }


def summarize_cases(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row.get('error'):
            continue
        key = (
            row.get('model', ''),
            row.get('case_index', ''),
            row.get('orig_id', ''),
            row.get('tile_img_id', ''),
        )
        variant = row.get('variant', '')
        if variant in VARIANTS:
            grouped[key][variant] = row

    case_rows = []
    for key, variants in sorted(grouped.items()):
        model, case_index, orig_id, tile_img_id = key
        diag = diagnose_case(variants)
        complete = int(
            all(variant in variants for variant in BASE_VARIANTS) and
            any(variant in variants for variant in SCALE_VARIANTS))
        case_rows.append({
            'model': model,
            'case_index': case_index,
            'orig_id': orig_id,
            'tile_img_id': tile_img_id,
            'complete_variants': complete,
            **diag,
        })
    return case_rows


def summarize_models(case_rows):
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[row['model']].append(row)

    summary_rows = []
    for model, items in sorted(grouped.items()):
        counts = Counter(row['diagnosis'] for row in items)
        summary_rows.append({
            'model': model,
            'cases': len(items),
            'complete_cases': sum(int(row['complete_variants'])
                                  for row in items),
            'cls_reg_scale_decoupling': counts.get(
                'cls_reg_scale_decoupling', 0),
            'context_shortcut': counts.get('context_shortcut', 0),
            'scale_prior_sensitive': counts.get('scale_prior_sensitive', 0),
            'not_reproduced_after_rerun': counts.get(
                'not_reproduced_after_rerun', 0),
            'insufficient_match': counts.get('insufficient_match', 0),
            'other_or_mixed': counts.get('other_or_mixed', 0),
        })
    return summary_rows


def write_markdown(path, statuses, summary_rows, case_rows, combined_csv,
                   case_csv, summary_csv):
    lines = [
        '# E9 Scale Counterfactual Combined Summary',
        '',
        '本报告合并 GPU6/GPU7 的尺度反事实 probe 输出，用于判断 '
        'top confusion pair 是否来自上下文捷径、尺度先验敏感，或分类-回归'
        '尺度解耦。',
        '',
        '## Input Status',
        '',
        '| out_dir | status | rows | csv_path |',
        '|---|---|---:|---|',
    ]
    for row in statuses:
        lines.append(
            f'| `{row["out_dir"]}` | {row["status"]} | {row["rows"]} | '
            f'`{row["csv_path"]}` |')

    lines.extend([
        '',
        '## Model Summary',
        '',
        '| model | cases | complete_cases | cls_reg_scale_decoupling | context_shortcut | scale_prior_sensitive | not_reproduced_after_rerun | insufficient_match | other_or_mixed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in summary_rows:
        lines.append(
            f'| {row["model"]} | {row["cases"]} | '
            f'{row["complete_cases"]} | '
            f'{row["cls_reg_scale_decoupling"]} | '
            f'{row["context_shortcut"]} | '
            f'{row["scale_prior_sensitive"]} | '
            f'{row["not_reproduced_after_rerun"]} | '
            f'{row["insufficient_match"]} | '
            f'{row["other_or_mixed"]} |')

    lines.extend([
        '',
        '## Case Summary',
        '',
        '| model | case_index | orig_id | original_cls | neutral_same_cls | scale_variant | neutral_scale_cls | diagnosis |',
        '|---|---:|---|---|---|---|---|---|',
    ])
    for row in case_rows[:80]:
        lines.append(
            f'| {row["model"]} | {row["case_index"]} | '
            f'{row["orig_id"]} | {row["original_cls"]} | '
            f'{row["neutral_same_cls"]} | {row["scale_variant"]} | '
            f'{row["neutral_scale_cls"]} | '
            f'{row["diagnosis"]} |')

    lines.extend([
        '',
        '## Artifacts',
        '',
        f'- combined_csv: `{combined_csv}`',
        f'- case_csv: `{case_csv}`',
        f'- summary_csv: `{summary_csv}`',
        '',
        '## Interpretation',
        '',
        '- `cls_reg_scale_decoupling`: 去上下文且保持 GT 原尺度后仍输出错误类别，是最强机制证据。',
        '- `context_shortcut`: 原图错、去上下文同尺度变对，说明邻域/背景触发了语义捷径。',
        '- `scale_prior_sensitive`: 同尺度变对，但缩放到配置的类别先验尺度后变错，说明模型使用尺度线索，但没有把它作为原尺度分类约束。',
    ])
    if not case_rows:
        lines.extend([
            '',
            '当前还没有非 dry-run 的 E9 结果。若 GPU6/GPU7 仍忙，本报告会保持 pending 状态。',
        ])
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    input_dirs = [item.strip() for item in args.input_dirs.split(',')
                  if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, statuses = load_inputs(input_dirs, args.include_dry_run)
    combined_csv = out_dir / 'e9_counterfactual_probe_combined.csv'
    case_csv = out_dir / 'e9_case_diagnosis.csv'
    summary_csv = out_dir / 'e9_model_diagnosis_summary.csv'
    report_md = out_dir / 'e9_scale_counterfactual_combined_report.md'

    write_csv(combined_csv, rows)
    case_rows = summarize_cases(rows)
    summary_rows = summarize_models(case_rows)
    write_csv(case_csv, case_rows)
    write_csv(summary_csv, summary_rows)
    write_markdown(report_md, statuses, summary_rows, case_rows,
                   combined_csv, case_csv, summary_csv)

    print(f'combined_csv={combined_csv}')
    print(f'case_csv={case_csv}')
    print(f'summary_csv={summary_csv}')
    print(f'report_md={report_md}')


if __name__ == '__main__':
    main()
