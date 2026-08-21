#!/usr/bin/env python
"""Build a Markdown report for the full rotation-robustness experiment."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ANGLES = list(range(0, 360, 15))
RIGHT_ANGLES = {0, 90, 180, 270}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-tsv', required=True)
    parser.add_argument('--out-md', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--title', default='Full Rotation Robustness Results')
    parser.add_argument('--gpus', default='')
    parser.add_argument('--data-root', default='')
    parser.add_argument('--diagnostics-tsv', default='')
    parser.add_argument('--run-log', default='')
    return parser.parse_args()


def read_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='|'))


def latest_task_rows(rows):
    latest = {}
    for row in rows:
        key = (row.get('stage', ''), row.get('model', ''),
               row.get('dataset', ''), row.get('angle', ''))
        latest[key] = row
    return list(latest.values())


def metric_value(value):
    try:
        if value in ('', 'NA', None):
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def fmt(value, digits=4):
    number = metric_value(value)
    if number is None:
        return 'NA'
    return f'{number:.{digits}f}'


def md_escape(value):
    if value is None:
        return ''
    return str(value).replace('|', '\\|')


def code(value):
    if not value:
        return ''
    return f'`{md_escape(value)}`'


def status_counts(rows):
    counts = Counter()
    for row in rows:
        status = row.get('status', 'UNKNOWN')
        if status == 'OK':
            key = 'OK'
        elif status == 'DRY_RUN':
            key = 'DRY_RUN'
        elif status.startswith('SKIP'):
            key = 'SKIP'
        else:
            key = 'FAIL'
        counts[key] += 1
    return counts


def latest_ok_by_key(rows, stage):
    latest = {}
    for row in rows:
        if row.get('stage') != stage or row.get('status') != 'OK':
            continue
        key = (row.get('model', ''), row.get('angle', ''),
               row.get('dataset', ''))
        latest[key] = row
    return latest


def angle_summary(rows):
    by_model_angle = {}
    for row in rows:
        if row.get('stage') != 'angle_sweep' or row.get('status') != 'OK':
            continue
        angle = row.get('angle', '')
        try:
            angle_int = int(float(angle))
        except ValueError:
            continue
        map_value = metric_value(row.get('map'))
        if map_value is None:
            continue
        by_model_angle[(row.get('model', ''), angle_int)] = row

    models = sorted({model for model, _ in by_model_angle})
    rows_out = []
    for model in models:
        values = {}
        for angle in ANGLES:
            row = by_model_angle.get((model, angle))
            if row is None:
                continue
            value = metric_value(row.get('map'))
            if value is not None:
                values[angle] = value
        if not values:
            continue
        m0 = values.get(0)
        nonzero = [v for a, v in values.items() if a != 0]
        right = [v for a, v in values.items() if a in RIGHT_ANGLES]
        non_right = [v for a, v in values.items() if a not in RIGHT_ANGLES]
        worst_angle = min(values, key=values.get)
        best_angle = max(values, key=values.get)
        mean_nonzero = sum(nonzero) / len(nonzero) if nonzero else None
        rows_out.append({
            'model': model,
            'mAP_0': m0,
            'mean_rotated': mean_nonzero,
            'rotation_gap': (m0 - mean_nonzero)
            if m0 is not None and mean_nonzero is not None else None,
            'worst_angle': worst_angle,
            'worst_mAP': values[worst_angle],
            'best_angle': best_angle,
            'best_mAP': values[best_angle],
            'RSI': values[best_angle] - values[worst_angle],
            'right_angle_mean': sum(right) / len(right) if right else None,
            'non_right_angle_mean':
            sum(non_right) / len(non_right) if non_right else None,
            'completed_angles': len(values),
        })
    return rows_out, by_model_angle


def append_angle_sections(lines, rows):
    summary_rows, by_model_angle = angle_summary(rows)
    if not summary_rows:
        lines.append('## Angle Sweep')
        lines.append('')
        lines.append('No completed angle-sweep rows yet.')
        lines.append('')
        return

    lines.append('## Angle Sweep Metrics')
    lines.append('')
    lines.append(
        '| model | completed | mAP@0 | mean rotated | rotation gap | worst angle | worst mAP | best angle | best mAP | RSI | right-angle mean | non-right-angle mean |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in summary_rows:
        lines.append(
            f'| {md_escape(row["model"])} | {row["completed_angles"]} | '
            f'{fmt(row["mAP_0"])} | {fmt(row["mean_rotated"])} | '
            f'{fmt(row["rotation_gap"])} | {row["worst_angle"]} | '
            f'{fmt(row["worst_mAP"])} | {row["best_angle"]} | '
            f'{fmt(row["best_mAP"])} | {fmt(row["RSI"])} | '
            f'{fmt(row["right_angle_mean"])} | '
            f'{fmt(row["non_right_angle_mean"])} |')
    lines.append('')

    lines.append('## Angle Sweep mAP')
    lines.append('')
    header = '| model | ' + ' | '.join(str(a) for a in ANGLES) + ' |'
    sep = '|---|' + '|'.join(['---:'] * len(ANGLES)) + '|'
    lines.append(header)
    lines.append(sep)
    for model in sorted({model for model, _ in by_model_angle}):
        cells = []
        for angle in ANGLES:
            row = by_model_angle.get((model, angle))
            cells.append(fmt(row.get('map')) if row else 'NA')
        lines.append(f'| {md_escape(model)} | ' + ' | '.join(cells) + ' |')
    lines.append('')


def append_stage_table(lines, rows, stage, title):
    stage_rows = [row for row in rows if row.get('stage') == stage]
    if not stage_rows:
        return
    lines.append(f'## {title}')
    lines.append('')
    lines.append(
        '| model | dataset | angle | status | batch | mAP | AP50 | log | predictions | diagnosis |')
    lines.append('|---|---|---:|---|---:|---:|---:|---|---|---|')
    for row in stage_rows:
        lines.append(
            f'| {md_escape(row.get("model", ""))} | '
            f'{md_escape(row.get("dataset", ""))} | '
            f'{md_escape(row.get("angle", ""))} | '
            f'{md_escape(row.get("status", ""))} | '
            f'{md_escape(row.get("batch_size", ""))} | '
            f'{fmt(row.get("map"))} | {fmt(row.get("ap50"))} | '
            f'{code(row.get("log", ""))} | {code(row.get("predictions", ""))} | '
            f'{md_escape(row.get("diagnosis", ""))} |')
    lines.append('')


def append_failures(lines, rows):
    failures = [
        row for row in rows
        if row.get('status') not in ('OK', 'DRY_RUN')
        and not row.get('status', '').startswith('SKIP(done)')
    ]
    if not failures:
        return
    lines.append('## Failures And Skips')
    lines.append('')
    lines.append(
        '| stage | model | dataset | angle | status | batch | diagnosis | log |')
    lines.append('|---|---|---|---:|---|---:|---|---|')
    for row in failures:
        lines.append(
            f'| {md_escape(row.get("stage", ""))} | '
            f'{md_escape(row.get("model", ""))} | '
            f'{md_escape(row.get("dataset", ""))} | '
            f'{md_escape(row.get("angle", ""))} | '
            f'{md_escape(row.get("status", ""))} | '
            f'{md_escape(row.get("batch_size", ""))} | '
            f'{md_escape(row.get("diagnosis", ""))} | '
            f'{code(row.get("log", ""))} |')
    lines.append('')


def append_diagnostics(lines, diagnostics_path):
    rows = read_rows(diagnostics_path) if diagnostics_path else []
    rows = [row for row in rows if row.get('scope') == 'all']
    if not rows:
        return
    lines.append('## Classification vs Localization Diagnostics')
    lines.append('')
    lines.append(
        '| model | angle | gts | dets | recall@IoU50 | mean TP score | class-wrong rate | loc-fail rate | missed rate | fp/image |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        lines.append(
            f'| {md_escape(row.get("model", ""))} | '
            f'{md_escape(row.get("angle", ""))} | '
            f'{md_escape(row.get("gts", ""))} | '
            f'{md_escape(row.get("dets", ""))} | '
            f'{fmt(row.get("recall_at_iou50"), 6)} | '
            f'{fmt(row.get("mean_tp_score"), 6)} | '
            f'{fmt(row.get("class_wrong_rate"), 6)} | '
            f'{fmt(row.get("localization_fail_rate"), 6)} | '
            f'{fmt(row.get("missed_rate"), 6)} | '
            f'{fmt(row.get("fp_per_image"), 6)} |')
    lines.append('')


def main():
    args = parse_args()
    all_rows = read_rows(args.results_tsv)
    rows = latest_task_rows(all_rows)
    counts = status_counts(rows)
    lines = [
        f'# {args.title}',
        '',
        f'- generated_at: `{datetime.now().strftime("%F %T")}`',
        f'- out_root: `{args.out_root}`',
        f'- results_tsv: `{args.results_tsv}`',
        f'- diagnostics_tsv: `{args.diagnostics_tsv}`',
        f'- run_log: `{args.run_log}`',
        f'- gpus: `{args.gpus}`',
        f'- data_root: `{args.data_root}`',
        f'- current_task_rows: `{len(rows)}`',
        f'- raw_history_rows: `{len(all_rows)}`',
        f'- status_counts: `{dict(counts)}`',
        '',
    ]

    append_angle_sections(lines, rows)
    append_stage_table(lines, rows, 'tta_eval', 'Rotation TTA Upper Bound')
    append_stage_table(lines, rows, 'diagnostic', 'Diagnostic Artifacts')
    append_diagnostics(lines, args.diagnostics_tsv)
    append_failures(lines, rows)

    lines.append('## Raw Result Rows')
    lines.append('')
    lines.append(
        '| stage | model | dataset | angle | epoch | batch | status | mAP | AP50 | checkpoint | log | predictions | diagnosis |')
    lines.append('|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|')
    for row in all_rows:
        lines.append(
            f'| {md_escape(row.get("stage", ""))} | '
            f'{md_escape(row.get("model", ""))} | '
            f'{md_escape(row.get("dataset", ""))} | '
            f'{md_escape(row.get("angle", ""))} | '
            f'{md_escape(row.get("epoch", ""))} | '
            f'{md_escape(row.get("batch_size", ""))} | '
            f'{md_escape(row.get("status", ""))} | '
            f'{fmt(row.get("map"))} | {fmt(row.get("ap50"))} | '
            f'{code(row.get("checkpoint", ""))} | '
            f'{code(row.get("log", ""))} | '
            f'{code(row.get("predictions", ""))} | '
            f'{md_escape(row.get("diagnosis", ""))} |')
    lines.append('')

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
