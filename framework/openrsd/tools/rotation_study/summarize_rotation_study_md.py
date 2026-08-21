#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--out-md', required=True)
    parser.add_argument('--title', default='Rotation Study P0-P3 Results')
    return parser.parse_args()


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def fmt_float(value, digits=4):
    if value in (None, '', 'NA'):
        return 'NA'
    try:
        return f'{float(value):.{digits}f}'
    except ValueError:
        return str(value)


def md(value):
    return str(value if value is not None else '').replace('|', '\\|')


def code(value):
    value = str(value if value is not None else '')
    return f'`{md(value)}`' if value else ''


def status_counts(rows):
    counts = {}
    for row in rows:
        status = row.get('status', 'UNKNOWN')
        key = status if status in ('OK', 'SMOKE_OK') else (
            'FAIL' if status.startswith('FAIL') else status)
        counts[key] = counts.get(key, 0) + 1
    return counts


def latest_rows(rows):
    latest = {}
    for row in rows:
        key = (
            row.get('stage', ''),
            row.get('dataset', ''),
            row.get('task_granularity', ''),
            row.get('model', ''),
            row.get('eval_split', ''),
            row.get('angle', ''),
        )
        latest[key] = row
    return list(latest.values())


def append_stage_table(lines, rows, stage):
    stage_rows = [r for r in rows if r.get('stage') == stage]
    if not stage_rows:
        return
    lines.append(f'## {stage}')
    lines.append('')
    lines.append('| dataset | granularity | model | split | rotation | angle | status | mAP | AP50 | AP07 | AP12 | predictions | log |')
    lines.append('|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|')
    for row in stage_rows:
        lines.append(
            f'| {md(row.get("dataset"))} | {md(row.get("task_granularity"))} | '
            f'{md(row.get("model"))} | {md(row.get("eval_split"))} | '
            f'{md(row.get("rotation_type"))} | {md(row.get("angle"))} | '
            f'{md(row.get("status"))} | {fmt_float(row.get("mAP"))} | '
            f'{fmt_float(row.get("AP50"))} | {fmt_float(row.get("ap07_mAP"))} | '
            f'{fmt_float(row.get("ap12_mAP"))} | {code(row.get("prediction_path"))} | '
            f'{code(row.get("log_path"))} |')
    lines.append('')


def append_p0_gap_table(lines, rows):
    ok_rows = [r for r in rows if r.get('stage') == 'P0' and r.get('status') == 'OK']
    by_key = {}
    for row in ok_rows:
        key = (row.get('dataset'), row.get('task_granularity'), row.get('model'))
        by_key.setdefault(key, {})[row.get('rotation_type')] = row
    gap_rows = []
    for key, split_rows in by_key.items():
        clean = split_rows.get('clean')
        rotated = split_rows.get('rotated') or split_rows.get('non_right')
        if not clean or not rotated:
            continue
        clean_map = clean.get('mAP')
        rot_map = rotated.get('mAP')
        try:
            gap = float(clean_map) - float(rot_map)
            rel_gap = gap / float(clean_map) if float(clean_map) else None
        except (TypeError, ValueError):
            gap = rel_gap = None
        gap_rows.append((key, clean_map, rot_map, gap, rel_gap))
    if not gap_rows:
        return
    lines.append('## P0 Rotation Gap Matrix')
    lines.append('')
    lines.append('| dataset | granularity | model | clean mAP | rotated mAP | gap | relative gap |')
    lines.append('|---|---|---|---:|---:|---:|---:|')
    for (dataset, gran, model), clean_map, rot_map, gap, rel_gap in gap_rows:
        lines.append(
            f'| {md(dataset)} | {md(gran)} | {md(model)} | '
            f'{fmt_float(clean_map)} | {fmt_float(rot_map)} | '
            f'{fmt_float(gap)} | {fmt_float(rel_gap)} |')
    lines.append('')


def append_fair1m5(lines, out_root):
    paths = sorted((out_root / 'p1_granularity').glob('*.csv'))
    if not paths:
        return
    lines.append('## P1 FAIR1M-5 Remap')
    lines.append('')
    for path in paths:
        rows = read_csv(path)
        lines.append(f'### {path.name}')
        lines.append('')
        lines.append('| split | granularity | class | AP50 |')
        lines.append('|---|---|---|---:|')
        for row in rows:
            lines.append(
                f'| {md(row.get("split"))} | {md(row.get("granularity"))} | '
                f'{md(row.get("class"))} | {fmt_float(row.get("AP50"))} |')
        lines.append('')


def append_error_decomp(lines, out_root):
    path = out_root / 'error_decomp' / 'p3_error_decomp.csv'
    rows = read_csv(path)
    if not rows:
        return
    lines.append('## P3 Error Decomposition')
    lines.append('')
    lines.append('| dataset | model | split | scope | images | gts | dets/img | recall@50 | correct | wrong class | loc fail | missed | false pos |')
    lines.append('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        if row.get('scope') != 'all':
            continue
        lines.append(
            f'| {md(row.get("dataset"))} | {md(row.get("model"))} | '
            f'{md(row.get("split"))} | {md(row.get("scope"))} | '
            f'{md(row.get("images"))} | {md(row.get("gts"))} | '
            f'{fmt_float(row.get("dets_per_image"))} | '
            f'{fmt_float(row.get("recall_at_iou50"))} | {md(row.get("correct"))} | '
            f'{md(row.get("wrong_class_given_matched"))} | '
            f'{md(row.get("localization_fail"))} | {md(row.get("missed"))} | '
            f'{md(row.get("false_positive"))} |')
    lines.append('')


def append_angle_summary(lines, rows):
    ok_rows = [r for r in rows if r.get('stage') == 'P2' and r.get('status') == 'OK']
    by_model = {}
    for row in ok_rows:
        try:
            angle = int(float(row.get('angle')))
            map_value = float(row.get('mAP'))
        except (TypeError, ValueError):
            continue
        key = (row.get('dataset'), row.get('model'))
        by_model.setdefault(key, {})[angle] = map_value
    if not by_model:
        return
    lines.append('## P2 Angle Response Summary')
    lines.append('')
    lines.append('| dataset | model | completed angles | mAP@0 | worst angle | worst mAP | mean nonzero | RSI |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for (dataset, model), values in sorted(by_model.items()):
        worst_angle = min(values, key=values.get)
        m0 = values.get(0)
        nonzero = [v for a, v in values.items() if a != 0]
        mean_nonzero = sum(nonzero) / len(nonzero) if nonzero else None
        rsi = (m0 - mean_nonzero) if m0 is not None and mean_nonzero is not None else None
        lines.append(
            f'| {md(dataset)} | {md(model)} | {len(values)} | {fmt_float(m0)} | '
            f'{worst_angle} | {fmt_float(values[worst_angle])} | '
            f'{fmt_float(mean_nonzero)} | {fmt_float(rsi)} |')
    lines.append('')


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    rows = latest_rows(read_csv(out_root / 'results_rotation_study.csv'))
    lines = [
        f'# {args.title}',
        '',
        f'- generated_at: `{datetime.now().strftime("%F %T")}`',
        f'- out_root: `{out_root}`',
        f'- result_csv: `{out_root / "results_rotation_study.csv"}`',
        '',
    ]
    counts = status_counts(rows)
    if counts:
        lines.append('## Status')
        lines.append('')
        lines.append('| status | count |')
        lines.append('|---|---:|')
        for key, value in sorted(counts.items()):
            lines.append(f'| {md(key)} | {value} |')
        lines.append('')
    append_p0_gap_table(lines, rows)
    append_angle_summary(lines, rows)
    append_fair1m5(lines, out_root)
    append_error_decomp(lines, out_root)
    for stage in ('P0', 'P1', 'P2', 'P3'):
        append_stage_table(lines, rows, stage)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
