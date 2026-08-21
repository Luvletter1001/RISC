#!/usr/bin/env python
"""Refine E9 case-level mechanism labels.

The first E9 summarizer uses a small set of coarse labels.  This script keeps
those labels intact and adds a finer diagnosis that separates stable
other-class attractors, class shifts after scale change, context shortcuts,
and missing-match failure modes.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_CASE_CSV = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'e9_scale_counterfactual_manifest_full_combined/e9_case_diagnosis.csv')
DEFAULT_OUT_DIR = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'p1_e9_refined_labels')

CLASS_FAMILIES = {
    'plane': 'aviation_context',
    'airport': 'aviation_context',
    'helicopter': 'aviation_context',
    'helipad': 'aviation_context',
    'ship': 'maritime_context',
    'harbor': 'maritime_context',
    'small-vehicle': 'vehicle_like',
    'large-vehicle': 'vehicle_like',
    'storage-tank': 'industrial_storage',
    'container-crane': 'industrial_storage',
    'tennis-court': 'court_field_like',
    'basketball-court': 'court_field_like',
    'baseball-diamond': 'court_field_like',
    'soccer-ball-field': 'court_field_like',
    'ground-track-field': 'court_field_like',
    'swimming-pool': 'court_field_like',
    'bridge': 'road_structure',
    'roundabout': 'road_structure',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-csv', default=DEFAULT_CASE_CSV)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
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


def class_family(cls):
    if not cls or cls == 'no_match':
        return 'no_match'
    return CLASS_FAMILIES.get(cls, 'other_class')


def relation(cls, target_class, impossible_class):
    if not cls or cls == 'no_match':
        return 'no_match'
    if cls == target_class:
        return 'target'
    if cls == impossible_class:
        return 'impossible'
    return 'other'


def classify_case(row):
    target = row.get('target_class', '')
    impossible = row.get('impossible_class', '')
    original = row.get('original_cls', '')
    same = row.get('neutral_same_cls', '')
    small = row.get('neutral_small_cls', '')
    original_rel = relation(original, target, impossible)
    same_rel = relation(same, target, impossible)
    small_rel = relation(small, target, impossible)
    original_family = class_family(original)
    same_family = class_family(same)
    small_family = class_family(small)

    if original_rel == 'no_match':
        refined = 'insufficient_original_match'
    elif same_rel == 'no_match':
        refined = 'insufficient_neutral_same_match'
    elif same_rel == 'impossible':
        if small_rel == 'no_match':
            refined = 'cls_reg_scale_decoupling_small_unmatched'
        else:
            refined = 'cls_reg_scale_decoupling'
    elif small_rel == 'no_match':
        refined = 'insufficient_neutral_small_match'
    elif original_rel == 'impossible' and same_rel == 'target':
        if small_rel == 'impossible':
            refined = 'context_shortcut_scale_prior_sensitive'
        elif small_rel == 'target':
            refined = 'context_shortcut_scale_stable'
        else:
            refined = 'context_shortcut_small_scale_other'
    elif same_rel == 'target' and small_rel == 'impossible':
        refined = 'scale_prior_sensitive'
    elif original_rel == 'target' and same_rel == 'target':
        if small_rel == 'target':
            refined = 'not_reproduced_scale_stable'
        elif small_rel == 'other':
            refined = 'not_reproduced_small_scale_other'
        else:
            refined = 'not_reproduced_after_rerun'
    elif same_rel == 'other' and small_rel == 'other':
        if same == small:
            refined = 'neutral_stable_other_attractor'
        elif same_family == small_family:
            refined = 'neutral_other_class_shift'
        else:
            refined = 'neutral_cross_family_shift'
    elif same_rel == 'other' and small_rel == 'target':
        refined = 'neutral_same_other_small_target'
    elif same_rel == 'other' and small_rel == 'impossible':
        refined = 'neutral_same_other_small_impossible'
    elif same_rel == 'target' and small_rel == 'other':
        refined = 'target_same_small_other'
    else:
        refined = 'mixed_unclassified'

    flags = []
    if original_rel == 'impossible':
        flags.append('original_impossible')
    if same_rel == 'target':
        flags.append('same_target')
    if same_rel == 'impossible':
        flags.append('same_impossible')
    if same_rel == 'other':
        flags.append('same_other')
    if small_rel == 'target':
        flags.append('small_target')
    if small_rel == 'impossible':
        flags.append('small_impossible')
    if small_rel == 'other':
        flags.append('small_other')
    if 'no_match' in {original_rel, same_rel, small_rel}:
        flags.append('has_no_match')

    return {
        'original_relation': original_rel,
        'neutral_same_relation': same_rel,
        'neutral_small_relation': small_rel,
        'original_family': original_family,
        'neutral_same_family': same_family,
        'neutral_small_family': small_family,
        'refined_diagnosis': refined,
        'mechanism_flags': ';'.join(flags),
    }


def enrich_rows(rows):
    enriched = []
    for row in rows:
        enriched.append({**row, **classify_case(row)})
    return enriched


def summarize_by_model(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get('model', '')].append(row)
    out = []
    labels = sorted(set(row['refined_diagnosis'] for row in rows))
    for model, items in sorted(grouped.items()):
        counts = Counter(row['refined_diagnosis'] for row in items)
        record = {'model': model, 'cases': len(items)}
        for label in labels:
            record[label] = counts.get(label, 0)
        out.append(record)
    return out


def summarize_transitions(rows):
    counts = Counter()
    for row in rows:
        key = (
            row['original_relation'],
            row['neutral_same_relation'],
            row['neutral_small_relation'],
            row['neutral_same_family'],
            row['neutral_small_family'],
            row['refined_diagnosis'],
        )
        counts[key] += 1
    out = []
    for key, count in counts.most_common():
        original_rel, same_rel, small_rel, same_family, small_family, label = key
        out.append({
            'cases': count,
            'original_relation': original_rel,
            'neutral_same_relation': same_rel,
            'neutral_small_relation': small_rel,
            'neutral_same_family': same_family,
            'neutral_small_family': small_family,
            'refined_diagnosis': label,
        })
    return out


def write_markdown(path, rows, model_rows, transition_rows, refined_csv,
                   model_csv, transition_csv):
    coarse = Counter(row.get('diagnosis', '') for row in rows)
    refined = Counter(row['refined_diagnosis'] for row in rows)
    family_same = Counter(row['neutral_same_family'] for row in rows)
    family_small = Counter(row['neutral_small_family'] for row in rows)
    lines = [
        '# P1 E9 Refined Diagnosis Labels',
        '',
        '本报告把 E9 的 coarse diagnosis 细分为更可审稿的机制标签，'
        '重点拆开原来的 `other_or_mixed`。',
        '',
        '## Artifacts',
        '',
        f'- refined_csv: `{refined_csv}`',
        f'- model_summary_csv: `{model_csv}`',
        f'- transition_summary_csv: `{transition_csv}`',
        '',
        '## Coarse Diagnosis Counts',
        '',
        '| coarse_diagnosis | cases |',
        '|---|---:|',
    ]
    for label, count in coarse.most_common():
        lines.append(f'| `{label}` | {count} |')
    lines.extend([
        '',
        '## Refined Diagnosis Counts',
        '',
        '| refined_diagnosis | cases |',
        '|---|---:|',
    ])
    for label, count in refined.most_common():
        lines.append(f'| `{label}` | {count} |')
    lines.extend([
        '',
        '## Neutral Same-Scale Family Counts',
        '',
        '| family | cases |',
        '|---|---:|',
    ])
    for label, count in family_same.most_common():
        lines.append(f'| `{label}` | {count} |')
    lines.extend([
        '',
        '## Neutral Small-Scale Family Counts',
        '',
        '| family | cases |',
        '|---|---:|',
    ])
    for label, count in family_small.most_common():
        lines.append(f'| `{label}` | {count} |')
    lines.extend([
        '',
        '## Top Transition Patterns',
        '',
        '| cases | original_relation | neutral_same_relation | neutral_small_relation | neutral_same_family | neutral_small_family | refined_diagnosis |',
        '|---:|---|---|---|---|---|---|',
    ])
    for row in transition_rows[:30]:
        lines.append(
            f'| {row["cases"]} | {row["original_relation"]} | '
            f'{row["neutral_same_relation"]} | '
            f'{row["neutral_small_relation"]} | '
            f'{row["neutral_same_family"]} | '
            f'{row["neutral_small_family"]} | '
            f'`{row["refined_diagnosis"]}` |')
    lines.extend([
        '',
        '## Model Summary',
        '',
    ])
    if model_rows:
        labels = [key for key in model_rows[0] if key not in {'model', 'cases'}]
        lines.append('| model | cases | ' + ' | '.join(labels) + ' |')
        lines.append('|---|---:|' + '|'.join('---:' for _ in labels) + '|')
        for row in model_rows:
            values = ' | '.join(str(row.get(label, 0)) for label in labels)
            lines.append(f'| `{row["model"]}` | {row["cases"]} | {values} |')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = enrich_rows(read_csv_rows(args.case_csv))
    model_rows = summarize_by_model(rows)
    transition_rows = summarize_transitions(rows)
    refined_csv = out_dir / 'e9_case_diagnosis_refined.csv'
    model_csv = out_dir / 'e9_refined_model_summary.csv'
    transition_csv = out_dir / 'e9_refined_transition_summary.csv'
    report_md = out_dir / 'e9_refined_diagnosis_report.md'
    write_csv(refined_csv, rows)
    write_csv(model_csv, model_rows)
    write_csv(transition_csv, transition_rows)
    write_markdown(report_md, rows, model_rows, transition_rows,
                   refined_csv, model_csv, transition_csv)
    print(f'refined_csv={refined_csv}')
    print(f'model_summary_csv={model_csv}')
    print(f'transition_summary_csv={transition_csv}')
    print(f'report_md={report_md}')
    print(f'cases={len(rows)}')


if __name__ == '__main__':
    main()
