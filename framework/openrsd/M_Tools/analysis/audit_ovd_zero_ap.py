#!/usr/bin/env python3
"""Audit zero-AP open-vocabulary classes using existing OpenRSD OVD outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DOTA1_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]
CORE_CLASSES = [
    'plane', 'harbor', 'helicopter', 'large-vehicle', 'storage-tank',
    'bridge', 'ship', 'small-vehicle'
]


def canonical(name: str) -> str:
    return name.lower().replace('_', '-').replace(' ', '-')


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def load_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value, 'cpu'):
        return value.cpu().numpy()
    return np.asarray(value)


def dota_gt_counts(repo_root: Path, angle: str, classes: Sequence[str]) -> dict[str, int]:
    counts = {name: 0 for name in classes}
    ann_dir = repo_root / 'data/DOTA1_1024_500/angle_sweep_val/realistic' / f'angle_{angle}' / 'annfiles'
    for ann_file in ann_dir.glob('*.txt'):
        for line in ann_file.read_text(encoding='utf-8', errors='replace').splitlines():
            parts = line.strip().split()
            if len(parts) >= 9 and parts[8] in counts:
                counts[parts[8]] += 1
    return counts


def prediction_stats(pkl_path: Path, classes: Sequence[str]) -> dict[str, dict[str, Any]]:
    stats = {
        name: {
            'prediction_count': 0,
            'mean_confidence': 0.0,
            'score_sum': 0.0,
            'top_predicted_labels': {},
        }
        for name in classes
    }
    if not pkl_path.exists():
        return stats
    for sample in load_pickle(pkl_path):
        inst = sample.get('pred_instances', {})
        labels = to_numpy(inst.get('labels', [])).astype(int)
        scores = to_numpy(inst.get('scores', [])).astype(float)
        for label, score in zip(labels, scores):
            if 0 <= label < len(classes):
                class_name = classes[label]
            else:
                class_name = f'INVALID_{label}'
            if class_name not in stats:
                stats[class_name] = {
                    'prediction_count': 0,
                    'mean_confidence': 0.0,
                    'score_sum': 0.0,
                    'top_predicted_labels': {},
                }
            item = stats[class_name]
            item['prediction_count'] += 1
            item['score_sum'] += float(score)
            item['top_predicted_labels'][class_name] = item['top_predicted_labels'].get(class_name, 0) + 1
    for item in stats.values():
        count = item['prediction_count']
        item['mean_confidence'] = item['score_sum'] / count if count else 0.0
        item['top_predicted_labels'] = ','.join(
            f'{k}:{v}' for k, v in sorted(item['top_predicted_labels'].items(), key=lambda kv: kv[1], reverse=True)[:5])
        item.pop('score_sum', None)
    return stats


def support_norms(prompt_support_dir: Path, prompt_key: str) -> dict[str, dict[str, Any]]:
    pkl_path = prompt_support_dir / f'{prompt_key}.pkl'
    out: dict[str, dict[str, Any]] = {}
    if not pkl_path.exists():
        return out
    support = load_pickle(pkl_path)
    for class_name, payload in support.items():
        item: dict[str, Any] = {'support_exists': True}
        if isinstance(payload, dict):
            text = payload.get('text_embeds')
            visual = payload.get('visual_embeds')
            item['text_embedding_norm'] = float(np.linalg.norm(np.asarray(text))) if text is not None else None
            item['support_visual_embedding_norm'] = float(np.linalg.norm(np.asarray(visual))) if visual is not None else None
            item['support_visual_count'] = int(np.asarray(visual).shape[0]) if visual is not None and np.asarray(visual).ndim > 1 else 0
        out[canonical(class_name)] = item
    return out


def choose_source_row(existing_work_dir: Path, preferred_prompt: str, angle: str) -> tuple[dict[str, Any], list[str]]:
    sources = [
        existing_work_dir / 'exp_ovd2/exp_ovd2_results.json',
        existing_work_dir / 'exp_ovd3/exp_ovd3_results.json',
        existing_work_dir / 'exp_ovd1/exp_ovd1_results.json',
    ]
    notes = []
    for source in sources:
        report = load_json(source)
        rows = [r for r in report.get('rows', []) if r.get('status') == 'OK' and r.get('angle') == angle]
        preferred = [r for r in rows if r.get('prompt_key') in {preferred_prompt, f'ovd3_{preferred_prompt}'} and r.get('head') == 'alignment']
        if preferred:
            return preferred[0], notes + [f'selected {source} preferred {preferred_prompt}']
        if rows:
            return rows[0], notes + [f'selected {source} first OK row']
    return {}, notes + ['no source row found']


def audit(args: argparse.Namespace) -> dict[str, Any]:
    row, notes = choose_source_row(args.existing_work_dir, args.prompt_key, args.angle)
    classes = list(row.get('per_class_ap50', {}).keys()) or [
        'plane', 'bridge', 'ship', 'small-vehicle', 'large-vehicle',
        'storage-tank', 'harbor', 'roundabout', 'helicopter'
    ]
    gt_counts = dota_gt_counts(args.repo_root, args.angle, DOTA1_CLASSES)
    pred = prediction_stats(Path(row.get('predictions', '')), classes)
    support = support_norms(args.existing_work_dir / 'prompt_support', row.get('prompt_key', args.prompt_key))
    mapping = row.get('class_mapping') or {}
    per_class_ap = row.get('per_class_ap50', {})
    rows = []
    for class_name in CORE_CLASSES:
        class_key = canonical(class_name)
        support_item = support.get(class_key, {})
        pred_item = pred.get(class_name, {'prediction_count': 0, 'mean_confidence': 0.0, 'top_predicted_labels': ''})
        ap = per_class_ap.get(class_name)
        if ap is None and class_name not in classes:
            diagnosis = 'NOT_IN_EVALUATOR_CLASS_LIST'
        elif gt_counts.get(class_name, 0) == 0:
            diagnosis = 'NO_GT_IN_SELECTED_ANGLE'
        elif pred_item.get('prediction_count', 0) == 0:
            diagnosis = 'NO_PREDICTION_FOR_CLASS'
        elif ap == 0:
            diagnosis = 'PREDICTIONS_EXIST_BUT_AP_ZERO_CHECK_IOU_SCORE_OR_LABEL_MAPPING'
        else:
            diagnosis = 'NONZERO_OR_NOT_ZERO_AP'
        rows.append({
            'class': class_name,
            'dota_gt_label_id': DOTA1_CLASSES.index(class_name) if class_name in DOTA1_CLASSES else None,
            'evaluator_label_id': classes.index(class_name) if class_name in classes else None,
            'gt_count': gt_counts.get(class_name, 0),
            'prediction_count': pred_item.get('prediction_count', 0),
            'mean_confidence': pred_item.get('mean_confidence', 0.0),
            'ap50': ap,
            'top_predicted_labels': pred_item.get('top_predicted_labels', ''),
            'text_embedding_norm': support_item.get('text_embedding_norm'),
            'support_visual_exists': bool(support_item.get('support_exists')),
            'support_visual_embedding_norm': support_item.get('support_visual_embedding_norm'),
            'support_visual_count': support_item.get('support_visual_count', 0),
            'prompt_reencoded': row.get('support_meta', {}).get('prompt_reencoded'),
            'prompt_text_or_mapping': mapping.get(class_name, ''),
            'diagnosis': diagnosis,
        })
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'status': 'DONE' if row else 'FAILED',
        'angle': args.angle,
        'source_row': row,
        'notes': notes,
        'dota_class_list': DOTA1_CLASSES,
        'prompt_class_list': row.get('support_meta', {}).get('prompt_texts', []),
        'evaluator_class_list': classes,
        'label_id_mapping': {
            name: {
                'dota_gt_id': DOTA1_CLASSES.index(name) if name in DOTA1_CLASSES else None,
                'evaluator_id': classes.index(name) if name in classes else None,
            }
            for name in sorted(set(DOTA1_CLASSES) | set(classes))
        },
        'csv': str(args.out_csv),
        'rows': rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--existing-work-dir', type=Path, default=Path('/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508'))
    parser.add_argument('--angle', default='000')
    parser.add_argument('--prompt-key', default='F3_orientation_aware')
    parser.add_argument('--out-csv', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    audit(parse_args())


if __name__ == '__main__':
    main()
