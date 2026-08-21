#!/usr/bin/env python3
"""Scan all rotation angles and plot small-vehicle ratios per model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import site
import statistics
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / 'vis' / 'P0148__1024__651___0' / 'dataset')
DEFAULT_BASE_RESULTS = (
    PROJECT_ROOT / 'vis' / 'P0148__1024__651___0' / 'results.pkl')
DEFAULT_CONFIG = (
    PROJECT_ROOT / 'M_configs' / 'Vis' /
    'A12_flex_rtm_v3_1_DOTA2only_ss_train_vis.py')
DEFAULT_FINETUNE_CKPT = (
    PROJECT_ROOT / 'results' /
    'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train' / 'epoch_12.pth')
DEFAULT_NOTEXT_CKPT = (
    PROJECT_ROOT / 'results' /
    'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00' /
    'epoch_12.pth')
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / 'workdir_vis' / 'rotation_small_vehicle_scan' /
    'P0148__1024__651___0')
DEFAULT_PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
DEFAULT_EXTRA_PYTHONPATH = Path('/tmp/openrsd_extra')
DEFAULT_EXPECTED_ANGLES = list(range(0, 360, 5))
ANGLE_PATTERN = re.compile(r'_rot(\d{3})(?:\.[^.]+)?$')
SMALL_VEHICLE_LABEL = 'small-vehicle'
PLOT_COLORS = OrderedDict([
    ('base', '#e84a8a'),
    ('finetune', '#7a4cff'),
    ('notext_cls', '#31b768'),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export per-angle small-vehicle statistics and curves.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset-root',
        default=str(DEFAULT_DATASET_ROOT),
        help='Dataset root that contains images/ and annfiles/.')
    parser.add_argument(
        '--img-dir',
        default='images',
        help='Image subdirectory under dataset-root.')
    parser.add_argument(
        '--ann-dir',
        default='annfiles',
        help='Annotation subdirectory under dataset-root.')
    parser.add_argument(
        '--base-results',
        default=str(DEFAULT_BASE_RESULTS),
        help='Precomputed base-model results.pkl from SimpleRun.')
    parser.add_argument(
        '--config',
        default=str(DEFAULT_CONFIG),
        help='Config used by tools/test.py for finetuned models.')
    parser.add_argument(
        '--finetune-checkpoint',
        default=str(DEFAULT_FINETUNE_CKPT),
        help='Checkpoint for the finetuned model.')
    parser.add_argument(
        '--notext-checkpoint',
        default=str(DEFAULT_NOTEXT_CKPT),
        help='Checkpoint for the notext_cls model.')
    parser.add_argument(
        '--out-dir',
        default=str(DEFAULT_OUT_DIR),
        help='Output directory for dumps, tables, and plots.')
    parser.add_argument(
        '--python',
        default=str(DEFAULT_PYTHON),
        help='Python executable used to call tools/test.py.')
    parser.add_argument(
        '--vis-score-thr',
        type=float,
        default=0.3,
        help='Score threshold used for the comparable visible-prediction stats.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size passed to the vis dataloader env vars.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Num workers passed to the vis dataloader env vars.')
    parser.add_argument(
        '--cuda-visible-devices',
        default='0',
        help='CUDA_VISIBLE_DEVICES passed to tools/test.py.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate finetuned dump pkls even if they already exist.')
    return parser.parse_args()


def normalize_label(label: Any) -> str:
    text = str(label).strip().lower().replace('_', '-')
    return re.sub(r'\s+', '-', text)


def parse_angle(name: str) -> int:
    match = ANGLE_PATTERN.search(name)
    if not match:
        raise ValueError(f'failed to parse angle from name: {name}')
    return int(match.group(1))


def to_list(value: Any) -> list[Any]:
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def ensure_extra_pythonpath(extra_dir: Path) -> None:
    extra_dir.mkdir(parents=True, exist_ok=True)
    link = extra_dir / 'M_AD'
    target = PROJECT_ROOT / 'M_AD'
    if link.is_symlink() or link.exists():
        try:
            if link.resolve() == target.resolve():
                return
        except FileNotFoundError:
            pass
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError(f'cannot replace existing directory: {link}')
        link.unlink()
    os.symlink(target, link)


def strip_user_site_paths() -> None:
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        user_sites = {Path(user_site).resolve()}
    else:
        user_sites = {Path(path).resolve() for path in user_site}

    cleaned = []
    cwd = Path.cwd().resolve()
    for path_item in sys.path:
        abs_path = cwd if path_item == '' else Path(path_item).resolve()
        if abs_path in user_sites:
            continue
        cleaned.append(path_item)
    sys.path = cleaned


def compute_label_stats(labels: list[str]) -> dict[str, Any]:
    normalized = [normalize_label(label) for label in labels]
    counts = Counter(normalized)
    total = len(normalized)
    dominant_class = ''
    dominant_count = 0
    if counts:
        dominant_class, dominant_count = counts.most_common(1)[0]
    small_vehicle_preds = counts.get(SMALL_VEHICLE_LABEL, 0)
    return dict(
        total_preds=total,
        small_vehicle_preds=small_vehicle_preds,
        small_vehicle_ratio=(
            small_vehicle_preds / total if total else 0.0),
        non_small_vehicle_preds=total - small_vehicle_preds,
        dominant_class=dominant_class,
        dominant_count=dominant_count,
        unique_classes=len(counts),
        class_counts=dict(sorted(counts.items())))


def run_test_dump(
        *,
        model_name: str,
        checkpoint: Path,
        dump_path: Path,
        work_dir: Path,
        log_path: Path,
        args: argparse.Namespace) -> None:
    if dump_path.exists() and not args.force:
        return

    ensure_extra_pythonpath(DEFAULT_EXTRA_PYTHONPATH)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONPATH'] = str(DEFAULT_EXTRA_PYTHONPATH)
    env['MPLCONFIGDIR'] = str(PROJECT_ROOT / 'SimpleRun' / '.mplconfig')
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    env['OPENRSD_VIS_DATA_ROOT'] = str(Path(args.dataset_root))
    env['OPENRSD_VIS_IMG_DIR'] = args.img_dir
    env['OPENRSD_VIS_ANN_DIR'] = args.ann_dir
    env['OPENRSD_VIS_BATCH_SIZE'] = str(args.batch_size)
    env['OPENRSD_VIS_NUM_WORKERS'] = str(args.num_workers)

    cmd = [
        args.python,
        str(PROJECT_ROOT / 'tools' / 'test.py'),
        str(args.config),
        str(checkpoint),
        '--work-dir',
        str(work_dir),
        '--out',
        str(dump_path),
    ]

    with open(log_path, 'w', encoding='utf-8') as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'{model_name} dump failed, see log: {log_path}')


def build_base_rows(base_results_path: Path) -> dict[int, dict[str, Any]]:
    with open(base_results_path, 'rb') as handle:
        data = pickle.load(handle)

    rows: dict[int, dict[str, Any]] = {}
    for image_key, payload in data.items():
        angle = parse_angle(image_key)
        vis_labels = to_list(payload.get('texts', []))
        vis_stats = compute_label_stats(vis_labels)
        scores = [float(score) for score in to_list(payload.get('scores', []))]
        rows[angle] = dict(
            image_name=image_key,
            source_path=str(base_results_path),
            vis=vis_stats,
            raw=None,
            score_min=min(scores) if scores else None,
            score_mean=(
                sum(scores) / len(scores) if scores else None),
            score_max=max(scores) if scores else None,
        )
    return rows


def build_dump_rows(
        dump_path: Path,
        vis_score_thr: float) -> dict[int, dict[str, Any]]:
    with open(dump_path, 'rb') as handle:
        data = pickle.load(handle)

    rows: dict[int, dict[str, Any]] = {}
    for sample in data:
        image_name = Path(sample['img_path']).name
        angle = parse_angle(image_name)
        pred_instances = sample['pred_instances']
        cls_list = [normalize_label(label) for label in to_list(
            sample.get('cls_list', []))]
        labels = [int(label) for label in to_list(pred_instances['labels'])]
        scores = [float(score) for score in to_list(pred_instances['scores'])]
        raw_labels = [cls_list[label] for label in labels]
        vis_labels = [
            label for label, score in zip(raw_labels, scores)
            if score >= vis_score_thr
        ]
        rows[angle] = dict(
            image_name=image_name,
            source_path=str(dump_path),
            vis=compute_label_stats(vis_labels),
            raw=compute_label_stats(raw_labels),
            score_min=min(scores) if scores else None,
            score_mean=(
                sum(scores) / len(scores) if scores else None),
            score_max=max(scores) if scores else None,
        )
    return rows


def validate_angles(
        *,
        rows_by_model: OrderedDict[str, dict[int, dict[str, Any]]],
        expected_angles: list[int]) -> None:
    for model_name, rows in rows_by_model.items():
        missing = sorted(set(expected_angles) - set(rows))
        extra = sorted(set(rows) - set(expected_angles))
        if missing or extra:
            raise RuntimeError(
                f'{model_name} angles mismatch, missing={missing}, extra={extra}')


def build_long_rows(
        rows_by_model: OrderedDict[str, dict[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    long_rows: list[dict[str, Any]] = []
    for model_name, rows in rows_by_model.items():
        for angle in sorted(rows):
            payload = rows[angle]
            vis = payload['vis']
            raw = payload['raw']
            long_rows.append(dict(
                model=model_name,
                angle_deg=angle,
                image_name=payload['image_name'],
                source_path=payload['source_path'],
                vis_total_preds=vis['total_preds'],
                vis_small_vehicle_preds=vis['small_vehicle_preds'],
                vis_small_vehicle_ratio=vis['small_vehicle_ratio'],
                vis_non_small_vehicle_preds=vis['non_small_vehicle_preds'],
                vis_dominant_class=vis['dominant_class'],
                vis_dominant_count=vis['dominant_count'],
                vis_unique_classes=vis['unique_classes'],
                raw_total_preds=raw['total_preds'] if raw else None,
                raw_small_vehicle_preds=(
                    raw['small_vehicle_preds'] if raw else None),
                raw_small_vehicle_ratio=(
                    raw['small_vehicle_ratio'] if raw else None),
                raw_non_small_vehicle_preds=(
                    raw['non_small_vehicle_preds'] if raw else None),
                raw_dominant_class=raw['dominant_class'] if raw else None,
                raw_dominant_count=raw['dominant_count'] if raw else None,
                raw_unique_classes=raw['unique_classes'] if raw else None,
                score_min=payload['score_min'],
                score_mean=payload['score_mean'],
                score_max=payload['score_max'],
            ))
    return long_rows


def build_wide_rows(
        rows_by_model: OrderedDict[str, dict[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    wide_rows: list[dict[str, Any]] = []
    all_angles = sorted(next(iter(rows_by_model.values())).keys())
    for angle in all_angles:
        row: dict[str, Any] = {'angle_deg': angle}
        for model_name, rows in rows_by_model.items():
            vis = rows[angle]['vis']
            row[f'{model_name}_vis_small_vehicle_ratio'] = (
                vis['small_vehicle_ratio'])
            row[f'{model_name}_vis_small_vehicle_preds'] = (
                vis['small_vehicle_preds'])
            row[f'{model_name}_vis_total_preds'] = vis['total_preds']
            row[f'{model_name}_vis_dominant_class'] = vis['dominant_class']
        wide_rows.append(row)
    return wide_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f'no rows to write: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
        path: Path,
        *,
        args: argparse.Namespace,
        rows_by_model: OrderedDict[str, dict[int, dict[str, Any]]]) -> None:
    payload = dict(
        dataset_root=str(Path(args.dataset_root)),
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        vis_score_thr=args.vis_score_thr,
        models={
            model_name: {
                str(angle): row for angle, row in sorted(rows.items())
            }
            for model_name, rows in rows_by_model.items()
        })
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def summarize_model(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ratios = [payload['vis']['small_vehicle_ratio'] for payload in rows.values()]
    full_angles = [
        angle for angle, payload in sorted(rows.items())
        if payload['vis']['total_preds'] > 0
        and payload['vis']['small_vehicle_ratio'] == 1.0
    ]
    return dict(
        mean_ratio=statistics.fmean(ratios),
        median_ratio=statistics.median(ratios),
        min_ratio=min(ratios),
        max_ratio=max(ratios),
        all_small_vehicle_angles=full_angles,
    )


def write_markdown_summary(
        path: Path,
        *,
        args: argparse.Namespace,
        rows_by_model: OrderedDict[str, dict[int, dict[str, Any]]],
        wide_rows: list[dict[str, Any]]) -> None:
    lines = [
        '# Small-Vehicle Rotation Scan',
        '',
        f'- Dataset root: `{Path(args.dataset_root)}`',
        f'- Visualization score threshold: `{args.vis_score_thr}`',
        '',
        '## Model Summary',
        '',
        '| Model | Mean Ratio | Median Ratio | Min Ratio | Max Ratio | All-Small-Vehicle Angles |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for model_name, rows in rows_by_model.items():
        summary = summarize_model(rows)
        full_angles = ', '.join(
            f'{angle:03d}' for angle in summary['all_small_vehicle_angles'])
        lines.append(
            f'| {model_name} | '
            f'{summary["mean_ratio"]:.4f} | '
            f'{summary["median_ratio"]:.4f} | '
            f'{summary["min_ratio"]:.4f} | '
            f'{summary["max_ratio"]:.4f} | '
            f'{full_angles or "-"} |')

    lines.extend([
        '',
        '## Per-Angle Table',
        '',
        '| Angle | Base Ratio | Finetune Ratio | Notext Ratio | Base Preds | Finetune Preds | Notext Preds |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in wide_rows:
        lines.append(
            f'| {row["angle_deg"]:03d} | '
            f'{row["base_vis_small_vehicle_ratio"]:.4f} | '
            f'{row["finetune_vis_small_vehicle_ratio"]:.4f} | '
            f'{row["notext_cls_vis_small_vehicle_ratio"]:.4f} | '
            f'{row["base_vis_total_preds"]} | '
            f'{row["finetune_vis_total_preds"]} | '
            f'{row["notext_cls_vis_total_preds"]} |')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def save_plot(
        path: Path,
        rows_by_model: OrderedDict[str, dict[int, dict[str, Any]]]) -> None:
    strip_user_site_paths()
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_ratio, ax_count) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)

    for model_name, rows in rows_by_model.items():
        angles = sorted(rows)
        ratios = [rows[angle]['vis']['small_vehicle_ratio'] for angle in angles]
        counts = [rows[angle]['vis']['total_preds'] for angle in angles]
        color = PLOT_COLORS.get(model_name)
        ax_ratio.plot(
            angles, ratios, marker='o', markersize=4, linewidth=2,
            color=color, label=model_name)
        ax_count.plot(
            angles, counts, marker='o', markersize=4, linewidth=2,
            color=color, label=model_name)

    ax_ratio.set_title(
        'Small-vehicle ratio across 72 rotation angles')
    ax_ratio.set_ylabel('Small-vehicle ratio')
    ax_ratio.set_ylim(-0.02, 1.02)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(loc='best')

    ax_count.set_title('Visible prediction count across 72 rotation angles')
    ax_count.set_xlabel('Rotation angle (deg)')
    ax_count.set_ylabel('Visible predictions')
    ax_count.set_xticks(DEFAULT_EXPECTED_ANGLES[::3])
    ax_count.grid(True, alpha=0.3)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.dataset_root = str(Path(args.dataset_root))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_results_path = Path(args.base_results)
    finetune_dump = out_dir / 'finetune_dump.pkl'
    notext_dump = out_dir / 'notext_cls_dump.pkl'

    run_test_dump(
        model_name='finetune',
        checkpoint=Path(args.finetune_checkpoint),
        dump_path=finetune_dump,
        work_dir=out_dir / 'finetune_work_dir',
        log_path=out_dir / 'finetune_test.log',
        args=args)
    run_test_dump(
        model_name='notext_cls',
        checkpoint=Path(args.notext_checkpoint),
        dump_path=notext_dump,
        work_dir=out_dir / 'notext_cls_work_dir',
        log_path=out_dir / 'notext_cls_test.log',
        args=args)

    rows_by_model: OrderedDict[str, dict[int, dict[str, Any]]] = OrderedDict([
        ('base', build_base_rows(base_results_path)),
        ('finetune', build_dump_rows(finetune_dump, args.vis_score_thr)),
        ('notext_cls', build_dump_rows(notext_dump, args.vis_score_thr)),
    ])
    validate_angles(
        rows_by_model=rows_by_model, expected_angles=DEFAULT_EXPECTED_ANGLES)

    long_rows = build_long_rows(rows_by_model)
    wide_rows = build_wide_rows(rows_by_model)

    long_csv_path = out_dir / 'small_vehicle_rotation_stats_long.csv'
    wide_csv_path = out_dir / 'small_vehicle_rotation_stats_wide.csv'
    summary_md_path = out_dir / 'small_vehicle_rotation_summary.md'
    json_path = out_dir / 'small_vehicle_rotation_stats.json'
    plot_path = out_dir / 'small_vehicle_rotation_curve.png'
    run_cfg_path = out_dir / 'run_config.json'

    write_csv(long_csv_path, long_rows)
    write_csv(wide_csv_path, wide_rows)
    write_markdown_summary(
        summary_md_path,
        args=args,
        rows_by_model=rows_by_model,
        wide_rows=wide_rows)
    write_json(json_path, args=args, rows_by_model=rows_by_model)
    save_plot(plot_path, rows_by_model)

    with open(run_cfg_path, 'w', encoding='utf-8') as handle:
        json.dump(
            dict(
                dataset_root=args.dataset_root,
                img_dir=args.img_dir,
                ann_dir=args.ann_dir,
                base_results=str(base_results_path),
                config=str(args.config),
                finetune_checkpoint=str(args.finetune_checkpoint),
                notext_checkpoint=str(args.notext_checkpoint),
                vis_score_thr=args.vis_score_thr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                cuda_visible_devices=args.cuda_visible_devices,
            ),
            handle,
            indent=2,
            sort_keys=True)

    print(f'Saved long CSV: {long_csv_path}')
    print(f'Saved wide CSV: {wide_csv_path}')
    print(f'Saved summary: {summary_md_path}')
    print(f'Saved JSON: {json_path}')
    print(f'Saved plot: {plot_path}')


if __name__ == '__main__':
    main()
