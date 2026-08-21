#!/usr/bin/env python3
"""DOTA1 rotation-TTA follow-up analyses from saved angle-sweep predictions."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT_DIR / 'tools'
for path in (str(ROOT_DIR), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

import mmengine  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmengine import Config  # noqa: E402
from mmengine.evaluator import Evaluator  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

from mmdet.registry import DATASETS  # noqa: E402
from mmdet.utils import register_all_modules as register_mmdet  # noqa: E402
from mmrotate.evaluation import eval_rbbox_map  # noqa: E402
from mmrotate.utils import register_all_modules as register_mmrotate  # noqa: E402

register_mmdet(init_default_scope=False)
register_mmrotate(init_default_scope=False)


DATA_ROOT = ROOT_DIR / 'data/DOTA1_1024_500'
DEFAULT_OUT_ROOT = ROOT_DIR / 'work_dirs/dota1_exp_ab_20260507_003353'
DEFAULT_RESULTMD_DIR = ROOT_DIR / 'resultmd'
DEFAULT_PYTHON = '/data/zcy/anaconda3/envs/openrsd/bin/python'
ANGLES = ('000', '030', '060', '090', '120', '150', '180', '210', '240',
          '270', '300', '330')
FOCUS_CLASSES = {
    'bridge', 'harbor', 'ship', 'small-vehicle', 'large-vehicle', 'plane',
    'storage-tank', 'roundabout', 'helicopter'
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    config: Path
    checkpoint: Path
    infer_root: Path
    canonical_merged: Path
    smoke_batch: int


def build_model_specs(out_root: Path) -> dict[str, ModelSpec]:
    infer_root = out_root / 'exp_a_tta_infer'
    merge_root = out_root / 'exp_a_tta_merge'
    return {
        'rtmdet_l':
        ModelSpec(
            key='rtmdet_l',
            display='Rotated RTMDet-L 3xMS',
            config=ROOT_DIR /
            'M_configs/RotationStudy/rotated_rtmdet_l_dota1_ms_eval.py',
            checkpoint=ROOT_DIR /
            'weights/rotated_rtmdet_l-3x-dota_ms-2738da34.pth',
            infer_root=infer_root / 'rtmdet_l',
            canonical_merged=merge_root / 'rtmdet_l/merged_predictions.pkl',
            smoke_batch=24),
        'h2rbox_v2':
        ModelSpec(
            key='h2rbox_v2',
            display='H2RBox-v2 1xMS+RR',
            config=ROOT_DIR /
            'M_configs/RotationStudy/h2rbox_v2_r50_fpn_dota1_ms_rr_eval.py',
            checkpoint=ROOT_DIR /
            'weights/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth',
            infer_root=infer_root / 'h2rbox_v2',
            canonical_merged=merge_root / 'h2rbox_v2/merged_predictions.pkl',
            smoke_batch=32),
        'retinanet_msrr':
        ModelSpec(
            key='retinanet_msrr',
            display='Rotated RetinaNet R50 MS+RR',
            config=ROOT_DIR /
            'M_configs/RotationStudy/rotated_retinanet_r50_msrr_dota1_eval.py',
            checkpoint=ROOT_DIR /
            'weights/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth',
            infer_root=infer_root / 'retinanet_msrr',
            canonical_merged=merge_root /
            'retinanet_msrr/merged_predictions.pkl',
            smoke_batch=48),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        choices=('all', 'sanity', 'curve', 'classwise', 'summary'),
        default='all')
    parser.add_argument('--models', nargs='+',
                        default=['rtmdet_l', 'h2rbox_v2', 'retinanet_msrr'])
    parser.add_argument('--angles', nargs='+', default=list(ANGLES))
    parser.add_argument('--views', nargs='+', default=list(ANGLES))
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument('--target-merge-root', type=Path, default=None)
    parser.add_argument('--analysis-root', type=Path, default=None)
    parser.add_argument('--resultmd-dir', type=Path, default=DEFAULT_RESULTMD_DIR)
    parser.add_argument('--python-bin', default=DEFAULT_PYTHON)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--offline-num-workers', type=int, default=0)
    parser.add_argument('--classwise-nproc', type=int, default=4)
    parser.add_argument('--force-merge', action='store_true')
    parser.add_argument('--force-eval', action='store_true')
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime('%F %T')


def log(message: str) -> None:
    print(f'[{now()}] {message}', flush=True)


def normalize_img_id(img_id) -> str:
    if isinstance(img_id, (int, np.integer)):
        return str(int(img_id))
    return str(img_id)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value, 'cpu'):
        return value.cpu().numpy()
    return np.asarray(value)


def angle_int(angle: str) -> int:
    return int(angle)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return 'NA'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'NA'
    if math.isnan(number):
        return 'NA'
    return f'{number:.{digits}f}'


def empty_gt_instances() -> dict:
    return {
        'labels': torch.zeros((0, ), dtype=torch.long),
        'bboxes': torch.zeros((0, 5), dtype=torch.float32),
    }


def empty_ignored_instances() -> dict:
    return {
        'labels': torch.zeros((0, ), dtype=torch.long),
        'bboxes': torch.zeros((0, 5), dtype=torch.float32),
    }


def normalize_sample(sample: dict, pred_img_id=None) -> dict:
    sample = dict(sample)
    if not sample.get('gt_instances'):
        sample['gt_instances'] = empty_gt_instances()
    if not sample.get('ignored_instances'):
        sample['ignored_instances'] = empty_ignored_instances()
    if pred_img_id is not None and 'img_id' not in sample:
        sample['img_id'] = pred_img_id
    return sample


def md_path(path: Path | str) -> str:
    return f'`{path}`'


def metric_from_results(results: dict) -> dict:
    def get_value(*keys):
        for key in keys:
            if key in results:
                return float(results[key])
        return None

    return {
        'map': get_value('dota/mAP', 'mAP'),
        'ap50': get_value('dota/AP50', 'AP50'),
        'raw': {str(k): float(v) for k, v in results.items()}
    }


def parse_metrics_from_log(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors='ignore')
    map_match = re.findall(r'dota/mAP[:\'"]?\s*[:=]?\s*([0-9]*\.?[0-9]+)',
                           text)
    ap50_match = re.findall(r'dota/AP50[:\'"]?\s*[:=]?\s*([0-9]*\.?[0-9]+)',
                             text)
    if not map_match and not ap50_match:
        return None
    return {
        'map': float(map_match[-1]) if map_match else None,
        'ap50': float(ap50_match[-1]) if ap50_match else None,
    }


class EvalContext:
    def __init__(self, spec: ModelSpec, angle: str, offline_num_workers: int):
        self.spec = spec
        self.angle = angle
        self.cfg = Config.fromfile(str(spec.config))
        init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.cfg.merge_from_dict({
            'test_dataloader.dataset.data_root':
            str(DATA_ROOT),
            'test_dataloader.dataset.ann_file':
            f'angle_sweep_val/realistic/angle_{angle}/annfiles/',
            'test_dataloader.dataset.data_prefix.img_path':
            f'angle_sweep_val/realistic/angle_{angle}/images/',
            'test_dataloader.num_workers':
            offline_num_workers,
            'test_dataloader.persistent_workers':
            False,
        })
        self.dataset = DATASETS.build(self.cfg.test_dataloader.dataset)
        self.dataset_meta = self.dataset.metainfo
        self.classes = tuple(self.dataset_meta['classes'])
        self.gt_by_id = self._build_gt_by_id()
        evaluator_cfg = self.cfg.val_evaluator
        if isinstance(evaluator_cfg, (list, tuple)):
            evaluator_cfg = evaluator_cfg[0]
        self.evaluator_cfg = self.cfg.val_evaluator
        self.predict_box_type = evaluator_cfg.get('predict_box_type', 'rbox')
        self.use_07_metric = evaluator_cfg.get('eval_mode',
                                               '11points') == '11points'

    def _build_gt_by_id(self) -> dict[str, dict]:
        dataloader = Runner.build_dataloader(self.cfg.test_dataloader)
        gt_by_id = {}
        for batch in dataloader:
            for data_sample in batch['data_samples']:
                data = data_sample.to_dict()
                gt_by_id[normalize_img_id(data['img_id'])] = data
        return gt_by_id

    def merge_with_gt(self, predictions: Iterable[dict]) -> list[dict]:
        pred_map = {}
        for pred in predictions:
            pid = normalize_img_id(pred['img_id'])
            if pid in pred_map:
                raise ValueError(f'Duplicate img_id in predictions: {pid}')
            pred_map[pid] = pred

        missing_in_pkl = set(self.gt_by_id) - set(pred_map)
        if missing_in_pkl:
            sample = next(iter(missing_in_pkl))
            raise ValueError(
                f'{len(missing_in_pkl)} dataset image(s) missing from pkl; '
                f'example={sample}')

        merged = []
        for pid in sorted(pred_map):
            pred = pred_map[pid]
            if pid in self.gt_by_id:
                sample = normalize_sample(self.gt_by_id[pid], pred['img_id'])
            else:
                sample = {
                    'img_id': pred['img_id'],
                    'gt_instances': empty_gt_instances(),
                    'ignored_instances': empty_ignored_instances(),
                }
            sample['pred_instances'] = pred['pred_instances']
            merged.append(sample)
        return merged

    def evaluate_overall(self, pkl_path: Path, log_path: Path) -> dict:
        predictions = mmengine.load(str(pkl_path))
        samples = self.merge_with_gt(predictions)
        evaluator = Evaluator(self.evaluator_cfg)
        evaluator.dataset_meta = self.dataset_meta
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w', encoding='utf-8') as out:
            out.write(f'config={self.spec.config}\n')
            out.write(f'predictions={pkl_path}\n')
            out.write(f'target_angle={self.angle}\n')
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                        out):
                    results = evaluator.offline_evaluate(samples)
                out.write(f'{results}\n')
            except Exception:  # noqa: BLE001
                traceback.print_exc(file=out)
                raise
        metrics = metric_from_results(results)
        metrics['predictions'] = str(pkl_path)
        metrics['log'] = str(log_path)
        return metrics

    def evaluate_classwise(self, pkl_path: Path, log_path: Path,
                           nproc: int) -> dict:
        predictions = mmengine.load(str(pkl_path))
        samples = self.merge_with_gt(predictions)
        annotations = []
        det_results = []
        num_classes = len(self.classes)
        for sample in samples:
            gt_instances = sample['gt_instances']
            ignore_instances = sample['ignored_instances']
            if not gt_instances:
                ann = {
                    'labels': np.zeros((0, ), dtype=np.int64),
                    'bboxes': np.zeros((0, 5), dtype=np.float32),
                    'labels_ignore': np.zeros((0, ), dtype=np.int64),
                    'bboxes_ignore': np.zeros((0, 5), dtype=np.float32),
                }
            else:
                if not ignore_instances:
                    labels_ignore = np.zeros((0, ), dtype=np.int64)
                    bboxes_ignore = np.zeros((0, 5), dtype=np.float32)
                else:
                    labels_ignore = to_numpy(
                        ignore_instances['labels']).astype(np.int64)
                    bboxes_ignore = to_numpy(
                        ignore_instances['bboxes']).astype(np.float32)
                ann = {
                    'labels': to_numpy(gt_instances['labels']).astype(
                        np.int64),
                    'bboxes': to_numpy(gt_instances['bboxes']).astype(
                        np.float32),
                    'labels_ignore': labels_ignore,
                    'bboxes_ignore': bboxes_ignore,
                }
            annotations.append(ann)

            pred_instances = sample['pred_instances']
            boxes = to_numpy(pred_instances['bboxes']).astype(np.float32)
            labels = to_numpy(pred_instances['labels']).astype(np.int64)
            scores = to_numpy(pred_instances['scores']).astype(np.float32)
            per_class = []
            for label in range(num_classes):
                indices = np.where(labels == label)[0]
                per_class.append(
                    np.hstack(
                        [boxes[indices], scores[indices].reshape((-1, 1))]))
            det_results.append(per_class)

        mean_ap, cls_results = eval_rbbox_map(
            det_results,
            annotations,
            iou_thr=0.5,
            use_07_metric=self.use_07_metric,
            box_type=self.predict_box_type,
            dataset=self.classes,
            logger='silent',
            nproc=nproc)
        rows = []
        for class_name, cls_result in zip(self.classes, cls_results):
            ap = cls_result['ap']
            if isinstance(ap, np.ndarray):
                ap = float(np.asarray(ap).reshape(-1)[0])
            else:
                ap = float(ap)
            num_gts = cls_result['num_gts']
            if isinstance(num_gts, np.ndarray):
                num_gts = int(np.asarray(num_gts).reshape(-1)[0])
            rows.append({
                'class': class_name,
                'ap50': ap,
                'num_gts': num_gts,
                'num_dets': int(cls_result['num_dets']),
            })
        result = {
            'map': float(mean_ap),
            'classes': rows,
            'predictions': str(pkl_path),
            'log': str(log_path),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result


class ContextCache:
    def __init__(self, offline_num_workers: int):
        self.offline_num_workers = offline_num_workers
        self._cache: dict[tuple[str, str], EvalContext] = {}

    def get(self, spec: ModelSpec, angle: str) -> EvalContext:
        key = (spec.key, angle)
        if key not in self._cache:
            log(f'Build GT/evaluator context: {spec.key} angle_{angle}')
            self._cache[key] = EvalContext(spec, angle,
                                           self.offline_num_workers)
        return self._cache[key]


def select_specs(args: argparse.Namespace) -> list[ModelSpec]:
    all_specs = build_model_specs(args.out_root)
    specs = []
    for model in args.models:
        if model not in all_specs:
            raise KeyError(f'Unknown model: {model}')
        specs.append(all_specs[model])
    return specs


def require_inputs(specs: list[ModelSpec], angles: list[str],
                   views: list[str]) -> None:
    missing = []
    for spec in specs:
        for path in (spec.config, spec.checkpoint, spec.canonical_merged):
            if not path.exists():
                missing.append(str(path))
        for angle in sorted(set(angles) | set(views)):
            pred = spec.infer_root / f'angle_{angle}/predictions.pkl'
            if not pred.exists():
                missing.append(str(pred))
    for angle in angles:
        ann_dir = DATA_ROOT / f'angle_sweep_val/realistic/angle_{angle}/annfiles'
        img_dir = DATA_ROOT / f'angle_sweep_val/realistic/angle_{angle}/images'
        if not ann_dir.exists():
            missing.append(str(ann_dir))
        if not img_dir.exists():
            missing.append(str(img_dir))
    if missing:
        preview = '\n'.join(f'  - {item}' for item in missing[:20])
        raise FileNotFoundError(
            f'Missing {len(missing)} required input(s):\n{preview}')


def ensure_target_merge(spec: ModelSpec, target_angle: str,
                        views: list[str], args: argparse.Namespace) -> Path:
    out_dir = args.target_merge_root / spec.key / f'target_{target_angle}'
    out_path = out_dir / 'merged_predictions.pkl'
    log_path = out_dir / 'merge.log'
    if out_path.exists() and not args.force_merge:
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        str(ROOT_DIR / 'M_Tools/analysis/rotation_tta_merge.py'),
        '--out',
        str(out_path),
        '--target-angle',
        str(angle_int(target_angle)),
        '--img-shape',
        '1024',
        '1024',
        '--score-thr',
        '0.05',
        '--pre-nms-topk',
        '4000',
        '--nms-iou',
        '0.1',
        '--max-per-img',
        '2000',
    ]
    for view in views:
        cmd.extend([
            '--prediction',
            str(spec.infer_root / f'angle_{view}/predictions.pkl'),
            '--angle',
            str(angle_int(view)),
        ])

    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONPATH'] = f'{ROOT_DIR}:{TOOLS_DIR}'
    env['MPLCONFIGDIR'] = '/tmp/mplconfig'
    log(f'Merge {spec.key} target_{target_angle} from {len(views)} views')
    with log_path.open('w', encoding='utf-8') as log_file:
        log_file.write(' '.join(cmd) + '\n')
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f'Merge failed: {log_path}')
    return out_path


def eval_overall_cached(ctx: EvalContext, pkl: Path, log_path: Path,
                        force_eval: bool) -> dict:
    cache_path = log_path.with_suffix('.json')
    if cache_path.exists() and log_path.exists() and not force_eval:
        return json.loads(cache_path.read_text())
    metrics = ctx.evaluate_overall(pkl, log_path)
    cache_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def run_sanity(args: argparse.Namespace, specs: list[ModelSpec],
               cache: ContextCache) -> list[dict]:
    rows = []
    for spec in specs:
        ctx = cache.get(spec, '000')
        single_pkl = spec.infer_root / 'angle_000/predictions.pkl'
        single_log = args.analysis_root / 'sanity' / spec.key / 'single_angle000.log'
        tta_log = args.analysis_root / 'sanity' / spec.key / 'tta_merged_angle000.log'
        single = eval_overall_cached(ctx, single_pkl, single_log,
                                     args.force_eval)
        tta = eval_overall_cached(ctx, spec.canonical_merged, tta_log,
                                  args.force_eval)
        rows.append({
            'model': spec.key,
            'display': spec.display,
            'single_map': single['map'],
            'single_ap50': single['ap50'],
            'tta_map': tta['map'],
            'tta_ap50': tta['ap50'],
            'delta_ap50': tta['ap50'] - single['ap50'],
            'single_pkl': str(single_pkl),
            'tta_pkl': str(spec.canonical_merged),
            'single_log': single['log'],
            'tta_log': tta['log'],
        })
    out_json = args.analysis_root / 'sanity_results.json'
    out_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def run_curve(args: argparse.Namespace, specs: list[ModelSpec],
              cache: ContextCache) -> list[dict]:
    rows = []
    for spec in specs:
        for target in args.angles:
            ctx = cache.get(spec, target)
            single_pkl = spec.infer_root / f'angle_{target}/predictions.pkl'
            merged_pkl = ensure_target_merge(spec, target, args.views, args)
            base_dir = args.analysis_root / 'repair_curve' / spec.key / f'target_{target}'
            single = eval_overall_cached(ctx, single_pkl,
                                         base_dir / 'single.log',
                                         args.force_eval)
            tta = eval_overall_cached(ctx, merged_pkl, base_dir / 'tta.log',
                                      args.force_eval)
            rows.append({
                'model': spec.key,
                'display': spec.display,
                'angle': target,
                'single_map': single['map'],
                'single_ap50': single['ap50'],
                'tta_map': tta['map'],
                'tta_ap50': tta['ap50'],
                'delta_ap50': tta['ap50'] - single['ap50'],
                'single_pkl': str(single_pkl),
                'tta_pkl': str(merged_pkl),
                'single_log': single['log'],
                'tta_log': tta['log'],
            })
    out_json = args.analysis_root / 'repair_curve_results.json'
    out_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def classwise_by_class(result: dict) -> dict[str, dict]:
    return {row['class']: row for row in result['classes']}


def run_classwise(args: argparse.Namespace, specs: list[ModelSpec],
                  cache: ContextCache) -> list[dict]:
    rows = []
    for spec in specs:
        for target in args.angles:
            ctx = cache.get(spec, target)
            single_pkl = spec.infer_root / f'angle_{target}/predictions.pkl'
            merged_pkl = ensure_target_merge(spec, target, args.views, args)
            base_dir = args.analysis_root / 'classwise' / spec.key / f'target_{target}'
            single_json = base_dir / 'single.json'
            tta_json = base_dir / 'tta.json'
            if single_json.exists() and not args.force_eval:
                single = json.loads(single_json.read_text())
            else:
                single = ctx.evaluate_classwise(single_pkl, single_json,
                                                args.classwise_nproc)
            if tta_json.exists() and not args.force_eval:
                tta = json.loads(tta_json.read_text())
            else:
                tta = ctx.evaluate_classwise(merged_pkl, tta_json,
                                             args.classwise_nproc)
            single_by_class = classwise_by_class(single)
            tta_by_class = classwise_by_class(tta)
            for class_name in ctx.classes:
                s = single_by_class[class_name]
                t = tta_by_class[class_name]
                rows.append({
                    'model': spec.key,
                    'display': spec.display,
                    'angle': target,
                    'class': class_name,
                    'single_ap50': s['ap50'],
                    'tta_ap50': t['ap50'],
                    'delta_ap50': t['ap50'] - s['ap50'],
                    'num_gts': s['num_gts'],
                    'single_num_dets': s['num_dets'],
                    'tta_num_dets': t['num_dets'],
                })
    out_json = args.analysis_root / 'classwise_results.json'
    out_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def group_by(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else float('nan')


def write_sanity_md(args: argparse.Namespace, rows: list[dict]) -> Path:
    md_path_out = args.resultmd_dir / 'dota1_exp_a_rotation_tta_merge_eval_canonical_20260507.md'
    lines = [
        '# DOTA1 Experiment A: canonical Rotation TTA fairness sanity',
        '',
        f'- generated_at: `{now()}`',
        f'- out_root: `{args.out_root}`',
        '- target GT/images: `data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/`',
        '- evaluator: `tools/openrsd_eval_metric.py` logic via `Evaluator.offline_evaluate` with GT merged by `img_id`',
        '',
        '## Same-offline-evaluator AP50',
        '',
        '| model | single angle_000 AP50 | TTA merge AP50 | delta | single pkl | TTA pkl |',
        '|---|---:|---:|---:|---|---|',
    ]
    for row in rows:
        lines.append(
            f'| {row["display"]} | {fmt(row["single_ap50"])} | '
            f'{fmt(row["tta_ap50"])} | {fmt(row["delta_ap50"])} | '
            f'{md_path(row["single_pkl"])} | {md_path(row["tta_pkl"])} |')
    lines.extend([
        '',
        '## Notes',
        '',
        '- This replaces the previous comparison that mixed online `test.py` single-view metrics with offline TTA metrics.',
        '- Both columns above are evaluated through the same fixed offline path.',
        '- `dota/mAP` is also recorded in the per-run logs below.',
        '',
        '## Logs',
        '',
        '| model | single eval log | TTA eval log |',
        '|---|---|---|',
    ])
    for row in rows:
        lines.append(
            f'| {row["display"]} | {md_path(row["single_log"])} | '
            f'{md_path(row["tta_log"])} |')
    lines.append('')
    md_path_out.parent.mkdir(parents=True, exist_ok=True)
    md_path_out.write_text('\n'.join(lines), encoding='utf-8')
    return md_path_out


def summarize_curve(rows: list[dict]) -> list[dict]:
    summary = []
    for model, model_rows in group_by(rows, 'model').items():
        single_vals = [r['single_ap50'] for r in model_rows]
        tta_vals = [r['tta_ap50'] for r in model_rows]
        gains = [r['delta_ap50'] for r in model_rows]
        worst_single = min(model_rows, key=lambda r: r['single_ap50'])
        worst_tta = min(model_rows, key=lambda r: r['tta_ap50'])
        display = model_rows[0]['display']
        summary.append({
            'model': model,
            'display': display,
            'single_mean': mean(single_vals),
            'tta_mean': mean(tta_vals),
            'mean_gain': mean(gains),
            'single_range': max(single_vals) - min(single_vals),
            'tta_range': max(tta_vals) - min(tta_vals),
            'worst_single_angle': worst_single['angle'],
            'worst_single_ap50': worst_single['single_ap50'],
            'worst_tta_angle': worst_tta['angle'],
            'worst_tta_ap50': worst_tta['tta_ap50'],
        })
    return summary


def write_curve_md(args: argparse.Namespace, rows: list[dict]) -> Path:
    md_path_out = args.resultmd_dir / 'dota1_rotation_tta_full_angle_repair_curve_20260507.md'
    lines = [
        '# DOTA1 full-angle Rotation TTA repair curve',
        '',
        f'- generated_at: `{now()}`',
        f'- target angles: `{", ".join(args.angles)}`',
        f'- TTA views per target: `{", ".join(args.views)}`',
        f'- target merge root: `{args.target_merge_root}`',
        '',
        '## Summary',
        '',
        '| model | single mean AP50 | TTA mean AP50 | mean gain | single range | TTA range | single worst | TTA worst |',
        '|---|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in summarize_curve(rows):
        lines.append(
            f'| {row["display"]} | {fmt(row["single_mean"])} | '
            f'{fmt(row["tta_mean"])} | {fmt(row["mean_gain"])} | '
            f'{fmt(row["single_range"])} | {fmt(row["tta_range"])} | '
            f'angle_{row["worst_single_angle"]} ({fmt(row["worst_single_ap50"])}) | '
            f'angle_{row["worst_tta_angle"]} ({fmt(row["worst_tta_ap50"])}) |')
    lines.append('')
    lines.append('## Per-model curves')
    lines.append('')
    for model, model_rows in group_by(rows, 'model').items():
        model_rows = sorted(model_rows, key=lambda r: int(r['angle']))
        lines.append(f'### {model_rows[0]["display"]}')
        lines.append('')
        lines.append('| angle | single AP50 | TTA AP50 | TTA gain | single pkl | TTA pkl |')
        lines.append('|---:|---:|---:|---:|---|---|')
        for row in model_rows:
            lines.append(
                f'| {row["angle"]} | {fmt(row["single_ap50"])} | '
                f'{fmt(row["tta_ap50"])} | {fmt(row["delta_ap50"])} | '
                f'{md_path(row["single_pkl"])} | {md_path(row["tta_pkl"])} |')
        lines.append('')
    md_path_out.parent.mkdir(parents=True, exist_ok=True)
    md_path_out.write_text('\n'.join(lines), encoding='utf-8')
    return md_path_out


def aggregate_classwise(rows: list[dict]) -> list[dict]:
    aggregated = []
    for model, model_rows in group_by(rows, 'model').items():
        for class_name, class_rows in group_by(model_rows, 'class').items():
            aggregated.append({
                'model':
                model,
                'display':
                class_rows[0]['display'],
                'class':
                class_name,
                'focus':
                class_name in FOCUS_CLASSES,
                'single_mean_ap50':
                mean(r['single_ap50'] for r in class_rows),
                'tta_mean_ap50':
                mean(r['tta_ap50'] for r in class_rows),
                'delta_mean_ap50':
                mean(r['delta_ap50'] for r in class_rows),
                'max_gain_angle':
                max(class_rows, key=lambda r: r['delta_ap50'])['angle'],
                'max_damage_angle':
                min(class_rows, key=lambda r: r['delta_ap50'])['angle'],
            })
    return aggregated


def write_classwise_md(args: argparse.Namespace, rows: list[dict]) -> Path:
    md_path_out = args.resultmd_dir / 'dota1_rotation_tta_classwise_gain_20260507.md'
    aggregated = aggregate_classwise(rows)
    lines = [
        '# DOTA1 class-wise Rotation TTA gain',
        '',
        f'- generated_at: `{now()}`',
        '- AP50 values are averaged over the 12 target angles.',
        '- Focus classes: `bridge, harbor, ship, small-vehicle, large-vehicle, plane, storage-tank, roundabout, helicopter`.',
        '',
    ]
    for model, model_rows in group_by(aggregated, 'model').items():
        model_rows = sorted(model_rows, key=lambda r: r['delta_mean_ap50'],
                            reverse=True)
        lines.append(f'## {model_rows[0]["display"]}')
        lines.append('')
        lines.append('| class | focus | single mean AP50 | TTA mean AP50 | mean gain | max-gain angle | max-damage angle |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for row in model_rows:
            focus = 'yes' if row['focus'] else ''
            lines.append(
                f'| {row["class"]} | {focus} | '
                f'{fmt(row["single_mean_ap50"])} | {fmt(row["tta_mean_ap50"])} | '
                f'{fmt(row["delta_mean_ap50"])} | {row["max_gain_angle"]} | '
                f'{row["max_damage_angle"]} |')
        lines.append('')
        lines.append('Top gains: ' + ', '.join(
            f'{r["class"]} ({fmt(r["delta_mean_ap50"])})'
            for r in model_rows[:5]))
        lines.append('')
        lines.append('Top damages: ' + ', '.join(
            f'{r["class"]} ({fmt(r["delta_mean_ap50"])})'
            for r in sorted(model_rows, key=lambda r: r['delta_mean_ap50'])[:5]))
        lines.append('')
    out_json = args.analysis_root / 'classwise_aggregated.json'
    out_json.write_text(json.dumps(aggregated, indent=2), encoding='utf-8')
    md_path_out.parent.mkdir(parents=True, exist_ok=True)
    md_path_out.write_text('\n'.join(lines), encoding='utf-8')
    return md_path_out


def read_or_empty(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def smoke_rows(args: argparse.Namespace, specs: list[ModelSpec]) -> list[dict]:
    root = args.analysis_root / 'gpu_smoke'
    rows = []
    for spec in specs:
        smoke_dir = root / spec.key / 'angle_000'
        log_path = smoke_dir / 'test.log'
        pred_path = smoke_dir / 'predictions.pkl'
        metrics = parse_metrics_from_log(log_path)
        rows.append({
            'model': spec.key,
            'display': spec.display,
            'batch_size': spec.smoke_batch,
            'status': 'OK' if pred_path.exists() and metrics else 'NA',
            'map': metrics.get('map') if metrics else None,
            'ap50': metrics.get('ap50') if metrics else None,
            'predictions': str(pred_path),
            'log': str(log_path),
        })
    return rows


def write_master_md(args: argparse.Namespace, specs: list[ModelSpec],
                    sanity_rows: list[dict], curve_rows: list[dict],
                    classwise_rows: list[dict], doc_paths: dict[str,
                                                               Path]) -> Path:
    md_path_out = args.resultmd_dir / 'dota1_rotation_tta_followup_master_20260507.md'
    curve_summary = summarize_curve(curve_rows) if curve_rows else []
    aggregated = aggregate_classwise(classwise_rows) if classwise_rows else []
    lines = [
        '# DOTA1 Rotation TTA follow-up master summary',
        '',
        f'- generated_at: `{now()}`',
        f'- out_root: `{args.out_root}`',
        f'- analysis_root: `{args.analysis_root}`',
        f'- GPUs requested for smoke: `{args.gpus}`',
        '',
        '## Subtask docs',
        '',
        f'- Fairness sanity: {md_path(doc_paths.get("sanity", ""))}',
        f'- Full-angle repair curve: {md_path(doc_paths.get("curve", ""))}',
        f'- Class-wise gain: {md_path(doc_paths.get("classwise", ""))}',
        '',
        '## 4-GPU smoke status',
        '',
        '| model | per-GPU batch | status | AP50 | log |',
        '|---|---:|---|---:|---|',
    ]
    for row in smoke_rows(args, specs):
        lines.append(
            f'| {row["display"]} | {row["batch_size"]} | {row["status"]} | '
            f'{fmt(row["ap50"])} | {md_path(row["log"])} |')
    lines.extend([
        '',
        '## Fairness sanity',
        '',
        '| model | single angle_000 AP50 | TTA merge AP50 | delta |',
        '|---|---:|---:|---:|',
    ])
    for row in sanity_rows:
        lines.append(
            f'| {row["display"]} | {fmt(row["single_ap50"])} | '
            f'{fmt(row["tta_ap50"])} | {fmt(row["delta_ap50"])} |')
    lines.extend([
        '',
        '## Repair curve summary',
        '',
        '| model | single mean AP50 | TTA mean AP50 | mean gain | single range | TTA range |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in curve_summary:
        lines.append(
            f'| {row["display"]} | {fmt(row["single_mean"])} | '
            f'{fmt(row["tta_mean"])} | {fmt(row["mean_gain"])} | '
            f'{fmt(row["single_range"])} | {fmt(row["tta_range"])} |')
    lines.extend([
        '',
        '## Class-wise strongest effects',
        '',
        '| model | largest mean gain | largest mean damage |',
        '|---|---|---|',
    ])
    for model, model_rows in group_by(aggregated, 'model').items():
        gain = max(model_rows, key=lambda r: r['delta_mean_ap50'])
        damage = min(model_rows, key=lambda r: r['delta_mean_ap50'])
        lines.append(
            f'| {gain["display"]} | {gain["class"]} ({fmt(gain["delta_mean_ap50"])}) | '
            f'{damage["class"]} ({fmt(damage["delta_mean_ap50"])}) |')
    lines.append('')
    md_path_out.parent.mkdir(parents=True, exist_ok=True)
    md_path_out.write_text('\n'.join(lines), encoding='utf-8')
    return md_path_out


def main() -> None:
    args = parse_args()
    if args.target_merge_root is None:
        args.target_merge_root = args.out_root / 'exp_a_tta_target_merge'
    if args.analysis_root is None:
        args.analysis_root = args.out_root / 'exp_a_tta_followup_analysis'
    args.angles = [f'{int(a):03d}' for a in args.angles]
    args.views = [f'{int(v):03d}' for v in args.views]
    args.resultmd_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    args.target_merge_root.mkdir(parents=True, exist_ok=True)

    specs = select_specs(args)
    require_inputs(specs, args.angles, args.views)
    cache = ContextCache(args.offline_num_workers)
    doc_paths: dict[str, Path] = {}

    if args.task in ('all', 'sanity'):
        sanity_rows = run_sanity(args, specs, cache)
        doc_paths['sanity'] = write_sanity_md(args, sanity_rows)
    else:
        sanity_rows = read_or_empty(args.analysis_root / 'sanity_results.json')

    if args.task in ('all', 'curve'):
        curve_rows = run_curve(args, specs, cache)
        doc_paths['curve'] = write_curve_md(args, curve_rows)
    else:
        curve_rows = read_or_empty(args.analysis_root / 'repair_curve_results.json')

    if args.task in ('all', 'classwise'):
        classwise_rows = run_classwise(args, specs, cache)
        doc_paths['classwise'] = write_classwise_md(args, classwise_rows)
    else:
        classwise_rows = read_or_empty(args.analysis_root / 'classwise_results.json')

    if args.task in ('all', 'summary'):
        if 'sanity' not in doc_paths:
            candidate = args.resultmd_dir / 'dota1_exp_a_rotation_tta_merge_eval_canonical_20260507.md'
            if candidate.exists():
                doc_paths['sanity'] = candidate
        if 'curve' not in doc_paths:
            candidate = args.resultmd_dir / 'dota1_rotation_tta_full_angle_repair_curve_20260507.md'
            if candidate.exists():
                doc_paths['curve'] = candidate
        if 'classwise' not in doc_paths:
            candidate = args.resultmd_dir / 'dota1_rotation_tta_classwise_gain_20260507.md'
            if candidate.exists():
                doc_paths['classwise'] = candidate
        master = write_master_md(args, specs, sanity_rows, curve_rows,
                                 classwise_rows, doc_paths)
        doc_paths['master'] = master

    log('Written docs:')
    for name, path in doc_paths.items():
        log(f'  {name}: {path}')


if __name__ == '__main__':
    main()
