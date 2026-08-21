#!/usr/bin/env python3
"""Evaluate one checkpoint on val and export worst per-image AP50 cases.

AP50 is a dataset-level metric. For image ranking, this script defines
"image AP50" as:

1. Compute AP50 independently for each class on one image.
2. Average the AP50 values over active classes in that image.
3. Active classes are classes that have GTs or predictions in that image.

This keeps false-positive-only classes penalized, while classes absent from
both GT and prediction are ignored for that image.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'SimpleRun' / '.mplconfig'))

from tools.openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

import numpy as np
import torch
from mmdet.evaluation.functional import average_precision
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import (DATA_SAMPLERS, DATASETS, FUNCTIONS,
                               init_default_scope)
from mmengine.runner.checkpoint import load_checkpoint
from mmrotate.evaluation.functional.mean_ap import tpfp_default
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


DEFAULT_CONFIG = Path(
    '/data1/zcy/OpenRSD/results/'
    'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train/'
    'A12_flex_rtm_v3_1_DOTA2only_ss_train.py')
DEFAULT_CHECKPOINT = Path(
    '/data1/zcy/OpenRSD/results/'
    'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train/epoch_12.pth')
DEFAULT_OUT_TXT = DEFAULT_CHECKPOINT.parent / 'worst_val_ap50_top200_epoch12.txt'
CSV_FIELDS = [
    'file_name', 'image_ap50', 'num_gts', 'num_dets', 'num_active_classes',
    'img_path'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate val split and export worst per-image AP50 names.')
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help='Model config path.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help='Checkpoint path to evaluate.')
    parser.add_argument(
        '--device',
        default='cpu',
        help='Torch device used for inference, e.g. cuda:0 or cpu.')
    parser.add_argument(
        '--gpus',
        default='4,5,8,9',
        help='Comma separated GPU ids for parallel validation. Empty means '
        'run in single-process mode using --device.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Per-process batch size. Multi-GPU mode uses this value per card.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Override val dataloader num_workers.')
    parser.add_argument(
        '--topk',
        type=int,
        default=200,
        help='How many worst images to export.')
    parser.add_argument(
        '--min-num-gts',
        type=int,
        default=20,
        help='Only rank images with num_gts strictly greater than this.')
    parser.add_argument(
        '--max-patches-per-source',
        type=int,
        default=3,
        help='Maximum kept patches from the same source image.')
    parser.add_argument(
        '--out-txt',
        type=Path,
        default=DEFAULT_OUT_TXT,
        help='Output txt path. One file name per line, worst first.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=None,
        help='Optional override for model test score threshold.')
    parser.add_argument(
        '--merge-interval',
        type=float,
        default=5.0,
        help='Seconds between live merges of partial shard results.')
    parser.add_argument(
        '--worker-rank',
        type=int,
        default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--worker-count',
        type=int,
        default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--worker-gpu',
        default='',
        help=argparse.SUPPRESS)
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def resolve_runtime_path(path_value: str | Path, *, config_path: Path) -> Path:
    raw_path = Path(path_value)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            Path.cwd() / raw_path,
            PROJECT_ROOT / raw_path,
            config_path.parent / raw_path,
        ])

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered_candidates.append(resolved)
        if candidate.name == 'Step5_3_Prepare_Visual_Text_DINOv2_support.pkl':
            alt = Path(str(candidate).replace('/train/', '/ss_train/', 1)).resolve()
            if alt not in seen:
                seen.add(alt)
                ordered_candidates.append(alt)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    if raw_path.name:
        matches = sorted((PROJECT_ROOT / 'data').rglob(raw_path.name))
        if len(matches) == 1:
            return matches[0].resolve()

    return ordered_candidates[0] if ordered_candidates else raw_path


def prepare_model_cfg(cfg: Config, *, config_path: Path) -> None:
    for key in ('normalized_class_dict', 'neg_support_data', 'pca_meta_pth'):
        path_value = cfg.model.get(key)
        if path_value:
            cfg.model[key] = str(
                resolve_runtime_path(path_value, config_path=config_path))

    support_feat_dict = cfg.model.get('support_feat_dict')
    val_dataset_flag = cfg.model.get('val_dataset_flag')
    if support_feat_dict and val_dataset_flag in support_feat_dict:
        cfg.model.support_feat_dict = {
            val_dataset_flag: str(
                resolve_runtime_path(
                    support_feat_dict[val_dataset_flag],
                    config_path=config_path))
        }

    backbone = cfg.model.get('backbone')
    if backbone and 'init_cfg' in backbone:
        backbone['init_cfg'] = None


def build_dataloader(dataloader_cfg: dict,
                     *,
                     seed: int,
                     subset_indices: list[int] | None = None
                     ) -> tuple[Any, Any, DataLoader]:
    dataset_cfg = deepcopy(dataloader_cfg['dataset'])
    dataset = DATASETS.build(dataset_cfg)
    dataloader_dataset = Subset(dataset, subset_indices) \
        if subset_indices is not None else dataset

    sampler_cfg = deepcopy(dataloader_cfg.get(
        'sampler', dict(type='DefaultSampler', shuffle=False)))
    sampler = DATA_SAMPLERS.build(
        sampler_cfg,
        default_args=dict(dataset=dataloader_dataset, seed=seed))

    collate_fn_cfg = deepcopy(
        dataloader_cfg.get('collate_fn', dict(type='pseudo_collate')))
    collate_type = collate_fn_cfg.pop('type')
    collate_impl = FUNCTIONS.get(collate_type)
    collate_fn = lambda batch: collate_impl(batch, **collate_fn_cfg)

    num_workers = int(dataloader_cfg.get('num_workers', 0))
    persistent_workers = bool(
        dataloader_cfg.get('persistent_workers', False) and num_workers > 0)

    dataloader = DataLoader(
        dataloader_dataset,
        batch_size=int(dataloader_cfg.get('batch_size', 1)),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        drop_last=bool(dataloader_cfg.get('drop_last', False)))
    return dataset, dataloader_dataset, dataloader


def build_model(cfg: Config,
                *,
                config_path: Path,
                checkpoint_path: Path,
                device: str,
                score_thr: float | None):
    prepare_model_cfg(cfg, config_path=config_path)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))

    if score_thr is not None:
        cfg.model.setdefault('test_cfg', dict())
        cfg.model.test_cfg['score_thr'] = score_thr

    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    load_checkpoint(model, str(checkpoint_path), map_location='cpu')
    model.to(device)
    model.eval()
    return model


def resolve_eval_meta(cfg: Config) -> tuple[str, bool, float]:
    evaluator_cfg = cfg.get('val_evaluator', cfg.get('test_evaluator', {}))
    if isinstance(evaluator_cfg, list):
        evaluator_cfg = evaluator_cfg[0] if evaluator_cfg else {}
    box_type = evaluator_cfg.get('predict_box_type', 'rbox')
    use_07_metric = evaluator_cfg.get('eval_mode', '11points') == '11points'
    iou_thrs = evaluator_cfg.get('iou_thrs', 0.5)
    if isinstance(iou_thrs, list):
        iou_thr = float(iou_thrs[0])
    else:
        iou_thr = float(iou_thrs)
    return box_type, use_07_metric, iou_thr


def to_box_array(boxes: Any, *, box_type: str) -> np.ndarray:
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    array = to_numpy(boxes).astype(np.float32)
    dim = 5 if box_type == 'rbox' else 8
    if array.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    return array.reshape(-1, dim)


def to_label_array(labels: Any) -> np.ndarray:
    array = to_numpy(labels).astype(np.int64)
    if array.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return array.reshape(-1)


def empty_boxes(*, box_type: str) -> np.ndarray:
    dim = 5 if box_type == 'rbox' else 8
    return np.zeros((0, dim), dtype=np.float32)


def compute_class_ap50(*,
                       det_boxes: np.ndarray,
                       det_scores: np.ndarray,
                       gt_boxes: np.ndarray,
                       gt_boxes_ignore: np.ndarray,
                       iou_thr: float,
                       box_type: str,
                       use_07_metric: bool) -> float | None:
    num_dets = det_boxes.shape[0]
    num_gts = gt_boxes.shape[0]

    if num_gts == 0 and num_dets == 0:
        return None
    if num_dets == 0:
        return 0.0

    det_with_scores = np.concatenate(
        [det_boxes.astype(np.float32), det_scores[:, None].astype(np.float32)],
        axis=1)
    tp, fp = tpfp_default(
        det_with_scores,
        gt_boxes.astype(np.float32),
        gt_boxes_ignore.astype(np.float32),
        iou_thr=iou_thr,
        box_type=box_type,
        area_ranges=None)

    if num_gts == 0:
        return 0.0 if float(fp.sum()) > 0 else None

    sort_inds = np.argsort(-det_with_scores[:, -1])
    tp = np.cumsum(tp[0, sort_inds])
    fp = np.cumsum(fp[0, sort_inds])
    eps = np.finfo(np.float32).eps
    recalls = tp / max(num_gts, eps)
    precisions = tp / np.maximum(tp + fp, eps)
    mode = '11points' if use_07_metric else 'area'
    ap = average_precision(recalls, precisions, mode)
    return float(ap)


def compute_image_ap50(*,
                       pred_boxes: np.ndarray,
                       pred_scores: np.ndarray,
                       pred_labels: np.ndarray,
                       gt_boxes: np.ndarray,
                       gt_labels: np.ndarray,
                       gt_boxes_ignore: np.ndarray,
                       gt_labels_ignore: np.ndarray,
                       num_classes: int,
                       iou_thr: float,
                       box_type: str,
                       use_07_metric: bool) -> tuple[float, list[dict[str, Any]]]:
    class_rows: list[dict[str, Any]] = []
    class_aps: list[float] = []

    for class_id in range(num_classes):
        det_mask = pred_labels == class_id
        gt_mask = gt_labels == class_id
        gt_ignore_mask = gt_labels_ignore == class_id

        cls_det_boxes = pred_boxes[det_mask]
        cls_det_scores = pred_scores[det_mask]
        cls_gt_boxes = gt_boxes[gt_mask]
        cls_gt_boxes_ignore = gt_boxes_ignore[gt_ignore_mask]

        ap = compute_class_ap50(
            det_boxes=cls_det_boxes,
            det_scores=cls_det_scores,
            gt_boxes=cls_gt_boxes,
            gt_boxes_ignore=cls_gt_boxes_ignore,
            iou_thr=iou_thr,
            box_type=box_type,
            use_07_metric=use_07_metric)
        if ap is None:
            continue

        class_aps.append(ap)
        class_rows.append(
            dict(
                class_id=class_id,
                ap50=ap,
                num_gts=int(cls_gt_boxes.shape[0]),
                num_dets=int(cls_det_boxes.shape[0]),
            ))

    image_ap50 = float(np.mean(class_aps)) if class_aps else 1.0
    return image_ap50, class_rows


def extract_gt(sample: Any, *, box_type: str):
    gt_instances = sample.gt_instances
    gt_boxes = to_box_array(gt_instances.bboxes, box_type=box_type)
    gt_labels = to_label_array(gt_instances.labels)

    ignored_instances = getattr(sample, 'ignored_instances', None)
    if ignored_instances is None:
        gt_boxes_ignore = empty_boxes(box_type=box_type)
        gt_labels_ignore = np.zeros((0,), dtype=np.int64)
    else:
        gt_boxes_ignore = to_box_array(
            getattr(ignored_instances, 'bboxes', empty_boxes(box_type=box_type)),
            box_type=box_type)
        gt_labels_ignore = to_label_array(
            getattr(ignored_instances, 'labels', np.zeros((0,), dtype=np.int64)))

    return gt_boxes, gt_labels, gt_boxes_ignore, gt_labels_ignore


def extract_pred(sample: Any, *, box_type: str):
    pred_instances = sample.pred_instances
    pred_boxes = to_box_array(pred_instances.bboxes, box_type=box_type)
    pred_labels = to_label_array(pred_instances.labels)
    pred_scores = to_numpy(pred_instances.scores).astype(np.float32).reshape(-1)
    return pred_boxes, pred_scores, pred_labels


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return '--:--:--'
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def evaluate_images(model,
                    dataloader: DataLoader,
                    *,
                    num_classes: int,
                    box_type: str,
                    iou_thr: float,
                    use_07_metric: bool,
                    desc: str = 'Evaluating val AP50',
                    on_row: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_images = None
    dataset_obj = getattr(dataloader, 'dataset', None)
    if dataset_obj is not None:
        try:
            total_images = len(dataset_obj)
        except TypeError:
            total_images = None

    progress = tqdm(
        total=total_images,
        desc=desc,
        unit='img',
        dynamic_ncols=True)
    try:
        for data_batch in dataloader:
            input_samples = data_batch['data_samples']
            with torch.no_grad():
                outputs = model.test_step(data_batch)

            batch_count = 0
            for in_sample, out_sample in zip(input_samples, outputs):
                gt_boxes, gt_labels, gt_boxes_ignore, gt_labels_ignore = extract_gt(
                    in_sample, box_type=box_type)
                pred_boxes, pred_scores, pred_labels = extract_pred(
                    out_sample, box_type=box_type)

                image_ap50, class_rows = compute_image_ap50(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    gt_boxes_ignore=gt_boxes_ignore,
                    gt_labels_ignore=gt_labels_ignore,
                    num_classes=num_classes,
                    iou_thr=iou_thr,
                    box_type=box_type,
                    use_07_metric=use_07_metric)

                rows.append(
                    dict(
                        file_name=Path(in_sample.img_path).name,
                        img_path=in_sample.img_path,
                        image_ap50=image_ap50,
                        num_gts=int(gt_boxes.shape[0]),
                        num_dets=int(pred_boxes.shape[0]),
                        num_active_classes=len(class_rows),
                        class_rows=class_rows,
                    ))
                if on_row is not None:
                    on_row(rows[-1], len(rows))
                batch_count += 1
            progress.update(batch_count)
    finally:
        progress.close()
    return rows


def write_text_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        f.writelines(lines)
    os.replace(tmp_path, path)


def write_csv_atomic(path: Path,
                     header: list[str],
                     rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    with tmp_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp_path, path)


def get_source_image_id(file_name: str) -> str:
    return file_name.split('__', 1)[0]


def filter_ranked_rows(sorted_rows: list[dict[str, Any]], *,
                       min_num_gts: int,
                       max_patches_per_source: int) -> list[dict[str, Any]]:
    kept_per_source = Counter()
    filtered_rows: list[dict[str, Any]] = []
    for row in sorted_rows:
        if row['num_gts'] <= min_num_gts:
            continue
        source_image_id = get_source_image_id(row['file_name'])
        if kept_per_source[source_image_id] >= max_patches_per_source:
            continue
        kept_per_source[source_image_id] += 1
        filtered_rows.append(row)
    return filtered_rows


def save_outputs(rows: list[dict[str, Any]], *, out_txt: Path, topk: int,
                 min_num_gts: int,
                 max_patches_per_source: int) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda item: (item['image_ap50'], item['file_name']))
    filtered_rows = filter_ranked_rows(
        sorted_rows,
        min_num_gts=min_num_gts,
        max_patches_per_source=max_patches_per_source)
    worst_rows = filtered_rows[:topk]

    write_text_atomic(
        out_txt,
        [f"{row['file_name']}\n" for row in worst_rows])

    csv_path = out_txt.with_suffix('.csv')
    write_csv_atomic(
        csv_path,
        [
            'rank', 'file_name', 'image_ap50', 'num_gts', 'num_dets',
            'num_active_classes', 'img_path'
        ],
        [[
            rank,
            row['file_name'],
            f"{row['image_ap50']:.6f}",
            row['num_gts'],
            row['num_dets'],
            row['num_active_classes'],
            row['img_path'],
        ] for rank, row in enumerate(worst_rows, start=1)])

    all_csv_path = out_txt.with_name(out_txt.stem + '_all.csv')
    write_csv_atomic(
        all_csv_path,
        [
            'rank', 'file_name', 'image_ap50', 'num_gts', 'num_dets',
            'num_active_classes', 'img_path'
        ],
        [[
            rank,
            row['file_name'],
            f"{row['image_ap50']:.6f}",
            row['num_gts'],
            row['num_dets'],
            row['num_active_classes'],
            row['img_path'],
        ] for rank, row in enumerate(sorted_rows, start=1)])

    filtered_all_csv_path = out_txt.with_name(out_txt.stem + '_filtered_all.csv')
    write_csv_atomic(
        filtered_all_csv_path,
        [
            'rank', 'file_name', 'image_ap50', 'num_gts', 'num_dets',
            'num_active_classes', 'img_path'
        ],
        [[
            rank,
            row['file_name'],
            f"{row['image_ap50']:.6f}",
            row['num_gts'],
            row['num_dets'],
            row['num_active_classes'],
            row['img_path'],
        ] for rank, row in enumerate(filtered_rows, start=1)])


def parse_gpu_ids(text: str) -> list[int]:
    return [int(chunk.strip()) for chunk in text.split(',') if chunk.strip()]


def get_live_dir(out_txt: Path) -> Path:
    return out_txt.parent / f'{out_txt.stem}_live'


def get_shard_csv_path(live_dir: Path, rank: int, gpu_id: str) -> Path:
    return live_dir / f'shard_r{rank:02d}_gpu{gpu_id}.csv'


def get_worker_log_path(live_dir: Path, rank: int, gpu_id: int) -> Path:
    return live_dir / f'worker_r{rank:02d}_gpu{gpu_id}.log'


def get_done_path(live_dir: Path, rank: int, gpu_id: str) -> Path:
    return live_dir / f'shard_r{rank:02d}_gpu{gpu_id}.done'


def init_partial_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def append_partial_row(csv_path: Path, row: dict[str, Any]) -> None:
    record = {field: row[field] for field in CSV_FIELDS}
    with csv_path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(record)
        f.flush()
        os.fsync(f.fileno())


def load_partial_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                file_name = str(raw.get('file_name', '')).strip()
                if not file_name:
                    continue
                rows.append(
                    dict(
                        file_name=file_name,
                        image_ap50=float(raw['image_ap50']),
                        num_gts=int(raw['num_gts']),
                        num_dets=int(raw['num_dets']),
                        num_active_classes=int(raw['num_active_classes']),
                        img_path=str(raw['img_path']).strip(),
                    ))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def merge_live_outputs(*, live_dir: Path, out_txt: Path, topk: int,
                       min_num_gts: int,
                       max_patches_per_source: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(live_dir.glob('shard_r*_gpu*.csv')):
        rows.extend(load_partial_rows(csv_path))
    save_outputs(
        rows,
        out_txt=out_txt,
        topk=topk,
        min_num_gts=min_num_gts,
        max_patches_per_source=max_patches_per_source)
    return rows


def shard_indices(total: int, rank: int, count: int) -> list[int]:
    return list(range(rank, total, count))


def _raise_stop_signal(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f'signal {signum}')


def get_total_val_images(config_path: Path) -> int:
    os.chdir(PROJECT_ROOT)
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(str(config_path))
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    dataset_cfg = deepcopy(cfg.val_dataloader['dataset'])
    dataset = DATASETS.build(dataset_cfg)
    return len(dataset)


def run_worker(args: argparse.Namespace) -> int:
    assert args.worker_rank is not None
    assert args.worker_count is not None

    os.chdir(PROJECT_ROOT)
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(str(args.config))
    cfg.launcher = 'none'
    cfg.load_from = str(args.checkpoint)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))

    dataloader_cfg = deepcopy(cfg.val_dataloader)
    dataloader_cfg['batch_size'] = args.batch_size
    if args.num_workers is not None:
        dataloader_cfg['num_workers'] = args.num_workers
        dataloader_cfg['persistent_workers'] = args.num_workers > 0

    dataset_cfg = deepcopy(dataloader_cfg['dataset'])
    dataset = DATASETS.build(dataset_cfg)
    subset_indices = shard_indices(len(dataset), args.worker_rank, args.worker_count)

    base_dataset, _, dataloader = build_dataloader(
        dataloader_cfg,
        seed=int(cfg.get('seed', 2024)),
        subset_indices=subset_indices)

    box_type, use_07_metric, iou_thr = resolve_eval_meta(cfg)
    model = build_model(
        cfg,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_thr=args.score_thr)

    live_dir = get_live_dir(args.out_txt)
    gpu_tag = args.worker_gpu or str(args.worker_rank)
    shard_csv = get_shard_csv_path(live_dir, args.worker_rank, gpu_tag)
    done_path = get_done_path(live_dir, args.worker_rank, gpu_tag)
    init_partial_csv(shard_csv)
    if done_path.exists():
        done_path.unlink()

    evaluate_images(
        model,
        dataloader,
        num_classes=len(base_dataset.metainfo['classes']),
        box_type=box_type,
        iou_thr=iou_thr,
        use_07_metric=use_07_metric,
        desc=f'GPU {gpu_tag} rank {args.worker_rank}',
        on_row=lambda row, _: append_partial_row(shard_csv, row))

    done_path.write_text('done\n', encoding='utf-8')
    print(
        f'Worker rank={args.worker_rank} gpu={gpu_tag} '
        f'processed={len(subset_indices)} shard_csv={shard_csv}')
    return 0


def terminate_worker_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return


def force_kill_worker_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_multi_gpu(args: argparse.Namespace, gpu_ids: list[int]) -> int:
    live_dir = get_live_dir(args.out_txt)
    if live_dir.exists():
        shutil.rmtree(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)

    total_images = get_total_val_images(args.config)
    merge_live_outputs(
        live_dir=live_dir,
        out_txt=args.out_txt,
        topk=args.topk,
        min_num_gts=args.min_num_gts,
        max_patches_per_source=args.max_patches_per_source)

    workers: list[dict[str, Any]] = []
    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _raise_stop_signal)

    progress = tqdm(
        total=total_images,
        desc='Multi-GPU val AP50',
        unit='img',
        dynamic_ncols=True)
    start_time = time.time()
    script_path = Path(__file__).resolve()
    for rank, gpu_id in enumerate(gpu_ids):
        log_path = get_worker_log_path(live_dir, rank, gpu_id)
        log_file = log_path.open('w', encoding='utf-8', buffering=1)
        cmd = [
            sys.executable,
            str(script_path),
            '--config',
            str(args.config),
            '--checkpoint',
            str(args.checkpoint),
            '--device',
            'cuda:0',
            '--gpus',
            '',
            '--batch-size',
            str(args.batch_size),
            '--topk',
            str(args.topk),
            '--min-num-gts',
            str(args.min_num_gts),
            '--max-patches-per-source',
            str(args.max_patches_per_source),
            '--out-txt',
            str(args.out_txt),
            '--merge-interval',
            str(args.merge_interval),
            '--worker-rank',
            str(rank),
            '--worker-count',
            str(len(gpu_ids)),
            '--worker-gpu',
            str(gpu_id),
        ]
        if args.num_workers is not None:
            cmd.extend(['--num-workers', str(args.num_workers)])
        if args.score_thr is not None:
            cmd.extend(['--score-thr', str(args.score_thr)])

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env.setdefault('PYTHONNOUSERSITE', '1')
        env.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'SimpleRun' / '.mplconfig'))
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True)
        workers.append(
            dict(
                rank=rank,
                gpu_id=gpu_id,
                proc=proc,
                log_file=log_file,
                log_path=log_path,
            ))
        print(f'START worker rank={rank} gpu={gpu_id} log={log_path}')

    try:
        while True:
            time.sleep(max(0.5, min(args.merge_interval, 5.0)))
            merged_rows = merge_live_outputs(
                live_dir=live_dir,
                out_txt=args.out_txt,
                topk=args.topk,
                min_num_gts=args.min_num_gts,
                max_patches_per_source=args.max_patches_per_source)
            active = 0
            for worker in workers:
                ret = worker['proc'].poll()
                if ret is None:
                    active += 1
                    continue
                if worker['log_file'] is not None:
                    worker['log_file'].close()
                    worker['log_file'] = None
                if ret != 0:
                    merge_live_outputs(
                        live_dir=live_dir,
                        out_txt=args.out_txt,
                        topk=args.topk,
                        min_num_gts=args.min_num_gts,
                        max_patches_per_source=args.max_patches_per_source)
                    raise RuntimeError(
                        f'Worker rank={worker["rank"]} gpu={worker["gpu_id"]} '
                        f'failed with code {ret}. See {worker["log_path"]}')
            processed = min(len(merged_rows), total_images)
            progress.n = processed
            elapsed = max(time.time() - start_time, 1e-6)
            speed = processed / elapsed
            remaining = max(total_images - processed, 0)
            eta = remaining / speed if speed > 0 and processed > 0 else None
            progress.set_postfix_str(
                f'active={active} speed={speed:.2f} img/s eta={format_seconds(eta)}')
            progress.refresh()
            print(
                f'LIVE MERGE rows={len(merged_rows)} active_workers={active} '
                f'out={args.out_txt}')
            if active == 0:
                break
    except KeyboardInterrupt:
        print('Interrupted. Merging current partial results before stopping workers...')
        merged_rows = merge_live_outputs(
            live_dir=live_dir,
            out_txt=args.out_txt,
            topk=args.topk,
            min_num_gts=args.min_num_gts,
            max_patches_per_source=args.max_patches_per_source)
        progress.n = min(len(merged_rows), total_images)
        elapsed = max(time.time() - start_time, 1e-6)
        speed = progress.n / elapsed
        remaining = max(total_images - progress.n, 0)
        eta = remaining / speed if speed > 0 and progress.n > 0 else None
        progress.set_postfix_str(
            f'active=stopping speed={speed:.2f} img/s eta={format_seconds(eta)}')
        progress.refresh()
        for worker in workers:
            terminate_worker_group(worker['proc'])
        deadline = time.time() + 10
        while time.time() < deadline and any(
                worker['proc'].poll() is None for worker in workers):
            time.sleep(0.5)
        for worker in workers:
            force_kill_worker_group(worker['proc'])
        merged_rows = merge_live_outputs(
            live_dir=live_dir,
            out_txt=args.out_txt,
            topk=args.topk,
            min_num_gts=args.min_num_gts,
            max_patches_per_source=args.max_patches_per_source)
        progress.n = min(len(merged_rows), total_images)
        progress.refresh()
        return 130
    finally:
        progress.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for worker in workers:
            if worker['log_file'] is not None:
                worker['log_file'].close()

    merged_rows = merge_live_outputs(
        live_dir=live_dir,
        out_txt=args.out_txt,
        topk=args.topk,
        min_num_gts=args.min_num_gts,
        max_patches_per_source=args.max_patches_per_source)
    print(f'Val images merged: {len(merged_rows)}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output txt: {args.out_txt}')
    print(f'Output csv: {args.out_txt.with_suffix(".csv")}')
    print(f'Full ranking csv: {args.out_txt.with_name(args.out_txt.stem + "_all.csv")}')
    print(
        f'Filtered ranking csv: '
        f'{args.out_txt.with_name(args.out_txt.stem + "_filtered_all.csv")}')
    print(
        'Ranking filters: '
        f'num_gts>{args.min_num_gts}, '
        f'max_patches_per_source={args.max_patches_per_source}')
    print(f'Live shard dir: {live_dir}')
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(f'config not found: {args.config}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be > 0')
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')
    if args.topk <= 0:
        raise ValueError('--topk must be > 0')
    if args.min_num_gts < 0:
        raise ValueError('--min-num-gts must be >= 0')
    if args.max_patches_per_source <= 0:
        raise ValueError('--max-patches-per-source must be > 0')
    if args.merge_interval <= 0:
        raise ValueError('--merge-interval must be > 0')
    if args.worker_rank is not None:
        if args.worker_count is None or args.worker_count <= 0:
            raise ValueError('worker mode requires --worker-count > 0')
    else:
        _ = parse_gpu_ids(args.gpus) if args.gpus.strip() else []


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.worker_rank is not None:
        raise SystemExit(run_worker(args))

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus.strip() else []
    if gpu_ids:
        raise SystemExit(run_multi_gpu(args, gpu_ids))

    os.chdir(PROJECT_ROOT)
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(str(args.config))
    cfg.launcher = 'none'
    cfg.load_from = str(args.checkpoint)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))

    dataloader_cfg = deepcopy(cfg.val_dataloader)
    if args.batch_size is not None:
        dataloader_cfg['batch_size'] = args.batch_size
    if args.num_workers is not None:
        dataloader_cfg['num_workers'] = args.num_workers
        dataloader_cfg['persistent_workers'] = args.num_workers > 0

    dataset, _, dataloader = build_dataloader(
        dataloader_cfg,
        seed=int(cfg.get('seed', 2024)))

    class_names = list(dataset.metainfo['classes'])
    box_type, use_07_metric, iou_thr = resolve_eval_meta(cfg)
    model = build_model(
        cfg,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_thr=args.score_thr)

    rows = evaluate_images(
        model,
        dataloader,
        num_classes=len(class_names),
        box_type=box_type,
        iou_thr=iou_thr,
        use_07_metric=use_07_metric,
        desc=f'Evaluating val AP50 on {args.device}')
    save_outputs(
        rows,
        out_txt=args.out_txt,
        topk=args.topk,
        min_num_gts=args.min_num_gts,
        max_patches_per_source=args.max_patches_per_source)

    print(f'Val images evaluated: {len(rows)}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output txt: {args.out_txt}')
    print(f'Output csv: {args.out_txt.with_suffix(".csv")}')
    print(f'Full ranking csv: {args.out_txt.with_name(args.out_txt.stem + "_all.csv")}')
    print(
        f'Filtered ranking csv: '
        f'{args.out_txt.with_name(args.out_txt.stem + "_filtered_all.csv")}')
    print(
        'Ranking filters: '
        f'num_gts>{args.min_num_gts}, '
        f'max_patches_per_source={args.max_patches_per_source}')


if __name__ == '__main__':
    main()
