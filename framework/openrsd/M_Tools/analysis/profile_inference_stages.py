#!/usr/bin/env python3
"""Profile detector inference stages on cached MMEngine test batches."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / 'tools'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from openrsd_env import preload_installed_mmengine  # noqa: E402

preload_installed_mmengine()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmdet.registry import RUNNERS  # noqa: E402
from mmdet.utils import setup_cache_size_limit_of_dynamo  # noqa: E402

try:  # noqa: E402
    from openrsd_config_switches import sync_openrsd_feature_switches
except Exception:  # pragma: no cover - profiling fallback
    sync_openrsd_feature_switches = None


StageAccumulator = dict[str, list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--work-dir')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=100)
    parser.add_argument('--warmup-iters', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--cudnn-benchmark', action='store_true')
    parser.add_argument(
        '--single-stage-without-nms',
        action='store_true',
        help='Profile single-stage detector postprocess with with_nms=False.')
    return parser.parse_args()


def set_dataloader_runtime(cfg: Config, batch_size: int,
                           num_workers: int) -> None:
    for key in ('test_dataloader', 'val_dataloader'):
        if key not in cfg:
            continue
        cfg[key].batch_size = batch_size
        cfg[key].num_workers = num_workers
        cfg[key].persistent_workers = bool(num_workers > 0)
        if 'sampler' in cfg[key]:
            cfg[key].sampler.shuffle = False


def build_runner(cfg: Config) -> Runner:
    if 'runner_type' not in cfg:
        return Runner.from_cfg(cfg)
    return RUNNERS.build(cfg)


def batch_size_of(data_batch: Any) -> int:
    if isinstance(data_batch, dict):
        samples = data_batch.get('data_samples')
        if samples is not None:
            return len(samples)
        inputs = data_batch.get('inputs')
        if hasattr(inputs, 'shape'):
            return int(inputs.shape[0])
        if isinstance(inputs, (list, tuple)):
            return len(inputs)
    return 0


def count_detections(outputs: Any) -> int:
    total = 0
    for sample in outputs or []:
        pred = getattr(sample, 'pred_instances', sample)
        scores = getattr(pred, 'scores', None)
        if scores is not None:
            total += int(scores.numel())
            continue
        bboxes = getattr(pred, 'bboxes', None)
        if bboxes is not None:
            total += int(len(bboxes))
    return total


def cuda_mb(value: int | float) -> float:
    return float(value) / 1024.0 / 1024.0


def sync_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def time_stage(device: torch.device, stages: StageAccumulator, name: str,
               fn: Callable[[], Any]) -> Any:
    sync_if_cuda(device)
    start = time.perf_counter()
    value = fn()
    sync_if_cuda(device)
    stages[name].append(time.perf_counter() - start)
    return value


def feature_shapes(feats: Any) -> list[list[int]]:
    if not isinstance(feats, (list, tuple)):
        return []
    shapes = []
    for feat in feats:
        if hasattr(feat, 'shape'):
            shapes.append([int(v) for v in feat.shape])
    return shapes


def run_single_stage_path(model: Any, data_batch: Any, device: torch.device,
                          stages: StageAccumulator,
                          with_nms: bool = True) -> tuple[Any, dict]:
    processed = time_stage(
        device, stages, 'preprocess',
        lambda: model.data_preprocessor(data_batch, False))
    batch_inputs = processed['inputs']
    batch_data_samples = processed['data_samples']

    feats = time_stage(device, stages, 'backbone',
                       lambda: model.backbone(batch_inputs))
    if getattr(model, 'with_neck', False):
        feats = time_stage(device, stages, 'neck', lambda: model.neck(feats))

    head_outs = time_stage(device, stages, 'dense_head_forward',
                           lambda: model.bbox_head.forward(feats))
    batch_img_metas = [sample.metainfo for sample in batch_data_samples]
    postprocess_stage = 'decode_filter_nms' if with_nms else (
        'decode_filter_no_nms')
    results_list = time_stage(
        device, stages, postprocess_stage,
        lambda: model.bbox_head.predict_by_feat(
            *head_outs,
            batch_img_metas=batch_img_metas,
            rescale=True,
            with_nms=with_nms))
    outputs = time_stage(
        device, stages, 'attach_results',
        lambda: model.add_pred_to_datasample(batch_data_samples, results_list))
    extra = {
        'path': 'single_stage_dense_head',
        'feature_shapes': feature_shapes(feats),
        'with_nms': with_nms,
    }
    return outputs, extra


def call_pre_decoder(model: Any, encoder_outputs: dict,
                     batch_data_samples: Any) -> tuple[dict, dict]:
    try:
        return model.pre_decoder(
            **encoder_outputs, batch_data_samples=batch_data_samples)
    except TypeError:
        return model.pre_decoder(**encoder_outputs)


def run_transformer_path(model: Any, data_batch: Any, device: torch.device,
                         stages: StageAccumulator) -> tuple[Any, dict]:
    processed = time_stage(
        device, stages, 'preprocess',
        lambda: model.data_preprocessor(data_batch, False))
    batch_inputs = processed['inputs']
    batch_data_samples = processed['data_samples']

    feats = time_stage(device, stages, 'backbone',
                       lambda: model.backbone(batch_inputs))
    if getattr(model, 'with_neck', False):
        feats = time_stage(device, stages, 'neck', lambda: model.neck(feats))

    encoder_inputs, decoder_inputs = time_stage(
        device, stages, 'pre_transformer_flatten_pos',
        lambda: model.pre_transformer(feats, batch_data_samples))
    encoder_outputs = time_stage(
        device, stages, 'transformer_encoder',
        lambda: model.forward_encoder(**encoder_inputs))
    tmp_dec_inputs, head_inputs = time_stage(
        device, stages, 'pre_decoder_topk_query_init',
        lambda: call_pre_decoder(model, encoder_outputs, batch_data_samples))
    decoder_inputs.update(tmp_dec_inputs)
    decoder_outputs = time_stage(
        device, stages, 'transformer_decoder',
        lambda: model.forward_decoder(**decoder_inputs))
    head_inputs.update(decoder_outputs)
    results_list = time_stage(
        device, stages, 'set_head_predict_no_nms',
        lambda: model.bbox_head.predict(
            **head_inputs,
            batch_data_samples=batch_data_samples,
            rescale=True))
    outputs = time_stage(
        device, stages, 'attach_results',
        lambda: model.add_pred_to_datasample(batch_data_samples, results_list))
    extra = {
        'path': 'detr_style_set_prediction',
        'feature_shapes': feature_shapes(feats),
        'num_queries': int(getattr(model, 'num_queries', 0)),
    }
    return outputs, extra


def run_profiled_batch(model: Any, data_batch: Any, device: torch.device,
                       stages: StageAccumulator,
                       single_stage_with_nms: bool) -> tuple[Any, dict]:
    if hasattr(model, 'pre_transformer') and hasattr(model, 'forward_decoder'):
        return run_transformer_path(model, data_batch, device, stages)
    return run_single_stage_path(
        model, data_batch, device, stages, with_nms=single_stage_with_nms)


def summarize_stages(stages: StageAccumulator, image_count: int) -> dict:
    total_stage_sec = sum(sum(values) for values in stages.values())
    stage_summary = {}
    for name, values in stages.items():
        total = sum(values)
        stage_summary[name] = {
            'iters': len(values),
            'total_sec': total,
            'mean_iter_sec': statistics.mean(values) if values else 0.0,
            'median_iter_sec': statistics.median(values) if values else 0.0,
            'p90_iter_sec': percentile(values, 0.90),
            'ms_per_image': (
                total / image_count * 1000.0 if image_count else 0.0),
            'share_of_profiled_total': (
                total / total_stage_sec if total_stage_sec > 0 else 0.0),
        }
    return {
        'total_stage_sec': total_stage_sec,
        'images': image_count,
        'ms_per_image': (
            total_stage_sec / image_count * 1000.0 if image_count else 0.0),
        'images_per_sec': (
            image_count / total_stage_sec if total_stage_sec > 0 else 0.0),
        'stages': stage_summary,
    }


def main() -> None:
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.load_from = args.checkpoint
    cfg.resume = False
    cfg.work_dir = args.work_dir or str(
        Path(args.out).resolve().parent / f'{args.label}_runner')
    set_dataloader_runtime(cfg, args.batch_size, args.num_workers)
    if sync_openrsd_feature_switches is not None:
        sync_openrsd_feature_switches(cfg)

    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested, but CUDA unavailable.')
        torch.cuda.set_device(device)

    runner = build_runner(cfg)
    runner.load_or_resume()
    model = runner.model
    model.eval()

    dataloader = runner.test_loop.dataloader
    cached_batches = []
    cached_images = 0
    for data_batch in dataloader:
        bs = batch_size_of(data_batch)
        cached_batches.append(data_batch)
        cached_images += bs
        if cached_images >= args.max_images:
            break
    if not cached_batches:
        raise RuntimeError('No profiling batches were produced.')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model_loaded_allocated_mb = cuda_mb(torch.cuda.memory_allocated(device))
        model_loaded_reserved_mb = cuda_mb(torch.cuda.memory_reserved(device))
    else:
        model_loaded_allocated_mb = None
        model_loaded_reserved_mb = None

    warmup_stages: StageAccumulator = defaultdict(list)
    warmup_outputs = None
    warmup_extra = {}
    with torch.inference_mode():
        for data_batch in cached_batches[:args.warmup_iters]:
            warmup_outputs, warmup_extra = run_profiled_batch(
                model, data_batch, device, warmup_stages,
                single_stage_with_nms=not args.single_stage_without_nms)
    sync_if_cuda(device)
    if device.type == 'cuda':
        post_warmup_allocated_mb = cuda_mb(torch.cuda.memory_allocated(device))
        post_warmup_reserved_mb = cuda_mb(torch.cuda.memory_reserved(device))
    else:
        post_warmup_allocated_mb = None
        post_warmup_reserved_mb = None

    stages: StageAccumulator = defaultdict(list)
    timed_images = 0
    timed_detections = 0
    last_extra = warmup_extra
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.repeat):
            for data_batch in cached_batches:
                outputs, last_extra = run_profiled_batch(
                    model, data_batch, device, stages,
                    single_stage_with_nms=not args.single_stage_without_nms)
                timed_images += batch_size_of(data_batch)
                timed_detections += count_detections(outputs)

    stage_profile = summarize_stages(stages, timed_images)
    result = {
        'label': args.label,
        'config': str(Path(args.config)),
        'checkpoint': str(Path(args.checkpoint)),
        'device': str(device),
        'cuda_device_name': (
            torch.cuda.get_device_name(device)
            if device.type == 'cuda' else None),
        'device_name': (
            torch.cuda.get_device_name(device)
            if device.type == 'cuda' else platform.processor()),
        'model_class': model.__class__.__name__,
        'bbox_head_class': model.bbox_head.__class__.__name__,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_images': args.max_images,
        'cached_batches': len(cached_batches),
        'cached_images': cached_images,
        'warmup_iters': args.warmup_iters,
        'repeat': args.repeat,
        'stage_profile': stage_profile,
        'detections_per_image': (
            timed_detections / timed_images if timed_images > 0 else 0.0),
        'warmup_detections_last_iter': count_detections(warmup_outputs),
        'extra': last_extra,
        'memory_mb': {
            'model_loaded_allocated': model_loaded_allocated_mb,
            'model_loaded_reserved': model_loaded_reserved_mb,
            'post_warmup_allocated': post_warmup_allocated_mb,
            'post_warmup_reserved': post_warmup_reserved_mb,
            'timed_peak_allocated': (
                cuda_mb(torch.cuda.max_memory_allocated(device))
                if device.type == 'cuda' else None),
            'timed_peak_reserved': (
                cuda_mb(torch.cuda.max_memory_reserved(device))
                if device.type == 'cuda' else None),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
