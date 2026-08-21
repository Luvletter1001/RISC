#!/usr/bin/env python3
"""Benchmark MMEngine detector inference speed and CUDA memory.

The timed region starts after a data batch has been fetched, so the primary
numbers measure model preprocessing, forward, prediction decode, and
post-processing/NMS through ``model.test_step``. A secondary end-to-end loop
number includes dataloader iteration overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


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
except Exception:  # pragma: no cover - benchmark fallback
    sync_openrsd_feature_switches = None


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
    parser.add_argument('--skip-e2e-loop', action='store_true')
    return parser.parse_args()


def set_dataloader_runtime(cfg: Config, batch_size: int, num_workers: int) -> None:
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
        pred = getattr(sample, 'pred_instances', None)
        if pred is None:
            continue
        scores = getattr(pred, 'scores', None)
        if scores is not None:
            total += int(scores.numel())
            continue
        bboxes = getattr(pred, 'bboxes', None)
        if bboxes is not None:
            total += int(len(bboxes))
    return total


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def cuda_mb(value: int | float) -> float:
    return float(value) / 1024.0 / 1024.0


def sync_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def summarize_times(iter_times: list[float], image_count: int) -> dict[str, Any]:
    total_time = sum(iter_times)
    return {
        'iters': len(iter_times),
        'images': image_count,
        'total_time_sec': total_time,
        'mean_iter_sec': statistics.mean(iter_times) if iter_times else 0.0,
        'median_iter_sec': statistics.median(iter_times) if iter_times else 0.0,
        'p90_iter_sec': percentile(iter_times, 0.90),
        'images_per_sec': image_count / total_time if total_time > 0 else 0.0,
        'ms_per_image': (total_time / image_count * 1000.0)
        if image_count > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.load_from = args.checkpoint
    cfg.resume = False
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = str(Path(args.out).resolve().parent / f'{args.label}_runner')
    set_dataloader_runtime(cfg, args.batch_size, args.num_workers)
    if sync_openrsd_feature_switches is not None:
        sync_openrsd_feature_switches(cfg)

    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested, but CUDA is unavailable.')
        torch.cuda.set_device(device)

    runner = build_runner(cfg)
    runner.load_or_resume()
    model = runner.model
    model.eval()

    # Build/cache deterministic data batches before timing model execution.
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
        raise RuntimeError('No benchmark batches were produced.')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model_loaded_allocated_mb = cuda_mb(torch.cuda.memory_allocated(device))
        model_loaded_reserved_mb = cuda_mb(torch.cuda.memory_reserved(device))
    else:
        model_loaded_allocated_mb = None
        model_loaded_reserved_mb = None

    warmup_outputs = []
    with torch.inference_mode():
        for data_batch in cached_batches[:args.warmup_iters]:
            warmup_outputs = model.test_step(data_batch)
    sync_if_cuda(device)
    if device.type == 'cuda':
        post_warmup_allocated_mb = cuda_mb(torch.cuda.memory_allocated(device))
        post_warmup_reserved_mb = cuda_mb(torch.cuda.memory_reserved(device))
    else:
        post_warmup_allocated_mb = None
        post_warmup_reserved_mb = None

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    iter_times: list[float] = []
    timed_images = 0
    timed_detections = 0
    with torch.inference_mode():
        for _ in range(args.repeat):
            for data_batch in cached_batches:
                bs = batch_size_of(data_batch)
                start = time.perf_counter()
                outputs = model.test_step(data_batch)
                sync_if_cuda(device)
                elapsed = time.perf_counter() - start
                iter_times.append(elapsed)
                timed_images += bs
                timed_detections += count_detections(outputs)

    model_only = summarize_times(iter_times, timed_images)
    if device.type == 'cuda':
        peak_allocated_mb = cuda_mb(torch.cuda.max_memory_allocated(device))
        peak_reserved_mb = cuda_mb(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated_mb = None
        peak_reserved_mb = None

    e2e_loop = None
    if not args.skip_e2e_loop:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        loop_times: list[float] = []
        loop_images = 0
        data_iter = iter(runner.test_loop.dataloader)
        with torch.inference_mode():
            while loop_images < args.max_images:
                start = time.perf_counter()
                try:
                    data_batch = next(data_iter)
                except StopIteration:
                    break
                bs = batch_size_of(data_batch)
                _ = model.test_step(data_batch)
                sync_if_cuda(device)
                loop_times.append(time.perf_counter() - start)
                loop_images += bs
        e2e_loop = summarize_times(loop_times, loop_images)

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
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_images': args.max_images,
        'cached_batches': len(cached_batches),
        'cached_images': cached_images,
        'warmup_iters': args.warmup_iters,
        'repeat': args.repeat,
        'model_only': model_only,
        'end_to_end_loop': e2e_loop,
        'detections_per_image': (
            timed_detections / timed_images if timed_images > 0 else 0.0),
        'warmup_detections_last_iter': count_detections(warmup_outputs),
        'memory_mb': {
            'model_loaded_allocated': model_loaded_allocated_mb,
            'model_loaded_reserved': model_loaded_reserved_mb,
            'post_warmup_allocated': post_warmup_allocated_mb,
            'post_warmup_reserved': post_warmup_reserved_mb,
            'timed_peak_allocated': peak_allocated_mb,
            'timed_peak_reserved': peak_reserved_mb,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
