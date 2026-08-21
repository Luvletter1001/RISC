#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / 'M_configs' / 'Vis' / 'redet_re50_refpn_1x_dota_vis.py')
DEFAULT_CHECKPOINT = Path(
    '/data1/zcy/OpenRSD/weights/redet_re50_fpn_1x_dota_le90-724ab2da.pth')
DEFAULT_VIS_ROOT = Path('/data1/zcy/OpenRSD/vis')
DEFAULT_OUT_ROOT = Path(
    '/data1/zcy/OpenRSD/workdir_vis/redet_re50_fpn_1x_dota_le90')
DEFAULT_PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
DEFAULT_EXTRA_PYTHONPATH = Path('/tmp/openrsd_extra')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch visualization inference for vis/*/dataset using ReDet.')
    parser.add_argument(
        '--vis-root',
        default=str(DEFAULT_VIS_ROOT),
        help='Root folder that contains vis/<subdir>/vis and vis/<subdir>/dataset.')
    parser.add_argument(
        '--out-root',
        default=str(DEFAULT_OUT_ROOT),
        help='Root output folder.')
    parser.add_argument(
        '--config',
        default=str(DEFAULT_CONFIG),
        help='Visualization config for ReDet.')
    parser.add_argument(
        '--checkpoint',
        default=str(DEFAULT_CHECKPOINT),
        help='Checkpoint path.')
    parser.add_argument(
        '--python',
        default=str(DEFAULT_PYTHON),
        help='Python executable.')
    parser.add_argument(
        '--gpus',
        default='0,1,6,7',
        help='Comma separated GPU ids.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Per-process batch size.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Per-process num_workers.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Visualization score threshold.')
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=4,
        help='Maximum concurrent inference processes.')
    parser.add_argument(
        '--subset',
        default='',
        help='Comma separated subfolder names to run. Empty means all.')
    return parser.parse_args()


def ensure_extra_pythonpath(extra_dir: Path):
    extra_dir.mkdir(parents=True, exist_ok=True)
    link = extra_dir / 'M_AD'
    target = PROJECT_ROOT / 'M_AD'
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    os.symlink(target, link)


def discover_datasets(vis_root: Path, subset_names):
    datasets = []
    for subdir in sorted(vis_root.iterdir()):
        if not subdir.is_dir():
            continue
        if subset_names and subdir.name not in subset_names:
            continue
        legacy_vis_dir = subdir / 'vis'
        dataset_root = subdir / 'dataset'
        img_dir = dataset_root / 'images'
        ann_dir = dataset_root / 'annfiles'
        if legacy_vis_dir.is_dir() and img_dir.is_dir() and ann_dir.is_dir():
            datasets.append(subdir)
    return datasets


def build_env(args, gpu_id: int, dataset_root: Path):
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['MPLCONFIGDIR'] = str(PROJECT_ROOT / 'SimpleRun' / '.mplconfig')
    env['PYTHONPATH'] = str(DEFAULT_EXTRA_PYTHONPATH)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['OPENRSD_VIS_DATA_ROOT'] = str(dataset_root)
    env['OPENRSD_VIS_IMG_DIR'] = 'images'
    env['OPENRSD_VIS_ANN_DIR'] = 'annfiles'
    env['OPENRSD_VIS_BATCH_SIZE'] = str(args.batch_size)
    env['OPENRSD_VIS_NUM_WORKERS'] = str(args.num_workers)
    env.setdefault('LOCAL_RANK', '0')
    return env


def start_job(args, gpu_id: int, subdir: Path, out_root: Path):
    out_dir = out_root / subdir.name
    work_dir = out_dir / 'work_dir'
    log_path = out_dir / 'inference.log'
    os.makedirs(out_dir, exist_ok=True)

    env = build_env(args, gpu_id, subdir / 'dataset')
    cmd = [
        args.python,
        str(PROJECT_ROOT / 'tools' / 'openrsd_test.py'),
        args.config,
        args.checkpoint,
        '--work-dir',
        str(work_dir),
        '--show-dir',
        'vis',
        '--cfg-options',
        f'default_hooks.visualization.score_thr={args.score_thr}',
        # The official checkpoint bundled here was trained with MMRotate 0.x.
        # Its bbox coder angle stds differ from the imported MMRotate 1.x
        # config; keep decoding aligned with checkpoint meta.
        'model.roi_head.bbox_head.0.bbox_coder.target_stds=[0.1,0.1,0.2,0.2,1.0]',
        'model.roi_head.bbox_head.1.bbox_coder.target_stds=[0.05,0.05,0.1,0.1,0.5]',
    ]
    log_file = open(log_path, 'w')
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT)
    return {
        'name': subdir.name,
        'gpu': gpu_id,
        'out_dir': out_dir,
        'proc': proc,
        'log_file': log_file,
    }


def finalize_job(job):
    job['log_file'].close()
    if job['proc'].returncode != 0:
        raise RuntimeError(
            f'inference failed for {job["name"]}, '
            f'see {job["out_dir"] / "inference.log"}')

    work_dir = job['out_dir'] / 'work_dir'
    timestamp_dirs = [path for path in work_dir.iterdir() if path.is_dir()]
    if not timestamp_dirs:
        raise RuntimeError(f'no timestamp dir found under {work_dir}')

    timestamp_dir = max(timestamp_dirs, key=lambda path: path.stat().st_mtime)
    src_vis_dir = timestamp_dir / 'vis'
    dst_vis_dir = job['out_dir'] / 'vis'
    if not src_vis_dir.is_dir():
        raise RuntimeError(f'visualization dir not found: {src_vis_dir}')

    if dst_vis_dir.exists():
        shutil.rmtree(dst_vis_dir)
    shutil.copytree(src_vis_dir, dst_vis_dir)


def main():
    args = parse_args()
    ensure_extra_pythonpath(DEFAULT_EXTRA_PYTHONPATH)

    vis_root = Path(args.vis_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not vis_root.is_dir():
        raise FileNotFoundError(f'vis root not found: {vis_root}')
    if not Path(args.config).is_file():
        raise FileNotFoundError(f'config not found: {args.config}')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not Path(args.python).is_file():
        raise FileNotFoundError(f'python not found: {args.python}')

    subset_names = {name for name in args.subset.split(',') if name}
    datasets = discover_datasets(vis_root, subset_names)
    if not datasets:
        raise RuntimeError(f'No matching dataset folders found under {vis_root}')

    gpu_ids = [int(value) for value in args.gpus.split(',') if value]
    if not gpu_ids:
        raise RuntimeError('No GPU ids provided')
    max_parallel = min(args.max_parallel, len(gpu_ids))
    if max_parallel <= 0:
        raise RuntimeError('--max-parallel must be > 0')

    print(f'Found {len(datasets)} dataset folders')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output root: {out_root}')
    print(f'GPUs: {gpu_ids}, max_parallel: {max_parallel}')

    pending = list(datasets)
    running = []
    start_time = time.time()

    while pending or running:
        while pending and len(running) < max_parallel:
            used_gpus = {job['gpu'] for job in running}
            free_gpus = [gpu for gpu in gpu_ids if gpu not in used_gpus]
            if not free_gpus:
                break
            subdir = pending.pop(0)
            gpu_id = free_gpus[0]
            job = start_job(args, gpu_id, subdir, out_root)
            running.append(job)
            print(f'START {subdir.name} on GPU {gpu_id}')

        time.sleep(2)
        still_running = []
        for job in running:
            ret = job['proc'].poll()
            if ret is None:
                still_running.append(job)
                continue
            print(f'INFER DONE {job["name"]} on GPU {job["gpu"]}')
            finalize_job(job)
            print(f'VIS DONE {job["name"]}')
        running = still_running

    cost = time.time() - start_time
    print(f'All done in {cost / 60:.1f} minutes')
    print(f'Outputs: {out_root}')


if __name__ == '__main__':
    main()
