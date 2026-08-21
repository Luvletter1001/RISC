import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / 'M_configs' / 'Vis' /
    'A12_flex_rtm_v3_1_DOTA2only_ss_train_vis.py')
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / 'results' / 'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train' /
    'epoch_12.pth')
DEFAULT_VIS_ROOT = PROJECT_ROOT / 'vis'
DEFAULT_OUT_ROOT = PROJECT_ROOT / 'vis_infer_epoch12'
DEFAULT_PYTHON = Path('/data/zcy/anaconda3/envs/openrsd/bin/python')
DEFAULT_EXTRA_PYTHONPATH = Path('/tmp/openrsd_extra')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch inference and visualization for vis/*/dataset.')
    parser.add_argument(
        '--vis-root', default=str(DEFAULT_VIS_ROOT), help='Root vis folder.')
    parser.add_argument(
        '--out-root',
        default=str(DEFAULT_OUT_ROOT),
        help='Root output folder.')
    parser.add_argument(
        '--config',
        default=str(DEFAULT_CONFIG),
        help='Inference config for tools/test.py.')
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
        default='6,7',
        help='Comma separated GPU ids.')
    parser.add_argument(
        '--batch-size', type=int, default=8, help='Per-process batch size.')
    parser.add_argument(
        '--num-workers', type=int, default=4, help='Per-process num_workers.')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Visualization threshold.')
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=2,
        help='Maximum concurrent inference processes.')
    parser.add_argument(
        '--subset',
        default='',
        help='Comma separated folder names to run. Empty means all.')
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
        dataset_root = subdir / 'dataset'
        img_dir = dataset_root / 'images'
        ann_dir = dataset_root / 'annfiles'
        if img_dir.is_dir() and ann_dir.is_dir():
            datasets.append(subdir)
    return datasets


def build_env(args, gpu_id: int, dataset_root: Path, out_dir: Path):
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['MPLCONFIGDIR'] = str(PROJECT_ROOT / 'SimpleRun' / '.mplconfig')
    env['PYTHONPATH'] = str(DEFAULT_EXTRA_PYTHONPATH)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['OPENRSD_VIS_DATA_ROOT'] = str(dataset_root / 'dataset')
    env['OPENRSD_VIS_IMG_DIR'] = 'images'
    env['OPENRSD_VIS_ANN_DIR'] = 'annfiles'
    env['OPENRSD_VIS_BATCH_SIZE'] = str(args.batch_size)
    env['OPENRSD_VIS_NUM_WORKERS'] = str(args.num_workers)
    env['OPENRSD_VIS_RESULTS_PREFIX'] = str(out_dir / 'predictions')
    return env


def start_job(args, gpu_id: int, subdir: Path, out_root: Path):
    out_dir = out_root / subdir.name
    work_dir = out_dir / 'work_dir'
    log_pth = out_dir / 'inference.log'
    os.makedirs(out_dir, exist_ok=True)

    env = build_env(args, gpu_id, subdir, out_dir)
    cmd = [
        args.python,
        str(PROJECT_ROOT / 'tools' / 'test.py'),
        args.config,
        args.checkpoint,
        '--work-dir',
        str(work_dir),
        '--show-dir',
        'vis',
        '--cfg-options',
        f'default_hooks.visualization.score_thr={args.score_thr}',
    ]
    log_file = open(log_pth, 'w')
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT)
    return {
        'name': subdir.name,
        'gpu': gpu_id,
        'subdir': subdir,
        'out_dir': out_dir,
        'proc': proc,
        'log_file': log_file,
    }


def finalize_job(args, job):
    job['log_file'].close()
    if job['proc'].returncode != 0:
        raise RuntimeError(
            f'inference failed for {job["name"]}, see {job["out_dir"] / "inference.log"}')
    work_dir = job['out_dir'] / 'work_dir'
    timestamp_dirs = [p for p in work_dir.iterdir() if p.is_dir()]
    if not timestamp_dirs:
        raise RuntimeError(f'no timestamp dir found under {work_dir}')
    timestamp_dir = max(timestamp_dirs, key=lambda p: p.stat().st_mtime)
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

    subset_names = {x for x in args.subset.split(',') if x}
    datasets = discover_datasets(vis_root, subset_names)
    if not datasets:
        raise RuntimeError(f'No dataset folders found under {vis_root}')

    gpu_ids = [int(x) for x in args.gpus.split(',') if x]
    if not gpu_ids:
        raise RuntimeError('No GPU ids provided')
    max_parallel = min(args.max_parallel, len(gpu_ids))

    print(f'Found {len(datasets)} dataset folders')
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
            finalize_job(args, job)
            print(f'VIS DONE {job["name"]}')
        running = still_running

    cost = time.time() - start_time
    print(f'All done in {cost / 60:.1f} minutes')
    print(f'Outputs: {out_root}')


if __name__ == '__main__':
    main()
