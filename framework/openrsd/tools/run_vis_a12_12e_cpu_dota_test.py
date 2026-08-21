import argparse
import copy
import importlib
import os
import pickle
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = Path('/data/zcy/dataset/test_ss/images')
DEFAULT_VIS_CONFIG = (
    PROJECT_ROOT / 'M_configs' / 'Vis' /
    'A12_flex_rtm_v3_1_DOTA2only_ss_train_vis.py')
DEFAULT_CONFIG = Path(
    '/data1/zcy/OpenRSD/results/'
    'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train/'
    'A12_flex_rtm_v3_1_DOTA2only_ss_train.py')
DEFAULT_CHECKPOINT = Path(
    '/data1/zcy/OpenRSD/results/'
    'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train/epoch_12.pth')
DEFAULT_OUTPUT_ROOT = Path('/data1/zcy/OpenRSD/workdir_vis/A12_12e_dota_test_cpu')
DEFAULT_BASE_ANN = Path(
    '/data1/zcy/OpenRSD/SimpleRun/test_dataset/annfiles/'
    'P1860__1024__932___714.pkl')
IMAGE_SUFFIXES = {'.png'}


def _path_matches(path_item: str, target: str) -> bool:
    if path_item == '':
        path_item = os.getcwd()
    return os.path.abspath(path_item) == target


# Force MMEngine to pick CPU before importing it.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault(
    'MPLCONFIGDIR', str(PROJECT_ROOT / 'SimpleRun' / '.mplconfig'))
os.environ.setdefault('LOCAL_RANK', '0')
os.chdir(PROJECT_ROOT)

# Prefer the conda-env mmengine package. The repository also contains a local
# mmengine checkout that shadows the installed package but is incomplete.
sys.path = [
    p for p in sys.path
    if not p.startswith('/home/zcy/.local/')
    if not _path_matches(p, str(PROJECT_ROOT))
]

from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmdet.utils import register_all_modules as register_all_modules_mmdet  # noqa: E402
from mmrotate.utils import register_all_modules  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize DOTA test_ss images with A12 epoch_12 on CPU.')
    parser.add_argument(
        '--image-dir',
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help='Input image directory.')
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help='Finetuned config path.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help='Finetuned checkpoint path.')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help='All outputs will be written under this directory.')
    parser.add_argument(
        '--base-ann',
        type=Path,
        default=DEFAULT_BASE_ANN,
        help='Template annfile copied to each image stem.')
    parser.add_argument(
        '--ann-dir',
        type=Path,
        default=None,
        help='Optional annfiles directory. Defaults to <output-root>/annfiles.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Visualization score threshold.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='CPU test batch size.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='CPU dataloader workers.')
    parser.add_argument(
        '--force-rebuild-annfiles',
        action='store_true',
        help='Rewrite annfiles even if they already exist.')
    return parser.parse_args()


def load_pickle(file_path: Path):
    with file_path.open('rb') as f:
        return pickle.load(f)


def save_pickle(obj, file_path: Path):
    with file_path.open('wb') as f:
        pickle.dump(obj, f)


def list_images(image_dir: Path):
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def ensure_annfiles(image_dir: Path, ann_dir: Path, base_ann: Path,
                    force_rebuild: bool):
    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f'No images found under {image_dir}')

    ann_dir.mkdir(parents=True, exist_ok=True)
    template = load_pickle(base_ann)
    created = 0
    reused = 0
    for image_path in images:
        ann_path = ann_dir / f'{image_path.stem}.pkl'
        if ann_path.exists() and not force_rebuild:
            reused += 1
            continue
        save_pickle(template, ann_path)
        created += 1
    return images, created, reused


def ensure_custom_import(cfg: Config, import_path: str):
    custom_imports = cfg.get(
        'custom_imports',
        dict(
            allow_failed_imports=False,
            imports=[],
        ))
    imports = list(custom_imports.get('imports', []))
    if import_path not in imports:
        imports.append(import_path)
    custom_imports['allow_failed_imports'] = False
    custom_imports['imports'] = imports
    cfg.custom_imports = custom_imports


def disable_pretrained_init(cfg: Config):
    backbone = cfg.model.get('backbone')
    if backbone and 'init_cfg' in backbone:
        backbone['init_cfg'] = None


def load_reference_vis_metainfo():
    vis_cfg = Config.fromfile(str(DEFAULT_VIS_CONFIG))
    reference = vis_cfg.get('vis_metainfo')
    if reference is None:
        reference = vis_cfg.get('metainfo')
    return copy.deepcopy(reference or {})


def normalize_metainfo(metainfo, reference_metainfo=None):
    metainfo = copy.deepcopy(metainfo or {})
    reference_metainfo = copy.deepcopy(reference_metainfo or {})
    classes = list(metainfo.get('classes', []))
    if not classes:
        if reference_metainfo:
            return reference_metainfo
        return metainfo

    ref_classes = list(reference_metainfo.get('classes', []))
    ref_palette = list(reference_metainfo.get('palette', []))
    if ref_classes == classes and len(ref_palette) >= len(classes):
        metainfo['palette'] = ref_palette[:len(classes)]
        return metainfo

    palette = list(metainfo.get('palette', []))
    if len(palette) >= len(classes):
        metainfo['palette'] = palette[:len(classes)]
        return metainfo

    metainfo['palette'] = [
        ((37 * i) % 256, (97 * i) % 256, (157 * i) % 256)
        for i in range(len(classes))
    ]
    return metainfo


def build_test_pipeline(img_scale):
    return [
        dict(
            type='mmdet.LoadImageFromFile',
            file_client_args=dict(backend='disk')),
        dict(type='LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
        dict(
            type='ConvertBoxTypeSafe',
            box_type_mapping=dict(gt_bboxes='rbox')),
        dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
        dict(
            type='mmdet.Pad',
            size=img_scale,
            pad_val=dict(img=(114, 114, 114))),
        dict(
            type='PackDetInputsMM',
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            )),
    ]


def build_runtime_cfg(args, ann_dir: Path):
    cfg = Config.fromfile(str(args.config))
    cfg.launcher = 'none'
    cfg.load_from = str(args.checkpoint)
    cfg.work_dir = str(args.output_root / 'work_dir')

    ensure_custom_import(cfg, 'M_AD.engine.hooks.pred_only_visualization_hook')
    importlib.import_module('M_AD.engine.hooks.pred_only_visualization_hook')
    disable_pretrained_init(cfg)

    default_hooks = copy.deepcopy(cfg.get('default_hooks', {}))
    default_hooks['visualization'] = dict(
        type='PredOnlyDetVisualizationHook',
        draw=True,
        show=False,
        wait_time=0,
        test_out_dir='vis',
        score_thr=args.score_thr)
    cfg.default_hooks = default_hooks

    img_scale = (1024, 1024)
    metainfo = copy.deepcopy(cfg.get('metainfo'))
    if metainfo is None:
        metainfo = copy.deepcopy(cfg.test_dataloader.dataset.metainfo)
    reference_metainfo = load_reference_vis_metainfo()
    metainfo = normalize_metainfo(metainfo, reference_metainfo)

    test_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTADatasetOnline',
            data_root=str(args.image_dir.parent),
            ann_file=str(ann_dir),
            data_prefix=dict(img_path=args.image_dir.name),
            img_shape=img_scale,
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False),
            test_mode=True,
            pipeline=build_test_pipeline(img_scale)))

    cfg.test_dataloader = test_dataloader
    cfg.val_dataloader = copy.deepcopy(test_dataloader)
    cfg.test_evaluator = []
    cfg.val_evaluator = []
    cfg.resume = False
    return cfg


def collect_vis_dir(output_root: Path):
    work_dir = output_root / 'work_dir'
    if not work_dir.is_dir():
        raise RuntimeError(f'work_dir not found: {work_dir}')

    timestamp_dirs = [path for path in work_dir.iterdir() if path.is_dir()]
    if not timestamp_dirs:
        raise RuntimeError(f'No timestamp directories found under {work_dir}')

    latest_dir = max(timestamp_dirs, key=lambda path: path.stat().st_mtime)
    src_vis_dir = latest_dir / 'vis'
    if not src_vis_dir.is_dir():
        raise RuntimeError(f'Visualization directory not found: {src_vis_dir}')

    dst_vis_dir = output_root / 'vis'
    if dst_vis_dir.exists():
        shutil.rmtree(dst_vis_dir)
    shutil.copytree(src_vis_dir, dst_vis_dir)
    return src_vis_dir, dst_vis_dir


def validate_args(args):
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f'image dir not found: {args.image_dir}')
    if not args.config.is_file():
        raise FileNotFoundError(f'config not found: {args.config}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not args.base_ann.is_file():
        raise FileNotFoundError(f'base ann not found: {args.base_ann}')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be > 0')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')


def main():
    args = parse_args()
    validate_args(args)

    args.output_root.mkdir(parents=True, exist_ok=True)
    ann_dir = args.ann_dir if args.ann_dir is not None else args.output_root / 'annfiles'

    images, created, reused = ensure_annfiles(
        image_dir=args.image_dir,
        ann_dir=ann_dir,
        base_ann=args.base_ann,
        force_rebuild=args.force_rebuild_annfiles)

    cfg = build_runtime_cfg(args, ann_dir)
    runtime_cfg_path = args.output_root / 'runtime_config.py'
    cfg.dump(str(runtime_cfg_path))

    print(f'CPU mode: CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f'Input images: {args.image_dir}')
    print(f'Image count: {len(images)}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Config: {args.config}')
    print(f'Runtime config: {runtime_cfg_path}')
    print(f'Annfiles: {ann_dir} (created={created}, reused={reused})')
    print(f'Output root: {args.output_root}')

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    runner = Runner.from_cfg(cfg)
    runner.test()

    src_vis_dir, dst_vis_dir = collect_vis_dir(args.output_root)
    print(f'Raw vis dir: {src_vis_dir}')
    print(f'Collected vis dir: {dst_vis_dir}')


if __name__ == '__main__':
    main()
