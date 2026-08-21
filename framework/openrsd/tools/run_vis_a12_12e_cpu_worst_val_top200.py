#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

from openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

# Reuse the existing CPU visualization setup so the inference path, hook,
# and palette stay aligned with the current DOTA test visualization script.
import run_vis_a12_12e_cpu_dota_test as vis_common


PROJECT_ROOT = vis_common.PROJECT_ROOT
DEFAULT_IMAGE_DIR = PROJECT_ROOT / 'data' / 'DOTA2_1024_500' / 'ss_val' / 'images'
DEFAULT_TXT_PATH = (
    vis_common.DEFAULT_CHECKPOINT.parent / 'worst_val_ap50_top200_epoch12.txt')
DEFAULT_RANKING_CSV = DEFAULT_TXT_PATH.with_name(
    DEFAULT_TXT_PATH.stem + '_all.csv')
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / 'workdir_vis' /
    f'{DEFAULT_TXT_PATH.stem}_cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize the worst val AP50 top-200 DOTA images on CPU.')
    parser.add_argument(
        '--txt-path',
        type=Path,
        default=DEFAULT_TXT_PATH,
        help='Fallback txt file with one image file name per line.')
    parser.add_argument(
        '--ranking-csv',
        type=Path,
        default=DEFAULT_RANKING_CSV,
        help='Full ranking csv. When present, selection is rebuilt from it.')
    parser.add_argument(
        '--image-dir',
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help='Validation image directory.')
    parser.add_argument(
        '--config',
        type=Path,
        default=vis_common.DEFAULT_CONFIG,
        help='Finetuned config path.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=vis_common.DEFAULT_CHECKPOINT,
        help='Finetuned checkpoint path.')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help='Outputs will be written under this directory.')
    parser.add_argument(
        '--base-ann',
        type=Path,
        default=vis_common.DEFAULT_BASE_ANN,
        help='Template annfile copied to each selected image stem.')
    parser.add_argument(
        '--ann-dir',
        type=Path,
        default=None,
        help='Optional annfiles directory. Defaults to <output-root>/annfiles.')
    parser.add_argument(
        '--topk',
        type=int,
        default=200,
        help='How many filtered images to visualize.')
    parser.add_argument(
        '--min-num-gts',
        type=int,
        default=20,
        help='Only keep images with num_gts strictly greater than this.')
    parser.add_argument(
        '--max-patches-per-source',
        type=int,
        default=3,
        help='Maximum kept patches from the same source image.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Visualization score threshold.')
    parser.add_argument(
        '--allow-txt-fallback',
        action='store_true',
        help='Use --txt-path if --ranking-csv is missing.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='CPU test batch size.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='CPU dataloader workers. Use 0 in restricted environments.')
    parser.add_argument(
        '--force-rebuild-annfiles',
        action='store_true',
        help='Rewrite annfiles even if they already exist.')
    return parser.parse_args()


def read_image_names(txt_path: Path):
    image_names = []
    seen = set()
    for raw_line in txt_path.read_text(encoding='utf-8').splitlines():
        image_name = raw_line.strip()
        if not image_name or image_name in seen:
            continue
        seen.add(image_name)
        image_names.append(image_name)
    if not image_names:
        raise RuntimeError(f'No image names found in {txt_path}')
    return image_names


def get_source_image_id(file_name: str) -> str:
    return file_name.split('__', 1)[0]


def select_image_names_from_ranking(ranking_csv: Path, *, topk: int,
                                    min_num_gts: int,
                                    max_patches_per_source: int):
    selected_rows = []
    kept_per_source = Counter()
    seen = set()

    with ranking_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['file_name'].strip()
            if not file_name or file_name in seen:
                continue

            seen.add(file_name)
            num_gts = int(row['num_gts'])
            if num_gts <= min_num_gts:
                continue

            source_image_id = get_source_image_id(file_name)
            if kept_per_source[source_image_id] >= max_patches_per_source:
                continue

            kept_per_source[source_image_id] += 1
            row['source_image_id'] = source_image_id
            selected_rows.append(row)
            if len(selected_rows) >= topk:
                break

    if not selected_rows:
        raise RuntimeError(
            f'No images remain after filtering {ranking_csv} with '
            f'num_gts > {min_num_gts} and '
            f'max {max_patches_per_source} patches per source image.')
    return selected_rows


def select_image_names(args):
    if args.ranking_csv.is_file():
        selected_rows = select_image_names_from_ranking(
            args.ranking_csv,
            topk=args.topk,
            min_num_gts=args.min_num_gts,
            max_patches_per_source=args.max_patches_per_source)
        return [row['file_name'] for row in selected_rows], selected_rows

    if not args.allow_txt_fallback:
        raise FileNotFoundError(
            f'ranking csv not found: {args.ranking_csv}. '
            'Pass --allow-txt-fallback to use the raw txt list instead.')

    image_names = read_image_names(args.txt_path)[:args.topk]
    selected_rows = [
        dict(
            rank=str(index),
            file_name=image_name,
            image_ap50='',
            num_gts='',
            num_dets='',
            num_active_classes='',
            img_path='',
            source_image_id=get_source_image_id(image_name))
        for index, image_name in enumerate(image_names, start=1)
    ]
    return image_names, selected_rows


def write_selection_csv(selected_rows, output_root: Path):
    csv_path = output_root / 'selected_images.csv'
    fieldnames = [
        'rank',
        'file_name',
        'image_ap50',
        'num_gts',
        'num_dets',
        'num_active_classes',
        'img_path',
        'source_image_id',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})
    return csv_path


def ensure_selected_annfiles(image_names, image_dir: Path, ann_dir: Path,
                             base_ann: Path, force_rebuild: bool):
    missing = [
        image_name for image_name in image_names
        if not (image_dir / image_name).is_file()
    ]
    if missing:
        preview = ', '.join(missing[:5])
        suffix = '' if len(missing) <= 5 else ', ...'
        raise FileNotFoundError(
            f'{len(missing)} selected images are missing under {image_dir}: '
            f'{preview}{suffix}')

    target_stems = {Path(image_name).stem for image_name in image_names}
    removed = 0
    if ann_dir.exists():
        for ann_path in ann_dir.glob('*.pkl'):
            if ann_path.stem not in target_stems:
                ann_path.unlink()
                removed += 1

    ann_dir.mkdir(parents=True, exist_ok=True)
    template = vis_common.load_pickle(base_ann)
    created = 0
    reused = 0
    for image_name in image_names:
        ann_path = ann_dir / f'{Path(image_name).stem}.pkl'
        if ann_path.exists() and not force_rebuild:
            reused += 1
            continue
        vis_common.save_pickle(template, ann_path)
        created += 1
    return created, reused, removed


def validate_args(args):
    if not args.txt_path.is_file():
        raise FileNotFoundError(f'txt file not found: {args.txt_path}')
    if not args.ranking_csv.is_file() and not args.allow_txt_fallback:
        raise FileNotFoundError(f'ranking csv not found: {args.ranking_csv}')
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f'image dir not found: {args.image_dir}')
    if not args.config.is_file():
        raise FileNotFoundError(f'config not found: {args.config}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not args.base_ann.is_file():
        raise FileNotFoundError(f'base ann not found: {args.base_ann}')
    if args.topk <= 0:
        raise ValueError('--topk must be > 0')
    if args.min_num_gts < 0:
        raise ValueError('--min-num-gts must be >= 0')
    if args.max_patches_per_source <= 0:
        raise ValueError('--max-patches-per-source must be > 0')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be > 0')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')


def main():
    args = parse_args()
    validate_args(args)

    args.output_root.mkdir(parents=True, exist_ok=True)
    ann_dir = args.ann_dir if args.ann_dir is not None else args.output_root / 'annfiles'

    image_names, selected_rows = select_image_names(args)
    created, reused, removed = ensure_selected_annfiles(
        image_names=image_names,
        image_dir=args.image_dir,
        ann_dir=ann_dir,
        base_ann=args.base_ann,
        force_rebuild=args.force_rebuild_annfiles)

    selected_manifest_path = args.output_root / 'selected_images.txt'
    selected_manifest_path.write_text(
        ''.join(f'{image_name}\n' for image_name in image_names),
        encoding='utf-8')
    selected_csv_path = write_selection_csv(selected_rows, args.output_root)

    cfg = vis_common.build_runtime_cfg(args, ann_dir)
    runtime_cfg_path = args.output_root / 'runtime_config.py'
    cfg.dump(str(runtime_cfg_path))

    unique_sources = len({
        row['source_image_id'] for row in selected_rows
    })

    print(f'CPU mode: CUDA_VISIBLE_DEVICES={vis_common.os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f'Selection source: {args.ranking_csv if args.ranking_csv.is_file() else args.txt_path}')
    print(f'Input images: {args.image_dir}')
    print(
        f'Selection rules: topk={args.topk}, num_gts>{args.min_num_gts}, '
        f'max_patches_per_source={args.max_patches_per_source}')
    print(f'Selected image count: {len(image_names)}')
    print(f'Unique source images: {unique_sources}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Config: {args.config}')
    print(f'Runtime config: {runtime_cfg_path}')
    print(
        f'Annfiles: {ann_dir} '
        f'(created={created}, reused={reused}, removed_stale={removed})')
    print(f'Selected manifest: {selected_manifest_path}')
    print(f'Selected csv: {selected_csv_path}')
    print(f'Output root: {args.output_root}')

    vis_common.register_all_modules_mmdet(init_default_scope=False)
    vis_common.register_all_modules(init_default_scope=False)

    runner = vis_common.Runner.from_cfg(cfg)
    runner.test()

    src_vis_dir, dst_vis_dir = vis_common.collect_vis_dir(args.output_root)
    print(f'Raw vis dir: {src_vis_dir}')
    print(f'Collected vis dir: {dst_vis_dir}')


if __name__ == '__main__':
    main()
