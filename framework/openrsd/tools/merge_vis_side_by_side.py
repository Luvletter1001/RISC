import argparse
from pathlib import Path

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_LEFT_ROOT = Path('/data1/zcy/OpenRSD/workdir_vis/pre_vs_finetune')
DEFAULT_RIGHT_ROOT = Path('/data1/zcy/OpenRSD/workdir_vis/notext_cls')
DEFAULT_OUT_ROOT = Path('/data1/zcy/OpenRSD/workdir_vis/pre_vs_finetune_vs_notext_cls')
VALID_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge vis images from two roots side by side.')
    parser.add_argument(
        '--left-root',
        default=str(DEFAULT_LEFT_ROOT),
        help='Root containing legacy vis/<subdir>/vis images.')
    parser.add_argument(
        '--right-root',
        default=str(DEFAULT_RIGHT_ROOT),
        help='Root containing finetuned vis/<subdir>/vis images.')
    parser.add_argument(
        '--out-root',
        default=str(DEFAULT_OUT_ROOT),
        help='Output root for merged images.')
    parser.add_argument(
        '--subset',
        default='',
        help='Comma separated folder names to process. Empty means all common folders.')
    parser.add_argument(
        '--output-ext',
        default='.png',
        choices=['.png', '.jpg'],
        help='Merged image extension.')
    return parser.parse_args()


def collect_vis_dirs(root: Path):
    vis_dirs = {}
    if not root.is_dir():
        raise FileNotFoundError(f'root not found: {root}')
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        vis_dir = subdir / 'vis'
        if vis_dir.is_dir():
            vis_dirs[subdir.name] = vis_dir
    return vis_dirs


def collect_images(vis_dir: Path):
    images = {}
    for path in sorted(vis_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            images[path.stem] = path
    return images


def merge_pair(left_path: Path, right_path: Path, out_path: Path):
    with Image.open(left_path) as left_img, Image.open(right_path) as right_img:
        left_img = left_img.convert('RGB')
        right_img = right_img.convert('RGB')

        out_img = Image.new(
            'RGB',
            (left_img.width + right_img.width,
             max(left_img.height, right_img.height)),
            (255, 255, 255))
        out_img.paste(left_img, (0, 0))
        out_img.paste(right_img, (left_img.width, 0))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == '.jpg':
            out_img.save(out_path, quality=95)
        else:
            out_img.save(out_path)


def main():
    args = parse_args()
    left_root = Path(args.left_root)
    right_root = Path(args.right_root)
    out_root = Path(args.out_root)
    subset = {item for item in args.subset.split(',') if item}

    left_dirs = collect_vis_dirs(left_root)
    right_dirs = collect_vis_dirs(right_root)
    common_names = sorted(set(left_dirs) & set(right_dirs))
    if subset:
        common_names = [name for name in common_names if name in subset]

    if not common_names:
        raise RuntimeError('no common vis folders found')

    total_outputs = 0
    skipped_folder_count = 0

    for name in common_names:
        left_images = collect_images(left_dirs[name])
        right_images = collect_images(right_dirs[name])
        common_stems = sorted(set(left_images) & set(right_images))
        if not common_stems:
            skipped_folder_count += 1
            print(f'SKIP {name}: no common image stems')
            continue

        out_dir = out_root / name / 'vis'
        out_dir.mkdir(parents=True, exist_ok=True)

        left_only = len(set(left_images) - set(right_images))
        right_only = len(set(right_images) - set(left_images))
        print(
            f'FOLDER {name}: matched={len(common_stems)} '
            f'left_only={left_only} right_only={right_only}')

        for stem in common_stems:
            out_path = out_dir / f'{stem}{args.output_ext}'
            merge_pair(left_images[stem], right_images[stem], out_path)
            total_outputs += 1

    print(
        f'DONE folders={len(common_names)} '
        f'skipped_without_matches={skipped_folder_count} '
        f'images={total_outputs} out_root={out_root}')


if __name__ == '__main__':
    main()
