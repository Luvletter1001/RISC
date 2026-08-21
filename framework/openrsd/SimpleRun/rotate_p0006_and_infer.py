import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2


PROJ_ROOT = Path('/data1/zcy/OpenRSD')
DEFAULT_IMAGE_STEM = 'P1819__1024__424___0'
DEFAULT_VIS_ROOT = PROJ_ROOT / 'vis'
DEFAULT_CONFIG = (
    './M_configs/Step2_A10_Large_Pretrain_Stage3/'
    'A10_flex_rtm_v3_1_formal.py'
)
DEFAULT_CHECKPOINT = (
    './results/MMR_AD_A10_flex_rtm_v3_1_formal/'
    'epoch_24_weights_only.pth'
)
BASE_ANN = PROJ_ROOT / 'SimpleRun/test_dataset/annfiles/P1860__1024__932___714.pkl'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rotate one DOTA image every 5 degrees and run OpenRSD inference.')
    parser.add_argument(
        '--source-image',
        default=None,
        help='Source image path. If omitted, the script uses the clean DOTA '
             'test image when available.')
    parser.add_argument(
        '--pic-name',
        default=None,
        help='Name used for output folders and rotated image stems. Default '
             'uses source image stem, or the built-in default image name when '
             '--source-image is omitted.')
    parser.add_argument(
        '--angle-step',
        type=int,
        default=5,
        help='Rotation interval in degrees.')
    parser.add_argument(
        '--dataset-dir',
        default=None,
        help='Output dataset dir containing images/ and annfiles/. Default: '
             '/data1/zcy/OpenRSD/vis/<pic-name>/dataset')
    parser.add_argument(
        '--result-dir',
        default=None,
        help='Output dir for results.pkl and visualized detections. Default: '
             '/data1/zcy/OpenRSD/vis/<pic-name>')
    parser.add_argument(
        '--gpu',
        default=os.environ.get('OPENRSD_GPU', '4'),
        help='GPU id passed to step1_inference.py.')
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='Model config used for inference.')
    parser.add_argument(
        '--checkpoint',
        default=DEFAULT_CHECKPOINT,
        help='Official A10 checkpoint. Default uses the converted weights_only '
             'file generated from epoch_24.pth.')
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Only generate rotated images and annfiles.')
    parser.add_argument(
        '--vis-box-only',
        action='store_true',
        help='Detection vis: polygons only, no class name or score labels.',
    )
    return parser.parse_args()


def find_source_image(source_image, pic_name):
    if source_image:
        src = Path(source_image)
        if not src.exists():
            raise FileNotFoundError(f'Source image not found: {src}')
        return src

    candidates = [
        DEFAULT_VIS_ROOT / f'{pic_name}.jpg',
        Path('/data/zcy/dataset/test_ss/images') / f'{pic_name}.png',
        Path('/data/zcy/dataset/test_ss/images') / f'{pic_name}.jpg',
        PROJ_ROOT / 'SimpleRun/results/vis' / f'{pic_name}.jpg',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'Could not find source image for {pic_name}. Checked:\n' +
        '\n'.join(str(p) for p in candidates))


def rotate_keep_size(img, angle):
    h, w = img.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114))


def save_rotated_images(source_path, dataset_dir, angle_step, pic_name):
    if angle_step <= 0 or 360 % angle_step != 0:
        raise ValueError('angle-step must be a positive divisor of 360.')

    image_dir = dataset_dir / 'images'
    ann_dir = dataset_dir / 'annfiles'
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(source_path))
    if img is None:
        raise RuntimeError(f'Failed to read source image: {source_path}')

    with open(BASE_ANN, 'rb') as f:
        base_ann = pickle.load(f)

    saved = []
    for angle in range(0, 360, angle_step):
        rotated = rotate_keep_size(img, angle)
        stem = f'{pic_name}_rot{angle:03d}'
        out_png = image_dir / f'{stem}.png'
        out_jpg = image_dir / f'{stem}.jpg'
        ok = cv2.imwrite(str(out_png), rotated)
        if not ok:
            raise RuntimeError(f'Failed to write rotated image: {out_png}')
        ok = cv2.imwrite(str(out_jpg), rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f'Failed to write rotated image: {out_jpg}')

        out_ann = ann_dir / f'{stem}.pkl'
        with open(out_ann, 'wb') as f:
            pickle.dump(base_ann, f)
        saved.append(out_png)

    return image_dir, ann_dir, saved


def run_inference(args, dataset_dir, ann_dir, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('PYTHONNOUSERSITE', '1')
    env.setdefault(
        'MPLCONFIGDIR',
        str(PROJ_ROOT / 'SimpleRun/.mplconfig'))
    env.update({
        'OPENRSD_GPU': str(args.gpu),
        'OPENRSD_CONFIG': args.config,
        'OPENRSD_CHECKPOINT': args.checkpoint,
        'OPENRSD_DATA_ROOT': str(dataset_dir),
        'OPENRSD_IMG_DIR': 'images',
        'OPENRSD_ANN_DIR': str(ann_dir),
        'OPENRSD_OUT_RESULTS': str(result_dir / 'results.pkl'),
        'OPENRSD_VIS_OUT_DIR': str(result_dir / 'vis'),
        'OPENRSD_VIS_IMG_EXT': '.jpg',
        'OPENRSD_SAVE_VIS': '1',
        'OPENRSD_SAVE_EMPTY_VIS': '1',
    })
    if args.vis_box_only:
        env['OPENRSD_VIS_BOX_ONLY'] = '1'

    cmd = [sys.executable, str(PROJ_ROOT / 'SimpleRun/step1_inference.py')]
    print('Running inference:')
    print(' '.join(cmd))
    subprocess.run(cmd, cwd=str(PROJ_ROOT), env=env, check=True)


def main():
    args = parse_args()
    search_pic_name = args.pic_name or DEFAULT_IMAGE_STEM
    source_path = find_source_image(args.source_image, search_pic_name)
    pic_name = args.pic_name or source_path.stem
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (
        DEFAULT_VIS_ROOT / pic_name / 'dataset')
    result_dir = Path(args.result_dir) if args.result_dir else (
        DEFAULT_VIS_ROOT / pic_name)

    print(f'Source image: {source_path}')
    print(f'pic_name: {pic_name}')
    print(f'Dataset dir: {dataset_dir}')
    print(f'Result dir: {result_dir}')
    image_dir, ann_dir, saved = save_rotated_images(
        source_path, dataset_dir, args.angle_step, pic_name)
    print(f'Saved {len(saved)} rotated images to: {image_dir}')
    print(f'Saved annfiles to: {ann_dir}')

    if args.skip_inference:
        print('Skip inference requested.')
        return

    run_inference(args, dataset_dir, ann_dir, result_dir)
    print(f'Inference pkl: {result_dir / "results.pkl"}')
    print(f'Visualized detections: {result_dir / "vis"}')


if __name__ == '__main__':
    main()
