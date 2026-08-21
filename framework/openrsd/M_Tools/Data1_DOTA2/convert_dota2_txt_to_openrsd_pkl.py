import argparse
import os
import pickle
from pathlib import Path

import numpy as np


DOTA2_CLASSES = [
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DOTA2 txt annfiles to OpenRSD online pkl labels.')
    parser.add_argument(
        '--ann-dir',
        default='./data/DOTA2_1024_500/ss_train/annfiles',
        help='Directory containing DOTA-format txt annotation files.')
    parser.add_argument(
        '--out-dir',
        default='./data/DOTA2_1024_500/ss_train/Step6_Format_labels',
        help='Output directory for OpenRSD DOTADatasetOnline pkl files.')
    parser.add_argument(
        '--keep-empty',
        action='store_true',
        help='Keep empty txt files as empty pkl files. By default they are skipped.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output pkl files.')
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Convert only the first N txt files. 0 means convert all.')
    return parser.parse_args()


def load_dota_txt(txt_path):
    texts = []
    polys = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].startswith('imagesource') or parts[0].startswith('gsd'):
                continue
            if len(parts) < 10:
                raise ValueError(
                    f'Invalid DOTA line in {txt_path}:{line_no}: {line.strip()}')

            cls_name = parts[8]
            if cls_name not in DOTA2_CLASSES:
                raise ValueError(
                    f'Unknown DOTA2 class "{cls_name}" in {txt_path}:{line_no}')

            polys.append([float(x) for x in parts[:8]])
            texts.append(cls_name)

    return texts, np.asarray(polys, dtype=np.float32).reshape(-1, 8)


def main():
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)

    if not ann_dir.is_dir():
        raise FileNotFoundError(f'Annotation directory not found: {ann_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)

    txt_paths = sorted(ann_dir.glob('*.txt'))
    if args.limit > 0:
        txt_paths = txt_paths[:args.limit]

    converted = 0
    skipped_existing = 0
    skipped_empty = 0

    for txt_path in txt_paths:
        out_path = out_dir / f'{txt_path.stem}.pkl'
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        texts, polys = load_dota_txt(txt_path)
        if len(texts) == 0 and not args.keep_empty:
            skipped_empty += 1
            continue

        out_data = dict(
            visual_embeds=None,
            texts=texts,
            text_embeds=None,
            polys=polys,
            cls_list=DOTA2_CLASSES,
        )
        with open(out_path, 'wb') as f:
            pickle.dump(out_data, f)
        converted += 1

    print(f'Txt files: {len(txt_paths)}')
    print(f'Converted: {converted}')
    print(f'Skipped existing: {skipped_existing}')
    print(f'Skipped empty: {skipped_empty}')
    print(f'Output: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
