import argparse
import os
import pickle
from pathlib import Path

import numpy as np


DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DOTA txt annfiles to OpenRSD online pkl labels.')
    parser.add_argument(
        '--ann-dir',
        default='/data/zcy/dataset/trainval_ss/annfiles',
        help='Directory containing DOTA-format txt annotation files.')
    parser.add_argument(
        '--out-dir',
        default='./data/OpenRSD_DOTA_trainval_ss/annfiles',
        help='Output directory for OpenRSD pkl annotation files.')
    parser.add_argument(
        '--keep-empty',
        action='store_true',
        help='Keep empty annotation files. By default empty txt files are skipped.')
    return parser.parse_args()


def load_dota_txt(txt_path):
    texts = []
    polys = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10 or parts[0].startswith('imagesource') or parts[0].startswith('gsd'):
                continue
            cls_name = parts[8]
            if cls_name not in DOTA_CLASSES:
                raise ValueError(f'Unknown DOTA class "{cls_name}" in {txt_path}')
            polys.append([float(x) for x in parts[:8]])
            texts.append(cls_name)
    return texts, np.asarray(polys, dtype=np.float32).reshape(-1, 8)


def main():
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_empty = 0
    for txt_path in sorted(ann_dir.glob('*.txt')):
        texts, polys = load_dota_txt(txt_path)
        if len(texts) == 0 and not args.keep_empty:
            skipped_empty += 1
            continue

        out_data = dict(
            visual_embeds=None,
            texts=texts,
            text_embeds=None,
            polys=polys,
            cls_list=DOTA_CLASSES,
        )
        with open(out_dir / f'{txt_path.stem}.pkl', 'wb') as f:
            pickle.dump(out_data, f)
        converted += 1

    print(f'Converted: {converted}')
    print(f'Skipped empty: {skipped_empty}')
    print(f'Output: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
