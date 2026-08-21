#!/usr/bin/env python3
"""Convert DOTA1 txt annfiles to OpenRSD DOTADatasetOnline pkl labels."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


DOTA1_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--keep-empty', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def load_dota_txt(txt_path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    polys: list[list[float]] = []
    with txt_path.open('r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts or parts[0].startswith('imagesource') or parts[0].startswith('gsd'):
                continue
            if len(parts) < 9:
                raise ValueError(f'Invalid DOTA line in {txt_path}:{line_no}: {line.strip()}')
            cls_name = parts[8]
            if cls_name not in DOTA1_CLASSES:
                raise ValueError(f'Unknown DOTA1 class "{cls_name}" in {txt_path}:{line_no}')
            polys.append([float(x) for x in parts[:8]])
            texts.append(cls_name)
    return texts, np.asarray(polys, dtype=np.float32).reshape(-1, 8)


def main() -> None:
    args = parse_args()
    if not args.ann_dir.is_dir():
        raise FileNotFoundError(f'Annotation directory not found: {args.ann_dir}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt_paths = sorted(args.ann_dir.glob('*.txt'))
    if args.limit > 0:
        txt_paths = txt_paths[:args.limit]

    converted = 0
    skipped_existing = 0
    skipped_empty = 0
    for txt_path in txt_paths:
        out_path = args.out_dir / f'{txt_path.stem}.pkl'
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        texts, polys = load_dota_txt(txt_path)
        if not texts and not args.keep_empty:
            skipped_empty += 1
            continue
        with out_path.open('wb') as f:
            pickle.dump(
                dict(
                    visual_embeds=None,
                    texts=texts,
                    text_embeds=None,
                    polys=polys,
                    cls_list=DOTA1_CLASSES,
                ),
                f,
            )
        converted += 1

    print(f'Txt files: {len(txt_paths)}')
    print(f'Converted: {converted}')
    print(f'Skipped existing: {skipped_existing}')
    print(f'Skipped empty: {skipped_empty}')
    print(f'Output: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
