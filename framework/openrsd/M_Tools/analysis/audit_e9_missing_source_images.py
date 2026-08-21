#!/usr/bin/env python
"""Audit and optionally recover missing E9 source tile images.

The E9 counterfactual probe records ``missing_source_image`` when a split
tile is absent from the configured source image directory.  This script finds
those missing tile IDs, searches local dataset roots for exact tile files and
source original images, and can reconstruct the 1024 split tile from a found
original image.
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_COMBINED_CSV = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'e9_scale_counterfactual_manifest_full_combined/'
    'e9_counterfactual_probe_combined.csv')
DEFAULT_OUT_DIR = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'e9_missing_source_audit')
DEFAULT_SEARCH_ROOTS = [
    '/data/zcy/dataset',
    '/data1/zcy/datasets',
    '/data1/zcy/OpenRSD',
]
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
TILE_RE = re.compile(
    r'^(?P<orig>.+)__(?P<size>\d+)__(?P<x>\d+)___(?P<y>\d+)$')
PADDING_VALUE = (104, 116, 124)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined-csv', default=DEFAULT_COMBINED_CSV)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--search-roots', nargs='*', default=DEFAULT_SEARCH_ROOTS)
    parser.add_argument('--recover-tiles', action='store_true')
    parser.add_argument('--recovered-image-dir',
                        default=str(Path(DEFAULT_OUT_DIR) / 'recovered_images'))
    parser.add_argument('--link-found-tiles', action='store_true')
    parser.add_argument('--linked-image-dir',
                        default=str(Path(DEFAULT_OUT_DIR) / 'found_tile_links'))
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def wanted_names(tiles, origs):
    names = set()
    for stem in sorted(set(tiles) | set(origs)):
        for suffix in IMAGE_SUFFIXES:
            names.add(stem + suffix)
    return names


def search_files(roots, names):
    found = defaultdict(list)
    root_list = [Path(root) for root in roots if Path(root).exists()]
    for root in root_list:
        for dirpath, dirnames, filenames in os.walk(root):
            # Keep the search finite inside the repo; work_dirs contains many
            # generated visualizations that are not source images.
            if Path(dirpath).parts[:4] == Path('/data1/zcy/OpenRSD/work_dirs').parts:
                dirnames[:] = []
                continue
            for fn in filenames:
                if fn in names:
                    found[fn].append(str(Path(dirpath) / fn))
    return found


def hits_for_stem(found, stem):
    hits = []
    for suffix in IMAGE_SUFFIXES:
        hits.extend(found.get(stem + suffix, []))
    return sorted(hits)


def choose_canonical_tile_path(hits):
    if not hits:
        return ''
    preferred_prefixes = [
        '/data/zcy/dataset/trainval_ss/images/',
        '/data1/zcy/datasets/trainval_ss/images/',
    ]
    for prefix in preferred_prefixes:
        for hit in hits:
            if hit.startswith(prefix):
                return hit
    png_hits = [hit for hit in hits if hit.lower().endswith('.png')]
    if png_hits:
        return png_hits[0]
    return hits[0]


def link_tile(tile_stem, source_path, linked_dir):
    if not source_path:
        return ''
    linked_dir = Path(linked_dir)
    linked_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(source_path).suffix
    out_path = linked_dir / f'{tile_stem}{suffix}'
    if out_path.exists() or out_path.is_symlink():
        if out_path.is_symlink() and os.readlink(out_path) == source_path:
            return str(out_path)
        out_path.unlink()
    out_path.symlink_to(source_path)
    return str(out_path)


def parse_tile(stem):
    match = TILE_RE.match(stem)
    if not match:
        raise ValueError(f'not a DOTA split tile stem: {stem}')
    return {
        'orig_id': match.group('orig'),
        'size': int(match.group('size')),
        'x': int(match.group('x')),
        'y': int(match.group('y')),
    }


def recover_tile(tile_stem, original_path, out_dir):
    tile = parse_tile(tile_stem)
    img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if img is None:
        return '', 'imread_failed'
    size = tile['size']
    x = tile['x']
    y = tile['y']
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(PADDING_VALUE, dtype=np.uint8)
    crop = img[y:y + size, x:x + size]
    if crop.size == 0:
        return '', 'empty_crop'
    h, w = crop.shape[:2]
    canvas[:h, :w] = crop
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{tile_stem}.png'
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        return '', 'imwrite_failed'
    return str(out_path), 'recovered_from_original'


def main():
    args = parse_args()
    rows = read_rows(args.combined_csv)
    missing = [row for row in rows if row.get('error') == 'missing_source_image']
    tiles = sorted(set(row['tile_img_id'] for row in missing))
    origs = sorted(set(row['orig_id'] for row in missing))
    found = search_files(args.search_roots, wanted_names(tiles, origs))

    tile_rows = []
    recovered = []
    for tile_stem in tiles:
        tile_info = parse_tile(tile_stem)
        tile_hits = hits_for_stem(found, tile_stem)
        orig_hits = hits_for_stem(found, tile_info['orig_id'])
        canonical_tile_path = choose_canonical_tile_path(tile_hits)
        linked_path = ''
        if args.link_found_tiles and canonical_tile_path:
            linked_path = link_tile(
                tile_stem, canonical_tile_path, args.linked_image_dir)
        recovered_path = ''
        recover_status = ''
        if args.recover_tiles and not tile_hits and orig_hits:
            recovered_path, recover_status = recover_tile(
                tile_stem, orig_hits[0], args.recovered_image_dir)
        elif tile_hits:
            recover_status = 'exact_tile_exists'
        elif orig_hits:
            recover_status = 'original_found_not_recovered'
        else:
            recover_status = 'not_found'
        row = {
            'tile_img_id': tile_stem,
            'orig_id': tile_info['orig_id'],
            'tile_x': tile_info['x'],
            'tile_y': tile_info['y'],
            'tile_size': tile_info['size'],
            'missing_rows': sum(1 for r in missing if r['tile_img_id'] == tile_stem),
            'models': ';'.join(sorted(set(
                r['model'] for r in missing if r['tile_img_id'] == tile_stem))),
            'exact_tile_hits': len(tile_hits),
            'exact_tile_paths': ';'.join(tile_hits),
            'canonical_tile_path': canonical_tile_path,
            'linked_tile_path': linked_path,
            'original_hits': len(orig_hits),
            'original_paths': ';'.join(orig_hits),
            'recovered_path': recovered_path,
            'recover_status': recover_status,
        }
        tile_rows.append(row)
        if recovered_path:
            recovered.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'missing_tile_audit.csv', tile_rows)
    write_csv(out_dir / 'missing_tile_canonical_sources.csv', [{
        'tile_img_id': row['tile_img_id'],
        'missing_rows': row['missing_rows'],
        'models': row['models'],
        'canonical_tile_path': row['canonical_tile_path'],
        'linked_tile_path': row['linked_tile_path'],
    } for row in tile_rows])
    by_status = Counter(row['recover_status'] for row in tile_rows)
    by_orig = Counter(row['orig_id'] for row in missing)
    report = [
        '# E9 Missing Source Image Audit',
        '',
        f'- combined_csv: `{args.combined_csv}`',
        f'- missing_rows: `{len(missing)}`',
        f'- unique_missing_tiles: `{len(tiles)}`',
        f'- unique_missing_originals: `{len(origs)}`',
        f'- search_roots: `{", ".join(args.search_roots)}`',
        f'- recovered_tiles: `{len(recovered)}`',
        f'- linked_tiles: `{sum(1 for row in tile_rows if row["linked_tile_path"])}`',
        '',
        '## Status Counts',
        '',
        '| status | tiles |',
        '|---|---:|',
    ]
    for key, value in sorted(by_status.items()):
        report.append(f'| `{key}` | {value} |')
    report.extend([
        '',
        '## Missing Rows By Original',
        '',
        '| orig_id | missing_rows |',
        '|---|---:|',
    ])
    for key, value in sorted(by_orig.items()):
        report.append(f'| `{key}` | {value} |')
    report.extend([
        '',
        '## Per Tile',
        '',
        '| tile_img_id | missing_rows | exact_tile_hits | original_hits | recover_status | canonical_tile_path | linked_tile_path |',
        '|---|---:|---:|---:|---|---|---|',
    ])
    for row in tile_rows:
        report.append(
            f'| `{row["tile_img_id"]}` | {row["missing_rows"]} | '
            f'{row["exact_tile_hits"]} | {row["original_hits"]} | '
            f'`{row["recover_status"]}` | `{row["canonical_tile_path"]}` | '
            f'`{row["linked_tile_path"]}` |')
    (out_dir / 'missing_source_audit.md').write_text(
        '\n'.join(report) + '\n', encoding='utf-8')

    print(f'missing_rows={len(missing)}')
    print(f'unique_missing_tiles={len(tiles)}')
    print(f'unique_missing_originals={len(origs)}')
    print(f'status_counts={dict(by_status)}')
    print(f'audit_csv={out_dir / "missing_tile_audit.csv"}')
    print(f'canonical_csv={out_dir / "missing_tile_canonical_sources.csv"}')
    print(f'audit_md={out_dir / "missing_source_audit.md"}')
    if args.recover_tiles:
        print(f'recovered_image_dir={args.recovered_image_dir}')
    if args.link_found_tiles:
        print(f'linked_image_dir={args.linked_image_dir}')


if __name__ == '__main__':
    main()
