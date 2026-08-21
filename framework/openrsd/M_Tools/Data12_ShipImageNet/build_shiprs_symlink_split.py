#!/usr/bin/env python
"""Build a deterministic train/val symlink split for ShipRSImageNet_DOTA."""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deterministic ShipRSImageNet_DOTA symlink split.")
    parser.add_argument(
        "--src-train",
        default="/data1/zcy/datasets/ShipRSImageNet_DOTA/train",
        help="Source train directory containing images, labelTxt and Step6_Format_labels.")
    parser.add_argument(
        "--dst-root",
        default="/data1/zcy/datasets/ShipRSImageNet_DOTA_split_20260619",
        help="Destination dataset root to create.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument(
        "--overwrite-symlinks",
        action="store_true",
        help="Replace existing symlinks in the destination. Regular files are never overwritten.")
    return parser.parse_args()


def collect_complete_stems(src_train):
    img_dir = src_train / "images"
    ann_dir = src_train / "labelTxt"
    pkl_dir = src_train / "Step6_Format_labels"
    image_stems = {p.stem for p in img_dir.glob("*.png")}
    ann_stems = {p.stem for p in ann_dir.glob("*.txt")}
    pkl_stems = {p.stem for p in pkl_dir.glob("*.pkl")}
    complete = sorted(image_stems & ann_stems & pkl_stems)
    return complete, {
        "image_count": len(image_stems),
        "ann_count": len(ann_stems),
        "pkl_count": len(pkl_stems),
        "complete_count": len(complete),
        "image_without_ann": len(image_stems - ann_stems),
        "ann_without_image": len(ann_stems - image_stems),
        "complete_missing_pkl": len((image_stems & ann_stems) - pkl_stems),
    }


def read_classes(ann_path):
    classes = []
    for raw in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) >= 9:
            classes.append(parts[8])
    return classes


def safe_symlink(src, dst, overwrite_symlinks):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and overwrite_symlinks:
            dst.unlink()
        else:
            if dst.is_symlink() and Path(os.readlink(dst)) == src:
                return
            raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    dst.symlink_to(src)


def write_split(src_train, dst_root, split_name, stems, overwrite_symlinks):
    img_src = src_train / "images"
    ann_src = src_train / "labelTxt"
    pkl_src = src_train / "Step6_Format_labels"
    class_counter = Counter()
    object_count = 0
    for stem in stems:
        safe_symlink(
            (img_src / f"{stem}.png").resolve(),
            dst_root / split_name / "images" / f"{stem}.png",
            overwrite_symlinks,
        )
        ann_path = ann_src / f"{stem}.txt"
        safe_symlink(
            ann_path.resolve(),
            dst_root / split_name / "labelTxt" / f"{stem}.txt",
            overwrite_symlinks,
        )
        safe_symlink(
            (pkl_src / f"{stem}.pkl").resolve(),
            dst_root / split_name / "Step6_Format_labels" / f"{stem}.pkl",
            overwrite_symlinks,
        )
        classes = read_classes(ann_path)
        class_counter.update(classes)
        object_count += len(classes)
    return {
        "images": len(stems),
        "annfiles": len(stems),
        "pkl": len(stems),
        "objects": object_count,
        "classes": dict(sorted(class_counter.items())),
    }


def main():
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    src_train = Path(args.src_train)
    dst_root = Path(args.dst_root)
    complete, source_stats = collect_complete_stems(src_train)
    if not complete:
        raise RuntimeError(f"No complete samples found under {src_train}")

    stems = list(complete)
    random.Random(args.seed).shuffle(stems)
    val_count = max(1, int(round(len(stems) * args.val_fraction)))
    val_stems = sorted(stems[:val_count])
    train_stems = sorted(stems[val_count:])

    train_stats = write_split(
        src_train, dst_root, "train", train_stems, args.overwrite_symlinks)
    val_stats = write_split(
        src_train, dst_root, "val", val_stems, args.overwrite_symlinks)

    (dst_root / "splits").mkdir(parents=True, exist_ok=True)
    (dst_root / "splits" / "train.txt").write_text(
        "\n".join(train_stems) + "\n", encoding="utf-8")
    (dst_root / "splits" / "val.txt").write_text(
        "\n".join(val_stems) + "\n", encoding="utf-8")

    manifest = {
        "src_train": str(src_train),
        "dst_root": str(dst_root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "source_stats": source_stats,
        "train": train_stats,
        "val": val_stats,
    }
    (dst_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
