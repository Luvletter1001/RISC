#!/usr/bin/env python
"""Build a deterministic internal train/val split for DOTA-style datasets.

The script links existing files instead of copying images. It is intended for
datasets that only have a train-like split but already contain OpenRSD-ready
files:

    source/images/*.png
    source/labelTxt/*.txt
    source/Step6_Format_labels/*.pkl

Output layout:

    out/train/images, out/train/labelTxt, out/train/annfiles,
    out/train/Step6_Format_labels
    out/val/images, out/val/labelTxt, out/val/annfiles,
    out/val/Step6_Format_labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-modulo", type=int, default=5)
    parser.add_argument("--val-remainder", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing symlinks in the output split. Real files are never removed.",
    )
    return parser.parse_args()


def stable_bucket(stem: str, modulo: int) -> int:
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = images_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def parse_label_file(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if not parts or parts[0].startswith(("imagesource", "gsd")):
            continue
        if len(parts) >= 9:
            counts[parts[8]] += 1
    return counts


def make_link(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace symlinks")
        if not dst.is_symlink():
            raise FileExistsError(f"{dst} exists and is not a symlink; refusing to overwrite")
        dst.unlink()
    os.symlink(src, dst)


def link_sample(source: dict[str, Path], out_root: Path, split: str, overwrite: bool) -> None:
    stem = source["label"].stem
    image = source["image"]
    label = source["label"]
    pkl = source["pkl"]
    split_root = out_root / split
    make_link(image, split_root / "images" / image.name, overwrite)
    make_link(label, split_root / "labelTxt" / label.name, overwrite)
    make_link(label, split_root / "annfiles" / label.name, overwrite)
    make_link(pkl, split_root / "Step6_Format_labels" / f"{stem}.pkl", overwrite)


def collect_samples(source_dir: Path) -> tuple[list[dict[str, Path]], dict[str, object]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labelTxt"
    pkls_dir = source_dir / "Step6_Format_labels"
    missing_images: list[str] = []
    missing_pkls: list[str] = []
    samples: list[dict[str, Path]] = []

    for label in sorted(labels_dir.glob("*.txt")):
        stem = label.stem
        image = find_image(images_dir, stem)
        pkl = pkls_dir / f"{stem}.pkl"
        if image is None:
            missing_images.append(stem)
            continue
        if not pkl.exists():
            missing_pkls.append(stem)
            continue
        samples.append({"image": image, "label": label, "pkl": pkl})

    audit = {
        "source_dir": str(source_dir),
        "label_count": len(list(labels_dir.glob("*.txt"))),
        "sample_count": len(samples),
        "missing_image_count": len(missing_images),
        "missing_pkl_count": len(missing_pkls),
        "missing_image_examples": missing_images[:20],
        "missing_pkl_examples": missing_pkls[:20],
    }
    return samples, audit


def summarize_split(samples: list[dict[str, Path]]) -> dict[str, object]:
    class_counts: Counter[str] = Counter()
    nonempty = 0
    for sample in samples:
        counts = parse_label_file(sample["label"])
        if counts:
            nonempty += 1
        class_counts.update(counts)
    return {
        "image_count": len(samples),
        "nonempty_label_count": nonempty,
        "object_count": int(sum(class_counts.values())),
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
    }


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if args.val_modulo <= 1:
        raise ValueError("--val-modulo must be > 1")
    if not 0 <= args.val_remainder < args.val_modulo:
        raise ValueError("--val-remainder must satisfy 0 <= r < modulo")

    samples, audit = collect_samples(source_dir)
    if audit["missing_image_count"] or audit["missing_pkl_count"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    if not samples:
        raise RuntimeError(f"No samples found under {source_dir}")

    split_samples = {"train": [], "val": []}
    for sample in samples:
        stem = sample["label"].stem
        split = "val" if stable_bucket(stem, args.val_modulo) == args.val_remainder else "train"
        split_samples[split].append(sample)
        link_sample(sample, out_dir, split, args.overwrite)

    summary = {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "split_rule": {
            "hash": "sha1(stem)[:12]",
            "val_modulo": args.val_modulo,
            "val_remainder": args.val_remainder,
        },
        "audit": audit,
        "train": summarize_split(split_samples["train"]),
        "val": summarize_split(split_samples["val"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "internal_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
