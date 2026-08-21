#!/usr/bin/env python
"""Convert OpenRSD pkl labels to DOTA-style txt annfiles."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenRSD label pkl files to DOTA txt annotations.")
    parser.add_argument("--pkl-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional image directory. When set, create empty txt files for images "
        "without a matching pkl label.")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_label(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    texts = list(data.get("texts") or [])
    polys = np.asarray(data.get("polys") if data.get("polys") is not None else [])
    if len(texts) == 0:
        return [], np.zeros((0, 8), dtype=np.float32)
    polys = polys.astype(np.float32).reshape(-1, 8)
    if len(texts) != len(polys):
        raise ValueError(
            f"{path}: texts/polys length mismatch: {len(texts)} vs {len(polys)}")
    return texts, polys


def _fmt_num(value: float) -> str:
    rounded = round(float(value), 6)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def _write_txt(path: Path, texts: list[str], polys: np.ndarray, difficulty: int) -> None:
    lines = []
    for cls_name, poly in zip(texts, polys):
        coords = " ".join(_fmt_num(value) for value in poly.tolist())
        lines.append(f"{coords} {cls_name} {difficulty}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _image_stems(image_dir: Path) -> set[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return {p.stem for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}


def main() -> None:
    args = parse_args()
    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.out_dir)
    if not pkl_dir.is_dir():
        raise FileNotFoundError(f"pkl-dir not found: {pkl_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_existing = 0
    objects = 0
    pkl_stems = set()
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        pkl_stems.add(pkl_path.stem)
        out_path = out_dir / f"{pkl_path.stem}.txt"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        texts, polys = _load_label(pkl_path)
        _write_txt(out_path, texts, polys, args.difficulty)
        converted += 1
        objects += len(texts)

    empty_from_images = 0
    missing_label_stems: list[str] = []
    if args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image-dir not found: {image_dir}")
        for stem in sorted(_image_stems(image_dir) - pkl_stems):
            out_path = out_dir / f"{stem}.txt"
            missing_label_stems.append(stem)
            if out_path.exists() and not args.overwrite:
                continue
            out_path.write_text("", encoding="utf-8")
            empty_from_images += 1

    print(f"pkl_dir: {pkl_dir}")
    print(f"out_dir: {out_dir}")
    print(f"converted: {converted}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"objects: {objects}")
    print(f"empty_from_images: {empty_from_images}")
    if missing_label_stems:
        print("missing_label_stems: " + ",".join(missing_label_stems[:20]))
    print(f"abs_out_dir: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
