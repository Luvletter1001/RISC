#!/usr/bin/env python
"""Rebuild a labeled dataset subset from OpenRSD image and pkl-label zips.

The script extracts only images that have a matching pkl label. This avoids
turning missing labels into false negative empty annotations.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-zip", required=True)
    parser.add_argument("--label-zip", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _stem_to_member(zip_path: Path, exts: set[str]) -> tuple[dict[str, str], dict[str, int]]:
    mapping: dict[str, str] = {}
    duplicates: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in exts:
                continue
            stem = Path(name).stem
            if stem in mapping:
                duplicates[stem] = duplicates.get(stem, 1) + 1
                continue
            mapping[stem] = name
    return mapping, duplicates


def _fmt_num(value: float) -> str:
    rounded = round(float(value), 6)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def _label_to_txt(label_bytes: bytes, difficulty: int) -> tuple[str, int, Counter[str]]:
    data = pickle.load(io.BytesIO(label_bytes))
    texts = list(data.get("texts") or [])
    polys = data.get("polys")
    if not texts:
        return "", 0, Counter()
    if polys is None:
        raise ValueError("label has texts but no polys")

    lines: list[str] = []
    classes: Counter[str] = Counter()
    for cls_name, poly in zip(texts, polys):
        values = list(poly)
        if len(values) != 8:
            raise ValueError(f"expected 8 polygon values, got {len(values)}")
        coords = " ".join(_fmt_num(v) for v in values)
        cls_name = str(cls_name)
        lines.append(f"{coords} {cls_name} {difficulty}")
        classes[cls_name] += 1
    return "\n".join(lines) + "\n", len(lines), classes


def _copy_zip_member(zf: zipfile.ZipFile, member: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def main() -> None:
    args = parse_args()
    image_zip = Path(args.image_zip)
    label_zip = Path(args.label_zip)
    out_root = Path(args.out_root)
    out_img_dir = out_root / "images"
    out_pkl_dir = out_root / "Step6_Format_labels"
    out_txt_dir = out_root / "labelTxt"
    started = time.time()

    image_members, image_dupes = _stem_to_member(image_zip, IMAGE_EXTS)
    label_members, label_dupes = _stem_to_member(label_zip, {".pkl"})
    common_stems = sorted(set(image_members) & set(label_members))
    if args.limit > 0:
        common_stems = common_stems[:args.limit]

    summary = {
        "image_zip": str(image_zip),
        "label_zip": str(label_zip),
        "out_root": str(out_root),
        "image_members": len(image_members),
        "label_members": len(label_members),
        "common_stems": len(common_stems),
        "image_without_label": len(set(image_members) - set(label_members)),
        "label_without_image": len(set(label_members) - set(image_members)),
        "image_duplicate_stems": len(image_dupes),
        "label_duplicate_stems": len(label_dupes),
        "dry_run": bool(args.dry_run),
        "limit": int(args.limit),
    }

    if args.dry_run:
        summary["elapsed_sec"] = round(time.time() - started, 3)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_pkl_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_existing = 0
    objects = 0
    class_counter: Counter[str] = Counter()
    errors: list[str] = []

    with zipfile.ZipFile(image_zip) as izf, zipfile.ZipFile(label_zip) as lzf:
        for stem in common_stems:
            img_member = image_members[stem]
            img_suffix = Path(img_member).suffix
            out_img = out_img_dir / f"{stem}{img_suffix}"
            out_pkl = out_pkl_dir / f"{stem}.pkl"
            out_txt = out_txt_dir / f"{stem}.txt"

            if out_img.exists() and out_pkl.exists() and out_txt.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            try:
                label_bytes = lzf.read(label_members[stem])
                txt, obj_count, classes = _label_to_txt(label_bytes, args.difficulty)
                _copy_zip_member(izf, img_member, out_img)
                out_pkl.write_bytes(label_bytes)
                out_txt.write_text(txt, encoding="utf-8")
                converted += 1
                objects += obj_count
                class_counter.update(classes)
            except Exception as exc:  # noqa: BLE001 - keep rebuilding other samples.
                errors.append(f"{stem}: {exc}")
                if len(errors) >= 20:
                    break

    summary.update(
        {
            "converted": converted,
            "skipped_existing": skipped_existing,
            "objects": objects,
            "classes": len(class_counter),
            "class_counts": dict(class_counter.most_common()),
            "errors": errors,
            "out_images": str(out_img_dir),
            "out_pkl_labels": str(out_pkl_dir),
            "out_txt_labels": str(out_txt_dir),
            "elapsed_sec": round(time.time() - started, 3),
        }
    )
    summary_path = out_root / "rebuild_labeled_subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
