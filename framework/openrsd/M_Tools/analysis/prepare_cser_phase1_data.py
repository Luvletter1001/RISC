#!/usr/bin/env python3
"""Prepare source-disjoint runtime views for the CSER Phase-1 gate."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Iterable


DOTA1_CLASSES = [
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "ground-track-field",
    "harbor",
    "helicopter",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
]


def select_source_stems(
        image_names: Iterable[str],
        archive_names: Iterable[str],
        heldout_stems: set[str]) -> list[str]:
    """Select sorted stems that have both PNG image and PKL annotation."""
    image_stems = {
        Path(name).stem for name in image_names
        if Path(name).suffix.lower() == ".png"
    }
    pkl_stems = {
        Path(name).stem for name in archive_names
        if Path(name).suffix.lower() == ".pkl"
    }
    return sorted((image_stems & pkl_stems) - set(heldout_stems))


def select_support_classes(support_data: dict, classes: list[str]) -> dict:
    """Return an exact ordered support subset and reject missing classes."""
    missing = [class_name for class_name in classes
               if class_name not in support_data]
    if missing:
        raise KeyError(f"support data is missing classes: {missing}")
    return {class_name: support_data[class_name] for class_name in classes}


def _heldout_stems(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["tiles"]
    return [str(row["tile_id"]) for row in rows]


def _index_files(directories: list[Path], suffix: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob(f"*{suffix}")):
            index.setdefault(path.stem, path.resolve())
    return index


def _link_once(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(
            f"existing symlink has different target: {destination}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(source.resolve())


def _extract_selected_pkls(
        archive_path: Path,
        selected_stems: list[str],
        output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = {
            Path(name).stem: name
            for name in archive.namelist()
            if Path(name).suffix.lower() == ".pkl"
        }
        for stem in selected_stems:
            member = members[stem]
            destination = output_dir / f"{stem}.pkl"
            expected_size = archive.getinfo(member).file_size
            if destination.is_file() and destination.stat().st_size == expected_size:
                extracted += 1
                continue
            temporary = destination.with_suffix(".pkl.tmp")
            with archive.open(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
            extracted += 1
    return extracted


def _write_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> dict:
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    heldout_order = _heldout_stems(args.heldout_json)
    heldout = set(heldout_order)

    image_names = [path.name for path in args.train_image_dir.glob("*.png")]
    with zipfile.ZipFile(args.formatted_zip) as archive:
        archive_names = archive.namelist()
    selected = select_source_stems(image_names, archive_names, heldout)
    train_ann_dir = out_root / "train_formatted_pkls"
    extracted = _extract_selected_pkls(
        args.formatted_zip, selected, train_ann_dir)

    image_index = _index_files(args.val_image_dir, ".png")
    ann_index = _index_files(args.val_ann_dir, ".txt")
    val_root = out_root / "s2_val_view"
    val_image_dir = val_root / "images"
    val_ann_dir = val_root / "annfiles"
    val_image_dir.mkdir(parents=True, exist_ok=True)
    val_ann_dir.mkdir(parents=True, exist_ok=True)
    recoverable = []
    missing_images = []
    missing_annotations = []
    for stem in heldout_order:
        image_path = image_index.get(stem)
        ann_path = ann_index.get(stem)
        if image_path is None:
            missing_images.append(stem)
        if ann_path is None:
            missing_annotations.append(stem)
        if image_path is None or ann_path is None:
            continue
        _link_once(image_path, val_image_dir / f"{stem}.png")
        _link_once(ann_path, val_ann_dir / f"{stem}.txt")
        recoverable.append(stem)

    with args.support_pkl.open("rb") as handle:
        support_data = pickle.load(handle)
    classes = list(support_data)
    runtime_meta = out_root / "runtime_meta"
    runtime_meta.mkdir(parents=True, exist_ok=True)
    dota1_support_path = runtime_meta / "dota1_visual_text_support.pkl"
    _write_pickle(
        dota1_support_path,
        select_support_classes(support_data, DOTA1_CLASSES),
    )
    _write_pickle(
        runtime_meta / "normalized_class_dict.pkl",
        {class_name: class_name for class_name in classes},
    )
    _write_pickle(
        runtime_meta / "neg_supports_unused.pkl",
        {"neg_dict": {
            "Data1_DOTA2": {class_name: [] for class_name in classes},
            "Data1_DOTA1": {class_name: [] for class_name in DOTA1_CLASSES},
        }},
    )
    _write_pickle(runtime_meta / "pca_meta_unused.pkl", {})

    payload = {
        "status": "DONE",
        "train_image_dir": str(args.train_image_dir.resolve()),
        "formatted_zip": str(args.formatted_zip.resolve()),
        "train_formatted_pkl_dir": str(train_ann_dir),
        "train_selected_count": len(selected),
        "train_extracted_count": extracted,
        "heldout_count": len(heldout),
        "train_heldout_intersection_count": len(set(selected) & heldout),
        "val_recoverable_count": len(recoverable),
        "val_image_dir": str(val_image_dir),
        "val_ann_dir": str(val_ann_dir),
        "missing_image_count": len(missing_images),
        "missing_annotation_count": len(missing_annotations),
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "support_pkl": str(args.support_pkl.resolve()),
        "support_classes": classes,
        "dota1_support_pkl": str(dota1_support_path),
        "dota1_support_classes": DOTA1_CLASSES,
        "normalized_class_dict": str(
            runtime_meta / "normalized_class_dict.pkl"),
        "neg_support_data": str(runtime_meta / "neg_supports_unused.pkl"),
        "pca_meta": str(runtime_meta / "pca_meta_unused.pkl"),
    }
    (out_root / "prepare_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-image-dir", type=Path, required=True)
    parser.add_argument("--formatted-zip", type=Path, required=True)
    parser.add_argument("--heldout-json", type=Path, required=True)
    parser.add_argument("--support-pkl", type=Path, required=True)
    parser.add_argument(
        "--val-image-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--val-ann-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = prepare(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
