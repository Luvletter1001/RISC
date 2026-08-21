#!/usr/bin/env python
"""Collapse all classes in DOTA-format annfiles to one class name."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite DOTA txt annotations with one target class name.")
    parser.add_argument("--ann-dir", required=True, help="Input DOTA annfiles dir.")
    parser.add_argument("--out-dir", required=True, help="Output annfiles dir.")
    parser.add_argument("--target-class", default="ship")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files.")
    return parser.parse_args()


def collapse_file(src: Path, dst: Path, target_class: str) -> tuple[int, int]:
    converted = 0
    skipped = 0
    lines_out: list[str] = []
    for raw in src.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 9:
            skipped += 1
            continue
        parts[8] = target_class
        lines_out.append(" ".join(parts))
        converted += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""),
                   encoding="utf-8")
    return converted, skipped


def main() -> None:
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ann_dir.glob("*.txt"))
    converted_files = 0
    converted_objects = 0
    skipped_lines = 0
    skipped_existing = 0
    for src in files:
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        objects, skipped = collapse_file(src, dst, args.target_class)
        converted_files += 1
        converted_objects += objects
        skipped_lines += skipped

    print(f"ann_dir: {ann_dir}")
    print(f"out_dir: {out_dir}")
    print(f"converted_files: {converted_files}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"converted_objects: {converted_objects}")
    print(f"skipped_lines: {skipped_lines}")


if __name__ == "__main__":
    main()
