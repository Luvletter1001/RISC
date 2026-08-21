#!/usr/bin/env python
"""Select top SISE cases and dataset indices for targeted hook runs."""

import argparse
import csv
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localized-records-csv", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-indices", required=True)
    parser.add_argument("--pairs", default="small-vehicle->plane,small-vehicle->tennis-court")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--metric-z-thr", type=float, default=4.0)
    parser.add_argument("--score-thr", type=float, default=0.999)
    return parser.parse_args()


def get_img_id(sample):
    if isinstance(sample, dict):
        if sample.get("img_id") is not None:
            return str(sample["img_id"])
        metainfo = sample.get("metainfo", {}) or {}
        return str(metainfo.get("img_id"))
    return str(getattr(sample, "img_id"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    pairs = {item for item in args.pairs.split(",") if item.strip()}
    with Path(args.predictions).open("rb") as f:
        predictions = pickle.load(f)
    img_to_index = {get_img_id(sample): idx for idx, sample in enumerate(predictions)}
    candidates = []
    for row in read_csv(args.localized_records_csv):
        if row["pair"] not in pairs:
            continue
        if row["correct"] != "False":
            continue
        if row["is_sibling"] == "True":
            continue
        if float(row["log_area_z"]) < args.metric_z_thr:
            continue
        if float(row["score"]) < args.score_thr:
            continue
        image_index = img_to_index.get(row["img_id"])
        if image_index is None:
            continue
        item = dict(row)
        item["dataset_index"] = image_index
        candidates.append(item)
    candidates.sort(
        key=lambda row: (float(row["score"]), float(row["log_area_z"])),
        reverse=True)
    selected = []
    seen_images = set()
    for row in candidates:
        if row["img_id"] in seen_images:
            continue
        selected.append(row)
        seen_images.add(row["img_id"])
        if len(selected) >= args.limit:
            break
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, selected)
    Path(args.out_indices).write_text(
        "\n".join(str(row["dataset_index"]) for row in selected) + "\n",
        encoding="utf-8")
    print({
        "selected": len(selected),
        "out_csv": args.out_csv,
        "out_indices": args.out_indices,
        "indices": [row["dataset_index"] for row in selected],
    })


if __name__ == "__main__":
    main()
