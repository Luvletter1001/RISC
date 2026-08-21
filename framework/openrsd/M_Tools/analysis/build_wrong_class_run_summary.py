#!/usr/bin/env python
"""Build one-row CSV summaries for wrong-class DOTAV2 runs."""

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "model",
    "config",
    "checkpoint",
    "output_dir",
    "mAP",
    "AP50",
    "images_scanned",
    "detections_checked",
    "gt_objects_checked",
    "localized_correct_class",
    "wrong_class_iou_gt0p7",
    "jsonl",
    "csv",
    "summary_json",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wrong-summary-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--eval-json", default="")
    return parser.parse_args()


def format_metric(value):
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def latest_eval_json(output_dir):
    candidates = sorted((Path(output_dir) / "test_work").glob("*/*.json"))
    if not candidates:
        raise FileNotFoundError(f"no eval json found under {output_dir}/test_work")
    return candidates[-1]


def read_eval_metrics(output_dir, eval_json=""):
    path = Path(eval_json) if eval_json else latest_eval_json(output_dir)
    data = json.loads(path.read_text())
    try:
        return format_metric(data["dota/mAP"]), format_metric(data["dota/AP50"])
    except KeyError as exc:
        raise KeyError(f"missing metric {exc} in {path}") from exc


def build_summary_row(model, config, checkpoint, output_dir, wrong_summary_json,
                      eval_json=""):
    output_dir = Path(output_dir)
    wrong_summary_json = Path(wrong_summary_json)
    summary = json.loads(wrong_summary_json.read_text())
    stats = summary.get("stats", {})
    map_value, ap50_value = read_eval_metrics(output_dir, eval_json)
    return {
        "model": model,
        "config": config,
        "checkpoint": checkpoint,
        "output_dir": str(output_dir),
        "mAP": map_value,
        "AP50": ap50_value,
        "images_scanned": str(int(stats.get("images_scanned", 0))),
        "detections_checked": str(int(stats.get("detections_checked", 0))),
        "gt_objects_checked": str(int(stats.get("gt_objects_checked", 0))),
        "localized_correct_class": str(
            int(stats.get("localized_correct_class", 0))
        ),
        "wrong_class_iou_gt0p7": str(
            int(summary.get("selected_count", stats.get("localized_wrong_class", 0)))
        ),
        "jsonl": str(summary.get("out_jsonl", output_dir / "wrong_class_iou_gt0p7.jsonl")),
        "csv": str(summary.get("out_csv", output_dir / "wrong_class_iou_gt0p7.csv")),
        "summary_json": str(wrong_summary_json),
    }


def write_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    row = build_summary_row(
        model=args.model,
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        wrong_summary_json=args.wrong_summary_json,
        eval_json=args.eval_json,
    )
    write_summary_csv(args.out_csv, [row])
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
