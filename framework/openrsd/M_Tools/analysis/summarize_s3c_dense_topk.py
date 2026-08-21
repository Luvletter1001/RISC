#!/usr/bin/env python
"""Summarize dense pre-NMS top-k rows dumped by S3C head calibration."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--focus-classes", default="small-vehicle,plane,tennis-court")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def summarize_rows(rows: list[dict], focus_classes: set[str]) -> dict:
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[str(row.get("stage"))].append(row)

    stage_summary = {}
    for stage, stage_rows in sorted(by_stage.items()):
        flagged = [row for row in stage_rows if row.get("s3c_flagged")]
        focus = [row for row in stage_rows if row.get("class_name") in focus_classes]
        z_values = [
            abs(float(row.get("log_area_z", 0.0) or 0.0))
            for row in stage_rows
        ]
        scores = [
            float(row.get("score", 0.0) or 0.0)
            for row in stage_rows
        ]
        stage_summary[stage] = {
            "rows": len(stage_rows),
            "images": len({row.get("image_id") for row in stage_rows}),
            "flagged_rows": len(flagged),
            "flagged_rate": len(flagged) / max(len(stage_rows), 1),
            "focus_rows": len(focus),
            "class_counts": Counter(row.get("class_name") for row in stage_rows),
            "flagged_class_counts": Counter(row.get("class_name") for row in flagged),
            "mean_score": mean(scores) if scores else 0.0,
            "mean_abs_log_area_z": mean(z_values) if z_values else 0.0,
            "top_flagged_examples": sorted(
                flagged,
                key=lambda row: (
                    -abs(float(row.get("log_area_z", 0.0) or 0.0)),
                    int(row.get("rank", 999999) or 999999),
                ),
            )[:20],
        }

    before = stage_summary.get("before", {})
    after = stage_summary.get("after", {})
    return {
        "total_rows": len(rows),
        "focus_classes": sorted(focus_classes),
        "stages": stage_summary,
        "delta": {
            "flagged_rows_before_minus_after": (
                int(before.get("flagged_rows", 0))
                - int(after.get("flagged_rows", 0))),
            "focus_rows_before_minus_after": (
                int(before.get("focus_rows", 0))
                - int(after.get("focus_rows", 0))),
            "mean_score_before_minus_after": (
                float(before.get("mean_score", 0.0))
                - float(after.get("mean_score", 0.0))),
        },
    }


def jsonable(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path: Path, payload: dict) -> None:
    lines = [
        "# S3C Dense Pre-NMS Top-K Summary",
        "",
        f"- total_rows: `{payload['total_rows']}`",
        f"- focus_classes: `{', '.join(payload['focus_classes'])}`",
        "",
        "| stage | rows | images | flagged_rows | flagged_rate | focus_rows | mean_score | mean_abs_log_area_z |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, item in payload["stages"].items():
        lines.append(
            f"| {stage} | {item['rows']} | {item['images']} | "
            f"{item['flagged_rows']} | {item['flagged_rate']:.4f} | "
            f"{item['focus_rows']} | {item['mean_score']:.6f} | "
            f"{item['mean_abs_log_area_z']:.4f} |")
    lines.extend([
        "",
        "## Delta",
        "",
        f"- flagged_rows_before_minus_after: `{payload['delta']['flagged_rows_before_minus_after']}`",
        f"- focus_rows_before_minus_after: `{payload['delta']['focus_rows_before_minus_after']}`",
        f"- mean_score_before_minus_after: `{payload['delta']['mean_score_before_minus_after']:.6f}`",
        "",
        "## Top Flagged Examples",
        "",
        "| stage | image_id | rank | class | score | log_area_z | area |",
        "|---|---|---:|---|---:|---:|---:|",
    ])
    for stage, item in payload["stages"].items():
        for row in item.get("top_flagged_examples", [])[:10]:
            lines.append(
                f"| {stage} | {row.get('image_id')} | {row.get('rank')} | "
                f"{row.get('class_name')} | {float(row.get('score', 0.0) or 0.0):.6f} | "
                f"{float(row.get('log_area_z', 0.0) or 0.0):.4f} | "
                f"{float(row.get('area', 0.0) or 0.0):.2f} |")
    path.write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def main():
    args = parse_args()
    rows = load_rows(Path(args.jsonl))
    focus_classes = {
        item.strip()
        for item in args.focus_classes.split(",")
        if item.strip()
    }
    payload = summarize_rows(rows, focus_classes)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    if args.out_md:
        write_md(Path(args.out_md), jsonable(payload))
    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    print(json.dumps({
        "out_json": str(out_json),
        "total_rows": payload["total_rows"],
        "delta": payload["delta"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
