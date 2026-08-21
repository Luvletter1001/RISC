#!/usr/bin/env python
"""Extract structured AP metrics from an mmengine test log."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def extract_metrics(log_text: str) -> dict:
    pattern = re.compile(
        r"Epoch\(test\)\s+\[\d+/\d+\]\s+"
        r"dota/mAP:\s+(?P<map>[0-9.]+)\s+"
        r"dota/AP50:\s+(?P<ap50>[0-9.]+)\s+"
        r"dota/IoU_50_Detail:\s+(?P<detail>\{.*?\})\s+"
        r"data_time:",
        re.DOTALL,
    )
    matches = list(pattern.finditer(log_text))
    if not matches:
        raise ValueError("No final Epoch(test) metric line found in run log")
    match = matches[-1]
    detail = ast.literal_eval(match.group("detail"))
    return {
        "dota/mAP": float(match.group("map")),
        "dota/AP50": float(match.group("ap50")),
        "dota/IoU_50_Detail": detail,
    }


def write_markdown(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    detail = metrics["dota/IoU_50_Detail"]
    lines = [
        "# Prediction Metric JSON Eval",
        "",
        f"- source: `mmengine run log`",
        f"- run_log: `{payload['run_log']}`",
        f"- predictions: `{payload['predictions']}`",
        f"- mAP: `{metrics['dota/mAP']}`",
        f"- AP50: `{metrics['dota/AP50']}`",
        f"- small-vehicle_AP50: `{detail.get('small-vehicle', {}).get('ap')}`",
        "",
        "| class | AP50 | recall | num_dets | num_gts |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in sorted(detail):
        info = detail[class_name]
        lines.append(
            f"| {class_name} | {info.get('ap')} | {info.get('recall')} | "
            f"{info.get('num_dets')} | {info.get('num_gts')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = Path(args.run_log)
    metrics = extract_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
    payload = {
        "config": args.config,
        "predictions": args.predictions,
        "run_log": str(log_path),
        "metric_source": "tools/test.py mmengine log",
        "metrics": metrics,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8",
    )
    write_markdown(Path(args.out_md), payload)
    print(json.dumps({
        "out_json": str(out_json),
        "mAP": metrics["dota/mAP"],
        "AP50": metrics["dota/AP50"],
        "small_vehicle_AP50": metrics["dota/IoU_50_Detail"]
        .get("small-vehicle", {})
        .get("ap"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
