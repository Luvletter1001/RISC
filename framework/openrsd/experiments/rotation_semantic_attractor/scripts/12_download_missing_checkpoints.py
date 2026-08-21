#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from common import add_common_args, exp_path, read_jsonish
from experiments.rotation_semantic_attractor.src.utils.io import write_csv


def _download(url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=30) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(target)
    return "DOWNLOADED"


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--model-registry", default=str(exp_path("configs", "model_registry.yaml")))
    parser.add_argument("--models", default="roi_trans,s2anet,rotated_fcos,rotated_atss")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    registry = read_jsonish(args.model_registry)["models"]
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = []
    for model_key in selected:
        cfg = registry[model_key]
        checkpoint = Path(cfg["checkpoint"])
        url = cfg.get("checkpoint_url", "")
        status = "AVAILABLE" if checkpoint.exists() else "MISSING"
        reason = ""
        if status == "MISSING" and not url:
            reason = "missing_checkpoint_url"
        elif status == "MISSING" and not args.check_only:
            try:
                status = _download(url, checkpoint)
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                status = "FAILED"
        rows.append(
            {
                "model_name": model_key,
                "checkpoint": str(checkpoint),
                "checkpoint_url": url,
                "exists": checkpoint.exists(),
                "status": status,
                "reason": reason,
            }
        )
    out = Path(args.output_dir or exp_path("outputs", "inventory")) / "checkpoint_download_status.csv"
    write_csv(out, rows)
    print(f"checkpoint_status={out}")
    if any(row["status"] == "FAILED" for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
