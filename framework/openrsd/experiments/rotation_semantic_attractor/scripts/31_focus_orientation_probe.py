#!/usr/bin/env python3
"""Probe FourierOrientationLearner on synthetic oriented textures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from M_AD.models.utils.focus_fourier_orientation import FourierOrientationLearner


DEFAULT_OUT_DIR = Path("resultmd/exp_focus_ovd_20260608/orientation_probe")


def texture(angle: float, size: int = 64, freq: float = 5.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size),
        torch.linspace(-1.0, 1.0, size),
        indexing="ij")
    coord = xx * math.cos(angle) + yy * math.sin(angle)
    img = torch.sin(2.0 * math.pi * freq * coord)
    return img[None, None]


def periodic_error(a: float, b: float) -> float:
    diff = abs((a - b + math.pi / 2.0) % math.pi - math.pi / 2.0)
    return float(diff)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    learner = FourierOrientationLearner(
        patch_size=9, num_angle_bins=36, harmonic_orders=(2, 4, 6))
    rows = []
    for deg in [0, 30, 60, 90, 120, 150]:
        gt = math.radians(deg)
        out = learner(texture(gt))
        theta = float(out["theta"][0, 32, 32].item())
        conf = float(out["confidence"][0, 32, 32].item())
        rows.append({
            "input_degrees": deg,
            "estimated_degrees": theta * 180.0 / math.pi,
            "periodic_error_degrees": periodic_error(theta, gt) * 180.0 / math.pi,
            "confidence": conf,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "orientation_probe.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "PROBE_WRITTEN",
        "csv": str(csv_path),
        "mean_periodic_error_degrees": sum(
            r["periodic_error_degrees"] for r in rows) / len(rows),
        "min_confidence": min(r["confidence"] for r in rows),
        "max_confidence": max(r["confidence"] for r in rows),
    }
    (args.out_dir / "orientation_probe_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# FOCUS-OVD Orientation Probe", "", "| input | estimated | error | confidence |",
          "| ---: | ---: | ---: | ---: |"]
    for r in rows:
        md.append(
            f"| {r['input_degrees']} | {r['estimated_degrees']:.2f} | "
            f"{r['periodic_error_degrees']:.2f} | {r['confidence']:.4f} |")
    (args.out_dir / "orientation_probe_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
