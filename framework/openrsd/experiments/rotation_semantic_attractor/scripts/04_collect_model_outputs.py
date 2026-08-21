#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args
from experiments.rotation_semantic_attractor.src.utils.io import write_json


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    files = sorted(str(p) for p in (run_dir / "canonical_predictions").glob("*/*/*.json"))
    out = Path(args.output_dir or run_dir / "metrics") / "collected_outputs.json"
    write_json(out, {"num_prediction_files": len(files), "prediction_files": files[:100]})
    print(f"collected_outputs={out}")


if __name__ == "__main__":
    main()

