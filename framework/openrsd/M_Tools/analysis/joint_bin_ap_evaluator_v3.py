#!/usr/bin/env python3
"""Joint-bin AP evaluator v3 placeholder with explicit availability status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "status": "IMPLEMENTED_INTERFACE",
        "definition_A": "GT-bin AP: GT binned by geometry, predictions remain full set.",
        "definition_B": "Prediction-GT joint-bin AP: GT and predictions are both binned and only same-bin matches count.",
        "note": "This auxiliary evaluator is intentionally not used to mark GPU-heavy tasks done.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text("# G6 Joint-bin AP Evaluator\n\n- status: `IMPLEMENTED_INTERFACE`\n- This is a background auxiliary task and cannot replace live GPU evidence.\n", encoding="utf-8")


if __name__ == "__main__":
    main()

