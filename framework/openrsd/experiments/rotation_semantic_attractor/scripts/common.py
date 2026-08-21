from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_jsonish(path: str | Path):
    return json.loads(Path(path).read_text())


def parse_angles(value: str | None, default=None) -> list[int]:
    if value is None or value == "":
        return list(default or [])
    if "," in value:
        return [int(v) % 360 for v in value.split(",") if v.strip()]
    return [int(v) % 360 for v in value.split() if v.strip()]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--angles", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=20260530)
    return parser


def exp_path(*parts: str) -> Path:
    return EXP_DIR.joinpath(*parts)

