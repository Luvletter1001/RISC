"""CPU-only CLI for validating and serializing OVD object-orbit P0 records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from M_Tools.analysis.ovd_orbit_p0 import (
    OrbitP0Error,
    ScoreCarrier,
    build_manifest,
    collapse_carriers,
    write_receipt,
)


def _carrier_from_record(value: Any, *, line_number: int) -> ScoreCarrier:
    if not isinstance(value, dict):
        raise OrbitP0Error(f'line {line_number} must contain a JSON object')
    required = {'view_id', 'scene_id', 'source', 'box', 'scores'}
    missing = sorted(required - set(value))
    if missing:
        raise OrbitP0Error(f'line {line_number} is missing {",".join(missing)}')
    source = value['source']
    if not isinstance(source, list) or len(source) != 2:
        raise OrbitP0Error(f'line {line_number} source must be a two-item list')
    return ScoreCarrier(
        view_id=value['view_id'],
        scene_id=value['scene_id'],
        source=(source[0], source[1]),
        box=np.asarray(value['box'], dtype=np.float32),
        scores=np.asarray(value['scores'], dtype=np.float32),
        calibrated_scores=(
            None if value.get('calibrated_scores') is None
            else np.asarray(value['calibrated_scores'], dtype=np.float32)),
    )


def load_carriers(path: Path | str) -> tuple[ScoreCarrier, ...]:
    source = Path(path)
    if not source.is_file():
        raise OrbitP0Error(f'input JSONL does not exist: {source}')
    carriers = []
    for line_number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OrbitP0Error(f'line {line_number} is not valid JSON') from exc
        carriers.append(_carrier_from_record(value, line_number=line_number))
    return collapse_carriers(carriers)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-jsonl', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--vocabulary-hash', required=True)
    parser.add_argument('--prompt-hash', required=True)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    carriers = load_carriers(args.input_jsonl)
    manifest = build_manifest(
        carriers,
        vocabulary_hash=args.vocabulary_hash,
        prompt_hash=args.prompt_hash,
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    receipt = write_receipt(args.output_dir, manifest, carriers)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
