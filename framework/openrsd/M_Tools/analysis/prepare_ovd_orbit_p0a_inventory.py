"""Publish a P0-A diagnostic inventory package from sealed CPU-only inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from M_Tools.analysis import ovd_orbit_p0a_inventory as inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-scene-plan', required=True, type=Path)
    parser.add_argument('--source-plan-sha256', required=True)
    parser.add_argument('--source-input-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    return parser


def _print_receipt(artifacts: dict[str, bytes]) -> None:
    print(artifacts['receipt.json'].decode('utf-8'), end='')


def main(argv: Sequence[str] | None = None) -> int:
    """Publish a no-forward diagnostic package or a minimal P0 input failure."""
    args = _parser().parse_args(argv)
    try:
        source_plan = inventory.load_canonical_json(
            args.source_scene_plan, label='source scene plan')
        source_input_manifest = inventory.load_canonical_json(
            args.source_input_manifest, label='source input manifest')
        try:
            rehashed_source_plan = inventory.sha256_file(args.source_scene_plan)
        except OSError as error:
            raise inventory.P0AInventoryError(
                'source scene plan input rehash failed') from error
        if rehashed_source_plan != source_plan.source_sha256:
            raise inventory.P0AInventoryError(
                'source scene plan hash does not match loaded snapshot')
        if args.source_plan_sha256 != source_plan.source_sha256:
            raise inventory.P0AInventoryError(
                'source_plan_sha256 must match loaded source scene plan')
        artifacts = inventory.build_inventory_artifacts(
            source_plan=source_plan,
            source_plan_sha256=source_plan.source_sha256,
            source_input_manifest=source_input_manifest,
        )
    except inventory.P0AInventoryError as error:
        artifacts = inventory.build_inventory_failure_artifacts(error)
        inventory.publish_inventory_artifacts(args.output_dir, artifacts)
        _print_receipt(artifacts)
        return 2

    inventory.publish_inventory_artifacts(args.output_dir, artifacts)
    _print_receipt(artifacts)
    receipt = json.loads(artifacts['receipt.json'])
    return 0 if receipt['status'] == 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD' else 2


if __name__ == '__main__':
    raise SystemExit(main())
