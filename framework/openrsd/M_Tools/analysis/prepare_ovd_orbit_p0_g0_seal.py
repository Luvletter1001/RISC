"""Publish a deterministic P0 G0 input-seal package from explicit paths."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from M_Tools.analysis import ovd_orbit_p0_g0 as g0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-json", required=True, type=Path)
    parser.add_argument("--candidate-scene-plan", required=True, type=Path)
    parser.add_argument("--object-inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _print_receipt(artifacts: dict[str, bytes]) -> None:
    print(artifacts["receipt.json"].decode("utf-8"), end="")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and publish one complete input package or a fail-stop receipt."""
    args = _parser().parse_args(argv)
    try:
        authority = g0.validate_authority(g0.load_canonical_json(
            args.authority_json, label="authority"))
        candidate_plan = g0.load_canonical_json(
            args.candidate_scene_plan, label="candidate scene plan")
        object_rows = g0.load_object_rows(args.object_inventory)
        try:
            candidate_file_sha256 = g0.sha256_file(args.candidate_scene_plan)
        except FileExistsError:
            raise
        except OSError as error:
            raise g0.G0SealError(
                "candidate scene plan input rehash failed") from error
        try:
            object_file_sha256 = g0.sha256_file(args.object_inventory)
        except FileExistsError:
            raise
        except OSError as error:
            raise g0.G0SealError(
                "object inventory input rehash failed") from error
        candidate_plan_sha256 = candidate_plan.source_sha256
        object_inventory_sha256 = object_rows.source_sha256
        if candidate_file_sha256 != candidate_plan_sha256:
            raise g0.G0SealError(
                "candidate scene plan hash does not match loaded snapshot")
        if object_file_sha256 != object_inventory_sha256:
            raise g0.G0SealError(
                "object inventory hash does not match loaded snapshot")
        asset_hashes = g0.verify_authority_assets(authority)
        candidates = g0.validate_candidate_plan(candidate_plan)
        candidate_asset_hashes = g0.verify_candidate_assets(candidates)
        artifacts = g0.build_g0_artifacts(
            authority=authority,
            candidate_plan=candidate_plan,
            object_rows=object_rows,
            candidate_plan_sha256=candidate_plan_sha256,
            object_inventory_sha256=object_inventory_sha256,
            asset_hashes=asset_hashes,
            candidate_asset_hashes=candidate_asset_hashes,
        )
    except g0.G0SealError as error:
        artifacts = g0.build_failure_artifacts(error)
        g0.publish_artifacts(args.output_dir, artifacts)
        _print_receipt(artifacts)
        return 2
    g0.publish_artifacts(args.output_dir, artifacts)
    _print_receipt(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
