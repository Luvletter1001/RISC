"""Publish a P0-A diagnostic authority package from explicit CPU-only inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from M_Tools.analysis import ovd_orbit_p0a_authority as authority


class _SinglePath(argparse.Action):
    """Reject duplicated scalar path flags before any authority input is read."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f'{option_string} may be supplied only once')
        setattr(namespace, self.dest, values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--inventory-receipt', required=True, type=Path,
                        action=_SinglePath)
    parser.add_argument('--inventory-diagnostics', required=True, type=Path,
                        action=_SinglePath)
    parser.add_argument('--object-inventory', required=True, type=Path,
                        action=_SinglePath)
    parser.add_argument('--source-input-manifest', required=True, type=Path,
                        action=_SinglePath)
    parser.add_argument('--support-asset', required=True, type=Path,
                        action=_SinglePath)
    parser.add_argument('--oracle-code-file', action='append', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path, action=_SinglePath)
    return parser


def _oracle_code_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    code_paths = tuple(args.oracle_code_file)
    if len(code_paths) != 3:
        raise authority.P0AAuthorityError(
            '--oracle-code-file must be supplied exactly three times')
    return code_paths


def _print_receipt(artifacts: dict[str, bytes]) -> None:
    print(artifacts['receipt.json'].decode('utf-8'), end='')


def main(argv: Sequence[str] | None = None) -> int:
    """Publish authority success or the minimal input-failure package."""
    args = _parser().parse_args(argv)
    try:
        code_paths = _oracle_code_paths(args)
        artifacts = authority.build_authority_artifacts(
            inventory_receipt_path=args.inventory_receipt,
            inventory_diagnostics_path=args.inventory_diagnostics,
            object_inventory_path=args.object_inventory,
            source_input_manifest_path=args.source_input_manifest,
            support_asset_path=args.support_asset,
            oracle_code_files=code_paths,
        )
        consumed = authority.consumed_input_digest_ledger(artifacts)
        rehashed = authority.rehash_authority_input_ledger(
            inventory_receipt_path=args.inventory_receipt,
            inventory_diagnostics_path=args.inventory_diagnostics,
            object_inventory_path=args.object_inventory,
            source_input_manifest_path=args.source_input_manifest,
            support_asset_path=args.support_asset,
            oracle_code_files=code_paths,
        )
        if rehashed != consumed:
            raise authority.P0AAuthorityError(
                'authority inputs changed after consumed snapshot')
    except authority.P0AAuthorityError as error:
        artifacts = authority.build_authority_failure_artifacts(error)
        authority.publish_authority_artifacts(args.output_dir, artifacts)
        _print_receipt(artifacts)
        return 2
    authority.publish_authority_artifacts(args.output_dir, artifacts)
    _print_receipt(artifacts)
    receipt = json.loads(artifacts['receipt.json'])
    return 0 if receipt['status'] == 'P0A_DIAGNOSTIC_AUTHORITY_READY_NO_FORWARD' else 2


if __name__ == '__main__':
    raise SystemExit(main())
