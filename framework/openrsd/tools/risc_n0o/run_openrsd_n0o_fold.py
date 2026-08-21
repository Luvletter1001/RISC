#!/usr/bin/env python3
"""Authorization guard for the future OpenRSD N0-O GPU fold runtime."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


V3_MANIFEST_SHA256 = (
    '646b8702d60a3dbe36a35871b5599459807b8399deafac98d3056a1f04d8351c')
FOLD_IDS = frozenset({'c4_a', 'c4_b', 'c8_a', 'c8_b'})
SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')


class AuthorizationError(RuntimeError):
    """Raised before lazy runtime imports when GPU use is unauthorized."""


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True).encode('utf-8') + b'\n'


def load_authorization(
        path: Path | str,
        *,
        fold_id: str,
        requested_scenes: int,
        expected_preflight_sha256: str):
    path = Path(path)
    if not path.is_file():
        raise AuthorizationError('GPU authorization receipt is missing')
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorizationError('authorization JSON is invalid') from error
    if raw != _canonical_json_bytes(value):
        raise AuthorizationError('authorization receipt is not canonical')
    if value.get('status') != 'GPU_SMOKE_AUTHORIZED':
        raise AuthorizationError('authorization status is invalid')
    if value.get('input_manifest_sha256') != V3_MANIFEST_SHA256:
        raise AuthorizationError('authorization input manifest is invalid')
    if (not isinstance(expected_preflight_sha256, str)
            or SHA256_PATTERN.fullmatch(expected_preflight_sha256) is None
            or value.get('preflight_receipt_sha256')
            != expected_preflight_sha256):
        raise AuthorizationError('authorization preflight identity is invalid')
    allowed_folds = value.get('allowed_folds')
    if (not isinstance(allowed_folds, list)
            or not allowed_folds
            or any(item not in FOLD_IDS for item in allowed_folds)
            or len(allowed_folds) != len(set(allowed_folds))):
        raise AuthorizationError('authorization fold list is invalid')
    if fold_id not in allowed_folds:
        raise AuthorizationError('requested fold is not authorized')
    limit = value.get('max_scenes_per_fold')
    if type(limit) is not int or limit <= 0:
        raise AuthorizationError('authorized scene bound must be positive')
    if type(requested_scenes) is not int or requested_scenes <= 0:
        raise AuthorizationError('requested scene count must be positive')
    if requested_scenes > limit:
        raise AuthorizationError('requested scene count exceeds authorization')
    return value


def run_authorized_fold(
        *,
        authorization: Path | str,
        fold_id: str,
        requested_scenes: int,
        expected_preflight_sha256: str,
        lazy_runtime_loader):
    load_authorization(
        authorization,
        fold_id=fold_id,
        requested_scenes=requested_scenes,
        expected_preflight_sha256=expected_preflight_sha256)
    runtime = lazy_runtime_loader()
    return runtime(fold_id, requested_scenes)


def _unavailable_runtime():
    raise AuthorizationError(
        'GPU runtime is not published in the current no-GPU phase')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--fold-id', choices=sorted(FOLD_IDS), required=True)
    parser.add_argument('--requested-scenes', type=int, required=True)
    parser.add_argument('--preflight-sha256', required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_authorized_fold(
        authorization=args.authorization,
        fold_id=args.fold_id,
        requested_scenes=args.requested_scenes,
        expected_preflight_sha256=args.preflight_sha256,
        lazy_runtime_loader=_unavailable_runtime)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
