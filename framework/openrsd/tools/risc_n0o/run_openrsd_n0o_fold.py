#!/usr/bin/env python3
"""Authorization guard for the future OpenRSD N0-O GPU fold runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


V3_MANIFEST_SHA256 = (
    '646b8702d60a3dbe36a35871b5599459807b8399deafac98d3056a1f04d8351c')
V3_SCENE_PLAN_SHA256 = (
    '0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35')
V3_SUPPORT_LEDGER_SHA256 = (
    '5cb1efb1f5bf65bb701a5b99daeded0751e2498961180d3f86ff2d9a80bbaa61')
FOLD_IDS = frozenset({'c4_a', 'c4_b', 'c8_a', 'c8_b'})
MAX_SCENES_PER_FOLD = 40
PREFLIGHT_ARTIFACT_NAMES = frozenset({
    'model_ledger.jsonl',
    'module_origins.json',
    'preflight_report.json',
    'resolved_config.py',
})
COMMITTED_PREFLIGHT_RECEIPT = Path(
    '/data1/zcy/RISC/docs/provenance/risc_openrsd_n0o_preflight_v2/'
    'PREFLIGHT_READY_GPU_NOT_AUTHORIZED.json')


class AuthorizationError(RuntimeError):
    """Raised before lazy runtime imports when GPU use is unauthorized."""


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True).encode('utf-8') + b'\n'


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _committed_preflight_sha256() -> str:
    path = Path(COMMITTED_PREFLIGHT_RECEIPT)
    if not path.is_file():
        raise AuthorizationError('committed preflight receipt is missing')
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorizationError('committed preflight JSON is invalid') from error
    if raw != _canonical_json_bytes(value):
        raise AuthorizationError('committed preflight receipt is not canonical')
    if (value.get('schema') != 'risc-openrsd-n0o-preflight-receipt-v2'
            or value.get('status') != 'PREFLIGHT_READY_GPU_NOT_AUTHORIZED'
            or value.get('gpu_smoke_authorized') is not False):
        raise AuthorizationError('committed preflight receipt is invalid')
    artifacts = value.get('artifacts')
    if not isinstance(artifacts, dict) or set(
            artifacts) != PREFLIGHT_ARTIFACT_NAMES:
        raise AuthorizationError('committed preflight artifact set is invalid')
    for name, record in sorted(artifacts.items()):
        artifact_path = path.parent / name
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise AuthorizationError(
                'committed preflight artifact is missing')
        if (not isinstance(record, dict)
                or type(record.get('byte_count')) is not int
                or record['byte_count'] <= 0
                or not isinstance(record.get('sha256'), str)
                or len(record['sha256']) != 64):
            raise AuthorizationError(
                'committed preflight artifact record is invalid')
        if (artifact_path.stat().st_size != record['byte_count']
                or _sha256_file(artifact_path) != record['sha256']):
            raise AuthorizationError(
                'committed preflight artifact hash mismatch')
    report_path = path.parent / 'preflight_report.json'
    report_raw = report_path.read_bytes()
    try:
        report = json.loads(report_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorizationError('preflight report JSON is invalid') from error
    if report_raw != _canonical_json_bytes(report):
        raise AuthorizationError('preflight report is not canonical')
    if report.get('schema') != 'risc-openrsd-n0o-preflight-report-v2':
        raise AuthorizationError('preflight report schema is invalid')
    sources = report.get('sources')
    source_paths = {
        'preflight_source': Path(__file__).with_name(
            'preflight_openrsd_n0o.py'),
        'protocol_source': Path(__file__).with_name(
            'openrsd_n0o_protocol.py'),
        'runner_source': Path(__file__).resolve(),
    }
    if not isinstance(sources, dict):
        raise AuthorizationError('preflight source chain is invalid')
    for name, source_path in source_paths.items():
        if sources.get(name) != _sha256_file(source_path):
            raise AuthorizationError('preflight source chain mismatch')
    if (sources.get('input_manifest') != V3_MANIFEST_SHA256
            or sources.get('scene_plan') != V3_SCENE_PLAN_SHA256
            or sources.get('support_ledger') != V3_SUPPORT_LEDGER_SHA256):
        raise AuthorizationError('preflight input source chain mismatch')
    return hashlib.sha256(raw).hexdigest()


def load_authorization(
        path: Path | str,
        *,
        fold_id: str,
        requested_scenes: int):
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
    if value.get('preflight_receipt_sha256') != _committed_preflight_sha256():
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
    if limit > MAX_SCENES_PER_FOLD:
        raise AuthorizationError('authorized scene bound cannot exceed 40')
    if type(requested_scenes) is not int or requested_scenes <= 0:
        raise AuthorizationError('requested scene count must be positive')
    if requested_scenes > MAX_SCENES_PER_FOLD:
        raise AuthorizationError('requested scene count cannot exceed 40')
    if requested_scenes > limit:
        raise AuthorizationError('requested scene count exceeds authorization')
    return value


def run_authorized_fold(
        *,
        authorization: Path | str,
        fold_id: str,
        requested_scenes: int,
        lazy_runtime_loader):
    load_authorization(
        authorization,
        fold_id=fold_id,
        requested_scenes=requested_scenes)
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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_authorized_fold(
        authorization=args.authorization,
        fold_id=args.fold_id,
        requested_scenes=args.requested_scenes,
        lazy_runtime_loader=_unavailable_runtime)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
