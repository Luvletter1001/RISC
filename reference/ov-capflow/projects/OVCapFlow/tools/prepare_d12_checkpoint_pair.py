#!/usr/bin/env python3
"""Build the immutable paired checkpoints for D12 terminal XYWH transport.

Both outputs are complete target-model state dictionaries constructed from
one initialized target state.  They share every compatible generic tensor.
The candidate alone receives rows 0:4 of decoder terminal branches 0:5 from
the converted raw GroundingDINO source.  Publication is fail-closed and never
replaces an existing final or pending artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from mmengine import Config
from mmengine.utils import import_modules_from_strings
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

from projects.OVCapFlow.tools.prepare_cleanstart_checkpoint import (
    convert_groundingdino_state_dict,
    filter_compatible_state_dict,
    reject_forbidden_namespaces,
    reject_forbidden_source,
    sha256_file,
    unwrap_model_state,
)


EXPECTED_SOURCE_SHA256 = (
    '3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799')
SEED = 20260716
TERMINAL_LAYER_INDEX = 4
DECODER_BRANCHES = tuple(range(6))
ALL_REGRESSION_BRANCHES = tuple(range(7))
XYWH_ROW_COUNT = 4
TARGET_REG_DIM = 5
EMBED_DIMS = 256
AUTHORIZED_POSITION_COUNT = len(DECODER_BRANCHES) * (
    XYWH_ROW_COUNT * EMBED_DIMS + XYWH_ROW_COUNT)


def _weight_key(branch: int) -> str:
    return (
        f'bbox_head.reg_branches.{branch}.{TERMINAL_LAYER_INDEX}.weight')


def _bias_key(branch: int) -> str:
    return f'bbox_head.reg_branches.{branch}.{TERMINAL_LAYER_INDEX}.bias'


def allowed_transport_keys() -> tuple[str, ...]:
    """Return the exact twelve decoder-terminal keys authorized by D12."""
    return tuple(
        f'bbox_head.reg_branches.{branch}.{TERMINAL_LAYER_INDEX}.{suffix}'
        for branch in DECODER_BRANCHES
        for suffix in ('weight', 'bias'))


def _clone_state(state: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
    cloned = OrderedDict()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError('state must map string keys to tensors')
        cloned[key] = value.detach().cpu().clone()
    return cloned


def _validate_target_initialization(target: Mapping[str, Tensor]) -> None:
    for branch in ALL_REGRESSION_BRANCHES:
        weight_key = _weight_key(branch)
        bias_key = _bias_key(branch)
        if weight_key not in target or bias_key not in target:
            raise KeyError(
                f'missing target terminal for regression branch {branch}')
        weight = target[weight_key]
        bias = target[bias_key]
        if list(weight.shape) != [TARGET_REG_DIM, EMBED_DIMS]:
            raise ValueError(
                f'target terminal shape mismatch for {weight_key}: '
                f'{list(weight.shape)}')
        if list(bias.shape) != [TARGET_REG_DIM]:
            raise ValueError(
                f'target terminal shape mismatch for {bias_key}: '
                f'{list(bias.shape)}')
        if torch.count_nonzero(weight).item() != 0:
            if branch == 6:
                raise ValueError('branch 6 must be all-zero: ' + weight_key)
            raise ValueError(
                f'target terminal weight must be all-zero: {weight_key}')
        expected_bias = torch.zeros_like(bias)
        if not torch.equal(bias, expected_bias):
            if branch == 6:
                raise ValueError('branch 6 must be all-zero: ' + bias_key)
            raise ValueError(
                f'target terminal bias violates initialized contract: '
                f'{bias_key}')

    for key in (_weight_key(6), _bias_key(6)):
        if torch.count_nonzero(target[key]).item() != 0:
            raise ValueError(f'branch 6 must be all-zero: {key}')


def _validate_source_terminals(
        target: Mapping[str, Tensor],
        converted_source: Mapping[str, Tensor]) -> None:
    for branch in DECODER_BRANCHES:
        for key, expected_shape in (
                (_weight_key(branch), [XYWH_ROW_COUNT, EMBED_DIMS]),
                (_bias_key(branch), [XYWH_ROW_COUNT])):
            if key not in converted_source:
                raise KeyError('missing source terminal: ' + key)
            source_value = converted_source[key]
            target_value = target[key]
            if list(source_value.shape) != expected_shape:
                raise ValueError(
                    f'source terminal shape mismatch for {key}: '
                    f'{list(source_value.shape)}')
            if source_value.dtype != target_value.dtype:
                raise ValueError(
                    f'terminal dtype mismatch for {key}: '
                    f'{source_value.dtype} != {target_value.dtype}')
            if not torch.isfinite(source_value).all().item():
                raise ValueError('non-finite source terminal: ' + key)


def validate_pair_delta(
        control: Mapping[str, Tensor],
        candidate: Mapping[str, Tensor],
        converted_source: Mapping[str, Tensor]) -> dict[str, Any]:
    """Prove that a materialized pair obeys the complete D12 delta boundary."""
    if set(control) != set(candidate):
        raise ValueError('candidate/control key sets differ')
    allowed = set(allowed_transport_keys())

    for key in (_weight_key(6), _bias_key(6)):
        if not torch.equal(control[key], candidate[key]):
            raise ValueError('branch 6 differs between candidate and control')
        if torch.count_nonzero(control[key]).item() != 0:
            raise ValueError('branch 6 must be all-zero: ' + key)

    actual_unequal = 0
    for branch in DECODER_BRANCHES:
        weight_key = _weight_key(branch)
        bias_key = _bias_key(branch)
        if not torch.equal(
                candidate[weight_key][:XYWH_ROW_COUNT],
                converted_source[weight_key]):
            raise ValueError('candidate source XYWH mismatch: ' + weight_key)
        if not torch.equal(
                candidate[bias_key][:XYWH_ROW_COUNT],
                converted_source[bias_key]):
            raise ValueError('candidate source XYWH mismatch: ' + bias_key)
        if not torch.equal(
                candidate[weight_key][XYWH_ROW_COUNT:],
                control[weight_key][XYWH_ROW_COUNT:]):
            raise ValueError('candidate/control angle row differs: ' + weight_key)
        if not torch.equal(
                candidate[bias_key][XYWH_ROW_COUNT:],
                control[bias_key][XYWH_ROW_COUNT:]):
            raise ValueError('candidate/control angle row differs: ' + bias_key)
        actual_unequal += torch.count_nonzero(
            candidate[weight_key][:XYWH_ROW_COUNT]
            != control[weight_key][:XYWH_ROW_COUNT]).item()
        actual_unequal += torch.count_nonzero(
            candidate[bias_key][:XYWH_ROW_COUNT]
            != control[bias_key][:XYWH_ROW_COUNT]).item()

    for key in sorted(control):
        if key in allowed:
            continue
        if not torch.equal(control[key], candidate[key]):
            raise ValueError('state differs outside allowed transport: ' + key)

    return {
        'transported_keys': list(allowed_transport_keys()),
        'authorized_position_count': AUTHORIZED_POSITION_COUNT,
        'actual_unequal_position_count': int(actual_unequal),
        'bitwise_equal_outside_allowed_keys': True,
        'source_xywh_exact': True,
        'angle_rows_equal': True,
        'branch6_equal': True,
        'branch6_all_zero': True,
    }


def build_d12_pair_states(
        target_state: Mapping[str, Tensor],
        converted_source: Mapping[str, Tensor],
) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor],
           dict[str, Any]]:
    """Materialize the complete control/candidate pair from one target state."""
    _validate_target_initialization(target_state)
    _validate_source_terminals(target_state, converted_source)

    control = _clone_state(target_state)
    compatible, _ = filter_compatible_state_dict(
        converted_source, target_state)
    for key, value in compatible.items():
        control[key] = value.detach().cpu().clone()
    control = OrderedDict((key, control[key]) for key in sorted(control))
    candidate = _clone_state(control)

    for branch in DECODER_BRANCHES:
        weight_key = _weight_key(branch)
        bias_key = _bias_key(branch)
        candidate[weight_key][:XYWH_ROW_COUNT].copy_(
            converted_source[weight_key])
        candidate[bias_key][:XYWH_ROW_COUNT].copy_(
            converted_source[bias_key])

    summary = validate_pair_delta(control, candidate, converted_source)
    return control, candidate, summary


def _git_commit() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip()


def _file_record(path: Path, artifact_path: Path | None = None) -> dict:
    artifact_path = Path(artifact_path if artifact_path is not None else path)
    return {
        'path': str(Path(path).resolve()),
        'size_bytes': artifact_path.stat().st_size,
        'sha256': sha256_file(artifact_path),
    }


def build_manifest(
        *,
        source_path: Path,
        config_path: Path,
        control_path: Path,
        candidate_path: Path,
        control: Mapping[str, Tensor],
        candidate: Mapping[str, Tensor],
        summary: Mapping[str, Any],
        git_commit: str,
        control_artifact_path: Path | None = None,
        candidate_artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic manifest after checkpoint bytes are staged."""
    if set(control) != set(candidate):
        raise ValueError('cannot manifest different checkpoint key sets')
    required_checks = {
        'bitwise_equal_outside_allowed_keys',
        'source_xywh_exact',
        'angle_rows_equal',
        'branch6_equal',
        'branch6_all_zero',
    }
    if not required_checks <= set(summary):
        raise ValueError('pair summary is missing required checks')
    if not all(bool(summary[key]) for key in required_checks):
        raise ValueError('pair summary contains a failed safety check')

    return {
        'schema_version': 1,
        'git_commit': git_commit,
        'source': _file_record(Path(source_path)),
        'config': _file_record(Path(config_path)),
        'outputs': {
            'control': _file_record(
                Path(control_path), control_artifact_path),
            'candidate': _file_record(
                Path(candidate_path), candidate_artifact_path),
        },
        'checkpoint_payload_keys': ['state_dict'],
        'included_keys': sorted(control),
        'included_shapes': {
            key: list(control[key].shape) for key in sorted(control)
        },
        'transport': {
            'transported_keys': list(summary['transported_keys']),
            'authorized_position_count': int(
                summary['authorized_position_count']),
            'actual_unequal_position_count': int(
                summary['actual_unequal_position_count']),
            'query_transport': {'enabled': False},
            'reference_transport': {'enabled': False},
            'dn_transport': {'enabled': False},
            'classification_transport': {'enabled': False},
        },
        'checks': {
            key: bool(summary[key]) for key in sorted(required_checks)
        },
        'optimizer_state_present': False,
        'scheduler_state_present': False,
        'ema_state_present': False,
        'teacher_state_present': False,
        'forbidden_hits': [],
    }


def _collision_paths(paths: Sequence[Path]) -> list[str]:
    collisions = set()
    for raw_path in paths:
        path = Path(raw_path)
        if os.path.lexists(path):
            collisions.add(str(path))
        if path.parent.exists():
            for pending in path.parent.glob(path.name + '.pending.*'):
                if os.path.lexists(pending):
                    collisions.add(str(pending))
    return sorted(collisions)


def _pending_path(final_path: Path, index: int) -> Path:
    return final_path.with_name(
        f'{final_path.name}.pending.{os.getpid()}.{index}')


def _write_bytes(stream, payload: bytes) -> None:
    stream.write(payload)


def _stage_artifacts(
        payloads: Sequence[tuple[Path, bytes | Callable]]) -> list[
            tuple[Path, Path]]:
    final_paths = [Path(path).expanduser().resolve() for path, _ in payloads]
    collisions = _collision_paths(final_paths)
    if collisions:
        raise FileExistsError(
            'output collision: ' + ', '.join(collisions))
    for path in final_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    collisions = _collision_paths(final_paths)
    if collisions:
        raise FileExistsError(
            'output collision: ' + ', '.join(collisions))

    staged = []
    for index, ((_, payload), final_path) in enumerate(
            zip(payloads, final_paths)):
        pending_path = _pending_path(final_path, index)
        try:
            with pending_path.open('xb') as stream:
                if isinstance(payload, bytes):
                    _write_bytes(stream, payload)
                elif callable(payload):
                    payload(stream)
                else:
                    raise TypeError('artifact payload must be bytes or callable')
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            raise
        staged.append((pending_path, final_path))
    return staged


def _publish_staged_no_clobber(
        staged: Sequence[tuple[Path, Path]]) -> None:
    for pending_path, final_path in staged:
        os.link(pending_path, final_path)
        directory_fd = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    for pending_path, _ in staged:
        pending_path.unlink()


def publish_artifacts_no_clobber(
        payloads: Sequence[tuple[Path, bytes | Callable]]) -> None:
    """Stage and atomically hard-link artifacts without replacing anything."""
    staged = _stage_artifacts(payloads)
    _publish_staged_no_clobber(staged)


def _target_state_from_config(
        config_path: Path, seed: int = SEED) -> OrderedDict[str, Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    model_cfg = copy.deepcopy(cfg.model)
    model = MODELS.build(model_cfg)
    model.init_weights()
    return _clone_state(model.state_dict())


def _save_checkpoint(stream, state: Mapping[str, Tensor]) -> None:
    torch.save({'state_dict': state}, stream)


def prepare_d12_checkpoint_pair(
        *,
        source: Path,
        config: Path,
        control_output: Path,
        candidate_output: Path,
        manifest_output: Path,
) -> dict[str, Any]:
    """Construct, validate, and publish one immutable D12 checkpoint pair."""
    source = Path(source).expanduser().resolve()
    config = Path(config).expanduser().resolve()
    control_output = Path(control_output).expanduser().resolve()
    candidate_output = Path(candidate_output).expanduser().resolve()
    manifest_output = Path(manifest_output).expanduser().resolve()
    final_paths = [control_output, candidate_output, manifest_output]
    collisions = _collision_paths(final_paths)
    if collisions:
        raise FileExistsError(
            'output collision: ' + ', '.join(collisions))

    reject_forbidden_source(source)
    actual_source_sha = sha256_file(source)
    if actual_source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            'source SHA256 mismatch: expected ' + EXPECTED_SOURCE_SHA256 +
            ', got ' + actual_source_sha)
    checkpoint = torch.load(str(source), map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError('checkpoint root must be a mapping')
    original_state = unwrap_model_state(checkpoint)
    reject_forbidden_namespaces(original_state)
    converted_source = convert_groundingdino_state_dict(original_state)
    reject_forbidden_namespaces(converted_source)

    target_state = _target_state_from_config(config)
    control, candidate, summary = build_d12_pair_states(
        target_state, converted_source)

    checkpoint_staged = _stage_artifacts([
        (
            control_output,
            lambda stream: _save_checkpoint(stream, control)),
        (
            candidate_output,
            lambda stream: _save_checkpoint(stream, candidate)),
    ])
    control_pending = checkpoint_staged[0][0]
    candidate_pending = checkpoint_staged[1][0]
    manifest = build_manifest(
        source_path=source,
        config_path=config,
        control_path=control_output,
        candidate_path=candidate_output,
        control=control,
        candidate=candidate,
        summary=summary,
        git_commit=_git_commit(),
        control_artifact_path=control_pending,
        candidate_artifact_path=candidate_pending)
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    manifest_staged = _stage_artifacts([
        (manifest_output, manifest_payload),
    ])
    _publish_staged_no_clobber(checkpoint_staged + manifest_staged)

    for role, output in (
            ('control', control_output), ('candidate', candidate_output)):
        if sha256_file(output) != manifest['outputs'][role]['sha256']:
            raise RuntimeError(role + ' checkpoint hash changed during publish')
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--control-output', required=True, type=Path)
    parser.add_argument('--candidate-output', required=True, type=Path)
    parser.add_argument('--manifest-output', required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_d12_checkpoint_pair(
        source=args.source,
        config=args.config,
        control_output=args.control_output,
        candidate_output=args.candidate_output,
        manifest_output=args.manifest_output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
