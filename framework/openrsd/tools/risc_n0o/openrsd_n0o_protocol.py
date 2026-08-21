#!/usr/bin/env python3
"""CPU protocol consumer for the sealed OpenRSD N0-O v3 inputs."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as torch_functional

from tools.risc_n0o.prepare_openrsd_n0o_input_seal import (
    SealError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_authoritative_manifest,
    load_text_mapping,
    sha256_bytes,
    sha256_file,
    validate_scene_plan,
)


V3_MANIFEST_SHA256 = (
    '646b8702d60a3dbe36a35871b5599459807b8399deafac98d3056a1f04d8351c')
RUNNER_PROTOCOL = 'risc-openrsd-n0o-runner-v1'


class ProtocolError(RuntimeError):
    """Raised before execution when N0-O runtime authority is invalid."""


@dataclass(frozen=True)
class ProtocolBundle:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    scene_records: dict[str, Mapping[str, Any]]
    support_rows: dict[str, Mapping[str, Any]]


def _read_canonical_json(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError('{} is invalid JSON'.format(path.name)) from error
    if raw != canonical_json_bytes(value):
        raise ProtocolError('{} is not canonical JSON'.format(path.name))
    return value


def _read_canonical_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    raw = path.read_bytes()
    try:
        rows = tuple(json.loads(line) for line in raw.splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError('{} is invalid JSONL'.format(path.name)) from error
    if raw != canonical_jsonl_bytes(rows):
        raise ProtocolError('{} is not canonical JSONL'.format(path.name))
    return rows


def load_protocol_bundle(root: Path | str) -> ProtocolBundle:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ProtocolError('v3 input directory is missing')
    try:
        manifest = load_authoritative_manifest(root / 'input_manifest.json')
    except SealError as error:
        raise ProtocolError(str(error)) from error
    manifest_sha = sha256_file(root / 'input_manifest.json')
    if manifest_sha != V3_MANIFEST_SHA256:
        raise ProtocolError('input manifest SHA-256 is not exact v3')
    scene_path = root / 'scene_plan_40.json'
    ledger_path = root / 'support_ledger.jsonl'
    if sha256_file(scene_path) != manifest['artifacts'][
            'scene_plan_40.json']['sha256']:
        raise ProtocolError('scene plan hash mismatch')
    if sha256_file(ledger_path) != manifest['support']['ledger_sha256']:
        raise ProtocolError('support ledger hash mismatch')
    scene_plan = _read_canonical_json(scene_path)
    try:
        scene_summary = validate_scene_plan(scene_plan, scenes_per_fold=40)
    except SealError as error:
        raise ProtocolError(str(error)) from error
    if scene_summary['scene_count'] != 160:
        raise ProtocolError('scene plan must contain 160 rows')
    support_rows_raw = _read_canonical_jsonl(ledger_path)
    if len(support_rows_raw) != 160:
        raise ProtocolError('support ledger must contain 160 rows')
    scene_records = {}
    for row in scene_plan['records']:
        scene_id = row['scene_id']
        if scene_id in scene_records:
            raise ProtocolError('duplicate scene plan identity')
        scene_records[scene_id] = row
    support_rows = {}
    for row in support_rows_raw:
        scene_id = row.get('scene_id')
        if scene_id in support_rows:
            raise ProtocolError('duplicate support ledger identity')
        support_rows[scene_id] = row
    if set(scene_records) != set(support_rows):
        raise ProtocolError('scene/support ledger identities mismatch')
    for scene_id in scene_records:
        scene = scene_records[scene_id]
        support = support_rows[scene_id]
        for field in ('fold_id', 'fold_index', 'group_order'):
            if scene[field] != support[field]:
                raise ProtocolError(
                    'scene/support {} mismatch'.format(field))
    return ProtocolBundle(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        scene_records=scene_records,
        support_rows=support_rows,
    )


def view_specs(group_order: int) -> list[dict[str, Any]]:
    if group_order == 4:
        angles = (90, 180, 270)
    elif group_order == 8:
        angles = (45, 90, 135, 180, 225, 270, 315)
    else:
        raise ProtocolError('group order must be 4 or 8')
    views = [
        {'view_id': 'rot000_a', 'angle_deg': 0, 'forward_ordinal': 0},
        {'view_id': 'rot000_b', 'angle_deg': 0, 'forward_ordinal': 1},
    ]
    views.extend({
        'view_id': 'rot{:03d}'.format(angle),
        'angle_deg': angle,
        'forward_ordinal': index + 2,
    } for index, angle in enumerate(angles))
    return views


def build_model_ledger(bundle: ProtocolBundle) -> list[dict[str, Any]]:
    rows = []
    for scene_id, scene in bundle.scene_records.items():
        support = bundle.support_rows[scene_id]
        rows.append({
            'schema': 'risc-openrsd-n0o-model-row-v1',
            'protocol': RUNNER_PROTOCOL,
            'input_manifest_sha256': bundle.manifest_sha256,
            'scene_plan_sha256': bundle.manifest['scene_plan']['sha256'],
            'support_ledger_sha256': bundle.manifest['support'][
                'ledger_sha256'],
            'scene_id': scene_id,
            'fold_id': scene['fold_id'],
            'fold_index': scene['fold_index'],
            'group_order': scene['group_order'],
            'tile_name': scene['tile_name'],
            'image_path': scene['image_path'],
            'image_sha256': scene['image_sha256'],
            'support_source_sha256': support['source_tensor_sha256'],
            'support_mapped_sha256': support['mapped_tensor_sha256'],
            'support_prompt_indices': [
                item['indices'] for item in support['selections']],
            'support_row_sha256': sha256_bytes(canonical_json_bytes(support)),
            'views': view_specs(scene['group_order']),
            'shard_relative_path': '{}/{}_{}.npz'.format(
                scene['fold_id'], scene['scene_rank'], scene_id),
        })
    return rows


def model_ledger_bytes(rows) -> bytes:
    return canonical_jsonl_bytes(rows)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    value = np.asarray(value, dtype='<f4', order='C')
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


class SupportCache:
    """Reconstruct and cache v3 support tensors without any CUDA access."""

    def __init__(self, bundle, support_data, mapping):
        self.bundle = bundle
        self.support_data = support_data
        self.mapping = mapping
        self.classes = tuple(bundle.manifest['support']['class_order'])
        self._cache = {}

    @classmethod
    def from_bundle(cls, bundle: ProtocolBundle):
        support_path = Path(bundle.manifest['support']['source']['path'])
        checkpoint_path = Path(bundle.manifest['parent']['path'])
        if sha256_file(support_path) != bundle.manifest['support'][
                'source']['sha256']:
            raise ProtocolError('support source hash mismatch')
        if sha256_file(checkpoint_path) != bundle.manifest['parent']['sha256']:
            raise ProtocolError('parent checkpoint hash mismatch')
        with support_path.open('rb') as stream:
            support_data = pickle.load(stream)
        mapping = load_text_mapping(checkpoint_path)
        return cls(bundle, support_data, mapping)

    def for_scene(self, scene_id: str):
        if scene_id in self._cache:
            return self._cache[scene_id]
        if scene_id not in self.bundle.support_rows:
            raise ProtocolError('scene is absent from support ledger')
        row = self.bundle.support_rows[scene_id]
        if [item['class_name'] for item in row['selections']] != list(
                self.classes):
            raise ProtocolError('support class order mismatch')
        source = np.asarray([
            self.support_data[item['class_name']]['text_embeds'][
                item['indices']]
            for item in row['selections']
        ], dtype='<f4', order='C')
        if sha256_bytes(source.tobytes(order='C')) != row[
                'source_tensor_sha256']:
            raise ProtocolError('support source tensor hash mismatch')
        source_tensor = torch.from_numpy(source)
        with torch.no_grad():
            hidden = torch_functional.linear(
                source_tensor,
                torch.from_numpy(self.mapping.first_weight),
                torch.from_numpy(self.mapping.first_bias))
            hidden = torch.relu(hidden)
            mapped = torch_functional.linear(
                hidden,
                torch.from_numpy(self.mapping.second_weight),
                torch.from_numpy(self.mapping.second_bias))
        mapped = mapped.contiguous()
        if tensor_sha256(mapped) != row['mapped_tensor_sha256']:
            raise ProtocolError('support mapped tensor hash mismatch')
        labels = torch.arange(len(self.classes), dtype=torch.long)
        labels = labels[:, None].expand(len(self.classes), source.shape[1])
        result = (
            mapped.reshape(1, -1, mapped.shape[-1]),
            labels.contiguous().reshape(1, -1),
        )
        self._cache[scene_id] = result
        return result


def audit_support_bundle(cache: SupportCache) -> dict[str, Any]:
    """Reconstruct every sealed scene and verify the aggregate mapped bytes."""
    digest = hashlib.sha256()
    byte_count = 0
    scene_count = 0
    for scene_id in cache.bundle.support_rows:
        support, _ = cache.for_scene(scene_id)
        value = support[0].reshape(18, 7, 256).detach().cpu().contiguous()
        array = np.asarray(value.numpy(), dtype='<f4', order='C')
        raw = array.tobytes(order='C')
        digest.update(raw)
        byte_count += len(raw)
        scene_count += 1
    expected = cache.bundle.manifest['support']
    if scene_count != expected['ledger_row_count']:
        raise ProtocolError('support aggregate scene count mismatch')
    if byte_count != expected['mapped_tensor_bundle_byte_count']:
        raise ProtocolError('support aggregate byte count mismatch')
    bundle_sha256 = digest.hexdigest()
    if bundle_sha256 != expected['mapped_tensor_bundle_sha256']:
        raise ProtocolError('support aggregate hash mismatch')
    return {
        'scene_count': scene_count,
        'mapped_tensor_bundle_byte_count': byte_count,
        'mapped_tensor_bundle_sha256': bundle_sha256,
    }


__all__ = [
    'ProtocolBundle', 'ProtocolError', 'RUNNER_PROTOCOL', 'SupportCache',
    'V3_MANIFEST_SHA256', 'audit_support_bundle', 'build_model_ledger',
    'load_protocol_bundle', 'model_ledger_bytes', 'tensor_sha256', 'view_specs',
]
