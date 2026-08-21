#!/usr/bin/env python3
"""Build the CPU-only RISC OpenRSD N0-O input seal."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')
SUPPORT_PROTOCOL = 'risc-openrsd-n0o-support-v1'
TEXT_MAPPING_KEYS = (
    'text_support_mapping.0.weight',
    'text_support_mapping.0.bias',
    'text_support_mapping.2.weight',
    'text_support_mapping.2.bias',
)
FOLD_CONTRACT = {
    'c4_a': (0, 4, [0, 90, 180, 270]),
    'c4_b': (1, 4, [0, 90, 180, 270]),
    'c8_a': (2, 8, [0, 45, 90, 135, 180, 225, 270, 315]),
    'c8_b': (3, 8, [0, 45, 90, 135, 180, 225, 270, 315]),
}


class SealError(RuntimeError):
    """Raised when a frozen N0-O input contract is violated."""


@dataclass(frozen=True)
class TextMapping:
    first_weight: np.ndarray
    first_bias: np.ndarray
    second_weight: np.ndarray
    second_bias: np.ndarray


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str


@dataclass(frozen=True)
class SealAuthority:
    checkpoint: Asset
    scene_plan: Asset
    support_pickle: Asset
    historical_assets: dict[str, Asset]
    supporting_assets: dict[str, Asset]
    s0_assets: dict[str, Asset]
    image_dir: Path
    annotation_dir: Path
    image_count: int
    annotation_count: int
    nonempty_annotation_count: int
    empty_annotation_count: int
    classes: tuple[str, ...]
    scenes_per_fold: int
    support_shot: int
    text_width: int
    hidden_width: int
    mapped_width: int
    s0_base_commit: str


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError('canonical JSON encoding failed') from error
    return text.encode('utf-8') + b'\n'


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b''.join(canonical_json_bytes(dict(row)) for row in rows)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_file_hash(asset: Asset, description: str) -> dict[str, Any]:
    path = Path(asset.path)
    _require_sha256(asset.sha256, '{} sha256'.format(description))
    if not path.is_file():
        raise SealError('{} file is missing: {}'.format(description, path))
    observed = sha256_file(path)
    if observed != asset.sha256:
        raise SealError('{} hash mismatch'.format(description))
    return {
        'path': str(path),
        'byte_count': path.stat().st_size,
        'sha256': observed,
    }


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise SealError('{} must be lowercase SHA-256'.format(field))


def _require_unique(values, description: str) -> None:
    if len(values) != len(set(values)):
        raise SealError('{} must be unique'.format(description))


def validate_scene_plan(plan: Mapping[str, Any], *, scenes_per_fold: int = 40
                        ) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise SealError('scene plan must be a mapping')
    if type(scenes_per_fold) is not int or scenes_per_fold <= 0:
        raise SealError('scenes_per_fold must be a positive integer')
    mouth = plan.get('dataset_mouth')
    if not isinstance(mouth, Mapping):
        raise SealError('dataset_mouth is missing')
    if mouth.get('raw_tile_count') != 13833:
        raise SealError('raw tile count must equal 13833')
    if mouth.get('known_openrsd_filtered_tile_count') != 6605:
        raise SealError('filtered tile count must equal 6605')
    if plan.get('nonempty_tiles_observed') != 6605:
        raise SealError('nonempty_tiles_observed must equal 6605')

    fold_specs = plan.get('fold_specs')
    if not isinstance(fold_specs, list) or len(fold_specs) != 4:
        raise SealError('four fold specs are required')
    observed_specs = {}
    for spec in fold_specs:
        if not isinstance(spec, Mapping):
            raise SealError('fold spec must be a mapping')
        fold_id = spec.get('fold_id')
        if fold_id not in FOLD_CONTRACT:
            raise SealError('fold spec contains unknown fold')
        if spec.get('group_order') != FOLD_CONTRACT[fold_id][1]:
            raise SealError('fold spec group order mismatch')
        if spec.get('scenes') != scenes_per_fold:
            raise SealError('fold spec scene count mismatch')
        observed_specs[fold_id] = spec
    if set(observed_specs) != set(FOLD_CONTRACT):
        raise SealError('fold spec identities mismatch')

    records = plan.get('records')
    if not isinstance(records, list):
        raise SealError('records must be a list')
    expected_records = scenes_per_fold * len(FOLD_CONTRACT)
    if len(records) != expected_records:
        raise SealError(
            'scene record count must equal {}'.format(expected_records))

    scene_ids = []
    image_paths = []
    annotation_paths = []
    fold_counts = Counter()
    group_counts = Counter()
    for row_index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise SealError('scene record must be a mapping')
        scene_id = row.get('scene_id')
        if not isinstance(scene_id, str) or not scene_id:
            raise SealError('scene_id must be non-empty')
        if scene_id == 'P0148':
            raise SealError('P0148 is forbidden')
        fold_id = row.get('fold_id')
        if fold_id not in FOLD_CONTRACT:
            raise SealError('record fold_id is invalid')
        expected_index, expected_order, expected_angles = FOLD_CONTRACT[
            fold_id]
        if row.get('fold_index') != expected_index:
            raise SealError('fold_index mismatch')
        if row.get('group_order') != expected_order:
            raise SealError('group_order mismatch')
        if row.get('angles_deg') != expected_angles:
            raise SealError('angles mismatch')
        image_path = row.get('image_path')
        annotation_path = row.get('annotation_path')
        if not isinstance(image_path, str) or not image_path:
            raise SealError('image_path must be non-empty')
        if not isinstance(annotation_path, str) or not annotation_path:
            raise SealError('annotation_path must be non-empty')
        _require_sha256(row.get('image_sha256'), 'image_sha256')
        _require_sha256(row.get('annotation_sha256'), 'annotation_sha256')
        scene_ids.append(scene_id)
        image_paths.append(image_path)
        annotation_paths.append(annotation_path)
        fold_counts[fold_id] += 1
        group_counts[str(expected_order)] += 1

    _require_unique(scene_ids, 'unique scene identities')
    _require_unique(image_paths, 'unique image paths')
    _require_unique(annotation_paths, 'unique annotation paths')
    for fold_id in FOLD_CONTRACT:
        if fold_counts[fold_id] != scenes_per_fold:
            raise SealError('fold {} record count mismatch'.format(fold_id))

    return {
        'scene_count': len(records),
        'fold_counts': {
            fold_id: fold_counts[fold_id] for fold_id in FOLD_CONTRACT
        },
        'group_counts': {
            '4': group_counts['4'],
            '8': group_counts['8'],
        },
    }


def _support_index_key(scene_id: str, class_name: str, index: int) -> bytes:
    return hashlib.sha256(
        SUPPORT_PROTOCOL.encode('utf-8') + b'\0'
        + scene_id.encode('utf-8') + b'\0'
        + class_name.encode('utf-8') + b'\0'
        + str(index).encode('ascii')
    ).digest()


def select_support_indices(
        scene_id: str,
        support_data: Mapping[str, Any],
        *,
        classes,
        shot: int = 7) -> dict[str, list[int]]:
    if not isinstance(scene_id, str) or not scene_id:
        raise SealError('scene_id must be non-empty')
    if type(shot) is not int or shot <= 0:
        raise SealError('shot must be a positive integer')
    if not isinstance(support_data, Mapping):
        raise SealError('support data must be a mapping')
    classes = tuple(classes)
    if not classes or len(classes) != len(set(classes)):
        raise SealError('classes must be non-empty and unique')

    selected = {}
    for class_name in classes:
        if class_name not in support_data:
            raise SealError('support data is missing class {}'.format(
                class_name))
        class_data = support_data[class_name]
        if not isinstance(class_data, Mapping):
            raise SealError('support class data must be a mapping')
        embeddings = class_data.get('text_embeds')
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise SealError('text embeddings must be a 2D numpy array')
        if embeddings.dtype != np.float32:
            raise SealError('text embeddings must be float32')
        if embeddings.shape[0] < shot:
            raise SealError('every class must have at least {} prompts'.format(
                shot))
        if not np.isfinite(embeddings).all():
            raise SealError('text embeddings must be finite')
        ranked = sorted(
            range(embeddings.shape[0]),
            key=lambda index: (
                _support_index_key(scene_id, class_name, index), index))
        selected[class_name] = ranked[:shot]
    return selected


def _mapping_array(value: Any, key: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise SealError('{} must be a tensor'.format(key))
    if value.dtype != torch.float32:
        raise SealError('{} must be float32'.format(key))
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    if not np.isfinite(array).all():
        raise SealError('{} must be finite'.format(key))
    return array


def load_text_mapping(checkpoint_path: Path | str) -> TextMapping:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise SealError('checkpoint does not exist')
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise SealError('checkpoint must be a mapping')
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping):
        raise SealError('checkpoint must contain raw state_dict')
    missing = [key for key in TEXT_MAPPING_KEYS if key not in state]
    if missing:
        raise SealError('raw state_dict is missing {}'.format(', '.join(
            missing)))
    arrays = {
        key: _mapping_array(state[key], key) for key in TEXT_MAPPING_KEYS
    }
    first_weight = arrays[TEXT_MAPPING_KEYS[0]]
    first_bias = arrays[TEXT_MAPPING_KEYS[1]]
    second_weight = arrays[TEXT_MAPPING_KEYS[2]]
    second_bias = arrays[TEXT_MAPPING_KEYS[3]]
    if first_weight.ndim != 2:
        raise SealError('first mapping weight must be 2D')
    if first_bias.shape != (first_weight.shape[0],):
        raise SealError('first mapping bias shape mismatch')
    if second_weight.ndim != 2 or second_weight.shape[1] != first_weight.shape[0]:
        raise SealError('second mapping weight shape mismatch')
    if second_bias.shape != (second_weight.shape[0],):
        raise SealError('second mapping bias shape mismatch')
    return TextMapping(
        first_weight=first_weight,
        first_bias=first_bias,
        second_weight=second_weight,
        second_bias=second_bias,
    )


def build_support_row(
        scene_record: Mapping[str, Any],
        support_data: Mapping[str, Any],
        mapping: TextMapping,
        *,
        classes,
        shot: int = 7):
    classes = tuple(classes)
    scene_id = scene_record.get('scene_id')
    selected = select_support_indices(
        scene_id, support_data, classes=classes, shot=shot)
    source_by_class = []
    selections = []
    for class_name in classes:
        indices = selected[class_name]
        source_by_class.append(
            support_data[class_name]['text_embeds'][indices])
        selections.append({
            'class_name': class_name,
            'indices': list(indices),
        })
    source = np.asarray(source_by_class, dtype='<f4', order='C')
    if source.shape[-1] != mapping.first_weight.shape[1]:
        raise SealError('support width does not match text mapping input')
    hidden = source @ mapping.first_weight.T + mapping.first_bias
    hidden = np.maximum(hidden, np.float32(0.0))
    mapped = hidden @ mapping.second_weight.T + mapping.second_bias
    mapped = np.asarray(mapped, dtype='<f4', order='C')
    if not np.isfinite(mapped).all():
        raise SealError('mapped support tensor must be finite')
    source_bytes = source.tobytes(order='C')
    mapped_bytes = mapped.tobytes(order='C')
    row = {
        'schema': 'risc-openrsd-n0o-support-row-v1',
        'scene_id': scene_id,
        'fold_id': scene_record.get('fold_id'),
        'fold_index': scene_record.get('fold_index'),
        'group_order': scene_record.get('group_order'),
        'selections': selections,
        'source_tensor_dtype': np.dtype('<f4').str,
        'source_tensor_shape': list(source.shape),
        'source_tensor_byte_count': len(source_bytes),
        'source_tensor_sha256': sha256_bytes(source_bytes),
        'mapped_tensor_dtype': np.dtype('<f4').str,
        'mapped_tensor_shape': list(mapped.shape),
        'mapped_tensor_byte_count': len(mapped_bytes),
        'mapped_tensor_sha256': sha256_bytes(mapped_bytes),
    }
    return row, mapped


def _verify_mapping_dimensions(
        mapping: TextMapping, authority: SealAuthority) -> None:
    if mapping.first_weight.shape != (
            authority.hidden_width, authority.text_width):
        raise SealError('first text mapping dimensions mismatch')
    if mapping.first_bias.shape != (authority.hidden_width,):
        raise SealError('first text mapping bias dimensions mismatch')
    if mapping.second_weight.shape != (
            authority.mapped_width, authority.hidden_width):
        raise SealError('second text mapping dimensions mismatch')
    if mapping.second_bias.shape != (authority.mapped_width,):
        raise SealError('second text mapping bias dimensions mismatch')


def _count_files(directory: Path) -> tuple[int, int, int]:
    if not directory.is_dir():
        raise SealError('dataset directory is missing: {}'.format(directory))
    count = 0
    nonempty = 0
    empty = 0
    for path in directory.iterdir():
        if not path.is_file():
            continue
        count += 1
        if path.stat().st_size:
            nonempty += 1
        else:
            empty += 1
    return count, nonempty, empty


def _verify_dataset_inventory(authority: SealAuthority) -> dict[str, Any]:
    image_count, _, _ = _count_files(authority.image_dir)
    annotation_count, nonempty_count, empty_count = _count_files(
        authority.annotation_dir)
    expected = (
        authority.image_count,
        authority.annotation_count,
        authority.nonempty_annotation_count,
        authority.empty_annotation_count,
    )
    observed = (
        image_count, annotation_count, nonempty_count, empty_count)
    if observed != expected:
        raise SealError(
            'dataset inventory mismatch: expected {}, got {}'.format(
                expected, observed))
    return {
        'image_dir': str(authority.image_dir),
        'annotation_dir': str(authority.annotation_dir),
        'image_count': image_count,
        'annotation_count': annotation_count,
        'nonempty_annotation_count': nonempty_count,
        'empty_annotation_count': empty_count,
    }


def _verify_selected_files(plan: Mapping[str, Any]) -> None:
    for row in plan['records']:
        image = Path(row['image_path'])
        annotation = Path(row['annotation_path'])
        if not image.is_file() or sha256_file(image) != row['image_sha256']:
            raise SealError(
                'selected image hash mismatch: {}'.format(image))
        if (not annotation.is_file()
                or sha256_file(annotation) != row['annotation_sha256']):
            raise SealError(
                'selected annotation hash mismatch: {}'.format(annotation))


def _verify_asset_group(assets: Mapping[str, Asset]) -> dict[str, Any]:
    return {
        name: require_file_hash(asset, name)
        for name, asset in sorted(assets.items())
    }


def _load_support_pickle(path: Path) -> Mapping[str, Any]:
    with path.open('rb') as stream:
        value = pickle.load(stream)
    if not isinstance(value, Mapping):
        raise SealError('support pickle must contain a mapping')
    return value


def build_seal_artifacts(
        authority: SealAuthority) -> tuple[dict[str, bytes], dict[str, Any]]:
    checkpoint_record = require_file_hash(
        authority.checkpoint, 'A10 raw checkpoint')
    scene_plan_record = require_file_hash(
        authority.scene_plan, 'scene plan')
    support_pickle_record = require_file_hash(
        authority.support_pickle, 'support pickle')
    historical_records = _verify_asset_group(authority.historical_assets)
    supporting_records = _verify_asset_group(authority.supporting_assets)
    s0_records = _verify_asset_group(authority.s0_assets)
    dataset_record = _verify_dataset_inventory(authority)

    scene_plan_bytes = authority.scene_plan.path.read_bytes()
    try:
        scene_plan = json.loads(scene_plan_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError('scene plan JSON is invalid') from error
    scene_summary = validate_scene_plan(
        scene_plan, scenes_per_fold=authority.scenes_per_fold)
    if tuple(scene_plan.get('prompt', ())) != authority.classes:
        raise SealError('scene plan prompt order mismatch')
    _verify_selected_files(scene_plan)

    support_data = _load_support_pickle(authority.support_pickle.path)
    mapping = load_text_mapping(authority.checkpoint.path)
    _verify_mapping_dimensions(mapping, authority)
    rows = []
    mapped_bundle_digest = hashlib.sha256()
    mapped_bundle_bytes = 0
    for scene_record in scene_plan['records']:
        row, mapped = build_support_row(
            scene_record,
            support_data,
            mapping,
            classes=authority.classes,
            shot=authority.support_shot)
        mapped_bytes = mapped.tobytes(order='C')
        mapped_bundle_digest.update(mapped_bytes)
        mapped_bundle_bytes += len(mapped_bytes)
        rows.append(row)
    ledger_bytes = canonical_jsonl_bytes(rows)

    artifacts_without_manifest = {
        'scene_plan_40.json': scene_plan_bytes,
        'support_ledger.jsonl': ledger_bytes,
    }
    artifact_records = {
        name: {
            'byte_count': len(content),
            'sha256': sha256_bytes(content),
        }
        for name, content in sorted(artifacts_without_manifest.items())
    }
    script_path = Path(__file__).resolve()
    manifest = {
        'schema': 'risc-openrsd-n0o-input-manifest-v1',
        'status': 'SEALED_INPUTS_GPU_NOT_AUTHORIZED',
        'protocol': 'risc-openrsd-n0o-input-v1',
        'paper_mouth': {
            'dataset_len': 6605,
            'filter_empty_gt': True,
            'resize_scale': [1024, 1024],
            'support_type': 'text',
            'num_val_prompts': 7,
            'val_using_aux': False,
            'checkpoint_mode': 'raw',
            'dota_mAP': '0x1.68ef2daaf7a0fp-1',
            'dota_AP50': '0x1.68f5c28f5c28fp-1',
        },
        'execution_boundary': {
            'gpu_authorized': False,
            'model_forward_executed': False,
            'new_prediction_read': False,
            'new_metric_computed': False,
            'training_executed': False,
            'checkpoint_written': False,
        },
        'environment': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'numpy': np.__version__,
        },
        'builder': {
            'path': str(script_path),
            'sha256': sha256_file(script_path),
        },
        'parent': {
            **checkpoint_record,
            'state_source': 'state_dict',
            'ema_state_excluded': True,
            'text_mapping_keys': list(TEXT_MAPPING_KEYS),
        },
        'historical_assets': historical_records,
        'supporting_assets': supporting_records,
        's0': {
            'base_commit': authority.s0_base_commit,
            'assets': s0_records,
        },
        'dataset': dataset_record,
        'scene_plan': {
            **scene_plan_record,
            'row_count': scene_summary['scene_count'],
            'fold_counts': scene_summary['fold_counts'],
            'group_counts': scene_summary['group_counts'],
            'p0148_excluded': True,
        },
        'support': {
            'source': support_pickle_record,
            'selection_protocol': SUPPORT_PROTOCOL,
            'class_order': list(authority.classes),
            'class_count': len(authority.classes),
            'shot': authority.support_shot,
            'ledger_row_count': len(rows),
            'ledger_sha256': sha256_bytes(ledger_bytes),
            'mapped_tensor_shape_per_scene': [
                len(authority.classes),
                authority.support_shot,
                authority.mapped_width,
            ],
            'mapped_tensor_dtype': np.dtype('<f4').str,
            'mapped_tensor_bundle_byte_count': mapped_bundle_bytes,
            'mapped_tensor_bundle_sha256': mapped_bundle_digest.hexdigest(),
        },
        'artifacts': artifact_records,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    artifacts = {
        **artifacts_without_manifest,
        'input_manifest.json': manifest_bytes,
    }
    summary = {
        'status': manifest['status'],
        'scene_count': len(rows),
        'ledger_sha256': sha256_bytes(ledger_bytes),
        'mapped_tensor_bundle_sha256': mapped_bundle_digest.hexdigest(),
    }
    return artifacts, summary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, 'renameat2', None)
    if renameat2 is None:
        raise SealError('atomic no-replace publication is unavailable')
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise SealError('output directory already exists: {}'.format(
            destination))
    raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_artifacts(output_dir: Path | str, artifacts: Mapping[str, bytes]
                      ) -> None:
    output_dir = Path(output_dir)
    if os.path.lexists(str(output_dir)):
        raise SealError('output directory already exists: {}'.format(
            output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=output_dir.name + '.', dir=str(output_dir.parent)))
    try:
        for name, content in artifacts.items():
            if Path(name).name != name or not isinstance(content, bytes):
                raise SealError('artifact names and bytes must be canonical')
            path = temporary / name
            with path.open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(temporary)
        _rename_no_replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_input_seal(
        output_dir: Path | str,
        *,
        authority: SealAuthority | None = None) -> dict[str, Any]:
    if authority is None:
        authority = FROZEN_AUTHORITY
    if os.path.lexists(str(output_dir)):
        raise SealError('output directory already exists: {}'.format(
            output_dir))
    artifacts, summary = build_seal_artifacts(authority)
    publish_artifacts(output_dir, artifacts)
    return summary


RISC_ROOT = Path(__file__).resolve().parents[4]
P77E_ROOT = Path(
    '/data1/zcy/GSOVD/work_dirs/'
    'p77e_paper_a10_epoch24_dota2_text7_alignment_filtered_scale1024_20260706')
SUPPORT_ROOT = Path(
    '/data1/zcy/OpenRSD/work_dirs/'
    'dotav2_p4_lowtext_lser_sise_gpu67_20260617_153546/eval_bundle')


FROZEN_AUTHORITY = SealAuthority(
    checkpoint=Asset(
        path=Path(
            '/data1/zcy/OpenRSD/results/'
            'MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth'),
        sha256='097585080a4c95f23370840da546acfdcb85093480454133ad2a34876cc20bd6'),
    scene_plan=Asset(
        path=Path(
            '/data1/zcy/OV-CapFlow/work_dirs/risc_er/'
            'n0_set_orbit_20260817/scene_plan_40.json'),
        sha256='0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35'),
    support_pickle=Asset(
        path=SUPPORT_ROOT / 'DOTA2_1024_500/ss_train/'
        'Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        sha256='4ea3572d8184bfbfa556d051efaaea06d575bcb9a4e9f05efe3cc3564e5d737c'),
    historical_assets={
        'p77e_audit': Asset(
            P77E_ROOT / 'audit.json',
            '163478453291ce0c99d3c3eb0c14aaebe25ee8f0f9e7fa23d889fbd3b11fdef3'),
        'p77e_predictions': Asset(
            P77E_ROOT / 'predictions.pkl',
            'dedd99f81c84638a93e2d367b52c98774ff22723f4fd76832e37e258e227a6a7'),
        'p77e_resolved_config': Asset(
            P77E_ROOT / 'A10_flex_rtm_v3_1_formal.py',
            '2b4919b20492c92e5f4dc02fa0f76ef1cb2997527c0db4bc80d84793f3f5925b'),
        'p77e_result': Asset(
            P77E_ROOT / 'eval_results.json',
            '7202dbab088da8ec66270268b443c98ede5848507e797c6800c7b331b94ac836'),
        'p77e_runner': Asset(
            Path('/data1/zcy/GSOVD/.lab/workspace/p77e_paper_dota2_prompt_eval.py'),
            '39054ab7519f9fe3b0490a203faa3b00218a40f289cd019adf3c4c630918ce72'),
        'p77e_source_config': Asset(
            Path('/data1/zcy/GSOVD/.lab/tmp/openrsd_head_20260706/'
                 'M_configs/Step2_A10_Large_Pretrain_Stage3/'
                 'A10_flex_rtm_v3_1_formal.py'),
            '51fec95ef0c4fff121f7683115caca72721a942a2dfc2afcc72de7686a97cd96'),
    },
    supporting_assets={
        'neg_support_data': Asset(
            SUPPORT_ROOT / 'Neg_supports_v2.pkl',
            '51014c73b587f51f043e7dbe35a716bbcb0f70a57ac39c6ecf2cd5e50fb9a042'),
        'normalized_class_dict': Asset(
            SUPPORT_ROOT / 'normalized_class_dict.pkl',
            'a82b5da5c20ab5f48d93cd4e6a8c84da49254a75b725e764ba776b96af1e276b'),
        'pca_meta': Asset(
            SUPPORT_ROOT / '7_25_pca_meta_DINOv2_256.pkl',
            '8eb6eca687939858faa6b2989f4b997a12d518f34d740fb13b7761c51a7f7acd'),
    },
    s0_assets={
        'adapter': Asset(
            RISC_ROOT / 'framework/openrsd/M_AD/models/utils/'
            'risc_final_readout.py',
            '21c6f768df344bb496faaf5b73cb7f523d336379606f1102493d001adfb982a1'),
        'capture': Asset(
            RISC_ROOT / 'framework/openrsd/experiments/'
            'rotation_semantic_attractor/src/model_adapters/'
            'openrsd_hook_registry.py',
            'cf71fbd262700f4a3e787591703daa52c5fbf9697e922fb7e1e5b79c05899afc'),
        'head': Asset(
            RISC_ROOT / 'framework/openrsd/M_AD/models/dense_heads/'
            'Flex_Rrtmdet_head_v3_1.py',
            'eab2c717eecb70ac5c15ffdde3d3549dc3bf8106dcb3f7bedad1b8a72dda5ed3'),
        's0_config': Asset(
            RISC_ROOT / 'framework/openrsd/M_configs/Diagnostics/'
            'risc_openrsd_a10_final_readout_s0.py',
            '6318d20b72738b716335cb64134204007144f3d6ae534728fb697ed8a10787c5'),
    },
    image_dir=Path('/data1/zcy/datasets/DOTA2_1024_500/ss_val/images'),
    annotation_dir=Path('/data1/zcy/datasets/DOTA2_1024_500/ss_val/annfiles'),
    image_count=13833,
    annotation_count=13833,
    nonempty_annotation_count=6605,
    empty_annotation_count=7228,
    classes=(
        'airport', 'baseball-diamond', 'basketball-court', 'bridge',
        'container-crane', 'ground-track-field', 'harbor', 'helicopter',
        'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
        'small-vehicle', 'soccer-ball-field', 'storage-tank',
        'swimming-pool', 'tennis-court'),
    scenes_per_fold=40,
    support_shot=7,
    text_width=768,
    hidden_width=1024,
    mapped_width=256,
    s0_base_commit='93cabbcf5d383ff4641d07186c1d3c5681443ad7',
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_input_seal(args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
