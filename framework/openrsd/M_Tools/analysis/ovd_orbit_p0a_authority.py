"""Pure authority for selecting the P0-A diagnostic primary objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any


_REQUIRED_ROW_KEYS = frozenset({
    'annotation_sha256', 'box', 'class_name', 'object_id', 'overlap',
    'scene_id', 'size',
})
_SHA256_RE = re.compile(r'[0-9a-f]{64}\Z')
_AREA_MIN = 43.5
_AREA_MAX = 3265.0
_MAX_OVERLAP = 0.05
_MINIMUM_CLASS_COUNT = 5
_ZERO_OBSERVED_CLASSES = ('container-crane', 'helicopter', 'helipad')
_FULL_CLASS_ORDER = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
)
_INVENTORY_READY_STATUS = 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD'
_AUTHORITY_READY_STATUS = 'P0A_DIAGNOSTIC_AUTHORITY_READY_NO_FORWARD'
_INVENTORY_RECEIPT_SCHEMA = 'risc-n0-set-orbit-p0a-inventory-receipt-v1'
_INVENTORY_DIAGNOSTICS_SCHEMA = 'risc-n0-set-orbit-p0a-inventory-diagnostics-v1'
_AUTHORITY_SCHEMA = 'risc-n0-set-orbit-p0a-authority-v1'
_AUTHORITY_DIAGNOSTICS_SCHEMA = 'risc-n0-set-orbit-p0a-authority-diagnostics-v1'
_AUTHORITY_RECEIPT_SCHEMA = 'risc-n0-set-orbit-p0a-authority-receipt-v1'
_EXPECTED_CODE_PATHS = (
    'M_Tools/analysis/ovd_orbit_p0.py',
    'M_Tools/analysis/ovd_orbit_p0_export.py',
    'M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py',
)
_FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
_OPENAT_SUPPORTED = os.open in getattr(os, 'supports_dir_fd', ())
_MISSING = object()


class P0AAuthorityError(ValueError):
    """Raised when P0-A primary-selection input violates the sealed policy."""


@dataclass(frozen=True)
class PrimarySelection:
    """Immutable result of applying the fixed P0-A primary-object policy."""

    policy: Mapping[str, Any]
    primary_rows: tuple[Mapping[str, Any], ...]
    primary_object_count: int
    supported_classes: tuple[str, ...]
    non_primary_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CanonicalSnapshot:
    """Immutable canonical JSON mapping bound to its exact source digest."""

    data: Mapping[str, Any]
    source_sha256: str


class _InventoryRows(tuple):
    """Frozen inventory rows retaining the digest of their JSONL snapshot."""

    def __new__(
            cls, values: Iterable[Mapping[str, Any]], source_sha256: str,
    ) -> '_InventoryRows':
        instance = super().__new__(cls, values)
        instance.source_sha256 = source_sha256
        return instance


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one value as compact, key-sorted UTF-8 JSON followed by LF."""
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False,
    ) + '\n').encode('utf-8')


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of bytes-like input."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file, streamed in bounded chunks."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise P0AAuthorityError('chunk_size must be a positive integer')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_snapshot(path: str | Path, label: str) -> tuple[bytes, str]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise P0AAuthorityError(f'{label} could not read {source}') from error
    return raw, sha256_bytes(raw)


def _reject_json_constant(value: str) -> None:
    raise P0AAuthorityError('JSON constants must be finite')


def _decode_canonical_json(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode('utf-8'), parse_constant=_reject_json_constant)
        canonical = canonical_json_bytes(value)
    except (UnicodeDecodeError, json.JSONDecodeError, P0AAuthorityError,
            TypeError, ValueError, UnicodeError) as error:
        raise P0AAuthorityError(f'{label} must contain valid canonical JSON') from error
    if raw != canonical or not isinstance(value, Mapping):
        raise P0AAuthorityError(f'{label} bytes must be canonical JSON mapping')
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def load_canonical_json(path: str | Path, *, label: str) -> CanonicalSnapshot:
    """Load one mapping only from a canonical, byte-hashed JSON snapshot."""
    _require_nonempty_string(label, 'label')
    raw, digest = _read_snapshot(path, label)
    return CanonicalSnapshot(_deep_freeze(_decode_canonical_json(raw, label)), digest)


def _load_canonical_json_snapshot(
        raw: bytes, digest: str, label: str,
) -> CanonicalSnapshot:
    return CanonicalSnapshot(_deep_freeze(_decode_canonical_json(raw, label)), digest)


def load_inventory_rows(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load exactly canonical inventory JSONL records from one byte snapshot."""
    raw, digest = _read_snapshot(path, 'object inventory')
    return _decode_inventory_rows(raw, digest)


def _decode_inventory_rows(
        raw: bytes, digest: str,
) -> tuple[Mapping[str, Any], ...]:
    if not raw:
        raise P0AAuthorityError('object inventory must not be empty')
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b'\n') for line in lines):
        raise P0AAuthorityError('object inventory must be newline-delimited canonical JSON')
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line.decode('utf-8'), parse_constant=_reject_json_constant)
            canonical = canonical_json_bytes(value)
        except (UnicodeDecodeError, json.JSONDecodeError, P0AAuthorityError,
                TypeError, ValueError, UnicodeError) as error:
            raise P0AAuthorityError(
                f'object inventory row {index} must be canonical JSON') from error
        if line != canonical:
            raise P0AAuthorityError(
                f'object inventory row {index} must be canonical JSON')
        rows.append(_freeze_row(value, index))
    object_ids = tuple(row['object_id'] for row in rows)
    if len(set(object_ids)) != len(object_ids):
        raise P0AAuthorityError('object inventory object_id values must be unique')
    return _InventoryRows(_sort_rows(rows), digest)


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P0AAuthorityError(f'{label} must be a nonempty string')
    return value


def _require_finite_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P0AAuthorityError(f'{label} must be a finite nonnegative number')
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise P0AAuthorityError(f'{label} must be a finite nonnegative number')
    return number


def _validate_box(value: Any) -> tuple[float, float, float, float, float]:
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or len(value) != 5):
        raise P0AAuthorityError('box must contain exactly five finite numbers')
    numbers: list[float] = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, Real):
            raise P0AAuthorityError('box must contain exactly five finite numbers')
        converted = float(number)
        if not math.isfinite(converted):
            raise P0AAuthorityError('box must contain exactly five finite numbers')
        numbers.append(converted)
    return tuple(numbers)  # type: ignore[return-value]


def _freeze_row(row: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise P0AAuthorityError(f'row {index} must be a mapping')
    if frozenset(row) != _REQUIRED_ROW_KEYS or len(row) != len(_REQUIRED_ROW_KEYS):
        raise P0AAuthorityError(f'row {index} must contain exactly seven required keys')
    annotation_sha256 = row['annotation_sha256']
    if not isinstance(annotation_sha256, str) or not _SHA256_RE.fullmatch(annotation_sha256):
        raise P0AAuthorityError(
            f'row {index} annotation_sha256 must be a lowercase SHA-256 digest')
    return MappingProxyType({
        'annotation_sha256': annotation_sha256,
        'box': _validate_box(row['box']),
        'class_name': _require_nonempty_string(row['class_name'], f'row {index} class_name'),
        'object_id': _require_nonempty_string(row['object_id'], f'row {index} object_id'),
        'overlap': _require_finite_nonnegative_number(row['overlap'], f'row {index} overlap'),
        'scene_id': _require_nonempty_string(row['scene_id'], f'row {index} scene_id'),
        'size': _require_finite_nonnegative_number(row['size'], f'row {index} size'),
    })


def _sort_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(sorted(rows, key=lambda row: (row['scene_id'], row['object_id'])))


def _fixed_policy() -> Mapping[str, Any]:
    return MappingProxyType({
        'area_min': _AREA_MIN,
        'area_max': _AREA_MAX,
        'max_overlap_iou': _MAX_OVERLAP,
        'min_objects_per_class': _MINIMUM_CLASS_COUNT,
        'selection_basis': 'p0a_inventory_gt_quantiles_v1',
    })


def select_primary_objects(
        rows: Iterable[Mapping[str, Any]], *,
        zero_observed_classes: Any = _MISSING,
        **policy_overrides: Any) -> PrimarySelection:
    """Select immutable P0-A primary objects using the non-overridable policy."""
    if policy_overrides:
        names = ', '.join(sorted(policy_overrides))
        raise P0AAuthorityError(f'fixed P0-A policy cannot be overridden: {names}')
    if zero_observed_classes is _MISSING:
        raise P0AAuthorityError('zero_observed_classes declaration is required')
    if (type(zero_observed_classes) is not tuple
            or zero_observed_classes != _ZERO_OBSERVED_CLASSES):
        raise P0AAuthorityError(
            'zero_observed_classes must exactly declare '
            "('container-crane', 'helicopter', 'helipad')")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise P0AAuthorityError('rows must be an iterable of mappings')

    frozen_rows = tuple(_freeze_row(row, index) for index, row in enumerate(rows))
    object_ids = tuple(row['object_id'] for row in frozen_rows)
    if len(set(object_ids)) != len(object_ids):
        raise P0AAuthorityError('object_id values must be unique')
    observed_classes = {row['class_name'] for row in frozen_rows}
    zero_observed = set(_ZERO_OBSERVED_CLASSES)
    if observed_classes & zero_observed:
        raise P0AAuthorityError('declared zero-observed classes cannot appear in rows')

    geometry_rows = []
    geometry_excluded = []
    for row in frozen_rows:
        if _AREA_MIN <= row['size'] <= _AREA_MAX and row['overlap'] <= _MAX_OVERLAP:
            geometry_rows.append(row)
        else:
            geometry_excluded.append(row)

    class_counts: dict[str, int] = {}
    for row in geometry_rows:
        class_name = row['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    supported_classes = tuple(sorted(
        class_name for class_name, count in class_counts.items()
        if count >= _MINIMUM_CLASS_COUNT))
    supported = set(supported_classes)
    primary_rows = _sort_rows(row for row in geometry_rows if row['class_name'] in supported)
    unsupported_rows = _sort_rows(
        row for row in geometry_rows if row['class_name'] not in supported)
    diagnostics = MappingProxyType({
        'class_counts': MappingProxyType(dict(sorted(class_counts.items()))),
        'geometry_excluded': _sort_rows(geometry_excluded),
        'unsupported_class': unsupported_rows,
    })
    return PrimarySelection(
        policy=_fixed_policy(),
        primary_rows=primary_rows,
        primary_object_count=len(primary_rows),
        supported_classes=supported_classes,
        non_primary_diagnostics=diagnostics,
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise P0AAuthorityError(f'{label} must be a lowercase SHA-256 hex digest')
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P0AAuthorityError(f'{label} must be a mapping')
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise P0AAuthorityError(f'{label} must be a nonnegative integer')
    return value


def _snapshot_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, CanonicalSnapshot):
        raise P0AAuthorityError(f'{label} must be a canonical snapshot')
    _require_sha256(value.source_sha256, f'{label}.source_sha256')
    return _require_mapping(value.data, f'{label}.data')


def validate_inventory_receipt(
        receipt: CanonicalSnapshot, *, inventory_diagnostics: CanonicalSnapshot,
        object_inventory_rows: Sequence[Mapping[str, Any]],
        candidate_scene_plan_path: str | Path,
        inventory_result_path: str | Path,
        candidate_scene_plan_sha256: str | None = None,
        inventory_result_sha256: str | None = None,
) -> None:
    """Require a receipt to bind all four exact inventory artifact snapshots."""
    receipt_data = _snapshot_mapping(receipt, 'inventory receipt')
    diagnostics_data = _snapshot_mapping(
        inventory_diagnostics, 'inventory diagnostics')
    if receipt_data.get('schema') != _INVENTORY_RECEIPT_SCHEMA:
        raise P0AAuthorityError('inventory receipt schema is not sealed P0-A schema')
    if receipt_data.get('status') != _INVENTORY_READY_STATUS:
        raise P0AAuthorityError('inventory receipt status is not inventory-ready')
    if diagnostics_data.get('schema') != _INVENTORY_DIAGNOSTICS_SCHEMA:
        raise P0AAuthorityError('inventory diagnostics schema is not sealed P0-A schema')
    if diagnostics_data.get('status') != _INVENTORY_READY_STATUS:
        raise P0AAuthorityError('inventory diagnostics status is not inventory-ready')
    if _require_nonnegative_int(
            diagnostics_data.get('retained_object_count'),
            'inventory diagnostics.retained_object_count') != len(object_inventory_rows):
        raise P0AAuthorityError(
            'inventory diagnostics retained_object_count must match inventory rows')
    artifact_sha256 = _require_mapping(
        receipt_data.get('artifact_sha256'), 'inventory receipt.artifact_sha256')
    required = {
        'candidate_scene_plan.json', 'inventory_diagnostics.json',
        'object_inventory.jsonl', 'result.md',
    }
    if set(artifact_sha256) != required:
        raise P0AAuthorityError('inventory receipt must bind exactly four artifacts')
    candidate_digest = (sha256_file(candidate_scene_plan_path)
                        if candidate_scene_plan_sha256 is None else _require_sha256(
                            candidate_scene_plan_sha256,
                            'candidate_scene_plan_sha256'))
    result_digest = (sha256_file(inventory_result_path)
                     if inventory_result_sha256 is None else _require_sha256(
                         inventory_result_sha256, 'inventory_result_sha256'))
    observed = {
        'candidate_scene_plan.json': candidate_digest,
        'inventory_diagnostics.json': inventory_diagnostics.source_sha256,
        'object_inventory.jsonl': _require_sha256(
            getattr(object_inventory_rows, 'source_sha256', None),
            'object inventory.source_sha256'),
        'result.md': result_digest,
    }
    for name, digest in observed.items():
        if _require_sha256(artifact_sha256[name], f'inventory receipt.{name}') != digest:
            raise P0AAuthorityError(f'inventory receipt digest mismatch for {name}')


def validate_source_support_manifest(
        source_input_manifest: CanonicalSnapshot, *,
        support_asset_path: str | Path,
        support_asset_snapshot_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate the one exact opaque text7 support declaration and bytes."""
    manifest = _snapshot_mapping(source_input_manifest, 'source input manifest')
    if manifest.get('schema') != 'risc-openrsd-n0o-input-manifest-v1':
        raise P0AAuthorityError('source input manifest schema is not sealed')
    paper_mouth = _require_mapping(
        manifest.get('paper_mouth'), 'source input manifest.paper_mouth')
    if paper_mouth.get('support_type') != 'text':
        raise P0AAuthorityError(
            'source input manifest paper_mouth.support_type must be text')
    support = _require_mapping(manifest.get('support'), 'source input manifest.support')
    if support.get('shot') != 7:
        raise P0AAuthorityError('source input manifest support shot must be 7')
    if support.get('class_count') != 18:
        raise P0AAuthorityError('source input manifest support class_count must be 18')
    class_order = support.get('class_order')
    if not isinstance(class_order, tuple) or class_order != _FULL_CLASS_ORDER:
        raise P0AAuthorityError('source input manifest support class_order must be sealed')
    source = _require_mapping(support.get('source'), 'source input manifest.support.source')
    declared_path = _require_nonempty_string(
        source.get('path'), 'source input manifest.support.source.path')
    supplied_path = str(Path(support_asset_path))
    if declared_path != supplied_path:
        raise P0AAuthorityError('supplied support path must exactly match source manifest')
    expected_hash = _require_sha256(
        source.get('sha256'), 'source input manifest.support.source.sha256')
    observed_hash = (sha256_file(support_asset_path)
                     if support_asset_snapshot_sha256 is None else _require_sha256(
                         support_asset_snapshot_sha256,
                         'support_asset_snapshot_sha256'))
    if observed_hash != expected_hash:
        raise P0AAuthorityError('support asset hash mismatch')
    return MappingProxyType({
        'class_order': _FULL_CLASS_ORDER,
        'sha256': observed_hash,
        'shot': 7,
        'source_path': supplied_path,
        'support_type': 'text',
    })


def _oracle_open_flags() -> tuple[int, int]:
    """Return fail-closed directory and file flags for pinned oracle reads."""
    nofollow = getattr(os, 'O_NOFOLLOW', None)
    directory = getattr(os, 'O_DIRECTORY', None)
    if nofollow is None or directory is None or not _OPENAT_SUPPORTED:
        raise P0AAuthorityError('safe no-follow openat oracle snapshot is unavailable')
    common = os.O_RDONLY | nofollow | getattr(os, 'O_CLOEXEC', 0)
    return common | directory, common


def _open_pinned_framework_root(root: Path, directory_flags: int) -> int:
    try:
        descriptor = os.open(root, directory_flags)
    except OSError as error:
        raise P0AAuthorityError('trusted framework root could not safely open') from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise P0AAuthorityError('trusted framework root must be a directory')
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_oracle_descriptor_snapshot(
        root_fd: int, relative_path: str, *, directory_flags: int,
        file_flags: int,
) -> tuple[bytes, str]:
    """Read one exact relative regular file through a pinned root descriptor."""
    components = relative_path.split('/')
    if not components or any(not component or component in {'.', '..'}
                             for component in components):
        raise P0AAuthorityError('oracle code identity must be a safe relative path')
    try:
        current_fd = os.dup(root_fd)
    except OSError as error:
        raise P0AAuthorityError('trusted framework root descriptor could not duplicate') from error
    descriptor: int | None = None
    try:
        for component in components[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except OSError as error:
                raise P0AAuthorityError(
                    'oracle code parent could not safely open') from error
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise P0AAuthorityError('oracle code parent must be a directory')
            except BaseException:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        try:
            descriptor = os.open(components[-1], file_flags, dir_fd=current_fd)
        except OSError as error:
            raise P0AAuthorityError('oracle code file could not safely open') from error
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise P0AAuthorityError('oracle code file could not stat') from error
        if not stat.S_ISREG(metadata.st_mode):
            raise P0AAuthorityError('oracle code file must be a regular file')
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            chunks.append(chunk)
        return b''.join(chunks), digest.hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(current_fd)


def _trusted_oracle_code_snapshots(
        code_paths: Sequence[str | Path],
) -> tuple[tuple[bytes, str], ...]:
    if isinstance(code_paths, (str, bytes)) or not isinstance(code_paths, Sequence):
        raise P0AAuthorityError('oracle_code_files must be a sequence of three paths')
    if len(code_paths) != 3:
        raise P0AAuthorityError('oracle_code_files must contain exactly three files')
    root = Path(_FRAMEWORK_ROOT).resolve()
    directory_flags, file_flags = _oracle_open_flags()
    root_fd = _open_pinned_framework_root(root, directory_flags)
    try:
        snapshots: list[tuple[bytes, str]] = []
        for expected, supplied in zip(_EXPECTED_CODE_PATHS, code_paths):
            expected_path = root / expected
            raw_path = Path(supplied)
            candidate = raw_path if raw_path.is_absolute() else root / raw_path
            if candidate != expected_path:
                raise P0AAuthorityError(
                    'oracle code path is not an exact trusted-root path')
            snapshots.append(_read_oracle_descriptor_snapshot(
                root_fd, expected, directory_flags=directory_flags,
                file_flags=file_flags))
        return tuple(snapshots)
    finally:
        os.close(root_fd)


def _code_identities(
        snapshots: Sequence[tuple[bytes, str]],
) -> tuple[Mapping[str, str], ...]:
    if len(snapshots) != 3:
        raise P0AAuthorityError('oracle code snapshots must contain exactly three files')
    return tuple(MappingProxyType({'path': path, 'sha256': digest})
                 for path, (_, digest) in zip(_EXPECTED_CODE_PATHS, snapshots))


def _authority_result_markdown(selection: PrimarySelection) -> bytes:
    return (
        '# P0-A 诊断权威摘要\n\n'
        f'状态：{_AUTHORITY_READY_STATUS}\n\n'
        f'主对象数：{selection.primary_object_count}\n'
        f'支持类别数：{len(selection.supported_classes)}\n\n'
        '仅封存诊断输入与代码身份；未运行模型、检查点或 GPU。\n'
        '未执行模型前向、checkpoint 加载、GPU、AP 或 P0 指标计算。\n'
    ).encode('utf-8')


def _consumed_input_digest_ledger(
        *, snapshots: Mapping[str, tuple[bytes, str]],
        code_snapshots: Sequence[tuple[bytes, str]],
) -> Mapping[str, str]:
    """Return the exact hashes of every byte source consumed by the builder."""
    if set(snapshots) != {
            'receipt', 'diagnostics', 'inventory', 'manifest', 'support',
            'candidate', 'result'}:
        raise P0AAuthorityError('authority input snapshot ledger is incomplete')
    if len(code_snapshots) != len(_EXPECTED_CODE_PATHS):
        raise P0AAuthorityError('authority code snapshot ledger is incomplete')
    return MappingProxyType({
        'candidate_scene_plan.json': snapshots['candidate'][1],
        'inventory_diagnostics.json': snapshots['diagnostics'][1],
        'object_inventory.jsonl': snapshots['inventory'][1],
        'receipt.json': snapshots['receipt'][1],
        'result.md': snapshots['result'][1],
        'source_input_manifest.json': snapshots['manifest'][1],
        'support_asset': snapshots['support'][1],
        **{path: digest for path, (_, digest) in zip(
            _EXPECTED_CODE_PATHS, code_snapshots)},
    })


def consumed_input_digest_ledger(artifacts: Mapping[str, bytes]) -> Mapping[str, str]:
    """Read and validate the builder's consumed-input digest ledger from artifacts."""
    if not isinstance(artifacts, Mapping):
        raise P0AAuthorityError('authority artifacts must be a mapping')
    payload = artifacts.get('authority_diagnostics.json')
    if not isinstance(payload, bytes):
        raise P0AAuthorityError('authority diagnostics artifact is required')
    diagnostics = _decode_canonical_json(payload, 'authority diagnostics artifact')
    ledger = _require_mapping(
        diagnostics.get('consumed_input_sha256'), 'consumed_input_sha256')
    required = {
        'candidate_scene_plan.json', 'inventory_diagnostics.json',
        'object_inventory.jsonl', 'receipt.json', 'result.md',
        'source_input_manifest.json', 'support_asset', *_EXPECTED_CODE_PATHS,
    }
    if set(ledger) != required:
        raise P0AAuthorityError('consumed_input_sha256 keys are not sealed')
    return MappingProxyType({
        name: _require_sha256(digest, f'consumed_input_sha256.{name}')
        for name, digest in ledger.items()
    })


def rehash_authority_input_ledger(
        *, inventory_receipt_path: str | Path,
        inventory_diagnostics_path: str | Path,
        object_inventory_path: str | Path,
        source_input_manifest_path: str | Path,
        support_asset_path: str | Path,
        oracle_code_files: Sequence[str | Path],
) -> Mapping[str, str]:
    """Rehash all CLI inputs using the same names as a consumed-input ledger."""
    if (isinstance(oracle_code_files, (str, bytes))
            or not isinstance(oracle_code_files, Sequence)
            or len(oracle_code_files) != len(_EXPECTED_CODE_PATHS)):
        raise P0AAuthorityError('oracle_code_files must contain exactly three files')
    receipt_path = Path(inventory_receipt_path)
    paths = {
        'candidate_scene_plan.json': receipt_path.with_name('candidate_scene_plan.json'),
        'inventory_diagnostics.json': Path(inventory_diagnostics_path),
        'object_inventory.jsonl': Path(object_inventory_path),
        'receipt.json': receipt_path,
        'result.md': receipt_path.with_name('result.md'),
        'source_input_manifest.json': Path(source_input_manifest_path),
        'support_asset': Path(support_asset_path),
        **{name: Path(path) for name, path in zip(
            _EXPECTED_CODE_PATHS, oracle_code_files)},
    }
    try:
        return MappingProxyType({name: sha256_file(path) for name, path in paths.items()})
    except OSError as error:
        raise P0AAuthorityError('authority input rehash failed') from error


def build_authority_artifacts(
        *, inventory_receipt_path: str | Path,
        inventory_diagnostics_path: str | Path,
        object_inventory_path: str | Path,
        source_input_manifest_path: str | Path,
        support_asset_path: str | Path,
        oracle_code_files: Sequence[str | Path],
) -> dict[str, bytes]:
    """Build the deterministic no-forward P0-A diagnostic authority package."""
    code_paths = tuple(oracle_code_files) if not isinstance(
        oracle_code_files, (str, bytes)) else ()
    snapshots = {
        'receipt': _read_snapshot(inventory_receipt_path, 'inventory receipt'),
        'diagnostics': _read_snapshot(
            inventory_diagnostics_path, 'inventory diagnostics'),
        'inventory': _read_snapshot(object_inventory_path, 'object inventory'),
        'manifest': _read_snapshot(source_input_manifest_path, 'source input manifest'),
        'support': _read_snapshot(support_asset_path, 'support asset'),
        'candidate': _read_snapshot(
            Path(inventory_receipt_path).with_name('candidate_scene_plan.json'),
            'candidate scene plan'),
        'result': _read_snapshot(
            Path(inventory_receipt_path).with_name('result.md'),
            'inventory result'),
    }
    code_snapshots = _trusted_oracle_code_snapshots(code_paths)
    receipt = _load_canonical_json_snapshot(*snapshots['receipt'], 'inventory receipt')
    diagnostics = _load_canonical_json_snapshot(
        *snapshots['diagnostics'], 'inventory diagnostics')
    rows = _decode_inventory_rows(*snapshots['inventory'])
    manifest = _load_canonical_json_snapshot(
        *snapshots['manifest'], 'source input manifest')
    candidate_path = Path(inventory_receipt_path).with_name('candidate_scene_plan.json')
    result_path = Path(inventory_receipt_path).with_name('result.md')
    validate_inventory_receipt(
        receipt, inventory_diagnostics=diagnostics, object_inventory_rows=rows,
        candidate_scene_plan_path=candidate_path, inventory_result_path=result_path,
        candidate_scene_plan_sha256=snapshots['candidate'][1],
        inventory_result_sha256=snapshots['result'][1])
    support = validate_source_support_manifest(
        manifest, support_asset_path=support_asset_path,
        support_asset_snapshot_sha256=snapshots['support'][1])
    if support['sha256'] != snapshots['support'][1]:
        raise P0AAuthorityError('support asset snapshot hash mismatch')
    identities = _code_identities(code_snapshots)
    consumed_input_sha256 = _consumed_input_digest_ledger(
        snapshots=snapshots, code_snapshots=code_snapshots)
    unrecognized = sorted({row['class_name'] for row in rows} - set(support['class_order']))
    if unrecognized:
        raise P0AAuthorityError(
            'object inventory contains classes outside source support vocabulary')
    selection = select_primary_objects(
        rows, zero_observed_classes=_ZERO_OBSERVED_CLASSES)
    observed_classes = tuple(sorted({row['class_name'] for row in rows}))
    authority = {
        'all_classes': list(_FULL_CLASS_ORDER),
        'condition': {'support_type': 'text', 'shot': 7},
        'condition_id': 'historical_text7_support_v1',
        'inventory_sha256': {
            'inventory_diagnostics.json': snapshots['diagnostics'][1],
            'object_inventory.jsonl': snapshots['inventory'][1],
            'receipt.json': snapshots['receipt'][1],
        },
        'observed_classes': list(observed_classes),
        'oracle_adapter_type': 'dense-carrier-fallback',
        'oracle_code_identities': [dict(item) for item in identities],
        'policy': dict(selection.policy),
        'primary_object_count': selection.primary_object_count,
        'prompt_stability_status': 'NOT_TESTED_SINGLE_CONDITION',
        'schema': _AUTHORITY_SCHEMA,
        'score_field': 'raw_native_foreground_logits',
        'source_schema': 'level-row-v1',
        'status': _AUTHORITY_READY_STATUS,
        'shot': 7,
        'support_asset_path': support['source_path'],
        'support_asset_sha256': support['sha256'],
        'supported_classes': list(selection.supported_classes),
        'support_type': 'text',
        'zero_observed_classes': list(_ZERO_OBSERVED_CLASSES),
    }
    primary_ids = b''.join(canonical_json_bytes({'object_id': row['object_id']})
                           for row in selection.primary_rows)
    authority_diagnostics = {
        'consumed_input_sha256': dict(consumed_input_sha256),
        'observed_class_count': len(observed_classes),
        'primary_object_count': selection.primary_object_count,
        'schema': _AUTHORITY_DIAGNOSTICS_SCHEMA,
        'status': _AUTHORITY_READY_STATUS,
        'supported_class_count': len(selection.supported_classes),
        'zero_observed_class_count': len(_ZERO_OBSERVED_CLASSES),
    }
    artifacts = {
        'p0a_diagnostic_authority.json': canonical_json_bytes(authority),
        'primary_object_ids.jsonl': primary_ids,
        'authority_diagnostics.json': canonical_json_bytes(authority_diagnostics),
    }
    artifacts['receipt.json'] = canonical_json_bytes({
        'artifact_sha256': {
            name: sha256_bytes(payload) for name, payload in artifacts.items()},
        'schema': _AUTHORITY_RECEIPT_SCHEMA,
        'status': _AUTHORITY_READY_STATUS,
    })
    artifacts['result.md'] = _authority_result_markdown(selection)
    # The receipt must bind every non-receipt artifact, including the human receipt.
    artifacts['receipt.json'] = canonical_json_bytes({
        'artifact_sha256': {
            name: sha256_bytes(payload)
            for name, payload in artifacts.items() if name != 'receipt.json'},
        'schema': _AUTHORITY_RECEIPT_SCHEMA,
        'status': _AUTHORITY_READY_STATUS,
    })
    return artifacts


def build_authority_failure_artifacts(error: P0AAuthorityError) -> dict[str, bytes]:
    """Build the intentionally minimal package for an authority input failure."""
    if not isinstance(error, P0AAuthorityError):
        raise TypeError('error must be a P0AAuthorityError')
    status = 'P0_INPUT_FAIL_STOP'
    message = str(error)
    return {
        'receipt.json': canonical_json_bytes({
            'error': message,
            'schema': _AUTHORITY_RECEIPT_SCHEMA,
            'status': status,
        }),
        'authority_diagnostics.json': canonical_json_bytes({
            'error': message,
            'schema': _AUTHORITY_DIAGNOSTICS_SCHEMA,
            'status': status,
        }),
        'result.md': (
            '# P0-A 诊断权威失败\n\n'
            f'输入权威校验失败：{message}\n\n'
            '未运行模型、检查点、GPU 或指标计算。\n'
        ).encode('utf-8'),
    }


def _validated_authority_artifact_items(
        artifacts: Mapping[str, bytes],
) -> tuple[tuple[str, bytes], ...]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise P0AAuthorityError('artifacts must be a nonempty mapping')
    items: list[tuple[str, bytes]] = []
    names: set[str] = set()
    for name, payload in artifacts.items():
        if not isinstance(name, str) or not name:
            raise P0AAuthorityError('artifact names must be nonempty strings')
        path = Path(name)
        if (path.is_absolute() or path.name != name or name in {'.', '..'}
                or '/' in name or '\\' in name or '\x00' in name):
            raise P0AAuthorityError(f'unsafe artifact name: {name!r}')
        if name in names:
            raise P0AAuthorityError(f'duplicate artifact name: {name!r}')
        if not isinstance(payload, bytes) or not payload:
            raise P0AAuthorityError(f'artifact {name!r} must be nonempty bytes')
        names.add(name)
        items.append((name, payload))
    return tuple(items)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a same-parent directory without overwriting a target."""
    if os.name != 'posix':
        raise P0AAuthorityError('no-replace directory publication is unavailable')
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise P0AAuthorityError(
            'no-replace directory publication is unavailable') from error
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def _raise_if_output_exists(output_dir: Path) -> None:
    try:
        output_dir.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(output_dir))


def publish_authority_artifacts(
        output_dir: Path | str, artifacts: Mapping[str, bytes],
) -> None:
    """Fsync and no-replace publish an authority artifact package."""
    items = _validated_authority_artifact_items(artifacts)
    target = Path(output_dir)
    if not target.name or target.name in {'.', '..'}:
        raise P0AAuthorityError('output_dir must name a new directory')
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    _raise_if_output_exists(target)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{target.name}.tmp-', dir=parent))
    try:
        for name, payload in items:
            destination = temporary / name
            with destination.open('xb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    'P0AAuthorityError', 'PrimarySelection', 'CanonicalSnapshot',
    'canonical_json_bytes',
    'sha256_bytes', 'sha256_file', 'load_canonical_json', 'load_inventory_rows',
    'select_primary_objects', 'validate_inventory_receipt',
    'validate_source_support_manifest', 'build_authority_artifacts',
    'consumed_input_digest_ledger', 'rehash_authority_input_ledger',
    'build_authority_failure_artifacts', 'publish_authority_artifacts',
]
