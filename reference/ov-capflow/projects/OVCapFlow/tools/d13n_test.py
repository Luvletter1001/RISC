#!/usr/bin/env python3
"""Run one deterministic five-rank D13-N evaluation and publish evidence."""

import argparse
import ast
import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import sys
from contextlib import contextmanager
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from mmengine import Config
from mmengine.dist import (all_gather_object, barrier, broadcast_object_list,
                           get_dist_info)
from mmengine.runner import Runner

from projects.OVCapFlow.ov_capflow.d13n_dump_results import (
    D13NDumpCollisionError)
from projects.OVCapFlow.ov_capflow.no_replace import publish_json_noreplace
from projects.OVCapFlow.tools.validate_dotav2_q600_dump import (
    load_cpu, validate_records)

# Importing the package registers the project model, hooks, and dump metric.
import projects.OVCapFlow.ov_capflow  # noqa: F401,E402


ROLES = (
    'stage0-parent',
    'proxy-control',
    'proxy-candidate',
    'raw-control',
    'raw-candidate',
)
ROLE_CONFIG = {
    'stage0-parent': ('control', 'proxy400'),
    'proxy-control': ('control', 'proxy400'),
    'proxy-candidate': ('candidate', 'proxy400'),
    'raw-control': ('raw-control', 'raw13833'),
    'raw-candidate': ('raw-candidate', 'raw13833'),
}
ROLE_PORTS = {
    'stage0-parent': (29845,),
    'proxy-control': (29842, 29846),
    'proxy-candidate': (29841, 29847),
    'raw-control': (29843,),
    'raw-candidate': (29844,),
}
RAW_DISABLED_FIELDS = (
    'train_cfg',
    'train_dataloader',
    'optim_wrapper',
    'param_scheduler',
)
DERIVED_DELETED_FIELDS = [
    'model.bbox_head.existence_loss_weight',
    'model.freeze_except_patterns',
    'custom_hooks[0]',
]
EXPECTED_CUDA_VISIBLE_DEVICES = '5,6,7,8,9'
EXPECTED_LOCAL_RANKS = [0, 1, 2, 3, 4]
EXPECTED_WORLD_SIZE = 5
QUERIES_PER_IMAGE = 600
RECORDS_BY_MOUTH = {'proxy400': 400, 'raw13833': 13833}
TORCHRUN_PYTHON = '/data/zcy/anaconda3/envs/mmdet/bin/python'
TEST_SCRIPT = 'projects/OVCapFlow/tools/d13n_test.py'


class OutputCollisionError(FileExistsError):
    """One immutable output final/pending path is already occupied."""


class IdentityError(ValueError):
    """A role, config, checkpoint, or topology identity is invalid."""


def _default_checkpoint_loader(path: Path) -> Any:
    return torch.load(str(path), map_location='cpu')


class RuntimeFacade:
    """Injectable boundary around MMEngine, validation, and collectives."""

    def __init__(
            self,
            load_config: Callable[[str], Config] = Config.fromfile,
            build_runner: Callable[[Config], Any] = Runner.from_cfg,
            checkpoint_loader: Callable[[Path], Any] =
            _default_checkpoint_loader,
            load_records: Callable[[Path], Any] = load_cpu,
            validate_records: Callable[..., Mapping] = validate_records,
            get_dist_info: Callable[[], Tuple[int, int]] = get_dist_info,
            all_gather_object: Callable[[Any], list] = all_gather_object,
            barrier: Callable[[], None] = barrier,
            broadcast_object_list: Callable[[list], None] =
            broadcast_object_list) -> None:
        self.load_config = load_config
        self.build_runner = build_runner
        self.checkpoint_loader = checkpoint_loader
        self.load_records = load_records
        self.validate_records = validate_records
        self.get_dist_info = get_dist_info
        self.all_gather_object = all_gather_object
        self.barrier = barrier
        self.broadcast_object_list = broadcast_object_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--launcher', choices=('pytorch',), required=True)
    parser.add_argument('--role', choices=ROLES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--metrics-out', type=Path, required=True)
    parser.add_argument('--identity-out', type=Path, required=True)
    parser.add_argument('--derive-adapter-free-parent', action='store_true')
    return parser


def _absolute_lexical_path(path: os.PathLike) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _physical_target_without_final_resolution(path: Path) -> Path:
    """Resolve parent aliases while retaining the literal final component."""
    try:
        parent = path.parent.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise OutputCollisionError(
            'failed to resolve output parent: {}'.format(path.parent)) \
            from error
    return parent / path.name


def _path_collisions(path: Path) -> list:
    collisions = []
    if os.path.lexists(path):
        collisions.append(str(path))
    if path.parent.exists():
        prefix = path.name + '.pending.'
        collisions.extend(
            str(item) for item in path.parent.iterdir()
            if item.name.startswith(prefix) and os.path.lexists(item))
    return sorted(set(collisions))


def resolve_and_preflight_outputs(
        out: os.PathLike, metrics_out: os.PathLike,
        identity_out: os.PathLike) -> Tuple[Path, Path, Path]:
    """Resolve three lexical paths and reject any alias/final/pending."""
    resolved = tuple(_absolute_lexical_path(path) for path in (
        out, metrics_out, identity_out))
    if len(set(resolved)) != 3:
        raise OutputCollisionError('output paths must be pairwise distinct')
    physical = tuple(
        _physical_target_without_final_resolution(path) for path in resolved)
    if len(set(physical)) != 3:
        raise OutputCollisionError(
            'output paths contain a physical alias through parent directories')
    collisions = []
    for path in resolved:
        collisions.extend(_path_collisions(path))
    if collisions:
        raise OutputCollisionError(
            'output collision: ' + ', '.join(sorted(set(collisions))))
    return resolved


def _require_input_file(path: os.PathLike, name: str) -> Path:
    resolved = _absolute_lexical_path(path)
    try:
        is_file = resolved.is_file()
    except OSError as error:
        raise IdentityError(
            'failed to inspect {}: {}'.format(name, resolved)) from error
    if not is_file:
        raise IdentityError('{} must be an existing file: {}'.format(
            name, resolved))
    return resolved


class CheckpointBinding:
    """One open checkpoint inode, its digest, and its proc-fd runtime path."""

    def __init__(self, original_path: Path, descriptor: int,
                 sha256: str) -> None:
        self.original_path = original_path
        self.descriptor = descriptor
        self.runtime_path = Path('/proc/self/fd/{}'.format(descriptor))
        self.sha256 = sha256


_CHECKPOINT_SNAPSHOT_SCHEMA = 'd13n-checkpoint-snapshot-v1'
_FICLONE = 0x40049409
_FICLONE_UNSUPPORTED = {
    errno.EINVAL,
    errno.ENOTTY,
    errno.EOPNOTSUPP,
    errno.EXDEV,
    getattr(errno, 'ENOSYS', 38),
}


class CheckpointSnapshot:
    """One verified read-only descriptor for the shared immutable inode."""

    def __init__(self, descriptor: int, authority: Mapping) -> None:
        self.descriptor = descriptor
        self.runtime_path = Path('/proc/self/fd/{}'.format(descriptor))
        self.sha256 = authority['sha256']
        self.dev = authority['dev']
        self.ino = authority['ino']
        self.size = authority['size']
        self.authority = dict(authority)
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            os.close(self.descriptor)
            self._closed = True


def _sha256_open_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def _copy_checkpoint_bytes(source: int, destination: int, size: int) -> None:
    """Copy exactly one source snapshot using only the kernel sendfile path."""
    offset = 0
    while offset < size:
        copied = os.sendfile(
            destination, source, offset, min(size - offset, 1024 ** 3))
        if copied <= 0:
            raise OSError('kernel checkpoint copy ended before source size')
        offset += copied
    os.ftruncate(destination, size)


def _clone_or_copy_checkpoint(source: int, destination: int,
                              size: int) -> None:
    try:
        fcntl.ioctl(destination, _FICLONE, source)
    except OSError as error:
        if error.errno not in _FICLONE_UNSUPPORTED:
            raise
        _copy_checkpoint_bytes(source, destination, size)


def _validate_checkpoint_snapshot_authority(authority: Any) -> dict:
    required = {'schema', 'proc_path', 'sha256', 'dev', 'ino', 'size'}
    if not isinstance(authority, Mapping) or set(authority) != required:
        raise IdentityError('malformed immutable checkpoint authority')
    value = dict(authority)
    if value['schema'] != _CHECKPOINT_SNAPSHOT_SCHEMA:
        raise IdentityError('invalid immutable checkpoint authority schema')
    if (type(value['proc_path']) is not str
            or re.fullmatch(
                r'/proc/[1-9][0-9]*/fd/[0-9]+', value['proc_path']) is None):
        raise IdentityError('invalid immutable checkpoint proc path')
    if (type(value['sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', value['sha256']) is None):
        raise IdentityError('invalid immutable checkpoint SHA256')
    for field in ('dev', 'ino', 'size'):
        if (type(value[field]) is not int or isinstance(value[field], bool)
                or value[field] < 0):
            raise IdentityError(
                'invalid immutable checkpoint {}'.format(field))
    return value


def _verify_checkpoint_snapshot(descriptor: int, authority: Mapping) -> None:
    observed = os.fstat(descriptor)
    expected_identity = (
        authority['dev'], authority['ino'], authority['size'])
    if (observed.st_dev, observed.st_ino, observed.st_size) != expected_identity:
        raise IdentityError('immutable checkpoint inode metadata mismatch')
    if observed.st_mode & 0o777 != 0o400:
        raise IdentityError('immutable checkpoint mode must be 0400')
    if _sha256_open_descriptor(descriptor) != authority['sha256']:
        raise IdentityError('immutable checkpoint SHA256 mismatch')


def _create_rank0_checkpoint_snapshot(
        binding: CheckpointBinding) -> CheckpointSnapshot:
    """Create, unlink, and verify rank zero's single shared snapshot."""
    parent_flags = (os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
                    | getattr(os, 'O_CLOEXEC', 0))
    try:
        parent_descriptor = os.open(binding.original_path.parent, parent_flags)
    except OSError as error:
        raise IdentityError(
            'failed to pin checkpoint snapshot parent') from error
    writable_descriptor = -1
    readonly_descriptor = -1
    snapshot_name = None
    identity = None
    try:
        create_flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | getattr(os, 'O_CLOEXEC', 0))
        for _ in range(128):
            candidate = '.{}.d13n-checkpoint-snapshot-{}'.format(
                binding.original_path.stem, secrets.token_hex(12))
            try:
                writable_descriptor = os.open(
                    candidate,
                    create_flags,
                    0o600,
                    dir_fd=parent_descriptor)
                snapshot_name = candidate
                break
            except FileExistsError:
                continue
        if writable_descriptor < 0 or snapshot_name is None:
            raise IdentityError(
                'failed to allocate exclusive checkpoint snapshot')
        created = os.fstat(writable_descriptor)
        identity = (created.st_dev, created.st_ino)
        source_size = os.fstat(binding.descriptor).st_size
        _clone_or_copy_checkpoint(
            binding.descriptor, writable_descriptor, source_size)
        os.fsync(writable_descriptor)
        os.fchmod(writable_descriptor, 0o400)

        read_flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
        readonly_descriptor = os.open(
            snapshot_name, read_flags, dir_fd=parent_descriptor)
        os.close(writable_descriptor)
        writable_descriptor = -1
        observed = os.fstat(readonly_descriptor)
        if (observed.st_dev, observed.st_ino) != identity:
            raise IdentityError(
                'checkpoint snapshot identity changed before unlink')
        _cleanup_owned_snapshot(parent_descriptor, snapshot_name, identity)
        snapshot_name = None
        authority = {
            'schema': _CHECKPOINT_SNAPSHOT_SCHEMA,
            'proc_path': '/proc/{}/fd/{}'.format(
                os.getpid(), readonly_descriptor),
            'sha256': binding.sha256,
            'dev': observed.st_dev,
            'ino': observed.st_ino,
            'size': observed.st_size,
        }
        authority = _validate_checkpoint_snapshot_authority(authority)
        _verify_checkpoint_snapshot(readonly_descriptor, authority)
        snapshot = CheckpointSnapshot(readonly_descriptor, authority)
        readonly_descriptor = -1
        return snapshot
    except IdentityError:
        raise
    except OSError as error:
        raise IdentityError(
            'failed to create immutable checkpoint snapshot: {}'.format(
                error)) from error
    finally:
        try:
            if snapshot_name is not None:
                _cleanup_open_owned_snapshot(
                    parent_descriptor, snapshot_name, identity,
                    writable_descriptor)
        finally:
            try:
                if writable_descriptor >= 0:
                    os.close(writable_descriptor)
                if readonly_descriptor >= 0:
                    os.close(readonly_descriptor)
            finally:
                os.close(parent_descriptor)


def _open_checkpoint_snapshot(authority: Mapping) -> CheckpointSnapshot:
    authority = _validate_checkpoint_snapshot_authority(authority)
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    descriptor = -1
    try:
        descriptor = os.open(authority['proc_path'], flags)
        _verify_checkpoint_snapshot(descriptor, authority)
        snapshot = CheckpointSnapshot(descriptor, authority)
        descriptor = -1
        return snapshot
    except IdentityError:
        raise
    except OSError as error:
        raise IdentityError(
            'failed to open immutable checkpoint snapshot: {}'.format(
                error)) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def establish_checkpoint_snapshot(
        runtime: RuntimeFacade, rank: int,
        binding: CheckpointBinding) -> Tuple[dict, Optional[CheckpointSnapshot]]:
    """Broadcast rank zero's verified authority or one shared failure."""
    envelope = [None]
    owner = None
    if rank == 0:
        try:
            owner = _create_rank0_checkpoint_snapshot(binding)
            envelope[0] = {
                'schema': _CHECKPOINT_SNAPSHOT_SCHEMA,
                'status': 'ok',
                'authority': owner.authority,
            }
        except Exception as error:
            envelope[0] = {
                'schema': _CHECKPOINT_SNAPSHOT_SCHEMA,
                'status': 'error',
                'message': 'failed to establish immutable checkpoint '
                           'snapshot: {}'.format(error),
            }
    try:
        runtime.broadcast_object_list(envelope)
    except Exception:
        if owner is not None:
            owner.close()
        raise
    payload = envelope[0]
    if (not isinstance(payload, Mapping)
            or payload.get('schema') != _CHECKPOINT_SNAPSHOT_SCHEMA):
        if owner is not None:
            owner.close()
        raise IdentityError('malformed checkpoint snapshot broadcast')
    if payload.get('status') == 'error':
        if owner is not None:
            owner.close()
        raise IdentityError(str(payload.get(
            'message', 'immutable checkpoint snapshot failed')))
    if payload.get('status') != 'ok':
        if owner is not None:
            owner.close()
        raise IdentityError('invalid checkpoint snapshot broadcast status')
    try:
        authority = _validate_checkpoint_snapshot_authority(
            payload.get('authority'))
    except Exception:
        if owner is not None:
            owner.close()
        raise
    return authority, owner


@contextmanager
def open_checkpoint_binding(path: os.PathLike):
    """Open and hash a checkpoint once, retaining its inode through the run."""
    original = _absolute_lexical_path(path)
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(original, flags)
        try:
            digest = _sha256_open_descriptor(descriptor)
        except Exception:
            os.close(descriptor)
            raise
    except OSError as error:
        raise IdentityError(
            'failed to open/hash checkpoint: {}'.format(original)) from error
    binding = CheckpointBinding(original, descriptor, digest)
    try:
        yield binding
    finally:
        os.close(descriptor)


def _read_config_bytes(parent_descriptor: int, name: str,
                       display_path: Path) -> bytes:
    flags = (os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
             | getattr(os, 'O_NOFOLLOW', 0))
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        try:
            chunks = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise IdentityError(
            'failed to open/read config: {}'.format(display_path)) from error
    return b''.join(chunks)


def _cleanup_owned_snapshot(parent_descriptor: int, name: str,
                            identity: Tuple[int, int]) -> None:
    try:
        observed = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as error:
        raise IdentityError(
            'exclusive snapshot vanished before cleanup') from error
    except OSError as error:
        raise IdentityError(
            'failed to inspect exclusive snapshot') from error
    if (observed.st_dev, observed.st_ino) != identity:
        raise IdentityError(
            'exclusive snapshot identity changed before cleanup')
    try:
        os.unlink(name, dir_fd=parent_descriptor)
    except OSError as error:
        raise IdentityError(
            'failed to clean exclusive snapshot') from error


def _open_descriptor_identity_for_cleanup(descriptor: int) -> Tuple[int, int]:
    """Recover an inode identity from an owned open descriptor."""
    try:
        observed = os.fstat(descriptor)
    except OSError as fstat_error:
        try:
            observed = os.stat('/proc/self/fd/{}'.format(descriptor))
        except OSError as proc_error:
            raise IdentityError(
                'failed to recover exclusive snapshot identity') from proc_error
        if observed is None:
            raise IdentityError(
                'failed to recover exclusive snapshot identity') from fstat_error
    return observed.st_dev, observed.st_ino


def _cleanup_open_owned_snapshot(parent_descriptor: int, name: str,
                                 identity: Optional[Tuple[int, int]],
                                 descriptor: int) -> None:
    """Remove only the name still bound to the open snapshot descriptor."""
    if identity is None:
        if descriptor < 0:
            raise IdentityError('exclusive snapshot identity is unavailable')
        identity = _open_descriptor_identity_for_cleanup(descriptor)
    _cleanup_owned_snapshot(parent_descriptor, name, identity)


def _config_base_literals(payload: bytes, display_path: Path) -> list:
    """Return ordered literal ``_base_`` strings and their AST nodes."""
    try:
        parsed = ast.parse(payload.decode('utf-8'), filename=str(display_path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise IdentityError(
            'failed to parse config dependency: {}'.format(display_path)) \
            from error
    assignments = []
    for node in parsed.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == '_base_'
               for target in node.targets):
            assignments.append(node)
    if len(assignments) > 1:
        raise IdentityError(
            'config contains multiple _base_ assignments: {}'.format(
                display_path))
    if not assignments:
        return []
    value = assignments[0].value
    if isinstance(value, ast.Constant) and type(value.value) is str:
        elements = [value]
    elif isinstance(value, ast.List):
        elements = value.elts
    else:
        raise IdentityError(
            'config _base_ must be a literal path or path list: {}'.format(
                display_path))
    if (not elements
            or any(not isinstance(element, ast.Constant)
                   or type(element.value) is not str or not element.value
                   for element in elements)):
        raise IdentityError(
            'config _base_ paths must be nonempty built-in strings')
    literals = [(element.value, element) for element in elements]
    for base, _ in literals:
        if os.path.isabs(base):
            raise IdentityError(
                'config snapshot requires local relative _base_ paths')
    return literals


def _relative_config_bases(payload: bytes, display_path: Path) -> list:
    """Return literal relative Python ``_base_`` paths without execution."""
    return [base for base, _ in _config_base_literals(payload, display_path)]


def _config_node_byte_span(payload: bytes, node: ast.AST,
                           display_path: Path) -> Tuple[int, int]:
    """Translate one AST node's UTF-8 offsets into payload byte offsets."""
    if (getattr(node, 'end_lineno', None) is None
            or getattr(node, 'end_col_offset', None) is None):
        raise IdentityError(
            'config parser did not retain literal offsets: {}'.format(
                display_path))
    line_starts = [0]
    for offset, value in enumerate(payload):
        if value == ord('\n'):
            line_starts.append(offset + 1)
    try:
        start = line_starts[node.lineno - 1] + node.col_offset
        end = line_starts[node.end_lineno - 1] + node.end_col_offset
    except (AttributeError, IndexError) as error:
        raise IdentityError(
            'invalid config literal offsets: {}'.format(display_path)) \
            from error
    if start < 0 or end < start or end > len(payload):
        raise IdentityError(
            'config literal offsets escape payload: {}'.format(display_path))
    return start, end


def _rewrite_config_local_bases(payload: bytes, source: Path,
                                snapshot_names: Mapping) -> bytes:
    """Rewrite only local base literals to flat sibling snapshot names."""
    replacements = []
    for base, node in _config_base_literals(payload, source):
        if '::' in base:
            continue
        resolved = Path(os.path.normpath(os.path.join(
            str(source.parent), base)))
        try:
            snapshot_name = snapshot_names[resolved]
        except KeyError as error:
            raise IdentityError(
                'config dependency is absent from pinned closure: {}'.format(
                    resolved)) from error
        start, end = _config_node_byte_span(payload, node, source)
        replacements.append(
            (start, end, repr('./{}'.format(snapshot_name)).encode('utf-8')))
    rewritten = payload
    for start, end, replacement in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten


def _descriptor_identity_via_proc(descriptor: int) -> Tuple[int, int]:
    """Inspect a pinned directory through its procfs descriptor link."""
    try:
        observed = os.stat('/proc/self/fd/{}'.format(descriptor))
    except OSError as error:
        raise IdentityError(
            'failed to inspect pinned config directory') from error
    return observed.st_dev, observed.st_ino


def _pin_config_ancestor_directories(
        entry_parent_descriptor: int) -> Tuple[Path, dict, list]:
    """Pin the entry's original physical ancestor chain before any reads."""
    descriptors = []
    directory_flags = (os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
                       | getattr(os, 'O_CLOEXEC', 0)
                       | getattr(os, 'O_NOFOLLOW', 0))
    try:
        target = os.readlink(
            '/proc/self/fd/{}'.format(entry_parent_descriptor))
        if not os.path.isabs(target) or target.endswith(' (deleted)'):
            raise IdentityError(
                'config parent has no stable absolute physical path')
        physical_parent = Path(os.path.normpath(target))
        root_descriptor = os.open('/', directory_flags)
        descriptors.append(root_descriptor)
        directories = {Path('/'): root_descriptor}
        current = Path('/')
        current_descriptor = root_descriptor
        for component in physical_parent.parts[1:]:
            current_descriptor = os.open(
                component, directory_flags, dir_fd=current_descriptor)
            descriptors.append(current_descriptor)
            current = current / component
            directories[current] = current_descriptor
        if (_descriptor_identity_via_proc(current_descriptor)
                != _descriptor_identity_via_proc(
                    entry_parent_descriptor)):
            raise IdentityError(
                'config parent identity changed while pinning ancestry')
        return physical_parent, directories, descriptors
    except IdentityError:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as error:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise IdentityError(
            'failed to pin config ancestor directories') from error


def _pin_config_logical_directory(directory: Path, directories: dict,
                                  descriptors: list) -> int:
    """Resolve a logical directory only through already pinned ancestors."""
    directory = Path(os.path.normpath(os.fspath(directory)))
    if not directory.is_absolute():
        raise IdentityError('config dependency directory must be absolute')
    if directory in directories:
        return directories[directory]
    missing = []
    ancestor = directory
    while ancestor not in directories:
        if ancestor == ancestor.parent:
            raise IdentityError(
                'config dependency has no pinned logical ancestor')
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    directory_flags = (os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
                       | getattr(os, 'O_CLOEXEC', 0)
                       | getattr(os, 'O_NOFOLLOW', 0))
    current_descriptor = directories[ancestor]
    current = ancestor
    try:
        for component in reversed(missing):
            current_descriptor = os.open(
                component, directory_flags, dir_fd=current_descriptor)
            descriptors.append(current_descriptor)
            current = current / component
            directories[current] = current_descriptor
    except OSError as error:
        raise IdentityError(
            'failed to pin config dependency parent: {}'.format(
                directory)) from error
    return directories[directory]


def _config_dependency_closure(entry_path: Path,
                               entry_payload: bytes,
                               directories: dict,
                               descriptors: list) -> dict:
    """Snapshot the recursive local-relative config inheritance closure."""
    closure = {entry_path: entry_payload}
    pending = [(entry_path, entry_payload)]
    seen = {entry_path}
    while pending:
        current, payload = pending.pop()
        if current.suffix != '.py':
            raise IdentityError(
                'D13-N config dependencies must be Python files')
        for base in _relative_config_bases(payload, current):
            if '::' in base:
                continue
            resolved = Path(os.path.normpath(os.path.join(
                str(current.parent), base)))
            if resolved in seen:
                continue
            if len(seen) >= 256:
                raise IdentityError('config dependency closure is too large')
            base_parent_descriptor = _pin_config_logical_directory(
                resolved.parent, directories, descriptors)
            base_payload = _read_config_bytes(
                base_parent_descriptor, resolved.name, resolved)
            seen.add(resolved)
            closure[resolved] = base_payload
            pending.append((resolved, base_payload))
    return closure


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError('config snapshot write made no progress')
        offset += written


def _reserve_flat_config_snapshots(
        parent_descriptor: int, stem: str, closure: Mapping,
        records: dict) -> None:
    """Atomically reserve one sibling snapshot file for every source."""
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
             | getattr(os, 'O_CLOEXEC', 0)
             | getattr(os, 'O_NOFOLLOW', 0))
    for source in closure:
        for _ in range(128):
            name = '.{}.d13n-snapshot-{}.py'.format(
                stem, secrets.token_hex(12))
            try:
                descriptor = os.open(
                    name, flags, 0o600, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            record = {
                'name': name,
                'descriptor': descriptor,
                'identity': None,
            }
            records[source] = record
            record['identity'] = _open_descriptor_identity_for_cleanup(
                descriptor)
            break
        else:
            raise IdentityError(
                'failed to allocate exclusive flat config snapshot')


def _populate_flat_config_snapshots(closure: Mapping, records: Mapping) -> None:
    """Write rewritten source bytes and seal every reserved snapshot."""
    snapshot_names = {
        source: record['name'] for source, record in records.items()
    }
    for source, payload in closure.items():
        rewritten = _rewrite_config_local_bases(
            payload, source, snapshot_names)
        descriptor = records[source]['descriptor']
        _write_all(descriptor, rewritten)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)


def _cleanup_flat_config_snapshots(parent_descriptor: int,
                                   records: Mapping) -> None:
    """Remove only names still bound to their atomically created files."""
    first_error = None
    for record in reversed(list(records.values())):
        try:
            identity = record['identity']
            if identity is None:
                identity = _open_descriptor_identity_for_cleanup(
                    record['descriptor'])
            _cleanup_owned_snapshot(
                parent_descriptor, record['name'], identity)
        except IdentityError as error:
            if first_error is None:
                first_error = error
        finally:
            try:
                os.close(record['descriptor'])
            except OSError as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        if isinstance(first_error, IdentityError):
            raise first_error
        raise IdentityError(
            'failed to clean flat config snapshot') from first_error


def load_hashed_config_snapshot(
        path: Path, loader: Callable[[str], Any]) -> Tuple[Any, str]:
    """Load one hashed closure from atomically created flat snapshots."""
    path = _absolute_lexical_path(path)
    parent_flags = (os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
                    | getattr(os, 'O_CLOEXEC', 0))
    try:
        parent_descriptor = os.open(path.parent, parent_flags)
    except OSError as error:
        raise IdentityError(
            'failed to pin config parent: {}'.format(path.parent)) from error
    records = {}
    try:
        ancestry_descriptors = []
        try:
            physical_parent, pinned_directories, ancestry_descriptors = \
                _pin_config_ancestor_directories(parent_descriptor)
            entry_path = physical_parent / path.name
            payload = _read_config_bytes(
                parent_descriptor, path.name, entry_path)
            digest = hashlib.sha256(payload).hexdigest()
            closure = _config_dependency_closure(
                entry_path, payload, pinned_directories,
                ancestry_descriptors)
        finally:
            for descriptor in reversed(ancestry_descriptors):
                os.close(descriptor)
        _reserve_flat_config_snapshots(
            parent_descriptor, path.stem, closure, records)
        _populate_flat_config_snapshots(closure, records)
        try:
            snapshot_path = '/proc/self/fd/{}/{}'.format(
                parent_descriptor, records[entry_path]['name'])
            loaded = loader(snapshot_path)
        except Exception as error:
            raise IdentityError(
                'failed to load config snapshot: {}'.format(error)) from error
    except IdentityError:
        raise
    except OSError as error:
        raise IdentityError(
            'failed to create/write config snapshot beside {}'.format(path)) \
            from error
    finally:
        try:
            _cleanup_flat_config_snapshots(parent_descriptor, records)
        finally:
            os.close(parent_descriptor)
    return loaded, digest


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True).encode('utf-8')


def _canonical_json_sha256(payload: Mapping,
                           omit: Optional[str] = None) -> str:
    value = dict(payload)
    if omit is not None:
        value.pop(omit, None)
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _resolved_config_sha256(cfg: Config) -> str:
    try:
        return _canonical_json_sha256(cfg.to_dict())
    except Exception as error:
        raise IdentityError(
            'failed to canonicalize resolved config identity') from error


def validate_role_config(cfg: Config, role: str,
                         derive_adapter_free_parent: bool) -> str:
    """Require exact role/mouth metadata and raw test-only disabling."""
    if role not in ROLE_CONFIG:
        raise IdentityError('unknown D13-N role: {}'.format(role))
    expected_role, expected_mouth = ROLE_CONFIG[role]
    if cfg.get('d13n_role') != expected_role:
        raise IdentityError(
            'config d13n_role must be {!r} for {}'.format(
                expected_role, role))
    if cfg.get('d13n_mouth') != expected_mouth:
        raise IdentityError(
            'config d13n_mouth must be {!r} for {}'.format(
                expected_mouth, role))
    if role == 'stage0-parent':
        if not derive_adapter_free_parent:
            raise IdentityError(
                'stage0-parent requires --derive-adapter-free-parent')
    elif derive_adapter_free_parent:
        raise IdentityError(
            '--derive-adapter-free-parent is valid only for stage0-parent')
    if expected_mouth == 'raw13833':
        for field in RAW_DISABLED_FIELDS:
            if cfg.get(field, object()) is not None:
                raise IdentityError(
                    'raw role requires top-level {}=None'.format(field))
    return expected_mouth


def validate_stage0_checkpoint_identity(
        cfg: Config, checkpoint: Path, checkpoint_sha256: str,
        step: int) -> None:
    """Bind adapter-free stage0 evaluation to the configured E24 bytes."""
    if step != 24:
        raise IdentityError('stage0-parent requires the authoritative Epoch 24')
    expected = cfg.get('d13n_parent_sha256')
    if (type(expected) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected) is None):
        raise IdentityError(
            'config d13n_parent_sha256 must be a lowercase SHA256')
    if checkpoint_sha256 != expected:
        raise IdentityError(
            'stage0-parent checkpoint SHA256 does not match config: {}'.format(
                checkpoint))


def derive_adapter_free_parent(cfg: Config) -> Tuple[Config, list]:
    """Deep-copy and delete exactly the three frozen adapter-owned fields."""
    derived = Config(copy.deepcopy(cfg.to_dict()))
    try:
        del derived.model.bbox_head.existence_loss_weight
        del derived.model.freeze_except_patterns
    except (AttributeError, KeyError, TypeError) as error:
        raise IdentityError(
            'adapter-free parent deletion fields are missing') from error
    hooks = list(derived.get('custom_hooks', []))
    matches = [
        index for index, hook in enumerate(hooks)
        if isinstance(hook, Mapping)
        and hook.get('type') == 'D13NParentEvalModeHook'
    ]
    if len(matches) != 1:
        raise IdentityError(
            'adapter-free parent requires a single '
            'D13NParentEvalModeHook')
    hook_index = matches[0]
    del hooks[hook_index]
    derived.custom_hooks = hooks
    deleted = [
        DERIVED_DELETED_FIELDS[0],
        DERIVED_DELETED_FIELDS[1],
        'custom_hooks[{}]'.format(hook_index),
    ]
    if deleted != DERIVED_DELETED_FIELDS:
        raise IdentityError(
            'D13NParentEvalModeHook must be custom_hooks[0]')
    return derived, deleted


def _metric_type(metric: Any) -> Any:
    return metric.get('type') if isinstance(metric, Mapping) else None


def configure_test_config(cfg: Config, checkpoint: os.PathLike,
                          output: os.PathLike) -> Config:
    """Bind the exact checkpoint and append/replace only the dump metric."""
    configured = Config(copy.deepcopy(cfg.to_dict()))
    configured.launcher = 'pytorch'
    configured.load_from = str(_absolute_lexical_path(checkpoint))
    configured.resume = False
    current = configured.get('test_evaluator')
    evaluators = list(current) if isinstance(current, (list, tuple)) else [
        current]
    if not evaluators or any(item is None for item in evaluators):
        raise IdentityError('test_evaluator must contain the official metric')
    dump_cfg = {
        'type': 'D13NNoReplaceDumpResults',
        'out_file_path': str(_absolute_lexical_path(output)),
        'collect_device': 'cpu',
    }
    dump_indices = [
        index for index, item in enumerate(evaluators)
        if _metric_type(item) in ('DumpResults', 'D13NNoReplaceDumpResults')
    ]
    if len(dump_indices) > 1:
        raise IdentityError('test_evaluator contains multiple dump metrics')
    if dump_indices:
        evaluators[dump_indices[0]] = dump_cfg
    else:
        evaluators.append(dump_cfg)
    if not any(_metric_type(item) == 'DOTAMetric' for item in evaluators):
        raise IdentityError('test_evaluator must preserve DOTAMetric')
    configured.test_evaluator = evaluators
    return configured


def _unwrap_state_dict(checkpoint: Any) -> dict:
    if not isinstance(checkpoint, Mapping):
        raise IdentityError('checkpoint must be a mapping')
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if not isinstance(state, Mapping):
        raise IdentityError('checkpoint model state must be a mapping')
    normalized = {}
    for key, value in state.items():
        if type(key) is not str:
            raise IdentityError('checkpoint state keys must be strings')
        normalized[re.sub(r'^module\.', '', key)] = value
    return normalized


def validate_stage0_checkpoint_load(
        model: Any, checkpoint: Path,
        checkpoint_loader: Callable[[Path], Any]) -> None:
    """Load adapter-free E24 and require zero missing/unexpected keys."""
    try:
        payload = checkpoint_loader(checkpoint)
        incompatible = model.load_state_dict(
            _unwrap_state_dict(payload), strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
    except IdentityError:
        raise
    except Exception as error:
        raise IdentityError(
            'failed to load stage0-parent checkpoint') from error
    if missing:
        raise IdentityError(
            'stage0-parent checkpoint has missing keys: {}'.format(missing))
    if unexpected:
        raise IdentityError(
            'stage0-parent checkpoint has unexpected keys: {}'.format(
                unexpected))


def _model_for_state_load(model: Any) -> Any:
    return model.module if hasattr(model, 'module') else model


def _validate_global_runtime_environment() -> None:
    """Fail before model construction on launcher-global GPU/NCCL drift."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != EXPECTED_CUDA_VISIBLE_DEVICES:
        raise IdentityError('CUDA_VISIBLE_DEVICES must be 5,6,7,8,9')
    if os.environ.get('NCCL_P2P_DISABLE') != '1':
        raise IdentityError('NCCL_P2P_DISABLE must equal 1')
    if os.environ.get('NCCL_IB_DISABLE') != '1':
        raise IdentityError('NCCL_IB_DISABLE must equal 1')

def _identity_envelope(operation: Callable[[], Mapping]) -> dict:
    try:
        return {'ok': True, 'value': dict(operation())}
    except IdentityError as error:
        return {'ok': False, 'kind': 'identity', 'message': str(error)}
    except Exception as error:
        return {
            'ok': False,
            'kind': 'identity',
            'message': 'post-init identity check failed: {}'.format(error),
        }


def _require_sha256_field(payload: Mapping, field: str) -> str:
    value = payload.get(field)
    if (type(value) is not str
            or re.fullmatch(r'[0-9a-f]{64}', value) is None):
        raise IdentityError('{} must be a lowercase SHA256'.format(field))
    return value


def converge_post_init_identity(
        runtime: RuntimeFacade,
        local_operation: Callable[[], Mapping]) -> dict:
    """Converge every rank-local identity/load result in one all-gather."""
    local_envelope = _identity_envelope(local_operation)
    gathered = runtime.all_gather_object(local_envelope)
    if not isinstance(gathered, list) or len(gathered) != EXPECTED_WORLD_SIZE:
        raise IdentityError(
            'post-init identity gather must return exactly five envelopes')
    failures = []
    values = []
    for index, envelope in enumerate(gathered):
        if not isinstance(envelope, Mapping):
            failures.append('rank {}: malformed identity envelope'.format(
                index))
            continue
        if envelope.get('ok') is not True:
            failures.append('rank {}: {}'.format(
                index, envelope.get('message', 'identity failure')))
            continue
        value = envelope.get('value')
        if not isinstance(value, Mapping):
            failures.append('rank {}: malformed identity payload'.format(
                index))
            continue
        values.append(dict(value))
    if failures:
        raise IdentityError('; '.join(failures))
    if len(values) != EXPECTED_WORLD_SIZE:
        raise IdentityError('post-init identity payload count mismatch')

    ranks = [value.get('rank') for value in values]
    local_ranks = [value.get('local_rank') for value in values]
    world_sizes = [value.get('world_size') for value in values]
    env_world_sizes = [value.get('env_world_size') for value in values]
    topology_values = ranks + local_ranks + world_sizes + env_world_sizes
    if any(type(value) is not int for value in topology_values):
        raise IdentityError('rank/world topology values must be built-in ints')
    if ranks != EXPECTED_LOCAL_RANKS:
        raise IdentityError('distributed ranks must be exactly 0,1,2,3,4')
    if local_ranks != EXPECTED_LOCAL_RANKS:
        raise IdentityError('local ranks must be exactly 0,1,2,3,4')
    if world_sizes != [EXPECTED_WORLD_SIZE] * EXPECTED_WORLD_SIZE:
        raise IdentityError('distributed world_size must equal 5 on every rank')
    if env_world_sizes != [EXPECTED_WORLD_SIZE] * EXPECTED_WORLD_SIZE:
        raise IdentityError('WORLD_SIZE must equal 5 on every rank')

    hash_fields = (
        'config_sha256',
        'checkpoint_sha256',
        'resolved_config_sha256',
    )
    expected_hashes = {
        field: _require_sha256_field(values[0], field)
        for field in hash_fields
    }
    for index, value in enumerate(values[1:], start=1):
        observed = {
            field: _require_sha256_field(value, field)
            for field in hash_fields
        }
        if observed != expected_hashes:
            raise IdentityError(
                'input/resolved hash identity differs on rank {}'.format(
                    index))
    return {
        'rank': (local_envelope.get('value', {}).get('rank')
                 if local_envelope.get('ok') is True else None),
        'world_size': EXPECTED_WORLD_SIZE,
        'local_ranks': local_ranks,
        **expected_hashes,
    }


def _local_post_init_identity(
        runtime: RuntimeFacade, config_sha256: str,
        checkpoint_sha256: str, resolved_config_sha256: str,
        stage0_load: Optional[Callable[[], None]] = None) -> dict:
    rank, world_size = runtime.get_dist_info()
    try:
        env_world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    except (KeyError, TypeError, ValueError) as error:
        raise IdentityError(
            'WORLD_SIZE and LOCAL_RANK must be integer environment values') \
            from error
    if (type(rank) is not int or isinstance(rank, bool)
            or rank not in EXPECTED_LOCAL_RANKS):
        raise IdentityError('distributed rank must be in 0..4')
    if type(world_size) is not int or world_size != EXPECTED_WORLD_SIZE:
        raise IdentityError('distributed world_size must equal 5')
    if env_world_size != EXPECTED_WORLD_SIZE:
        raise IdentityError('WORLD_SIZE must equal 5')
    if (type(local_rank) is not int or isinstance(local_rank, bool)
            or local_rank not in EXPECTED_LOCAL_RANKS):
        raise IdentityError('LOCAL_RANK must be in 0..4')
    if stage0_load is not None:
        stage0_load()
    return {
        'rank': rank,
        'world_size': world_size,
        'env_world_size': env_world_size,
        'local_rank': local_rank,
        'config_sha256': config_sha256,
        'checkpoint_sha256': checkpoint_sha256,
        'resolved_config_sha256': resolved_config_sha256,
    }


def _infer_checkpoint_step(path: Path) -> int:
    match = re.search(r'(?:^|/)epoch_(\d+)\.pth$', str(path))
    if match is None:
        raise IdentityError(
            'checkpoint path must end in epoch_INTEGER.pth')
    return int(match.group(1))


def _finite_metric(metrics: Mapping, key: str) -> float:
    if key not in metrics:
        raise RuntimeError('Runner.test metrics lack {}'.format(key))
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError('{} must be a real scalar'.format(key))
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError('{} must be finite'.format(key))
    return converted


def _validate_dump(runtime: RuntimeFacade, path: Path,
                   mouth: str) -> dict:
    expected_records = RECORDS_BY_MOUTH[mouth]
    summary = dict(runtime.validate_records(
        runtime.load_records(path),
        expected_records=expected_records,
        queries_per_image=QUERIES_PER_IMAGE,
        num_classes=18))
    expected = {
        'records': expected_records,
        'unique_image_ids': expected_records,
        'prediction_rows': expected_records * QUERIES_PER_IMAGE,
        'all_cpu_finite': True,
    }
    exact_integer_fields = (
        'records', 'unique_image_ids', 'prediction_rows')
    exact_types = all(
        type(summary.get(field)) is int for field in exact_integer_fields)
    exact_types = exact_types and summary.get('all_cpu_finite') is True
    if (set(summary) != set(expected) or not exact_types
            or summary != expected):
        raise RuntimeError(
            'prediction dump validation summary mismatch: {}'.format(summary))
    return summary


def _build_metrics_record(role: str, mouth: str, step: int,
                          metrics: Mapping, config_path: Path,
                          config_sha256: str, checkpoint_path: Path,
                          checkpoint_sha256: str,
                          summary: Mapping) -> dict:
    record = {
        'schema': 'd13n-official-metrics-v1',
        'role': role,
        'mouth': mouth,
        'step': step,
        'dota/mAP': _finite_metric(metrics, 'dota/mAP'),
        'dota/AP50': _finite_metric(metrics, 'dota/AP50'),
        'config_path': str(config_path),
        'config_sha256': config_sha256,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': checkpoint_sha256,
        'records': summary['records'],
        'queries_per_image': QUERIES_PER_IMAGE,
        'prediction_rows': summary['prediction_rows'],
    }
    record['report_sha256'] = _canonical_json_sha256(
        record, omit='report_sha256')
    return record


def _phase_error(error: Exception) -> dict:
    if isinstance(error, OutputCollisionError):
        kind = 'collision'
    elif isinstance(error, IdentityError):
        kind = 'identity'
    else:
        kind = 'runtime'
    return {'ok': False, 'kind': kind, 'message': str(error)}


def _raise_phase_error(payload: Mapping) -> None:
    message = str(payload.get('message', 'distributed rank-0 phase failed'))
    if payload.get('kind') == 'collision':
        raise OutputCollisionError(message)
    if payload.get('kind') == 'identity':
        raise IdentityError(message)
    raise RuntimeError(message)


def _publish_json_output(path: Path, payload: Mapping) -> None:
    """Classify FileExists only when this output is actually occupied."""
    try:
        publish_json_noreplace(path, payload)
    except FileExistsError as error:
        if _path_collisions(path):
            raise OutputCollisionError(str(error)) from error
        raise


def _rank0_phase(runtime: RuntimeFacade, rank: int,
                 operation: Callable[[], Any]) -> Any:
    envelope = [None]
    if rank == 0:
        try:
            envelope[0] = {'ok': True, 'value': operation()}
        except Exception as error:
            envelope[0] = _phase_error(error)
    runtime.broadcast_object_list(envelope)
    if not isinstance(envelope[0], Mapping):
        raise RuntimeError('rank-0 phase broadcast is malformed')
    if not envelope[0].get('ok'):
        _raise_phase_error(envelope[0])
    return envelope[0].get('value')


def _build_identity(
        role: str, mouth: str, config_path: Path, config_sha256: str,
        resolved_config_sha256: str, deleted_fields: list,
        checkpoint_path: Path, checkpoint_sha256: str,
        predictions_path: Path, metrics_path: Path, summary: Mapping,
        local_ranks: list) -> dict:
    return {
        'schema': 'd13n-test-identity-v1',
        'role': role,
        'mouth': mouth,
        'config_path': str(config_path),
        'config_sha256': config_sha256,
        'resolved_config_sha256': resolved_config_sha256,
        'derived_parent': bool(deleted_fields),
        'derived_deleted_fields': deleted_fields,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': checkpoint_sha256,
        'predictions_path': str(predictions_path),
        'predictions_sha256': sha256_file(predictions_path),
        'official_metrics_path': str(metrics_path),
        'official_metrics_sha256': sha256_file(metrics_path),
        'records': summary['records'],
        'unique_image_ids': summary['unique_image_ids'],
        'queries_per_image': QUERIES_PER_IMAGE,
        'prediction_rows': summary['prediction_rows'],
        'finite': summary['all_cpu_finite'],
        'world_size': EXPECTED_WORLD_SIZE,
        'local_ranks': local_ranks,
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'nccl_p2p_disable': os.environ['NCCL_P2P_DISABLE'],
        'nccl_ib_disable': os.environ['NCCL_IB_DISABLE'],
    }


def execute(args: argparse.Namespace,
            runtime: Optional[RuntimeFacade] = None) -> int:
    """Execute one already-parsed D13-N test invocation."""
    runtime = runtime or RuntimeFacade()
    out, metrics_out, identity_out = resolve_and_preflight_outputs(
        args.out, args.metrics_out, args.identity_out)
    config_path = _require_input_file(args.config, 'config')
    checkpoint_path = _require_input_file(args.checkpoint, 'checkpoint')
    step = _infer_checkpoint_step(checkpoint_path)
    with open_checkpoint_binding(checkpoint_path) as checkpoint_binding:
        cfg, config_sha = load_hashed_config_snapshot(
            config_path, runtime.load_config)
        checkpoint_sha = checkpoint_binding.sha256
        if not isinstance(cfg, Config):
            try:
                cfg = Config(cfg)
            except Exception as error:
                raise IdentityError(
                    'config loader did not return a config') from error
        mouth = validate_role_config(
            cfg, args.role, args.derive_adapter_free_parent)
        deleted_fields = []
        if args.role == 'stage0-parent':
            validate_stage0_checkpoint_identity(
                cfg, checkpoint_path, checkpoint_sha, step)
            cfg, deleted_fields = derive_adapter_free_parent(cfg)
        # The canonical resolved identity retains the positional absolute path.
        # The runner's private load boundary is rebound to the held inode only
        # after MMEngine initializes distributed state.
        cfg = configure_test_config(cfg, checkpoint_path, out)
        resolved_sha = _resolved_config_sha256(cfg)
        _validate_global_runtime_environment()
        resolve_and_preflight_outputs(out, metrics_out, identity_out)
        runner = runtime.build_runner(cfg)
        snapshot_holder = {'value': None}
        try:
            snapshot_rank, _ = runtime.get_dist_info()
            snapshot_authority, snapshot_owner = establish_checkpoint_snapshot(
                runtime, snapshot_rank, checkpoint_binding)
            snapshot_holder['value'] = snapshot_owner

            def local_snapshot_identity() -> dict:
                if snapshot_holder['value'] is None:
                    snapshot_holder['value'] = _open_checkpoint_snapshot(
                        snapshot_authority)
                snapshot = snapshot_holder['value']
                runner._load_from = str(snapshot.runtime_path)

                def stage0_load() -> None:
                    validate_stage0_checkpoint_load(
                        _model_for_state_load(runner.model),
                        snapshot.runtime_path,
                        runtime.checkpoint_loader)

                return _local_post_init_identity(
                    runtime,
                    config_sha,
                    checkpoint_sha,
                    resolved_sha,
                    stage0_load if args.role == 'stage0-parent' else None)

            converged = converge_post_init_identity(
                runtime, local_snapshot_identity)
            rank = converged['rank']
            local_ranks = converged['local_ranks']
            if args.role == 'stage0-parent':
                runner._has_loaded = True
            try:
                metrics = runner.test()
            except D13NDumpCollisionError as error:
                if error.out_file_path == str(out):
                    raise OutputCollisionError(
                        error.original_message) from error
                raise
            if not isinstance(metrics, Mapping):
                raise RuntimeError('Runner.test must return a metrics mapping')

            def publish_metrics_phase() -> dict:
                summary = _validate_dump(runtime, out, mouth)
                record = _build_metrics_record(
                    args.role, mouth, step, metrics, config_path, config_sha,
                    checkpoint_path, checkpoint_sha, summary)
                _publish_json_output(metrics_out, record)
                return {'summary': summary, 'metrics': record}

            phase = _rank0_phase(runtime, rank, publish_metrics_phase)
            runtime.barrier()

            def publish_identity_phase() -> dict:
                identity = _build_identity(
                    args.role, mouth, config_path, config_sha, resolved_sha,
                    deleted_fields, checkpoint_path, checkpoint_sha, out,
                    metrics_out, phase['summary'], local_ranks)
                _publish_json_output(identity_out, identity)
                return identity

            _rank0_phase(runtime, rank, publish_identity_phase)
            runtime.barrier()
            return 0
        finally:
            snapshot = snapshot_holder['value']
            if snapshot is not None:
                snapshot.close()


def build_torchrun_argv(
        config: os.PathLike, checkpoint: os.PathLike, role: str,
        out: os.PathLike, metrics_out: os.PathLike,
        identity_out: os.PathLike, port: int,
        derive_adapter_free_parent: bool = False) -> list:
    """Return the frozen five-process torchrun argv; environment is separate."""
    if role not in ROLES:
        raise IdentityError('unknown D13-N role: {}'.format(role))
    if derive_adapter_free_parent and role != 'stage0-parent':
        raise IdentityError(
            'derive flag is valid only for stage0-parent')
    if role == 'stage0-parent' and not derive_adapter_free_parent:
        raise IdentityError('stage0-parent requires the derive flag')
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise IdentityError('port must be an integer in 1..65535')
    if port not in ROLE_PORTS[role]:
        raise IdentityError(
            '{} port must be one of {}'.format(role, ROLE_PORTS[role]))
    argv = [
        TORCHRUN_PYTHON,
        '-m',
        'torch.distributed.run',
        '--nproc_per_node=5',
        '--master_port={}'.format(port),
        TEST_SCRIPT,
        str(config),
        str(checkpoint),
        '--launcher',
        'pytorch',
        '--role',
        role,
        '--out',
        str(out),
        '--metrics-out',
        str(metrics_out),
        '--identity-out',
        str(identity_out),
    ]
    if derive_adapter_free_parent:
        argv.append('--derive-adapter-free-parent')
    return argv


def main(argv: Optional[Sequence[str]] = None,
         runtime: Optional[RuntimeFacade] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args, runtime=runtime)
    except OutputCollisionError as error:
        print('D13-N output collision: {}'.format(error), file=sys.stderr)
        return 3
    except IdentityError as error:
        print('D13-N identity failure: {}'.format(error), file=sys.stderr)
        return 4
    except Exception as error:
        print('D13-N runtime failure: {}'.format(error), file=sys.stderr)
        return 5


if __name__ == '__main__':
    raise SystemExit(main())
