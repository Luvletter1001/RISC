"""Publish immutable evidence files without replacing existing bytes."""

import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, List, Union


PathLike = Union[str, os.PathLike]
StreamWriter = Callable[[BinaryIO], None]


def _absolute_lexical_path(path: PathLike) -> Path:
    """Make a path absolute without following its final symlink."""
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def _collision_paths(path: PathLike) -> List[str]:
    """Return an existing final or any same-final pending path."""
    path = _absolute_lexical_path(path)
    collisions = set()
    if os.path.lexists(path):
        collisions.add(str(path))
    if path.parent.exists():
        pending_prefix = path.name + '.pending.'
        for pending in path.parent.iterdir():
            if (pending.name.startswith(pending_prefix)
                    and os.path.lexists(pending)):
                collisions.add(str(pending))
    return sorted(collisions)


def _raise_for_collisions(path: Path) -> None:
    collisions = _collision_paths(path)
    if collisions:
        raise FileExistsError('output collision: ' + ', '.join(collisions))


def publish_stream_noreplace(path: PathLike, writer: StreamWriter) -> None:
    """Stream one file to exclusive staging and atomically link its final.

    Failed or interrupted publications intentionally retain their pending file
    as evidence. Only a pending file whose final link and directory fsync both
    succeed is removed.
    """
    if not callable(writer):
        raise TypeError('writer must be callable')
    final_path = _absolute_lexical_path(path)
    _raise_for_collisions(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _raise_for_collisions(final_path)

    pending_path = final_path.with_name(
        f'{final_path.name}.pending.{os.getpid()}.0')
    with pending_path.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())

    os.link(pending_path, final_path)
    directory_fd = os.open(final_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    pending_path.unlink()


def publish_bytes_noreplace(path: PathLike, payload: bytes) -> None:
    """Publish an immutable bytes payload."""
    if type(payload) is not bytes:
        raise TypeError('payload must be bytes')
    publish_stream_noreplace(path, lambda stream: stream.write(payload))


def publish_json_noreplace(path: PathLike, payload: Any) -> None:
    """Publish sorted compact UTF-8 JSON as exactly one newline-ended line."""
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=True) + '\n').encode('utf-8')
    publish_bytes_noreplace(path, encoded)


__all__ = [
    'publish_bytes_noreplace',
    'publish_json_noreplace',
    'publish_stream_noreplace',
]
