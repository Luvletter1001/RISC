import json
import os
from pathlib import Path

import pytest


def _api():
    from projects.OVCapFlow.ov_capflow import no_replace

    return no_replace


def test_publish_bytes_is_exclusive_fsyncs_file_and_directory(
        tmp_path, monkeypatch):
    api = _api()
    target = tmp_path / 'nested' / 'artifact.bin'
    fsynced = []
    real_fsync = os.fsync

    def record_fsync(fd):
        fsynced.append(os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(api.os, 'fsync', record_fsync)
    api.publish_bytes_noreplace(target, b'payload')

    assert target.read_bytes() == b'payload'
    assert len(fsynced) == 2
    assert not list(target.parent.glob(target.name + '.pending.*'))


def test_publish_stream_writes_directly_to_pending_file(tmp_path):
    api = _api()
    target = tmp_path / 'large.bin'
    stream_types = []

    def writer(stream):
        stream_types.append(type(stream))
        stream.write(b'first')
        stream.write(b'-second')

    api.publish_stream_noreplace(target, writer)

    assert target.read_bytes() == b'first-second'
    assert stream_types and hasattr(stream_types[0], 'write')


def test_publish_json_is_sorted_compact_utf8_jsonl(tmp_path):
    api = _api()
    target = tmp_path / 'report.jsonl'

    api.publish_json_noreplace(target, {'z': 1, 'message': '中文', 'a': [2]})

    assert target.read_bytes() == (
        '{"a":[2],"message":"中文","z":1}\n'.encode('utf-8'))
    assert json.loads(target.read_text(encoding='utf-8')) == {
        'a': [2], 'message': '中文', 'z': 1}


@pytest.mark.parametrize('collision_kind', ['final', 'pending'])
def test_publish_refuses_existing_final_or_pending_without_changing_bytes(
        tmp_path, collision_kind):
    api = _api()
    target = tmp_path / 'artifact.bin'
    collision = target
    if collision_kind == 'pending':
        collision = tmp_path / 'artifact.bin.pending.123.0'
    collision.write_bytes(b'preserve-me')

    with pytest.raises(FileExistsError, match='output collision'):
        api.publish_bytes_noreplace(target, b'new-bytes')

    assert collision.read_bytes() == b'preserve-me'
    if collision_kind == 'pending':
        assert not target.exists()


def test_publish_runs_collision_scan_before_and_after_directory_creation(
        tmp_path, monkeypatch):
    api = _api()
    target = tmp_path / 'nested' / 'artifact.bin'
    scans = []
    real_collision_paths = api._collision_paths

    def record_scan(path):
        scans.append(Path(path))
        return real_collision_paths(path)

    monkeypatch.setattr(api, '_collision_paths', record_scan)
    api.publish_bytes_noreplace(target, b'payload')

    lexical_target = Path(os.path.abspath(target))
    assert scans == [lexical_target, lexical_target]


def test_failed_link_preserves_only_unpublished_pending_evidence(
        tmp_path, monkeypatch):
    api = _api()
    target = tmp_path / 'artifact.bin'

    def fail_link(_source, _target):
        raise FileExistsError('simulated publication race')

    monkeypatch.setattr(api.os, 'link', fail_link)
    with pytest.raises(FileExistsError, match='simulated publication race'):
        api.publish_bytes_noreplace(target, b'evidence')

    assert not target.exists()
    pending = list(tmp_path.glob('artifact.bin.pending.*'))
    assert len(pending) == 1
    assert pending[0].read_bytes() == b'evidence'


@pytest.mark.parametrize('target_kind', ['dangling', 'valid'])
def test_publish_refuses_requested_final_symlink_without_following_it(
        tmp_path, target_kind):
    api = _api()
    target = tmp_path / 'artifact.bin'
    link_target = tmp_path / 'link-target.bin'
    if target_kind == 'valid':
        link_target.write_bytes(b'target-bytes')
    target.symlink_to(link_target.name)
    original_link = os.readlink(target)

    with pytest.raises(FileExistsError, match='output collision'):
        api.publish_bytes_noreplace(target, b'new-bytes')

    assert target.is_symlink()
    assert os.path.lexists(target)
    assert os.readlink(target) == original_link
    if target_kind == 'valid':
        assert link_target.read_bytes() == b'target-bytes'
    else:
        assert not os.path.lexists(link_target)


@pytest.mark.parametrize('pending_kind', ['file', 'dangling_symlink'])
def test_pending_scan_treats_final_glob_metacharacters_literally(
        tmp_path, pending_kind):
    api = _api()
    target = tmp_path / 'artifact[1]*?.bin'
    pending = tmp_path / (target.name + '.pending.123.0')
    if pending_kind == 'file':
        pending.write_bytes(b'pending-evidence')
    else:
        pending.symlink_to('missing-pending-target')
    original_link = os.readlink(pending) if pending.is_symlink() else None

    with pytest.raises(FileExistsError, match='output collision'):
        api.publish_bytes_noreplace(target, b'new-bytes')

    assert not os.path.lexists(target)
    assert os.path.lexists(pending)
    if pending_kind == 'file':
        assert pending.read_bytes() == b'pending-evidence'
    else:
        assert pending.is_symlink()
        assert os.readlink(pending) == original_link


@pytest.mark.parametrize(
    'non_finite',
    [float('nan'), float('inf'), float('-inf')],
    ids=['nan', 'positive_inf', 'negative_inf'])
def test_publish_json_rejects_non_finite_values_without_filesystem_effects(
        tmp_path, non_finite):
    api = _api()
    target = tmp_path / 'not-created' / 'report.json'

    with pytest.raises(ValueError, match='Out of range float values'):
        api.publish_json_noreplace(target, {'value': non_finite})

    assert not target.parent.exists()
    assert not os.path.lexists(target)
