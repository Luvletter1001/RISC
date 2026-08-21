import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / 'tools' / 'risc_n0o' / 'run_openrsd_n0o_fold.py')


def load_runner():
    module_name = 'run_openrsd_n0o_fold'
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_receipt(path, runner, **overrides):
    value = {
        'status': 'GPU_SMOKE_AUTHORIZED',
        'input_manifest_sha256': runner.V3_MANIFEST_SHA256,
        'preflight_receipt_sha256': 'b' * 64,
        'allowed_folds': ['c4_a'],
        'max_scenes_per_fold': 1,
    }
    value.update(overrides)
    path.write_text(json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(',', ':'), sort_keys=True) + '\n')
    return path


def test_import_is_stdlib_only_and_absent_receipt_blocks_lazy_runtime(tmp_path):
    before_torch = 'torch' in sys.modules
    runner = load_runner()
    events = []

    with pytest.raises(runner.AuthorizationError, match='missing'):
        runner.run_authorized_fold(
            authorization=tmp_path / 'missing.json',
            fold_id='c4_a',
            requested_scenes=1,
            expected_preflight_sha256='b' * 64,
            lazy_runtime_loader=lambda: events.append('runtime'))

    assert ('torch' in sys.modules) is before_torch
    assert events == []


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'status': 'PREFLIGHT_READY_GPU_NOT_AUTHORIZED'}, 'status'),
        ({'input_manifest_sha256': 'a' * 64}, 'input manifest'),
        ({'preflight_receipt_sha256': 'c' * 64}, 'preflight'),
        ({'allowed_folds': ['c8_a']}, 'fold'),
        ({'max_scenes_per_fold': 0}, 'positive'),
    ],
)
def test_receipt_drift_blocks_before_lazy_runtime(
        tmp_path, overrides, message):
    runner = load_runner()
    receipt = write_receipt(tmp_path / 'authorization.json', runner, **overrides)
    events = []

    with pytest.raises(runner.AuthorizationError, match=message):
        runner.run_authorized_fold(
            authorization=receipt,
            fold_id='c4_a',
            requested_scenes=1,
            expected_preflight_sha256='b' * 64,
            lazy_runtime_loader=lambda: events.append('runtime'))

    assert events == []


def test_scene_request_cannot_exceed_authorized_bound(tmp_path):
    runner = load_runner()
    receipt = write_receipt(tmp_path / 'authorization.json', runner)

    with pytest.raises(runner.AuthorizationError, match='exceeds'):
        runner.run_authorized_fold(
            authorization=receipt,
            fold_id='c4_a',
            requested_scenes=2,
            expected_preflight_sha256='b' * 64,
            lazy_runtime_loader=lambda: object())


def test_valid_receipt_calls_lazy_runtime_once(tmp_path):
    runner = load_runner()
    receipt = write_receipt(tmp_path / 'authorization.json', runner)
    events = []

    result = runner.run_authorized_fold(
        authorization=receipt,
        fold_id='c4_a',
        requested_scenes=1,
        expected_preflight_sha256='b' * 64,
        lazy_runtime_loader=lambda: (
            events.append('runtime') or
            (lambda fold_id, scenes: (fold_id, scenes))))

    assert events == ['runtime']
    assert result == ('c4_a', 1)
