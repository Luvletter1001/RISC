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


def write_preflight_receipt(path, runner, *, runner_sha256=None):
    directory = path.parent
    report = runner._canonical_json_bytes({
        'schema': 'risc-openrsd-n0o-preflight-report-v2',
        'sources': {
            'input_manifest': runner.V3_MANIFEST_SHA256,
            'preflight_source': runner._sha256_file(
                SCRIPT_PATH.with_name('preflight_openrsd_n0o.py')),
            'protocol_source': runner._sha256_file(
                SCRIPT_PATH.with_name('openrsd_n0o_protocol.py')),
            'runner_source': runner_sha256 or runner._sha256_file(SCRIPT_PATH),
            'scene_plan': runner.V3_SCENE_PLAN_SHA256,
            'support_ledger': runner.V3_SUPPORT_LEDGER_SHA256,
        },
    })
    primary = {
        'model_ledger.jsonl': b'{"scene_id":"P0001"}\n',
        'module_origins.json': b'{"modules":[]}\n',
        'preflight_report.json': report,
        'resolved_config.py': b'model = dict()\n',
    }
    for name, content in primary.items():
        (directory / name).write_bytes(content)
    value = {
        'schema': 'risc-openrsd-n0o-preflight-receipt-v2',
        'status': 'PREFLIGHT_READY_GPU_NOT_AUTHORIZED',
        'artifacts': {
            name: {
                'byte_count': len(content),
                'sha256': runner._sha256_file(directory / name),
            }
            for name, content in sorted(primary.items())
        },
        'gpu_smoke_authorized': False,
    }
    path.write_bytes(runner._canonical_json_bytes(value))
    return path


def write_receipt(path, runner, preflight_path, **overrides):
    value = {
        'status': 'GPU_SMOKE_AUTHORIZED',
        'input_manifest_sha256': runner.V3_MANIFEST_SHA256,
        'preflight_receipt_sha256': runner._sha256_file(preflight_path),
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
    preflight = write_preflight_receipt(tmp_path / 'preflight.json', runner)
    runner.COMMITTED_PREFLIGHT_RECEIPT = preflight
    events = []

    with pytest.raises(runner.AuthorizationError, match='missing'):
        runner.run_authorized_fold(
            authorization=tmp_path / 'missing.json',
            fold_id='c4_a',
            requested_scenes=1,
            lazy_runtime_loader=lambda: events.append('runtime'))

    assert ('torch' in sys.modules) is before_torch
    assert events == []
    assert {action.dest for action in runner.build_parser()._actions} == {
        'authorization', 'fold_id', 'help', 'requested_scenes'}


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
    preflight = write_preflight_receipt(tmp_path / 'preflight.json', runner)
    runner.COMMITTED_PREFLIGHT_RECEIPT = preflight
    if 'preflight_receipt_sha256' in overrides:
        overrides['preflight_receipt_sha256'] = 'c' * 64
    receipt = write_receipt(
        tmp_path / 'authorization.json', runner, preflight, **overrides)
    events = []

    with pytest.raises(runner.AuthorizationError, match=message):
        runner.run_authorized_fold(
            authorization=receipt,
            fold_id='c4_a',
            requested_scenes=1,
            lazy_runtime_loader=lambda: events.append('runtime'))

    assert events == []


def test_scene_request_cannot_exceed_authorized_bound(tmp_path):
    runner = load_runner()
    preflight = write_preflight_receipt(tmp_path / 'preflight.json', runner)
    runner.COMMITTED_PREFLIGHT_RECEIPT = preflight
    receipt = write_receipt(tmp_path / 'authorization.json', runner, preflight)

    with pytest.raises(runner.AuthorizationError, match='exceeds'):
        runner.run_authorized_fold(
            authorization=receipt,
            fold_id='c4_a',
            requested_scenes=2,
            lazy_runtime_loader=lambda: object())


def test_scene_bound_cannot_exceed_sealed_fold_size(tmp_path):
    runner = load_runner()
    preflight = write_preflight_receipt(tmp_path / 'preflight.json', runner)
    runner.COMMITTED_PREFLIGHT_RECEIPT = preflight
    receipt = write_receipt(
        tmp_path / 'authorization.json', runner, preflight,
        max_scenes_per_fold=41)

    with pytest.raises(runner.AuthorizationError, match='40'):
        runner.run_authorized_fold(
            authorization=receipt,
            fold_id='c4_a',
            requested_scenes=41,
            lazy_runtime_loader=lambda: object())


def test_committed_artifact_drift_blocks_lazy_runtime(tmp_path):
    runner = load_runner()
    preflight = write_preflight_receipt(tmp_path / 'preflight.json', runner)
    runner.COMMITTED_PREFLIGHT_RECEIPT = preflight
    receipt = write_receipt(tmp_path / 'authorization.json', runner, preflight)
    (tmp_path / 'model_ledger.jsonl').write_bytes(b'drift\n')
    events = []

    with pytest.raises(runner.AuthorizationError, match='artifact'):
        runner.run_authorized_fold(
            authorization=receipt,
            fold_id='c4_a',
            requested_scenes=1,
            lazy_runtime_loader=lambda: events.append('runtime'))
    assert events == []


def test_committed_report_must_chain_current_runner_source(tmp_path):
    runner = load_runner()
    preflight = write_preflight_receipt(
        tmp_path / 'preflight.json', runner, runner_sha256='0' * 64)
    runner.COMMITTED_PREFLIGHT_RECEIPT = preflight
    receipt = write_receipt(tmp_path / 'authorization.json', runner, preflight)
    events = []

    with pytest.raises(runner.AuthorizationError, match='source'):
        runner.run_authorized_fold(
            authorization=receipt,
            fold_id='c4_a',
            requested_scenes=1,
            lazy_runtime_loader=lambda: events.append('runtime'))
    assert events == []


def test_valid_receipt_calls_lazy_runtime_once(tmp_path):
    runner = load_runner()
    preflight = write_preflight_receipt(tmp_path / 'preflight.json', runner)
    runner.COMMITTED_PREFLIGHT_RECEIPT = preflight
    receipt = write_receipt(tmp_path / 'authorization.json', runner, preflight)
    events = []

    result = runner.run_authorized_fold(
        authorization=receipt,
        fold_id='c4_a',
        requested_scenes=1,
        lazy_runtime_loader=lambda: (
            events.append('runtime') or
            (lambda fold_id, scenes: (fold_id, scenes))))

    assert events == ['runtime']
    assert result == ('c4_a', 1)
