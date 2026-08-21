import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / 'tools' / 'risc_n0o' / 'preflight_openrsd_n0o.py')


def load_preflight():
    module_name = 'preflight_openrsd_n0o'
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_module_import_is_stdlib_only_and_parser_has_no_gpu_option():
    before_torch = 'torch' in sys.modules
    preflight = load_preflight()
    parser = preflight.build_parser()
    actions = {action.dest for action in parser._actions}

    assert ('torch' in sys.modules) is before_torch
    assert actions == {'help', 'output_dir'}


def test_hybrid_paths_are_risc_first_and_require_both_roots(tmp_path):
    preflight = load_preflight()
    risc = tmp_path / 'risc'
    clean = tmp_path / 'clean'
    risc.mkdir()
    clean.mkdir()

    paths = preflight.plan_hybrid_paths(risc, clean, existing=['/installed'])

    assert paths[:3] == [str(risc.resolve()), str(clean.resolve()), '/installed']
    with pytest.raises(preflight.PreflightError, match='clean root'):
        preflight.plan_hybrid_paths(risc, tmp_path / 'missing', existing=[])


def test_origin_audit_requires_declared_root_and_hash(tmp_path):
    preflight = load_preflight()
    risc = tmp_path / 'risc'
    clean = tmp_path / 'clean'
    risc.mkdir()
    clean.mkdir()
    head = risc / 'head.py'
    dataset = clean / 'dataset.py'
    head.write_text('head = 1\n')
    dataset.write_text('dataset = 1\n')

    rows = preflight.validate_module_origins(
        {
            'head': (head, 'risc'),
            'dataset': (dataset, 'clean'),
        },
        risc_root=risc,
        clean_root=clean)

    assert [row['authority'] for row in rows] == ['clean', 'risc']
    assert all(len(row['sha256']) == 64 for row in rows)
    outside = tmp_path / 'outside.py'
    outside.write_text('outside = 1\n')
    with pytest.raises(preflight.PreflightError, match='outside'):
        preflight.validate_module_origins(
            {'bad': (outside, 'risc')}, risc_root=risc, clean_root=clean)


def test_checkpoint_load_audit_has_exact_three_missing_and_no_unexpected():
    preflight = load_preflight()
    expected = set(preflight.EXPECTED_RISC_MISSING_KEYS)

    assert len(expected) == 17
    assert {
        'bbox_head.risc_final_readout.raw_alpha',
        'bbox_head.risc_final_readout.down.weight',
        'bbox_head.risc_final_readout.up.weight',
    }.issubset(expected)

    report = preflight.audit_checkpoint_load_result(
        missing_keys=sorted(expected),
        unexpected_keys=[],
        common_tensor_count=1129,
        common_tensors_exact=True)

    assert report['missing_keys'] == sorted(expected)
    assert report['unexpected_keys'] == []
    assert report['common_tensors_exact'] is True
    with pytest.raises(preflight.PreflightError, match='missing keys'):
        preflight.audit_checkpoint_load_result(
            missing_keys=[], unexpected_keys=[], common_tensor_count=1129,
            common_tensors_exact=True)
    with pytest.raises(preflight.PreflightError, match='unexpected keys'):
        preflight.audit_checkpoint_load_result(
            missing_keys=sorted(expected), unexpected_keys=['bad'],
            common_tensor_count=1129, common_tensors_exact=True)


def test_optional_missing_module_allowlist_requires_disabled_modules():
    preflight = load_preflight()

    class Module:
        def __init__(self, enable):
            self.enable = enable

    class Head:
        focus_text_anchor_calibration = Module(False)
        focus_fourier_head_gate = Module(False)
        focus_text_logit_mixer = Module(False)
        counter_support_ratio = Module(False)

    class Model:
        bbox_head = Head()

    report = preflight.audit_disabled_optional_modules(Model())
    assert report == {
        'counter_support_ratio': False,
        'focus_fourier_head_gate': False,
        'focus_text_anchor_calibration': False,
        'focus_text_logit_mixer': False,
    }
    Model.bbox_head.focus_text_logit_mixer.enable = True
    with pytest.raises(preflight.PreflightError, match='must be disabled'):
        preflight.audit_disabled_optional_modules(Model())


def test_model_build_audit_never_calls_forward_or_cuda():
    preflight = load_preflight()
    events = []

    class FakeModel:
        def forward(self, *args, **kwargs):
            events.append('forward')
            raise AssertionError('forward called')

        predict = forward
        test_step = forward

    report = preflight.run_model_build_audit(
        build_model=lambda: (events.append('build') or FakeModel()),
        load_model=lambda model: {
            'missing_keys': list(preflight.EXPECTED_RISC_MISSING_KEYS),
            'unexpected_keys': [],
            'common_tensor_count': 1129,
            'common_tensors_exact': True,
            'parameter_count': 10,
        },
        cuda_initialized=lambda: False)

    assert events == ['build']
    assert report['parameter_count'] == 10
    with pytest.raises(preflight.PreflightError, match='CUDA'):
        preflight.run_model_build_audit(
            build_model=FakeModel,
            load_model=lambda model: {},
            cuda_initialized=lambda: True)


def test_runtime_scope_is_initialized_from_resolved_config():
    preflight = load_preflight()
    calls = []

    preflight.initialize_runtime_scope(
        'mmrotate', initializer=lambda value: calls.append(value))

    assert calls == ['mmrotate']
    with pytest.raises(preflight.PreflightError, match='default scope'):
        preflight.initialize_runtime_scope(None, initializer=calls.append)


def test_preflight_publication_is_canonical_and_no_replace(tmp_path):
    preflight = load_preflight()
    output = tmp_path / 'preflight'
    artifacts = preflight.build_preflight_artifacts(
        resolved_config='model = dict()\n',
        origins=[{'module': 'head', 'authority': 'risc'}],
        model_ledger=[{'scene_id': 'P0001'}],
        model_report={'parameter_count': 10},
        source_hashes={'protocol': 'a' * 64})

    preflight.publish_preflight(output, artifacts)

    assert set(path.name for path in output.iterdir()) == {
        'PREFLIGHT_READY_GPU_NOT_AUTHORIZED.json',
        'model_ledger.jsonl',
        'module_origins.json',
        'preflight_report.json',
        'resolved_config.py',
    }
    receipt = json.loads(
        (output / 'PREFLIGHT_READY_GPU_NOT_AUTHORIZED.json').read_text())
    assert receipt['status'] == 'PREFLIGHT_READY_GPU_NOT_AUTHORIZED'
    with pytest.raises(preflight.PreflightError, match='already exists'):
        preflight.publish_preflight(output, artifacts)
