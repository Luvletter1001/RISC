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
    installed = tmp_path / 'installed'
    risc.mkdir()
    clean.mkdir()
    installed.mkdir()
    head = risc / 'head.py'
    dataset = clean / 'dataset.py'
    mmengine = installed / 'mmengine.py'
    head.write_text('head = 1\n')
    dataset.write_text('dataset = 1\n')
    mmengine.write_text('version = 1\n')

    rows = preflight.validate_module_origins(
        {
            'head': (head, 'risc'),
            'dataset': (dataset, 'clean'),
            'mmengine': (mmengine, 'installed'),
        },
        risc_root=risc,
        clean_root=clean,
        installed_root=installed)

    assert [row['authority'] for row in rows] == [
        'clean', 'risc', 'installed']
    assert all(len(row['sha256']) == 64 for row in rows)
    outside = tmp_path / 'outside.py'
    outside.write_text('outside = 1\n')
    with pytest.raises(preflight.PreflightError, match='outside'):
        preflight.validate_module_origins(
            {'bad': (outside, 'risc')}, risc_root=risc, clean_root=clean,
            installed_root=installed)


def test_runtime_origin_contract_covers_frameworks_and_all_custom_imports():
    preflight = load_preflight()
    expected_custom = {
        'M_AD.engine.runner.meta_remove_runer',
        'M_AD.datasets.transforms.formatting',
        'M_AD.datasets.transforms.loading',
        'M_AD.datasets.transforms.transforms',
        'M_AD.datasets.dota_online_v1',
        'M_AD.datasets.samplers.one_task_sampler',
        'M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner',
        'M_AD.models.detectors.Flex_Rtmdet_v3_1_formal',
        'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1',
        'M_AD.models.roi_heads.CLIP_VP_head_v1',
        'M_AD.models.necks.promopt_cspnext_pafpn',
        'M_AD.evaluation.metrics.detail_dota_metric',
    }

    contract = preflight.RUNTIME_MODULE_AUTHORITIES

    expected = expected_custom | {
        'M_AD.models.utils.risc_final_readout',
        ('experiments.rotation_semantic_attractor.src.model_adapters.'
         'openrsd_hook_registry'),
        'mmengine',
        'mmcv',
        'mmdet',
        'mmrotate',
    }
    assert set(contract) == expected
    assert contract['mmengine'] == contract['mmcv'] == 'installed'
    assert contract['mmdet'] == contract['mmrotate'] == 'risc'
    assert expected_custom.issubset(contract)


def test_manifest_asset_validation_rejects_path_hash_or_size_drift(tmp_path):
    preflight = load_preflight()
    asset = tmp_path / 'asset.bin'
    asset.write_bytes(b'authority\n')
    record = {
        'path': str(asset),
        'byte_count': len(asset.read_bytes()),
        'sha256': preflight._sha256_file(asset),
    }

    assert preflight.validate_manifest_asset(
        asset, record, description='asset')['sha256'] == record['sha256']
    with pytest.raises(preflight.PreflightError, match='path'):
        preflight.validate_manifest_asset(
            tmp_path / 'other.bin', record, description='asset')
    asset.write_bytes(b'drift\n')
    with pytest.raises(preflight.PreflightError, match='byte count|hash'):
        preflight.validate_manifest_asset(asset, record, description='asset')


def test_transitive_base_configs_are_exact_preflight_authority():
    preflight = load_preflight()

    report = preflight.audit_transitive_config_assets()

    assert set(report) == {
        'base_config_rtmdet',
        'base_config_settings',
    }
    assert report['base_config_rtmdet']['sha256'] == (
        '6c8ab580172cbbe82dfdda1dfb3755cce4738f58ff2bf91b9cf2e8c0926a74a7')
    assert report['base_config_settings']['sha256'] == (
        '44aeb53ec7e322d0bf87390c392a3af3f2fdf3fc0b9efa94490c8dde8502770a')


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


def test_adapter_initial_debug_identity_requires_enabled_zero_state():
    preflight = load_preflight()

    class Adapter:
        enabled = True
        raw_alpha = 0.0

        @staticmethod
        def debug_state():
            return {
                'enabled': True,
                'alpha': 0.0,
                'max_delta_norm_ratio': 0.0,
            }

    class Model:
        class bbox_head:
            risc_final_readout = Adapter()

    assert preflight.audit_adapter_initial_state(Model()) == {
        'enabled': True,
        'alpha': 0.0,
        'max_delta_norm_ratio': 0.0,
    }
    Model.bbox_head.risc_final_readout.enabled = False
    with pytest.raises(preflight.PreflightError, match='enabled'):
        preflight.audit_adapter_initial_state(Model())


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
        source_hashes={
            'input_manifest': 'a' * 64,
            'preflight_source': 'b' * 64,
            'protocol_source': 'c' * 64,
            'runner_source': 'd' * 64,
            'scene_plan': 'e' * 64,
            'support_ledger': 'f' * 64,
        })

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

    missing_runner = {
        key: value for key, value in {
            'input_manifest': 'a' * 64,
            'preflight_source': 'b' * 64,
            'protocol_source': 'c' * 64,
            'runner_source': 'd' * 64,
            'scene_plan': 'e' * 64,
            'support_ledger': 'f' * 64,
        }.items() if key != 'runner_source'
    }
    with pytest.raises(preflight.PreflightError, match='source hash'):
        preflight.build_preflight_artifacts(
            resolved_config='model = dict()\n',
            origins=[],
            model_ledger=[],
            model_report={},
            source_hashes=missing_runner)
