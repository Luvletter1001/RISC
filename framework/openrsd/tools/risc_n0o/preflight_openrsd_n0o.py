#!/usr/bin/env python3
"""CPU-only preflight for the sealed OpenRSD N0-O runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


EXPECTED_RISC_MISSING_KEYS = (
    'bbox_head.counter_support_ratio.bias',
    'bbox_head.counter_support_ratio.log_scale',
    'bbox_head.counter_support_ratio.strength',
    'bbox_head.counter_support_ratio.support_down.weight',
    'bbox_head.counter_support_ratio.support_up.weight',
    'bbox_head.focus_fourier_head_gate.class_mask',
    'bbox_head.focus_fourier_head_gate.raw_alpha',
    'bbox_head.focus_fourier_head_gate.raw_beta',
    'bbox_head.focus_text_anchor_calibration.class_mask',
    'bbox_head.focus_text_anchor_calibration.raw_alpha',
    'bbox_head.focus_text_anchor_calibration.raw_beta',
    'bbox_head.focus_text_logit_mixer.class_mask',
    'bbox_head.focus_text_logit_mixer.raw_beta',
    'bbox_head.focus_text_logit_mixer.raw_gamma',
    'bbox_head.risc_final_readout.down.weight',
    'bbox_head.risc_final_readout.raw_alpha',
    'bbox_head.risc_final_readout.up.weight',
)
OPTIONAL_MODULE_NAMES = (
    'counter_support_ratio',
    'focus_fourier_head_gate',
    'focus_text_anchor_calibration',
    'focus_text_logit_mixer',
)
RISC_ROOT = Path('/data1/zcy/RISC/framework/openrsd')
CLEAN_ROOT = Path('/data1/zcy/GSOVD/.lab/tmp/openrsd_head_20260706')
INSTALLED_ROOT = Path(
    '/data/zcy/anaconda3/envs/openrsd/lib/python3.10/site-packages')
V3_ROOT = Path('/data1/zcy/RISC/docs/provenance/risc_openrsd_n0o_v3')
HISTORICAL_CONFIG = (
    CLEAN_ROOT / 'M_configs/Step2_A10_Large_Pretrain_Stage3/'
    'A10_flex_rtm_v3_1_formal.py')
TRANSITIVE_CONFIG_ASSETS = {
    'base_config_rtmdet': {
        'path': str(HISTORICAL_CONFIG.with_name('base_rtmdet_l.py')),
        'byte_count': 2352,
        'sha256': (
            '6c8ab580172cbbe82dfdda1dfb3755cce4738f58ff2bf91b9cf2e8c0926a74a7'),
    },
    'base_config_settings': {
        'path': str(HISTORICAL_CONFIG.with_name(
            'base_settings_dior_rtmdet.py')),
        'byte_count': 5462,
        'sha256': (
            '44aeb53ec7e322d0bf87390c392a3af3f2fdf3fc0b9efa94490c8dde8502770a'),
    },
}
RUNTIME_MODULE_AUTHORITIES = {
    'M_AD.datasets.dota_online_v1': 'clean',
    'M_AD.datasets.samplers.one_task_sampler': 'clean',
    'M_AD.datasets.transforms.formatting': 'clean',
    'M_AD.datasets.transforms.loading': 'clean',
    'M_AD.datasets.transforms.transforms': 'clean',
    'M_AD.engine.runner.meta_remove_runer': 'risc',
    'M_AD.evaluation.metrics.detail_dota_metric': 'risc',
    'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1': 'risc',
    'M_AD.models.detectors.Flex_Rtmdet_v3_1_formal': 'risc',
    'M_AD.models.necks.promopt_cspnext_pafpn': 'risc',
    'M_AD.models.roi_heads.CLIP_VP_head_v1': 'risc',
    'M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner': (
        'risc'),
    'M_AD.models.utils.risc_final_readout': 'risc',
    'experiments.rotation_semantic_attractor.src.model_adapters.'
    'openrsd_hook_registry': 'risc',
    'mmcv': 'installed',
    'mmdet': 'risc',
    'mmengine': 'installed',
    'mmrotate': 'risc',
}
REQUIRED_SOURCE_HASH_KEYS = frozenset({
    'input_manifest',
    'preflight_source',
    'protocol_source',
    'runner_source',
    'scene_plan',
    'support_ledger',
})


class PreflightError(RuntimeError):
    """Raised when CPU preflight cannot authorize a future GPU smoke."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser


def _resolved_directory(path: Path | str, description: str) -> Path:
    value = Path(path).resolve()
    if not value.is_dir():
        raise PreflightError('{} is missing'.format(description))
    return value


def plan_hybrid_paths(
        risc_root: Path | str,
        clean_root: Path | str,
        *,
        existing) -> list[str]:
    risc = _resolved_directory(risc_root, 'RISC root')
    clean = _resolved_directory(clean_root, 'clean root')
    output = [str(risc), str(clean)]
    for item in existing:
        text = str(item)
        if text not in output:
            output.append(text)
    return output


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_module_origins(
        modules: Mapping[str, tuple[Path | str, str]],
        *,
        risc_root: Path | str,
        clean_root: Path | str,
        installed_root: Path | str) -> list[dict[str, Any]]:
    risc = _resolved_directory(risc_root, 'RISC root')
    clean = _resolved_directory(clean_root, 'clean root')
    installed = _resolved_directory(installed_root, 'installed root')
    rows = []
    for module_name, (path_value, authority) in sorted(modules.items()):
        path = Path(path_value).resolve()
        if not path.is_file():
            raise PreflightError('{} origin file is missing'.format(
                module_name))
        if authority == 'risc':
            allowed = risc
        elif authority == 'clean':
            allowed = clean
        elif authority == 'installed':
            allowed = installed
        else:
            raise PreflightError('module authority is invalid')
        if not _within(path, allowed):
            raise PreflightError(
                '{} origin is outside declared {} root'.format(
                    module_name, authority))
        rows.append({
            'module': module_name,
            'authority': authority,
            'path': str(path),
            'byte_count': path.stat().st_size,
            'sha256': _sha256_file(path),
        })
    return rows


def validate_manifest_asset(
        path: Path | str,
        record: Mapping[str, Any],
        *,
        description: str) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        recorded_path = Path(record['path']).resolve()
        expected_bytes = record['byte_count']
        expected_sha256 = record['sha256']
    except (KeyError, TypeError) as error:
        raise PreflightError(
            '{} manifest record is invalid'.format(description)) from error
    if path != recorded_path:
        raise PreflightError('{} path does not match manifest'.format(
            description))
    if not path.is_file():
        raise PreflightError('{} file is missing'.format(description))
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise PreflightError('{} byte count mismatch'.format(description))
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PreflightError('{} hash mismatch'.format(description))
    return {
        'path': str(path),
        'byte_count': actual_bytes,
        'sha256': actual_sha256,
    }


def audit_transitive_config_assets() -> dict[str, dict[str, Any]]:
    return {
        name: validate_manifest_asset(
            record['path'], record, description=name)
        for name, record in sorted(TRANSITIVE_CONFIG_ASSETS.items())
    }


def audit_manifest_runtime_assets(bundle) -> dict[str, dict[str, Any]]:
    manifest = bundle.manifest
    records = {
        'historical_config': manifest['historical_assets'][
            'p77e_source_config'],
        'parent_checkpoint': manifest['parent'],
        'support_source': manifest['support']['source'],
        'neg_support_data': manifest['supporting_assets']['neg_support_data'],
        'normalized_class_dict': manifest['supporting_assets'][
            'normalized_class_dict'],
        'pca_meta': manifest['supporting_assets']['pca_meta'],
        's0_adapter': manifest['s0']['assets']['adapter'],
        's0_capture': manifest['s0']['assets']['capture'],
        's0_head': manifest['s0']['assets']['head'],
        's0_config': manifest['s0']['assets']['s0_config'],
    }
    paths = {
        'historical_config': HISTORICAL_CONFIG,
        **{name: record['path'] for name, record in records.items()
           if name != 'historical_config'},
    }
    manifest_report = {
        name: validate_manifest_asset(
            paths[name], record, description=name)
        for name, record in sorted(records.items())
    }
    overlap = set(manifest_report).intersection(TRANSITIVE_CONFIG_ASSETS)
    if overlap:
        raise PreflightError('runtime asset identities overlap')
    return {**manifest_report, **audit_transitive_config_assets()}


def audit_checkpoint_load_result(
        *,
        missing_keys,
        unexpected_keys,
        common_tensor_count: int,
        common_tensors_exact: bool) -> dict[str, Any]:
    missing = sorted(missing_keys)
    unexpected = sorted(unexpected_keys)
    if missing != sorted(EXPECTED_RISC_MISSING_KEYS):
        raise PreflightError('checkpoint missing keys are not exact')
    if unexpected:
        raise PreflightError('checkpoint unexpected keys are not empty')
    if type(common_tensor_count) is not int or common_tensor_count <= 0:
        raise PreflightError('common tensor count must be positive')
    if common_tensors_exact is not True:
        raise PreflightError('common checkpoint tensors are not exact')
    return {
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'common_tensor_count': common_tensor_count,
        'common_tensors_exact': True,
    }


def audit_disabled_optional_modules(model) -> dict[str, bool]:
    head = getattr(model, 'bbox_head', None)
    if head is None:
        raise PreflightError('model bbox_head is missing')
    report = {}
    for name in OPTIONAL_MODULE_NAMES:
        module = getattr(head, name, None)
        if module is None or not hasattr(module, 'enable'):
            raise PreflightError('{} module identity is missing'.format(name))
        enabled = bool(module.enable)
        if enabled:
            raise PreflightError('{} must be disabled'.format(name))
        report[name] = enabled
    return report


def audit_adapter_initial_state(model) -> dict[str, Any]:
    adapter = getattr(
        getattr(model, 'bbox_head', None), 'risc_final_readout', None)
    if adapter is None:
        raise PreflightError('RISC adapter identity is missing')
    if getattr(adapter, 'enabled', None) is not True:
        raise PreflightError('RISC adapter must be enabled')
    debug_state = adapter.debug_state()
    expected = {
        'enabled': True,
        'alpha': 0.0,
        'max_delta_norm_ratio': 0.0,
    }
    if debug_state != expected:
        raise PreflightError('RISC adapter initial debug state is not identity')
    raw_alpha = adapter.raw_alpha
    if hasattr(raw_alpha, 'detach'):
        raw_alpha = raw_alpha.detach().cpu().item()
    if float(raw_alpha) != 0.0:
        raise PreflightError('RISC adapter raw alpha is not zero')
    return expected


def run_model_build_audit(
        *,
        build_model,
        load_model,
        cuda_initialized) -> dict[str, Any]:
    if cuda_initialized():
        raise PreflightError('CUDA is initialized before CPU preflight')
    model = build_model()
    if cuda_initialized():
        raise PreflightError('CUDA initialized during model construction')
    raw_report = dict(load_model(model))
    if cuda_initialized():
        raise PreflightError('CUDA initialized during checkpoint load')
    audit = audit_checkpoint_load_result(
        missing_keys=raw_report.pop('missing_keys'),
        unexpected_keys=raw_report.pop('unexpected_keys'),
        common_tensor_count=raw_report.pop('common_tensor_count'),
        common_tensors_exact=raw_report.pop('common_tensors_exact'))
    audit.update(raw_report)
    del model
    return audit


def initialize_runtime_scope(scope_name, *, initializer) -> None:
    if not isinstance(scope_name, str) or not scope_name:
        raise PreflightError('resolved config default scope is missing')
    initializer(scope_name)


def _helpers():
    from tools.risc_n0o.prepare_openrsd_n0o_input_seal import (
        canonical_json_bytes,
        canonical_jsonl_bytes,
        publish_artifacts,
        SealError,
        sha256_bytes,
    )
    return (
        canonical_json_bytes, canonical_jsonl_bytes, publish_artifacts,
        SealError, sha256_bytes)


def build_preflight_artifacts(
        *,
        resolved_config: str,
        origins,
        model_ledger,
        model_report,
        source_hashes) -> dict[str, bytes]:
    (canonical_json_bytes, canonical_jsonl_bytes, _, _,
     sha256_bytes) = _helpers()
    if (set(source_hashes) != REQUIRED_SOURCE_HASH_KEYS
            or any(not isinstance(value, str) or len(value) != 64
                   for value in source_hashes.values())):
        raise PreflightError('preflight source hash contract is not exact')
    config_bytes = resolved_config.encode('utf-8')
    if not config_bytes.endswith(b'\n'):
        config_bytes += b'\n'
    origins_bytes = canonical_json_bytes({
        'schema': 'risc-openrsd-n0o-module-origins-v2',
        'modules': list(origins),
    })
    ledger_bytes = canonical_jsonl_bytes(model_ledger)
    report = {
        'schema': 'risc-openrsd-n0o-preflight-report-v2',
        'execution_boundary': {
            'cuda_initialized': False,
            'gpu_used': False,
            'model_forward_executed': False,
            'prediction_created': False,
            'training_executed': False,
        },
        'model': dict(model_report),
        'sources': dict(source_hashes),
    }
    report_bytes = canonical_json_bytes(report)
    primary = {
        'resolved_config.py': config_bytes,
        'module_origins.json': origins_bytes,
        'model_ledger.jsonl': ledger_bytes,
        'preflight_report.json': report_bytes,
    }
    artifact_rows = {
        name: {
            'byte_count': len(content),
            'sha256': sha256_bytes(content),
        }
        for name, content in sorted(primary.items())
    }
    receipt_bytes = canonical_json_bytes({
        'schema': 'risc-openrsd-n0o-preflight-receipt-v2',
        'status': 'PREFLIGHT_READY_GPU_NOT_AUTHORIZED',
        'artifacts': artifact_rows,
        'gpu_smoke_authorized': False,
    })
    return {
        **primary,
        'PREFLIGHT_READY_GPU_NOT_AUTHORIZED.json': receipt_bytes,
    }


def publish_preflight(
        output_dir: Path | str,
        artifacts: Mapping[str, bytes]) -> None:
    _, _, publish_artifacts, seal_error, _ = _helpers()
    try:
        publish_artifacts(output_dir, artifacts)
    except seal_error as error:
        raise PreflightError(str(error)) from error


def _configure_hybrid_imports() -> None:
    sys.path[:] = plan_hybrid_paths(
        RISC_ROOT, CLEAN_ROOT, existing=sys.path)
    import importlib
    importlib.invalidate_caches()
    for package_name, root in (
            ('M_AD', RISC_ROOT / 'M_AD'),
            ('M_AD', CLEAN_ROOT / 'M_AD')):
        package = sys.modules.get(package_name)
        if package is not None and hasattr(package, '__path__'):
            path_text = str(root)
            if path_text not in package.__path__:
                package.__path__.append(path_text)


def _import_installed_frameworks() -> None:
    import importlib
    original_paths = list(sys.path)
    filtered_paths = []
    for entry in original_paths:
        candidate = Path(entry or os.getcwd()).resolve()
        if (_within(candidate, RISC_ROOT.resolve())
                or _within(candidate, CLEAN_ROOT.resolve())):
            continue
        filtered_paths.append(entry)
    try:
        sys.path[:] = filtered_paths
        importlib.invalidate_caches()
        for module_name in ('mmengine', 'mmcv'):
            module = importlib.import_module(module_name)
            path = Path(module.__file__).resolve()
            if not _within(path, INSTALLED_ROOT.resolve()):
                raise PreflightError(
                    '{} must resolve from installed environment'.format(
                        module_name))
    finally:
        sys.path[:] = original_paths
        importlib.invalidate_caches()


def _runtime_config(bundle):
    from mmengine.config import Config
    cfg = Config.fromfile(str(HISTORICAL_CONFIG))
    cfg.launcher = 'none'
    cfg.work_dir = '/data1/zcy/RISC_PRELIGHT_NO_RUN'
    cfg.load_from = bundle.manifest['parent']['path']
    cfg.model.val_support_classes = bundle.manifest['support']['class_order']
    cfg.model.val_dataset_flag = 'Data1_DOTA2'
    cfg.model.val_using_aux = False
    cfg.model.support_type = 'text'
    cfg.model.num_val_prompts = 7
    cfg.model.support_feat_dict = {
        'Data1_DOTA2': bundle.manifest['support']['source']['path']}
    cfg.model.neg_support_data = bundle.manifest[
        'supporting_assets']['neg_support_data']['path']
    cfg.model.normalized_class_dict = bundle.manifest[
        'supporting_assets']['normalized_class_dict']['path']
    cfg.model.pca_meta_pth = bundle.manifest[
        'supporting_assets']['pca_meta']['path']
    cfg.model.bbox_head.risc_final_readout = dict(
        enabled=True,
        rank=8,
        init_alpha=0.0,
        max_alpha=0.1,
        max_delta_norm_ratio=0.05,
        init_seed=20260822)
    return cfg


def _actual_module_origins():
    import importlib
    values = {}
    for module_name, authority in RUNTIME_MODULE_AUTHORITIES.items():
        module = importlib.import_module(module_name)
        values[module_name] = (Path(module.__file__), authority)
    return validate_module_origins(
        values,
        risc_root=RISC_ROOT,
        clean_root=CLEAN_ROOT,
        installed_root=INSTALLED_ROOT)


def _load_actual_checkpoint(model, checkpoint_path):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping):
        raise PreflightError('raw checkpoint state_dict is missing')
    incompatible = model.load_state_dict(state, strict=False)
    model_state = model.state_dict()
    common = sorted(set(state).intersection(model_state))
    common_exact = True
    for key in common:
        source = state[key]
        target = model_state[key]
        if (getattr(source, 'shape', None) != getattr(target, 'shape', None)
                or not torch.equal(source.detach().cpu(), target.detach().cpu())):
            common_exact = False
            break
    return {
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'common_tensor_count': len(common),
        'common_tensors_exact': common_exact,
        'parameter_count': sum(
            parameter.numel() for parameter in model.parameters()),
        'trainable_parameter_count': sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
        'adapter_initial_state': audit_adapter_initial_state(model),
        'disabled_optional_modules': audit_disabled_optional_modules(model),
    }


def run_real_preflight(output_dir: Path | str) -> dict[str, Any]:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('PYTHONNOUSERSITE', '1')
    _import_installed_frameworks()
    _configure_hybrid_imports()

    import gc
    import torch
    from mmengine.registry import init_default_scope
    from mmdet.utils import register_all_modules as register_mmdet
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules as register_mmrotate
    from tools.risc_n0o.openrsd_n0o_protocol import (
        SupportCache,
        V3_MANIFEST_SHA256,
        audit_support_bundle,
        build_model_ledger,
        load_protocol_bundle,
    )

    if torch.cuda.is_initialized():
        raise PreflightError('CUDA was initialized before CPU preflight')
    bundle = load_protocol_bundle(V3_ROOT)
    manifest_assets = audit_manifest_runtime_assets(bundle)
    support_cache = SupportCache.from_bundle(bundle)
    support_report = audit_support_bundle(support_cache)
    del support_cache
    gc.collect()
    ledger = build_model_ledger(bundle)
    cfg = _runtime_config(bundle)
    register_mmdet(init_default_scope=False)
    register_mmrotate(init_default_scope=False)
    initialize_runtime_scope(
        cfg.get('default_scope'), initializer=init_default_scope)
    origins = _actual_module_origins()

    def build_model():
        return MODELS.build(cfg.model)

    def load_model(model):
        return _load_actual_checkpoint(
            model, bundle.manifest['parent']['path'])

    report = run_model_build_audit(
        build_model=build_model,
        load_model=load_model,
        cuda_initialized=torch.cuda.is_initialized)
    report['manifest_assets'] = manifest_assets
    report['support_reconstruction'] = support_report
    gc.collect()
    if torch.cuda.is_initialized():
        raise PreflightError('CUDA initialized during CPU preflight')
    source_hashes = {
        'input_manifest': V3_MANIFEST_SHA256,
        'scene_plan': bundle.manifest['scene_plan']['sha256'],
        'support_ledger': bundle.manifest['support']['ledger_sha256'],
        'preflight_source': _sha256_file(Path(__file__).resolve()),
        'protocol_source': _sha256_file(
            Path(__file__).with_name('openrsd_n0o_protocol.py')),
        'runner_source': _sha256_file(
            Path(__file__).with_name('run_openrsd_n0o_fold.py')),
    }
    artifacts = build_preflight_artifacts(
        resolved_config=cfg.pretty_text,
        origins=origins,
        model_ledger=ledger,
        model_report=report,
        source_hashes=source_hashes)
    publish_preflight(output_dir, artifacts)
    _, _, _, _, sha256_bytes = _helpers()
    return {
        'status': 'PREFLIGHT_READY_GPU_NOT_AUTHORIZED',
        'model_ledger_rows': len(ledger),
        'receipt_sha256': sha256_bytes(
            artifacts['PREFLIGHT_READY_GPU_NOT_AUTHORIZED.json']),
        'common_tensor_count': report['common_tensor_count'],
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_real_preflight(args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
