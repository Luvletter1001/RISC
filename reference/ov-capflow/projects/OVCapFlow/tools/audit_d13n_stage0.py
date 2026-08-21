#!/usr/bin/env python3
"""Run the D13-N Stage-0 checks through existing project audit paths."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
from mmrotate.registry import DATASETS, MODELS
from mmrotate.utils import register_all_modules

import projects.OVCapFlow.ov_capflow  # noqa: F401
from projects.OVCapFlow.ov_capflow.d13n_mode_hook import (
    D13NParentEvalModeHook, hash_named_tensor_state)
from projects.OVCapFlow.ov_capflow.existence_residual import ExistenceResidual
from projects.OVCapFlow.ov_capflow.freeze_except_hook import (
    D13N_PARAMETER_NAMES, require_d13n_trainable_parameters)
from projects.OVCapFlow.ov_capflow.no_replace import publish_json_noreplace
from projects.OVCapFlow.tools import d13n_runtime
from projects.OVCapFlow.tools import audit_d12_preflight as d12_preflight
from projects.OVCapFlow.tools.audit_d12_preflight import _run_role_model
from projects.OVCapFlow.tools.audit_dotav2_mouth import (
    audit_mouth, validate_contract)
from projects.OVCapFlow.tools.audit_open_vocabulary import (
    audit_prompt_variants, set_prompt_metainfo)
from projects.OVCapFlow.tools.audit_strict_inference import (
    record_forbidden_runtime_calls, scan_calls)
from projects.OVCapFlow.tools.d13n_test import (
    _absolute_lexical_path, _validate_global_runtime_environment,
    build_torchrun_argv)
from projects.OVCapFlow.tools.validate_dotav2_q600_dump import (
    as_tensor, load_cpu, validate_records)

SCHEMA = 'd13n-stage0-v1'
GATE_NAMES = ('identity_load', 'static_d13n', 'zero_step_equality',
              'protocol_regression', 'real_finite_smoke')
MODEL_EVIDENCE_KEYS = ('parent_parameter_names', 'parent_buffer_names',
                       'parent_parameter_sha256', 'parent_buffer_sha256')
ROLES = ('stage0-parent', 'proxy-control', 'proxy-candidate')
ROLE_PORTS = (29845, 29846, 29847)
ROLE_CONFIG = ('control', 'control', 'candidate')
ADAPTER_NAMES = tuple(D13N_PARAMETER_NAMES)

class Stage0ChildError(RuntimeError):
    pass
class ScientificGateError(RuntimeError):
    pass
def _json_copy(value, name):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be finite JSON') from error
def report_sha256(report):
    payload = dict(report)
    payload.pop('report_sha256', None)
    return d13n_runtime.canonical_json_sha256(payload)
def collect_stage0_fingerprint(repo, parent, control_config, candidate_config):
    """Bind only the commit and three inputs needed by the thin Stage-0."""
    repo = Path(repo).resolve(strict=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(repo),
                            check=True, stdout=subprocess.PIPE,
                            encoding='utf-8').stdout.strip()
    sources = dict(parent=parent, control_config=control_config,
                   candidate_config=candidate_config)
    inputs = {}
    for name, source in sources.items():
        path = Path(source).expanduser()
        path = (path if path.is_absolute() else repo / path).resolve(strict=True)
        inputs[name] = {'path': str(path), 'sha256': d13n_runtime.sha256_file(path)}
    value = {'schema': 'd13n-stage0-inputs-v1', 'git': {'commit': commit},
             'inputs': inputs}
    value['fingerprint_sha256'] = d13n_runtime.canonical_json_sha256(value)
    return value
def _validate_model_evidence(evidence):
    if type(evidence) is not dict or list(evidence) != list(
            MODEL_EVIDENCE_KEYS):
        raise ValueError('model_evidence must contain the exact four keys')
    names = evidence['parent_parameter_names'] + evidence[
        'parent_buffer_names']
    if (any(type(name) is not str or not name for name in names) or
            len(names) != len(set(names))):
        raise ValueError('model_evidence names must be unique strings')
    for key in MODEL_EVIDENCE_KEYS[2:]:
        value = evidence[key]
        if (type(value) is not str or len(value) != 64 or
                any(character not in '0123456789abcdef'
                    for character in value)):
            raise ValueError(f'model_evidence {key} must be lowercase SHA256')
def build_stage0_report(*, pre_run_commit, fingerprint, inputs, artifacts,
                        gates, model_evidence):
    """Aggregate detailed evidence under exactly five scientific checks."""
    if type(gates) is not dict or list(gates) != list(GATE_NAMES):
        raise ValueError('gates must be exactly the five core checks')
    gates = _json_copy(gates, 'gates')
    if any(type(value) is not dict or type(value.get('passed')) is not bool
           for value in gates.values()):
        raise ValueError('each core gate needs an exact passed boolean')
    model_evidence = _json_copy(model_evidence, 'model_evidence')
    _validate_model_evidence(model_evidence)
    failures = [name for name, value in gates.items()
                if value['passed'] is not True]
    fingerprint = _json_copy(fingerprint, 'fingerprint')
    report = {
        'schema': SCHEMA,
        'status': 'PASS' if not failures else 'FAIL',
        'pre_run_commit': str(pre_run_commit),
        'fingerprint': fingerprint,
        'fingerprint_sha256': fingerprint.get('fingerprint_sha256',
            d13n_runtime.canonical_json_sha256(fingerprint)),
        'inputs': _json_copy(inputs, 'inputs'),
        'artifacts': _json_copy(artifacts, 'artifacts'),
        'gates': gates,
        'model_evidence': model_evidence,
        'failures': failures,
    }
    report['report_sha256'] = report_sha256(report)
    return report
def _validate_report(report):
    if type(report) is not dict or report.get('schema') != SCHEMA:
        raise ValueError('invalid D13-N Stage-0 report')
    fingerprint = report.get('fingerprint')
    if type(fingerprint) is not dict:
        raise ValueError('Stage-0 fingerprint is invalid')
    unhashed = dict(fingerprint)
    observed_hash = unhashed.pop('fingerprint_sha256', None)
    if observed_hash != d13n_runtime.canonical_json_sha256(unhashed):
        raise ValueError('Stage-0 fingerprint self-hash is invalid')
    if (report.get('pre_run_commit') != fingerprint.get('git', {}).get('commit')
            or report.get('inputs') != fingerprint.get('inputs')):
        raise ValueError('Stage-0 report differs from fingerprint authority')
    rebuilt = build_stage0_report(
        pre_run_commit=report.get('pre_run_commit'),
        fingerprint=report.get('fingerprint'), inputs=report.get('inputs'),
        artifacts=report.get('artifacts'), gates=report.get('gates'),
        model_evidence=report.get('model_evidence'))
    if report != rebuilt:
        raise ValueError('Stage-0 report content or self-hash is invalid')
    return rebuilt
def publish_stage0_report(path, report):
    """Validate and publish one immutable Stage-0 report."""
    publish_json_noreplace(path, _validate_report(report))

def _tensor_equal(left, right):
    return (torch.is_tensor(left) and torch.is_tensor(right) and
            left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left, right))
def _first_tensor_mismatch(left, right):
    if (not torch.is_tensor(left) or not torch.is_tensor(right) or
            left.dtype != right.dtype or left.shape != right.shape):
        return None, None
    unequal = torch.nonzero(left != right, as_tuple=False)
    if unequal.numel() == 0:
        return None, None
    location = unequal[0].tolist()
    return int(location[0]), (int(location[1]) if len(location) > 1 else None)
def compare_prediction_dumps(parent_path, control_path, candidate_path,
                             expected_records=400, queries_per_image=600):
    """Validate and compare all three existing d13n_test dump artifacts."""
    records = [load_cpu(Path(path)) for path in (
        parent_path, control_path, candidate_path)]
    summaries = [validate_records(
        value, expected_records=expected_records,
        queries_per_image=queries_per_image, num_classes=18)
                 for value in records]
    pair_specs = (
        ('parent_control', 0, 1), ('parent_candidate', 0, 2),
        ('control_candidate', 1, 2),
    )
    pairwise = {}
    first = None
    source_order_equal = True
    for pair, left_index, right_index in pair_specs:
        left_records, right_records = records[left_index], records[right_index]
        checks = {'scores': True, 'labels': True, 'boxes': True}
        for record_index, (left, right) in enumerate(zip(
                left_records, right_records)):
            if left['img_id'] != right['img_id']:
                source_order_equal = False
                if first is None:
                    first = {
                        'pair': pair, 'record_index': record_index,
                        'field': 'source_order', 'row_index': None,
                        'column_index': None}
                continue
            for field, source in (
                    ('scores', 'scores'), ('labels', 'labels'),
                    ('boxes', 'bboxes')):
                left_tensor = as_tensor(left['pred_instances'][source], field)
                right_tensor = as_tensor(
                    right['pred_instances'][source], field)
                equal = _tensor_equal(left_tensor, right_tensor)
                checks[field] = checks[field] and equal
                if not equal and first is None:
                    row, column = _first_tensor_mismatch(
                        left_tensor, right_tensor)
                    first = {
                        'pair': pair, 'record_index': record_index,
                        'field': field, 'row_index': row,
                        'column_index': column}
        pairwise[pair] = checks
    passed = (source_order_equal and all(
        value for checks in pairwise.values() for value in checks.values()))
    return {
        'passed': passed, 'records': expected_records,
        'queries_per_image': queries_per_image,
        'prediction_rows': summaries[0]['prediction_rows'],
        'source_order_equal': source_order_equal,
        'pairwise': pairwise, 'first_mismatch': first,
        'validator_summaries': summaries,
    }
def load_parent_into_d13n(model, checkpoint_state):
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    return {
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
    }
def _assert_child_input_hashes(identities, inputs):
    expected_configs = {
        'stage0-parent': inputs['control_config']['sha256'],
        'proxy-control': inputs['control_config']['sha256'],
        'proxy-candidate': inputs['candidate_config']['sha256']}
    for role in ROLES:
        value = identities.get(role, {})
        if (value.get('checkpoint_sha256') != inputs['parent']['sha256'] or
                value.get('config_sha256') != expected_configs[role]):
            raise ValueError(f'{role} identity differs from fingerprint inputs')

def _assert_inputs_unchanged(inputs, sources=None):
    sources = sources or {}
    for name in ('parent', 'control_config', 'candidate_config'):
        identity = inputs[name]
        source = sources.get(name, identity['path'])
        if d13n_runtime.sha256_file(source) != identity['sha256']:
            raise ValueError(f'{name} changed after Stage-0 fingerprint')

def audit_identity_load(identities, load_evidence, fingerprint_inputs):
    """Conjoin d13n_test identity files with the two real adapter loads."""
    if type(identities) is not dict or set(identities) != set(ROLES):
        raise ValueError('identity evidence must contain the three roles')
    _assert_child_input_hashes(identities, fingerprint_inputs)
    hashes = set()
    summaries = {}
    for role in ROLES:
        value = identities[role]
        hash_fields = (
            'config_sha256', 'resolved_config_sha256', 'checkpoint_sha256',
            'predictions_sha256', 'official_metrics_sha256')
        path_fields = ('config_path', 'predictions_path',
                       'official_metrics_path')
        valid = (
            type(value) is dict and
            value.get('schema') == 'd13n-test-identity-v1' and
            value.get('role') == role and value.get('finite') is True and
            value.get('records') == 400 and
            value.get('queries_per_image') == 600 and
            value.get('world_size') == 5 and
            value.get('local_ranks') == [0, 1, 2, 3, 4] and
            value.get('cuda_visible_devices') == '5,6,7,8,9' and
            value.get('derived_parent') is (role == 'stage0-parent') and
            all(type(value.get(key)) is str and Path(value[key]).is_absolute()
                for key in path_fields) and
            all(type(value.get(key)) is str and len(value[key]) == 64 and
                set(value[key]) <= set('0123456789abcdef') for key in hash_fields))
        if not valid:
            raise ValueError(f'invalid d13n_test identity for {role}')
        hashes.add(value.get('checkpoint_sha256'))
        summary_keys = ('config_path', 'config_sha256', 'resolved_config_sha256',
            'checkpoint_sha256', 'records', 'queries_per_image', 'world_size',
            'local_ranks', 'cuda_visible_devices', 'derived_parent',
            'predictions_path', 'predictions_sha256',
            'official_metrics_path', 'official_metrics_sha256')
        summaries[role] = {key: value[key] for key in summary_keys}
    if len(hashes) != 1:
        raise ValueError('child checkpoint hashes differ')
    expected_load = {
        'missing_keys': list(ADAPTER_NAMES), 'unexpected_keys': []}
    if type(load_evidence) is not dict:
        raise ValueError('adapter load evidence is missing')
    for role in ROLES[1:]:
        if load_evidence.get(role) != expected_load:
            raise ValueError(f'{role} must load with exactly two adapter keys')
    return {
        'passed': True, 'roles': list(ROLES),
        'checkpoint_sha256': hashes.pop(),
        'adapter_loads': _json_copy(load_evidence, 'load evidence'),
        'identities': summaries,
        'ports': dict(zip(ROLES, ROLE_PORTS)),
    }

def _positive_zero(tensor):
    value = tensor.detach().cpu().contiguous()
    return bool(torch.equal(value, torch.zeros_like(value)) and
                not value.reshape(-1).view(torch.uint8).any().item())
def _optimizer_parameters(wrapper):
    optimizer = getattr(wrapper, 'optimizer', None)
    if optimizer is None:
        raise ValueError('actual MMEngine optimizer wrapper is required')
    return [parameter for group in optimizer.param_groups
            for parameter in group['params']]
def audit_static_d13n(model, optim_wrapper):
    """Check only D13-N adapter, freeze, mode, optimizer, and parent hashes."""
    model = model.module if hasattr(model, 'module') else model
    expected = require_d13n_trainable_parameters(model)
    named_parameters = dict(model.named_parameters())
    actual = _optimizer_parameters(optim_wrapper)
    if (len(actual) != 2 or len({id(value) for value in actual}) != 2 or
            {id(value) for value in actual} != {id(value) for value in expected}):
        raise ValueError('optimizer must contain the exact two adapter tensors')
    optimizer_names = [name for name in ADAPTER_NAMES
                       if id(named_parameters[name]) in {id(x) for x in actual}]
    adapter = model.bbox_head.existence_residual
    if (not _positive_zero(adapter.weight) or
            not _positive_zero(adapter.bias)):
        raise ValueError('D13-N adapter must contain positive zero bytes')
    scalars = sum(parameter.numel() for parameter in expected)
    if scalars != 257:
        raise ValueError('D13-N adapter must contain exactly 257 scalars')
    rng_before = torch.get_rng_state().clone()
    ExistenceResidual(adapter.in_features)
    rng_unchanged = torch.equal(rng_before, torch.get_rng_state())
    if not rng_unchanged:
        raise ValueError('D13-N adapter construction consumed RNG')
    hook = D13NParentEvalModeHook('candidate')
    hook.before_train_iter(type('RunnerView', (), {'model': model})(), 0)
    mode_ok = (not model.bbox_head.training and adapter.training)
    if not mode_ok:
        raise ValueError('D13-N parent/adapter mode authority failed')
    state = model.state_dict()
    parent_parameters = {
        name: value for name, value in named_parameters.items()
        if name not in ADAPTER_NAMES}
    parameter_names = set(named_parameters)
    parent_buffers = {
        name: value for name, value in state.items()
        if name not in parameter_names and name not in ADAPTER_NAMES}
    authority = {
        'parent_parameter_names': list(parent_parameters),
        'parent_buffer_names': list(parent_buffers),
        'parent_parameter_sha256': hash_named_tensor_state(parent_parameters),
        'parent_buffer_sha256': hash_named_tensor_state(parent_buffers),
    }
    return ({
        'passed': True, 'trainable_parameter_names': list(ADAPTER_NAMES),
        'optimizer_parameter_names': optimizer_names,
        'adapter_scalars': scalars, 'positive_zero': True,
        'constructor_rng_unchanged': rng_unchanged,
        'parent_eval_adapter_train': mode_ok,
    }, authority)
def audit_protocol_regression(*, dump_audit, strict, open_vocabulary, mouth):
    strict_calls = strict.get('forbidden_calls')
    runtime_calls = open_vocabulary.get('runtime_forbidden_calls')
    checks = {
        'strict_q600': dump_audit.get('passed') is True,
        'forbidden_calls': (
            strict.get('pass') is True and
            type(strict_calls) is list and not strict_calls and
            type(runtime_calls) is list and not runtime_calls),
        'open_vocabulary': open_vocabulary.get('pass') is True,
        'rotated_mouth': mouth.get('valid') is True,
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'evidence': {'strict': strict,
                         'open_vocabulary': open_vocabulary, 'mouth': mouth}}
def _run_d13n_role(**kwargs):
    original_builder = d12_preflight._build_loaded_model
    hook = D13NParentEvalModeHook(kwargs['role'])
    def build(*args, **builder_kwargs):
        model = original_builder(*args, **builder_kwargs)
        original_train = model.train
        def train(mode=True):
            result = original_train(mode)
            if mode:
                hook.before_train_iter(
                    type('RunnerView', (), {'model': model})(), 0)
            return result
        model.train = train
        return model
    d12_preflight._build_loaded_model = build
    try:
        return _run_role_model(**kwargs)
    finally:
        d12_preflight._build_loaded_model = original_builder
def run_real_finite_smoke(*, control_cfg, candidate_cfg, control_state,
                          candidate_state, raw_batches, device):
    """Reuse D12's real batch forward/backward path without optimizer.step."""
    expected = ('empty', 'ordinary', 'dense_1223')
    if type(raw_batches) is not dict or list(raw_batches) != list(expected):
        raise ValueError('raw_batches must have the exact ordered three cases')
    cases = {}
    for name, raw_batch in raw_batches.items():
        roles = {}
        for role, cfg, state in (
                ('control', control_cfg, control_state),
                ('candidate', candidate_cfg, candidate_state)):
            roles[role] = _run_d13n_role(
                role=role, cfg=cfg, state=state, raw_batch=raw_batch,
                device=device, seed=20260716)
        losses = [value['loss_gradient'] for value in roles.values()]
        checks = {
            'real_batch': bool(raw_batch.get('data_samples')),
            'losses_finite': all(
                value['individual_losses'] and all(torch.isfinite(
                    torch.tensor(number)).item()
                    for number in value['individual_losses'].values())
                for value in losses),
            'gradients_finite': all(value['gradients_finite']
                                    for value in losses),
            'candidate_adapter_gradient': (
                roles['candidate']['loss_gradient']['gradient_norm'] > 0),
            'parameters_unchanged': all(value['parameters_unchanged']
                                        for value in losses),
            'prediction_after_backward': all(
                value['prediction_succeeds_after_backward']
                for value in losses),
        }
        cases[name] = {'passed': all(checks.values()), **checks,
                       'roles': roles}
    return {'passed': all(case['passed'] for case in cases.values()),
            'cases': cases}
def build_stage0_jobs(control_config, candidate_config, checkpoint,
                      output_root):
    configs = {'control': control_config, 'candidate': candidate_config}
    jobs = []
    for role, port, config_name in zip(ROLES, ROLE_PORTS, ROLE_CONFIG):
        stem = role.replace('stage0-', '').replace('proxy-', '')
        jobs.append(build_torchrun_argv(
            config=configs[config_name], checkpoint=checkpoint, role=role,
            out=Path(output_root) / f'{stem}.pkl',
            metrics_out=Path(output_root) / f'{stem}.metrics.json',
            identity_out=Path(output_root) / f'{stem}.identity.json',
            port=port, derive_adapter_free_parent=(role == 'stage0-parent')))
    return jobs
def _load_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise ValueError(f'JSON artifact must be an object: {path}')
    return value
def _checkpoint_state(path):
    payload = torch.load(str(path), map_location='cpu')
    state = payload.get('state_dict', payload)
    if not isinstance(state, Mapping):
        raise ValueError('parent checkpoint state_dict is invalid')
    return {(name[7:] if name.startswith('module.') else name): value
            for name, value in state.items()}
def _build_role(cfg, parent_state):
    model = MODELS.build(copy.deepcopy(cfg.model))
    load = load_parent_into_d13n(model, parent_state)
    wrapper = build_optim_wrapper(model, copy.deepcopy(cfg.optim_wrapper))
    return model, wrapper, load
def _open_vocabulary_audit(model, cfg):
    dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    dataloader_cfg.num_workers = 0
    dataloader_cfg.persistent_workers = False
    raw_batch = next(iter(Runner.build_dataloader(dataloader_cfg)))
    model.eval()
    prepared = model.data_preprocessor(copy.deepcopy(raw_batch), training=False)
    classes = list(cfg.val_dataloader.dataset.metainfo.classes)
    def run_variant(entities):
        samples = copy.deepcopy(prepared['data_samples'])
        for sample in samples:
            set_prompt_metainfo(sample, entities)
        with torch.no_grad():
            predictions = model.predict(prepared['inputs'], samples, rescale=True)
        return [len(sample.pred_instances) for sample in predictions]
    with record_forbidden_runtime_calls() as runtime_hits:
        report = audit_prompt_variants(
            model, {'canonical': classes, 'reversed': list(reversed(classes))},
            run_variant=run_variant, expected_queries=600, class_count=18,
            checkpoint_keys=parent_state_keys(model))
    report['runtime_forbidden_calls'] = runtime_hits
    report['pass'] = report['pass'] and not runtime_hits
    return report
def parent_state_keys(model):
    return [name for name in model.state_dict() if name not in ADAPTER_NAMES]
def _dataset_mouth(cfg):
    def root(dataset):
        data_root = Path(dataset.data_root)
        ann_parent = Path(dataset.ann_file).parent
        image_parent = Path(dataset.data_prefix.img_path).parent
        if ann_parent != image_parent:
            raise ValueError('annotation/image roots differ')
        return data_root / ann_parent
    train = cfg.train_dataloader.dataset
    val = cfg.val_dataloader.dataset
    report = audit_mouth(root(train), root(val))
    classes = list(val.metainfo.classes)
    validate_contract(report, expected_train=1600, expected_val=400,
                      expected_classes=classes)
    report['valid'] = True
    return report
def _fixed_real_batches(cfg):
    expected = {
        'empty': ('P0021__1024__1048___2096', 0),
        'ordinary': ('P0000__1024__1572___1048', 7),
        'dense_1223': ('P4076__1024__2620___0', 1223),
    }
    dataset = DATASETS.build(copy.deepcopy(cfg.train_dataloader.dataset))
    wanted = {image_id: (name, count)
              for name, (image_id, count) in expected.items()}
    batches = {}
    for index in range(len(dataset)):
        item = dataset[index]
        sample = item['data_samples']
        image_id = str(sample.metainfo.get('img_id'))
        if image_id not in wanted:
            continue
        name, count = wanted[image_id]
        if len(sample.gt_instances) != count:
            raise ValueError(f'{name} GT count differs from frozen authority')
        batches[name] = pseudo_collate([item])
        if len(batches) == len(expected):
            break
    if set(batches) != set(expected):
        raise ValueError('frozen empty/ordinary/dense samples are missing')
    return {name: batches[name] for name in expected}
def _artifact_paths(root):
    return {
        stem: {
            'predictions': root / f'{stem}.pkl',
            'metrics': root / f'{stem}.metrics.json',
            'identity': root / f'{stem}.identity.json',
        }
        for stem in ('parent', 'control', 'candidate')}
def _artifact_manifest(paths, identities):
    manifest = {}
    for role, stem in zip(ROLES, ('parent', 'control', 'candidate')):
        group = {}
        for kind, path in paths[stem].items():
            group[kind] = {'path': str(path),
                           'sha256': d13n_runtime.sha256_file(path)}
        identity = identities[role]
        declared_paths = {
            'predictions': identity['predictions_path'],
            'metrics': identity['official_metrics_path']}
        for kind, declared in declared_paths.items():
            actual = _absolute_lexical_path(paths[stem][kind])
            claimed = _absolute_lexical_path(declared)
            actual_pair = (actual, actual.parent.resolve(strict=False) / actual.name)
            claimed_pair = (claimed, claimed.parent.resolve(strict=False) / claimed.name)
            if actual_pair != claimed_pair:
                raise ValueError(f'{role} artifact path differs from child identity')
        if (group['predictions']['sha256'] != identity['predictions_sha256'] or
                group['metrics']['sha256'] !=
                identity['official_metrics_sha256']):
            raise ValueError(f'{role} artifact hash differs from child identity')
        manifest[stem] = group
    return manifest

def _preflight_outputs(root, report_path, fixed):
    targets = [root, report_path]
    for group in fixed.values():
        targets.extend(group.values())
    lexical = [_absolute_lexical_path(path) for path in targets]
    d13n_runtime.assert_paths_unoccupied(lexical)
    physical = [path.parent.resolve(strict=False) / path.name
                for path in lexical]
    if len(set(lexical)) != len(lexical) or len(set(physical)) != len(physical):
        raise FileExistsError('Stage-0 output paths contain an alias')

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-config', type=Path, required=True)
    parser.add_argument('--candidate-config', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser
def execute(args):
    _validate_global_runtime_environment()
    root = _absolute_lexical_path(args.output_root)
    report_path = _absolute_lexical_path(args.report)
    fixed = _artifact_paths(root)
    _preflight_outputs(root, report_path, fixed)
    d13n_runtime.assert_ports_free(list(ROLE_PORTS))
    register_all_modules(init_default_scope=True)
    input_sources = {
        'parent': args.parent, 'control_config': args.control_config,
        'candidate_config': args.candidate_config}
    fingerprint = collect_stage0_fingerprint(
        Path.cwd(), input_sources['parent'], input_sources['control_config'],
        input_sources['candidate_config'])
    os.mkdir(root)
    jobs = build_stage0_jobs(
        args.control_config, args.candidate_config, args.parent, root)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES='5,6,7,8,9',
                       NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')
    for job in jobs:
        try:
            subprocess.run(job, check=True, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            raise Stage0ChildError(str(error)) from error
    identities = {
        role: _load_json(fixed[stem]['identity'])
        for role, stem in zip(ROLES, ('parent', 'control', 'candidate'))}
    _assert_child_input_hashes(identities, fingerprint['inputs'])
    _assert_inputs_unchanged(fingerprint['inputs'], input_sources)
    artifacts = _artifact_manifest(fixed, identities)
    parent_state = _checkpoint_state(args.parent)
    control_cfg = Config.fromfile(str(args.control_config))
    candidate_cfg = Config.fromfile(str(args.candidate_config))
    _assert_inputs_unchanged(fingerprint['inputs'], input_sources)
    control, control_wrapper, control_load = _build_role(
        control_cfg, parent_state)
    candidate, candidate_wrapper, candidate_load = _build_role(
        candidate_cfg, parent_state)
    identity = audit_identity_load(identities, {
        'proxy-control': control_load, 'proxy-candidate': candidate_load},
        fingerprint['inputs'])
    try:
        control_static, control_authority = audit_static_d13n(
            control, control_wrapper)
        candidate_static, authority = audit_static_d13n(
            candidate, candidate_wrapper)
        static = {
            'passed': (control_static['passed'] and
                       candidate_static['passed'] and
                       control_authority == authority),
            'control': control_static, 'candidate': candidate_static,
            'parent_authority_equal': control_authority == authority}
        dump = compare_prediction_dumps(
            fixed['parent']['predictions'], fixed['control']['predictions'],
            fixed['candidate']['predictions'])
        candidate = candidate.cuda(0)
        opened = _open_vocabulary_audit(candidate, candidate_cfg)
        strict = {'forbidden_calls': scan_calls(
            'projects/OVCapFlow/ov_capflow')}
        strict['pass'] = not strict['forbidden_calls']
        protocol = audit_protocol_regression(
            dump_audit=dump, strict=strict, open_vocabulary=opened,
            mouth=_dataset_mouth(control_cfg))
        control_state = {name: value.detach().cpu()
                         for name, value in control.state_dict().items()}
        candidate_state = {name: value.detach().cpu()
                           for name, value in candidate.state_dict().items()}
        del control, control_wrapper, candidate, candidate_wrapper
        torch.cuda.empty_cache()
        smoke = run_real_finite_smoke(
            control_cfg=control_cfg, candidate_cfg=candidate_cfg,
            control_state=control_state, candidate_state=candidate_state,
            raw_batches=_fixed_real_batches(candidate_cfg), device='cuda:0')
    except Exception as error:
        raise ScientificGateError(str(error)) from error
    gates = {
        'identity_load': identity, 'static_d13n': static,
        'zero_step_equality': dump, 'protocol_regression': protocol,
        'real_finite_smoke': smoke,
    }
    _assert_inputs_unchanged(fingerprint['inputs'], input_sources)
    report = build_stage0_report(
        pre_run_commit=fingerprint['git']['commit'], fingerprint=fingerprint,
        inputs=fingerprint['inputs'], artifacts=artifacts, gates=gates,
        model_evidence=authority)
    publish_stage0_report(report_path, report)
    return 0 if report['status'] == 'PASS' else 5
def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except FileExistsError as error:
        print(f'D13-N Stage-0 collision: {error}', file=sys.stderr)
        return 3
    except Stage0ChildError as error:
        print(f'D13-N Stage-0 child failure: {error}', file=sys.stderr)
        return 6
    except ScientificGateError as error:
        print(f'D13-N Stage-0 scientific failure: {error}', file=sys.stderr)
        return 5
    except Exception as error:
        print(f'D13-N Stage-0 identity/load failure: {error}', file=sys.stderr)
        return 4
if __name__ == '__main__':
    raise SystemExit(main())
