import copy
import inspect
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from projects.OVCapFlow.tools import audit_d13n_stage0 as stage0


GATE_NAMES = [
    'identity_load', 'static_d13n', 'zero_step_equality',
    'protocol_regression', 'real_finite_smoke',
]
ADAPTER_NAMES = [
    'bbox_head.existence_residual.weight',
    'bbox_head.existence_residual.bias',
]


def _record(image_id, offset=0.0):
    return {
        'img_id': image_id,
        'pred_instances': {
            'bboxes': torch.arange(15, dtype=torch.float32).reshape(3, 5)
            + offset,
            'scores': torch.arange(3, dtype=torch.float32) + offset,
            'labels': torch.arange(3, dtype=torch.int64),
        },
        'gt_instances': {
            'bboxes': torch.empty((0, 5)),
            'labels': torch.empty((0,), dtype=torch.int64),
        },
    }


def _dump(path, records):
    with path.open('wb') as stream:
        pickle.dump(records, stream)


def _three_dumps(tmp_path):
    paths = [tmp_path / f'{role}.pkl'
             for role in ('parent', 'control', 'candidate')]
    for path in paths:
        _dump(path, [_record('a'), _record('b', 20.0)])
    return paths


def _passing_gate(name):
    return {'passed': True, 'check': name}


def _authority():
    return {
        'parent_parameter_names': ['backbone.weight'],
        'parent_buffer_names': ['running'],
        'parent_parameter_sha256': 'a' * 64,
        'parent_buffer_sha256': 'b' * 64,
    }


def _passing_report():
    gates = {name: _passing_gate(name) for name in GATE_NAMES}
    inputs = {'parent': {'sha256': 'c' * 64}}
    fingerprint = {
        'schema': 'd13n-stage0-inputs-v1', 'git': {'commit': '1' * 40},
        'inputs': inputs}
    fingerprint['fingerprint_sha256'] = \
        stage0.d13n_runtime.canonical_json_sha256(fingerprint)
    return stage0.build_stage0_report(
        pre_run_commit='1' * 40,
        fingerprint=fingerprint, inputs=inputs,
        artifacts={role: {kind: f'/tmp/{role}.{kind}'
                          for kind in ('predictions', 'metrics', 'identity')}
                   for role in ('parent', 'control', 'candidate')},
        gates=gates,
        model_evidence=_authority())


def test_report_has_only_five_top_level_checks_and_runtime_authority():
    report = _passing_report()
    assert report['schema'] == 'd13n-stage0-v1'
    assert report['status'] == 'PASS'
    assert list(report['gates']) == GATE_NAMES
    assert list(report['model_evidence']) == [
        'parent_parameter_names', 'parent_buffer_names',
        'parent_parameter_sha256', 'parent_buffer_sha256']
    assert report['report_sha256'] == stage0.report_sha256(report)


def test_report_is_conjunctive_and_names_failed_core_check():
    gates = {name: _passing_gate(name) for name in GATE_NAMES}
    gates['real_finite_smoke'] = {'passed': False, 'reason': 'non-finite'}
    report = stage0.build_stage0_report(
        pre_run_commit='1' * 40, fingerprint={}, inputs={}, artifacts={},
        gates=gates, model_evidence=_authority())
    assert report['status'] == 'FAIL'
    assert report['failures'] == ['real_finite_smoke']


def test_report_rejects_extra_gate_or_model_authority_key():
    report = _passing_report()
    gates = copy.deepcopy(report['gates'])
    gates['infrastructure'] = _passing_gate('infrastructure')
    with pytest.raises(ValueError, match='five core'):
        stage0.build_stage0_report(
            pre_run_commit='1' * 40, fingerprint={}, inputs={}, artifacts={},
            gates=gates, model_evidence=_authority())
    authority = _authority()
    authority['extra'] = True
    with pytest.raises(ValueError, match='model_evidence'):
        stage0.build_stage0_report(
            pre_run_commit='1' * 40, fingerprint={}, inputs={}, artifacts={},
            gates={name: _passing_gate(name) for name in GATE_NAMES},
            model_evidence=authority)
    bad = copy.deepcopy(report['fingerprint'])
    bad['fingerprint_sha256'] = '0' * 64
    forged = stage0.build_stage0_report(
        pre_run_commit='1' * 40, fingerprint=bad,
        inputs=report['inputs'], artifacts=report['artifacts'],
        gates=report['gates'], model_evidence=report['model_evidence'])
    with pytest.raises(ValueError, match='fingerprint'):
        stage0.publish_stage0_report('/tmp/unused-stage0.json', forged)


def test_publication_is_no_replace(tmp_path):
    path = tmp_path / 'stage0.json'
    stage0.publish_stage0_report(path, _passing_report())
    assert json.loads(path.read_text())['status'] == 'PASS'
    with pytest.raises(FileExistsError):
        stage0.publish_stage0_report(path, _passing_report())


def test_three_way_dump_equality_reuses_q600_validator(tmp_path):
    audit = stage0.compare_prediction_dumps(
        *_three_dumps(tmp_path), expected_records=2, queries_per_image=3)
    assert audit['passed'] is True
    assert audit['prediction_rows'] == 6
    assert audit['source_order_equal'] is True
    assert audit['pairwise'] == {
        'parent_control': {'scores': True, 'labels': True, 'boxes': True},
        'parent_candidate': {'scores': True, 'labels': True, 'boxes': True},
        'control_candidate': {'scores': True, 'labels': True, 'boxes': True},
    }
    assert audit['first_mismatch'] is None


@pytest.mark.parametrize('mutation,field,row,column', [
    ('order', 'source_order', None, None),
    ('score', 'scores', 1, None),
    ('label', 'labels', 2, None),
    ('box', 'boxes', 0, 4),
])
def test_three_way_dump_equality_reports_first_mismatch(
        tmp_path, mutation, field, row, column):
    paths = _three_dumps(tmp_path)
    records = [_record('a'), _record('b', 20.0)]
    if mutation == 'order':
        records.reverse()
    elif mutation == 'score':
        records[0]['pred_instances']['scores'][1] += 1
    elif mutation == 'label':
        records[0]['pred_instances']['labels'][2] = 8
    else:
        records[0]['pred_instances']['bboxes'][0, 4] += 1
    _dump(paths[2], records)
    audit = stage0.compare_prediction_dumps(
        *paths, expected_records=2, queries_per_image=3)
    assert audit['passed'] is False
    assert audit['first_mismatch'] == {
        'pair': 'parent_candidate', 'record_index': 0, 'field': field,
        'row_index': row, 'column_index': column}


def _identity(role, checkpoint_hash='a' * 64):
    config_hash = 'c' * 64 if role == 'proxy-candidate' else 'b' * 64
    return {
        'schema': 'd13n-test-identity-v1',
        'role': role,
        'config_path': f'/repo/{role}.py',
        'config_sha256': config_hash,
        'resolved_config_sha256': 'c' * 64,
        'checkpoint_sha256': checkpoint_hash,
        'finite': True,
        'records': 400,
        'queries_per_image': 600,
        'world_size': 5,
        'local_ranks': [0, 1, 2, 3, 4],
        'cuda_visible_devices': '5,6,7,8,9',
        'derived_parent': role == 'stage0-parent',
        'predictions_path': f'/tmp/{role}.pkl',
        'predictions_sha256': 'd' * 64,
        'official_metrics_path': f'/tmp/{role}.metrics.json',
        'official_metrics_sha256': 'e' * 64,
    }


def _fingerprint_inputs():
    return {
        'parent': {'path': '/tmp/parent.pth', 'sha256': 'a' * 64},
        'control_config': {'path': '/tmp/control.py', 'sha256': 'b' * 64},
        'candidate_config': {'path': '/tmp/candidate.py', 'sha256': 'c' * 64},
    }


def _adapter_load_evidence():
    parent_state = TinyModel().state_dict()
    for name in ADAPTER_NAMES:
        parent_state.pop(name)
    return {
        role: stage0.load_parent_into_d13n(TinyModel(), parent_state)
        for role in ('proxy-control', 'proxy-candidate')
    }


def test_identity_load_gate_aggregates_child_and_actual_load_authority():
    audit = stage0.audit_identity_load({
        role: _identity(role) for role in (
            'stage0-parent', 'proxy-control', 'proxy-candidate')},
        _adapter_load_evidence(), _fingerprint_inputs())
    assert audit['passed'] is True
    assert audit['checkpoint_sha256'] == 'a' * 64
    assert audit['roles'] == [
        'stage0-parent', 'proxy-control', 'proxy-candidate']
    assert list(audit['identities']) == audit['roles']
    assert audit['identities']['stage0-parent']['world_size'] == 5
    assert audit['identities']['stage0-parent']['predictions_sha256'] == 'd' * 64
    assert audit['ports'] == dict(zip(audit['roles'], [29845, 29846, 29847]))


def test_identity_load_gate_fails_closed_on_child_drift():
    identities = {role: _identity(role) for role in (
        'stage0-parent', 'proxy-control', 'proxy-candidate')}
    identities['proxy-candidate']['world_size'] = 4
    with pytest.raises(ValueError, match='proxy-candidate'):
        stage0.audit_identity_load(
            identities, _adapter_load_evidence(), _fingerprint_inputs())


class TinyModel(nn.Module):

    def __init__(self, negative_zero=False):
        super().__init__()
        self.backbone = nn.Linear(256, 256)
        self.bbox_head = nn.Module()
        self.bbox_head.existence_residual = nn.Linear(256, 1)
        self.register_buffer('running', torch.zeros(1))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        adapter = self.bbox_head.existence_residual
        adapter.weight.requires_grad_(True)
        adapter.bias.requires_grad_(True)
        nn.init.zeros_(adapter.weight)
        nn.init.zeros_(adapter.bias)
        if negative_zero:
            adapter.bias.data.copy_(torch.tensor([-0.0]))


def test_static_d13n_gate_binds_actual_optimizer_and_positive_zero(monkeypatch):
    model = TinyModel()
    expected = list(model.bbox_head.existence_residual.parameters())
    monkeypatch.setattr(
        stage0, 'require_d13n_trainable_parameters', lambda value: expected)
    wrapper = SimpleNamespace(
        optimizer=torch.optim.AdamW(expected, lr=1e-4))
    audit, authority = stage0.audit_static_d13n(model, wrapper)
    assert audit['passed'] is True
    assert audit['optimizer_parameter_names'] == ADAPTER_NAMES
    assert audit['adapter_scalars'] == 257
    assert list(authority) == list(_authority())


def test_static_d13n_gate_rejects_negative_zero(monkeypatch):
    model = TinyModel(negative_zero=True)
    expected = list(model.bbox_head.existence_residual.parameters())
    monkeypatch.setattr(
        stage0, 'require_d13n_trainable_parameters', lambda value: expected)
    wrapper = SimpleNamespace(optimizer=torch.optim.AdamW(expected, lr=1e-4))
    with pytest.raises(ValueError, match='positive zero'):
        stage0.audit_static_d13n(model, wrapper)


def test_protocol_gate_is_thin_conjunction_of_existing_audits():
    audit = stage0.audit_protocol_regression(
        dump_audit={'passed': True},
        strict={'pass': True, 'forbidden_calls': []},
        open_vocabulary={'pass': True, 'runtime_forbidden_calls': []},
        mouth={'valid': True})
    assert audit['passed'] is True
    assert audit['checks'] == {
        'strict_q600': True, 'forbidden_calls': True,
        'open_vocabulary': True, 'rotated_mouth': True}
    open_report = {'pass': True, 'runtime_forbidden_calls': [
        {'call': 'torch.topk', 'count': 1}]}
    assert stage0.audit_protocol_regression(
        dump_audit={'passed': True},
        strict={'pass': True, 'forbidden_calls': []},
        open_vocabulary=open_report, mouth={'valid': True})['passed'] is False
    for strict, opened in (({'pass': True}, {'pass': True,
                              'runtime_forbidden_calls': []}),
                           ({'pass': True, 'forbidden_calls': []},
                            {'pass': True})):
        assert stage0.audit_protocol_regression(
            dump_audit={'passed': True}, strict=strict,
            open_vocabulary=opened, mouth={'valid': True})['passed'] is False


def test_real_finite_smoke_calls_existing_real_forward_backward(monkeypatch):
    calls = []
    modes = []
    monkeypatch.setattr(
        stage0.d12_preflight, '_build_loaded_model',
        lambda *args, **kwargs: TinyModel())

    def run_role(**kwargs):
        calls.append(kwargs['role'])
        model = stage0.d12_preflight._build_loaded_model(None, None, None)
        model.train()
        modes.append((model.bbox_head.training,
                      model.bbox_head.existence_residual.training))
        return {
            'loss_gradient': {
                'individual_losses': {'loss_cls': 1.0},
                'total_loss': 1.0, 'gradient_norm': 0.5,
                'gradients_finite': True, 'parameters_unchanged': True,
                'prediction_succeeds_after_backward': True}}

    monkeypatch.setattr(stage0, '_run_role_model', run_role)
    audit = stage0.run_real_finite_smoke(
        control_cfg='control', candidate_cfg='candidate',
        control_state={}, candidate_state={}, raw_batches={
            name: {'data_samples': [1]}
            for name in ('empty', 'ordinary', 'dense_1223')},
        device='cuda:0')
    assert audit['passed'] is True
    assert calls == ['control', 'candidate'] * 3
    assert modes == [(False, True)] * 6
    assert list(audit['cases']) == ['empty', 'ordinary', 'dense_1223']
    for keys in ((), ('empty', 'ordinary'),
                 ('empty', 'ordinary', 'dense_1223', 'extra')):
        with pytest.raises(ValueError, match='exact ordered'):
            stage0.run_real_finite_smoke(
                control_cfg='c', candidate_cfg='x', control_state={},
                candidate_state={}, raw_batches={key: {} for key in keys},
                device='cuda:0')


def test_job_builder_delegates_to_existing_three_role_torchrun(monkeypatch,
                                                               tmp_path):
    calls = []
    monkeypatch.setattr(stage0, 'build_torchrun_argv',
                        lambda **kwargs: calls.append(kwargs) or [kwargs['role']])
    jobs = stage0.build_stage0_jobs(
        Path('control.py'), Path('candidate.py'), Path('epoch_24.pth'),
        tmp_path)
    assert [job[0] for job in jobs] == [
        'stage0-parent', 'proxy-control', 'proxy-candidate']
    assert [call['port'] for call in calls] == [29845, 29846, 29847]
    assert calls[0]['derive_adapter_free_parent'] is True
    assert all(not call.get('derive_adapter_free_parent', False)
               for call in calls[1:])
    paths = stage0._artifact_paths(tmp_path)
    identities = {role: _identity(role) for role in stage0.ROLES}
    for role, stem in zip(stage0.ROLES, paths):
        for path in paths[stem].values():
            path.write_bytes(path.name.encode())
        identities[role]['predictions_path'] = str(paths[stem]['predictions'])
        identities[role]['official_metrics_path'] = str(paths[stem]['metrics'])
        identities[role]['predictions_sha256'] = stage0.d13n_runtime.sha256_file(
            paths[stem]['predictions'])
        identities[role]['official_metrics_sha256'] = \
            stage0.d13n_runtime.sha256_file(paths[stem]['metrics'])
    manifest = stage0._artifact_manifest(paths, identities)
    assert sum(len(group) for group in manifest.values()) == 9
    assert all(set(item) == {'path', 'sha256'}
               for group in manifest.values() for item in group.values())
    identities['proxy-candidate']['predictions_path'] = str(tmp_path / 'other')
    with pytest.raises(ValueError, match='artifact path'):
        stage0._artifact_manifest(paths, identities)


def test_minimal_fingerprint_binds_only_commit_and_three_inputs(tmp_path):
    paths = [tmp_path / name for name in ('parent.pth', 'control.py',
                                          'candidate.py')]
    for path in paths:
        path.write_bytes(path.name.encode())
    value = stage0.collect_stage0_fingerprint(Path.cwd(), *paths)
    assert list(value) == ['schema', 'git', 'inputs', 'fingerprint_sha256']
    assert list(value['inputs']) == ['parent', 'control_config',
                                     'candidate_config']
    stage0._assert_inputs_unchanged(value['inputs'])
    paths[0].write_bytes(b'drift')
    with pytest.raises(ValueError, match='parent changed'):
        stage0._assert_inputs_unchanged(value['inputs'])
    assert inspect.getsource(stage0.execute).count(
        '_assert_inputs_unchanged') == 3


def test_execute_validates_parent_environment_before_any_action(monkeypatch):
    events = []
    monkeypatch.setattr(stage0, '_validate_global_runtime_environment',
                        lambda: events.append('env') or (_ for _ in ()).throw(
                            RuntimeError('environment drift')))
    monkeypatch.setattr(stage0, 'register_all_modules',
                        lambda **kwargs: events.append('registry'))
    with pytest.raises(RuntimeError, match='environment drift'):
        stage0.execute(SimpleNamespace())
    assert events == ['env']


def test_output_alias_and_scientific_exit_classification(tmp_path, monkeypatch):
    root = tmp_path / 'stage0'
    with pytest.raises(FileExistsError, match='alias'):
        stage0._preflight_outputs(root, root, stage0._artifact_paths(root))
    monkeypatch.setattr(stage0, 'execute',
                        lambda args: (_ for _ in ()).throw(
                            stage0.ScientificGateError('bad gate')))
    assert stage0.main(['--control-config=c', '--candidate-config=x',
                        '--parent=p', '--output-root=o', '--report=r']) == 5
    monkeypatch.setattr(stage0, 'execute',
                        lambda args: (_ for _ in ()).throw(ValueError('drift')))
    assert stage0.main(['--control-config=c', '--candidate-config=x',
                        '--parent=p', '--output-root=o', '--report=r']) == 4


@pytest.mark.parametrize('target', ['root', 'report'])
def test_preflight_rejects_dangling_final_symlink(tmp_path, target):
    root = tmp_path / 'root'
    report = tmp_path / 'report.json'
    selected = root if target == 'root' else report
    selected.symlink_to(tmp_path / 'missing-target')
    with pytest.raises(FileExistsError):
        stage0._preflight_outputs(root, report, stage0._artifact_paths(root))
