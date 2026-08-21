import copy
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


RUNTIME_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
                'tools' / 'd13n_runtime.py')


def _load_runtime():
    spec = importlib.util.spec_from_file_location('d13n_runtime', RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_runtime_constants_and_paths_are_exact():
    api = _load_runtime()
    assert api.PHYSICAL_GPUS == (5, 6, 7, 8, 9)
    assert api.GPU_UUIDS == {
        5: 'GPU-6c98da7c-69bd-4c20-5094-ff3b60141515',
        6: 'GPU-b30e9f80-7925-6816-9014-d228784c5a12',
        7: 'GPU-d614452d-c145-3751-ce23-f9e21dd0b535',
        8: 'GPU-f8f2c2c6-4c20-237b-f6df-ba5b5408573e',
        9: 'GPU-ff760a2d-9a1b-38ee-06a0-8c284e002545',
    }
    assert (
        api.CONTROL_PORT, api.CANDIDATE_PORT,
        api.RAW_CONTROL_PORT, api.RAW_CANDIDATE_PORT,
        api.STAGE0_PARENT_PORT, api.STAGE0_CONTROL_PORT,
        api.STAGE0_CANDIDATE_PORT,
    ) == (29842, 29841, 29843, 29844, 29845, 29846, 29847)
    assert api.E24 == Path(
        'work_dirs/dotav2_cleanstart/'
        'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/'
        'epoch_24.pth')
    assert api.CANONICAL_E24_DUMP == Path(
        'work_dirs/dotav2_cleanstart/'
        'eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl')
    assert api.TRAIN_MANIFEST == Path(
        'work_dirs/dotav2_cleanstart/subsets/'
        'seed20260715_rare4x/train/manifest.json')
    assert api.PROXY_MANIFEST == Path(
        'work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json')
    config_root = Path('configs/ov_capflow/dotav2')
    assert api.CONTROL_CONFIG == config_root / (
        'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
        'scale1024_batch2_rare4x_world5_d13n_control.py')
    assert api.CANDIDATE_CONFIG == config_root / (
        'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
        'scale1024_batch2_rare4x_world5_d13n_candidate.py')
    assert api.RAW_CONTROL_CONFIG == config_root / (
        'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
        'scale1024_batch2_rare4x_world5_d13n_control_raw13833.py')
    assert api.RAW_CANDIDATE_CONFIG == config_root / (
        'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
        'scale1024_batch2_rare4x_world5_d13n_candidate_raw13833.py')
    assert api.ANALYZER == Path(
        'projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py')
    assert api.DIAGNOSTICS == Path(
        'projects/OVCapFlow/tools/dotav2_q600_diagnostics.py')
    assert api.VALIDATOR == Path(
        'projects/OVCapFlow/tools/validate_dotav2_q600_dump.py')
    assert api.MONITOR_STARTUP_TIMEOUT_SECONDS == 180
    assert api.MONITOR_HEARTBEAT_MAX_AGE_SECONDS == 150


def test_hashes_are_canonical_and_json_publication_is_no_replace(tmp_path):
    api = _load_runtime()
    source = tmp_path / 'source.bin'
    source.write_bytes(b'abc\x00def')
    assert api.sha256_file(source) == hashlib.sha256(
        source.read_bytes()).hexdigest()
    left = {'z': [2, 1], 'a': {'value': '中文'}}
    right = {'a': {'value': '中文'}, 'z': [2, 1]}
    assert api.canonical_json_sha256(left) == api.canonical_json_sha256(right)
    output = tmp_path / 'report.json'
    api.publish_json_noreplace(output, left)
    assert output.read_bytes() == (
        json.dumps(left, allow_nan=False, ensure_ascii=False,
                   separators=(',', ':'), sort_keys=True) + '\n').encode()
    with pytest.raises(FileExistsError):
        api.publish_json_noreplace(output, right)
    assert json.loads(output.read_text()) == left


def test_normalized_pair_config_removes_only_preregistered_fields():
    api = _load_runtime()
    control = api.normalized_pair_config(api.CONTROL_CONFIG)
    candidate = api.normalized_pair_config(api.CANDIDATE_CONFIG)
    assert control == candidate
    assert 'd13n_role' not in control
    assert 'd13n_master_port' not in control
    assert 'd13n_audit_path' not in control
    assert 'work_dir' not in control
    assert 'existence_loss_weight' not in control['model']['bbox_head']
    assert 'role' not in control['custom_hooks'][0]
    assert 'audit_path' not in control['train_dataloader']['batch_sampler']
    assert control['d13n_pair_id'] == (
        'D13-N-Q600-existence-residual-world5-seed20260716')


def test_expected_environment_is_predeclared_and_every_scalar_is_frozen():
    api = _load_runtime()
    expected = {
        'interpreter': {
            'path': '/data/zcy/anaconda3/envs/mmdet/bin/python',
            'python_version': '3.8.19',
        },
        'software': {
            'torch': '1.12.1+cu113',
            'torch_cuda': '11.3',
            'mmengine': '0.10.4',
            'mmdet': '3.3.0',
            'mmrotate': '1.0.0rc1',
        },
        'driver': {'version': '580.173.02', 'reported_cuda': '13.0'},
        'gpus': [
            {'physical_index': index, 'name': 'NVIDIA A40', 'uuid': uuid}
            for index, uuid in api.GPU_UUIDS.items()
        ],
    }
    assert api.EXPECTED_ENVIRONMENT == expected
    assert api.validate_frozen_environment(copy.deepcopy(expected)) == expected
    scalar_paths = [
        ('interpreter', 'path'), ('interpreter', 'python_version'),
        *[('software', key) for key in expected['software']],
        ('driver', 'version'), ('driver', 'reported_cuda'),
    ]
    for outer, inner in scalar_paths:
        mutated = copy.deepcopy(expected)
        mutated[outer][inner] += '-drift'
        with pytest.raises(RuntimeError, match='environment'):
            api.validate_frozen_environment(mutated)
    for index in range(5):
        for field in ('physical_index', 'name', 'uuid'):
            mutated = copy.deepcopy(expected)
            value = mutated['gpus'][index][field]
            mutated['gpus'][index][field] = (
                value + '-drift' if type(value) is str else value + 1)
            with pytest.raises(RuntimeError, match='environment'):
                api.validate_frozen_environment(mutated)
    for mutated_gpus in (
            expected['gpus'][:-1], list(reversed(expected['gpus']))):
        mutated = copy.deepcopy(expected)
        mutated['gpus'] = copy.deepcopy(mutated_gpus)
        with pytest.raises(RuntimeError, match='environment'):
            api.validate_frozen_environment(mutated)


def _populate_fingerprint_inputs(api, repo):
    paths = {
        'parent': repo / api.E24,
        'canonical_raw_dump': repo / api.CANONICAL_E24_DUMP,
        'train_manifest': repo / api.TRAIN_MANIFEST,
        'proxy_manifest': repo / api.PROXY_MANIFEST,
        'control_config': repo / api.CONTROL_CONFIG,
        'candidate_config': repo / api.CANDIDATE_CONFIG,
        'raw_control_config': repo / api.RAW_CONTROL_CONFIG,
        'raw_candidate_config': repo / api.RAW_CANDIDATE_CONFIG,
        'analyzer': repo / api.ANALYZER,
        'diagnostics': repo / api.DIAGNOSTICS,
        'validator': repo / api.VALIDATOR,
    }
    for index, path in enumerate(paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes('frozen-input-{}'.format(index).encode())
    return paths


def test_collect_environment_validates_before_hashing_and_binds_all_inputs(
        tmp_path, monkeypatch):
    api = _load_runtime()
    paths = _populate_fingerprint_inputs(api, tmp_path)
    expected_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    monkeypatch.setattr(api, 'EXPECTED_INPUT_SHA256', expected_hashes)
    monkeypatch.setattr(
        api, '_collect_environment_observation',
        lambda: copy.deepcopy(api.EXPECTED_ENVIRONMENT))
    monkeypatch.setattr(
        api, '_git_identity',
        lambda repo: {'commit': 'a' * 40, 'tracked_dirty': False})
    events = []
    real_validate = api.validate_frozen_environment
    real_hash = api.sha256_file

    def validate_first(observed):
        events.append('validate')
        return real_validate(observed)

    def hash_after_validate(path):
        events.append('hash')
        return real_hash(path)

    monkeypatch.setattr(api, 'validate_frozen_environment', validate_first)
    monkeypatch.setattr(api, 'sha256_file', hash_after_validate)
    fingerprint = api.collect_environment_fingerprint(
        tmp_path, paths['control_config'], paths['candidate_config'],
        paths['raw_control_config'], paths['raw_candidate_config'])

    assert events[0] == 'validate'
    assert fingerprint['schema'] == 'd13n-environment-v1'
    assert fingerprint['interpreter'] == api.EXPECTED_ENVIRONMENT['interpreter']
    assert fingerprint['software'] == api.EXPECTED_ENVIRONMENT['software']
    assert fingerprint['driver'] == api.EXPECTED_ENVIRONMENT['driver']
    assert fingerprint['gpus'] == api.EXPECTED_ENVIRONMENT['gpus']
    assert fingerprint['git'] == {
        'commit': 'a' * 40, 'tracked_dirty': False}
    assert list(fingerprint['inputs']) == list(paths)
    for name, path in paths.items():
        assert fingerprint['inputs'][name] == {
            'path': str(path.resolve()), 'sha256': expected_hashes[name]}
    assert fingerprint['analyzer_source_commit'] == \
        'abd860d157d562aa9d30422c05e41e246effc8a7'
    unhashed = dict(fingerprint)
    digest = unhashed.pop('fingerprint_sha256')
    assert digest == api.canonical_json_sha256(unhashed)


@pytest.mark.parametrize('drift_name', [
    'parent', 'canonical_raw_dump', 'train_manifest', 'proxy_manifest',
    'control_config', 'candidate_config', 'raw_control_config',
    'raw_candidate_config', 'analyzer', 'diagnostics', 'validator',
])
def test_collect_environment_rejects_each_input_hash_drift(
        tmp_path, monkeypatch, drift_name):
    api = _load_runtime()
    paths = _populate_fingerprint_inputs(api, tmp_path)
    expected_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    expected_hashes[drift_name] = '0' * 64
    monkeypatch.setattr(api, 'EXPECTED_INPUT_SHA256', expected_hashes)
    monkeypatch.setattr(
        api, '_collect_environment_observation',
        lambda: copy.deepcopy(api.EXPECTED_ENVIRONMENT))
    monkeypatch.setattr(
        api, '_git_identity',
        lambda repo: {'commit': 'a' * 40, 'tracked_dirty': False})

    with pytest.raises(RuntimeError, match='input'):
        api.collect_environment_fingerprint(
            tmp_path, paths['control_config'], paths['candidate_config'],
            paths['raw_control_config'], paths['raw_candidate_config'])


def test_assert_fingerprint_equal_is_recursive_typed_and_hash_bound(
        tmp_path, monkeypatch):
    api = _load_runtime()
    paths = _populate_fingerprint_inputs(api, tmp_path)
    monkeypatch.setattr(api, 'EXPECTED_INPUT_SHA256', {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    })
    monkeypatch.setattr(
        api, '_collect_environment_observation',
        lambda: copy.deepcopy(api.EXPECTED_ENVIRONMENT))
    monkeypatch.setattr(
        api, '_git_identity',
        lambda repo: {'commit': 'a' * 40, 'tracked_dirty': False})
    expected = api.collect_environment_fingerprint(
        tmp_path, paths['control_config'], paths['candidate_config'],
        paths['raw_control_config'], paths['raw_candidate_config'])
    assert api.assert_fingerprint_equal(
        expected, copy.deepcopy(expected)) is None
    mutations = []
    changed_gpu = copy.deepcopy(expected)
    changed_gpu['gpus'][0]['uuid'] = 'GPU-drift'
    mutations.append(changed_gpu)
    changed_input = copy.deepcopy(expected)
    changed_input['inputs']['parent']['sha256'] = '0' * 64
    mutations.append(changed_input)
    changed_hash = copy.deepcopy(expected)
    changed_hash['fingerprint_sha256'] = '0' * 64
    mutations.append(changed_hash)
    changed_type = copy.deepcopy(expected)
    changed_type['git']['tracked_dirty'] = 0
    mutations.append(changed_type)
    for observed in mutations:
        with pytest.raises(RuntimeError, match='fingerprint'):
            api.assert_fingerprint_equal(expected, observed)


def _world5_processes(port=29842):
    config = 'configs/ov_capflow/dotav2/control.py'
    common_env = {
        'LOCAL_WORLD_SIZE': '5',
        'WORLD_SIZE': '5',
        'MASTER_PORT': str(port),
        'CUDA_VISIBLE_DEVICES': '5,6,7,8,9',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_IB_DISABLE': '1',
    }
    processes = [{
        'pid': 100,
        'ppid': 1,
        'argv': [
            'rtk', 'env', 'PYTHONNOUSERSITE=1',
            'CUDA_VISIBLE_DEVICES=5,6,7,8,9',
            '/data/zcy/anaconda3/envs/mmdet/bin/python', '-m',
            'torch.distributed.run', '--nproc_per_node=5',
            '--master_port={}'.format(port), 'tools/train.py', config,
        ],
        'environ': {},
        'listening_ports': [],
    }, {
        'pid': 101,
        'ppid': 100,
        'argv': [
            '/data/zcy/anaconda3/envs/mmdet/bin/python', '-m',
            'torch.distributed.run', '--nproc_per_node=5',
            '--master_port={}'.format(port), 'tools/train.py', config,
        ],
        'environ': {},
        'listening_ports': [port],
    }]
    for rank in range(5):
        environment = dict(common_env, LOCAL_RANK=str(rank))
        processes.append({
            'pid': 110 + rank,
            'ppid': 101,
            'argv': [
                '/data/zcy/anaconda3/envs/mmdet/bin/python',
                'tools/train.py', config, '--launcher', 'pytorch',
            ],
            'environ': environment,
            'listening_ports': [],
        })
    return processes


def test_process_classifier_distinguishes_rtk_wrapper_from_real_launcher():
    api = _load_runtime()
    snapshot = api.classify_process_snapshot(_world5_processes())
    assert [item['pid'] for item in snapshot['wrappers']] == [100]
    assert [item['pid'] for item in snapshot['launchers']] == [101]
    assert [item['pid'] for item in snapshot['workers']] == [
        110, 111, 112, 113, 114]
    assert api.assert_world5_topology(snapshot, 29842) == snapshot


@pytest.mark.parametrize('argv,listening_ports', [
    ([
        '/data/zcy/anaconda3/envs/mmdet/bin/python',
        'tools/train.py', 'configs/ov_capflow/dotav2/control.py',
    ], []),
    ([
        '/wrong/python', '-m', 'torch.distributed.run',
        '--nproc_per_node=5', '--master_port=29842',
        'tools/train.py', 'configs/ov_capflow/dotav2/control.py',
    ], [29842]),
    ([
        '/usr/bin/python3', 'tools/train.py',
        'configs/ov_capflow/dotav2/control.py', '--launcher', 'pytorch',
    ], []),
])
def test_world5_topology_rejects_train_like_or_selected_port_other(
        argv, listening_ports):
    api = _load_runtime()
    processes = _world5_processes()
    processes.append({
        'pid': 200,
        'ppid': 1,
        'argv': argv,
        'environ': {},
        'listening_ports': listening_ports,
    })

    snapshot = api.classify_process_snapshot(processes)
    assert [process['pid'] for process in snapshot['others']] == [200]
    with pytest.raises(RuntimeError, match='topology'):
        api.assert_world5_topology(snapshot, 29842)


def test_world5_topology_allows_ordinary_unrelated_other():
    api = _load_runtime()
    processes = _world5_processes()
    processes.append({
        'pid': 200,
        'ppid': 1,
        'argv': ['/usr/bin/python3', 'unrelated_job.py'],
        'environ': {},
        'listening_ports': [12345],
    })

    snapshot = api.classify_process_snapshot(processes)

    assert api.assert_world5_topology(snapshot, 29842) == snapshot


@pytest.mark.parametrize('escape', ['launcher_child', 'distributed_env'])
def test_world5_topology_rejects_other_process_escape(escape):
    api = _load_runtime()
    processes = _world5_processes()
    other = {
        'pid': 200,
        'ppid': 1,
        'argv': ['/wrong/python', 'worker_bootstrap.py'],
        'environ': {},
        'listening_ports': [],
    }
    if escape == 'launcher_child':
        other['ppid'] = 101
    else:
        other['environ'] = dict(
            processes[2]['environ'], LOCAL_RANK='3')
    processes.append(other)

    snapshot = api.classify_process_snapshot(processes)
    assert [process['pid'] for process in snapshot['others']] == [200]
    with pytest.raises(RuntimeError, match='topology'):
        api.assert_world5_topology(snapshot, 29842)


def test_world5_topology_allows_exact_forked_dataloader_worker_clone():
    api = _load_runtime()
    processes = _world5_processes()
    clone = copy.deepcopy(processes[2])
    clone['pid'] = 200
    clone['ppid'] = 110
    processes.append(clone)

    snapshot = api.classify_process_snapshot(processes)

    assert [process['pid'] for process in snapshot['workers']] == [
        110, 111, 112, 113, 114]
    assert [process['pid'] for process in snapshot['others']] == [200]
    assert api.assert_world5_topology(snapshot, 29842) == snapshot


@pytest.mark.parametrize('mutation', ['argv', 'environ', 'listener'])
def test_world5_topology_rejects_drifted_dataloader_worker_clone(mutation):
    api = _load_runtime()
    processes = _world5_processes()
    clone = copy.deepcopy(processes[2])
    clone['pid'] = 200
    clone['ppid'] = 110
    if mutation == 'argv':
        clone['argv'].append('--unexpected')
    elif mutation == 'environ':
        clone['environ']['WORLD_SIZE'] = '4'
    else:
        clone['listening_ports'] = [12345]
    processes.append(clone)

    snapshot = api.classify_process_snapshot(processes)
    assert [process['pid'] for process in snapshot['others']] == [200]
    with pytest.raises(RuntimeError, match='topology'):
        api.assert_world5_topology(snapshot, 29842)


@pytest.mark.parametrize('mutation', [
    'unreadable_env', 'missing_local_rank', 'wrong_rank',
    'wrong_local_world_size', 'wrong_world_size', 'wrong_port',
    'wrong_cuda_visible_devices', 'wrong_nccl_p2p', 'wrong_nccl_ib',
    'duplicate_worker', 'duplicate_launcher', 'wrong_listener',
    'missing_listener', 'extra_listener', 'missing_wrapper',
])
def test_world5_topology_fails_closed_on_every_process_contract(mutation):
    api = _load_runtime()
    processes = _world5_processes()
    if mutation == 'unreadable_env':
        processes[-1]['environ'] = None
    elif mutation == 'missing_local_rank':
        del processes[-1]['environ']['LOCAL_RANK']
    elif mutation == 'wrong_rank':
        processes[-1]['environ']['LOCAL_RANK'] = '3'
    elif mutation == 'wrong_local_world_size':
        processes[-1]['environ']['LOCAL_WORLD_SIZE'] = '4'
    elif mutation == 'wrong_world_size':
        processes[-1]['environ']['WORLD_SIZE'] = '4'
    elif mutation == 'wrong_cuda_visible_devices':
        processes[-1]['environ']['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
    elif mutation == 'wrong_nccl_p2p':
        processes[-1]['environ']['NCCL_P2P_DISABLE'] = '0'
    elif mutation == 'wrong_nccl_ib':
        processes[-1]['environ']['NCCL_IB_DISABLE'] = '0'
    elif mutation == 'duplicate_worker':
        duplicate = copy.deepcopy(processes[-1])
        duplicate['pid'] = 120
        processes.append(duplicate)
    elif mutation == 'duplicate_launcher':
        duplicate = copy.deepcopy(processes[1])
        duplicate['pid'] = 102
        processes.append(duplicate)
    elif mutation == 'wrong_port':
        processes[-1]['environ']['MASTER_PORT'] = '29999'
    elif mutation == 'wrong_listener':
        processes[1]['listening_ports'] = [29999]
    elif mutation == 'missing_listener':
        processes[1]['listening_ports'] = []
    elif mutation == 'extra_listener':
        processes[1]['listening_ports'] = [29842, 29999]
    else:
        processes.pop(0)
    snapshot = api.classify_process_snapshot(processes)
    with pytest.raises(RuntimeError, match='topology'):
        api.assert_world5_topology(snapshot, 29842)


def test_path_and_port_idle_checks_are_read_only_and_fail_closed(tmp_path):
    api = _load_runtime()
    output = tmp_path / 'artifact.json'
    assert api.assert_paths_unoccupied([output]) is None
    pending = output.with_name(output.name + '.pending.1.0')
    pending.write_bytes(b'evidence')
    with pytest.raises(FileExistsError):
        api.assert_paths_unoccupied([output])
    pending.unlink()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match='port'):
            api.assert_ports_free([port])
    finally:
        listener.close()
    assert api.assert_ports_free([port]) is None


READY_KEYS = [
    'schema', 'pid', 'proc_start_ticks', 'proc_cmdline_sha256',
    'created_monotonic_ns', 'owner_root', 'owner_log', 'monitor_log',
    'failure_lock', 'stage0_report_path', 'stage0_report_sha256',
    'pre_run_commit', 'report_sha256',
]


def _write_proc_identity(proc_root, pid, start_ticks, argv):
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    tail = ['S'] + ['0'] * 18 + [str(start_ticks)] + ['0'] * 4
    (process / 'stat').write_text(
        '{} (d13n monitor) {}\n'.format(pid, ' '.join(tail)))
    (process / 'cmdline').write_bytes(
        b'\0'.join(os.fsencode(value) for value in argv) + b'\0')


def _ready_fixture(api, tmp_path, *, create_process=True):
    owner_root = (tmp_path / 'formal-root').resolve()
    owner_root.mkdir()
    stage0 = (tmp_path / 'stage0.json').resolve()
    stage0.write_bytes(b'{"schema":"d13n-stage0-v1"}\n')
    owner_log = owner_root / 'owner.jsonl'
    monitor_log = owner_root / 'monitor.jsonl'
    failure_lock = owner_root / 'integrity_failure.lock'
    ready_path = owner_root / 'monitor.ready'
    terminal = owner_root / 'monitor_terminal.json'
    commit = 'b' * 40
    pid = 4321
    ticks = 987654
    argv = [
        '/data/zcy/anaconda3/envs/mmdet/bin/python',
        'projects/OVCapFlow/tools/monitor_d13n_run.py',
        '--owner-log', str(owner_log), '--owner-root', str(owner_root),
        '--stage0-report', str(stage0), '--pre-run-commit', commit,
        '--output', str(monitor_log), '--ready-output', str(ready_path),
        '--terminal-output', str(terminal),
        '--failure-lock', str(failure_lock), '--poll-seconds', '60',
    ]
    proc_root = tmp_path / 'proc'
    proc_root.mkdir()
    if create_process:
        _write_proc_identity(proc_root, pid, ticks, argv)
    payload = {
        'pid': pid,
        'proc_start_ticks': ticks,
        'proc_cmdline_sha256': hashlib.sha256(
            b'\0'.join(os.fsencode(value) for value in argv) + b'\0'
        ).hexdigest(),
        'created_monotonic_ns': 10_000_000_000,
        'owner_root': owner_root,
        'owner_log': owner_log,
        'monitor_log': monitor_log,
        'failure_lock': failure_lock,
        'stage0_report_path': stage0,
        'stage0_report_sha256': api.sha256_file(stage0),
        'pre_run_commit': commit,
    }
    expected = {
        key: (str(value.resolve()) if isinstance(value, Path) else value)
        for key, value in payload.items()
        if key in (
            'owner_root', 'owner_log', 'monitor_log', 'failure_lock',
            'stage0_report_path', 'stage0_report_sha256', 'pre_run_commit')
    }
    return {
        'payload': payload,
        'expected': expected,
        'ready_path': ready_path,
        'proc_root': proc_root,
        'stage0': stage0,
        'argv': argv,
    }


def test_monitor_ready_is_exact_hash_bound_no_replace_and_live(tmp_path):
    api = _load_runtime()
    fixture = _ready_fixture(api, tmp_path)
    report = api.publish_monitor_ready(
        fixture['ready_path'], fixture['payload'])

    assert list(report) == READY_KEYS
    assert report['schema'] == 'd13n-monitor-ready-v1'
    for key in ('owner_root', 'owner_log', 'monitor_log', 'failure_lock',
                'stage0_report_path'):
        assert report[key] == str(fixture['payload'][key].resolve())
    unhashed = dict(report)
    digest = unhashed.pop('report_sha256')
    assert digest == api.canonical_json_sha256(unhashed)
    assert api.validate_monitor_ready(
        fixture['ready_path'], fixture['expected'],
        proc_root=fixture['proc_root']) == report
    with pytest.raises(FileExistsError):
        api.publish_monitor_ready(
            fixture['ready_path'], fixture['payload'])


@pytest.mark.parametrize('mutation', [
    'never_started', 'dead', 'pid_reuse', 'wrong_cmdline_hash',
    'wrong_script', 'wrong_argv_root', 'wrong_bound_root', 'wrong_commit',
    'stage0_drift', 'wrong_stage0_hash',
])
def test_monitor_ready_rejects_process_or_binding_drift(tmp_path, mutation):
    api = _load_runtime()
    fixture = _ready_fixture(
        api, tmp_path, create_process=mutation != 'never_started')
    if mutation == 'wrong_cmdline_hash':
        fixture['payload']['proc_cmdline_sha256'] = '0' * 64
    elif mutation == 'wrong_bound_root':
        fixture['payload']['owner_root'] = tmp_path / 'other-root'
    elif mutation == 'wrong_commit':
        fixture['payload']['pre_run_commit'] = 'c' * 40
    elif mutation == 'wrong_stage0_hash':
        fixture['payload']['stage0_report_sha256'] = '0' * 64
    elif mutation in ('wrong_script', 'wrong_argv_root'):
        argv = list(fixture['argv'])
        if mutation == 'wrong_script':
            argv[1] = 'projects/OVCapFlow/tools/not_the_monitor.py'
        else:
            argv[argv.index('--owner-root') + 1] = str(tmp_path / 'wrong')
        process = fixture['proc_root'] / str(fixture['payload']['pid'])
        cmdline = b'\0'.join(
            os.fsencode(value) for value in argv) + b'\0'
        (process / 'cmdline').write_bytes(cmdline)
        fixture['payload']['proc_cmdline_sha256'] = hashlib.sha256(
            cmdline).hexdigest()
    report = api.publish_monitor_ready(
        fixture['ready_path'], fixture['payload'])
    process = fixture['proc_root'] / str(report['pid'])
    if mutation == 'dead':
        for child in process.iterdir():
            child.unlink()
        process.rmdir()
    elif mutation == 'pid_reuse':
        argv = fixture['argv']
        for child in process.iterdir():
            child.unlink()
        process.rmdir()
        _write_proc_identity(
            fixture['proc_root'], report['pid'],
            report['proc_start_ticks'] + 1, argv)
    elif mutation == 'stage0_drift':
        fixture['stage0'].write_bytes(b'drifted-stage0')
    with pytest.raises(RuntimeError, match='monitor ready'):
        api.validate_monitor_ready(
            fixture['ready_path'], fixture['expected'],
            proc_root=fixture['proc_root'])


SAMPLE_KEYS = [
    'schema', 'sequence', 'timestamp', 'monotonic_ns', 'monitor_pid',
    'proc_start_ticks', 'owner_event_sha256', 'previous_sample_sha256',
    'gpu', 'topology', 'console', 'training', 'd13n_telemetry', 'sampler',
    'checkpoint', 'fatal_patterns', 'status', 'sample_sha256',
]


def _monitor_sample(api, sequence, monotonic_ns, previous, owner_sha='d' * 64,
                    pid=4321, ticks=987654):
    sample = {
        'schema': 'd13n-monitor-sample-v1',
        'sequence': sequence,
        'timestamp': '2026-08-01T12:00:00+08:00',
        'monotonic_ns': monotonic_ns,
        'monitor_pid': pid,
        'proc_start_ticks': ticks,
        'owner_event_sha256': owner_sha,
        'previous_sample_sha256': previous,
        'gpu': [],
        'topology': {},
        'console': {},
        'training': {},
        'd13n_telemetry': {},
        'sampler': {},
        'checkpoint': {},
        'fatal_patterns': [],
        'status': 'RUNNING',
    }
    sample['sample_sha256'] = api.canonical_json_sha256(sample)
    return sample


def _append_jsonl(path, records, trailing=b''):
    payload = b''.join(
        (json.dumps(record, allow_nan=False, ensure_ascii=False,
                    separators=(',', ':'), sort_keys=True) + '\n').encode()
        for record in records)
    path.write_bytes(payload + trailing)


def test_monitor_heartbeat_accepts_exact_150_seconds_and_ignores_truncated_last(
        tmp_path):
    api = _load_runtime()
    monitor_log = tmp_path / 'monitor.jsonl'
    first = _monitor_sample(api, 1, 1_000_000_000, None)
    second = _monitor_sample(
        api, 2, 2_000_000_000, first['sample_sha256'])
    _append_jsonl(monitor_log, [first, second], trailing=b'{"truncated":')
    ready = {'pid': 4321, 'proc_start_ticks': 987654}

    observed = api.validate_monitor_heartbeat(
        monitor_log, ready, 'd' * 64,
        now_monotonic_ns=second['monotonic_ns'] + 150_000_000_000)

    assert list(observed) == SAMPLE_KEYS
    assert observed == second


@pytest.mark.parametrize('mutation', [
    'missing_first', 'truncated_only', 'bad_previous', 'bad_sample_hash',
    'wrong_owner', 'wrong_pid', 'wrong_start_ticks', 'future_timestamp',
    'too_old', 'sequence_gap',
])
def test_monitor_heartbeat_rejects_missing_chain_identity_or_age(
        tmp_path, mutation):
    api = _load_runtime()
    monitor_log = tmp_path / 'monitor.jsonl'
    ready = {'pid': 4321, 'proc_start_ticks': 987654}
    first = _monitor_sample(api, 1, 1_000_000_000, None)
    second = _monitor_sample(
        api, 2, 2_000_000_000, first['sample_sha256'])
    records = [first, second]
    trailing = b''
    now = second['monotonic_ns'] + 150_000_000_000
    if mutation == 'missing_first':
        records = []
    elif mutation == 'truncated_only':
        records = []
        trailing = b'{"schema":"d13n-monitor-sample-v1"}'
    elif mutation == 'bad_previous':
        second['previous_sample_sha256'] = '0' * 64
        second['sample_sha256'] = api.canonical_json_sha256({
            key: value for key, value in second.items()
            if key != 'sample_sha256'})
    elif mutation == 'bad_sample_hash':
        second['sample_sha256'] = '0' * 64
    elif mutation == 'wrong_owner':
        second['owner_event_sha256'] = '0' * 64
        second['sample_sha256'] = api.canonical_json_sha256({
            key: value for key, value in second.items()
            if key != 'sample_sha256'})
    elif mutation == 'wrong_pid':
        second['monitor_pid'] += 1
        second['sample_sha256'] = api.canonical_json_sha256({
            key: value for key, value in second.items()
            if key != 'sample_sha256'})
    elif mutation == 'wrong_start_ticks':
        second['proc_start_ticks'] += 1
        second['sample_sha256'] = api.canonical_json_sha256({
            key: value for key, value in second.items()
            if key != 'sample_sha256'})
    elif mutation == 'future_timestamp':
        now = second['monotonic_ns'] - 1
    elif mutation == 'too_old':
        now += 1
    else:
        second['sequence'] = 3
        second['sample_sha256'] = api.canonical_json_sha256({
            key: value for key, value in second.items()
            if key != 'sample_sha256'})
    _append_jsonl(monitor_log, records, trailing=trailing)

    with pytest.raises(RuntimeError, match='heartbeat'):
        api.validate_monitor_heartbeat(
            monitor_log, ready, 'd' * 64, now_monotonic_ns=now)


FAILURE_KEYS = [
    'schema', 'status', 'sequence', 'timestamp', 'owner_state',
    'failure_class', 'owner_event', 'sample', 'evidence_paths', 'failures',
    'report_sha256',
]
FAILURE_CLASSES = [
    'identity', 'topology', 'nonfinite_fatal', 'collision',
    'checkpoint_parent', 'checkpoint_buffer', 'optimizer_binding',
    'control_head', 'sampler', 'gpu_contract', 'input_runtime',
]


def test_pre_sample_integrity_failure_lock_is_exact_hash_bound_and_loadable(
        tmp_path):
    api = _load_runtime()
    present = tmp_path / 'console.log'
    present.write_bytes(b'failure evidence')
    missing = tmp_path / 'never-created.json'
    lock_path = tmp_path / 'integrity_failure.lock'
    report = api.publish_integrity_failure_lock(
        lock_path, sequence=0, timestamp='2026-08-01T12:00:00+08:00',
        owner_state='UNKNOWN', failure_class='input_runtime',
        owner_event=None, sample=None,
        evidence_paths=[present, missing], failures=['monitor never ready'])

    assert list(report) == FAILURE_KEYS
    assert report['schema'] == 'd13n-integrity-failure-v1'
    assert report['status'] == 'FAIL'
    assert report['owner_event'] == {'present': False, 'sha256': None}
    assert report['sample'] == {'present': False, 'sha256': None}
    assert report['evidence_paths'] == {
        str(present.resolve()): {
            'exists': True, 'sha256': api.sha256_file(present)},
        str(missing.resolve()): {'exists': False, 'sha256': None},
    }
    unhashed = dict(report)
    digest = unhashed.pop('report_sha256')
    assert digest == api.canonical_json_sha256(unhashed)
    assert api.load_integrity_failure_lock(lock_path) == report
    with pytest.raises(FileExistsError):
        api.publish_integrity_failure_lock(
            lock_path, sequence=0, timestamp='x', owner_state='UNKNOWN',
            failure_class='input_runtime', owner_event=None, sample=None,
            evidence_paths=[], failures=['duplicate'])


@pytest.mark.parametrize('failure_class', FAILURE_CLASSES)
def test_integrity_failure_lock_accepts_only_frozen_failure_classes(
        tmp_path, failure_class):
    api = _load_runtime()
    path = tmp_path / (failure_class + '.lock')
    report = api.publish_integrity_failure_lock(
        path, sequence=1, timestamp='2026-08-01T12:00:00+08:00',
        owner_state='TRAIN_CONTROL', failure_class=failure_class,
        owner_event={'event': 'sample'}, sample=None,
        evidence_paths=[], failures=['terminal'])
    assert report['failure_class'] == failure_class
    invalid = tmp_path / (failure_class + '.invalid')
    with pytest.raises((TypeError, ValueError), match='failure_class'):
        api.publish_integrity_failure_lock(
            invalid, sequence=1, timestamp='x', owner_state='TRAIN_CONTROL',
            failure_class=failure_class + '-invalid', owner_event=None,
            sample=None, evidence_paths=[], failures=['terminal'])
    assert not invalid.exists()


def test_integrity_failure_lock_uses_direct_exclusive_create_and_two_fsyncs(
        tmp_path, monkeypatch):
    api = _load_runtime()
    path = tmp_path / 'integrity_failure.lock'
    opened = []
    synced = []
    real_open = api.os.open
    real_fsync = api.os.fsync

    def track_open(target, flags, *args, **kwargs):
        descriptor = real_open(target, flags, *args, **kwargs)
        opened.append((os.fspath(target), flags, descriptor))
        return descriptor

    def track_fsync(descriptor):
        synced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(api.os, 'open', track_open)
    monkeypatch.setattr(api.os, 'fsync', track_fsync)
    api.publish_integrity_failure_lock(
        path, sequence=0, timestamp='2026-08-01T12:00:00+08:00',
        owner_state='UNKNOWN', failure_class='input_runtime',
        owner_event=None, sample=None, evidence_paths=[], failures=['fatal'])

    final_open = next(item for item in opened if item[0] == str(path.resolve()))
    assert final_open[1] & os.O_CREAT
    assert final_open[1] & os.O_EXCL
    assert final_open[2] in synced
    assert any(descriptor in synced and descriptor != final_open[2]
               for _, _, descriptor in opened)


CACHE_RECORD_KEYS = ['path', 'size', 'mtime_ns', 'sha256', 'sha_cached']


def test_checkpoint_hash_cache_records_identity_then_reuses_stable_key(
        tmp_path):
    api = _load_runtime()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'checkpoint-v1')
    cache = api.CheckpointHashCache()

    first = cache.hash_file(checkpoint)
    second = cache.hash_file(checkpoint)

    observed = checkpoint.stat()
    assert list(first) == CACHE_RECORD_KEYS
    assert first == {
        'path': str(checkpoint.resolve()),
        'size': observed.st_size,
        'mtime_ns': observed.st_mtime_ns,
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'sha_cached': False,
    }
    assert second == dict(first, sha_cached=True)


@pytest.mark.parametrize('mutation', ['size', 'mtime_ns'])
def test_checkpoint_hash_cache_rehashes_when_size_or_mtime_changes(
        tmp_path, mutation):
    api = _load_runtime()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'checkpoint-v1')
    cache = api.CheckpointHashCache()
    first = cache.hash_file(checkpoint)
    if mutation == 'size':
        checkpoint.write_bytes(b'checkpoint-version-two')
    else:
        checkpoint.write_bytes(b'checkpoint-v2')
        original = checkpoint.stat().st_mtime_ns
        os.utime(checkpoint, ns=(original + 1_000_000, original + 1_000_000))

    changed = cache.hash_file(checkpoint)

    assert changed['sha_cached'] is False
    assert changed['sha256'] == hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    assert (changed['size'], changed['mtime_ns']) != (
        first['size'], first['mtime_ns'])


def test_checkpoint_hash_cache_resolves_path_alias_to_one_cache_key(tmp_path):
    api = _load_runtime()
    physical = tmp_path / 'physical.pth'
    physical.write_bytes(b'checkpoint')
    alias = tmp_path / 'alias.pth'
    alias.symlink_to(physical)
    cache = api.CheckpointHashCache()

    first = cache.hash_file(alias)
    second = cache.hash_file(physical)

    assert first['path'] == second['path'] == str(physical.resolve())
    assert first['sha_cached'] is False
    assert second['sha_cached'] is True
    assert first['sha256'] == second['sha256']


def test_checkpoint_hash_cache_rehashes_same_size_restored_mtime_replace(
        tmp_path):
    api = _load_runtime()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'checkpoint-v1')
    original = checkpoint.stat()
    cache = api.CheckpointHashCache()
    first = cache.hash_file(checkpoint)
    replacement = tmp_path / 'replacement.pth'
    replacement.write_bytes(b'checkpoint-v2')
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, checkpoint)

    changed = cache.hash_file(checkpoint)

    assert (changed['size'], changed['mtime_ns']) == (
        first['size'], first['mtime_ns'])
    assert changed['sha_cached'] is False
    assert changed['sha256'] == hashlib.sha256(b'checkpoint-v2').hexdigest()
    assert changed['sha256'] != first['sha256']


def test_checkpoint_hash_cache_rejects_fifo_without_blocking(tmp_path):
    fifo = tmp_path / 'checkpoint.pth'
    os.mkfifo(fifo)
    program = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('d13n_runtime', sys.argv[1])
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
try:
    api.CheckpointHashCache().hash_file(sys.argv[2])
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(3)
"""

    completed = subprocess.run(
        [sys.executable, '-c', program, str(RUNTIME_PATH), str(fifo)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=5, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))

    assert completed.returncode == 0, completed.stderr.decode()


def test_checkpoint_hash_cache_closes_raw_fd_when_path_authority_fails(
        tmp_path, monkeypatch):
    api = _load_runtime()
    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'checkpoint')
    opened = []
    closed = []
    real_open = api.os.open
    real_close = api.os.close

    def track_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def track_close(descriptor):
        closed.append(descriptor)
        return real_close(descriptor)

    def fail_lstat(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(api.os, 'open', track_open)
    monkeypatch.setattr(api.os, 'close', track_close)
    monkeypatch.setattr(api.os, 'lstat', fail_lstat)

    with pytest.raises(RuntimeError, match='checkpoint.*open'):
        api.CheckpointHashCache().hash_file(checkpoint)

    assert len(opened) == 1
    assert closed == opened
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_checkpoint_hash_cache_stats_before_hashing_missing_path(
        tmp_path, monkeypatch):
    api = _load_runtime()
    calls = []
    monkeypatch.setattr(
        api, 'sha256_file', lambda path: calls.append(path) or '0' * 64)
    cache = api.CheckpointHashCache()

    with pytest.raises(FileNotFoundError):
        cache.hash_file(tmp_path / 'missing.pth')

    assert calls == []


def test_checkpoint_hash_cache_hash_error_is_fail_closed_and_never_cached(
        tmp_path, monkeypatch):
    api = _load_runtime()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'checkpoint')
    calls = []

    def fail_hash(stream):
        calls.append(stream.fileno())
        raise OSError('injected checkpoint read failure')

    monkeypatch.setattr(api, '_sha256_stream', fail_hash)
    cache = api.CheckpointHashCache()

    for _ in range(2):
        with pytest.raises(RuntimeError, match='checkpoint hash'):
            cache.hash_file(checkpoint)

    assert len(calls) == 2


import torch
from collections import OrderedDict


D13N_ADAPTER_NAMES = [
    'bbox_head.existence_residual.weight',
    'bbox_head.existence_residual.bias',
]
D13N_INTEGRITY_KEYS = [
    'schema', 'role', 'epoch', 'iter',
    'optimizer_parameter_names_by_group', 'optimizer_class',
    'live_optimizer_group_sizes', 'serialized_optimizer_group_sizes',
    'serialized_optimizer_parameter_ids_by_group',
    'initialized_optimizer_state_slots', 'parent_state_sha256',
    'adapter_state_sha256', 'config_sha256', 'integrity_sha256',
]


def _d13n_hash_named_tensor_state(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = json.dumps({
            'name': name,
            'dtype': str(value.dtype),
            'shape': list(value.shape),
        }, sort_keys=True, separators=(',', ':'),
                            ensure_ascii=True).encode('utf-8')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, byteorder='big'))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, byteorder='big'))
        digest.update(raw)
    return digest.hexdigest()


def _d13n_adamw_settings():
    return {
        'lr': 0.0001,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 0.05,
        'amsgrad': False,
        'foreach': None,
        'maximize': False,
        'capturable': False,
    }


def _d13n_synthetic_checkpoint(tmp_path, role, *, initial_lr=False):
    assert role in ('control', 'candidate')
    parent_parameters = {
        'backbone.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    parent_buffers = {
        'backbone.running_mean': torch.tensor([0.25, -0.5],
                                              dtype=torch.float32),
    }
    if role == 'control':
        adapter = {
            D13N_ADAPTER_NAMES[0]: torch.zeros(1, 256),
            D13N_ADAPTER_NAMES[1]: torch.zeros(1),
        }
    else:
        adapter = {
            D13N_ADAPTER_NAMES[0]: torch.full((1, 256), 0.25),
            D13N_ADAPTER_NAMES[1]: torch.tensor([0.125]),
        }
    state_dict = OrderedDict([
        *parent_parameters.items(),
        *parent_buffers.items(),
        *adapter.items(),
    ])
    settings = _d13n_adamw_settings()
    groups = []
    for parameter_id in (7, 19):
        group = dict(settings)
        if initial_lr:
            group['initial_lr'] = settings['lr']
        group['params'] = [parameter_id]
        groups.append(group)
    base = {'params': torch.zeros(1), **settings}
    if initial_lr:
        base['initial_lr'] = settings['lr']
    slots = {
        7: {
            'step': torch.tensor(1920.0, dtype=torch.float32),
            'exp_avg': torch.full((1, 256), 0.01),
            'exp_avg_sq': torch.full((1, 256), 0.001),
        },
        19: {
            'step': torch.tensor(1920.0, dtype=torch.float32),
            'exp_avg': torch.tensor([0.02], dtype=torch.float32),
            'exp_avg_sq': torch.tensor([0.002], dtype=torch.float32),
        },
    }
    config = "model = dict(d13n_role={!r})\n".format(role)
    parent_state = {**parent_parameters, **parent_buffers}
    integrity = {
        'schema': 'd13n-checkpoint-integrity-v1',
        'role': role,
        'epoch': 12,
        'iter': 1920,
        'optimizer_parameter_names_by_group': [
            [D13N_ADAPTER_NAMES[0]], [D13N_ADAPTER_NAMES[1]]],
        'optimizer_class': 'torch.optim.adamw.AdamW',
        'live_optimizer_group_sizes': [1, 1],
        'serialized_optimizer_group_sizes': [1, 1],
        'serialized_optimizer_parameter_ids_by_group': [[7], [19]],
        'initialized_optimizer_state_slots': 2,
        'parent_state_sha256': _d13n_hash_named_tensor_state(parent_state),
        'adapter_state_sha256': _d13n_hash_named_tensor_state(adapter),
        'config_sha256': hashlib.sha256(config.encode('utf-8')).hexdigest(),
    }
    integrity['integrity_sha256'] = hashlib.sha256(json.dumps(
        integrity, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')).hexdigest()
    assert list(integrity) == D13N_INTEGRITY_KEYS
    checkpoint = {
        'meta': {
            'epoch': 12,
            'iter': 1920,
            'cfg': config,
            'seed': 20260716,
            'experiment_name': 'd13n-{}-formal'.format(role),
            'time': '20260801_120000',
            'mmengine_version': '0.10.4',
            'dataset_meta': {'classes': ('plane', 'ship')},
            'd13n_integrity': integrity,
        },
        'state_dict': state_dict,
        'optimizer': {
            'state': slots,
            'param_groups': groups,
            'base_param_settings': base,
        },
        'message_hub': {'runtime_info': {}, 'log_scalars': {}},
        'param_schedulers': [],
    }
    checkpoint_path = tmp_path / '{}_epoch_12.pth'.format(role)
    torch.save(checkpoint, checkpoint_path)
    stage0_report = {
        'schema': 'd13n-stage0-v1',
        'status': 'PASS',
        'model_evidence': {
            'parent_parameter_names': list(parent_parameters),
            'parent_buffer_names': list(parent_buffers),
            'parent_parameter_sha256': _d13n_hash_named_tensor_state(
                parent_parameters),
            'parent_buffer_sha256': _d13n_hash_named_tensor_state(
                parent_buffers),
        },
    }
    expected = {
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_sha256': hashlib.sha256(
            checkpoint_path.read_bytes()).hexdigest(),
        'role': role,
        'endpoint_epoch': 12,
        'optimizer_parameter_names': D13N_ADAPTER_NAMES,
        'parent_parameter_sha256': stage0_report['model_evidence'][
            'parent_parameter_sha256'],
        'parent_buffer_sha256': stage0_report['model_evidence'][
            'parent_buffer_sha256'],
        'adapter_state_sha256': integrity['adapter_state_sha256'],
        'adapter_is_zero': role == 'control',
    }
    return checkpoint_path, stage0_report, expected


def test_audit_d13n_checkpoint_accepts_exact_control_and_candidate_fixture(
        tmp_path):
    api = _load_runtime()
    cache = api.CheckpointHashCache()
    for role, initial_lr in (('control', False), ('candidate', True)):
        checkpoint, stage0_report, expected = _d13n_synthetic_checkpoint(
            tmp_path, role, initial_lr=initial_lr)

        first = api.audit_d13n_checkpoint(
            checkpoint, stage0_report, role, cache)
        second = api.audit_d13n_checkpoint(
            checkpoint, stage0_report, role, cache)

        for key, value in expected.items():
            assert first[key] == value
            assert second[key] == value
        assert first['checkpoint_sha_cached'] is False
        assert second['checkpoint_sha_cached'] is True


def test_audit_d13n_checkpoint_uses_one_open_file_and_rejects_path_replace(
        tmp_path, monkeypatch):
    api = _load_runtime()
    checkpoint, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=False)
    cache = api.CheckpointHashCache()
    api.audit_d13n_checkpoint(
        checkpoint, stage0_report, 'candidate', cache)
    replacement = tmp_path / 'replacement.pth'
    replacement.write_bytes(b'not a checkpoint')
    real_load = api.torch.load
    loaded_from = []

    def replace_path_then_load(source, *args, **kwargs):
        loaded_from.append((source, source.seekable()))
        os.replace(replacement, checkpoint)
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(api.torch, 'load', replace_path_then_load)

    with pytest.raises(RuntimeError, match='audit load identity changed'):
        api.audit_d13n_checkpoint(
            checkpoint, stage0_report, 'candidate', cache)

    assert len(loaded_from) == 1
    assert hasattr(loaded_from[0][0], 'read')
    assert loaded_from[0][1] is True


def test_audit_d13n_checkpoint_rejects_in_place_aba_during_load(
        tmp_path, monkeypatch):
    api = _load_runtime()
    checkpoint, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=False)
    real_load = api.torch.load

    def mutate_ctime_after_load(source, *args, **kwargs):
        payload = real_load(source, *args, **kwargs)
        before = checkpoint.stat()
        with checkpoint.open('r+b') as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(first)
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(
            checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert checkpoint.stat().st_ctime_ns != before.st_ctime_ns
        return payload

    monkeypatch.setattr(api.torch, 'load', mutate_ctime_after_load)

    with pytest.raises(RuntimeError, match='checkpoint.*changed'):
        api.audit_d13n_checkpoint(
            checkpoint, stage0_report, 'candidate',
            api.CheckpointHashCache())


def test_audit_d13n_checkpoint_rejects_parent_directory_rename_during_load(
        tmp_path, monkeypatch):
    api = _load_runtime()
    formal_root = tmp_path / 'formal'
    formal_root.mkdir()
    checkpoint, stage0_report, _ = _d13n_synthetic_checkpoint(
        formal_root, 'candidate', initial_lr=False)
    moved_root = tmp_path / 'moved'
    real_load = api.torch.load

    def rename_parent_after_load(source, *args, **kwargs):
        payload = real_load(source, *args, **kwargs)
        formal_root.rename(moved_root)
        return payload

    monkeypatch.setattr(api.torch, 'load', rename_parent_after_load)

    with pytest.raises(RuntimeError, match='checkpoint.*path'):
        api.audit_d13n_checkpoint(
            checkpoint, stage0_report, 'candidate',
            api.CheckpointHashCache())


def test_audit_d13n_checkpoint_rejects_symlink_retarget_during_load(
        tmp_path, monkeypatch):
    api = _load_runtime()
    checkpoint, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=False)
    replacement = tmp_path / 'replacement.pth'
    replacement.write_bytes(checkpoint.read_bytes())
    alias = tmp_path / 'alias.pth'
    alias.symlink_to(checkpoint)
    real_load = api.torch.load

    def retarget_alias_after_load(source, *args, **kwargs):
        payload = real_load(source, *args, **kwargs)
        alias.unlink()
        alias.symlink_to(replacement)
        return payload

    monkeypatch.setattr(api.torch, 'load', retarget_alias_after_load)

    with pytest.raises(RuntimeError, match='checkpoint.*path'):
        api.audit_d13n_checkpoint(
            alias, stage0_report, 'candidate',
            api.CheckpointHashCache())


def test_audit_d13n_checkpoint_rejects_old_post_params_initial_lr_order(
        tmp_path):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=True)
    checkpoint = torch.load(source, map_location='cpu')
    groups = []
    for group in checkpoint['optimizer']['param_groups']:
        reordered = {
            key: value for key, value in group.items()
            if key not in ('params', 'initial_lr')
        }
        reordered['params'] = group['params']
        reordered['initial_lr'] = group['initial_lr']
        groups.append(reordered)
    checkpoint['optimizer']['param_groups'] = groups
    mutated = _d13n_save_mutation(source, checkpoint, 'old-initial-lr-order')

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, 'candidate',
            api.CheckpointHashCache())


class _D13NDictSubclass(dict):
    pass


class _D13NListSubclass(list):
    pass


class _D13NTupleSubclass(tuple):
    pass


@pytest.mark.parametrize('mutation', [
    'optimizer_dict_subclass',
    'optimizer_order',
    'optimizer_extra',
    'state_mapping_subclass',
    'param_groups_list_subclass',
    'param_groups_tuple',
    'group_dict_subclass',
    'params_list_subclass',
    'params_tuple',
    'base_dict_subclass',
    'slot_dict_subclass',
    'betas_list',
    'betas_tuple_subclass',
    'integrity_dict_subclass',
    'integrity_names_outer_list_subclass',
    'integrity_names_group_list_subclass',
])
def test_audit_d13n_checkpoint_rejects_optimizer_or_integrity_container_drift(
        tmp_path, mutation):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=True)
    checkpoint = torch.load(source, map_location='cpu')
    optimizer = checkpoint['optimizer']
    meta = checkpoint['meta']
    integrity = meta['d13n_integrity']

    if mutation == 'optimizer_dict_subclass':
        checkpoint['optimizer'] = _D13NDictSubclass(optimizer)
    elif mutation == 'optimizer_order':
        checkpoint['optimizer'] = {
            'param_groups': optimizer['param_groups'],
            'state': optimizer['state'],
            'base_param_settings': optimizer['base_param_settings'],
        }
    elif mutation == 'optimizer_extra':
        optimizer['unexpected'] = None
    elif mutation == 'state_mapping_subclass':
        optimizer['state'] = _D13NDictSubclass(optimizer['state'])
    elif mutation == 'param_groups_list_subclass':
        optimizer['param_groups'] = _D13NListSubclass(
            optimizer['param_groups'])
    elif mutation == 'param_groups_tuple':
        optimizer['param_groups'] = tuple(optimizer['param_groups'])
    elif mutation == 'group_dict_subclass':
        optimizer['param_groups'][0] = _D13NDictSubclass(
            optimizer['param_groups'][0])
    elif mutation == 'params_list_subclass':
        optimizer['param_groups'][0]['params'] = _D13NListSubclass([7])
    elif mutation == 'params_tuple':
        optimizer['param_groups'][0]['params'] = (7, )
    elif mutation == 'base_dict_subclass':
        optimizer['base_param_settings'] = _D13NDictSubclass(
            optimizer['base_param_settings'])
    elif mutation == 'slot_dict_subclass':
        optimizer['state'][7] = _D13NDictSubclass(optimizer['state'][7])
    elif mutation in ('betas_list', 'betas_tuple_subclass'):
        replacement_type = (
            list if mutation == 'betas_list' else _D13NTupleSubclass)
        for settings in [
                *optimizer['param_groups'],
                optimizer['base_param_settings']]:
            settings['betas'] = replacement_type((0.9, 0.999))
    elif mutation == 'integrity_dict_subclass':
        meta['d13n_integrity'] = _D13NDictSubclass(integrity)
    elif mutation == 'integrity_names_outer_list_subclass':
        integrity['optimizer_parameter_names_by_group'] = _D13NListSubclass(
            integrity['optimizer_parameter_names_by_group'])
    else:
        integrity['optimizer_parameter_names_by_group'][0] = (
            _D13NListSubclass(
                integrity['optimizer_parameter_names_by_group'][0]))

    mutated = tmp_path / '{}.pth'.format(mutation)
    torch.save(checkpoint, mutated)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, 'candidate', api.CheckpointHashCache())


class _D13NStringSubclass(str):
    pass


class _D13NIntSubclass(int):
    pass


class _D13NFloatSubclass(float):
    pass


def _d13n_refresh_integrity(checkpoint):
    integrity = checkpoint['meta']['d13n_integrity']
    unhashed = {
        key: value for key, value in integrity.items()
        if key != 'integrity_sha256'
    }
    integrity['integrity_sha256'] = hashlib.sha256(json.dumps(
        unhashed, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')).hexdigest()


def _d13n_save_mutation(source, checkpoint, mutation):
    target = source.with_name('mutated-{}.pth'.format(mutation))
    torch.save(checkpoint, target)
    return target


def _d13n_replace_exact_key(mapping, original, replacement):
    return {
        replacement if key == original else key: value
        for key, value in mapping.items()
    }


@pytest.mark.parametrize('location', [
    'integrity_schema', 'optimizer_state', 'slot_step', 'group_base_lr',
])
def test_audit_d13n_checkpoint_rejects_string_subclass_schema_keys(
        tmp_path, location):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=False)
    checkpoint = torch.load(source, map_location='cpu')
    optimizer = checkpoint['optimizer']
    subclass_key = _D13NStringSubclass(
        'schema' if location == 'integrity_schema' else
        'state' if location == 'optimizer_state' else
        'step' if location == 'slot_step' else 'lr')

    if location == 'integrity_schema':
        integrity = checkpoint['meta']['d13n_integrity']
        checkpoint['meta']['d13n_integrity'] = _d13n_replace_exact_key(
            integrity, 'schema', subclass_key)
    elif location == 'optimizer_state':
        checkpoint['optimizer'] = _d13n_replace_exact_key(
            optimizer, 'state', subclass_key)
    elif location == 'slot_step':
        optimizer['state'][7] = _d13n_replace_exact_key(
            optimizer['state'][7], 'step', subclass_key)
    else:
        optimizer['param_groups'] = [
            _d13n_replace_exact_key(group, 'lr', subclass_key)
            for group in optimizer['param_groups']
        ]
        optimizer['base_param_settings'] = _d13n_replace_exact_key(
            optimizer['base_param_settings'], 'lr', subclass_key)

    mutated = _d13n_save_mutation(source, checkpoint, location)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, 'candidate',
            api.CheckpointHashCache())


@pytest.mark.parametrize('key_type', [bool, _D13NIntSubclass])
def test_audit_d13n_checkpoint_rejects_nonexact_optimizer_state_int_keys(
        tmp_path, key_type):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=False)
    checkpoint = torch.load(source, map_location='cpu')
    state = checkpoint['optimizer']['state']
    replacement = key_type(7)
    checkpoint['optimizer']['state'] = {
        replacement: state[7],
        19: state[19],
    }
    mutated = _d13n_save_mutation(source, checkpoint, key_type.__name__)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, 'candidate',
            api.CheckpointHashCache())


@pytest.mark.parametrize('mutation', [
    'missing_integrity', 'none_integrity', 'missing_integrity_key',
    'extra_integrity_key', 'integrity_key_order', 'wrong_schema',
    'role_subclass', 'expected_role_subclass', 'wrong_expected_arm',
    'meta_cfg_subclass', 'empty_meta_cfg', 'meta_epoch_bool', 'meta_iter_bool',
    'integrity_epoch_bool', 'integrity_iter_bool', 'epoch_mismatch',
    'iter_mismatch', 'non_endpoint_epoch', 'non_endpoint_iter',
    'config_hash_drift', 'parent_hash_drift', 'adapter_hash_drift',
    'nested_name_tuple', 'nested_group_size_bool', 'selfhash_uppercase',
    'selfhash_nonhex', 'selfhash_mismatch',
])
def test_audit_d13n_checkpoint_rejects_integrity_role_or_runtime_drift(
        tmp_path, mutation):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=True)
    checkpoint = torch.load(source, map_location='cpu')
    meta = checkpoint['meta']
    integrity = meta['d13n_integrity']
    expected_role = 'candidate'
    refresh = False

    if mutation == 'missing_integrity':
        del meta['d13n_integrity']
    elif mutation == 'none_integrity':
        meta['d13n_integrity'] = None
    elif mutation == 'missing_integrity_key':
        del integrity['config_sha256']
        refresh = True
    elif mutation == 'extra_integrity_key':
        integrity['unexpected'] = None
        refresh = True
    elif mutation == 'integrity_key_order':
        reordered = {
            key: integrity[key]
            for key in reversed(D13N_INTEGRITY_KEYS)
        }
        meta['d13n_integrity'] = integrity = reordered
        refresh = True
    elif mutation == 'wrong_schema':
        integrity['schema'] = 'd13n-checkpoint-integrity-v2'
        refresh = True
    elif mutation == 'role_subclass':
        integrity['role'] = _D13NStringSubclass('candidate')
        refresh = True
    elif mutation == 'expected_role_subclass':
        expected_role = _D13NStringSubclass('candidate')
    elif mutation == 'wrong_expected_arm':
        expected_role = 'control'
    elif mutation == 'meta_cfg_subclass':
        meta['cfg'] = _D13NStringSubclass(meta['cfg'])
    elif mutation == 'empty_meta_cfg':
        meta['cfg'] = ''
        integrity['config_sha256'] = hashlib.sha256(b'').hexdigest()
        refresh = True
    elif mutation == 'meta_epoch_bool':
        meta['epoch'] = True
    elif mutation == 'meta_iter_bool':
        meta['iter'] = True
    elif mutation == 'integrity_epoch_bool':
        integrity['epoch'] = True
        refresh = True
    elif mutation == 'integrity_iter_bool':
        integrity['iter'] = True
        refresh = True
    elif mutation == 'epoch_mismatch':
        integrity['epoch'] = 11
        refresh = True
    elif mutation == 'iter_mismatch':
        integrity['iter'] = 1919
        refresh = True
    elif mutation == 'non_endpoint_epoch':
        meta['epoch'] = integrity['epoch'] = 11
        refresh = True
    elif mutation == 'non_endpoint_iter':
        meta['iter'] = integrity['iter'] = 1919
        refresh = True
    elif mutation == 'config_hash_drift':
        integrity['config_sha256'] = '0' * 64
        refresh = True
    elif mutation == 'parent_hash_drift':
        integrity['parent_state_sha256'] = '0' * 64
        refresh = True
    elif mutation == 'adapter_hash_drift':
        integrity['adapter_state_sha256'] = '0' * 64
        refresh = True
    elif mutation == 'nested_name_tuple':
        integrity['optimizer_parameter_names_by_group'][0] = (
            D13N_ADAPTER_NAMES[0], )
        refresh = True
    elif mutation == 'nested_group_size_bool':
        integrity['live_optimizer_group_sizes'][0] = True
        refresh = True
    elif mutation == 'selfhash_uppercase':
        integrity['integrity_sha256'] = integrity[
            'integrity_sha256'].upper()
    elif mutation == 'selfhash_nonhex':
        integrity['integrity_sha256'] = 'g' * 64
    else:
        integrity['integrity_sha256'] = '0' * 64

    if refresh:
        _d13n_refresh_integrity(checkpoint)
    mutated = _d13n_save_mutation(source, checkpoint, mutation)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, expected_role,
            api.CheckpointHashCache())


@pytest.mark.parametrize('mutation', [
    'names_reversed', 'names_duplicate', 'names_flat', 'name_unknown',
    'live_group_size_drift', 'serialized_group_size_drift',
    'initialized_slot_count_drift', 'metadata_ids_duplicate',
    'metadata_id_bool', 'metadata_ids_reversed', 'group_ids_reversed',
    'group_ids_duplicate', 'group_id_bool', 'state_keys_reversed',
    'state_key_bool', 'missing_state', 'extra_state', 'missing_slot_field',
    'extra_slot_field', 'slot_field_order', 'missing_optimizer_key',
    'one_group', 'extra_group', 'empty_group', 'two_ids_in_group',
])
def test_audit_d13n_checkpoint_rejects_name_id_group_or_slot_binding_drift(
        tmp_path, mutation):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=True)
    checkpoint = torch.load(source, map_location='cpu')
    optimizer = checkpoint['optimizer']
    state = optimizer['state']
    groups = optimizer['param_groups']
    integrity = checkpoint['meta']['d13n_integrity']
    refresh = False

    if mutation == 'names_reversed':
        integrity['optimizer_parameter_names_by_group'].reverse()
        refresh = True
    elif mutation == 'names_duplicate':
        integrity['optimizer_parameter_names_by_group'][1] = [
            D13N_ADAPTER_NAMES[0]]
        refresh = True
    elif mutation == 'names_flat':
        integrity['optimizer_parameter_names_by_group'] = list(
            D13N_ADAPTER_NAMES)
        refresh = True
    elif mutation == 'name_unknown':
        integrity['optimizer_parameter_names_by_group'][1] = [
            'bbox_head.existence_residual.unknown']
        refresh = True
    elif mutation == 'live_group_size_drift':
        integrity['live_optimizer_group_sizes'] = [2, 0]
        refresh = True
    elif mutation == 'serialized_group_size_drift':
        integrity['serialized_optimizer_group_sizes'] = [0, 2]
        refresh = True
    elif mutation == 'initialized_slot_count_drift':
        integrity['initialized_optimizer_state_slots'] = 1
        refresh = True
    elif mutation == 'metadata_ids_duplicate':
        integrity['serialized_optimizer_parameter_ids_by_group'] = [[7], [7]]
        refresh = True
    elif mutation == 'metadata_id_bool':
        integrity['serialized_optimizer_parameter_ids_by_group'] = (
            [[True], [19]])
        refresh = True
    elif mutation == 'metadata_ids_reversed':
        integrity['serialized_optimizer_parameter_ids_by_group'] = [[19], [7]]
        refresh = True
    elif mutation == 'group_ids_reversed':
        groups[0]['params'], groups[1]['params'] = (
            groups[1]['params'], groups[0]['params'])
    elif mutation == 'group_ids_duplicate':
        groups[1]['params'] = [7]
    elif mutation == 'group_id_bool':
        groups[0]['params'] = [True]
    elif mutation == 'state_keys_reversed':
        optimizer['state'] = {19: state[19], 7: state[7]}
    elif mutation == 'state_key_bool':
        optimizer['state'] = {True: state[7], 19: state[19]}
    elif mutation == 'missing_state':
        del state[19]
    elif mutation == 'extra_state':
        state[23] = copy.deepcopy(state[19])
    elif mutation == 'missing_slot_field':
        del state[7]['exp_avg']
    elif mutation == 'extra_slot_field':
        state[7]['max_exp_avg_sq'] = torch.zeros(1, 256)
    elif mutation == 'slot_field_order':
        state[7] = {
            'exp_avg': state[7]['exp_avg'],
            'step': state[7]['step'],
            'exp_avg_sq': state[7]['exp_avg_sq'],
        }
    elif mutation == 'missing_optimizer_key':
        del optimizer['base_param_settings']
    elif mutation == 'one_group':
        del groups[1]
    elif mutation == 'extra_group':
        groups.append(copy.deepcopy(groups[1]))
    elif mutation == 'empty_group':
        groups[0]['params'] = []
    else:
        groups[0]['params'] = [7, 19]

    if refresh:
        _d13n_refresh_integrity(checkpoint)
    mutated = _d13n_save_mutation(source, checkpoint, mutation)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, 'candidate',
            api.CheckpointHashCache())


@pytest.mark.parametrize('mutation', [
    'optimizer_class_drift', 'missing_group_field', 'extra_group_field',
    'group_key_order', 'missing_base_field', 'extra_base_field',
    'base_key_order', 'real_group_parity', 'base_group_parity',
    'initial_lr_one_group_missing', 'initial_lr_base_missing',
    'lr_int', 'lr_float_subclass', 'lr_negative', 'lr_nan',
    'eps_int', 'eps_zero', 'eps_infinite', 'weight_decay_int',
    'weight_decay_negative', 'beta_first_int', 'beta_second_subclass',
    'beta_out_of_domain', 'beta_nonfinite', 'beta_wrong_length',
    'amsgrad_true', 'foreach_false', 'maximize_true', 'capturable_true',
    'initial_lr_int', 'initial_lr_negative',
])
def test_audit_d13n_checkpoint_rejects_adamw_schema_domain_or_parity_drift(
        tmp_path, mutation):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=True)
    checkpoint = torch.load(source, map_location='cpu')
    optimizer = checkpoint['optimizer']
    groups = optimizer['param_groups']
    base = optimizer['base_param_settings']
    settings = [*groups, base]

    if mutation == 'optimizer_class_drift':
        checkpoint['meta']['d13n_integrity'][
            'optimizer_class'] = 'torch.optim.sgd.SGD'
        _d13n_refresh_integrity(checkpoint)
    elif mutation == 'missing_group_field':
        del groups[0]['eps']
    elif mutation == 'extra_group_field':
        groups[0]['fused'] = False
    elif mutation == 'group_key_order':
        group = groups[0]
        groups[0] = {
            'params': group['params'],
            **{key: value for key, value in group.items()
               if key != 'params'},
        }
    elif mutation == 'missing_base_field':
        del base['eps']
    elif mutation == 'extra_base_field':
        base['fused'] = False
    elif mutation == 'base_key_order':
        optimizer['base_param_settings'] = {
            **{key: value for key, value in base.items()
               if key != 'params'},
            'params': base['params'],
        }
    elif mutation == 'real_group_parity':
        groups[1]['lr'] = 0.0002
    elif mutation == 'base_group_parity':
        base['weight_decay'] = 0.01
    elif mutation == 'initial_lr_one_group_missing':
        del groups[0]['initial_lr']
    elif mutation == 'initial_lr_base_missing':
        del base['initial_lr']
    elif mutation == 'lr_int':
        for item in settings:
            item['lr'] = 0
    elif mutation == 'lr_float_subclass':
        for item in settings:
            item['lr'] = _D13NFloatSubclass(0.0001)
    elif mutation == 'lr_negative':
        for item in settings:
            item['lr'] = -0.0001
    elif mutation == 'lr_nan':
        for item in settings:
            item['lr'] = float('nan')
    elif mutation == 'eps_int':
        for item in settings:
            item['eps'] = 1
    elif mutation == 'eps_zero':
        for item in settings:
            item['eps'] = 0.0
    elif mutation == 'eps_infinite':
        for item in settings:
            item['eps'] = float('inf')
    elif mutation == 'weight_decay_int':
        for item in settings:
            item['weight_decay'] = 0
    elif mutation == 'weight_decay_negative':
        for item in settings:
            item['weight_decay'] = -0.01
    elif mutation == 'beta_first_int':
        for item in settings:
            item['betas'] = (0, 0.999)
    elif mutation == 'beta_second_subclass':
        for item in settings:
            item['betas'] = (0.9, _D13NFloatSubclass(0.999))
    elif mutation == 'beta_out_of_domain':
        for item in settings:
            item['betas'] = (0.9, 1.0)
    elif mutation == 'beta_nonfinite':
        for item in settings:
            item['betas'] = (float('nan'), 0.999)
    elif mutation == 'beta_wrong_length':
        for item in settings:
            item['betas'] = (0.9, )
    elif mutation == 'amsgrad_true':
        for item in settings:
            item['amsgrad'] = True
    elif mutation == 'foreach_false':
        for item in settings:
            item['foreach'] = False
    elif mutation == 'maximize_true':
        for item in settings:
            item['maximize'] = True
    elif mutation == 'capturable_true':
        for item in settings:
            item['capturable'] = True
    elif mutation == 'initial_lr_int':
        for item in settings:
            item['initial_lr'] = 0
    else:
        for item in settings:
            item['initial_lr'] = -0.0001

    mutated = _d13n_save_mutation(source, checkpoint, mutation)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, 'candidate',
            api.CheckpointHashCache())


def _d13n_inference_tensor(factory):
    with torch.inference_mode():
        value = factory()
    assert value.is_inference()
    return value


@pytest.mark.parametrize('mutation', [
    'base_parameter_subclass', 'base_meta_device', 'base_float64',
    'base_wrong_shape', 'base_nonzero', 'base_requires_grad', 'base_grad',
    'base_inference', 'base_fake_singleton_stride', 'base_offset_view',
    'base_conjugate', 'base_negative_view', 'step_parameter_subclass',
    'step_meta_device', 'step_float64', 'step_wrong_shape', 'step_zero',
    'step_fractional', 'step_nonfinite', 'step_requires_grad', 'step_grad',
    'step_inference', 'step_offset_view', 'moment_parameter_subclass',
    'moment_meta_device', 'moment_float64', 'moment_wrong_shape',
    'moment_nonfinite', 'second_moment_negative', 'moment_requires_grad',
    'moment_grad',
    'moment_inference', 'moment_fake_singleton_stride',
    'moment_offset_view', 'moment_conjugate', 'moment_negative_view',
    'moment_zero_pointer', 'six_state_identity_alias',
    'six_state_storage_alias', 'base_state_identity_alias',
    'base_state_storage_alias', 'optimizer_model_identity_alias',
    'optimizer_model_storage_alias',
])
def test_audit_d13n_checkpoint_rejects_noncanonical_tensor_or_storage_drift(
        tmp_path, monkeypatch, mutation):
    api = _load_runtime()
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, 'candidate', initial_lr=True)
    checkpoint = torch.load(source, map_location='cpu')
    optimizer = checkpoint['optimizer']
    base = optimizer['base_param_settings']
    weight_slot = optimizer['state'][7]
    bias_slot = optimizer['state'][19]

    if mutation == 'base_parameter_subclass':
        base['params'] = torch.nn.Parameter(torch.zeros(1),
                                            requires_grad=False)
    elif mutation == 'base_meta_device':
        base['params'] = torch.zeros(1, device='meta')
    elif mutation == 'base_float64':
        base['params'] = torch.zeros(1, dtype=torch.float64)
    elif mutation == 'base_wrong_shape':
        base['params'] = torch.zeros(2)
    elif mutation == 'base_nonzero':
        base['params'] = torch.ones(1)
    elif mutation == 'base_requires_grad':
        base['params'] = torch.zeros(1, requires_grad=True)
    elif mutation == 'base_grad':
        base['params'].grad = torch.zeros_like(base['params'])
    elif mutation == 'base_inference':
        base['params'] = _d13n_inference_tensor(lambda: torch.zeros(1))
    elif mutation == 'base_fake_singleton_stride':
        base['params'] = torch.empty_strided((1, ), (7, ))
        base['params'].zero_()
    elif mutation == 'base_offset_view':
        base['params'] = torch.zeros(2)[1:]
    elif mutation == 'base_conjugate':
        base['params'] = torch.zeros(1, dtype=torch.complex64).conj()
    elif mutation == 'base_negative_view':
        base['params'] = torch.zeros(1)._neg_view()
    elif mutation == 'step_parameter_subclass':
        weight_slot['step'] = torch.nn.Parameter(
            torch.tensor(1920.0), requires_grad=False)
    elif mutation == 'step_meta_device':
        weight_slot['step'] = torch.empty((), device='meta')
    elif mutation == 'step_float64':
        weight_slot['step'] = torch.tensor(1920.0, dtype=torch.float64)
    elif mutation == 'step_wrong_shape':
        weight_slot['step'] = torch.tensor([1920.0])
    elif mutation == 'step_zero':
        weight_slot['step'] = torch.tensor(0.0)
    elif mutation == 'step_fractional':
        weight_slot['step'] = torch.tensor(1.5)
    elif mutation == 'step_nonfinite':
        weight_slot['step'] = torch.tensor(float('nan'))
    elif mutation == 'step_requires_grad':
        weight_slot['step'] = torch.tensor(1920.0, requires_grad=True)
    elif mutation == 'step_grad':
        weight_slot['step'].grad = torch.zeros_like(weight_slot['step'])
    elif mutation == 'step_inference':
        weight_slot['step'] = _d13n_inference_tensor(
            lambda: torch.tensor(1920.0))
    elif mutation == 'step_offset_view':
        weight_slot['step'] = torch.tensor([1.0, 1920.0])[1]
    elif mutation == 'moment_parameter_subclass':
        weight_slot['exp_avg'] = torch.nn.Parameter(
            torch.zeros(1, 256), requires_grad=False)
    elif mutation == 'moment_meta_device':
        weight_slot['exp_avg'] = torch.empty((1, 256), device='meta')
    elif mutation == 'moment_float64':
        weight_slot['exp_avg'] = torch.zeros(1, 256, dtype=torch.float64)
    elif mutation == 'moment_wrong_shape':
        weight_slot['exp_avg'] = torch.zeros(256)
    elif mutation == 'moment_nonfinite':
        weight_slot['exp_avg'] = torch.full((1, 256), float('inf'))
    elif mutation == 'second_moment_negative':
        weight_slot['exp_avg_sq'] = torch.full((1, 256), -0.001)
    elif mutation == 'moment_requires_grad':
        weight_slot['exp_avg'] = torch.zeros(
            1, 256, requires_grad=True)
    elif mutation == 'moment_grad':
        weight_slot['exp_avg'].grad = torch.zeros_like(
            weight_slot['exp_avg'])
    elif mutation == 'moment_inference':
        weight_slot['exp_avg'] = _d13n_inference_tensor(
            lambda: torch.zeros(1, 256))
    elif mutation == 'moment_fake_singleton_stride':
        weight_slot['exp_avg'] = torch.empty_strided(
            (1, 256), (1, 1))
        weight_slot['exp_avg'].zero_()
    elif mutation == 'moment_offset_view':
        weight_slot['exp_avg'] = torch.zeros(257)[1:].reshape(1, 256)
    elif mutation == 'moment_conjugate':
        weight_slot['exp_avg'] = torch.zeros(
            1, 256, dtype=torch.complex64).conj()
    elif mutation == 'moment_negative_view':
        weight_slot['exp_avg'] = torch.zeros(1, 256)._neg_view()
    elif mutation == 'moment_zero_pointer':
        weight_slot['exp_avg'] = torch.empty(1, 0)
    elif mutation == 'six_state_identity_alias':
        weight_slot['exp_avg_sq'] = weight_slot['exp_avg']
    elif mutation == 'six_state_storage_alias':
        original = weight_slot['exp_avg']
        weight_slot['exp_avg_sq'] = torch.empty(0).set_(
            original.storage(), 0, original.size(), original.stride())
    elif mutation == 'base_state_identity_alias':
        shared = torch.zeros(1)
        base['params'] = shared
        bias_slot['exp_avg'] = shared
    elif mutation == 'base_state_storage_alias':
        original = base['params']
        bias_slot['exp_avg'] = torch.empty(0).set_(
            original.storage(), 0, original.size(), original.stride())
    elif mutation == 'optimizer_model_identity_alias':
        weight_slot['exp_avg'] = checkpoint['state_dict'][
            D13N_ADAPTER_NAMES[0]]
    else:
        original = checkpoint['state_dict'][D13N_ADAPTER_NAMES[0]]
        weight_slot['exp_avg'] = torch.empty(0).set_(
            original.storage(), 0, original.size(), original.stride())

    monkeypatch.setattr(
        api.torch, 'load', lambda *args, **kwargs: checkpoint)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            source, stage0_report, 'candidate',
            api.CheckpointHashCache())


@pytest.mark.parametrize('mutation', [
    'stage0_schema', 'stage0_status', 'missing_model_evidence',
    'missing_parameter_names', 'missing_buffer_names',
    'parameter_names_tuple', 'buffer_names_list_subclass',
    'parameter_names_reversed', 'parameter_name_unknown',
    'parameter_buffer_overlap', 'stage0_parameter_hash',
    'stage0_buffer_hash', 'parent_parameter_changed',
    'parent_buffer_changed', 'missing_parent_parameter',
    'missing_parent_buffer', 'unexpected_model_state',
    'control_adapter_nonzero',
])
def test_audit_d13n_checkpoint_rejects_stage0_or_model_state_authority_drift(
        tmp_path, mutation):
    api = _load_runtime()
    role = 'control' if mutation == 'control_adapter_nonzero' else 'candidate'
    source, stage0_report, _ = _d13n_synthetic_checkpoint(
        tmp_path, role, initial_lr=True)
    checkpoint = torch.load(source, map_location='cpu')
    evidence = stage0_report['model_evidence']
    state_dict = checkpoint['state_dict']

    if mutation == 'stage0_schema':
        stage0_report['schema'] = 'd13n-stage0-v2'
    elif mutation == 'stage0_status':
        stage0_report['status'] = 'FAIL'
    elif mutation == 'missing_model_evidence':
        del stage0_report['model_evidence']
    elif mutation == 'missing_parameter_names':
        del evidence['parent_parameter_names']
    elif mutation == 'missing_buffer_names':
        del evidence['parent_buffer_names']
    elif mutation == 'parameter_names_tuple':
        evidence['parent_parameter_names'] = tuple(
            evidence['parent_parameter_names'])
    elif mutation == 'buffer_names_list_subclass':
        evidence['parent_buffer_names'] = _D13NListSubclass(
            evidence['parent_buffer_names'])
    elif mutation == 'parameter_names_reversed':
        evidence['parent_parameter_names'] = [
            *reversed(evidence['parent_parameter_names'])]
        evidence['parent_parameter_names'].append('backbone.running_mean')
    elif mutation == 'parameter_name_unknown':
        evidence['parent_parameter_names'][0] = 'backbone.unknown'
    elif mutation == 'parameter_buffer_overlap':
        evidence['parent_buffer_names'] = list(
            evidence['parent_parameter_names'])
    elif mutation == 'stage0_parameter_hash':
        evidence['parent_parameter_sha256'] = '0' * 64
    elif mutation == 'stage0_buffer_hash':
        evidence['parent_buffer_sha256'] = '0' * 64
    elif mutation == 'parent_parameter_changed':
        state_dict['backbone.weight'][0, 0] += 1
    elif mutation == 'parent_buffer_changed':
        state_dict['backbone.running_mean'][0] += 1
    elif mutation == 'missing_parent_parameter':
        del state_dict['backbone.weight']
    elif mutation == 'missing_parent_buffer':
        del state_dict['backbone.running_mean']
    elif mutation == 'unexpected_model_state':
        state_dict['backbone.unregistered'] = torch.zeros(1)
    else:
        state_dict[D13N_ADAPTER_NAMES[0]].fill_(0.125)

    if mutation in (
            'parent_parameter_changed', 'parent_buffer_changed',
            'missing_parent_parameter', 'missing_parent_buffer',
            'unexpected_model_state', 'control_adapter_nonzero'):
        parent_state = {
            name: value for name, value in state_dict.items()
            if name not in D13N_ADAPTER_NAMES
        }
        adapter_state = {
            name: state_dict[name] for name in D13N_ADAPTER_NAMES
            if name in state_dict
        }
        integrity = checkpoint['meta']['d13n_integrity']
        integrity['parent_state_sha256'] = _d13n_hash_named_tensor_state(
            parent_state)
        integrity['adapter_state_sha256'] = _d13n_hash_named_tensor_state(
            adapter_state)
        _d13n_refresh_integrity(checkpoint)

    mutated = _d13n_save_mutation(source, checkpoint, mutation)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        api.audit_d13n_checkpoint(
            mutated, stage0_report, role, api.CheckpointHashCache())
