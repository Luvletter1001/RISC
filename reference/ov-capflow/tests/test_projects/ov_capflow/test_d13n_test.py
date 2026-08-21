import argparse
import errno
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine import Config


SCRIPT_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
               'tools' / 'd13n_test.py')
ADAPTER_KEYS = [
    'bbox_head.existence_residual.weight',
    'bbox_head.existence_residual.bias',
]
IDENTITY_KEYS = {
    'schema', 'role', 'mouth', 'config_path', 'config_sha256',
    'resolved_config_sha256', 'derived_parent',
    'derived_deleted_fields', 'checkpoint_path', 'checkpoint_sha256',
    'predictions_path', 'predictions_sha256', 'official_metrics_path',
    'official_metrics_sha256', 'records', 'unique_image_ids',
    'queries_per_image', 'prediction_rows', 'finite', 'world_size',
    'local_ranks', 'cuda_visible_devices', 'nccl_p2p_disable',
    'nccl_ib_disable',
}
METRIC_KEYS = {
    'schema', 'role', 'mouth', 'step', 'dota/mAP', 'dota/AP50',
    'config_path', 'config_sha256', 'checkpoint_path',
    'checkpoint_sha256', 'records', 'queries_per_image',
    'prediction_rows', 'report_sha256',
}


def _load_module():
    spec = importlib.util.spec_from_file_location('d13n_test', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_config(role='control', mouth='proxy400'):
    return Config({
        'd13n_role': role,
        'd13n_mouth': mouth,
        'model': {
            'bbox_head': {'existence_loss_weight': 0.0},
            'freeze_except_patterns': [
                r'^bbox_head\.existence_residual\.(weight|bias)$'
            ],
        },
        'custom_hooks': [
            {'type': 'D13NParentEvalModeHook', 'role': 'control'}
        ],
        'train_cfg': {'type': 'EpochBasedTrainLoop'},
        'train_dataloader': {'batch_size': 2},
        'optim_wrapper': {'type': 'OptimWrapper'},
        'param_scheduler': [],
        'test_evaluator': {
            'type': 'DOTAMetric',
            'metric': 'mAP',
            '_scope_': 'mmrotate',
            'iou_thrs': 0.5,
        },
        'work_dir': 'work_dirs/d13n-test',
        'resume': False,
    })


def _raw_config(role='raw-control'):
    cfg = _base_config(role=role, mouth='raw13833')
    cfg.train_cfg = None
    cfg.train_dataloader = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None
    return cfg


def _parse(api, tmp_path, role='stage0-parent', derive=True):
    argv = [
        str(tmp_path / 'config.py'),
        str(tmp_path / 'epoch_24.pth'),
        '--launcher', 'pytorch', '--role', role,
        '--out', str(tmp_path / 'predictions.pkl'),
        '--metrics-out', str(tmp_path / 'metrics.json'),
        '--identity-out', str(tmp_path / 'identity.json'),
    ]
    if derive:
        argv.append('--derive-adapter-free-parent')
    return api.build_parser().parse_args(argv)


def _canonical_hash(payload, omitted='report_sha256'):
    value = dict(payload)
    value.pop(omitted, None)
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(',', ':'), sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def test_parser_freezes_required_cli_and_role_enum(tmp_path):
    api = _load_module()
    args = _parse(api, tmp_path)

    assert args.launcher == 'pytorch'
    assert args.role == 'stage0-parent'
    assert args.derive_adapter_free_parent is True
    assert args.out.name == 'predictions.pkl'

    with pytest.raises(SystemExit) as missing:
        api.build_parser().parse_args([
            'config.py', 'epoch_24.pth', '--launcher', 'pytorch',
            '--role', 'proxy-control',
        ])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as bad_role:
        api.build_parser().parse_args([
            'config.py', 'epoch_24.pth', '--launcher', 'pytorch',
            '--role', 'invented', '--out', 'a.pkl', '--metrics-out', 'b.json',
            '--identity-out', 'c.json',
        ])
    assert bad_role.value.code == 2


@pytest.mark.parametrize(
    'role,config_role,mouth,derive',
    [
        ('stage0-parent', 'control', 'proxy400', True),
        ('proxy-control', 'control', 'proxy400', False),
        ('proxy-candidate', 'candidate', 'proxy400', False),
        ('raw-control', 'raw-control', 'raw13833', False),
        ('raw-candidate', 'raw-candidate', 'raw13833', False),
    ],
)
def test_role_config_contract_accepts_only_the_frozen_pair(
        role, config_role, mouth, derive):
    api = _load_module()
    cfg = (_raw_config(config_role) if mouth == 'raw13833'
           else _base_config(config_role, mouth))

    assert api.validate_role_config(
        cfg, role, derive_adapter_free_parent=derive) == mouth


def test_role_config_contract_rejects_mismatch_raw_training_and_derive_flag():
    api = _load_module()

    with pytest.raises(api.IdentityError, match='d13n_role'):
        api.validate_role_config(
            _base_config('candidate'), 'proxy-control', False)
    with pytest.raises(api.IdentityError, match='d13n_mouth'):
        api.validate_role_config(
            _base_config('control', 'raw13833'), 'proxy-control', False)
    with pytest.raises(api.IdentityError, match='train_cfg'):
        api.validate_role_config(
            _base_config('raw-control', 'raw13833'), 'raw-control', False)
    with pytest.raises(api.IdentityError, match='derive'):
        api.validate_role_config(
            _base_config('control'), 'stage0-parent', False)
    with pytest.raises(api.IdentityError, match='derive'):
        api.validate_role_config(
            _base_config('control'), 'proxy-control', True)


def test_adapter_free_parent_derivation_deletes_exactly_three_fields():
    api = _load_module()
    cfg = _base_config()
    before = cfg.to_dict()

    derived, deleted = api.derive_adapter_free_parent(cfg)

    assert cfg.to_dict() == before
    assert deleted == [
        'model.bbox_head.existence_loss_weight',
        'model.freeze_except_patterns',
        'custom_hooks[0]',
    ]
    assert 'existence_loss_weight' not in derived.model.bbox_head
    assert 'freeze_except_patterns' not in derived.model
    assert derived.custom_hooks == []
    expected = before
    del expected['model']['bbox_head']['existence_loss_weight']
    del expected['model']['freeze_except_patterns']
    expected['custom_hooks'] = []
    assert derived.to_dict() == expected


@pytest.mark.parametrize(
    'hooks',
    [[], [
        {'type': 'D13NParentEvalModeHook'},
        {'type': 'D13NParentEvalModeHook'},
    ], [{'type': 'OtherHook'}]],
)
def test_adapter_free_parent_requires_one_exact_mode_hook(hooks):
    api = _load_module()
    cfg = _base_config()
    cfg.custom_hooks = hooks

    with pytest.raises(api.IdentityError, match='single.*D13NParentEvalModeHook'):
        api.derive_adapter_free_parent(cfg)


def test_configure_runner_keeps_official_metric_and_appends_atomic_dump(
        tmp_path):
    api = _load_module()
    cfg = _base_config()
    checkpoint = tmp_path / 'epoch_24.pth'
    output = tmp_path / 'predictions.pkl'

    configured = api.configure_test_config(cfg, checkpoint, output)

    assert configured.launcher == 'pytorch'
    assert configured.load_from == str(checkpoint.resolve())
    assert configured.resume is False
    assert configured.test_evaluator[0] == cfg.test_evaluator
    assert configured.test_evaluator[1] == {
        'type': 'D13NNoReplaceDumpResults',
        'out_file_path': str(output.resolve()),
        'collect_device': 'cpu',
    }


def test_output_preflight_rejects_alias_final_and_pending_paths(tmp_path):
    api = _load_module()
    first = tmp_path / 'one.pkl'
    second = tmp_path / 'two.json'
    third = tmp_path / 'three.json'

    with pytest.raises(api.OutputCollisionError, match='distinct'):
        api.resolve_and_preflight_outputs(first, first, third)

    second.write_bytes(b'owned')
    with pytest.raises(api.OutputCollisionError, match='two.json'):
        api.resolve_and_preflight_outputs(first, second, third)
    second.unlink()
    pending = tmp_path / 'one.pkl.pending.123.0'
    pending.write_bytes(b'evidence')
    with pytest.raises(api.OutputCollisionError, match='pending'):
        api.resolve_and_preflight_outputs(first, second, third)
    assert pending.read_bytes() == b'evidence'


def test_output_preflight_rejects_symlinked_parent_physical_alias(tmp_path):
    api = _load_module()
    physical = tmp_path / 'physical'
    physical.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(physical, target_is_directory=True)

    with pytest.raises(api.OutputCollisionError, match='physical alias'):
        api.resolve_and_preflight_outputs(
            physical / 'artifact.bin', alias / 'artifact.bin',
            tmp_path / 'identity.json')


def test_second_scan_detects_parent_alias_created_during_config_load(
        tmp_path, monkeypatch):
    api = _load_module()
    first_parent = tmp_path / 'first'
    second_parent = tmp_path / 'second'
    first_parent.mkdir()
    second_parent.mkdir()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'marker = "A"\n')
    checkpoint.write_bytes(b'checkpoint')
    args = api.build_parser().parse_args([
        str(config), str(checkpoint), '--launcher', 'pytorch',
        '--role', 'proxy-control', '--out', str(first_parent / 'same.bin'),
        '--metrics-out', str(second_parent / 'same.bin'), '--identity-out',
        str(tmp_path / 'identity.json'),
    ])
    built = []
    runtime = _success_runtime(api, _base_config(), built, [])

    def racing_loader(path):
        second_parent.rmdir()
        second_parent.symlink_to(first_parent, target_is_directory=True)
        return _base_config()

    runtime.load_config = racing_loader
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    with pytest.raises(api.OutputCollisionError, match='physical alias'):
        api.execute(args, runtime=runtime)
    assert built == []


def test_checkpoint_guard_is_strict_only_for_stage0_adapter_free_parent(
        tmp_path):
    api = _load_module()
    checkpoint = tmp_path / 'epoch_24.pth'
    checkpoint.write_bytes(b'checkpoint')

    class Model:
        def __init__(self, missing, unexpected=()):
            self.missing = missing
            self.unexpected = unexpected

        def load_state_dict(self, state, strict=False):
            assert state == {'parent': 1}
            assert strict is False
            return SimpleNamespace(
                missing_keys=list(self.missing),
                unexpected_keys=list(self.unexpected))

    loader = lambda path: {'state_dict': {'module.parent': 1}}
    api.validate_stage0_checkpoint_load(Model([]), checkpoint, loader)

    with pytest.raises(api.IdentityError, match='missing'):
        api.validate_stage0_checkpoint_load(Model(ADAPTER_KEYS), checkpoint,
                                            loader)
    with pytest.raises(api.IdentityError, match='unexpected'):
        api.validate_stage0_checkpoint_load(Model([], ['foreign']), checkpoint,
                                            loader)


def test_stage0_checkpoint_identity_requires_e24_and_config_bound_sha(tmp_path):
    api = _load_module()
    checkpoint = tmp_path / 'epoch_24.pth'
    checkpoint.write_bytes(b'authoritative-e24')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = _base_config()
    cfg.d13n_parent_sha256 = digest

    api.validate_stage0_checkpoint_identity(cfg, checkpoint, digest, step=24)

    with pytest.raises(api.IdentityError, match='Epoch 24'):
        api.validate_stage0_checkpoint_identity(cfg, checkpoint, digest, step=12)
    with pytest.raises(api.IdentityError, match='SHA256'):
        api.validate_stage0_checkpoint_identity(
            cfg, checkpoint, '0' * 64, step=24)


@pytest.mark.parametrize(
    'role,port,derive',
    [
        ('stage0-parent', 29845, True),
        ('proxy-control', 29846, False),
        ('proxy-candidate', 29847, False),
        ('raw-control', 29843, False),
        ('raw-candidate', 29844, False),
    ],
)
def test_build_torchrun_argv_has_exact_prefix_order_and_stage0_flag(
        role, port, derive):
    api = _load_module()
    argv = api.build_torchrun_argv(
        'config.py', 'epoch_12.pth', role, 'pred.pkl', 'metrics.json',
        'identity.json', port, derive_adapter_free_parent=derive)

    assert argv == [
        '/data/zcy/anaconda3/envs/mmdet/bin/python',
        '-m', 'torch.distributed.run', '--nproc_per_node=5',
        '--master_port={}'.format(port),
        'projects/OVCapFlow/tools/d13n_test.py',
        'config.py', 'epoch_12.pth', '--launcher', 'pytorch',
        '--role', role, '--out', 'pred.pkl', '--metrics-out', 'metrics.json',
        '--identity-out', 'identity.json',
    ] + (['--derive-adapter-free-parent'] if derive else [])

    if role != 'stage0-parent':
        with pytest.raises(api.IdentityError, match='derive'):
            api.build_torchrun_argv(
                'config.py', 'epoch_12.pth', role, 'pred.pkl', 'metrics.json',
                'identity.json', port, derive_adapter_free_parent=True)

    with pytest.raises(api.IdentityError, match='port'):
        api.build_torchrun_argv(
            'config.py', 'epoch_12.pth', role, 'pred.pkl', 'metrics.json',
            'identity.json', port + 100,
            derive_adapter_free_parent=derive)


@pytest.mark.parametrize(
    'role,port', [('proxy-control', 29842), ('proxy-candidate', 29841)])
def test_build_torchrun_argv_accepts_formal_proxy_replay_ports(role, port):
    api = _load_module()

    argv = api.build_torchrun_argv(
        'config.py', 'epoch_12.pth', role, 'pred.pkl', 'metrics.json',
        'identity.json', port)

    assert '--master_port={}'.format(port) in argv


def _success_runtime(api, cfg, built, barriers):
    class Model:
        def load_state_dict(self, state, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    class Runner:
        def __init__(self, runner_cfg):
            self.cfg = runner_cfg
            self.model = Model()
            self._has_loaded = False

        def test(self):
            Path(self.cfg.test_evaluator[-1]['out_file_path']).write_bytes(
                b'complete-prediction-stream')
            return {'dota/mAP': 0.612345678901, 'dota/AP50': 0.612}

    def build_runner(runner_cfg):
        built.append(runner_cfg)
        return Runner(runner_cfg)

    def gather(value):
        return _five_rank_envelopes(value)

    def load_config(path):
        loaded = Config(cfg.to_dict())
        checkpoint = Path(path).with_name('epoch_24.pth')
        if checkpoint.exists():
            loaded.d13n_parent_sha256 = hashlib.sha256(
                checkpoint.read_bytes()).hexdigest()
        return loaded

    return api.RuntimeFacade(
        load_config=load_config,
        build_runner=build_runner,
        checkpoint_loader=lambda path: {'state_dict': {'parent': 1}},
        load_records=lambda path: ['synthetic-records'],
        validate_records=lambda records, **kwargs: {
            'records': kwargs['expected_records'],
            'unique_image_ids': kwargs['expected_records'],
            'prediction_rows': (
                kwargs['expected_records'] * kwargs['queries_per_image']),
            'all_cpu_finite': True,
        },
        get_dist_info=lambda: (0, 5),
        all_gather_object=gather,
        barrier=lambda: barriers.append('barrier'),
        broadcast_object_list=lambda payload: None,
    )


def _five_rank_envelopes(envelope):
    if not isinstance(envelope, dict) or envelope.get('ok') is not True:
        return [envelope] * 5
    values = []
    for rank in range(5):
        value = dict(envelope['value'], rank=rank, local_rank=rank)
        values.append({'ok': True, 'value': value})
    return values


def test_complete_facade_run_publishes_metrics_then_identity_with_real_hashes(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'marker = "frozen"\n')
    checkpoint.write_bytes(b'frozen checkpoint bytes')
    args = _parse(api, tmp_path)
    built = []
    barriers = []
    cfg = _base_config()
    cfg.d13n_parent_sha256 = hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    runtime = _success_runtime(api, cfg, built, barriers)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.execute(args, runtime=runtime) == 0

    predictions = tmp_path / 'predictions.pkl'
    metrics_path = tmp_path / 'metrics.json'
    identity_path = tmp_path / 'identity.json'
    assert built and built[0].test_evaluator[0]['type'] == 'DOTAMetric'
    assert built[0].test_evaluator[1]['type'] == 'D13NNoReplaceDumpResults'
    assert built[0].custom_hooks == []
    assert barriers == ['barrier', 'barrier']
    assert metrics_path.read_bytes().endswith(b'\n')
    assert len(metrics_path.read_text().splitlines()) == 1
    assert identity_path.read_bytes().endswith(b'\n')

    metrics = json.loads(metrics_path.read_text())
    identity = json.loads(identity_path.read_text())
    assert set(metrics) == METRIC_KEYS
    assert set(identity) == IDENTITY_KEYS
    assert metrics['schema'] == 'd13n-official-metrics-v1'
    assert metrics['dota/mAP'] == 0.612345678901
    assert metrics['dota/AP50'] == 0.612
    assert metrics['step'] == 24
    assert metrics['report_sha256'] == _canonical_hash(metrics)
    assert identity['schema'] == 'd13n-test-identity-v1'
    assert identity['derived_parent'] is True
    assert identity['derived_deleted_fields'] == [
        'model.bbox_head.existence_loss_weight',
        'model.freeze_except_patterns',
        'custom_hooks[0]',
    ]
    assert identity['records'] == identity['unique_image_ids'] == 400
    assert identity['queries_per_image'] == 600
    assert identity['prediction_rows'] == 240000
    assert identity['finite'] is True
    assert identity['world_size'] == 5
    assert identity['local_ranks'] == [0, 1, 2, 3, 4]
    assert identity['cuda_visible_devices'] == '5,6,7,8,9'
    assert identity['nccl_p2p_disable'] == '1'
    assert identity['nccl_ib_disable'] == '1'
    assert identity['config_sha256'] == hashlib.sha256(
        config.read_bytes()).hexdigest()
    assert identity['checkpoint_sha256'] == hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    assert identity['predictions_sha256'] == hashlib.sha256(
        predictions.read_bytes()).hexdigest()
    assert identity['official_metrics_sha256'] == hashlib.sha256(
        metrics_path.read_bytes()).hexdigest()

    diagnostics_path = (Path('projects/OVCapFlow/tools') /
                        'dotav2_q600_diagnostics.py')
    spec = importlib.util.spec_from_file_location(
        'dotav2_q600_diagnostics_for_d13n', diagnostics_path)
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)
    recovered = diagnostics.read_metric_record(metrics_path, step=24)
    assert recovered.mAP == metrics['dota/mAP']
    assert recovered.ap50 == metrics['dota/AP50']
    assert recovered.step == 24


def test_config_aba_executes_exclusive_same_directory_hashed_snapshot(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config_a = b'role = "A"\n'
    config_b = b'role = "B"\n'
    config.write_bytes(config_a)
    checkpoint.write_bytes(b'checkpoint A')
    args = _parse(api, tmp_path, role='proxy-control', derive=False)
    executed = []
    snapshot_paths = []

    def load_config(path):
        snapshot = Path(path)
        snapshot_paths.append(snapshot)
        config.write_bytes(config_b)
        executed.append(snapshot.read_bytes())
        config.write_bytes(config_a)
        return _base_config()

    runtime = _success_runtime(api, _base_config(), [], [])
    runtime.load_config = load_config
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.execute(args, runtime=runtime) == 0
    identity = json.loads((tmp_path / 'identity.json').read_text())
    assert executed == [config_a]
    assert snapshot_paths[0] != config
    assert str(snapshot_paths[0]).startswith('/proc/self/fd/')
    assert snapshot_paths[0].name.startswith(
        '.config.d13n-snapshot-')
    assert snapshot_paths[0].suffix == '.py'
    assert not snapshot_paths[0].exists()
    assert identity['config_sha256'] == hashlib.sha256(config_a).hexdigest()


def test_real_relative_base_config_loads_from_same_directory_snapshot():
    api = _load_module()
    config = Path(
        'configs/ov_capflow/dotav2/'
        'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
        'scale1024_batch2_rare4x_world5_d13n_control.py').resolve()
    pattern = '.' + config.stem + '.d13n-snapshot-*'
    before = set(config.parent.glob(pattern))

    loaded, digest = api.load_hashed_config_snapshot(
        config, lambda path: Config.fromfile(path))

    assert loaded.d13n_role == 'control'
    assert loaded.d13n_mouth == 'proxy400'
    assert digest == hashlib.sha256(config.read_bytes()).hexdigest()
    assert set(config.parent.glob(pattern)) == before


def test_flat_snapshot_rewrites_only_local_base_literal_and_keeps_scope(
        tmp_path):
    api = _load_module()
    config = tmp_path / 'top.py'
    base = tmp_path / 'base.py'
    base.write_text('local_marker = "local"\n')
    config.write_bytes(
        b'_base_ = [\n'
        b'    "./base.py",  # local entry\n'
        b"    'mmrotate::_base_/default_runtime.py',  # scoped entry\n"
        b']\n'
        b'top_marker = "top"\n')
    observed = []

    def inspect_then_load(path):
        observed.append(Path(path).read_bytes())
        snapshots = list(tmp_path.glob('.top.d13n-snapshot-*.py'))
        assert len(snapshots) == 2
        assert all(stat.S_IMODE(item.stat().st_mode) == 0o400
                   for item in snapshots)
        return Config.fromfile(path)

    loaded, _ = api.load_hashed_config_snapshot(config, inspect_then_load)

    assert loaded.local_marker == 'local'
    assert loaded.top_marker == 'top'
    assert b'"./base.py"' not in observed[0]
    assert b"'mmrotate::_base_/default_runtime.py'" in observed[0]
    assert b'# local entry' in observed[0]
    assert b'# scoped entry' in observed[0]
    assert list(tmp_path.glob('.top.d13n-snapshot-*.py')) == []


def test_config_parent_symlink_retarget_never_hybridizes_top_and_base(
        tmp_path):
    api = _load_module()
    directory_a = tmp_path / 'config-a'
    directory_b = tmp_path / 'config-b'
    directory_a.mkdir()
    directory_b.mkdir()
    (directory_a / 'base.py').write_text('base_marker = "A"\n')
    (directory_b / 'base.py').write_text('base_marker = "B"\n')
    top_a = b'_base_ = "./base.py"\ntop_marker = "A"\n'
    top_b = b'_base_ = "./base.py"\ntop_marker = "B"\n'
    (directory_a / 'top.py').write_bytes(top_a)
    (directory_b / 'top.py').write_bytes(top_b)
    alias = tmp_path / 'active-config'
    alias.symlink_to(directory_a, target_is_directory=True)

    def retarget_then_load(path):
        alias.unlink()
        alias.symlink_to(directory_b, target_is_directory=True)
        return Config.fromfile(path)

    loaded, digest = api.load_hashed_config_snapshot(
        alias / 'top.py', retarget_then_load)

    assert loaded.top_marker == 'A'
    assert loaded.base_marker == 'A'
    assert digest == hashlib.sha256(top_a).hexdigest()
    assert list(directory_a.glob('.top.d13n-snapshot-*')) == []
    assert list(directory_b.glob('.top.d13n-snapshot-*')) == []


def test_config_parent_path_replacement_cannot_substitute_same_snapshot_name(
        tmp_path):
    api = _load_module()
    config_dir = tmp_path / 'active-config'
    active_child = config_dir / 'nested'
    relocated = config_dir / 'relocated-nested-a'
    active_child.mkdir(parents=True)
    top_a = b'_base_ = "../middle.py"\ntop_marker = "A"\n'
    middle_a = b'_base_ = "./base.py"\nmiddle_marker = "A"\n'
    (active_child / 'top.py').write_bytes(top_a)
    (config_dir / 'middle.py').write_bytes(middle_a)
    (config_dir / 'base.py').write_text('base_marker = "A"\n')
    def replace_parent_and_snapshot(path):
        active_child.rename(relocated)
        active_child.mkdir()
        (active_child / 'top.py').write_text(
            '_base_ = "../middle.py"\ntop_marker = "B"\n')
        (config_dir / 'middle.py').write_text(
            '_base_ = "./base.py"\nmiddle_marker = "B"\n')
        (config_dir / 'base.py').write_text('base_marker = "B"\n')
        snapshots = list(relocated.glob('.top.d13n-snapshot-*.py'))
        assert snapshots
        replacements = []
        for snapshot in snapshots:
            replacement = active_child / snapshot.name
            replacement.write_bytes(
                snapshot.read_bytes().replace(b'"A"', b'"B"'))
            replacements.append(replacement)
        try:
            return Config.fromfile(path)
        finally:
            for replacement in replacements:
                replacement.unlink()

    loaded, digest = api.load_hashed_config_snapshot(
        active_child / 'top.py', replace_parent_and_snapshot)

    assert loaded.top_marker == 'A'
    assert loaded.middle_marker == 'A'
    assert loaded.base_marker == 'A'
    assert digest == hashlib.sha256(top_a).hexdigest()
    assert list(relocated.glob('.top.d13n-snapshot-*')) == []
    assert list(active_child.glob('.top.d13n-snapshot-*')) == []


def test_config_base_closure_reads_from_pinned_parent_before_replacement(
        tmp_path, monkeypatch):
    api = _load_module()
    active = tmp_path / 'active-config'
    relocated = tmp_path / 'relocated-config-a'
    active.mkdir()
    top_a = b'_base_ = "./base.py"\ntop_marker = "A"\n'
    (active / 'top.py').write_bytes(top_a)
    (active / 'base.py').write_text('base_marker = "A"\n')
    real_read = api._read_config_bytes
    calls = []

    def replace_after_entry_read(parent_descriptor, name, display_path):
        payload = real_read(parent_descriptor, name, display_path)
        calls.append((name, payload))
        if len(calls) == 1:
            active.rename(relocated)
            active.mkdir()
            (active / 'top.py').write_text(
                '_base_ = "./base.py"\ntop_marker = "B"\n')
            (active / 'base.py').write_text('base_marker = "B"\n')
        return payload

    monkeypatch.setattr(api, '_read_config_bytes', replace_after_entry_read)

    loaded, digest = api.load_hashed_config_snapshot(
        active / 'top.py', Config.fromfile)

    assert loaded.top_marker == 'A'
    assert loaded.base_marker == 'A'
    assert digest == hashlib.sha256(top_a).hexdigest()
    assert [name for name, _ in calls] == ['top.py', 'base.py']
    assert list(relocated.glob('.top.d13n-snapshot-*')) == []
    assert list(active.glob('.top.d13n-snapshot-*')) == []


def test_config_parent_move_cannot_rebind_dotdot_base_chain(
        tmp_path, monkeypatch):
    api = _load_module()
    parent_a = tmp_path / 'parent-a'
    parent_b = tmp_path / 'parent-b'
    child_a = parent_a / 'child'
    moved_child = parent_b / 'child'
    child_a.mkdir(parents=True)
    parent_b.mkdir()
    top_a = b'_base_ = "../middle.py"\ntop_marker = "A"\n'
    (child_a / 'top.py').write_bytes(top_a)
    (parent_a / 'middle.py').write_text(
        '_base_ = "./base.py"\nmiddle_marker = "A"\n')
    (parent_a / 'base.py').write_text('base_marker = "A"\n')
    (parent_b / 'middle.py').write_text(
        '_base_ = "./base.py"\nmiddle_marker = "B"\n')
    (parent_b / 'base.py').write_text('base_marker = "B"\n')
    real_read = api._read_config_bytes
    calls = []

    def move_after_entry_read(parent_descriptor, name, display_path):
        payload = real_read(parent_descriptor, name, display_path)
        calls.append((name, payload))
        if len(calls) == 1:
            child_a.rename(moved_child)
            child_a.mkdir()
            (child_a / 'top.py').write_text(
                '_base_ = "../middle.py"\ntop_marker = "replacement"\n')
        return payload

    monkeypatch.setattr(api, '_read_config_bytes', move_after_entry_read)

    loaded, digest = api.load_hashed_config_snapshot(
        child_a / 'top.py', Config.fromfile)

    assert loaded.top_marker == 'A'
    assert loaded.middle_marker == 'A'
    assert loaded.base_marker == 'A'
    assert digest == hashlib.sha256(top_a).hexdigest()
    assert [name for name, _ in calls] == ['top.py', 'middle.py', 'base.py']
    assert list(moved_child.glob('.top.d13n-snapshot-*')) == []
    assert list(child_a.glob('.top.d13n-snapshot-*')) == []


def test_config_first_snapshot_fstat_failure_recovers_from_open_descriptor(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'top.py'
    config.write_text('value = 1\n')
    real_fstat = api.os.fstat
    calls = []

    def fail_first_fstat(descriptor):
        calls.append(descriptor)
        if len(calls) == 1:
            raise OSError(errno.EIO, 'first fstat failed')
        return real_fstat(descriptor)

    monkeypatch.setattr(api.os, 'fstat', fail_first_fstat)

    loaded, digest = api.load_hashed_config_snapshot(
        config, Config.fromfile)

    assert loaded.value == 1
    assert digest == hashlib.sha256(config.read_bytes()).hexdigest()
    assert calls
    assert list(tmp_path.glob('.top.d13n-snapshot-*')) == []


def test_config_midway_flat_create_failure_cleans_prior_owned_files(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'top.py'
    config.write_text('_base_ = "./base.py"\ntop = 1\n')
    (tmp_path / 'base.py').write_text('base = 1\n')
    real_open = api.os.open
    calls = []

    def fail_second_flat_snapshot_create(path, flags, *args, **kwargs):
        if (type(path) is str and path.startswith('.top.d13n-snapshot-')
                and path.endswith('.py') and flags & api.os.O_EXCL
                and kwargs.get('dir_fd') is not None):
            calls.append(path)
            if len(calls) == 2:
                raise OSError(errno.EMFILE, 'second snapshot create failed')
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(api.os, 'open', fail_second_flat_snapshot_create)

    with pytest.raises(api.IdentityError, match='config snapshot'):
        api.load_hashed_config_snapshot(config, Config.fromfile)

    assert len(calls) == 2
    assert list(tmp_path.glob('.top.d13n-snapshot-*')) == []


def test_config_flat_writer_failure_cleans_every_owned_file(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'top.py'
    config.write_text('_base_ = "./base.py"\ntop = 1\n')
    (tmp_path / 'base.py').write_text('base = 1\n')
    real_write = api._write_all
    calls = []

    def fail_second_snapshot_write(descriptor, payload):
        calls.append(descriptor)
        if len(calls) == 2:
            raise OSError(errno.EIO, 'second snapshot write failed')
        return real_write(descriptor, payload)

    monkeypatch.setattr(api, '_write_all', fail_second_snapshot_write)

    with pytest.raises(api.IdentityError, match='config snapshot'):
        api.load_hashed_config_snapshot(config, Config.fromfile)

    assert len(calls) == 2
    assert list(tmp_path.glob('.top.d13n-snapshot-*')) == []


def test_config_loader_failure_cleans_every_owned_flat_file(tmp_path):
    api = _load_module()
    config = tmp_path / 'top.py'
    config.write_text('_base_ = "./base.py"\ntop = 1\n')
    (tmp_path / 'base.py').write_text('base = 1\n')

    def fail_loader(path):
        assert list(tmp_path.glob('.top.d13n-snapshot-*.py'))
        raise RuntimeError('loader failed')

    with pytest.raises(api.IdentityError, match='failed to load'):
        api.load_hashed_config_snapshot(config, fail_loader)

    assert list(tmp_path.glob('.top.d13n-snapshot-*')) == []


@pytest.mark.parametrize('external', [b'', b'external replacement'])
def test_config_cleanup_never_deletes_flat_external_replacement(
        tmp_path, external):
    api = _load_module()
    config = tmp_path / 'top.py'
    config.write_text('_base_ = "./base.py"\ntop = 1\n')
    (tmp_path / 'base.py').write_text('base = 1\n')
    replacement = {'path': None}

    def replace_entry_then_fail(path):
        owned = tmp_path / Path(path).name
        assert owned.is_file()
        owned.unlink()
        owned.write_bytes(external)
        replacement['path'] = owned
        raise RuntimeError('loader failed after replacement')

    with pytest.raises(api.IdentityError, match='identity changed'):
        api.load_hashed_config_snapshot(config, replace_entry_then_fail)

    assert replacement['path'].read_bytes() == external
    assert list(tmp_path.glob('.top.d13n-snapshot-*.py')) == [
        replacement['path']]


def test_config_unknown_snapshot_identity_never_deletes_replacement(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'top.py'
    config.write_text('value = 1\n')
    real_fstat = api.os.fstat
    replacement = {'path': None}
    calls = []

    def replace_then_fail_first_fstat(descriptor):
        calls.append(descriptor)
        if len(calls) == 1:
            owned = next(tmp_path.glob('.top.d13n-snapshot-*.py'))
            owned.unlink()
            owned.write_bytes(b'external = "replacement"\n')
            replacement['path'] = owned
            raise OSError(errno.EIO, 'first fstat failed after replacement')
        return real_fstat(descriptor)

    monkeypatch.setattr(api.os, 'fstat', replace_then_fail_first_fstat)

    with pytest.raises(api.IdentityError, match='identity changed'):
        api.load_hashed_config_snapshot(config, Config.fromfile)

    assert replacement['path'].read_bytes() == \
        b'external = "replacement"\n'


def test_config_flat_snapshot_success_does_not_leak_descriptors(tmp_path):
    api = _load_module()
    config = tmp_path / 'top.py'
    config.write_text('_base_ = "./base.py"\ntop = 1\n')
    (tmp_path / 'base.py').write_text('base = 1\n')
    before = len(list(Path('/proc/self/fd').iterdir()))

    for _ in range(4):
        loaded, _ = api.load_hashed_config_snapshot(
            config, Config.fromfile)
        assert loaded.top == loaded.base == 1

    assert len(list(Path('/proc/self/fd').iterdir())) == before
    assert list(tmp_path.glob('.top.d13n-snapshot-*')) == []


def test_readonly_config_parent_fails_closed_as_identity(tmp_path):
    api = _load_module()
    config_dir = tmp_path / 'readonly-config'
    config_dir.mkdir()
    config = config_dir / 'top.py'
    config.write_text('value = 1\n')
    config_dir.chmod(0o500)
    try:
        with pytest.raises(api.IdentityError, match='snapshot'):
            api.load_hashed_config_snapshot(config, Config.fromfile)
    finally:
        config_dir.chmod(0o700)


def test_checkpoint_aba_runner_reads_bound_open_inode_not_mutable_path(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'marker = "A"\n')
    checkpoint_a = b'checkpoint A'
    checkpoint_b = b'checkpoint B'
    checkpoint.write_bytes(checkpoint_a)
    args = _parse(api, tmp_path, role='proxy-control', derive=False)
    executed = []
    load_paths = []

    class Runner:
        model = object()

        def __init__(self, cfg):
            self._load_from = cfg.load_from
            self.cfg = cfg

        def test(self):
            replacement_b = tmp_path / 'checkpoint-b.tmp'
            replacement_b.write_bytes(checkpoint_b)
            replacement_b.replace(checkpoint)
            load_paths.append(Path(self._load_from))
            executed.append(Path(self._load_from).read_bytes())
            replacement_a = tmp_path / 'checkpoint-a.tmp'
            replacement_a.write_bytes(checkpoint_a)
            replacement_a.replace(checkpoint)
            Path(self.cfg.test_evaluator[-1]['out_file_path']).write_bytes(
                b'prediction stream')
            return {'dota/mAP': 0.5, 'dota/AP50': 0.5}

    runtime = _success_runtime(api, _base_config(), [], [])
    runtime.build_runner = lambda cfg: Runner(cfg)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.execute(args, runtime=runtime) == 0
    identity = json.loads((tmp_path / 'identity.json').read_text())
    assert executed == [checkpoint_a]
    assert str(load_paths[0]).startswith('/proc/self/fd/')
    assert identity['checkpoint_sha256'] == hashlib.sha256(
        checkpoint_a).hexdigest()


def test_proc_fd_checkpoint_binding_is_readable_by_real_torch_load(tmp_path):
    api = _load_module()
    from mmengine.runner.checkpoint import _load_checkpoint

    checkpoint = tmp_path / 'epoch_12.pth'
    torch.save({'state_dict': {'weight': torch.tensor([3.0])}}, checkpoint)

    with api.open_checkpoint_binding(checkpoint) as binding:
        assert api._sha256_open_descriptor(binding.descriptor) == binding.sha256
        assert api._sha256_open_descriptor(binding.descriptor) == binding.sha256
        assert api.os.lseek(binding.descriptor, 0, api.os.SEEK_CUR) == 0
        restored = torch.load(str(binding.runtime_path), map_location='cpu')
        assert torch.equal(
            restored['state_dict']['weight'], torch.tensor([3.0]))
        restored_mmengine = _load_checkpoint(
            str(binding.runtime_path), map_location='cpu')
        assert torch.equal(
            restored_mmengine['state_dict']['weight'], torch.tensor([3.0]))
        assert binding.sha256 == hashlib.sha256(
            checkpoint.read_bytes()).hexdigest()

    assert not Path(str(binding.runtime_path)).exists()


def test_immutable_checkpoint_snapshot_survives_same_inode_source_mutation(
        tmp_path):
    api = _load_module()
    from mmengine.runner.checkpoint import _load_checkpoint

    checkpoint = tmp_path / 'epoch_12.pth'
    torch.save({'state_dict': {'weight': torch.tensor([3.0])}}, checkpoint)
    initial_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with api.open_checkpoint_binding(checkpoint) as binding:
        snapshot = api._create_rank0_checkpoint_snapshot(binding)
        try:
            assert snapshot.sha256 == initial_sha
            assert stat.S_IMODE(os.fstat(snapshot.descriptor).st_mode) == 0o400
            assert str(snapshot.runtime_path).startswith('/proc/self/fd/')
            assert list(tmp_path.glob(
                '.epoch_12.d13n-checkpoint-snapshot-*')) == []

            torch.save(
                {'state_dict': {'weight': torch.tensor([9.0])}}, checkpoint)
            restored = _load_checkpoint(
                str(snapshot.runtime_path), map_location='cpu')
            assert torch.equal(
                restored['state_dict']['weight'], torch.tensor([3.0]))

            torch.save(
                {'state_dict': {'weight': torch.tensor([3.0])}}, checkpoint)
            restored_again = _load_checkpoint(
                str(snapshot.runtime_path), map_location='cpu')
            assert torch.equal(
                restored_again['state_dict']['weight'], torch.tensor([3.0]))
        finally:
            snapshot.close()

    assert not Path(str(snapshot.runtime_path)).exists()


def test_checkpoint_snapshot_full_copy_fallback_runs_once_and_leaves_no_name(
        tmp_path, monkeypatch):
    api = _load_module()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint_a = b'A' * 4096
    checkpoint.write_bytes(checkpoint_a)
    copy_calls = []
    real_copy = api._copy_checkpoint_bytes

    def unsupported_clone(*args):
        raise OSError(errno.EOPNOTSUPP, 'clone unsupported')

    def counted_copy(source, destination, size):
        copy_calls.append((source, destination, size))
        return real_copy(source, destination, size)

    monkeypatch.setattr(api.fcntl, 'ioctl', unsupported_clone)
    monkeypatch.setattr(api, '_copy_checkpoint_bytes', counted_copy)
    with api.open_checkpoint_binding(checkpoint) as binding:
        snapshot = api._create_rank0_checkpoint_snapshot(binding)
        try:
            checkpoint.write_bytes(b'B' * len(checkpoint_a))
            assert snapshot.runtime_path.read_bytes() == checkpoint_a
        finally:
            snapshot.close()

    assert len(copy_calls) == 1
    assert copy_calls[0][2] == len(checkpoint_a)
    assert list(tmp_path.glob(
        '.epoch_12.d13n-checkpoint-snapshot-*')) == []


def test_checkpoint_snapshot_full_copy_source_aba_fails_and_cleans_name(
        tmp_path, monkeypatch):
    api = _load_module()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint_a = b'A' * 4096
    checkpoint_b = b'B' * 4096
    checkpoint.write_bytes(checkpoint_a)
    real_copy = api._copy_checkpoint_bytes

    def unsupported_clone(*args):
        raise OSError(errno.EOPNOTSUPP, 'clone unsupported')

    def mutate_then_copy(source, destination, size):
        checkpoint.write_bytes(checkpoint_b)
        return real_copy(source, destination, size)

    monkeypatch.setattr(api.fcntl, 'ioctl', unsupported_clone)
    monkeypatch.setattr(api, '_copy_checkpoint_bytes', mutate_then_copy)
    with api.open_checkpoint_binding(checkpoint) as binding:
        with pytest.raises(api.IdentityError, match='SHA256 mismatch'):
            api._create_rank0_checkpoint_snapshot(binding)

    assert checkpoint.read_bytes() == checkpoint_b
    assert list(tmp_path.glob(
        '.epoch_12.d13n-checkpoint-snapshot-*')) == []


def test_checkpoint_first_snapshot_fstat_failure_cleans_owned_name(
        tmp_path, monkeypatch):
    api = _load_module()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'checkpoint')
    real_fstat = api.os.fstat
    calls = []

    def fail_first_fstat(descriptor):
        calls.append(descriptor)
        if len(calls) == 1:
            raise OSError(errno.EIO, 'first fstat failed')
        return real_fstat(descriptor)

    monkeypatch.setattr(api.os, 'fstat', fail_first_fstat)
    with api.open_checkpoint_binding(checkpoint) as binding:
        with pytest.raises(api.IdentityError, match='checkpoint snapshot'):
            api._create_rank0_checkpoint_snapshot(binding)

    assert calls
    assert list(tmp_path.glob(
        '.epoch_12.d13n-checkpoint-snapshot-*')) == []


def test_checkpoint_unknown_snapshot_identity_never_deletes_replacement(
        tmp_path, monkeypatch):
    api = _load_module()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'checkpoint')
    real_fstat = api.os.fstat
    replacement = {'path': None}
    calls = []

    def replace_then_fail_first_fstat(descriptor):
        calls.append(descriptor)
        if len(calls) == 1:
            owned = next(tmp_path.glob(
                '.epoch_12.d13n-checkpoint-snapshot-*'))
            owned.unlink()
            owned.write_bytes(b'external replacement')
            replacement['path'] = owned
            raise OSError(errno.EIO, 'first fstat failed after replacement')
        return real_fstat(descriptor)

    monkeypatch.setattr(api.os, 'fstat', replace_then_fail_first_fstat)
    with api.open_checkpoint_binding(checkpoint) as binding:
        with pytest.raises(api.IdentityError, match='identity changed'):
            api._create_rank0_checkpoint_snapshot(binding)

    assert replacement['path'].read_bytes() == b'external replacement'


def test_checkpoint_snapshot_creation_error_broadcasts_same_identity_failure(
        tmp_path, monkeypatch):
    api = _load_module()
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'tiny checkpoint')
    broadcasts = []

    def fail_create(binding):
        raise OSError('snapshot unavailable')

    monkeypatch.setattr(api, '_create_rank0_checkpoint_snapshot', fail_create)
    rank0_runtime = api.RuntimeFacade(
        get_dist_info=lambda: (0, 5),
        broadcast_object_list=lambda payload: broadcasts.append(payload[0]))
    with api.open_checkpoint_binding(checkpoint) as binding:
        with pytest.raises(api.IdentityError) as rank0_error:
            api.establish_checkpoint_snapshot(rank0_runtime, 0, binding)

        def replay(payload):
            payload[0] = broadcasts[0]

        peer_runtime = api.RuntimeFacade(
            get_dist_info=lambda: (1, 5), broadcast_object_list=replay)
        with pytest.raises(api.IdentityError) as peer_error:
            api.establish_checkpoint_snapshot(peer_runtime, 1, binding)

    assert str(rank0_error.value) == str(peer_error.value)
    assert 'snapshot unavailable' in str(rank0_error.value)


@pytest.mark.parametrize(
    'name,role,derive',
    [
        (
            'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
            'scale1024_batch2_rare4x_world5_d13n_control.py',
            'proxy-control', False,
        ),
        (
            'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
            'scale1024_batch2_rare4x_world5_d13n_candidate.py',
            'proxy-candidate', False,
        ),
        (
            'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
            'scale1024_batch2_rare4x_world5_d13n_control_raw13833.py',
            'raw-control', False,
        ),
        (
            'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
            'scale1024_batch2_rare4x_world5_d13n_candidate_raw13833.py',
            'raw-candidate', False,
        ),
    ],
)
def test_real_configs_validate_and_keep_official_plus_atomic_metrics(
        name, role, derive, tmp_path):
    api = _load_module()
    path = (Path('configs/ov_capflow/dotav2') / name).resolve()
    native = Config.fromfile(str(path))
    cfg, digest = api.load_hashed_config_snapshot(path, Config.fromfile)

    mouth = api.validate_role_config(cfg, role, derive)
    configured = api.configure_test_config(
        cfg, tmp_path / 'epoch_12.pth', tmp_path / 'predictions.pkl')

    assert mouth == ('raw13833' if role.startswith('raw-') else 'proxy400')
    assert [item['type'] for item in configured.test_evaluator] == [
        'DOTAMetric', 'D13NNoReplaceDumpResults']
    assert cfg.to_dict() == native.to_dict()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(api._resolved_config_sha256(configured)) == 64


def test_nonmain_rank_never_publishes_any_output(tmp_path, monkeypatch):
    api = _load_module()
    (tmp_path / 'config.py').write_bytes(b'config')
    checkpoint = tmp_path / 'epoch_24.pth'
    checkpoint.write_bytes(b'checkpoint')
    args = _parse(api, tmp_path, role='proxy-control', derive=False)
    barriers = []

    class Runner:
        model = object()

        def test(self):
            return {'dota/mAP': 0.5, 'dota/AP50': 0.5}

    with api.open_checkpoint_binding(checkpoint) as binding:
        snapshot_owner = api._create_rank0_checkpoint_snapshot(binding)
    try:
        broadcasts = iter([
            {
                'schema': 'd13n-checkpoint-snapshot-v1',
                'status': 'ok',
                'authority': snapshot_owner.authority,
            },
            {
                'ok': True,
                'value': {
                    'summary': {
                        'records': 400,
                        'unique_image_ids': 400,
                        'prediction_rows': 240000,
                        'all_cpu_finite': True,
                    },
                    'metrics': {},
                },
            },
            {'ok': True, 'value': {}},
        ])

        def broadcast(payload):
            payload[0] = next(broadcasts)

        def gather(value):
            return _five_rank_envelopes(value)

        runtime = api.RuntimeFacade(
            load_config=lambda path: _base_config(),
            build_runner=lambda cfg: Runner(),
            get_dist_info=lambda: (1, 5),
            all_gather_object=gather,
            barrier=lambda: barriers.append('barrier'),
            broadcast_object_list=broadcast,
        )
        monkeypatch.setenv('WORLD_SIZE', '5')
        monkeypatch.setenv('LOCAL_RANK', '1')
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
        monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
        monkeypatch.setenv('NCCL_IB_DISABLE', '1')

        assert api.execute(args, runtime=runtime) == 0
    finally:
        snapshot_owner.close()
    assert barriers == ['barrier', 'barrier']
    assert not (tmp_path / 'predictions.pkl').exists()
    assert not (tmp_path / 'metrics.json').exists()
    assert not (tmp_path / 'identity.json').exists()


def test_racing_metrics_collision_is_code3_and_preserves_bytes(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    args = _parse(api, tmp_path, role='proxy-control', derive=False)
    runtime = _success_runtime(api, _base_config(), [], [])
    original_test = runtime.build_runner

    def build(cfg):
        runner = original_test(cfg)
        test = runner.test

        def collide_after_preflight():
            metrics = test()
            (tmp_path / 'metrics.json').write_bytes(b'racing-owner')
            return metrics

        runner.test = collide_after_preflight
        return runner

    runtime.build_runner = build
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    with pytest.raises(api.OutputCollisionError):
        api.execute(args, runtime=runtime)
    assert (tmp_path / 'metrics.json').read_bytes() == b'racing-owner'
    assert not (tmp_path / 'identity.json').exists()


def test_late_real_dump_collision_maps_to3_and_preserves_competing_bytes(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    output = tmp_path / 'predictions.pkl'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch', '--role',
        'proxy-control', '--out', str(output), '--metrics-out',
        str(tmp_path / 'metrics.json'), '--identity-out',
        str(tmp_path / 'identity.json'),
    ]
    runtime = _success_runtime(api, _base_config(), [], [])

    class Runner:
        model = object()

        def __init__(self, cfg):
            self._load_from = cfg.load_from

        def test(self):
            output.write_bytes(b'competing-complete-dump')
            raise api.D13NDumpCollisionError(
                out_file_path=str(output),
                original_errno=None,
                original_filename=None,
                original_filename2=None,
                original_message='output collision: {}'.format(output))

    runtime.build_runner = lambda cfg: Runner(cfg)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.main(argv, runtime=runtime) == 3
    assert output.read_bytes() == b'competing-complete-dump'
    assert not (tmp_path / 'metrics.json').exists()
    assert not (tmp_path / 'identity.json').exists()


def test_arbitrary_late_file_exists_is_runtime5_not_collision3(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch', '--role',
        'proxy-control', '--out', str(tmp_path / 'predictions.pkl'),
        '--metrics-out', str(tmp_path / 'metrics.json'), '--identity-out',
        str(tmp_path / 'identity.json'),
    ]
    runtime = _success_runtime(api, _base_config(), [], [])

    class Runner:
        model = object()

        def __init__(self, cfg):
            self._load_from = cfg.load_from

        def test(self):
            raise FileExistsError('unrelated model cache exists')

    runtime.build_runner = lambda cfg: Runner(cfg)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.main(argv, runtime=runtime) == 5


@pytest.mark.parametrize('phase', ['metrics', 'identity'])
def test_publication_file_exists_without_target_collision_is_runtime5(
        tmp_path, monkeypatch, phase):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    metrics_out = tmp_path / 'metrics.json'
    identity_out = tmp_path / 'identity.json'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch', '--role',
        'proxy-control', '--out', str(tmp_path / 'predictions.pkl'),
        '--metrics-out', str(metrics_out), '--identity-out', str(identity_out),
    ]
    runtime = _success_runtime(api, _base_config(), [], [])
    real_publish = api.publish_json_noreplace

    def fail_unrelated(path, payload):
        failed_path = metrics_out if phase == 'metrics' else identity_out
        if Path(path) == failed_path:
            raise FileExistsError('unrelated publication cache exists')
        return real_publish(path, payload)

    monkeypatch.setattr(api, 'publish_json_noreplace', fail_unrelated)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.main(argv, runtime=runtime) == 5
    failed_path = metrics_out if phase == 'metrics' else identity_out
    assert not failed_path.exists()


def test_structured_dump_collision_for_different_target_is_runtime5(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    output = tmp_path / 'predictions.pkl'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch', '--role',
        'proxy-control', '--out', str(output), '--metrics-out',
        str(tmp_path / 'metrics.json'), '--identity-out',
        str(tmp_path / 'identity.json'),
    ]
    runtime = _success_runtime(api, _base_config(), [], [])

    class Runner:
        model = object()

        def __init__(self, cfg):
            self._load_from = cfg.load_from

        def test(self):
            raise api.D13NDumpCollisionError(
                out_file_path=str(tmp_path / 'other.pkl'),
                original_errno=None,
                original_filename=None,
                original_filename2=None,
                original_message='output collision: other')

    runtime.build_runner = lambda cfg: Runner(cfg)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.main(argv, runtime=runtime) == 5


def test_native_hardlink_eexist_for_exact_dump_maps_to3(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    output = tmp_path / 'predictions.pkl'
    pending = Path(str(output) + '.pending.42.0')
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch', '--role',
        'proxy-control', '--out', str(output), '--metrics-out',
        str(tmp_path / 'metrics.json'), '--identity-out',
        str(tmp_path / 'identity.json'),
    ]
    runtime = _success_runtime(api, _base_config(), [], [])

    class Runner:
        model = object()

        def __init__(self, cfg):
            self._load_from = cfg.load_from

        def test(self):
            output.write_bytes(b'competing-complete-dump')
            native = FileExistsError(
                errno.EEXIST, 'File exists', str(pending), None, str(output))
            raise api.D13NDumpCollisionError.from_error(str(output), native)

    runtime.build_runner = lambda cfg: Runner(cfg)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.main(argv, runtime=runtime) == 3
    assert output.read_bytes() == b'competing-complete-dump'


def test_source_checkpoint_drift_after_snapshot_does_not_change_evidence(
        tmp_path, monkeypatch):
    api = _load_module()
    (tmp_path / 'config.py').write_bytes(b'config')
    checkpoint = tmp_path / 'epoch_24.pth'
    checkpoint.write_bytes(b'checkpoint')
    args = _parse(api, tmp_path)
    cfg = _base_config()
    cfg.d13n_parent_sha256 = hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    runtime = _success_runtime(api, cfg, [], [])
    original_build = runtime.build_runner

    def build(cfg):
        runner = original_build(cfg)
        test = runner.test

        def drift_after_test():
            metrics = test()
            checkpoint.write_bytes(b'drifted-checkpoint')
            return metrics

        runner.test = drift_after_test
        return runner

    runtime.build_runner = build
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    initial_sha = hashlib.sha256(b'checkpoint').hexdigest()
    assert api.execute(args, runtime=runtime) == 0
    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    identity = json.loads((tmp_path / 'identity.json').read_text())
    assert checkpoint.read_bytes() == b'drifted-checkpoint'
    assert metrics['checkpoint_sha256'] == initial_sha
    assert identity['checkpoint_sha256'] == initial_sha


def test_real_execute_failures_map_identity_to4_and_runtime_to5(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch',
        '--role', 'stage0-parent', '--out', str(tmp_path / 'predictions.pkl'),
        '--metrics-out', str(tmp_path / 'metrics.json'), '--identity-out',
        str(tmp_path / 'identity.json'), '--derive-adapter-free-parent',
    ]
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')
    assert api.main(
        argv, runtime=_success_runtime(api, _base_config(), [], [])) == 4

    monkeypatch.setenv('WORLD_SIZE', '5')
    cfg = _base_config()
    cfg.d13n_parent_sha256 = hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    runtime = _success_runtime(api, cfg, [], [])
    original = runtime.build_runner

    def build(cfg):
        runner = original(cfg)
        runner.test = lambda: (_ for _ in ()).throw(RuntimeError('boom'))
        return runner

    runtime.build_runner = build
    assert api.main(argv, runtime=runtime) == 5


def test_collision_is_detected_before_runner_construction(tmp_path, monkeypatch):
    api = _load_module()
    (tmp_path / 'config.py').write_bytes(b'config')
    (tmp_path / 'epoch_24.pth').write_bytes(b'checkpoint')
    (tmp_path / 'predictions.pkl').write_bytes(b'owned')
    args = _parse(api, tmp_path)
    built = []
    barriers = []
    runtime = _success_runtime(api, _base_config(), built, barriers)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    with pytest.raises(api.OutputCollisionError):
        api.execute(args, runtime=runtime)

    assert built == []
    assert (tmp_path / 'predictions.pkl').read_bytes() == b'owned'


def test_second_collision_scan_closes_config_load_race_before_runner(
        tmp_path, monkeypatch):
    api = _load_module()
    (tmp_path / 'config.py').write_bytes(b'config')
    (tmp_path / 'epoch_24.pth').write_bytes(b'checkpoint')
    args = _parse(api, tmp_path)
    built = []
    barriers = []
    runtime = _success_runtime(api, _base_config(), built, barriers)

    def racing_config_loader(path):
        (tmp_path / 'identity.json.pending.99.0').write_bytes(b'other-owner')
        cfg = _base_config()
        cfg.d13n_parent_sha256 = hashlib.sha256(
            (tmp_path / 'epoch_24.pth').read_bytes()).hexdigest()
        return cfg

    runtime.load_config = racing_config_loader
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    with pytest.raises(api.OutputCollisionError, match='pending'):
        api.execute(args, runtime=runtime)

    assert built == []
    assert (tmp_path / 'identity.json.pending.99.0').read_bytes() == (
        b'other-owner')


def test_execute_rejects_nonfive_hash_consensus_after_dist_init_before_test(
        tmp_path, monkeypatch):
    api = _load_module()
    (tmp_path / 'config.py').write_bytes(b'config')
    (tmp_path / 'epoch_24.pth').write_bytes(b'checkpoint')
    args = _parse(api, tmp_path)
    built = []
    barriers = []
    cfg = _base_config()
    cfg.d13n_parent_sha256 = hashlib.sha256(
        (tmp_path / 'epoch_24.pth').read_bytes()).hexdigest()
    runtime = _success_runtime(api, cfg, built, barriers)
    def drifted_gather(envelope):
        gathered = _five_rank_envelopes(envelope)
        gathered[1]['value']['resolved_config_sha256'] = 'd' * 64
        return gathered

    runtime.all_gather_object = drifted_gather
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    with pytest.raises(api.IdentityError, match='hash identity'):
        api.execute(args, runtime=runtime)
    # Runner construction is MMEngine's distributed initialization boundary;
    # consensus must happen after it, while Runner.test remains uncalled.
    assert len(built) == 1
    assert not (tmp_path / 'predictions.pkl').exists()


def test_post_init_rank_local_failure_converges_identically_on_every_rank():
    api = _load_module()
    good = {
        'rank': 0,
        'world_size': 5,
        'env_world_size': 5,
        'local_rank': 0,
        'config_sha256': 'a' * 64,
        'checkpoint_sha256': 'b' * 64,
        'resolved_config_sha256': 'c' * 64,
    }
    gathered = [
        {'ok': True, 'value': dict(good, rank=0, local_rank=0)},
        {
            'ok': False,
            'kind': 'identity',
            'message': 'stage0 strict load failed on rank 1',
        },
        {'ok': True, 'value': dict(good, rank=2, local_rank=2)},
        {'ok': True, 'value': dict(good, rank=3, local_rank=3)},
        {'ok': True, 'value': dict(good, rank=4, local_rank=4)},
    ]
    messages = []
    for local_rank in range(5):
        runtime = api.RuntimeFacade(
            all_gather_object=lambda envelope: gathered)
        with pytest.raises(api.IdentityError) as captured:
            api.converge_post_init_identity(
                runtime,
                lambda rank=local_rank: dict(
                    good, rank=rank, local_rank=rank))
        messages.append(str(captured.value))

    assert len(set(messages)) == 1
    assert 'rank 1' in messages[0]


def test_execute_converges_remote_stage0_failure_before_runner_test(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    args = _parse(api, tmp_path)
    tested = []
    built = []
    cfg = _base_config()
    cfg.d13n_parent_sha256 = hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    runtime = _success_runtime(api, cfg, built, [])
    original_builder = runtime.build_runner

    def build(cfg):
        runner = original_builder(cfg)
        runner.test = lambda: tested.append(True)
        return runner

    runtime.build_runner = build

    def gathered(envelope):
        successful = dict(envelope)
        successful['ok'] = True
        values = []
        for rank in range(5):
            value = dict(successful['value'], rank=rank, local_rank=rank)
            values.append({'ok': True, 'value': value})
        values[3] = {
            'ok': False,
            'kind': 'identity',
            'message': 'stage0 strict load failed on rank 3',
        }
        return values

    runtime.all_gather_object = gathered
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    with pytest.raises(api.IdentityError, match='rank 3'):
        api.execute(args, runtime=runtime)
    assert len(built) == 1
    assert tested == []


@pytest.mark.parametrize(
    'field,bad_value',
    [
        ('records', 400.0),
        ('unique_image_ids', 400.0),
        ('prediction_rows', 240000.0),
        ('all_cpu_finite', 1),
    ],
)
def test_dump_summary_requires_exact_builtin_types(field, bad_value):
    api = _load_module()
    summary = {
        'records': 400,
        'unique_image_ids': 400,
        'prediction_rows': 240000,
        'all_cpu_finite': True,
    }
    summary[field] = bad_value
    runtime = api.RuntimeFacade(
        load_records=lambda path: ['records'],
        validate_records=lambda records, **kwargs: summary)

    with pytest.raises(RuntimeError, match='summary'):
        api._validate_dump(runtime, Path('predictions.pkl'), 'proxy400')


@pytest.mark.parametrize('failed_input', ['config', 'checkpoint'])
def test_input_open_hash_oserror_maps_to_identity4(
        failed_input, tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch', '--role',
        'proxy-control', '--out', str(tmp_path / 'predictions.pkl'),
        '--metrics-out', str(tmp_path / 'metrics.json'), '--identity-out',
        str(tmp_path / 'identity.json'),
    ]
    real_open = api.os.open

    def failing_open(path, flags, *rest, **kwargs):
        config_open = (
            failed_input == 'config'
            and Path(path) == Path(config.name)
            and kwargs.get('dir_fd') is not None)
        checkpoint_open = (
            failed_input == 'checkpoint' and Path(path) == checkpoint)
        if config_open or checkpoint_open:
            raise OSError('simulated input read failure')
        return real_open(path, flags, *rest, **kwargs)

    monkeypatch.setattr(api.os, 'open', failing_open)
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')
    assert api.main(
        argv, runtime=_success_runtime(api, _base_config(), [], [])) == 4


@pytest.mark.parametrize(
    'field,bad_value',
    [
        ('CUDA_VISIBLE_DEVICES', '0,1,2,3,4'),
        ('NCCL_P2P_DISABLE', '0'),
        ('NCCL_IB_DISABLE', '0'),
    ],
)
def test_global_gpu_nccl_drift_fails_before_runner_construction(
        field, bad_value, tmp_path, monkeypatch):
    api = _load_module()
    (tmp_path / 'config.py').write_bytes(b'config')
    (tmp_path / 'epoch_24.pth').write_bytes(b'checkpoint')
    args = _parse(api, tmp_path, role='proxy-control', derive=False)
    built = []
    runtime = _success_runtime(api, _base_config(), built, [])
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')
    monkeypatch.setenv(field, bad_value)

    with pytest.raises(api.IdentityError):
        api.execute(args, runtime=runtime)
    assert built == []


def test_invalid_local_rank_converges_only_after_runner_init(
        tmp_path, monkeypatch):
    api = _load_module()
    (tmp_path / 'config.py').write_bytes(b'config')
    (tmp_path / 'epoch_24.pth').write_bytes(b'checkpoint')
    args = _parse(api, tmp_path, role='proxy-control', derive=False)
    built = []
    runtime = _success_runtime(api, _base_config(), built, [])
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '9')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    with pytest.raises(api.IdentityError, match='LOCAL_RANK'):
        api.execute(args, runtime=runtime)
    assert len(built) == 1


def test_exit_code_mapping_is_frozen(tmp_path, monkeypatch):
    api = _load_module()
    valid = [
        'config.py', 'epoch_24.pth', '--launcher', 'pytorch',
        '--role', 'stage0-parent', '--out', 'pred.pkl',
        '--metrics-out', 'metrics.json', '--identity-out', 'identity.json',
        '--derive-adapter-free-parent',
    ]

    for error, code in (
            (api.OutputCollisionError('collision'), 3),
            (api.IdentityError('identity'), 4),
            (RuntimeError('runtime'), 5)):
        monkeypatch.setattr(api, 'execute', lambda args, runtime=None, e=error: (
            (_ for _ in ()).throw(e)))
        assert api.main(valid) == code


def test_unserializable_resolved_config_is_identity_code4(
        tmp_path, monkeypatch):
    api = _load_module()
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'epoch_24.pth'
    config.write_bytes(b'config')
    checkpoint.write_bytes(b'checkpoint')
    cfg = _base_config()
    cfg.unserializable_identity_value = {'not-json'}
    runtime = _success_runtime(api, cfg, [], [])
    runtime.build_runner = lambda cfg: pytest.fail(
        'identity failure must precede runner construction')
    argv = [
        str(config), str(checkpoint), '--launcher', 'pytorch', '--role',
        'proxy-control', '--out', str(tmp_path / 'predictions.pkl'),
        '--metrics-out', str(tmp_path / 'metrics.json'), '--identity-out',
        str(tmp_path / 'identity.json'),
    ]
    monkeypatch.setenv('WORLD_SIZE', '5')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,6,7,8,9')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('NCCL_IB_DISABLE', '1')

    assert api.main(argv, runtime=runtime) == 4
