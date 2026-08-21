import importlib.util
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (Path(__file__).parents[3] / 'projects' / 'OVCapFlow' /
               'tools' / 'queue_test_after_training.py')


def _load_module():
    spec = importlib.util.spec_from_file_location(
        'queue_test_after_training', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_complete_requires_target_epoch_metric():
    scheduler = _load_module()
    text = (
        '07/22 02:00:00 - mmengine - INFO - '
        'Epoch(val) [24][1730/1730] dota/mAP: 0.6100 '
        'dota/AP50: 0.6100')

    assert scheduler.validation_complete(text, target_epoch=24)
    assert not scheduler.validation_complete(text, target_epoch=18)
    assert not scheduler.validation_complete(
        'Epoch(train) [24][5916/5916]', target_epoch=24)


def test_build_test_spec_keeps_fixed_gpu_and_mouth_contract():
    scheduler = _load_module()

    command, environment = scheduler.build_test_spec(
        python=Path('/env/bin/python'),
        config=Path('/repo/base.py'),
        checkpoint=Path('/repo/epoch_24.pth'),
        work_dir=Path('/repo/eval'),
        prediction_out=Path('/repo/eval/predictions.pkl'),
        gpu_indices=(2, 3, 8, 9),
        port=29791,
    )

    assert '--nproc_per_node=4' in command
    assert '--master_port=29791' in command
    assert 'tools/test.py' in command
    assert '/repo/base.py' in command
    assert '/repo/epoch_24.pth' in command
    assert command[command.index('--out') + 1] == '/repo/eval/predictions.pkl'
    assert environment['CUDA_VISIBLE_DEVICES'] == '2,3,8,9'
    assert environment['NCCL_P2P_DISABLE'] == '1'
    assert environment['NCCL_IB_DISABLE'] == '1'
    assert environment['OMP_NUM_THREADS'] == '1'
    assert environment['PYTHONNOUSERSITE'] == '1'


def test_scheduler_status_requires_every_runtime_gate():
    scheduler = _load_module()

    ready = scheduler.scheduler_status(
        checkpoint_published=True,
        validation_finished=True,
        training_alive=False,
        gpus_idle=True,
        already_launched=False,
    )

    assert ready == {'ready': True, 'waiting_for': []}

    cases = (
        ({'checkpoint_published': False}, 'checkpoint'),
        ({'validation_finished': False}, 'validation'),
        ({'training_alive': True}, 'training_exit'),
        ({'gpus_idle': False}, 'gpu_idle'),
        ({'already_launched': True}, 'already_launched'),
    )
    defaults = {
        'checkpoint_published': True,
        'validation_finished': True,
        'training_alive': False,
        'gpus_idle': True,
        'already_launched': False,
    }
    for override, reason in cases:
        inputs = dict(defaults)
        inputs.update(override)
        status = scheduler.scheduler_status(**inputs)
        assert not status['ready']
        assert reason in status['waiting_for']


def test_state_log_detects_prior_evaluation_launch(tmp_path):
    scheduler = _load_module()
    state_log = tmp_path / 'queue.jsonl'
    state_log.write_text(
        '{"event":"waiting"}\n'
        '{"event":"evaluation_launched","pid":123}\n',
        encoding='utf-8',
    )

    assert scheduler.event_recorded(state_log, 'evaluation_launched')
    assert not scheduler.event_recorded(state_log, 'evaluation_complete')
    assert not scheduler.event_recorded(
        tmp_path / 'missing.jsonl', 'evaluation_launched')


def test_checkpoint_complete_requires_size_and_last_checkpoint(tmp_path):
    scheduler = _load_module()
    checkpoint = tmp_path / 'epoch_24.pth'
    with checkpoint.open('wb') as stream:
        stream.truncate(1_500_000_000)

    assert not scheduler.checkpoint_complete(checkpoint)

    (tmp_path / 'last_checkpoint').write_text(
        str(checkpoint), encoding='utf-8')
    assert scheduler.checkpoint_complete(checkpoint)

    (tmp_path / 'last_checkpoint').write_text(
        str(tmp_path / 'epoch_18.pth'), encoding='utf-8')
    assert not scheduler.checkpoint_complete(checkpoint)


def test_parse_compute_pids_handles_empty_and_numeric_rows():
    scheduler = _load_module()

    assert scheduler.parse_compute_pids('') == []
    assert scheduler.parse_compute_pids('1234\n5678\n') == [1234, 5678]
    assert scheduler.parse_compute_pids('No running processes found\n') == []


def test_build_validator_spec_forces_cpu(tmp_path):
    scheduler = _load_module()
    command, environment = scheduler.build_validator_spec(
        python=Path('/env/bin/python'),
        validator=Path('/repo/validate.py'),
        prediction_out=tmp_path / 'predictions.pkl',
    )

    assert command == [
        '/env/bin/python',
        '/repo/validate.py',
        str(tmp_path / 'predictions.pkl'),
    ]
    assert environment['CUDA_VISIBLE_DEVICES'] == ''
    assert environment['PYTHONNOUSERSITE'] == '1'


def test_gpus_idle_checks_only_requested_physical_indices():
    scheduler = _load_module()
    calls = []

    def query(index):
        calls.append(index)
        return [] if index != 8 else [999]

    assert not scheduler.gpus_idle((2, 3, 8, 9), query_fn=query)
    assert calls == [2, 3, 8]
    assert scheduler.gpus_idle((2, 3), query_fn=lambda _index: [])


def test_gpus_idle_treats_transient_nvidia_smi_failure_as_busy():
    scheduler = _load_module()

    def failing_query(_index):
        raise subprocess.CalledProcessError(
            returncode=9, cmd=['nvidia-smi'])

    assert not scheduler.gpus_idle((2, 3, 8, 9), query_fn=failing_query)


def test_matching_training_exists_reads_proc_cmdlines(tmp_path):
    scheduler = _load_module()
    config = Path('/repo/live.py')
    relative_script = tmp_path / '123'
    relative_script.mkdir()
    (relative_script / 'cmdline').write_bytes(
        b'python\x00tools/train.py\x00/repo/live.py\x00')

    assert scheduler.matching_training_exists(config, proc_root=tmp_path)
    assert not scheduler.matching_training_exists(
        Path('/repo/other.py'), proc_root=tmp_path)


def test_matching_training_exists_accepts_absolute_script_and_final_argv(
        tmp_path):
    scheduler = _load_module()
    process = tmp_path / '456'
    process.mkdir()
    (process / 'cmdline').write_bytes(
        b'python\x00/repo/tools/train.py\x00/repo/live.py')

    assert scheduler.matching_training_exists(
        Path('/repo/live.py'), proc_root=tmp_path)


def test_matching_training_exists_rejects_embedded_historical_command(
        tmp_path):
    scheduler = _load_module()
    tmux_server = tmp_path / '789'
    tmux_server.mkdir()
    (tmux_server / 'cmdline').write_bytes(
        b'tmux\x00new-session\x00bash -lc "python tools/train.py '
        b'/repo/live.py"\x00')

    assert not scheduler.matching_training_exists(
        Path('/repo/live.py'), proc_root=tmp_path)


def test_matching_training_exists_requires_exact_config_argv(tmp_path):
    scheduler = _load_module()
    process = tmp_path / '901'
    process.mkdir()
    (process / 'cmdline').write_bytes(
        b'python\x00tools/train.py\x00--config=/repo/live.py\x00')

    assert not scheduler.matching_training_exists(
        Path('/repo/live.py'), proc_root=tmp_path)


def test_matching_training_exists_skips_non_process_and_vanished_entries(
        tmp_path):
    scheduler = _load_module()
    (tmp_path / 'self').mkdir()
    (tmp_path / '234').mkdir()

    assert not scheduler.matching_training_exists(
        Path('/repo/live.py'), proc_root=tmp_path)


def test_read_monitor_process_alive_uses_latest_sample(tmp_path):
    scheduler = _load_module()
    monitor_log = tmp_path / 'monitor.jsonl'
    monitor_log.write_text(
        '{"process_group_alive":true,"type":"sample"}\n'
        '{"process_group_alive":false,"type":"sample"}\n'
        '{"reason":"all_ranks_dead","type":"summary"}\n',
        encoding='utf-8',
    )

    assert scheduler.read_monitor_process_alive(monitor_log) is False
    assert scheduler.read_monitor_process_alive(
        tmp_path / 'missing.jsonl') is None


def test_collect_status_uses_monitor_when_proc_namespace_hides_training(
        tmp_path):
    scheduler = _load_module()
    monitor_log = tmp_path / 'monitor.jsonl'
    monitor_log.write_text(
        '{"process_group_alive":true,"type":"sample"}\n',
        encoding='utf-8',
    )

    status = scheduler.collect_status(
        checkpoint=tmp_path / 'missing_epoch_24.pth',
        train_log=tmp_path / 'train.log',
        target_epoch=24,
        training_config=Path('/repo/live.py'),
        monitor_log=monitor_log,
        gpu_indices=(2, 3, 8, 9),
        state_log=tmp_path / 'queue.jsonl',
        training_fn=lambda _config: False,
        gpu_idle_fn=lambda _indices: False,
    )

    assert status['training_alive']
    assert not status['training_proc_visible']
    assert status['training_monitor_alive'] is True
    assert 'training_exit' in status['waiting_for']


def test_append_record_writes_timestamped_jsonl(tmp_path):
    scheduler = _load_module()
    state_log = tmp_path / 'queue.jsonl'

    scheduler.append_record(
        state_log,
        now_fn=lambda: '2026-07-21T14:00:00+08:00',
        event='waiting',
        waiting_for=['checkpoint'],
    )

    assert state_log.read_text(encoding='utf-8') == (
        '{"event": "waiting", "timestamp": '
        '"2026-07-21T14:00:00+08:00", '
        '"waiting_for": ["checkpoint"]}\n')


def test_collect_status_composes_live_runtime_evidence(tmp_path):
    scheduler = _load_module()
    checkpoint = tmp_path / 'epoch_24.pth'
    with checkpoint.open('wb') as stream:
        stream.truncate(1_500_000_000)
    (tmp_path / 'last_checkpoint').write_text(
        str(checkpoint), encoding='utf-8')
    train_log = tmp_path / 'train.log'
    train_log.write_text(
        'Epoch(val) [24][1730/1730] dota/mAP: 0.6100 '
        'dota/AP50: 0.6100\n',
        encoding='utf-8',
    )

    status = scheduler.collect_status(
        checkpoint=checkpoint,
        train_log=train_log,
        target_epoch=24,
        training_config=Path('/repo/live.py'),
        monitor_log=tmp_path / 'monitor.jsonl',
        gpu_indices=(2, 3, 8, 9),
        state_log=tmp_path / 'queue.jsonl',
        training_fn=lambda _config: False,
        gpu_idle_fn=lambda _indices: True,
    )

    assert status['ready']
    assert status['checkpoint_published']
    assert status['validation_finished']
    assert not status['training_alive']
    assert status['gpus_idle']


def test_cli_exposes_check_once_and_runtime_paths():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    for option in (
            '--check-once', '--training-config', '--train-log',
            '--checkpoint', '--monitor-log', '--eval-config', '--prediction-out',
            '--gpu-indices', '--state-log'):
        assert option in result.stdout
