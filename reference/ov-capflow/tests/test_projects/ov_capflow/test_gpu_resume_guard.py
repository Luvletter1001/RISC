import importlib.util
import subprocess
import sys
from pathlib import Path


GUARD_PATH = (Path(__file__).parents[3] / '.lab' / 'workspace' /
              'guard_gpu89_resume.py')
CHECKER_PATH = (Path(__file__).parents[3] / '.lab' / 'workspace' /
                'check_complete_checkpoint.py')


def _load_guard_module():
    spec = importlib.util.spec_from_file_location('guard_gpu89_resume', GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_checkpoint_path_uses_requested_epoch():
    guard = _load_guard_module()

    assert guard.target_checkpoint_path(Path('/tmp/full24'), 24) == Path(
        '/tmp/full24/epoch_24.pth')


def test_guard_cli_exposes_target_epoch():
    result = subprocess.run(
        [sys.executable, str(GUARD_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--target-epoch' in result.stdout


def test_resume_spec_uses_requested_physical_gpus_and_world_size():
    guard = _load_guard_module()
    build_resume_spec = getattr(guard, 'build_resume_spec', None)

    assert callable(build_resume_spec), 'generic resume spec is missing'
    command, environment = build_resume_spec(
        config=Path('/tmp/config.py'),
        gpu_indices=(2, 3, 8, 9),
        port=29789,
    )

    assert '--nproc_per_node=4' in command
    assert '--master_port=29789' in command
    assert '/tmp/config.py' in command
    assert environment['CUDA_VISIBLE_DEVICES'] == '2,3,8,9'
    assert environment['NCCL_P2P_DISABLE'] == '1'
    assert environment['NCCL_IB_DISABLE'] == '1'


def test_guard_cli_exposes_gpu_indices():
    result = subprocess.run(
        [sys.executable, str(GUARD_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--gpu-indices' in result.stdout


def test_monitor_spec_follows_relaunched_pid_and_physical_gpus():
    guard = _load_guard_module()
    build_monitor_spec = getattr(guard, 'build_monitor_spec', None)

    assert callable(build_monitor_spec), 'resume monitor spec is missing'
    command = build_monitor_spec(
        monitor_script=Path('/tmp/monitor.py'),
        pid=4321,
        gpu_indices=(2, 3, 8, 9),
        train_log=Path('/tmp/resume.log'),
        output=Path('/tmp/monitor.jsonl'),
    )

    assert command[0] == '/data/zcy/anaconda3/envs/mmdet/bin/python'
    assert command[1:4] == ['/tmp/monitor.py', '--pid', '4321']
    gpu_start = command.index('--gpu-indices') + 1
    assert command[gpu_start:gpu_start + 4] == ['2', '3', '8', '9']
    assert command[command.index('--train-log') + 1] == '/tmp/resume.log'
    assert command[command.index('--output') + 1] == '/tmp/monitor.jsonl'


def test_guard_cli_exposes_resume_monitor_options():
    result = subprocess.run(
        [sys.executable, str(GUARD_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--monitor-script' in result.stdout
    assert '--monitor-output' in result.stdout


def test_checkpoint_checker_rejects_missing_checkpoint(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH), str(tmp_path / 'epoch_1.pth')])

    assert result.returncode == 1


def test_checkpoint_checker_accepts_fully_published_checkpoint(tmp_path):
    checkpoint = tmp_path / 'epoch_1.pth'
    with checkpoint.open('wb') as stream:
        stream.truncate(1_500_000_000)
    (tmp_path / 'last_checkpoint').write_text(
        str(checkpoint), encoding='utf-8')

    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH), str(checkpoint)])

    assert result.returncode == 0
