#!/usr/bin/env python3
"""Queue a distributed test run after a training job fully exits."""

import argparse
import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple


def validation_complete(text: str, target_epoch: int) -> bool:
    """Return whether the target epoch's final DOTA validation was logged."""
    pattern = (
        r'Epoch\(val\) \[{}\]\[\d+/\d+\].*'
        r'dota/mAP:\s*\S+.*dota/AP50:\s*\S+'
    ).format(int(target_epoch))
    return re.search(pattern, text) is not None


def build_test_spec(
        python: Path,
        config: Path,
        checkpoint: Path,
        work_dir: Path,
        prediction_out: Path,
        gpu_indices: Sequence[int],
        port: int) -> Tuple[list, Dict[str, str]]:
    """Build the fixed-mouth distributed test command and environment."""
    gpu_indices = tuple(int(index) for index in gpu_indices)
    if not gpu_indices:
        raise ValueError('gpu_indices must not be empty')
    command = [
        str(python),
        '-m',
        'torch.distributed.launch',
        '--nproc_per_node={}'.format(len(gpu_indices)),
        '--master_port={}'.format(int(port)),
        'tools/test.py',
        str(config),
        str(checkpoint),
        '--launcher',
        'pytorch',
        '--work-dir',
        str(work_dir),
        '--out',
        str(prediction_out),
    ]
    environment = {
        'CUDA_VISIBLE_DEVICES': ','.join(str(index) for index in gpu_indices),
        'NCCL_P2P_DISABLE': '1',
        'NCCL_IB_DISABLE': '1',
        'OMP_NUM_THREADS': '1',
        'PYTHONNOUSERSITE': '1',
    }
    return command, environment


def scheduler_status(
        checkpoint_published: bool,
        validation_finished: bool,
        training_alive: bool,
        gpus_idle: bool,
        already_launched: bool) -> dict:
    """Combine runtime safety gates into a stable wait reason list."""
    waiting_for = []
    if not checkpoint_published:
        waiting_for.append('checkpoint')
    if not validation_finished:
        waiting_for.append('validation')
    if training_alive:
        waiting_for.append('training_exit')
    if not gpus_idle:
        waiting_for.append('gpu_idle')
    if already_launched:
        waiting_for.append('already_launched')
    return {'ready': not waiting_for, 'waiting_for': waiting_for}


def event_recorded(path: Path, event: str) -> bool:
    """Return whether a JSONL state log already contains an event."""
    try:
        with Path(path).open('r', encoding='utf-8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('event') == event:
                    return True
    except FileNotFoundError:
        return False
    return False


def checkpoint_complete(path: Path) -> bool:
    """Require a full-sized checkpoint atomically published by MMEngine."""
    path = Path(path)
    try:
        if path.stat().st_size < 1_500_000_000:
            return False
        pointer = Path((path.parent / 'last_checkpoint').read_text().strip())
        if not pointer.is_absolute():
            pointer = path.parent / pointer
        return pointer.resolve() == path.resolve()
    except (FileNotFoundError, OSError):
        return False


def parse_compute_pids(text: str) -> list:
    """Parse the PID-only nvidia-smi compute-app query."""
    pids = []
    for line in text.splitlines():
        value = line.strip()
        if not value or not value.isdigit():
            continue
        pids.append(int(value))
    return pids


def build_validator_spec(
        python: Path,
        validator: Path,
        prediction_out: Path) -> Tuple[list, Dict[str, str]]:
    """Build a GPU-hidden structural validation command."""
    command = [str(python), str(validator), str(prediction_out)]
    environment = {
        'CUDA_VISIBLE_DEVICES': '',
        'PYTHONNOUSERSITE': '1',
    }
    return command, environment


def query_gpu_compute_pids(index: int) -> list:
    """Return compute PIDs attached to one physical GPU index."""
    command = [
        'nvidia-smi',
        '--id={}'.format(int(index)),
        '--query-compute-apps=pid',
        '--format=csv,noheader,nounits',
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=15)
    return parse_compute_pids(completed.stdout)


def gpus_idle(
        gpu_indices: Sequence[int],
        query_fn: Callable[[int], list] = query_gpu_compute_pids) -> bool:
    """Require every requested physical GPU to have no compute process."""
    for index in gpu_indices:
        try:
            pids = query_fn(int(index))
        except (OSError, subprocess.SubprocessError):
            # A transient NVML/nvidia-smi failure must fail closed. The
            # scheduler will poll again instead of crashing or using a GPU
            # whose occupancy could not be established.
            return False
        if pids:
            return False
    return True


def matching_training_exists(
        config: Path,
        proc_root: Path = Path('/proc')) -> bool:
    """Find a live tools/train.py process using the exact config path."""
    needle = str(config).encode()
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = command.split(b'\0')
        has_train_script = any(
            arg == b'tools/train.py' or arg.endswith(b'/tools/train.py')
            for arg in argv)
        if needle in argv and has_train_script:
            return True
    return False


def read_monitor_process_alive(path: Path) -> Optional[bool]:
    """Read the latest process liveness sample from a monitor JSONL log.

    The reverse scan deliberately skips summary and malformed rows. This lets
    the scheduler work inside a PID namespace where the host training ranks
    are invisible under ``/proc`` while retaining the monitor's last direct
    observation of the process group.
    """
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except (FileNotFoundError, OSError):
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get('process_group_alive')
        if isinstance(value, bool):
            return value
    return None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def append_record(
        path: Path,
        now_fn: Optional[Callable[[], str]] = None,
        **record) -> None:
    """Append one timestamped, sorted JSON object atomically enough for JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record['timestamp'] = (now_fn or _now_iso)()
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')


def collect_status(
        checkpoint: Path,
        train_log: Path,
        target_epoch: int,
        training_config: Path,
        monitor_log: Path,
        gpu_indices: Sequence[int],
        state_log: Path,
        training_fn: Callable[[Path], bool] = matching_training_exists,
        gpu_idle_fn: Callable[[Sequence[int]], bool] = gpus_idle) -> dict:
    """Collect live evidence and combine it into the scheduler gate."""
    try:
        train_text = Path(train_log).read_text(
            encoding='utf-8', errors='replace')
    except FileNotFoundError:
        train_text = ''
    training_proc_visible = training_fn(training_config)
    training_monitor_alive = read_monitor_process_alive(monitor_log)
    evidence = {
        'checkpoint_published': checkpoint_complete(checkpoint),
        'validation_finished': validation_complete(train_text, target_epoch),
        'training_alive': (
            training_proc_visible or training_monitor_alive is True),
        'gpus_idle': gpu_idle_fn(gpu_indices),
        'already_launched': event_recorded(
            state_log, 'evaluation_launched'),
    }
    status = scheduler_status(**evidence)
    status.update(evidence)
    status.update({
        'training_proc_visible': training_proc_visible,
        'training_monitor_alive': training_monitor_alive,
    })
    return status


def port_available(port: int) -> bool:
    """Return whether a local distributed rendezvous port can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('127.0.0.1', int(port)))
        except OSError:
            return False
    return True


def test_metric_complete(text: str) -> bool:
    """Require a completed test metric line, not only a saved pickle."""
    return re.search(
        r'Epoch\(test\) \[\d+/\d+\].*dota/mAP:\s*\S+.*'
        r'dota/AP50:\s*\S+', text) is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--train-log', type=Path, required=True)
    parser.add_argument('--monitor-log', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target-epoch', type=int, default=24)
    parser.add_argument('--eval-config', type=Path, required=True)
    parser.add_argument('--eval-work-dir', type=Path, required=True)
    parser.add_argument('--prediction-out', type=Path, required=True)
    parser.add_argument('--eval-log', type=Path, required=True)
    parser.add_argument('--state-log', type=Path, required=True)
    parser.add_argument('--gpu-indices', type=int, nargs='+', required=True)
    parser.add_argument('--master-port', type=int, default=29791)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--validator', type=Path, required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--interval', type=float, default=30.0)
    parser.add_argument('--check-once', action='store_true')
    return parser


def _environment(extra: Dict[str, str], repo_root: Path) -> dict:
    environment = os.environ.copy()
    environment.update(extra)
    environment['PYTHONPATH'] = str(repo_root)
    return environment


def main() -> int:
    args = build_parser().parse_args()
    if args.interval <= 0:
        raise ValueError('interval must be positive')
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError('deadline must include a UTC offset')

    status = collect_status(
        checkpoint=args.checkpoint,
        train_log=args.train_log,
        target_epoch=args.target_epoch,
        training_config=args.training_config,
        monitor_log=args.monitor_log,
        gpu_indices=args.gpu_indices,
        state_log=args.state_log,
    )
    if args.check_once:
        print(json.dumps(status, sort_keys=True))
        return 0
    if status['already_launched']:
        append_record(
            args.state_log,
            event='duplicate_scheduler_exit',
            waiting_for=status['waiting_for'])
        return 0

    previous_wait = None
    while True:
        status = collect_status(
            checkpoint=args.checkpoint,
            train_log=args.train_log,
            target_epoch=args.target_epoch,
            training_config=args.training_config,
            monitor_log=args.monitor_log,
            gpu_indices=args.gpu_indices,
            state_log=args.state_log,
        )
        waiting = tuple(status['waiting_for'])
        if waiting != previous_wait:
            append_record(
                args.state_log,
                event='waiting' if waiting else 'ready',
                **status)
            previous_wait = waiting
        if status['ready']:
            break
        if datetime.now().astimezone() >= deadline:
            append_record(
                args.state_log,
                event='deadline_reached',
                **status)
            return 2
        time.sleep(args.interval)

    while not port_available(args.master_port):
        if datetime.now().astimezone() >= deadline:
            append_record(
                args.state_log,
                event='deadline_reached_waiting_for_port',
                port=args.master_port)
            return 2
        append_record(
            args.state_log,
            event='waiting_for_port',
            port=args.master_port)
        time.sleep(args.interval)

    if args.prediction_out.exists() or args.eval_log.exists():
        append_record(
            args.state_log,
            event='output_collision',
            prediction_exists=args.prediction_out.exists(),
            eval_log_exists=args.eval_log.exists())
        return 2

    args.eval_work_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_log.parent.mkdir(parents=True, exist_ok=True)
    command, test_environment = build_test_spec(
        python=args.python,
        config=args.eval_config,
        checkpoint=args.checkpoint,
        work_dir=args.eval_work_dir,
        prediction_out=args.prediction_out,
        gpu_indices=args.gpu_indices,
        port=args.master_port,
    )
    with args.eval_log.open('xb', buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=str(args.repo_root),
            env=_environment(test_environment, args.repo_root),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        append_record(
            args.state_log,
            event='evaluation_launched',
            pid=process.pid,
            command=command,
            gpu_indices=args.gpu_indices,
            eval_log=str(args.eval_log),
            prediction_out=str(args.prediction_out))
        return_code = process.wait()

    append_record(
        args.state_log,
        event='evaluation_exited',
        pid=process.pid,
        return_code=return_code)
    if return_code != 0:
        return return_code

    eval_text = args.eval_log.read_text(encoding='utf-8', errors='replace')
    if not test_metric_complete(eval_text):
        append_record(
            args.state_log,
            event='evaluation_metric_missing',
            eval_log=str(args.eval_log))
        return 3

    validator_command, validator_environment = build_validator_spec(
        python=args.python,
        validator=args.validator,
        prediction_out=args.prediction_out,
    )
    validated = subprocess.run(
        validator_command,
        cwd=str(args.repo_root),
        env=_environment(validator_environment, args.repo_root),
        capture_output=True,
        text=True,
    )
    append_record(
        args.state_log,
        event='dump_validation_exited',
        return_code=validated.returncode,
        stdout=validated.stdout.strip(),
        stderr=validated.stderr.strip())
    if validated.returncode != 0:
        return validated.returncode
    append_record(
        args.state_log,
        event='queue_complete',
        prediction_out=str(args.prediction_out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
