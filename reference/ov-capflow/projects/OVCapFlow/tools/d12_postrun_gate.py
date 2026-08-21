#!/usr/bin/env python3
"""Wait for and evaluate the frozen D12 paired epoch-12 endpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from projects.OVCapFlow.tools.audit_d12_preflight import (
    publish_json_no_clobber,
)
from projects.OVCapFlow.tools.prepare_cleanstart_checkpoint import sha256_file


CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
NOVEL_CLASSES = frozenset(
    ('airport', 'container-crane', 'helipad', 'helicopter'))
BASE_CLASSES = tuple(
    name for name in CLASSES if name not in NOVEL_CLASSES)
TARGET_EPOCH = 12
IDLE_GPU_INDICES = tuple(range(10))
MINIMUM_CHECKPOINT_BYTES = 1_500_000_000
DEFAULT_STAGE0 = Path(
    'work_dirs/dotav2_cleanstart/audits/'
    'd12_stage0_preflight_seed20260716.json')
THRESHOLDS = {
    'candidate_AP50_absolute_min': 0.4890,
    'candidate_AP50_control_delta_min': 0.020,
    'candidate_mAP_control_delta_min': 0.020,
    'candidate_novel4_absolute_min': 0.3675,
    'candidate_novel4_control_delta_min': 0.0,
    'candidate_base14_absolute_min': 0.464643,
    'candidate_base14_control_delta_min': -0.005,
}
REQUIRED_INTEGRITY = frozenset({
    'sampler',
    'stage0',
    'candidate_strict',
    'candidate_open_vocabulary',
    'control_strict',
    'control_open_vocabulary',
    'rotated_boxes',
    'fatal_free',
})

FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
METRIC = re.compile(
    rf'Epoch\(val\)\s+\[(?P<epoch>\d+)\]'
    rf'\[\s*(?P<done>\d+)/(?P<total>\d+)\].*?'
    rf'dota/mAP:\s*(?P<map>{FLOAT}).*?'
    rf'dota/AP50:\s*(?P<ap50>{FLOAT})')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
HEADER = re.compile(r'^\|\s*class\s*\|')
BORDER = re.compile(r'^\+\s*-')
CLASS_ROW = re.compile(
    r'^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<gts>\d+)\s*\|'
    r'\s*(?P<dets>\d+)\s*\|\s*(?P<recall>[^|\s]+)\s*\|'
    r'\s*(?P<ap>[^|\s]+)\s*\|$')
MAP_ROW = re.compile(
    r'^\|\s*mAP\s*\|\s*\|\s*\|\s*\|\s*(?P<map>[^|\s]+)\s*\|$')
FATAL_PATTERNS = (
    ('traceback', re.compile(r'Traceback \(most recent call last\):')),
    ('cuda_oom', re.compile(r'CUDA out of memory', re.IGNORECASE)),
    (
        'nccl_failure',
        re.compile(
            r'NCCL.*(?:error|abort|aborted|failed|failure)',
            re.IGNORECASE)),
    (
        'nonfinite',
        re.compile(
            r'(?:^|[\s:=,\[])' + r'(?:nan|[+-]?inf)'
            + r'(?:$|[\s,\]])',
            re.IGNORECASE)),
)


def exact_training_process_exists(
        config: Path, proc_root: Path = Path('/proc')) -> bool:
    """Find a live tools/train.py argv containing the exact config argument."""
    config = Path(config)
    needles = {
        os.fsencode(os.fspath(config)),
        os.fsencode(os.fspath(config.resolve())),
    }
    try:
        entries = Path(proc_root).iterdir()
    except OSError:
        return True
    try:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                argv = (entry / 'cmdline').read_bytes().split(b'\0')
            except (
                    FileNotFoundError,
                    PermissionError,
                    ProcessLookupError,
                    OSError):
                continue
            has_train = any(
                value == b'tools/train.py'
                or value.endswith(b'/tools/train.py')
                for value in argv)
            if has_train and any(needle in argv for needle in needles):
                return True
    except OSError:
        return True
    return False


def extract_complete_epoch_metric(
        text: str, target_epoch: int = TARGET_EPOCH) -> dict[str, Any] | None:
    """Return the last finite, fully completed target-epoch summary."""
    result = None
    for match in METRIC.finditer(text):
        epoch = int(match.group('epoch'))
        done = int(match.group('done'))
        total = int(match.group('total'))
        map_value = float(match.group('map'))
        ap50 = float(match.group('ap50'))
        if (epoch != int(target_epoch) or total <= 0 or done != total
                or not math.isfinite(map_value)
                or not math.isfinite(ap50)):
            continue
        result = {
            'epoch': epoch,
            'progress': [done, total],
            'mAP': map_value,
            'AP50': ap50,
        }
    return result


def pair_completion_status(
        *,
        candidate_training_alive: bool,
        control_training_alive: bool,
        candidate_metric_complete: bool,
        control_metric_complete: bool,
        candidate_checkpoint_published: bool,
        control_checkpoint_published: bool,
        candidate_sampler_complete: bool,
        control_sampler_complete: bool,
        sampler_checksums_match: bool,
        gpus_idle: bool,
        stage0_passed: bool,
        fatal_free: bool) -> dict[str, Any]:
    """Combine all endpoint evidence into stable fail-closed wait reasons."""
    waiting_for = []
    conditions = (
        (candidate_training_alive, 'candidate_training_exit', True),
        (control_training_alive, 'control_training_exit', True),
        (candidate_metric_complete, 'candidate_epoch12_metric', False),
        (control_metric_complete, 'control_epoch12_metric', False),
        (
            candidate_checkpoint_published,
            'candidate_checkpoint',
            False),
        (control_checkpoint_published, 'control_checkpoint', False),
        (candidate_sampler_complete, 'candidate_sampler', False),
        (control_sampler_complete, 'control_sampler', False),
        (sampler_checksums_match, 'sampler_checksum_match', False),
        (gpus_idle, 'gpu0_9_idle', False),
        (stage0_passed, 'stage0_pass', False),
        (fatal_free, 'fatal_free', False),
    )
    for value, reason, wait_when_true in conditions:
        if bool(value) == wait_when_true:
            waiting_for.append(reason)
    return {'ready': not waiting_for, 'waiting_for': waiting_for}


def checkpoint_published(
        checkpoint: Path,
        *,
        minimum_bytes: int = MINIMUM_CHECKPOINT_BYTES) -> bool:
    """Require a full checkpoint and exact MMEngine last pointer."""
    checkpoint = Path(checkpoint)
    try:
        if checkpoint.stat().st_size < int(minimum_bytes):
            return False
        pointer_text = (checkpoint.parent / 'last_checkpoint').read_text(
            encoding='utf-8').strip()
        if not pointer_text:
            return False
        pointer = Path(pointer_text)
        if not pointer.is_absolute():
            pointer = checkpoint.parent / pointer
        return pointer.resolve() == checkpoint.resolve()
    except (FileNotFoundError, OSError, UnicodeError):
        return False


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON artifact must be an object: ' + str(path))
    return value


def _validate_sampler_record(
        report: Mapping[str, Any], source: str) -> None:
    exact = {
        'world_size': 5,
        'dataset_size': 1600,
        'update_count': 160,
        'duplicate_count': 0,
        'missing_count': 0,
        'global_batch_size_min': 10,
        'global_batch_size_max': 10,
        'local_batch_size_min': 2,
        'local_batch_size_max': 2,
        'update_count_multiple': 1,
        'max_query_area_budget': 50_000_000,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(
                f'{source} sampler {key} mismatch: '
                f'{report.get(key)!r} != {expected!r}')
    query_area = report.get('max_estimated_query_area')
    if (not isinstance(query_area, int)
            or isinstance(query_area, bool)
            or query_area > 50_000_000
            or query_area < 0):
        raise ValueError(source + ' sampler max_estimated_query_area invalid')
    checksum = report.get('coverage_checksum')
    if not isinstance(checksum, str) or not checksum:
        raise ValueError(source + ' sampler coverage_checksum missing')


def validate_sampler_pair(
        candidate_path: Path, control_path: Path) -> dict[str, Any]:
    """Require exact-cover sampler reports with the same global ordering."""
    candidate = _load_json(candidate_path)
    control = _load_json(control_path)
    _validate_sampler_record(candidate, 'candidate')
    _validate_sampler_record(control, 'control')
    candidate_checksum = candidate['coverage_checksum']
    control_checksum = control['coverage_checksum']
    if candidate_checksum != control_checksum:
        raise ValueError('candidate/control sampler checksum mismatch')
    return {
        'pass': True,
        'candidate_path': str(Path(candidate_path).resolve()),
        'control_path': str(Path(control_path).resolve()),
        'coverage_checksum': candidate_checksum,
        'checksums_match': True,
        'dataset_size': 1600,
        'world_size': 5,
        'updates_per_epoch': 160,
        'samples_per_rank': 320,
        'candidate': candidate,
        'control': control,
    }


def _finite(
        token: str, field: str, source: str, line_number: int) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(
            f'{source}:{line_number}: invalid {field}: {token!r}') from error
    if not math.isfinite(value):
        raise ValueError(
            f'{source}:{line_number}: non-finite {field}: {token!r}')
    return value


def _close(left: float, right: float, tolerance: float = 0.00055) -> bool:
    return abs(left - right) <= tolerance


def _finish_epoch(
        *,
        source: str,
        line_number: int,
        rows: Mapping[str, float],
        gts: Mapping[str, int],
        table_map: float,
        summary_match: re.Match) -> dict[str, Any]:
    epoch = int(summary_match.group('epoch'))
    done = int(summary_match.group('done'))
    total = int(summary_match.group('total'))
    if total <= 0 or done != total:
        raise ValueError(
            f'{source}:{line_number}: incomplete validation progress')
    exact_map = _finite(
        summary_match.group('map'), 'dota/mAP', source, line_number)
    ap50 = _finite(
        summary_match.group('ap50'), 'dota/AP50', source, line_number)
    supported = [name for name in CLASSES if gts[name] > 0]
    if not supported:
        raise ValueError(source + ': all classes have zero ground truth')
    class_mean = sum(rows[name] for name in supported) / len(supported)
    if not _close(class_mean, table_map):
        raise ValueError(source + ': class AP mean does not match table mAP')
    if not _close(exact_map, table_map):
        raise ValueError(source + ': summary mAP does not match table mAP')
    if not _close(ap50, table_map):
        raise ValueError(source + ': AP50 does not match table mAP')
    return {
        'epoch': epoch,
        'progress': [done, total],
        'mAP': exact_map,
        'AP50': ap50,
        'novel4': sum(rows[name] for name in NOVEL_CLASSES) / 4,
        'base14': sum(rows[name] for name in BASE_CLASSES) / 14,
        'map_class_count': len(supported),
        'class_ap': {name: rows[name] for name in CLASSES},
        'class_gts': {name: gts[name] for name in CLASSES},
    }


def parse_epoch_records(text: str, *, source: str) -> list[dict[str, Any]]:
    """Parse complete ordered 18-class table/summary pairs."""
    records = []
    state = 'idle'
    rows: dict[str, float] = {}
    gts: dict[str, int] = {}
    next_class = 0
    table_map = 0.0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = ANSI.sub('', raw_line).strip()
        summary_match = METRIC.search(line)
        if state == 'idle':
            if HEADER.match(line):
                state = 'rows'
                rows = {}
                gts = {}
                next_class = 0
                continue
            if summary_match is not None:
                raise ValueError(
                    f'{source}:{line_number}: summary without class table')
            continue
        if state == 'rows':
            if not line or BORDER.match(line):
                continue
            map_match = MAP_ROW.match(line)
            if map_match is not None:
                if next_class != len(CLASSES):
                    raise ValueError(
                        f'{source}:{line_number}: incomplete class table: '
                        f'expected 18 rows, got {next_class}')
                table_map = _finite(
                    map_match.group('map'),
                    'table mAP',
                    source,
                    line_number)
                state = 'summary'
                continue
            row_match = CLASS_ROW.match(line)
            if row_match is None:
                raise ValueError(
                    f'{source}:{line_number}: malformed class table')
            if next_class >= len(CLASSES):
                raise ValueError(source + ': class table has too many rows')
            name = row_match.group('name').strip()
            expected = CLASSES[next_class]
            if name != expected:
                raise ValueError(
                    f'{source}:{line_number}: class order mismatch: '
                    f'{name!r} != {expected!r}')
            _finite(
                row_match.group('recall'),
                name + ' recall',
                source,
                line_number)
            rows[name] = _finite(
                row_match.group('ap'), name + ' AP', source, line_number)
            gts[name] = int(row_match.group('gts'))
            next_class += 1
            continue
        if state == 'summary':
            if not line or BORDER.match(line):
                continue
            if summary_match is None:
                raise ValueError(
                    f'{source}:{line_number}: table and summary not adjacent')
            record = _finish_epoch(
                source=source,
                line_number=line_number,
                rows=rows,
                gts=gts,
                table_map=table_map,
                summary_match=summary_match)
            if records and record['epoch'] <= records[-1]['epoch']:
                raise ValueError(source + ': epochs duplicated or out of order')
            records.append(record)
            state = 'idle'
    if state != 'idle':
        raise ValueError(source + ': incomplete class table at end of log')
    if not records:
        raise ValueError(source + ': no complete class table-summary pair')
    return records


def evaluate_frozen_gate(
        candidate: Mapping[str, Any],
        control: Mapping[str, Any],
        integrity: Mapping[str, bool]) -> dict[str, Any]:
    """Evaluate only the frozen epoch-12 D12 promotion conjunction."""
    for role, metric in (('candidate', candidate), ('control', control)):
        if int(metric.get('epoch', -1)) != TARGET_EPOCH:
            raise ValueError(role + ' metric is not epoch 12')
        for name in ('mAP', 'AP50', 'novel4', 'base14'):
            value = float(metric[name])
            if not math.isfinite(value):
                raise ValueError(role + ' metric must be finite: ' + name)
    missing_integrity = REQUIRED_INTEGRITY - set(integrity)
    if missing_integrity:
        raise ValueError(
            'missing integrity checks: ' + ', '.join(sorted(missing_integrity)))

    delta = {
        name: float(candidate[name]) - float(control[name])
        for name in ('mAP', 'AP50', 'novel4', 'base14')
    }
    check_specs = {
        'candidate_AP50_absolute': (
            float(candidate['AP50']),
            THRESHOLDS['candidate_AP50_absolute_min']),
        'candidate_AP50_delta': (
            delta['AP50'],
            THRESHOLDS['candidate_AP50_control_delta_min']),
        'candidate_mAP_delta': (
            delta['mAP'],
            THRESHOLDS['candidate_mAP_control_delta_min']),
        'candidate_novel4_absolute': (
            float(candidate['novel4']),
            THRESHOLDS['candidate_novel4_absolute_min']),
        'candidate_novel4_control': (
            delta['novel4'],
            THRESHOLDS['candidate_novel4_control_delta_min']),
        'candidate_base14_absolute': (
            float(candidate['base14']),
            THRESHOLDS['candidate_base14_absolute_min']),
        'candidate_base14_control': (
            delta['base14'],
            THRESHOLDS['candidate_base14_control_delta_min']),
    }
    checks = {}
    for name, (value, threshold) in check_specs.items():
        checks[name] = {
            'value': value,
            'threshold': threshold,
            'pass': value + 1e-12 >= threshold,
        }
    for name in sorted(REQUIRED_INTEGRITY):
        checks['integrity_' + name] = {
            'value': bool(integrity[name]),
            'threshold': True,
            'pass': bool(integrity[name]),
        }
    passed = all(check['pass'] for check in checks.values())
    return {
        'pass': passed,
        'decision': 'promote' if passed else 'discard',
        'thresholds': dict(THRESHOLDS),
        'delta': delta,
        'checks': checks,
    }


def fatal_log_hits(text: str) -> list[dict[str, Any]]:
    """Return precise fatal signatures without matching words like information."""
    hits = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for signature, pattern in FATAL_PATTERNS:
            if pattern.search(line):
                hits.append({
                    'signature': signature,
                    'line': line_number,
                    'text': line[-500:],
                })
    return hits


def _audit_spec(
        *,
        name: str,
        python: Path,
        script: Path,
        config: Path,
        checkpoint: Path,
        output: Path,
        physical_gpu: int) -> dict[str, Any]:
    temporary = output.with_name(
        f'{output.name}.subprocess.{os.getpid()}')
    return {
        'name': name,
        'physical_gpu': physical_gpu,
        'output': output,
        'temporary': temporary,
        'command': [
            str(python),
            str(script),
            str(config),
            '--checkpoint',
            str(checkpoint),
            '--output',
            str(temporary),
            '--device',
            'cuda:0',
        ],
        'environment': {
            'CUDA_VISIBLE_DEVICES': str(physical_gpu),
            'PYTHONNOUSERSITE': '1',
        },
    }


def build_audit_specs(
        *,
        python: Path,
        strict_script: Path,
        open_vocabulary_script: Path,
        candidate_config: Path,
        control_config: Path,
        candidate_checkpoint: Path,
        control_checkpoint: Path,
        output_root: Path) -> list[dict[str, Any]]:
    """Build four sequential exact-epoch audits on physical GPUs 0 and 5."""
    output_root = Path(output_root)
    return [
        _audit_spec(
            name='candidate_strict',
            python=python,
            script=strict_script,
            config=candidate_config,
            checkpoint=candidate_checkpoint,
            output=output_root / 'candidate_strict.json',
            physical_gpu=0),
        _audit_spec(
            name='candidate_open_vocabulary',
            python=python,
            script=open_vocabulary_script,
            config=candidate_config,
            checkpoint=candidate_checkpoint,
            output=output_root / 'candidate_open_vocabulary.json',
            physical_gpu=0),
        _audit_spec(
            name='control_strict',
            python=python,
            script=strict_script,
            config=control_config,
            checkpoint=control_checkpoint,
            output=output_root / 'control_strict.json',
            physical_gpu=5),
        _audit_spec(
            name='control_open_vocabulary',
            python=python,
            script=open_vocabulary_script,
            config=control_config,
            checkpoint=control_checkpoint,
            output=output_root / 'control_open_vocabulary.json',
            physical_gpu=5),
    ]


def _query_gpu_compute_pids(index: int) -> list[int]:
    completed = subprocess.run(
        [
            'nvidia-smi',
            f'--id={index}',
            '--query-compute-apps=pid',
            '--format=csv,noheader,nounits',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15)
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _gpus_idle() -> bool:
    try:
        return all(
            not _query_gpu_compute_pids(index)
            for index in IDLE_GPU_INDICES)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def _read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def _stage0_passed(path: Path) -> bool:
    try:
        return _load_json(path).get('pass') is True
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False


def _sampler_record_complete(path: Path) -> bool:
    try:
        report = _load_json(path)
        _validate_sampler_record(report, str(path))
        return True
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False


def _sampler_checksums_match(
        candidate_path: Path, control_path: Path) -> bool:
    try:
        return (
            _load_json(candidate_path).get('coverage_checksum')
            == _load_json(control_path).get('coverage_checksum')
            and _sampler_record_complete(candidate_path)
            and _sampler_record_complete(control_path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False


def _run_audit_spec(spec: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    output = Path(spec['output'])
    temporary = Path(spec['temporary'])
    output.parent.mkdir(parents=True, exist_ok=True)
    collisions = []
    if os.path.lexists(output):
        collisions.append(str(output))
    collisions.extend(
        str(path) for path in output.parent.glob(output.name + '.pending.*'))
    if os.path.lexists(temporary):
        collisions.append(str(temporary))
    if collisions:
        raise FileExistsError(
            'output collision: ' + ', '.join(sorted(collisions)))
    environment = os.environ.copy()
    environment.update(spec['environment'])
    environment['PYTHONPATH'] = str(repo_root)
    completed = subprocess.run(
        list(spec['command']),
        cwd=str(repo_root),
        env=environment,
        check=False)
    if not temporary.is_file():
        raise RuntimeError(
            f"{spec['name']} produced no JSON (returncode={completed.returncode})")
    report = _load_json(temporary)
    publish_json_no_clobber(report, output)
    temporary.unlink()
    return {
        'pass': report.get('pass') is True and completed.returncode == 0,
        'returncode': int(completed.returncode),
        'physical_gpu': int(spec['physical_gpu']),
        'path': str(output.resolve()),
        'sha256': sha256_file(output),
        'report': report,
    }


def _epoch12(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [dict(record) for record in records
               if int(record['epoch']) == TARGET_EPOCH]
    if len(matches) != 1:
        raise ValueError(
            f'expected exactly one epoch-12 record, got {len(matches)}')
    return matches[0]


def _collect_status(args: argparse.Namespace) -> dict[str, Any]:
    candidate_console = args.candidate_work_dir / 'console.log'
    control_console = args.control_work_dir / 'console.log'
    candidate_text = _read_text(candidate_console)
    control_text = _read_text(control_console)
    candidate_metric = extract_complete_epoch_metric(
        candidate_text, args.target_epoch)
    control_metric = extract_complete_epoch_metric(
        control_text, args.target_epoch)
    candidate_checkpoint = (
        args.candidate_work_dir / f'epoch_{args.target_epoch}.pth')
    control_checkpoint = (
        args.control_work_dir / f'epoch_{args.target_epoch}.pth')
    fatal_hits = {
        'candidate': fatal_log_hits(candidate_text),
        'control': fatal_log_hits(control_text),
    }
    evidence = {
        'candidate_training_alive': exact_training_process_exists(
            args.candidate_config),
        'control_training_alive': exact_training_process_exists(
            args.control_config),
        'candidate_metric_complete': candidate_metric is not None,
        'control_metric_complete': control_metric is not None,
        'candidate_checkpoint_published': checkpoint_published(
            candidate_checkpoint,
            minimum_bytes=args.minimum_checkpoint_bytes),
        'control_checkpoint_published': checkpoint_published(
            control_checkpoint,
            minimum_bytes=args.minimum_checkpoint_bytes),
        'candidate_sampler_complete': _sampler_record_complete(
            args.candidate_sampler_audit),
        'control_sampler_complete': _sampler_record_complete(
            args.control_sampler_audit),
        'sampler_checksums_match': _sampler_checksums_match(
            args.candidate_sampler_audit,
            args.control_sampler_audit),
        'gpus_idle': _gpus_idle(),
        'stage0_passed': _stage0_passed(args.stage0_report),
        'fatal_free': not any(fatal_hits.values()),
    }
    return {
        **pair_completion_status(**evidence),
        **evidence,
        'candidate_metric': candidate_metric,
        'control_metric': control_metric,
        'fatal_hits': fatal_hits,
        'candidate_checkpoint': str(candidate_checkpoint),
        'control_checkpoint': str(control_checkpoint),
    }


def run_postrun(args: argparse.Namespace) -> dict[str, Any]:
    """Wait for the exact pair, run integrity audits, and publish the gate."""
    output = Path(args.output).expanduser().resolve()
    if os.path.lexists(output) or list(
            output.parent.glob(output.name + '.pending.*')):
        raise FileExistsError('output collision: ' + str(output))
    if int(args.target_epoch) != TARGET_EPOCH:
        raise ValueError('target epoch must be exactly 12')
    if float(args.poll_seconds) <= 0:
        raise ValueError('poll seconds must be positive')
    if Path(args.candidate_config).resolve() == (
            Path(args.control_config).resolve()):
        raise ValueError('candidate/control configs must be distinct')
    for path in (
            args.candidate_config,
            args.control_config,
            args.stage0_report,
            args.strict_script,
            args.open_vocabulary_script):
        if not Path(path).is_file():
            raise ValueError('required input is missing: ' + str(path))

    seen_alive = {'candidate': False, 'control': False}
    previous_waiting = None
    while True:
        status = _collect_status(args)
        seen_alive['candidate'] |= status['candidate_training_alive']
        seen_alive['control'] |= status['control_training_alive']
        waiting = tuple(status['waiting_for'])
        if waiting != previous_waiting:
            print(json.dumps({
                'event': 'ready' if status['ready'] else 'waiting',
                'waiting_for': list(waiting),
                'candidate_metric': status['candidate_metric'],
                'control_metric': status['control_metric'],
            }, sort_keys=True), flush=True)
            previous_waiting = waiting
        if not status['fatal_free']:
            raise RuntimeError(
                'fatal training signature: '
                + json.dumps(status['fatal_hits'], sort_keys=True))
        for role in ('candidate', 'control'):
            if (seen_alive[role]
                    and not status[role + '_training_alive']
                    and not status[role + '_metric_complete']):
                raise RuntimeError(role + ' exited before epoch-12 metric')
        if status['ready']:
            break
        time.sleep(float(args.poll_seconds))

    candidate_console = args.candidate_work_dir / 'console.log'
    control_console = args.control_work_dir / 'console.log'
    candidate_records = parse_epoch_records(
        _read_text(candidate_console), source=str(candidate_console))
    control_records = parse_epoch_records(
        _read_text(control_console), source=str(control_console))
    candidate_metric = _epoch12(candidate_records)
    control_metric = _epoch12(control_records)
    sampler_report = validate_sampler_pair(
        args.candidate_sampler_audit,
        args.control_sampler_audit)
    stage0_report = _load_json(args.stage0_report)
    if stage0_report.get('pass') is not True:
        raise ValueError('Stage-0 report is not passing')

    candidate_checkpoint = (
        args.candidate_work_dir / 'epoch_12.pth').resolve()
    control_checkpoint = (
        args.control_work_dir / 'epoch_12.pth').resolve()
    audit_root = output.with_suffix('')
    specs = build_audit_specs(
        python=Path(sys.executable),
        strict_script=args.strict_script,
        open_vocabulary_script=args.open_vocabulary_script,
        candidate_config=args.candidate_config,
        control_config=args.control_config,
        candidate_checkpoint=candidate_checkpoint,
        control_checkpoint=control_checkpoint,
        output_root=audit_root)
    audit_reports = {
        spec['name']: _run_audit_spec(spec, args.repo_root)
        for spec in specs
    }
    integrity = {
        'sampler': sampler_report['pass'],
        'stage0': stage0_report.get('pass') is True,
        'candidate_strict': audit_reports['candidate_strict']['pass'],
        'candidate_open_vocabulary': (
            audit_reports['candidate_open_vocabulary']['pass']),
        'control_strict': audit_reports['control_strict']['pass'],
        'control_open_vocabulary': (
            audit_reports['control_open_vocabulary']['pass']),
        'rotated_boxes': (
            len(candidate_metric['class_ap']) == len(CLASSES)
            and len(control_metric['class_ap']) == len(CLASSES)),
        'fatal_free': True,
    }
    gate = evaluate_frozen_gate(
        candidate_metric, control_metric, integrity)
    report = {
        'schema_version': 1,
        'decision_endpoint': 'epoch_12_only',
        'decision': gate['decision'],
        'pass': gate['pass'],
        'candidate': candidate_metric,
        'control': control_metric,
        'delta': gate['delta'],
        'thresholds': gate['thresholds'],
        'checks': gate['checks'],
        'integrity': integrity,
        'sampler': sampler_report,
        'stage0': {
            'path': str(Path(args.stage0_report).resolve()),
            'sha256': sha256_file(args.stage0_report),
            'pass': True,
        },
        'audits': audit_reports,
        'artifacts': {
            'candidate_console': str(candidate_console.resolve()),
            'control_console': str(control_console.resolve()),
            'candidate_checkpoint': {
                'path': str(candidate_checkpoint),
                'sha256': sha256_file(candidate_checkpoint),
            },
            'control_checkpoint': {
                'path': str(control_checkpoint),
                'sha256': sha256_file(control_checkpoint),
            },
        },
        'trajectory': {
            'candidate': candidate_records,
            'control': control_records,
            'selection': 'epoch 12 only; earlier epochs are evidence only',
        },
        'next_action': (
            'separate_full_run_review_required'
            if gate['decision'] == 'promote'
            else 'discard_no_rescue'),
        'prohibited_automatic_actions': [
            'full24_launch',
            'retuning',
            'source_scaling',
            'extra_epoch',
            'extra_seed',
            'best_checkpoint_selection',
            'checkpoint_cherry_pick',
            'D11_stacking',
            'loss_change',
            'prompt_change',
        ],
    }
    publish_json_no_clobber(report, output)
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=root)
    parser.add_argument('--candidate-config', required=True, type=Path)
    parser.add_argument('--control-config', required=True, type=Path)
    parser.add_argument('--candidate-work-dir', required=True, type=Path)
    parser.add_argument('--control-work-dir', required=True, type=Path)
    parser.add_argument(
        '--candidate-sampler-audit', required=True, type=Path)
    parser.add_argument('--control-sampler-audit', required=True, type=Path)
    parser.add_argument('--stage0-report', type=Path, default=DEFAULT_STAGE0)
    parser.add_argument('--target-epoch', type=int, default=TARGET_EPOCH)
    parser.add_argument('--poll-seconds', type=float, default=30.0)
    parser.add_argument(
        '--minimum-checkpoint-bytes',
        type=int,
        default=MINIMUM_CHECKPOINT_BYTES)
    parser.add_argument(
        '--strict-script',
        type=Path,
        default=root / 'projects/OVCapFlow/tools/audit_strict_inference.py')
    parser.add_argument(
        '--open-vocabulary-script',
        type=Path,
        default=root / 'projects/OVCapFlow/tools/audit_open_vocabulary.py')
    parser.add_argument('--output', required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_postrun(args)
    except (
            FileExistsError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
            ValueError,
            RuntimeError) as error:
        print('D12 postrun failed closed: ' + str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['decision'] == 'promote' else 1


if __name__ == '__main__':
    raise SystemExit(main())
