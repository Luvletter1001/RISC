"""Shared fail-closed runtime primitives for the frozen D13-N experiment."""

import copy
import hashlib
import json
import math
import os
import platform
import re
import socket
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from mmengine.config import Config

from projects.OVCapFlow.ov_capflow.no_replace import (
    _collision_paths, publish_json_noreplace)


PHYSICAL_GPUS = (5, 6, 7, 8, 9)
GPU_UUIDS = {
    5: 'GPU-6c98da7c-69bd-4c20-5094-ff3b60141515',
    6: 'GPU-b30e9f80-7925-6816-9014-d228784c5a12',
    7: 'GPU-d614452d-c145-3751-ce23-f9e21dd0b535',
    8: 'GPU-f8f2c2c6-4c20-237b-f6df-ba5b5408573e',
    9: 'GPU-ff760a2d-9a1b-38ee-06a0-8c284e002545',
}

CONTROL_PORT = 29842
CANDIDATE_PORT = 29841
RAW_CONTROL_PORT = 29843
RAW_CANDIDATE_PORT = 29844
STAGE0_PARENT_PORT = 29845
STAGE0_CONTROL_PORT = 29846
STAGE0_CANDIDATE_PORT = 29847

E24 = Path(
    'work_dirs/dotav2_cleanstart/'
    'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth')
CANONICAL_E24_DUMP = Path(
    'work_dirs/dotav2_cleanstart/'
    'eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl')
TRAIN_MANIFEST = Path(
    'work_dirs/dotav2_cleanstart/subsets/'
    'seed20260715_rare4x/train/manifest.json')
PROXY_MANIFEST = Path(
    'work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json')

_CONFIG_ROOT = Path('configs/ov_capflow/dotav2')
CONTROL_CONFIG = _CONFIG_ROOT / (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_control.py')
CANDIDATE_CONFIG = _CONFIG_ROOT / (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_candidate.py')
RAW_CONTROL_CONFIG = _CONFIG_ROOT / (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_control_raw13833.py')
RAW_CANDIDATE_CONFIG = _CONFIG_ROOT / (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_candidate_raw13833.py')

ANALYZER = Path('projects/OVCapFlow/tools/analyze_dotav2_q600_dump.py')
DIAGNOSTICS = Path('projects/OVCapFlow/tools/dotav2_q600_diagnostics.py')
VALIDATOR = Path('projects/OVCapFlow/tools/validate_dotav2_q600_dump.py')

MONITOR_STARTUP_TIMEOUT_SECONDS = 180
MONITOR_HEARTBEAT_MAX_AGE_SECONDS = 150

EXPECTED_ENVIRONMENT = {
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
    'driver': {
        'version': '580.173.02',
        'reported_cuda': '13.0',
    },
    'gpus': [
        {
            'physical_index': index,
            'name': 'NVIDIA A40',
            'uuid': GPU_UUIDS[index],
        }
        for index in PHYSICAL_GPUS
    ],
}

_INPUT_NAMES = (
    'parent', 'canonical_raw_dump', 'train_manifest', 'proxy_manifest',
    'control_config', 'candidate_config', 'raw_control_config',
    'raw_candidate_config', 'analyzer', 'diagnostics', 'validator',
)
EXPECTED_INPUT_SHA256 = {
    'parent': 'a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8',
    'canonical_raw_dump': (
        '39c1d6d3193cbc9e7f8d4e6daace9fc95df86c0d0fc7f1be7870cb13ee092c45'),
    'train_manifest': (
        '1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290'),
    'proxy_manifest': (
        'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e'),
    'control_config': (
        '8315accaa269cfc2e45cf19e4eb9aac06267eb3c9108dd02d32b6942bdec80ad'),
    'candidate_config': (
        '164a4bbc38124f351bac5bf939c48a5bd8f3e949bfddb06554922c9d8cf14965'),
    'raw_control_config': (
        'e97efa11010c4b963bea78619475469827067619a712d2156900658d562439c1'),
    'raw_candidate_config': (
        '948f3aa6aa6563d6d2abc46231227a83ff3920a5523543f68576c967a6695b77'),
    'analyzer': (
        'f4f10734948b714dcbae726142e859b3ddac793ee5ab5c78d9a3f3436d99fa23'),
    'diagnostics': (
        'bd29dc88650e0db545479c60adaa6a9ee64da1734d3a1756c7cd787c7269c92d'),
    'validator': (
        '50e63a33823a64cdf8df84b4bdacc32eb0a486f9ea79b9f92e4c7e99c61cc1b3'),
}
ANALYZER_SOURCE_COMMIT = 'abd860d157d562aa9d30422c05e41e246effc8a7'


def sha256_file(path):
    """Return the lowercase SHA256 of a regular file's bytes."""
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError('hash input must be a regular file: {}'.format(
            resolved))
    digest = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload):
    """Hash canonical sorted compact UTF-8 JSON."""
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False,
        separators=(',', ':'), sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_keys = list(left)
        right_keys = list(right)
        return (
            len(left_keys) == len(right_keys) and
            all(type(a) is type(b) and a == b
                for a, b in zip(left_keys, right_keys)) and
            all(_typed_equal(left[a], right[b])
                for a, b in zip(left_keys, right_keys)))
    if type(left) in (list, tuple):
        return (len(left) == len(right) and
                all(_typed_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _require_exact_dict(value, keys, source):
    if (type(value) is not dict or list(value) != list(keys) or
            any(type(key) is not str for key in value)):
        raise RuntimeError('{} schema/order is invalid'.format(source))
    return value


def _is_lower_sha256(value):
    return (type(value) is str and len(value) == 64 and
            all(character in '0123456789abcdef' for character in value))


def validate_frozen_environment(observed):
    """Validate a complete observation against predeclared frozen values."""
    try:
        _require_exact_dict(
            observed, ('interpreter', 'software', 'driver', 'gpus'),
            'environment')
        _require_exact_dict(
            observed['interpreter'], ('path', 'python_version'),
            'environment interpreter')
        _require_exact_dict(
            observed['software'],
            ('torch', 'torch_cuda', 'mmengine', 'mmdet', 'mmrotate'),
            'environment software')
        _require_exact_dict(
            observed['driver'], ('version', 'reported_cuda'),
            'environment driver')
        if type(observed['gpus']) is not list:
            raise RuntimeError('environment gpus must be an exact list')
        for gpu in observed['gpus']:
            _require_exact_dict(
                gpu, ('physical_index', 'name', 'uuid'),
                'environment GPU')
        if not _typed_equal(observed, EXPECTED_ENVIRONMENT):
            raise RuntimeError(
                'environment differs from the frozen D13-N authority')
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError('environment validation failed closed') from error
    return copy.deepcopy(observed)


def _run_text(argv, cwd=None):
    try:
        completed = subprocess.run(
            list(argv), cwd=cwd, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, encoding='utf-8')
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            'environment observation command failed') from error
    return completed.stdout


def _collect_environment_observation():
    """Collect the complete local software, driver, and physical-GPU view."""
    try:
        import mmdet
        import mmengine
        import mmrotate
        import torch

        executable = str(Path(sys.executable).resolve(strict=True))
        software = {
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'mmengine': mmengine.__version__,
            'mmdet': mmdet.__version__,
            'mmrotate': mmrotate.__version__,
        }
        summary = _run_text(('nvidia-smi', ))
        cuda_match = re.search(r'CUDA Version:\s*([0-9.]+)', summary)
        if cuda_match is None:
            raise RuntimeError(
                'environment NVIDIA CUDA version is unreadable')
        query = _run_text((
            'nvidia-smi',
            '--query-gpu=index,name,uuid,driver_version',
            '--format=csv,noheader,nounits',
        ))
        physical = {}
        for line in query.splitlines():
            fields = [field.strip() for field in line.split(',')]
            if len(fields) != 4 or not fields[0].isdigit():
                raise RuntimeError(
                    'environment NVIDIA GPU query is malformed')
            index = int(fields[0])
            if index in physical:
                raise RuntimeError(
                    'environment NVIDIA GPU indices are duplicated')
            physical[index] = {
                'physical_index': index,
                'name': fields[1],
                'uuid': fields[2],
                'driver_version': fields[3],
            }
        selected = []
        driver_versions = set()
        for index in PHYSICAL_GPUS:
            if index not in physical:
                raise RuntimeError(
                    'environment lacks physical GPU {}'.format(index))
            row = physical[index]
            driver_versions.add(row['driver_version'])
            selected.append({
                'physical_index': row['physical_index'],
                'name': row['name'],
                'uuid': row['uuid'],
            })
        if len(driver_versions) != 1:
            raise RuntimeError(
                'environment selected GPUs report different drivers')
        return {
            'interpreter': {
                'path': executable,
                'python_version': platform.python_version(),
            },
            'software': software,
            'driver': {
                'version': next(iter(driver_versions)),
                'reported_cuda': cuda_match.group(1),
            },
            'gpus': selected,
        }
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            'environment observation failed closed') from error


def _git_identity(repo):
    repo = Path(repo).resolve(strict=True)
    commit = _run_text(
        ('git', 'rev-parse', 'HEAD'), cwd=str(repo)).strip()
    dirty_output = _run_text(
        ('git', 'status', '--porcelain', '--untracked-files=no'),
        cwd=str(repo))
    if (len(commit) != 40 or
            any(character not in '0123456789abcdef' for character in commit)):
        raise RuntimeError('environment git commit identity is invalid')
    return {'commit': commit, 'tracked_dirty': bool(dirty_output.strip())}


def _resolve_input(repo, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo / path
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            'input path is missing or unreadable: {}'.format(path)) from error
    if not resolved.is_file():
        raise RuntimeError('input path is not a regular file: {}'.format(
            resolved))
    return resolved


def collect_environment_fingerprint(
        repo, control_config, candidate_config, raw_control_config,
        raw_candidate_config):
    """Validate the complete environment before reading frozen input bytes."""
    observed = _collect_environment_observation()
    validated = validate_frozen_environment(observed)

    try:
        repo = Path(repo).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError('input repository root is invalid') from error
    if not repo.is_dir():
        raise RuntimeError('input repository root is not a directory')
    paths = {
        'parent': repo / E24,
        'canonical_raw_dump': repo / CANONICAL_E24_DUMP,
        'train_manifest': repo / TRAIN_MANIFEST,
        'proxy_manifest': repo / PROXY_MANIFEST,
        'control_config': control_config,
        'candidate_config': candidate_config,
        'raw_control_config': raw_control_config,
        'raw_candidate_config': raw_candidate_config,
        'analyzer': repo / ANALYZER,
        'diagnostics': repo / DIAGNOSTICS,
        'validator': repo / VALIDATOR,
    }
    if (type(EXPECTED_INPUT_SHA256) is not dict or
            list(EXPECTED_INPUT_SHA256) != list(_INPUT_NAMES) or
            any(not _is_lower_sha256(value)
                for value in EXPECTED_INPUT_SHA256.values())):
        raise RuntimeError('input SHA256 authority is invalid')

    inputs = {}
    for name in _INPUT_NAMES:
        resolved = _resolve_input(repo, paths[name])
        try:
            digest = sha256_file(resolved)
        except Exception as error:
            raise RuntimeError(
                'input hash failed for {}'.format(name)) from error
        if digest != EXPECTED_INPUT_SHA256[name]:
            raise RuntimeError('input hash drift for {}'.format(name))
        inputs[name] = {'path': str(resolved), 'sha256': digest}

    fingerprint = {
        'schema': 'd13n-environment-v1',
        'interpreter': validated['interpreter'],
        'software': validated['software'],
        'driver': validated['driver'],
        'gpus': validated['gpus'],
        'git': _git_identity(repo),
        'inputs': inputs,
        'analyzer_source_commit': ANALYZER_SOURCE_COMMIT,
    }
    fingerprint['fingerprint_sha256'] = canonical_json_sha256(fingerprint)
    return fingerprint


def _validate_fingerprint(value):
    _require_exact_dict(
        value,
        ('schema', 'interpreter', 'software', 'driver', 'gpus', 'git',
         'inputs', 'analyzer_source_commit', 'fingerprint_sha256'),
        'fingerprint')
    if type(value['schema']) is not str or value['schema'] != (
            'd13n-environment-v1'):
        raise RuntimeError('fingerprint schema is invalid')
    validate_frozen_environment({
        'interpreter': value['interpreter'],
        'software': value['software'],
        'driver': value['driver'],
        'gpus': value['gpus'],
    })
    git = _require_exact_dict(
        value['git'], ('commit', 'tracked_dirty'), 'fingerprint git')
    if (type(git['commit']) is not str or len(git['commit']) != 40 or
            any(character not in '0123456789abcdef'
                for character in git['commit']) or
            type(git['tracked_dirty']) is not bool):
        raise RuntimeError('fingerprint git identity is invalid')
    inputs = _require_exact_dict(
        value['inputs'], _INPUT_NAMES, 'fingerprint inputs')
    for name, identity in inputs.items():
        _require_exact_dict(
            identity, ('path', 'sha256'),
            'fingerprint input {}'.format(name))
        if (type(identity['path']) is not str or not identity['path'] or
                not Path(identity['path']).is_absolute() or
                not _is_lower_sha256(identity['sha256'])):
            raise RuntimeError(
                'fingerprint input {} identity is invalid'.format(name))
    if (type(value['analyzer_source_commit']) is not str or
            value['analyzer_source_commit'] != ANALYZER_SOURCE_COMMIT):
        raise RuntimeError('fingerprint analyzer source commit is invalid')
    if not _is_lower_sha256(value['fingerprint_sha256']):
        raise RuntimeError('fingerprint self-hash is invalid')
    unhashed = {
        key: item for key, item in value.items()
        if key != 'fingerprint_sha256'
    }
    if value['fingerprint_sha256'] != canonical_json_sha256(unhashed):
        raise RuntimeError('fingerprint self-hash mismatch')


def assert_fingerprint_equal(expected, observed):
    """Require two independently self-valid fingerprints to match by type."""
    try:
        _validate_fingerprint(expected)
        _validate_fingerprint(observed)
        if not _typed_equal(expected, observed):
            raise RuntimeError('fingerprint content differs')
    except Exception as error:
        if isinstance(error, RuntimeError) and str(error).startswith(
                'fingerprint'):
            raise
        raise RuntimeError('fingerprint validation failed closed') from error


_PROCESS_FIELDS = ('pid', 'ppid', 'argv', 'environ', 'listening_ports')


def _is_train_script(value):
    return (type(value) is str and
            (value == 'tools/train.py' or value.endswith('/tools/train.py')))


def _train_config(argv):
    positions = [
        index for index, value in enumerate(argv)
        if _is_train_script(value)
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return None
    config = argv[positions[0] + 1]
    return config if type(config) is str and config else None


def _is_python_torchrun(argv):
    return (
        type(argv) is list and len(argv) >= 4 and
        all(type(value) is str for value in argv) and
        argv[0] == EXPECTED_ENVIRONMENT['interpreter']['path'] and
        argv[1:3] == ['-m', 'torch.distributed.run'] and
        _train_config(argv) is not None)


def _rtk_python_suffix(argv):
    if (type(argv) is not list or len(argv) < 5 or
            any(type(value) is not str for value in argv) or
            Path(argv[0]).name != 'rtk' or argv[1] != 'env'):
        return None
    interpreter = EXPECTED_ENVIRONMENT['interpreter']['path']
    positions = [index for index, value in enumerate(argv)
                 if value == interpreter]
    if len(positions) != 1:
        return None
    suffix = argv[positions[0]:]
    return suffix if _is_python_torchrun(suffix) else None


def _is_direct_worker(argv):
    if (type(argv) is not list or len(argv) < 4 or
            any(type(value) is not str for value in argv) or
            argv[0] != EXPECTED_ENVIRONMENT['interpreter']['path'] or
            _train_config(argv) is None or
            'torch.distributed.run' in argv):
        return False
    launcher_values = []
    for index, value in enumerate(argv):
        if value == '--launcher' and index + 1 < len(argv):
            launcher_values.append(argv[index + 1])
        elif value.startswith('--launcher='):
            launcher_values.append(value.split('=', 1)[1])
    return launcher_values == ['pytorch']


def _require_process_record(process):
    if (type(process) is not dict or
            list(process) != list(_PROCESS_FIELDS)):
        raise RuntimeError('topology process schema/order is invalid')
    pid = process['pid']
    ppid = process['ppid']
    argv = process['argv']
    environ = process['environ']
    ports = process['listening_ports']
    if (type(pid) is not int or pid <= 0 or
            type(ppid) is not int or ppid < 0):
        raise RuntimeError('topology process PID identity is invalid')
    if (type(argv) is not list or
            any(type(value) is not str for value in argv)):
        raise RuntimeError('topology process argv is unreadable')
    if environ is not None and (
            type(environ) is not dict or
            any(type(key) is not str or type(value) is not str
                for key, value in environ.items())):
        raise RuntimeError('topology process environment is invalid')
    if (type(ports) is not list or
            any(type(port) is not int or not 1 <= port <= 65535
                for port in ports) or len(set(ports)) != len(ports)):
        raise RuntimeError('topology process listener list is invalid')


def classify_process_snapshot(processes):
    """Classify synthetic/read process evidence without mutating it."""
    if type(processes) is not list:
        raise RuntimeError('topology process snapshot must be an exact list')
    seen_pids = set()
    for process in processes:
        _require_process_record(process)
        if process['pid'] in seen_pids:
            raise RuntimeError('topology process PIDs are duplicated')
        seen_pids.add(process['pid'])
    launcher_pids = {
        process['pid'] for process in processes
        if _is_python_torchrun(process['argv'])
    }
    snapshot = {
        'wrappers': [],
        'launchers': [],
        'workers': [],
        'others': [],
    }
    for process in processes:
        argv = process['argv']
        if _rtk_python_suffix(argv) is not None:
            bucket = 'wrappers'
        elif _is_python_torchrun(argv):
            bucket = 'launchers'
        elif (_is_direct_worker(argv) and
              process['ppid'] in launcher_pids):
            bucket = 'workers'
        else:
            bucket = 'others'
        snapshot[bucket].append(process)
    for values in snapshot.values():
        values.sort(key=lambda process: process['pid'])
    return snapshot


def _argv_option(argv, names):
    values = []
    for index, argument in enumerate(argv):
        for name in names:
            if argument == name:
                if index + 1 >= len(argv):
                    return None
                values.append(argv[index + 1])
            elif argument.startswith(name + '='):
                values.append(argument[len(name) + 1:])
    return values[0] if len(values) == 1 else None


def _require_worker_environment(worker, master_port):
    environment = worker['environ']
    if type(environment) is not dict:
        raise RuntimeError(
            'topology worker environment is missing or unreadable')
    required = {
        'LOCAL_WORLD_SIZE': '5',
        'WORLD_SIZE': '5',
        'MASTER_PORT': str(master_port),
        'CUDA_VISIBLE_DEVICES': '5,6,7,8,9',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_IB_DISABLE': '1',
    }
    for name, expected in required.items():
        if (name not in environment or
                type(environment[name]) is not str or
                environment[name] != expected):
            raise RuntimeError(
                'topology worker environment {} is invalid'.format(name))
    rank = environment.get('LOCAL_RANK')
    if type(rank) is not str or rank not in ('0', '1', '2', '3', '4'):
        raise RuntimeError('topology worker LOCAL_RANK is invalid')
    return int(rank)


def _has_selected_distributed_environment(process, master_port):
    environment = process['environ']
    if type(environment) is not dict:
        return False
    required = {
        'LOCAL_WORLD_SIZE': '5',
        'WORLD_SIZE': '5',
        'MASTER_PORT': str(master_port),
        'CUDA_VISIBLE_DEVICES': '5,6,7,8,9',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_IB_DISABLE': '1',
    }
    return all(environment.get(name) == value
               for name, value in required.items())


def assert_world5_topology(snapshot, master_port):
    """Require one wrapper, one real launcher, and five direct workers."""
    if type(master_port) is not int or not 1 <= master_port <= 65535:
        raise RuntimeError('topology master port is invalid')
    _require_exact_dict(
        snapshot, ('wrappers', 'launchers', 'workers', 'others'),
        'topology snapshot')
    if any(type(snapshot[key]) is not list for key in snapshot):
        raise RuntimeError('topology snapshot buckets must be exact lists')
    if (len(snapshot['wrappers']) != 1 or
            len(snapshot['launchers']) != 1 or
            len(snapshot['workers']) != 5):
        raise RuntimeError(
            'topology must contain one wrapper, one launcher, five workers')
    relevant = [
        *snapshot['wrappers'], *snapshot['launchers'], *snapshot['workers']]
    all_processes = [*relevant, *snapshot['others']]
    for process in all_processes:
        _require_process_record(process)
    if len({process['pid'] for process in all_processes}) != len(
            all_processes):
        raise RuntimeError('topology process PIDs are duplicated')
    wrapper = snapshot['wrappers'][0]
    launcher = snapshot['launchers'][0]
    workers = snapshot['workers']
    worker_pids = {worker['pid'] for worker in workers}
    worker_by_pid = {worker['pid']: worker for worker in workers}
    for process in snapshot['others']:
        parent_worker = worker_by_pid.get(process['ppid'])
        if parent_worker is not None:
            if (_typed_equal(process['argv'], parent_worker['argv']) and
                    _typed_equal(
                        process['environ'], parent_worker['environ']) and
                    process['listening_ports'] == []):
                continue
            raise RuntimeError(
                'topology dataloader descendant differs from worker')
        argv = process['argv']
        if (any(_is_train_script(argument) for argument in argv) or
                'torch.distributed.run' in argv or
                master_port in process['listening_ports'] or
                process['ppid'] in (wrapper['pid'], launcher['pid']) or
                (process['ppid'] not in worker_pids and
                 _has_selected_distributed_environment(
                     process, master_port))):
            raise RuntimeError(
                'topology other process resembles selected training')

    wrapper_suffix = _rtk_python_suffix(wrapper['argv'])
    if (wrapper_suffix is None or not _is_python_torchrun(launcher['argv']) or
            not _typed_equal(wrapper_suffix, launcher['argv']) or
            launcher['ppid'] != wrapper['pid']):
        raise RuntimeError(
            'topology wrapper/launcher ancestry or argv is invalid')
    if (_argv_option(
            launcher['argv'], ('--nproc_per_node', '--nproc-per-node')) !=
            '5' or
            _argv_option(
                launcher['argv'], ('--master_port', '--master-port')) !=
            str(master_port)):
        raise RuntimeError('topology launcher world/port argv is invalid')
    launcher_config = _train_config(launcher['argv'])
    if launcher_config is None:
        raise RuntimeError('topology launcher config is invalid')

    ranks = []
    for worker in workers:
        if (not _is_direct_worker(worker['argv']) or
                worker['ppid'] != launcher['pid'] or
                _train_config(worker['argv']) != launcher_config):
            raise RuntimeError(
                'topology direct worker ancestry/argv is invalid')
        ranks.append(_require_worker_environment(worker, master_port))
    if sorted(ranks) != [0, 1, 2, 3, 4]:
        raise RuntimeError('topology worker ranks must be exactly 0..4')

    if (wrapper['listening_ports'] != [] or
            launcher['listening_ports'] != [master_port] or
            any(worker['listening_ports'] != [] for worker in workers)):
        raise RuntimeError(
            'topology must have exactly one launcher master-port listener')
    return snapshot


def _remove_required(mapping, key, source):
    if type(mapping) is not dict or key not in mapping:
        raise RuntimeError(
            'normalized D13-N config lacks {}.{}'.format(source, key))
    del mapping[key]


def normalized_pair_config(config_path):
    """Load a D13-N config and remove only preregistered arm fields."""
    try:
        normalized = copy.deepcopy(Config.fromfile(config_path).to_dict())
    except Exception as error:
        raise RuntimeError('failed to load D13-N pair config') from error
    if type(normalized) is not dict:
        raise RuntimeError('resolved D13-N config must be a dictionary')

    for key in ('d13n_role', 'd13n_master_port', 'd13n_audit_path',
                'work_dir'):
        _remove_required(normalized, key, 'config')
    model = normalized.get('model')
    if type(model) is not dict or type(model.get('bbox_head')) is not dict:
        raise RuntimeError('resolved D13-N config lacks model.bbox_head')
    _remove_required(
        model['bbox_head'], 'existence_loss_weight', 'model.bbox_head')

    hooks = normalized.get('custom_hooks')
    if type(hooks) is not list or len(hooks) != 1:
        raise RuntimeError('resolved D13-N config custom_hooks is invalid')
    _remove_required(hooks[0], 'role', 'custom_hooks[0]')

    dataloader = normalized.get('train_dataloader')
    if (type(dataloader) is not dict or
            type(dataloader.get('batch_sampler')) is not dict):
        raise RuntimeError(
            'resolved D13-N config lacks train_dataloader.batch_sampler')
    _remove_required(
        dataloader['batch_sampler'], 'audit_path',
        'train_dataloader.batch_sampler')
    return normalized


def assert_paths_unoccupied(paths):
    """Fail without mutation when a final or same-final pending path exists."""
    if type(paths) not in (list, tuple):
        raise TypeError('paths must be an exact list or tuple')
    collisions = []
    for path in paths:
        if not isinstance(path, (str, Path)):
            raise TypeError('artifact paths must be strings or Paths')
        collisions.extend(_collision_paths(path))
    if collisions:
        raise FileExistsError(
            'artifact collision: ' + ', '.join(sorted(set(collisions))))


def assert_ports_free(ports):
    """Probe TCP bind availability without retaining a listener."""
    if type(ports) not in (list, tuple):
        raise TypeError('ports must be an exact list or tuple')
    if len(set(ports)) != len(ports):
        raise ValueError('ports must be unique')
    for port in ports:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('port must be an exact integer in [1, 65535]')
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(('0.0.0.0', port))
        except OSError as error:
            raise RuntimeError('port {} is not free'.format(port)) from error
        finally:
            probe.close()


_READY_KEYS = (
    'schema', 'pid', 'proc_start_ticks', 'proc_cmdline_sha256',
    'created_monotonic_ns', 'owner_root', 'owner_log', 'monitor_log',
    'failure_lock', 'stage0_report_path', 'stage0_report_sha256',
    'pre_run_commit', 'report_sha256',
)
_READY_PAYLOAD_KEYS = _READY_KEYS[1:-1]
_READY_EXPECTED_KEYS = (
    'owner_root', 'owner_log', 'monitor_log', 'failure_lock',
    'stage0_report_path', 'stage0_report_sha256', 'pre_run_commit',
)


def _absolute_path_text(value, source):
    if not isinstance(value, (str, Path)):
        raise TypeError('{} must be a string or Path'.format(source))
    return str(Path(value).expanduser().resolve(strict=False))


def _validate_ready_report(report):
    if type(report) is not dict or set(report) != set(_READY_KEYS):
        raise RuntimeError('monitor ready schema is invalid')
    if (type(report['schema']) is not str or
            report['schema'] != 'd13n-monitor-ready-v1'):
        raise RuntimeError('monitor ready schema value is invalid')
    for name in ('pid', 'proc_start_ticks'):
        if type(report[name]) is not int or report[name] <= 0:
            raise RuntimeError(
                'monitor ready {} is invalid'.format(name))
    if (type(report['created_monotonic_ns']) is not int or
            report['created_monotonic_ns'] < 0):
        raise RuntimeError(
            'monitor ready created_monotonic_ns is invalid')
    for name in (
            'proc_cmdline_sha256', 'stage0_report_sha256',
            'report_sha256'):
        if not _is_lower_sha256(report[name]):
            raise RuntimeError(
                'monitor ready {} is invalid'.format(name))
    for name in (
            'owner_root', 'owner_log', 'monitor_log', 'failure_lock',
            'stage0_report_path'):
        value = report[name]
        if (type(value) is not str or not value or
                not Path(value).is_absolute()):
            raise RuntimeError(
                'monitor ready {} path is invalid'.format(name))
    commit = report['pre_run_commit']
    if (type(commit) is not str or len(commit) != 40 or
            any(character not in '0123456789abcdef'
                for character in commit)):
        raise RuntimeError('monitor ready pre-run commit is invalid')
    unhashed = {
        key: value for key, value in report.items()
        if key != 'report_sha256'
    }
    if report['report_sha256'] != canonical_json_sha256(unhashed):
        raise RuntimeError('monitor ready self-hash mismatch')


def publish_monitor_ready(path, payload):
    """Normalize and publish the immutable monitor process identity."""
    if (type(payload) is not dict or
            list(payload) != list(_READY_PAYLOAD_KEYS)):
        raise RuntimeError('monitor ready payload schema/order is invalid')
    report = {'schema': 'd13n-monitor-ready-v1'}
    for name in ('pid', 'proc_start_ticks', 'proc_cmdline_sha256',
                 'created_monotonic_ns'):
        report[name] = payload[name]
    for name in (
            'owner_root', 'owner_log', 'monitor_log', 'failure_lock'):
        report[name] = _absolute_path_text(payload[name], name)
    report['stage0_report_path'] = _absolute_path_text(
        payload['stage0_report_path'], 'stage0_report_path')
    report['stage0_report_sha256'] = payload['stage0_report_sha256']
    report['pre_run_commit'] = payload['pre_run_commit']
    report['report_sha256'] = canonical_json_sha256(report)
    if list(report) != list(_READY_KEYS):
        raise RuntimeError('monitor ready internal schema/order is invalid')
    _validate_ready_report(report)
    publish_json_noreplace(path, report)
    return report


def _json_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: {}'.format(key))
        result[key] = value
    return result


def _load_one_line_json(path, source):
    try:
        raw = Path(path).resolve(strict=True).read_bytes()
        if not raw.endswith(b'\n') or raw.count(b'\n') != 1:
            raise ValueError('JSON evidence must be exactly one complete line')
        value = json.loads(
            raw[:-1].decode('utf-8'),
            object_pairs_hook=_json_without_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError('non-finite JSON token: {}'.format(token))))
    except Exception as error:
        raise RuntimeError('{} is unreadable or malformed'.format(
            source)) from error
    if type(value) is not dict:
        raise RuntimeError('{} must contain a JSON object'.format(source))
    return value


def _read_proc_start_ticks(proc_root, pid):
    try:
        raw = (Path(proc_root) / str(pid) / 'stat').read_text(
            encoding='utf-8')
        close = raw.rfind(')')
        if close < 0:
            raise ValueError('missing process comm terminator')
        prefix = raw[:close + 1]
        if not prefix.startswith(str(pid) + ' ('):
            raise ValueError('PID prefix mismatch')
        tail = raw[close + 1:].strip().split()
        if len(tail) <= 19 or not tail[19].isdigit():
            raise ValueError('missing start ticks')
        ticks = int(tail[19])
        if ticks <= 0:
            raise ValueError('nonpositive start ticks')
        return ticks
    except Exception as error:
        raise RuntimeError(
            'monitor ready process is missing or stat is unreadable') from error


def _read_proc_cmdline(proc_root, pid):
    try:
        raw = (Path(proc_root) / str(pid) / 'cmdline').read_bytes()
        if not raw or not raw.endswith(b'\0'):
            raise ValueError('cmdline is incomplete')
        encoded = raw[:-1].split(b'\0')
        if not encoded or any(not value for value in encoded):
            raise ValueError('cmdline has empty argv entries')
        argv = [os.fsdecode(value) for value in encoded]
        if any(type(value) is not str or not value for value in argv):
            raise ValueError('cmdline argv is invalid')
        return raw, argv
    except Exception as error:
        raise RuntimeError(
            'monitor ready process is missing or cmdline is unreadable') \
            from error


def _expected_monitor_argv(report, ready_path):
    root = Path(report['owner_root'])
    return [
        EXPECTED_ENVIRONMENT['interpreter']['path'],
        'projects/OVCapFlow/tools/monitor_d13n_run.py',
        '--owner-log', report['owner_log'],
        '--owner-root', report['owner_root'],
        '--stage0-report', report['stage0_report_path'],
        '--pre-run-commit', report['pre_run_commit'],
        '--output', report['monitor_log'],
        '--ready-output', str(Path(ready_path).resolve(strict=False)),
        '--terminal-output', str((root / 'monitor_terminal.json').resolve()),
        '--failure-lock', report['failure_lock'],
        '--poll-seconds', '60',
    ]


def validate_monitor_ready(path, expected, proc_root=Path('/proc')):
    """Revalidate immutable bindings and the exact live monitor process."""
    try:
        report = _load_one_line_json(path, 'monitor ready')
        _validate_ready_report(report)
        if (type(expected) is not dict or
                list(expected) != list(_READY_EXPECTED_KEYS)):
            raise RuntimeError(
                'monitor ready expected binding schema/order is invalid')
        for name in _READY_EXPECTED_KEYS:
            if (type(expected[name]) is not str or
                    type(report[name]) is not str or
                    report[name] != expected[name]):
                raise RuntimeError(
                    'monitor ready expected {} binding differs'.format(name))

        root = Path(report['owner_root'])
        if (Path(path).resolve(strict=False).parent != root or
                any(Path(report[name]).parent != root for name in (
                    'owner_log', 'monitor_log', 'failure_lock'))):
            raise RuntimeError(
                'monitor ready output-root bindings are invalid')
        try:
            stage0_hash = sha256_file(report['stage0_report_path'])
        except Exception as error:
            raise RuntimeError(
                'monitor ready Stage-0 report is unreadable') from error
        if stage0_hash != report['stage0_report_sha256']:
            raise RuntimeError('monitor ready Stage-0 report hash drift')

        ticks = _read_proc_start_ticks(proc_root, report['pid'])
        if ticks != report['proc_start_ticks']:
            raise RuntimeError('monitor ready PID start ticks differ')
        raw_cmdline, argv = _read_proc_cmdline(proc_root, report['pid'])
        if hashlib.sha256(raw_cmdline).hexdigest() != report[
                'proc_cmdline_sha256']:
            raise RuntimeError('monitor ready process cmdline hash differs')
        if not _typed_equal(argv, _expected_monitor_argv(report, path)):
            raise RuntimeError('monitor ready process argv identity differs')
        return report
    except RuntimeError as error:
        if str(error).startswith('monitor ready'):
            raise
        raise RuntimeError('monitor ready validation failed closed') from error
    except Exception as error:
        raise RuntimeError('monitor ready validation failed closed') from error


FAILURE_CLASSES = (
    'identity', 'topology', 'nonfinite_fatal', 'collision',
    'checkpoint_parent', 'checkpoint_buffer', 'optimizer_binding',
    'control_head', 'sampler', 'gpu_contract', 'input_runtime',
)
_FAILURE_KEYS = (
    'schema', 'status', 'sequence', 'timestamp', 'owner_state',
    'failure_class', 'owner_event', 'sample', 'evidence_paths', 'failures',
    'report_sha256',
)


def _validate_json_value(value, source):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('{} contains a non-finite float'.format(source))
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item, source)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError('{} keys must be exact strings'.format(source))
        for item in value.values():
            _validate_json_value(item, source)
        return
    raise TypeError('{} contains a noncanonical JSON type'.format(source))


def _optional_evidence(value, source):
    if value is None:
        return {'present': False, 'sha256': None}
    if type(value) is not dict:
        raise TypeError('{} must be an exact dict or None'.format(source))
    _validate_json_value(value, source)
    return {'present': True, 'sha256': canonical_json_sha256(value)}


def _failure_path_evidence(paths):
    if type(paths) is not list:
        raise TypeError('evidence_paths must be an exact list')
    evidence = {}
    for path in paths:
        resolved = _absolute_path_text(path, 'evidence path')
        if resolved in evidence:
            raise ValueError('evidence_paths contains duplicate paths')
        try:
            candidate = Path(resolved)
            if not candidate.is_file():
                raise FileNotFoundError(resolved)
            digest = sha256_file(candidate)
        except Exception:
            evidence[resolved] = {'exists': False, 'sha256': None}
        else:
            evidence[resolved] = {'exists': True, 'sha256': digest}
    return evidence


def _validate_optional_evidence(value, source):
    if type(value) is not dict or set(value) != {'present', 'sha256'}:
        raise RuntimeError(
            'integrity failure {} schema is invalid'.format(source))
    if type(value['present']) is not bool:
        raise RuntimeError(
            'integrity failure {} presence is invalid'.format(source))
    if value['present']:
        if not _is_lower_sha256(value['sha256']):
            raise RuntimeError(
                'integrity failure {} hash is invalid'.format(source))
    elif value['sha256'] is not None:
        raise RuntimeError(
            'integrity failure absent {} must have null hash'.format(source))


def _validate_failure_report(report):
    if type(report) is not dict or set(report) != set(_FAILURE_KEYS):
        raise RuntimeError('integrity failure report schema is invalid')
    if (type(report['schema']) is not str or
            report['schema'] != 'd13n-integrity-failure-v1' or
            type(report['status']) is not str or
            report['status'] != 'FAIL'):
        raise RuntimeError('integrity failure report identity is invalid')
    if type(report['sequence']) is not int or report['sequence'] < 0:
        raise RuntimeError('integrity failure sequence is invalid')
    if type(report['timestamp']) is not str or not report['timestamp']:
        raise RuntimeError('integrity failure timestamp is invalid')
    if type(report['owner_state']) is not str or not report['owner_state']:
        raise RuntimeError('integrity failure owner_state is invalid')
    if (type(report['failure_class']) is not str or
            report['failure_class'] not in FAILURE_CLASSES):
        raise RuntimeError('integrity failure failure_class is invalid')
    _validate_optional_evidence(report['owner_event'], 'owner_event')
    _validate_optional_evidence(report['sample'], 'sample')
    evidence = report['evidence_paths']
    if type(evidence) is not dict:
        raise RuntimeError(
            'integrity failure evidence_paths must be an exact dict')
    for path, identity in evidence.items():
        if type(path) is not str or not Path(path).is_absolute():
            raise RuntimeError(
                'integrity failure evidence path is invalid')
        if type(identity) is not dict or set(identity) != {
                'exists', 'sha256'}:
            raise RuntimeError(
                'integrity failure evidence identity is invalid')
        if type(identity['exists']) is not bool:
            raise RuntimeError(
                'integrity failure evidence existence is invalid')
        if identity['exists']:
            if not _is_lower_sha256(identity['sha256']):
                raise RuntimeError(
                    'integrity failure evidence hash is invalid')
        elif identity['sha256'] is not None:
            raise RuntimeError(
                'integrity failure absent evidence must have null hash')
    failures = report['failures']
    if (type(failures) is not list or not failures or
            any(type(failure) is not str or not failure
                for failure in failures)):
        raise RuntimeError('integrity failure failures list is invalid')
    if report['owner_state'] == 'UNKNOWN':
        if (report['sequence'] != 0 or
                report['failure_class'] != 'input_runtime' or
                report['owner_event'] != {
                    'present': False, 'sha256': None} or
                report['sample'] != {'present': False, 'sha256': None}):
            raise RuntimeError(
                'integrity failure UNKNOWN pre-sample lock is invalid')
    if not _is_lower_sha256(report['report_sha256']):
        raise RuntimeError('integrity failure report hash is invalid')
    unhashed = {
        key: value for key, value in report.items()
        if key != 'report_sha256'
    }
    if report['report_sha256'] != canonical_json_sha256(unhashed):
        raise RuntimeError('integrity failure report self-hash mismatch')


def _publish_json_direct_exclusive(path, payload):
    final = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    encoded = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False,
            separators=(',', ':'), sort_keys=True) + '\n').encode('utf-8')
    final.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    file_descriptor = os.open(str(final), flags, 0o600)
    directory_descriptor = None
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(file_descriptor, encoded[offset:])
            if written <= 0:
                raise OSError('exclusive evidence write made no progress')
            offset += written
        os.fsync(file_descriptor)
        directory_descriptor = os.open(str(final.parent), os.O_RDONLY)
        os.fsync(directory_descriptor)
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(file_descriptor)


def publish_integrity_failure_lock(
        path, *, sequence, timestamp, owner_state, failure_class,
        owner_event, sample, evidence_paths, failures):
    """Latch one immutable terminal integrity failure directly at final."""
    if type(failure_class) is not str:
        raise TypeError('failure_class must be an exact string')
    if failure_class not in FAILURE_CLASSES:
        raise ValueError('failure_class is outside the frozen allowlist')
    if type(sequence) is not int or sequence < 0:
        raise ValueError('sequence must be a nonnegative exact integer')
    if type(timestamp) is not str or not timestamp:
        raise TypeError('timestamp must be a nonempty exact string')
    if type(owner_state) is not str or not owner_state:
        raise TypeError('owner_state must be a nonempty exact string')
    if (type(failures) is not list or not failures or
            any(type(failure) is not str or not failure
                for failure in failures)):
        raise TypeError('failures must be a nonempty exact string list')
    report = {
        'schema': 'd13n-integrity-failure-v1',
        'status': 'FAIL',
        'sequence': sequence,
        'timestamp': timestamp,
        'owner_state': owner_state,
        'failure_class': failure_class,
        'owner_event': _optional_evidence(owner_event, 'owner_event'),
        'sample': _optional_evidence(sample, 'sample'),
        'evidence_paths': _failure_path_evidence(evidence_paths),
        'failures': list(failures),
    }
    report['report_sha256'] = canonical_json_sha256(report)
    if list(report) != list(_FAILURE_KEYS):
        raise RuntimeError(
            'integrity failure internal schema/order is invalid')
    _validate_failure_report(report)
    _publish_json_direct_exclusive(path, report)
    return report


def load_integrity_failure_lock(path):
    """Load and self-validate an immutable terminal failure lock."""
    report = _load_one_line_json(path, 'integrity failure lock')
    _validate_failure_report(report)
    return report


_SAMPLE_KEYS = (
    'schema', 'sequence', 'timestamp', 'monotonic_ns', 'monitor_pid',
    'proc_start_ticks', 'owner_event_sha256', 'previous_sample_sha256',
    'gpu', 'topology', 'console', 'training', 'd13n_telemetry', 'sampler',
    'checkpoint', 'fatal_patterns', 'status', 'sample_sha256',
)


def _validate_monitor_sample(value):
    if type(value) is not dict or set(value) != set(_SAMPLE_KEYS):
        raise RuntimeError('heartbeat sample schema is invalid')
    sample = {key: value[key] for key in _SAMPLE_KEYS}
    if (type(sample['schema']) is not str or
            sample['schema'] != 'd13n-monitor-sample-v1'):
        raise RuntimeError('heartbeat sample identity is invalid')
    for name in ('sequence', 'monitor_pid', 'proc_start_ticks'):
        if type(sample[name]) is not int or sample[name] <= 0:
            raise RuntimeError(
                'heartbeat sample {} is invalid'.format(name))
    if (type(sample['monotonic_ns']) is not int or
            sample['monotonic_ns'] < 0):
        raise RuntimeError('heartbeat sample monotonic_ns is invalid')
    if type(sample['timestamp']) is not str or not sample['timestamp']:
        raise RuntimeError('heartbeat sample timestamp is invalid')
    if not _is_lower_sha256(sample['owner_event_sha256']):
        raise RuntimeError('heartbeat owner-event hash is invalid')
    previous = sample['previous_sample_sha256']
    if previous is not None and not _is_lower_sha256(previous):
        raise RuntimeError('heartbeat previous-sample hash is invalid')
    if type(sample['gpu']) is not list:
        raise RuntimeError('heartbeat GPU evidence must be an exact list')
    for name in (
            'topology', 'console', 'training', 'd13n_telemetry', 'sampler',
            'checkpoint'):
        if type(sample[name]) is not dict:
            raise RuntimeError(
                'heartbeat {} evidence must be an exact dict'.format(name))
    if (type(sample['fatal_patterns']) is not list or
            any(type(pattern) is not str
                for pattern in sample['fatal_patterns'])):
        raise RuntimeError('heartbeat fatal_patterns is invalid')
    if type(sample['status']) is not str or not sample['status']:
        raise RuntimeError('heartbeat sample status is invalid')
    if not _is_lower_sha256(sample['sample_sha256']):
        raise RuntimeError('heartbeat sample self-hash is invalid')
    _validate_json_value({
        key: item for key, item in sample.items()
        if key != 'sample_sha256'
    }, 'heartbeat sample')
    unhashed = {
        key: item for key, item in sample.items()
        if key != 'sample_sha256'
    }
    if sample['sample_sha256'] != canonical_json_sha256(unhashed):
        raise RuntimeError('heartbeat sample self-hash mismatch')
    return sample


def _load_complete_jsonl_records(path):
    try:
        raw = Path(path).resolve(strict=True).read_bytes()
    except Exception as error:
        raise RuntimeError('heartbeat log is missing or unreadable') from error
    last_newline = raw.rfind(b'\n')
    if last_newline < 0:
        return []
    complete = raw[:last_newline + 1].split(b'\n')[:-1]
    records = []
    for index, line in enumerate(complete, start=1):
        if not line:
            raise RuntimeError(
                'heartbeat complete record {} is empty'.format(index))
        try:
            record = json.loads(
                line.decode('utf-8'),
                object_pairs_hook=_json_without_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(
                        'non-finite JSON token: {}'.format(token))))
        except Exception as error:
            raise RuntimeError(
                'heartbeat complete record {} is malformed'.format(
                    index)) from error
        records.append(record)
    return records


def validate_monitor_heartbeat(
        monitor_log, ready, owner_event_sha256, *, now_monotonic_ns):
    """Validate every complete append-only heartbeat and return the latest."""
    try:
        if type(ready) is not dict:
            raise RuntimeError('heartbeat ready identity must be a dict')
        monitor_pid = ready.get('pid')
        start_ticks = ready.get('proc_start_ticks')
        if (type(monitor_pid) is not int or monitor_pid <= 0 or
                type(start_ticks) is not int or start_ticks <= 0):
            raise RuntimeError('heartbeat ready process identity is invalid')
        if not _is_lower_sha256(owner_event_sha256):
            raise RuntimeError('heartbeat expected owner hash is invalid')
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise RuntimeError('heartbeat current monotonic time is invalid')

        raw_records = _load_complete_jsonl_records(monitor_log)
        if not raw_records:
            raise RuntimeError('heartbeat first complete sample is missing')
        previous = None
        previous_monotonic_ns = None
        latest = None
        for expected_sequence, raw_record in enumerate(
                raw_records, start=1):
            sample = _validate_monitor_sample(raw_record)
            if sample['sequence'] != expected_sequence:
                raise RuntimeError('heartbeat sample sequence is not exact')
            if sample['previous_sample_sha256'] != previous:
                raise RuntimeError('heartbeat previous-sample chain is broken')
            if (previous_monotonic_ns is not None and
                    sample['monotonic_ns'] <= previous_monotonic_ns):
                raise RuntimeError(
                    'heartbeat monotonic sample times are not increasing')
            if (sample['monitor_pid'] != monitor_pid or
                    sample['proc_start_ticks'] != start_ticks):
                raise RuntimeError('heartbeat monitor process identity drift')
            if sample['owner_event_sha256'] != owner_event_sha256:
                raise RuntimeError('heartbeat owner-event identity drift')
            previous = sample['sample_sha256']
            previous_monotonic_ns = sample['monotonic_ns']
            latest = sample

        age = now_monotonic_ns - latest['monotonic_ns']
        if age < 0:
            raise RuntimeError('heartbeat latest sample is from the future')
        if age > MONITOR_HEARTBEAT_MAX_AGE_SECONDS * 1_000_000_000:
            raise RuntimeError('heartbeat latest sample is too old')
        return latest
    except RuntimeError as error:
        if str(error).startswith('heartbeat'):
            raise
        raise RuntimeError('heartbeat validation failed closed') from error
    except Exception as error:
        raise RuntimeError('heartbeat validation failed closed') from error


class CheckpointHashCache:
    """Cache trusted-checkpoint SHA256 by path, size, and mtime_ns.

    Cache records additionally bind the opened file's device, inode, and
    ctime. Checkpoint loading still uses pickle through ``torch.load`` and is
    only valid for trusted checkpoints; these consistency checks are not an
    untrusted-file security boundary.
    """

    def __init__(self):
        self._sha256_by_identity = {}

    @staticmethod
    def _resolve(path):
        try:
            return Path(path).expanduser().resolve(strict=True)
        except FileNotFoundError:
            raise
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                'checkpoint hash path resolution failed') from error

    @staticmethod
    def _fstat_descriptor(descriptor, source):
        try:
            observed = os.fstat(descriptor)
        except OSError as error:
            raise RuntimeError(
                'checkpoint {} fstat failed'.format(source)) from error
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError('checkpoint hash input is not a regular file')
        return observed

    @classmethod
    def _fstat(cls, stream, source):
        return cls._fstat_descriptor(stream.fileno(), source)

    @staticmethod
    def _authority(observed):
        return (observed.st_dev, observed.st_ino, observed.st_ctime_ns)

    @staticmethod
    def _path_authority(observed):
        return (
            observed.st_mode, observed.st_dev, observed.st_ino,
            observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns,
        )

    @staticmethod
    def _assert_same_file(before, after, source):
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, field) != getattr(after, field)
               for field in fields):
            raise RuntimeError(
                'checkpoint {} identity changed during read'.format(source))

    @classmethod
    def _assert_path_authority(
            cls, requested, resolved, requested_authority, descriptor_stat,
            source):
        try:
            current_requested = os.lstat(str(requested))
            current_resolved = requested.resolve(strict=True)
            path_stat = os.stat(str(resolved))
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                'checkpoint {} path authority is unavailable'.format(
                    source)) from error
        if (cls._path_authority(current_requested) != requested_authority or
                current_resolved != resolved):
            raise RuntimeError(
                'checkpoint {} path authority changed'.format(source))
        cls._assert_same_file(
            descriptor_stat, path_stat, '{} path'.format(source))

    def _hash_open_file(
            self, resolved, requested, requested_authority, before, stream):
        identity = (str(resolved), before.st_size, before.st_mtime_ns)
        authority = self._authority(before)
        cached_entry = self._sha256_by_identity.get(identity)
        cached = (cached_entry is not None and
                  cached_entry[0] == authority)
        digest = cached_entry[1] if cached else None
        if not cached:
            try:
                stream.seek(0)
                digest = _sha256_stream(stream)
            except Exception as error:
                raise RuntimeError('checkpoint hash read failed') from error
            after = self._fstat(stream, 'hash post-read')
            self._assert_same_file(before, after, 'hash')
            self._assert_path_authority(
                requested, resolved, requested_authority, after,
                'hash post-read')
            if not _is_lower_sha256(digest):
                raise RuntimeError('checkpoint hash digest is invalid')
            self._sha256_by_identity[identity] = (authority, digest)
        record = {
            'path': identity[0],
            'size': identity[1],
            'mtime_ns': identity[2],
            'sha256': digest,
            'sha_cached': cached,
        }
        return record, before

    def _open(self, path):
        try:
            requested = Path(os.path.abspath(
                os.fspath(Path(path).expanduser())))
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(
                'checkpoint hash input path is invalid') from error
        resolved = self._resolve(path)
        descriptor = None
        try:
            flags = (os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) |
                     getattr(os, 'O_CLOEXEC', 0))
            descriptor = os.open(str(resolved), flags)
            before = self._fstat_descriptor(descriptor, 'hash open')
            requested_stat = os.lstat(str(requested))
            requested_authority = self._path_authority(requested_stat)
            self._assert_path_authority(
                requested, resolved, requested_authority, before,
                'hash open')
            try:
                stream = os.fdopen(descriptor, 'rb')
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    'checkpoint hash stream open failed') from error
            descriptor = None
            try:
                seekable = stream.seekable()
            except (OSError, ValueError) as error:
                stream.close()
                raise RuntimeError(
                    'checkpoint hash stream check failed') from error
            if not seekable:
                stream.close()
                raise RuntimeError(
                    'checkpoint hash stream must be seekable')
            return (
                resolved, requested, requested_authority, before, stream)
        except OSError as error:
            raise RuntimeError('checkpoint hash open failed') from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def hash_file(self, path):
        (resolved, requested, requested_authority,
         before, stream) = self._open(path)
        with stream:
            record, _ = self._hash_open_file(
                resolved, requested, requested_authority, before, stream)
        return record

    def _hash_and_load(self, path):
        (resolved, requested, requested_authority,
         before, stream) = self._open(path)
        with stream:
            record, before = self._hash_open_file(
                resolved, requested, requested_authority, before, stream)
            try:
                stream.seek(0)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    'checkpoint audit seek failed') from error
            payload = _load_checkpoint_cpu(stream)
            after = self._fstat(stream, 'audit post-load')
            self._assert_same_file(before, after, 'audit load')
            self._assert_path_authority(
                requested, resolved, requested_authority, after,
                'audit post-load')
        return record, payload


def _sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


_ADAPTER_NAMES = (
    'bbox_head.existence_residual.weight',
    'bbox_head.existence_residual.bias',
)
_INTEGRITY_KEYS = (
    'schema', 'role', 'epoch', 'iter',
    'optimizer_parameter_names_by_group', 'optimizer_class',
    'live_optimizer_group_sizes', 'serialized_optimizer_group_sizes',
    'serialized_optimizer_parameter_ids_by_group',
    'initialized_optimizer_state_slots', 'parent_state_sha256',
    'adapter_state_sha256', 'config_sha256', 'integrity_sha256',
)


def _canonical_integrity_sha256(value):
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True,
        separators=(',', ':'), sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _hash_named_tensor_state(state, source):
    if not isinstance(state, Mapping):
        raise RuntimeError('{} must be a mapping'.format(source))
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if type(name) is not str or type(value) is not torch.Tensor:
            raise RuntimeError(
                '{} must map exact string names to Tensors'.format(source))
        try:
            tensor = value.detach().cpu().contiguous()
            header = json.dumps({
                'name': name,
                'dtype': str(tensor.dtype),
                'shape': list(tensor.shape),
            }, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True).encode('utf-8')
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        except Exception as error:
            raise RuntimeError(
                '{} tensor hashing failed'.format(source)) from error
        digest.update(len(header).to_bytes(8, byteorder='big'))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, byteorder='big'))
        digest.update(raw)
    return digest.hexdigest()


def _load_checkpoint_cpu(source):
    """Load one trusted checkpoint from its already-open authority file.

    ``torch.load`` uses pickle. The subsequent schema and identity validation
    detects experiment drift, but does not make an untrusted checkpoint safe.
    """
    try:
        checkpoint = torch.load(source, map_location='cpu')
    except Exception as error:
        raise RuntimeError('checkpoint audit CPU load failed') from error
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError('checkpoint audit root must be a mapping')
    for name in ('meta', 'state_dict', 'optimizer'):
        if name not in checkpoint:
            raise RuntimeError(
                'checkpoint audit lacks required {}'.format(name))
    if not isinstance(checkpoint['meta'], Mapping):
        raise RuntimeError('checkpoint audit meta must be a mapping')
    if not isinstance(checkpoint['state_dict'], Mapping):
        raise RuntimeError('checkpoint audit state_dict must be a mapping')
    return checkpoint


def _stage0_model_authority(stage0_report, state_dict):
    if (type(stage0_report) is not dict or
            stage0_report.get('schema') != 'd13n-stage0-v1' or
            stage0_report.get('status') != 'PASS'):
        raise RuntimeError('checkpoint audit Stage-0 report is invalid')
    evidence = stage0_report.get('model_evidence')
    evidence_keys = (
        'parent_parameter_names', 'parent_buffer_names',
        'parent_parameter_sha256', 'parent_buffer_sha256',
    )
    if type(evidence) is not dict or list(evidence) != list(evidence_keys):
        raise RuntimeError(
            'checkpoint audit Stage-0 model authority schema is invalid')
    parameter_names = evidence['parent_parameter_names']
    buffer_names = evidence['parent_buffer_names']
    if (type(parameter_names) is not list or
            type(buffer_names) is not list or
            any(type(name) is not str or not name
                for name in [*parameter_names, *buffer_names]) or
            len(set([*parameter_names, *buffer_names])) !=
            len(parameter_names) + len(buffer_names)):
        raise RuntimeError(
            'checkpoint audit Stage-0 parent names are invalid')
    state_names = list(state_dict)
    if any(type(name) is not str for name in state_names):
        raise RuntimeError('checkpoint audit model state names are invalid')
    expected_names = [*parameter_names, *buffer_names, *_ADAPTER_NAMES]
    if (len(state_names) != len(expected_names) or
            set(state_names) != set(expected_names)):
        raise RuntimeError(
            'checkpoint audit model state differs from Stage-0 authority')
    parameters = {name: state_dict[name] for name in parameter_names}
    buffers = {name: state_dict[name] for name in buffer_names}
    adapter = {name: state_dict[name] for name in _ADAPTER_NAMES}
    parent = {
        name: state_dict[name]
        for name in state_names if name not in _ADAPTER_NAMES
    }
    parameter_hash = _hash_named_tensor_state(
        parameters, 'checkpoint parent parameters')
    buffer_hash = _hash_named_tensor_state(
        buffers, 'checkpoint parent buffers')
    adapter_hash = _hash_named_tensor_state(
        adapter, 'checkpoint adapter state')
    parent_hash = _hash_named_tensor_state(
        parent, 'checkpoint parent state')
    if (not _is_lower_sha256(evidence['parent_parameter_sha256']) or
            evidence['parent_parameter_sha256'] != parameter_hash or
            not _is_lower_sha256(evidence['parent_buffer_sha256']) or
            evidence['parent_buffer_sha256'] != buffer_hash):
        raise RuntimeError(
            'checkpoint audit parent differs from Stage-0 authority')
    adapter_zero = all(
        torch.count_nonzero(value).item() == 0 for value in adapter.values())
    return {
        'parent_parameter_sha256': parameter_hash,
        'parent_buffer_sha256': buffer_hash,
        'parent_state_sha256': parent_hash,
        'adapter_state_sha256': adapter_hash,
        'adapter_is_zero': adapter_zero,
    }


def _validate_checkpoint_integrity(meta, evidence, expected_role):
    if (type(expected_role) is not str or
            expected_role not in ('control', 'candidate')):
        raise RuntimeError(
            'checkpoint audit expected role must be control or candidate')
    for name in ('epoch', 'iter', 'cfg', 'd13n_integrity'):
        if name not in meta:
            raise RuntimeError(
                'checkpoint audit meta lacks {}'.format(name))
    if type(meta['epoch']) is not int or meta['epoch'] != 12:
        raise RuntimeError(
            'checkpoint audit endpoint epoch must be exact integer 12')
    if type(meta['iter']) is not int or meta['iter'] != 1920:
        raise RuntimeError(
            'checkpoint audit endpoint iter must be exact integer 1920')
    config = meta['cfg']
    if type(config) is not str or not config:
        raise RuntimeError(
            'checkpoint audit meta.cfg must be a nonempty exact string')
    integrity = meta['d13n_integrity']
    _require_exact_dict(
        integrity, _INTEGRITY_KEYS, 'checkpoint audit integrity')
    digest = integrity['integrity_sha256']
    if not _is_lower_sha256(digest):
        raise RuntimeError(
            'checkpoint audit integrity self-hash is invalid')
    unhashed = {
        key: value for key, value in integrity.items()
        if key != 'integrity_sha256'
    }
    try:
        recomputed = _canonical_integrity_sha256(unhashed)
    except Exception as error:
        raise RuntimeError(
            'checkpoint audit integrity is not canonical JSON') from error
    if digest != recomputed:
        raise RuntimeError(
            'checkpoint audit integrity self-hash mismatch')

    if (type(integrity['schema']) is not str or
            integrity['schema'] != 'd13n-checkpoint-integrity-v1'):
        raise RuntimeError('checkpoint audit integrity identity is invalid')
    if (type(integrity['role']) is not str or
            integrity['role'] != expected_role):
        raise RuntimeError('checkpoint audit integrity role differs')
    for name in ('epoch', 'iter'):
        if (type(integrity[name]) is not int or
                integrity[name] != meta[name]):
            raise RuntimeError(
                'checkpoint audit integrity {} differs'.format(name))
    if (type(integrity['optimizer_class']) is not str or
            integrity['optimizer_class'] != 'torch.optim.adamw.AdamW'):
        raise RuntimeError(
            'checkpoint audit optimizer class is not exact AdamW')
    if (not _is_lower_sha256(integrity['parent_state_sha256']) or
            integrity['parent_state_sha256'] !=
            evidence['parent_state_sha256']):
        raise RuntimeError(
            'checkpoint audit integrity parent-state hash differs')
    if (not _is_lower_sha256(integrity['adapter_state_sha256']) or
            integrity['adapter_state_sha256'] !=
            evidence['adapter_state_sha256']):
        raise RuntimeError(
            'checkpoint audit integrity adapter-state hash differs')
    config_hash = hashlib.sha256(config.encode('utf-8')).hexdigest()
    if (not _is_lower_sha256(integrity['config_sha256']) or
            integrity['config_sha256'] != config_hash):
        raise RuntimeError(
            'checkpoint audit integrity config hash differs')
    if expected_role == 'control' and not evidence['adapter_is_zero']:
        raise RuntimeError(
            'checkpoint audit control adapter is not exactly zero')
    return integrity


_OPTIMIZER_KEYS = ('state', 'param_groups', 'base_param_settings')
_GROUP_KEYS = (
    'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach',
    'maximize', 'capturable', 'params',
)
_BASE_KEYS = (
    'params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach',
    'maximize', 'capturable',
)
_GROUP_INITIAL_LR_KEYS = (*_GROUP_KEYS[:-1], 'initial_lr', 'params')
_BASE_INITIAL_LR_KEYS = (*_BASE_KEYS, 'initial_lr')
_SLOT_KEYS = ('step', 'exp_avg', 'exp_avg_sq')


def _require_adamw_container(settings, keys, source):
    if type(settings) is not dict:
        raise RuntimeError(
            'checkpoint audit {} must be an exact dict'.format(source))
    actual = list(settings)
    if keys == _GROUP_KEYS:
        optional_keys = _GROUP_INITIAL_LR_KEYS
    elif keys == _BASE_KEYS:
        optional_keys = _BASE_INITIAL_LR_KEYS
    else:
        raise RuntimeError(
            'checkpoint audit {} expected schema is invalid'.format(source))
    valid = (list(keys), list(optional_keys))
    if (actual not in valid or
            any(type(key) is not str for key in settings)):
        raise RuntimeError(
            'checkpoint audit {} schema/order is invalid'.format(source))
    if type(settings['betas']) is not tuple:
        raise RuntimeError(
            'checkpoint audit {} betas must be an exact tuple'.format(
                source))


def _require_finite_float(value, field, source, *, positive=False):
    if (type(value) is not float or not math.isfinite(value) or
            (value <= 0 if positive else value < 0)):
        raise RuntimeError(
            'checkpoint audit {} {} must be an exact finite {}float'.format(
                source, field, 'positive ' if positive else 'nonnegative '))


def _validate_adamw_settings(settings, source):
    _require_finite_float(settings['lr'], 'lr', source)
    _require_finite_float(settings['eps'], 'eps', source, positive=True)
    _require_finite_float(
        settings['weight_decay'], 'weight_decay', source)
    betas = settings['betas']
    if (type(betas) is not tuple or len(betas) != 2 or
            any(type(beta) is not float or not math.isfinite(beta) or
                beta < 0 or beta >= 1 for beta in betas)):
        raise RuntimeError(
            'checkpoint audit {} betas must be two exact finite floats '
            'in [0, 1)'.format(source))
    if settings['amsgrad'] is not False:
        raise RuntimeError(
            'checkpoint audit {} amsgrad must be exactly False'.format(
                source))
    if settings['foreach'] is not None:
        raise RuntimeError(
            'checkpoint audit {} foreach must be exactly None'.format(
                source))
    if settings['maximize'] is not False:
        raise RuntimeError(
            'checkpoint audit {} maximize must be exactly False'.format(
                source))
    if settings['capturable'] is not False:
        raise RuntimeError(
            'checkpoint audit {} capturable must be exactly False'.format(
                source))
    if 'initial_lr' in settings:
        _require_finite_float(
            settings['initial_lr'], 'initial_lr', source)


def _nonparam_adamw_settings(settings):
    return {
        key: value for key, value in settings.items()
        if key != 'params'
    }


def _canonical_stride(shape):
    result = []
    running = 1
    for size in reversed(shape):
        result.append(running)
        running *= size
    return tuple(reversed(result))


def _validate_canonical_cpu_float32_tensor(value, source):
    if type(value) is not torch.Tensor:
        raise RuntimeError(
            'checkpoint audit {} must be an exact Tensor'.format(source))
    if (value.device.type != 'cpu' or value.dtype != torch.float32 or
            value.layout != torch.strided or value.requires_grad or
            value.grad is not None or value.is_inference() or
            value.numel() <= 0 or not value.is_contiguous() or
            value.stride() != _canonical_stride(value.shape) or
            value._base is not None or value.storage_offset() != 0 or
            value.is_conj() or value.is_neg()):
        raise RuntimeError(
            'checkpoint audit {} tensor is not exact canonical CPU '
            'float32 owned storage'.format(source))
    try:
        storage = value.storage()
        storage_size = storage.size()
        storage_pointer = storage.data_ptr()
        data_pointer = value.data_ptr()
    except Exception as error:
        raise RuntimeError(
            'checkpoint audit {} tensor storage is inaccessible'.format(
                source)) from error
    if (storage_size != value.numel() or storage_pointer <= 0 or
            data_pointer <= 0 or storage_pointer != data_pointer):
        raise RuntimeError(
            'checkpoint audit {} tensor storage pointer/size is invalid'.format(
                source))
    return ('cpu', storage_pointer)


def _validate_optimizer_tensors(optimizer, parameter_ids, state_dict):
    state = optimizer['state']
    base = optimizer['base_param_settings']['params']
    base_storage = _validate_canonical_cpu_float32_tensor(
        base, 'optimizer base params')
    if (base.shape != torch.Size([1]) or
            not torch.equal(base, torch.zeros_like(base))):
        raise RuntimeError(
            'checkpoint audit optimizer base params must be shape-[1] zero')

    state_tensors = []
    state_storages = []
    state_identities = set()
    for parameter_id, parameter_name in zip(parameter_ids, _ADAPTER_NAMES):
        parameter = state_dict[parameter_name]
        if (type(parameter) is not torch.Tensor or
                parameter.dtype != torch.float32):
            raise RuntimeError(
                'checkpoint audit adapter parameter tensor is invalid')
        slot = state[parameter_id]
        step = slot['step']
        step_storage = _validate_canonical_cpu_float32_tensor(
            step, 'optimizer state {} step'.format(parameter_id))
        if (step.shape != torch.Size([]) or
                not torch.isfinite(step).item()):
            raise RuntimeError(
                'checkpoint audit optimizer step must be a finite scalar')
        step_value = step.item()
        if step_value <= 0 or step_value != int(step_value):
            raise RuntimeError(
                'checkpoint audit optimizer step must be a positive integer')

        slot_values = [('step', step, step_storage)]
        for field in ('exp_avg', 'exp_avg_sq'):
            moment = slot[field]
            storage_key = _validate_canonical_cpu_float32_tensor(
                moment, 'optimizer state {} {}'.format(parameter_id, field))
            if (moment.shape != parameter.shape or
                    moment.dtype != parameter.dtype or
                    not torch.isfinite(moment).all().item()):
                raise RuntimeError(
                    'checkpoint audit optimizer {} moment shape/dtype/'
                    'finite contract is invalid'.format(field))
            if field == 'exp_avg_sq' and torch.lt(moment, 0).any().item():
                raise RuntimeError(
                    'checkpoint audit optimizer exp_avg_sq is negative')
            slot_values.append((field, moment, storage_key))
        for field, value, storage_key in slot_values:
            if id(value) in state_identities or storage_key in state_storages:
                raise RuntimeError(
                    'checkpoint audit six optimizer state tensors alias')
            state_identities.add(id(value))
            state_storages.append(storage_key)
            state_tensors.append((parameter_id, field, value))

    if base_storage in state_storages or id(base) in state_identities:
        raise RuntimeError(
            'checkpoint audit optimizer base aliases optimizer state')
    model_identities = set()
    model_storages = set()
    for name, value in state_dict.items():
        if type(value) is not torch.Tensor:
            raise RuntimeError(
                'checkpoint audit model state {} is not an exact Tensor'.format(
                    name))
        try:
            pointer = value.storage().data_ptr()
            data_pointer = value.data_ptr()
        except Exception as error:
            raise RuntimeError(
                'checkpoint audit model-state storage is inaccessible') \
                from error
        if value.numel() <= 0 or pointer <= 0 or data_pointer <= 0:
            raise RuntimeError(
                'checkpoint audit model-state storage pointer is invalid')
        model_identities.add(id(value))
        model_storages.add((value.device.type, pointer))
    for _, _, value in state_tensors:
        storage_key = ('cpu', value.storage().data_ptr())
        if id(value) in model_identities or storage_key in model_storages:
            raise RuntimeError(
                'checkpoint audit optimizer state aliases model state')
    if id(base) in model_identities or base_storage in model_storages:
        raise RuntimeError(
            'checkpoint audit optimizer base aliases model state')


def _validate_optimizer_binding(optimizer, integrity, state_dict):
    _require_exact_dict(
        optimizer, _OPTIMIZER_KEYS, 'checkpoint audit optimizer')
    state = optimizer['state']
    groups = optimizer['param_groups']
    base = optimizer['base_param_settings']
    if type(state) is not dict:
        raise RuntimeError(
            'checkpoint audit optimizer state must be an exact dict')
    if type(groups) is not list or len(groups) != 2:
        raise RuntimeError(
            'checkpoint audit optimizer groups must be exact [1, 1]')
    for index, group in enumerate(groups):
        _require_adamw_container(
            group, _GROUP_KEYS, 'optimizer group {}'.format(index))
        _validate_adamw_settings(
            group, 'optimizer group {}'.format(index))
        if type(group['params']) is not list or len(group['params']) != 1:
            raise RuntimeError(
                'checkpoint audit optimizer groups must be exact [1, 1]')
    _require_adamw_container(base, _BASE_KEYS, 'optimizer base settings')
    _validate_adamw_settings(base, 'optimizer base settings')
    first_settings = _nonparam_adamw_settings(groups[0])
    if (not _typed_equal(
            first_settings, _nonparam_adamw_settings(groups[1])) or
            not _typed_equal(
                first_settings, _nonparam_adamw_settings(base))):
        raise RuntimeError(
            'checkpoint audit AdamW groups/base settings differ')

    expected_names = [[_ADAPTER_NAMES[0]], [_ADAPTER_NAMES[1]]]
    names = integrity['optimizer_parameter_names_by_group']
    if (type(names) is not list or len(names) != 2 or
            any(type(group) is not list for group in names) or
            not _typed_equal(names, expected_names)):
        raise RuntimeError(
            'checkpoint audit optimizer parameter names are invalid')
    for field in (
            'live_optimizer_group_sizes',
            'serialized_optimizer_group_sizes'):
        sizes = integrity[field]
        if (type(sizes) is not list or len(sizes) != 2 or
                any(type(size) is not int for size in sizes) or
                sizes != [1, 1]):
            raise RuntimeError(
                'checkpoint audit {} is invalid'.format(field))
    if (type(integrity['initialized_optimizer_state_slots']) is not int or
            integrity['initialized_optimizer_state_slots'] != 2):
        raise RuntimeError(
            'checkpoint audit initialized optimizer slot count is invalid')

    recorded_ids = integrity[
        'serialized_optimizer_parameter_ids_by_group']
    if (type(recorded_ids) is not list or len(recorded_ids) != 2 or
            any(type(group) is not list or len(group) != 1
                for group in recorded_ids)):
        raise RuntimeError(
            'checkpoint audit recorded optimizer IDs are invalid')
    flattened_recorded = [value for group in recorded_ids for value in group]
    if (any(type(value) is not int for value in flattened_recorded) or
            len(set(flattened_recorded)) != 2):
        raise RuntimeError(
            'checkpoint audit recorded optimizer IDs must be unique ints')
    flattened_actual = [group['params'][0] for group in groups]
    if (any(type(value) is not int for value in flattened_actual) or
            flattened_actual != flattened_recorded):
        raise RuntimeError(
            'checkpoint audit optimizer group positions differ from metadata')
    if (any(type(value) is not int for value in state) or
            list(state) != flattened_recorded):
        raise RuntimeError(
            'checkpoint audit optimizer state key order differs')
    if len(state) != 2:
        raise RuntimeError(
            'checkpoint audit optimizer must have exactly two state slots')
    for parameter_id in flattened_recorded:
        slot = state[parameter_id]
        _require_exact_dict(
            slot, _SLOT_KEYS,
            'checkpoint audit optimizer slot {}'.format(parameter_id))
    _validate_optimizer_tensors(
        optimizer, flattened_recorded, state_dict)
    return list(_ADAPTER_NAMES)


def audit_d13n_checkpoint(checkpoint, stage0_report, role, cache):
    """Audit one trusted endpoint checkpoint before returning any claim.

    Checkpoints are loaded through pickle and must already be trusted. This
    validation binds experiment evidence; it is not a security boundary for
    attacker-controlled checkpoint files.
    """
    if type(role) is not str or role not in ('control', 'candidate'):
        raise RuntimeError(
            'checkpoint audit role must be control or candidate')
    if not isinstance(cache, CheckpointHashCache):
        raise TypeError('checkpoint audit cache must be CheckpointHashCache')
    hash_record, payload = cache._hash_and_load(checkpoint)
    evidence = _stage0_model_authority(
        stage0_report, payload['state_dict'])
    integrity = _validate_checkpoint_integrity(
        payload['meta'], evidence, role)
    names = _validate_optimizer_binding(
        payload['optimizer'], integrity, payload['state_dict'])
    return {
        'checkpoint_path': hash_record['path'],
        'checkpoint_sha256': hash_record['sha256'],
        'checkpoint_sha_cached': hash_record['sha_cached'],
        'role': role,
        'endpoint_epoch': payload['meta']['epoch'],
        'optimizer_parameter_names': names,
        'parent_parameter_sha256': evidence['parent_parameter_sha256'],
        'parent_buffer_sha256': evidence['parent_buffer_sha256'],
        'adapter_state_sha256': evidence['adapter_state_sha256'],
        'adapter_is_zero': evidence['adapter_is_zero'],
    }


__all__ = [
    'PHYSICAL_GPUS', 'GPU_UUIDS', 'CONTROL_PORT', 'CANDIDATE_PORT',
    'RAW_CONTROL_PORT', 'RAW_CANDIDATE_PORT', 'STAGE0_PARENT_PORT',
    'STAGE0_CONTROL_PORT', 'STAGE0_CANDIDATE_PORT', 'E24',
    'CANONICAL_E24_DUMP', 'TRAIN_MANIFEST', 'PROXY_MANIFEST',
    'CONTROL_CONFIG', 'CANDIDATE_CONFIG', 'RAW_CONTROL_CONFIG',
    'RAW_CANDIDATE_CONFIG', 'ANALYZER', 'DIAGNOSTICS', 'VALIDATOR',
    'MONITOR_STARTUP_TIMEOUT_SECONDS',
    'MONITOR_HEARTBEAT_MAX_AGE_SECONDS', 'EXPECTED_ENVIRONMENT',
    'EXPECTED_INPUT_SHA256', 'sha256_file', 'canonical_json_sha256',
    'publish_json_noreplace', 'validate_frozen_environment',
    'collect_environment_fingerprint', 'assert_fingerprint_equal',
    'classify_process_snapshot', 'assert_world5_topology',
    'normalized_pair_config', 'assert_paths_unoccupied',
    'assert_ports_free', 'publish_monitor_ready', 'validate_monitor_ready',
    'validate_monitor_heartbeat',
    'FAILURE_CLASSES', 'publish_integrity_failure_lock',
    'load_integrity_failure_lock', 'CheckpointHashCache',
    'audit_d13n_checkpoint',
]
