import json

import pytest

from projects.OVCapFlow.tools.monitor_gpu_run import (
    build_parser,
    detect_fatal_pattern,
    parse_nvidia_smi_csv,
    run_monitor,
    summarize_samples,
)


GPU4 = {
    'index': 4,
    'utilization_percent': 97.0,
    'memory_used_mib': 35903.0,
    'memory_total_mib': 46068.0,
    'power_draw_w': 254.4,
}


def test_parse_nvidia_smi_csv_preserves_physical_gpu_indices():
    text = (
        '4, 97, 35903, 46068, 254.40\n'
        '5, 100, 31869, 46068, 200.44\n')
    parsed = parse_nvidia_smi_csv(text, expected_indices=[4, 5])
    assert parsed[0] == GPU4
    assert parsed[1]['index'] == 5
    assert parsed[1]['utilization_percent'] == 100.0


def test_parse_nvidia_smi_csv_rejects_missing_or_unexpected_gpu():
    with pytest.raises(ValueError, match='GPU indices'):
        parse_nvidia_smi_csv(
            '4, 97, 35903, 46068, 254.40\n',
            expected_indices=[4, 5])


@pytest.mark.parametrize(
    ('line', 'expected'),
    [
        ('torch.OutOfMemoryError: CUDA out of memory', 'cuda_oom'),
        ('RuntimeError: NCCL communicator was aborted', 'nccl'),
        ('ChildFailedError: rank 2 exited', 'dead_rank'),
        ('loss_bbox: nan grad_norm: 12', 'non_finite'),
        ('Traceback (most recent call last):', 'traceback'),
    ])
def test_detect_fatal_pattern(line, expected):
    assert detect_fatal_pattern(line) == expected


def test_detect_fatal_pattern_does_not_confuse_normal_info_lines():
    assert detect_fatal_pattern(
        'mmengine - INFO - loss: 20.3 data_time: 0.02') is None
    assert detect_fatal_pattern(
        'CXX_FLAGS=-Wno-error USE_NCCL=ON -Werror=format USE_CUDA=ON') is None


def test_summarize_samples_reports_per_card_medians():
    samples = [
        {'gpus': [GPU4]},
        {'gpus': [{**GPU4, 'utilization_percent': 75.0,
                   'memory_used_mib': 34000.0}]},
        {'gpus': [{**GPU4, 'utilization_percent': 90.0,
                   'memory_used_mib': 35000.0}]},
    ]
    summary = summarize_samples(samples)
    assert summary['sample_count'] == 3
    assert summary['gpus']['4']['median_utilization_percent'] == 90.0
    assert summary['gpus']['4']['median_memory_used_mib'] == 35000.0
    assert summary['gpus']['4']['memory_total_mib'] == 46068.0


def test_run_monitor_writes_sample_and_summary_jsonl(tmp_path):
    output = tmp_path / 'monitor.jsonl'
    code = run_monitor(
        pid=123,
        gpu_indices=[4],
        interval=30,
        train_log=tmp_path / 'train.log',
        output=output,
        max_samples=1,
        query_fn=lambda indices: [GPU4],
        alive_fn=lambda: True,
        log_reader=lambda: '',
        sleep_fn=lambda _: None,
        now_fn=lambda: '2026-07-15T20:00:00+08:00')
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert code == 0
    assert records[0] == {
        'type': 'sample',
        'timestamp': '2026-07-15T20:00:00+08:00',
        'pid': 123,
        'process_group_alive': True,
        'fatal_pattern': None,
        'gpus': [GPU4],
    }
    assert records[1]['type'] == 'summary'
    assert records[1]['reason'] == 'max_samples'
    assert records[1]['sample_count'] == 1


def test_run_monitor_exits_nonzero_on_fatal_log_or_dead_group(tmp_path):
    common = dict(
        pid=123,
        gpu_indices=[4],
        interval=30,
        train_log=tmp_path / 'train.log',
        max_samples=1,
        query_fn=lambda indices: [GPU4],
        sleep_fn=lambda _: None,
        now_fn=lambda: '2026-07-15T20:00:00+08:00')
    assert run_monitor(
        output=tmp_path / 'fatal.jsonl',
        alive_fn=lambda: True,
        log_reader=lambda: 'RuntimeError: NCCL failure',
        **common) == 2
    assert run_monitor(
        output=tmp_path / 'dead.jsonl',
        alive_fn=lambda: False,
        log_reader=lambda: '',
        **common) == 3


def test_cli_requires_pid_log_output_and_defaults_to_30_seconds():
    args = build_parser().parse_args([
        '--pid', '123', '--gpu-indices', '4', '5', '6', '7',
        '--train-log', 'train.log', '--output', 'monitor.jsonl'])
    assert args.pid == 123
    assert args.gpu_indices == [4, 5, 6, 7]
    assert args.interval == 30
