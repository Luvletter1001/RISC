import json
import os
import subprocess
from pathlib import Path

import pytest


def _api():
    from projects.OVCapFlow.tools.d12_postrun_gate import (
        build_audit_specs,
        checkpoint_published,
        evaluate_frozen_gate,
        exact_training_process_exists,
        extract_complete_epoch_metric,
        fatal_log_hits,
        pair_completion_status,
        parse_epoch_records,
        publish_json_no_clobber,
        validate_sampler_pair,
    )
    return {
        'audit_specs': build_audit_specs,
        'checkpoint': checkpoint_published,
        'evaluate': evaluate_frozen_gate,
        'exact_process': exact_training_process_exists,
        'extract_metric': extract_complete_epoch_metric,
        'fatal_hits': fatal_log_hits,
        'pair_status': pair_completion_status,
        'parse_epochs': parse_epoch_records,
        'publish': publish_json_no_clobber,
        'samplers': validate_sampler_pair,
    }


CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')


def _epoch_table(epoch=12, ap=0.5, ap50=None):
    lines = [
        '+--------------------+-----+------+--------+-------+',
        '| class              | gts | dets | recall | ap    |',
        '+--------------------+-----+------+--------+-------+',
    ]
    for name in CLASSES:
        lines.append(f'| {name} | 10 | 20 | 0.600 | {ap:.6f} |')
    lines.extend([
        f'| mAP | | | | {ap:.6f} |',
        '+--------------------+-----+------+--------+-------+',
        (
            f'Epoch(val) [{epoch}][400/400] '
            f'dota/mAP: {ap:.6f} '
            f'dota/AP50: {(ap if ap50 is None else ap50):.6f}'
        ),
    ])
    return '\n'.join(lines) + '\n'


def _metric(map_value=0.5, ap50=0.5, novel4=0.5, base14=0.5):
    return {
        'epoch': 12,
        'mAP': map_value,
        'AP50': ap50,
        'novel4': novel4,
        'base14': base14,
    }


def _passing_metrics():
    control = _metric(
        map_value=0.470, ap50=0.470, novel4=0.3675, base14=0.470)
    candidate = _metric(
        map_value=0.490, ap50=0.490, novel4=0.3675, base14=0.465)
    return candidate, control


def test_exact_training_process_requires_train_script_and_exact_config(tmp_path):
    config = Path('configs/ov_capflow/dotav2/exact_d12.py')
    live = tmp_path / '100'
    live.mkdir()
    (live / 'cmdline').write_bytes(
        b'python\0tools/train.py\0' + str(config).encode()
        + b'\0--launcher\0pytorch\0')
    embedded = tmp_path / '101'
    embedded.mkdir()
    (embedded / 'cmdline').write_bytes(
        b'tmux\0python tools/train.py ' + str(config).encode())

    assert _api()['exact_process'](config, proc_root=tmp_path)
    assert not _api()['exact_process'](
        Path('configs/ov_capflow/dotav2/other.py'), proc_root=tmp_path)


def test_epoch12_metric_requires_complete_progress_and_finite_values():
    text = (
        'Epoch(val) [12][400/400] '
        'dota/mAP: 0.4912 dota/AP50: 0.4910\n')

    assert _api()['extract_metric'](text, 12) == {
        'epoch': 12,
        'progress': [400, 400],
        'mAP': pytest.approx(0.4912),
        'AP50': pytest.approx(0.4910),
    }
    assert _api()['extract_metric'](
        text.replace('[400/400]', '[399/400]'), 12) is None
    assert _api()['extract_metric'](
        text.replace('[12]', '[11]'), 12) is None
    assert _api()['extract_metric'](
        text.replace('0.4912', 'nan'), 12) is None


@pytest.mark.parametrize(
    'override,reason',
    [
        ({'candidate_training_alive': True}, 'candidate_training_exit'),
        ({'control_training_alive': True}, 'control_training_exit'),
        ({'candidate_metric_complete': False}, 'candidate_epoch12_metric'),
        ({'control_metric_complete': False}, 'control_epoch12_metric'),
        ({'candidate_checkpoint_published': False}, 'candidate_checkpoint'),
        ({'control_checkpoint_published': False}, 'control_checkpoint'),
        ({'candidate_sampler_complete': False}, 'candidate_sampler'),
        ({'control_sampler_complete': False}, 'control_sampler'),
        ({'sampler_checksums_match': False}, 'sampler_checksum_match'),
        ({'gpus_idle': False}, 'gpu0_9_idle'),
        ({'stage0_passed': False}, 'stage0_pass'),
        ({'fatal_free': False}, 'fatal_free'),
    ])
def test_pair_status_requires_every_completion_condition(override, reason):
    evidence = {
        'candidate_training_alive': False,
        'control_training_alive': False,
        'candidate_metric_complete': True,
        'control_metric_complete': True,
        'candidate_checkpoint_published': True,
        'control_checkpoint_published': True,
        'candidate_sampler_complete': True,
        'control_sampler_complete': True,
        'sampler_checksums_match': True,
        'gpus_idle': True,
        'stage0_passed': True,
        'fatal_free': True,
    }
    evidence.update(override)

    status = _api()['pair_status'](**evidence)

    assert status['ready'] is False
    assert reason in status['waiting_for']


def test_pair_status_ready_only_when_every_condition_holds():
    status = _api()['pair_status'](
        candidate_training_alive=False,
        control_training_alive=False,
        candidate_metric_complete=True,
        control_metric_complete=True,
        candidate_checkpoint_published=True,
        control_checkpoint_published=True,
        candidate_sampler_complete=True,
        control_sampler_complete=True,
        sampler_checksums_match=True,
        gpus_idle=True,
        stage0_passed=True,
        fatal_free=True)

    assert status == {'ready': True, 'waiting_for': []}


def test_checkpoint_publication_requires_size_and_last_pointer(tmp_path):
    checkpoint = tmp_path / 'epoch_12.pth'
    checkpoint.write_bytes(b'0123456789')
    assert not _api()['checkpoint'](checkpoint, minimum_bytes=10)

    (tmp_path / 'last_checkpoint').write_text(str(checkpoint))
    assert _api()['checkpoint'](checkpoint, minimum_bytes=10)

    (tmp_path / 'last_checkpoint').write_text(
        str(tmp_path / 'epoch_11.pth'))
    assert not _api()['checkpoint'](checkpoint, minimum_bytes=10)


def _sampler_report(checksum='same'):
    return {
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
        'max_estimated_query_area': 49_000_000,
        'max_query_area_budget': 50_000_000,
        'coverage_checksum': checksum,
    }


def test_sampler_pair_requires_exact_cover_and_matching_checksum(tmp_path):
    candidate = tmp_path / 'candidate.json'
    control = tmp_path / 'control.json'
    candidate.write_text(json.dumps(_sampler_report()))
    control.write_text(json.dumps(_sampler_report()))

    report = _api()['samplers'](candidate, control)

    assert report['pass'] is True
    assert report['samples_per_rank'] == 320
    assert report['checksums_match'] is True

    control.write_text(json.dumps(_sampler_report(checksum='different')))
    with pytest.raises(ValueError, match='checksum'):
        _api()['samplers'](candidate, control)

    bad = _sampler_report()
    bad['duplicate_count'] = 1
    control.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='duplicate_count'):
        _api()['samplers'](candidate, control)


def test_class_table_parser_returns_epoch12_aggregate_only_when_complete():
    text = _epoch_table(epoch=11, ap=0.4) + _epoch_table(epoch=12, ap=0.5)

    records = _api()['parse_epochs'](text, source='console.log')

    assert [record['epoch'] for record in records] == [11, 12]
    assert records[-1]['mAP'] == pytest.approx(0.5)
    assert records[-1]['AP50'] == pytest.approx(0.5)
    assert records[-1]['novel4'] == pytest.approx(0.5)
    assert records[-1]['base14'] == pytest.approx(0.5)
    assert list(records[-1]['class_ap']) == list(CLASSES)

    with pytest.raises(ValueError, match='incomplete class table'):
        _api()['parse_epochs'](
            _epoch_table().replace(
                '| tennis-court | 10 | 20 | 0.600 | 0.500000 |\n', ''),
            source='truncated.log')


def test_frozen_gate_accepts_all_inclusive_threshold_boundaries():
    candidate, control = _passing_metrics()
    integrity = {
        'sampler': True,
        'stage0': True,
        'candidate_strict': True,
        'candidate_open_vocabulary': True,
        'control_strict': True,
        'control_open_vocabulary': True,
        'rotated_boxes': True,
        'fatal_free': True,
    }

    report = _api()['evaluate'](candidate, control, integrity)

    assert report['pass'] is True
    assert report['decision'] == 'promote'
    assert report['delta'] == {
        'mAP': pytest.approx(0.020),
        'AP50': pytest.approx(0.020),
        'novel4': pytest.approx(0.0),
        'base14': pytest.approx(-0.005),
    }
    assert all(check['pass'] for check in report['checks'].values())


@pytest.mark.parametrize(
    'role,field,value,check',
    [
        ('candidate', 'AP50', 0.488999, 'candidate_AP50_absolute'),
        ('candidate', 'AP50', 0.489999, 'candidate_AP50_delta'),
        ('candidate', 'mAP', 0.489999, 'candidate_mAP_delta'),
        ('candidate', 'novel4', 0.367499, 'candidate_novel4_absolute'),
        ('candidate', 'novel4', 0.367499, 'candidate_novel4_control'),
        ('candidate', 'base14', 0.464642, 'candidate_base14_absolute'),
        ('candidate', 'base14', 0.464999, 'candidate_base14_control'),
        ('integrity', 'sampler', False, 'integrity_sampler'),
    ])
def test_frozen_gate_discards_when_any_conjunct_fails(
        role, field, value, check):
    candidate, control = _passing_metrics()
    integrity = {
        'sampler': True,
        'stage0': True,
        'candidate_strict': True,
        'candidate_open_vocabulary': True,
        'control_strict': True,
        'control_open_vocabulary': True,
        'rotated_boxes': True,
        'fatal_free': True,
    }
    if role == 'integrity':
        integrity[field] = value
    else:
        candidate[field] = value

    report = _api()['evaluate'](candidate, control, integrity)

    assert report['pass'] is False
    assert report['decision'] == 'discard'
    assert report['checks'][check]['pass'] is False


def test_frozen_gate_rejects_nonfinite_or_non_epoch12_metrics():
    candidate, control = _passing_metrics()
    candidate['mAP'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        _api()['evaluate'](candidate, control, {'sampler': True})

    candidate, control = _passing_metrics()
    candidate['epoch'] = 11
    with pytest.raises(ValueError, match='epoch 12'):
        _api()['evaluate'](candidate, control, {'sampler': True})


def test_fatal_scan_uses_specific_signatures():
    scan = _api()['fatal_hits']

    assert scan('normal information line\nloss_cls: 0.2\n') == []
    hits = scan(
        'Traceback (most recent call last):\n'
        'CUDA out of memory\nNCCL WARN connection aborted\nloss: nan\n')
    assert {hit['signature'] for hit in hits} == {
        'traceback', 'cuda_oom', 'nccl_failure', 'nonfinite'}


def test_audit_specs_bind_candidate_gpu0_and_control_gpu5(tmp_path):
    specs = _api()['audit_specs'](
        python=Path('/env/bin/python'),
        strict_script=Path('/repo/strict.py'),
        open_vocabulary_script=Path('/repo/open.py'),
        candidate_config=Path('/repo/candidate.py'),
        control_config=Path('/repo/control.py'),
        candidate_checkpoint=Path('/repo/candidate/epoch_12.pth'),
        control_checkpoint=Path('/repo/control/epoch_12.pth'),
        output_root=tmp_path)

    assert [spec['name'] for spec in specs] == [
        'candidate_strict',
        'candidate_open_vocabulary',
        'control_strict',
        'control_open_vocabulary',
    ]
    assert [spec['physical_gpu'] for spec in specs] == [0, 0, 5, 5]
    assert [
        spec['environment']['CUDA_VISIBLE_DEVICES'] for spec in specs
    ] == ['0', '0', '5', '5']
    assert all(
        spec['command'][spec['command'].index('--device') + 1] == 'cuda:0'
        for spec in specs)


def test_gate_report_publication_is_no_clobber(tmp_path):
    output = tmp_path / 'gate.json'

    _api()['publish']({'decision': 'discard'}, output)

    assert json.loads(output.read_text())['decision'] == 'discard'
    with pytest.raises(FileExistsError, match='output collision'):
        _api()['publish']({'decision': 'promote'}, output)


def test_cli_help_exposes_frozen_postrun_inputs():
    module = Path('projects/OVCapFlow/tools/d12_postrun_gate.py')
    completed = subprocess.run(
        [os.sys.executable, os.fspath(module), '--help'],
        capture_output=True,
        text=True,
        check=False)

    assert completed.returncode == 0
    for option in (
            '--candidate-config',
            '--control-config',
            '--candidate-work-dir',
            '--control-work-dir',
            '--candidate-sampler-audit',
            '--control-sampler-audit',
            '--target-epoch',
            '--poll-seconds',
            '--output'):
        assert option in completed.stdout
