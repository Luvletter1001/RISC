import json

import pytest

from projects.OVCapFlow.tools.audit_sampler_coverage import (
    audit_sampler_coverage,
    build_parser,
)


class _FakeDataset:

    def __init__(self, gt_counts):
        self.gt_counts = list(gt_counts)

    def __len__(self):
        return len(self.gt_counts)

    def get_data_info(self, index):
        return {'instances': [{} for _ in range(self.gt_counts[index])]}


def test_audit_sampler_coverage_replays_every_rank_exactly_once(tmp_path):
    output = tmp_path / 'coverage.json'
    report = audit_sampler_coverage(
        dataset=_FakeDataset([0, 1, 3, 20] * 7 + [2]),
        batch_size=3,
        batch_sampler_cfg={
            'num_matching_queries': 10,
            'num_dn_queries': 10,
            'max_query_area': 3000,
        },
        seed=17,
        epoch=11,
        world_size=4,
        output=output)

    assert report['dataset_size'] == 29
    assert report['epoch'] == 11
    assert report['world_size'] == 4
    assert report['duplicate_count'] == 0
    assert report['missing_count'] == 0
    assert len(set(report['rank_update_counts'])) == 1
    assert sum(report['rank_sample_counts']) == 29
    assert report['local_batch_size_min'] >= 1
    assert report['local_batch_size_max'] <= 3
    assert json.loads(output.read_text()) == report


def test_audit_sampler_coverage_records_query_geometry(tmp_path):
    output = tmp_path / 'grouped.json'
    report = audit_sampler_coverage(
        dataset=_FakeDataset([0, 2, 5, 50] * 4),
        batch_size=4,
        batch_sampler_cfg={
            'num_matching_queries': 1800,
            'num_dn_queries': 100,
            'max_query_area': 50_000_000,
        },
        seed=20260716,
        epoch=11,
        world_size=1,
        output=output)

    assert report['num_matching_queries'] == 1800
    assert report['num_dn_queries'] == 100
    assert report['max_query_area_budget'] == 50_000_000
    assert report['seed'] == 20260716
    assert report['coverage_checksum']


def test_audit_sampler_coverage_forwards_update_count_multiple(tmp_path):
    output = tmp_path / 'coverage_update_multiple.json'
    report = audit_sampler_coverage(
        dataset=_FakeDataset([1] * 18),
        batch_size=2,
        batch_sampler_cfg={
            'num_matching_queries': 1800,
            'num_dn_queries': 100,
            'max_query_area': 50_000_000,
            'update_count_multiple': 2,
        },
        seed=20260716,
        epoch=0,
        world_size=2,
        output=output)

    assert report['update_count_multiple'] == 2
    assert report['update_count'] % 2 == 0
    assert report['accumulation_split_count'] == 1
    assert report['duplicate_count'] == 0
    assert report['missing_count'] == 0


def test_sampler_coverage_cli_has_explicit_world_epoch_and_output():
    args = build_parser().parse_args([
        'config.py', '--world-size', '4', '--epoch', '11',
        '--seed', '20260716', '--output', 'coverage.json'])
    assert str(args.config) == 'config.py'
    assert args.world_size == 4
    assert args.epoch == 11
    assert args.seed == 20260716
    assert str(args.output) == 'coverage.json'


def test_independent_audit_builds_plans_without_sampler_side_effects(
        tmp_path, monkeypatch):
    output = tmp_path / 'coverage.json'

    def forbid_sampler_audit(*_args, **_kwargs):
        raise AssertionError('independent audit must call _build_plan directly')

    monkeypatch.setattr(
        'projects.OVCapFlow.ov_capflow.dn_budget_batch_sampler.'
        'DNQueryBudgetBatchSampler._write_audit',
        forbid_sampler_audit)
    report = audit_sampler_coverage(
        dataset=_FakeDataset([1] * 12),
        batch_size=2,
        batch_sampler_cfg={
            'num_matching_queries': 10,
            'num_dn_queries': 10,
            'max_query_area': 3000,
            'audit_path': 'must-not-be-forwarded.json',
            'audit_noreplace': True,
        },
        seed=17,
        epoch=6,
        world_size=2,
        output=output)

    assert report['rank_update_counts']
    assert json.loads(output.read_text(encoding='utf-8')) == report


def test_independent_audit_publishes_enriched_report_once_without_replacement(
        tmp_path):
    output = tmp_path / 'coverage.json'
    kwargs = dict(
        dataset=_FakeDataset([1] * 12),
        batch_size=2,
        batch_sampler_cfg={
            'num_matching_queries': 10,
            'num_dn_queries': 10,
            'max_query_area': 3000,
        },
        seed=17,
        epoch=6,
        world_size=2,
        output=output)
    report = audit_sampler_coverage(**kwargs)
    original = output.read_bytes()

    assert report['num_matching_queries'] == 10
    assert report['num_dn_queries'] == 10
    assert report['rank_update_counts']
    with pytest.raises(FileExistsError, match='output collision'):
        audit_sampler_coverage(**kwargs)
    assert output.read_bytes() == original
