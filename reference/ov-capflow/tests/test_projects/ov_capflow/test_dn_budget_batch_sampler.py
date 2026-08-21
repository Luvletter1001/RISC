import json
from inspect import signature
from pathlib import Path

import pytest

from projects.OVCapFlow.ov_capflow import DNQueryBudgetBatchSampler


class _FakeDataset:

    def __init__(self, gt_counts):
        self.gt_counts = list(gt_counts)

    def __len__(self):
        return len(self.gt_counts)

    def get_data_info(self, index):
        return {'instances': [{} for _ in range(self.gt_counts[index])]}


class _FakeSampler:

    def __init__(self, dataset, rank, world_size, seed=17, shuffle=True):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0


def _build_rank_samplers(
        gt_counts, *, epoch=0, batch_size=3, max_query_area=10**9,
        audit_path=None, update_count_multiple=1, audit_noreplace=False):
    dataset = _FakeDataset(gt_counts)
    batch_samplers = []
    for rank in range(4):
        sampler = _FakeSampler(dataset, rank=rank, world_size=4)
        sampler.epoch = epoch
        batch_samplers.append(DNQueryBudgetBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            num_matching_queries=10,
            num_dn_queries=10,
            max_query_area=max_query_area,
            update_count_multiple=update_count_multiple,
            audit_path=audit_path,
            audit_noreplace=audit_noreplace))
    return batch_samplers


def _materialize(batch_samplers):
    return [[list(batch) for batch in sampler] for sampler in batch_samplers]


def _flatten(rank_batches):
    return [index for batch in rank_batches for index in batch]


def test_four_ranks_cover_once_without_empty_or_unequal_updates():
    rank_batches = _materialize(_build_rank_samplers([1] * 29))

    all_indices = [
        index for batches in rank_batches for batch in batches
        for index in batch
    ]
    assert sorted(all_indices) == list(range(29))
    assert len(all_indices) == len(set(all_indices))
    assert len({len(batches) for batches in rank_batches}) == 1
    assert all(batch for batches in rank_batches for batch in batches)
    assert all(len(batch) <= 3
               for batches in rank_batches for batch in batches)
    for left_rank in range(4):
        for right_rank in range(left_rank + 1, 4):
            assert not (
                set(_flatten(rank_batches[left_rank]))
                & set(_flatten(rank_batches[right_rank])))


def test_same_epoch_repeats_and_next_epoch_changes_order():
    first = _materialize(_build_rank_samplers([1] * 29, epoch=2))
    repeated = _materialize(_build_rank_samplers([1] * 29, epoch=2))
    next_epoch = _materialize(_build_rank_samplers([1] * 29, epoch=3))

    assert repeated == first
    assert next_epoch != first


def test_update_count_multiple_splits_without_duplicate_or_missing_samples():
    batch_samplers = _build_rank_samplers(
        [1] * 29, update_count_multiple=2)
    rank_batches = _materialize(batch_samplers)

    assert len(rank_batches[0]) == 4
    assert all(len(batches) % 2 == 0 for batches in rank_batches)
    assert all(batch for batches in rank_batches for batch in batches)
    all_indices = [
        index for batches in rank_batches for batch in batches
        for index in batch
    ]
    assert sorted(all_indices) == list(range(29))
    assert len(all_indices) == len(set(all_indices))
    _, report = batch_samplers[0]._build_plan()
    assert report['accumulation_split_count'] == 1
    assert report['update_count_multiple'] == 2


def test_dense_sample_gets_singleton_local_batch():
    gt_counts = [1] * 23 + [20]
    rank_batches = _materialize(_build_rank_samplers(
        gt_counts, batch_size=3, max_query_area=3000))

    dense_batches = [
        batch for batches in rank_batches for batch in batches if 23 in batch
    ]
    assert dense_batches == [[23]]


def test_rank_zero_writes_complete_audit(tmp_path):
    audit_path = tmp_path / 'sampler.json'
    rank_batches = _materialize(_build_rank_samplers(
        [0, 1, 3, 20] * 7 + [2],
        batch_size=3,
        max_query_area=3000,
        audit_path=audit_path))

    report = json.loads(audit_path.read_text(encoding='utf-8'))
    assert report['dataset_size'] == 29
    assert report['world_size'] == 4
    assert report['update_count'] == len(rank_batches[0])
    assert report['duplicate_count'] == 0
    assert report['missing_count'] == 0
    assert report['max_gt'] == 20
    assert report['max_gt_sample']['gt_count'] == 20
    assert report['max_gt_sample']['index'] in {
        index for index, count in enumerate([0, 1, 3, 20] * 7 + [2])
        if count == 20
    }
    assert report['shrink_update_count'] > 0
    assert report['local_batch_size_min'] >= 1
    assert report['local_batch_size_max'] <= 3
    assert len(report['coverage_checksum']) == 64
    worst = report['max_query_area_batch']
    assert worst['estimated_query_area'] == report[
        'max_estimated_query_area']
    assert len(worst['indices']) == len(worst['gt_counts'])
    assert max(worst['gt_counts']) == 20


def test_audit_noreplace_defaults_false_and_preserves_legacy_writer(tmp_path):
    assert signature(DNQueryBudgetBatchSampler.__init__).parameters[
        'audit_noreplace'].default is False
    audit_path = tmp_path / 'legacy.json'
    audit_path.write_bytes(b'old-report')
    samplers = _build_rank_samplers(
        [1] * 29, audit_path=audit_path)
    list(samplers[0])
    _, report = samplers[0]._build_plan()

    assert audit_path.read_bytes() == json.dumps(
        report, indent=2).encode('utf-8')
    assert set(json.loads(audit_path.read_text(encoding='utf-8'))) == set(report)


def test_noreplace_audit_template_uses_report_epoch_without_write_text(
        tmp_path, monkeypatch):
    template = tmp_path / 'sampler_epoch_{epoch:02d}.json'
    sampler = _build_rank_samplers(
        [1] * 29,
        epoch=2,
        audit_path=template,
        audit_noreplace=True)[0]
    plan, report = sampler._build_plan()
    report = dict(report, epoch=7)
    monkeypatch.setattr(sampler, '_build_plan', lambda: (plan, report))

    def forbid_write_text(*_args, **_kwargs):
        raise AssertionError('no-replace publication must not use write_text')

    monkeypatch.setattr(Path, 'write_text', forbid_write_text)
    list(sampler)

    resolved = tmp_path / 'sampler_epoch_07.json'
    assert json.loads(resolved.read_text(encoding='utf-8')) == report
    assert not (tmp_path / 'sampler_epoch_02.json').exists()


def test_noreplace_audit_creates_distinct_epoch_paths(tmp_path):
    template = tmp_path / 'sampler_epoch_{epoch:02d}.json'

    list(_build_rank_samplers(
        [1] * 29, epoch=2, audit_path=template,
        audit_noreplace=True)[0])
    list(_build_rank_samplers(
        [1] * 29, epoch=3, audit_path=template,
        audit_noreplace=True)[0])

    assert (tmp_path / 'sampler_epoch_02.json').exists()
    assert (tmp_path / 'sampler_epoch_03.json').exists()


def test_only_rank_zero_publishes_noreplace_audit(tmp_path, monkeypatch):
    import projects.OVCapFlow.ov_capflow.dn_budget_batch_sampler as module

    template = tmp_path / 'sampler_epoch_{epoch:02d}.json'
    samplers = _build_rank_samplers(
        [1] * 29, epoch=4, audit_path=template,
        audit_noreplace=True)
    published = []

    def record_publish(path, report):
        published.append((Path(path), report['epoch']))

    monkeypatch.setattr(module, 'publish_json_noreplace', record_publish)
    for sampler in reversed(samplers):
        list(sampler)

    assert published == [(tmp_path / 'sampler_epoch_04.json', 4)]


def test_noreplace_audit_refuses_second_publication_without_clobber(tmp_path):
    template = tmp_path / 'sampler_epoch_{epoch:02d}.json'
    first = _build_rank_samplers(
        [1] * 29, epoch=5, audit_path=template,
        audit_noreplace=True)[0]
    list(first)
    target = tmp_path / 'sampler_epoch_05.json'
    original = target.read_bytes()

    second = _build_rank_samplers(
        [1] * 29, epoch=5, audit_path=template,
        audit_noreplace=True)[0]
    with pytest.raises(FileExistsError, match='output collision'):
        list(second)

    assert target.read_bytes() == original
