from pathlib import Path

from mmengine import Config

from projects.OVCapFlow.ov_capflow import DNQueryBudgetBatchSampler


CONFIG_DIR = Path('configs/ov_capflow/dotav2')
CONTROL = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d12_control.py')
CANDIDATE = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d12_xywh.py')
CONTROL_CHECKPOINT = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth')
CANDIDATE_CHECKPOINT = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth')


def _load(name):
    return Config.fromfile(CONFIG_DIR / name).to_dict()


def test_d12_world5_configs_have_frozen_matched_runtime_protocol():
    control = _load(CONTROL)
    candidate = _load(CANDIDATE)

    assert tuple(control['physical_gpus']) == (5, 6, 7, 8, 9)
    assert tuple(candidate['physical_gpus']) == (0, 1, 2, 3, 4)
    assert control['selected_world_size'] == 5
    assert candidate['selected_world_size'] == 5
    for cfg in (control, candidate):
        assert cfg['model']['num_queries'] == 600
        assert cfg['model']['train_query_groups'] == 3
        assert cfg['model']['bbox_head']['matching_query_groups'] == 3
        assert cfg['model']['decoder']['num_layers'] == 6
        assert cfg['train_dataloader']['batch_size'] == 2
        assert cfg['train_dataloader']['batch_sampler']['type'] == (
            'DNQueryBudgetBatchSampler')
        assert cfg['train_dataloader']['batch_sampler'][
            'num_matching_queries'] == 1800
        assert cfg['train_dataloader']['batch_sampler'][
            'num_dn_queries'] == 100
        assert cfg['train_dataloader']['batch_sampler'][
            'max_query_area'] == 50_000_000
        assert cfg['train_dataloader']['batch_sampler'][
            'update_count_multiple'] == 1
        assert cfg['optim_wrapper']['accumulative_counts'] == 1
        assert (
            cfg['train_dataloader']['batch_size']
            * cfg['selected_world_size']
            * cfg['optim_wrapper']['accumulative_counts']
        ) == 10
        assert cfg['randomness'] == {
            'seed': 20260716,
            'deterministic': False,
            'diff_rank_seed': False,
        }
        assert cfg['resume'] is False
        checkpoint = cfg['default_hooks']['checkpoint']
        assert checkpoint['by_epoch'] is True
        assert checkpoint['interval'] == 1
        assert checkpoint['max_keep_ckpts'] == -1
        assert checkpoint['save_best'] is None
        assert checkpoint['save_last'] is True

    assert divmod(1600, 2 * 5) == (160, 0)
    assert control['load_from'] == CONTROL_CHECKPOINT
    assert candidate['load_from'] == CANDIDATE_CHECKPOINT
    assert control['d12_pair_id'] == candidate['d12_pair_id']
    assert control['d12_role'] == 'control'
    assert candidate['d12_role'] == 'candidate'
    assert control['d12_only_scientific_delta'] == 'none'
    assert candidate['d12_only_scientific_delta'] == (
        'initialize_decoder_branches_0_5_terminal_xywh_from_raw_ogc')


def test_d12_candidate_has_only_approved_pair_differences():
    control = _load(CONTROL)
    candidate = _load(CANDIDATE)

    allowed_top_level = {
        'physical_gpus',
        'd12_role',
        'd12_only_scientific_delta',
        'load_from',
        'work_dir',
    }
    for key in allowed_top_level:
        control.pop(key)
        candidate.pop(key)
    control['train_dataloader']['batch_sampler'].pop('audit_path')
    candidate['train_dataloader']['batch_sampler'].pop('audit_path')

    assert candidate == control


class _Dataset:

    def __init__(self, size):
        self.data_list = [
            {'instances': [object()] * (index % 7)}
            for index in range(size)
        ]

    def __len__(self):
        return len(self.data_list)


class _RankSampler:

    def __init__(self, dataset, rank, world_size, seed):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = True
        self.epoch = 0


def test_d12_world5_sampler_exactly_covers_1600_without_padding():
    cfg = _load(CONTROL)
    dataset = _Dataset(1600)
    batch_cfg = cfg['train_dataloader']['batch_sampler']
    rank_batches = []
    reports = []
    for rank in range(5):
        sampler = DNQueryBudgetBatchSampler(
            sampler=_RankSampler(
                dataset=dataset,
                rank=rank,
                world_size=5,
                seed=cfg['randomness']['seed']),
            batch_size=cfg['train_dataloader']['batch_size'],
            num_matching_queries=batch_cfg['num_matching_queries'],
            num_dn_queries=batch_cfg['num_dn_queries'],
            max_query_area=batch_cfg['max_query_area'],
            update_count_multiple=batch_cfg['update_count_multiple'])
        batches, report = sampler._build_plan()
        rank_batches.append([update[rank] for update in batches])
        reports.append(report)

    assert all(len(batches) == 160 for batches in rank_batches)
    assert all(
        len(batch) == 2
        for batches in rank_batches
        for batch in batches)
    assert all(
        len({index for batch in batches for index in batch}) == 320
        for batches in rank_batches)
    coverage = [
        index
        for update in range(160)
        for rank in range(5)
        for index in rank_batches[rank][update]
    ]
    assert len(coverage) == len(set(coverage)) == 1600
    assert set(coverage) == set(range(1600))
    assert all(report['update_count'] == 160 for report in reports)
    assert all(report['duplicate_count'] == 0 for report in reports)
    assert all(report['missing_count'] == 0 for report in reports)
    assert all(report['global_batch_size_min'] == 10 for report in reports)
    assert all(report['global_batch_size_max'] == 10 for report in reports)
    assert all(
        report['max_estimated_query_area'] <= 50_000_000
        for report in reports)
    assert len({report['coverage_checksum'] for report in reports}) == 1
