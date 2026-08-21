import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import DATASETS, HOOKS, MODELS
from mmrotate.utils import register_all_modules

from projects.OVCapFlow.ov_capflow import DNQueryBudgetBatchSampler


CONFIG_DIR = Path('configs/ov_capflow/dotav2')
_EXPLICIT_CONFIG = os.getenv('OVCAPFLOW_DOTA2_CONFIG')
REAL_BATCH_CONFIGS = (
    [Path(_EXPLICIT_CONFIG)] if _EXPLICIT_CONFIG else [
        CONFIG_DIR / 'ov_capflow_swin-t_dotav2_c0_native_1e.py',
        CONFIG_DIR / 'ov_capflow_swin-t_dotav2_c1_parent_preserving_1e.py',
        CONFIG_DIR / 'ov_capflow_swin-t_dotav2_cleanstart_q600_base.py',
        CONFIG_DIR / 'ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py',
        CONFIG_DIR / 'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py',
    ])


class _RankSamplerView:

    def __init__(self, dataset, rank, world_size, seed, shuffle):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0


def _raw_batch_with_gt_counts(*gt_counts):
    return {
        'data_samples': [
            SimpleNamespace(gt_instances=[object()] * gt_count)
            for gt_count in gt_counts
        ]
    }


def _first_raw_batch_with_gt(dataloader):
    for _, raw_batch in zip(range(len(dataloader)), dataloader):
        total_gt = sum(
            len(sample.gt_instances)
            for sample in raw_batch['data_samples'])
        if total_gt > 0:
            return raw_batch, total_gt
    raise AssertionError('no non-empty GT batch found in dataloader')


def test_first_raw_batch_with_gt_selects_first_nonempty_batch():
    empty_batch = _raw_batch_with_gt_counts(0)
    first_nonempty_batch = _raw_batch_with_gt_counts(0, 2)
    later_nonempty_batch = _raw_batch_with_gt_counts(1)

    selected, total_gt = _first_raw_batch_with_gt([
        empty_batch,
        first_nonempty_batch,
        later_nonempty_batch,
    ])

    assert selected is first_nonempty_batch
    assert total_gt == 2


def test_first_raw_batch_with_gt_stops_at_dataloader_length():

    class _BoundedViewOfInfiniteEmptyBatches:

        def __init__(self):
            self.yield_count = 0

        def __len__(self):
            return 2

        def __iter__(self):
            while True:
                self.yield_count += 1
                yield _raw_batch_with_gt_counts(0)

    dataloader = _BoundedViewOfInfiniteEmptyBatches()

    with pytest.raises(AssertionError, match='no non-empty GT batch'):
        _first_raw_batch_with_gt(dataloader)

    assert dataloader.yield_count == 2


@pytest.mark.skipif(
    (os.getenv('OVCAPFLOW_RUN_DOTA2_REAL_BATCH') != '1' and
     os.getenv('RUN_OVCAPFLOW_DOTAV2_INTEGRATION') != '1'),
    reason='real DOTA2 integration is opt-in')
@pytest.mark.parametrize('config_path', REAL_BATCH_CONFIGS)
def test_real_dotav2_batch_forward_backward(config_path):
    assert torch.cuda.is_available()
    cfg = Config.fromfile(config_path)
    position_supervised = bool(
        cfg.model.bbox_head.get(
            'position_supervised_cfg', {}).get('enabled', False))
    if position_supervised:
        assert cfg.model.num_queries == 600
        assert cfg.model.train_query_groups == 3
        assert cfg.model.bbox_head.matching_query_groups == 3
        assert cfg.model.decoder.num_layers == 6
        assert not cfg.model.bbox_head.get(
            'balanced_cfg', {}).get('enabled', False)
    if os.getenv('OVCAPFLOW_PREFLIGHT_BATCH'):
        cfg.train_dataloader.batch_size = int(
            os.environ['OVCAPFLOW_PREFLIGHT_BATCH'])
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model).cuda().train()
    if 'cleanstart' in str(config_path):
        checkpoint = torch.load(cfg.load_from, map_location='cpu')
        incompatible = model.load_state_dict(
            checkpoint['state_dict'], strict=False)
        assert not incompatible.unexpected_keys
        expected_missing = {
            'bbox_head.cls_branches.0.bias',
            'bbox_head.cls_branches.1.bias',
            'bbox_head.cls_branches.2.bias',
            'bbox_head.cls_branches.3.bias',
            'bbox_head.cls_branches.4.bias',
            'bbox_head.cls_branches.5.bias',
            'bbox_head.cls_branches.6.bias',
            'bbox_head.reg_branches.0.4.weight',
            'bbox_head.reg_branches.0.4.bias',
            'bbox_head.reg_branches.1.4.weight',
            'bbox_head.reg_branches.1.4.bias',
            'bbox_head.reg_branches.2.4.weight',
            'bbox_head.reg_branches.2.4.bias',
            'bbox_head.reg_branches.3.4.weight',
            'bbox_head.reg_branches.3.4.bias',
            'bbox_head.reg_branches.4.4.weight',
            'bbox_head.reg_branches.4.4.bias',
            'bbox_head.reg_branches.5.4.weight',
            'bbox_head.reg_branches.5.4.bias',
            'bbox_head.reg_branches.6.4.weight',
            'bbox_head.reg_branches.6.4.bias',
            'query_initializer.query_embedding.weight',
            'query_initializer.reference_embedding.weight',
            'dn_query_generator.label_embedding.weight',
        }
        if (cfg.get('d11_role') == 'candidate' or
                cfg.get('d11_v2_role') == 'candidate'):
            expected_missing.remove(
                'query_initializer.query_embedding.weight')
        assert set(incompatible.missing_keys) == expected_missing

    expected_trainable = None
    if 'parent_preserving' in str(config_path):
        freeze_cfg = next(
            item for item in cfg.custom_hooks
            if item.get('type') == 'FreezeExceptHook')
        expected_trainable = HOOKS.build(freeze_cfg).apply(model)
        assert expected_trainable
        assert all('.semantic_fusion.' in name
                   for name in expected_trainable)

    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    batch, total_gt = _first_raw_batch_with_gt(dataloader)
    assert total_gt > 0
    batch = model.data_preprocessor(batch, training=True)
    losses = model.loss(batch['inputs'], batch['data_samples'])
    if position_supervised:
        for loss_name in ('loss_cls', 'dn_loss_cls'):
            assert loss_name in losses
            loss_value = losses[loss_name]
            assert torch.is_tensor(loss_value)
            assert torch.isfinite(loss_value).all()
            assert loss_value.requires_grad
    tensor_losses = [
        value for value in losses.values() if torch.is_tensor(value)
    ]
    assert tensor_losses
    assert all(torch.isfinite(value).all() for value in tensor_losses)
    total = sum(value for value in tensor_losses if value.requires_grad)
    assert torch.isfinite(total).all()
    total.backward()

    active = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert active
    finite_gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert finite_gradients
    assert torch.stack(finite_gradients).all()
    if expected_trainable is not None:
        assert set(active) <= set(expected_trainable)
        assert all('.semantic_fusion.' in name for name in active)
    else:
        assert model.bbox_head.last_balance_stats is None
        assert 'loss_capacity_mass' not in losses

    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            batch['inputs'], batch['data_samples'], rescale=True)
    assert len(predictions) == len(batch['data_samples'])
    assert all(
        len(sample.pred_instances) == cfg.model.num_queries
        for sample in predictions)
    for sample in predictions:
        pred_instances = sample.pred_instances
        assert (pred_instances.scores.shape == pred_instances.labels.shape ==
                (cfg.model.num_queries, ))
        assert pred_instances.bboxes.shape == (cfg.model.num_queries, 5)
        assert torch.isfinite(pred_instances.scores).all()
        assert torch.isfinite(pred_instances.bboxes).all()


@pytest.mark.skipif(
    os.getenv('RUN_OVCAPFLOW_DOTAV2_PREFLIGHT') != '1',
    reason='full DOTA2 sampler preflight is opt-in')
def test_full_dotav2_four_rank_sampler_preflight():
    cfg = Config.fromfile(CONFIG_DIR /
                          'ov_capflow_swin-t_dotav2_c0_native_1e.py')
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    assert len(dataset) == 47294

    batch_cfg = cfg.train_dataloader.batch_sampler
    audit_path = Path(
        'work_dirs/ov_capflow_dotav2/audits/sampler_preflight.json')
    rank_batches = []
    for rank in range(4):
        sampler = _RankSamplerView(
            dataset=dataset,
            rank=rank,
            world_size=4,
            seed=cfg.randomness.seed,
            shuffle=True)
        batch_sampler = DNQueryBudgetBatchSampler(
            sampler=sampler,
            batch_size=cfg.train_dataloader.batch_size,
            num_matching_queries=batch_cfg.num_matching_queries,
            num_dn_queries=batch_cfg.num_dn_queries,
            max_query_area=batch_cfg.max_query_area,
            audit_path=str(audit_path))
        rank_batches.append([list(batch) for batch in batch_sampler])

    assert len({len(batches) for batches in rank_batches}) == 1
    assert all(batch for batches in rank_batches for batch in batches)
    all_indices = [
        index for batches in rank_batches for batch in batches
        for index in batch
    ]
    assert sorted(all_indices) == list(range(47294))
    assert len(all_indices) == len(set(all_indices))

    report = json.loads(audit_path.read_text(encoding='utf-8'))
    assert report['dataset_size'] == 47294
    assert report['duplicate_count'] == 0
    assert report['missing_count'] == 0
    assert report['local_batch_size_min'] >= 1
    assert report['local_batch_size_max'] <= 6


@pytest.mark.skipif(
    os.getenv('RUN_OVCAPFLOW_DOTAV2_WORST_BATCH') != '1',
    reason='worst DOTA2 query-area batch is opt-in')
@pytest.mark.parametrize('config_name', [
    'ov_capflow_swin-t_dotav2_c0_native_1e.py',
    'ov_capflow_swin-t_dotav2_c1_parent_preserving_1e.py',
    'ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py',
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024.py',
    ('ov_capflow_swin-t_dotav2_cleanstart_q600_'
     'full24e_scale1024_rare4x_gpu89_batch2.py'),
])
def test_worst_dotav2_query_area_batch_forward_backward(config_name):
    assert torch.cuda.is_available()
    cfg = Config.fromfile(CONFIG_DIR / config_name)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    if 'full24e_scale1024_rare4x' in config_name:
        audit_path = Path(
            'work_dirs/dotav2_cleanstart/audits/'
            'full24_scale1024_rare4x_seed20260716_gpu89_b2_epochs/'
            'epoch_0.json')
    elif 'scale1024' in config_name:
        audit_path = Path(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_epochs/epoch_0.json')
    elif 'full24e' in config_name:
        audit_path = Path(
            'work_dirs/dotav2_cleanstart/audits/'
            'full24_grouped_sampler_world4.json')
    else:
        audit_path = Path(
            'work_dirs/ov_capflow_dotav2/audits/sampler_preflight.json')
    report = json.loads(audit_path.read_text(encoding='utf-8'))
    model = MODELS.build(cfg.model).cuda().train()
    if 'parent_preserving' in config_name:
        freeze_cfg = next(
            item for item in cfg.custom_hooks
            if item.get('type') == 'FreezeExceptHook')
        HOOKS.build(freeze_cfg).apply(model)
    max_gt = report['max_gt_sample']
    cases = [
        ('max_query_area', report['max_query_area_batch']['indices'],
         report['max_query_area_batch']['gt_counts']),
        ('max_gt_singleton', [max_gt['index']], [max_gt['gt_count']]),
    ]
    for case_name, indices, gt_counts in cases:
        raw_batch = pseudo_collate([dataset[index] for index in indices])
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        batch = model.data_preprocessor(raw_batch, training=True)
        losses = model.loss(batch['inputs'], batch['data_samples'])
        tensor_losses = [
            value for value in losses.values() if torch.is_tensor(value)
        ]
        assert tensor_losses
        assert all(torch.isfinite(value).all() for value in tensor_losses)
        sum(value for value in tensor_losses if value.requires_grad).backward()
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f'case={case_name} indices={indices} gt_counts={gt_counts} '
            f'peak_mib={peak_mib:.1f}')
        if 'scale1024' in config_name:
            assert peak_mib <= 42 * 1024
