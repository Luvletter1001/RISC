import copy
import re
import types
from collections import Counter
from pathlib import Path

import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmrotate.registry import DATASETS, MODELS
from mmrotate.utils import register_all_modules

from projects.OVCapFlow.ov_capflow import DNQueryBudgetBatchSampler


CONFIG_DIR = Path('configs/ov_capflow/dotav2')
CONTROL = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_control.py')
CANDIDATE = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_candidate.py')
RAW_CONTROL = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_control_raw13833.py')
RAW_CANDIDATE = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_candidate_raw13833.py')
E24_RAW_MOUTH = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_'
    'scale1024_rare4x_gpu89_batch2.py')
PROXY_RECIPE = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py')
PARENT = (
    'work_dirs/dotav2_cleanstart/'
    'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/'
    'epoch_24.pth')
PARENT_SHA256 = (
    'a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8')
TRAIN_MANIFEST_SHA256 = (
    '1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290')
PROXY_MANIFEST_SHA256 = (
    'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e')
CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
FREEZE_PATTERNS = [r'^bbox_head\.existence_residual\.(weight|bias)$']
TRAIN_ROOT = (
    '/data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/'
    'subsets/seed20260715_rare4x/train/')
PROXY_ROOT = (
    '/data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/'
    'subsets/seed20260715/')
TRAIN_PIPELINE = [
    {'type': 'mmdet.LoadImageFromFile'},
    {'type': 'mmdet.LoadAnnotations', 'with_bbox': True, 'box_type': 'qbox'},
    {
        'type': 'ConvertBoxType',
        'box_type_mapping': {'gt_bboxes': 'rbox'},
    },
    {'type': 'mmdet.Resize', 'scale': (1024, 1024), 'keep_ratio': True},
    {
        'type': 'mmdet.FilterAnnotations',
        'min_gt_bbox_wh': (0.01, 0.01),
    },
    {
        'type': 'mmdet.RandomFlip',
        'prob': 0.75,
        'direction': ['horizontal', 'vertical', 'diagonal'],
    },
    {
        'type': 'mmdet.PackDetInputs',
        'meta_keys': (
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'text', 'custom_entities'),
    },
]
VAL_PIPELINE = [
    {'type': 'mmdet.LoadImageFromFile'},
    {'type': 'mmdet.Resize', 'scale': (1024, 1024), 'keep_ratio': True},
    {'type': 'mmdet.LoadAnnotations', 'with_bbox': True, 'box_type': 'qbox'},
    {
        'type': 'ConvertBoxType',
        'box_type_mapping': {'gt_bboxes': 'rbox'},
    },
    {
        'type': 'mmdet.PackDetInputs',
        'meta_keys': (
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'text', 'custom_entities'),
    },
]
EXPECTED_PATHS = {
    CONTROL: {
        'work_dir': (
            'work_dirs/dotav2_cleanstart/'
            'd13n_world5_control_seed20260716_gpu56789_batch2'),
        'd13n_audit_path': (
            'work_dirs/dotav2_cleanstart/audits/'
            'd13n_world5_control_seed20260716_gpu56789_'
            'epoch_{epoch:02d}.json'),
    },
    CANDIDATE: {
        'work_dir': (
            'work_dirs/dotav2_cleanstart/'
            'd13n_world5_candidate_seed20260716_gpu56789_batch2'),
        'd13n_audit_path': (
            'work_dirs/dotav2_cleanstart/audits/'
            'd13n_world5_candidate_seed20260716_gpu56789_'
            'epoch_{epoch:02d}.json'),
    },
    RAW_CONTROL: {
        'work_dir': (
            'work_dirs/dotav2_cleanstart/'
            'd13n_raw13833_control_gpu56789'),
        'd13n_audit_path': (
            'work_dirs/dotav2_cleanstart/audits/'
            'd13n_raw13833_control_gpu56789_epoch_{epoch:02d}.json'),
    },
    RAW_CANDIDATE: {
        'work_dir': (
            'work_dirs/dotav2_cleanstart/'
            'd13n_raw13833_candidate_gpu56789'),
        'd13n_audit_path': (
            'work_dirs/dotav2_cleanstart/audits/'
            'd13n_raw13833_candidate_gpu56789_epoch_{epoch:02d}.json'),
    },
}


def _load(name):
    return Config.fromfile(CONFIG_DIR / name).to_dict()


def _assert_disabled(model):
    assert model['decoder']['enable_null_reservoir'] is False
    assert model['decoder']['layer_cfg']['enable_semantic_fusion'] is False
    assert model['decoder']['layer_cfg']['enable_density_capacity'] is False
    assert model['bbox_head']['balanced_cfg'] == {'enabled': False}
    assert model['bbox_head']['position_supervised_cfg'] == {'enabled': False}
    assert model['bbox_head']['adaptive_dn_cfg'] == {'enabled': False}
    assert model['bbox_head']['readout_cfg'] == {
        'temperature': 1.0,
        'power': 1.0,
        'use_capacity': False,
    }
    assert model['density_loss_cfg'] == {'weight': 0.0}
    assert model['null_loss_cfg'] == {}


def _normalize_arm(cfg):
    cfg = copy.deepcopy(cfg)
    for key in ('d13n_role', 'd13n_master_port', 'd13n_audit_path',
                'work_dir'):
        cfg.pop(key)
    cfg['model']['bbox_head'].pop('existence_loss_weight')
    cfg['custom_hooks'][0].pop('role')
    if cfg.get('train_dataloader') is not None:
        cfg['train_dataloader']['batch_sampler'].pop('audit_path')
    return cfg


def test_d13n_world5_training_configs_freeze_the_runtime_contract():
    control = _load(CONTROL)
    candidate = _load(CANDIDATE)

    for cfg, role, coefficient, port in (
            (control, 'control', 0.0, 29842),
            (candidate, 'candidate', 1.0, 29841)):
        assert tuple(cfg['physical_gpus']) == (5, 6, 7, 8, 9)
        assert cfg['selected_world_size'] == 5
        assert cfg['d13n_role'] == role
        assert cfg['d13n_mouth'] == 'proxy400'
        assert cfg['d13n_test_only'] is False
        assert cfg['d13n_master_port'] == port
        assert cfg['d13n_parent_sha256'] == PARENT_SHA256
        assert cfg['d13n_train_manifest_sha256'] == TRAIN_MANIFEST_SHA256
        assert cfg['d13n_proxy_manifest_sha256'] == PROXY_MANIFEST_SHA256
        assert cfg['load_from'] == PARENT
        assert cfg['resume'] is False
        assert cfg['randomness'] == {
            'seed': 20260716,
            'deterministic': False,
            'diff_rank_seed': False,
        }

        model = cfg['model']
        assert model['num_queries'] == 600
        assert model['train_query_groups'] == 3
        assert model['bbox_head']['matching_query_groups'] == 3
        assert model['decoder']['num_layers'] == 6
        assert model['bbox_head']['existence_loss_weight'] == coefficient
        assert model['freeze_except_patterns'] == FREEZE_PATTERNS
        _assert_disabled(model)

        loader = cfg['train_dataloader']
        sampler = loader['batch_sampler']
        assert loader['batch_size'] == 2
        assert loader['sampler'] == {
            'type': 'DefaultSampler',
            'shuffle': True,
            'round_up': False,
        }
        assert sampler['type'] == 'DNQueryBudgetBatchSampler'
        assert sampler['num_matching_queries'] == 1800
        assert sampler['num_dn_queries'] == 100
        assert sampler['max_query_area'] == 50_000_000
        assert sampler['update_count_multiple'] == 1
        assert sampler['audit_noreplace'] is True
        assert sampler['audit_path'] == cfg['d13n_audit_path']
        assert '{epoch:02d}' in sampler['audit_path']
        assert cfg['d13n_updates_per_epoch'] == 160
        assert cfg['d13n_total_updates'] == 1920
        assert cfg['train_cfg'] == {
            'type': 'VariableBatchEpochBasedTrainLoop',
            'max_epochs': 12,
            'val_interval': 1,
        }
        assert cfg['param_scheduler'] == [
            {
                'type': 'LinearLR',
                'start_factor': 0.1,
                'by_epoch': False,
                'begin': 0,
                'end': 500,
            },
            {
                'type': 'CosineAnnealingLR',
                'by_epoch': True,
                'begin': 0,
                'end': 12,
                'eta_min': 1e-6,
            },
        ]
        assert divmod(1600, loader['batch_size'] * 5) == (160, 0)

        train_dataset = loader['dataset']
        assert train_dataset['type'] == 'DOTAv2Dataset'
        assert train_dataset['data_root'] == TRAIN_ROOT
        assert train_dataset['ann_file'] == 'annfiles/'
        assert train_dataset['data_prefix'] == {'img_path': 'images/'}
        assert tuple(train_dataset['metainfo']['classes']) == CLASSES
        assert train_dataset['filter_cfg'] == {'filter_empty_gt': False}
        assert train_dataset['pipeline'] == TRAIN_PIPELINE
        assert train_dataset['return_classes'] is True

        proxy_mouth = cfg['val_dataloader']
        assert cfg['test_dataloader'] == proxy_mouth
        assert proxy_mouth['dataset']['type'] == 'DOTAv2Dataset'
        assert proxy_mouth['dataset']['data_root'] == PROXY_ROOT
        assert proxy_mouth['dataset']['ann_file'] == (
            's1_val_all18/annfiles/')
        assert proxy_mouth['dataset']['data_prefix'] == {
            'img_path': 's1_val_all18/images/'}
        assert tuple(proxy_mouth['dataset']['metainfo']['classes']) == CLASSES
        assert proxy_mouth['dataset']['filter_cfg'] == {
            'filter_empty_gt': False}
        assert proxy_mouth['dataset']['test_mode'] is True
        assert proxy_mouth['dataset']['pipeline'] == VAL_PIPELINE
        assert proxy_mouth['dataset']['return_classes'] is True
        assert cfg['val_evaluator'] == {
            'type': 'DOTAMetric',
            'metric': 'mAP',
            '_scope_': 'mmrotate',
            'iou_thrs': 0.5,
        }
        assert cfg['test_evaluator'] == cfg['val_evaluator']

        optim = cfg['optim_wrapper']
        assert optim == {
            'type': 'OptimWrapper',
            'constructor': 'D13NOptimWrapperConstructor',
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.0001,
                'weight_decay': 0.0001,
            },
            'clip_grad': {'max_norm': 0.1, 'norm_type': 2},
            'accumulative_counts': 1,
        }
        assert 'paramwise_cfg' not in optim
        assert loader['batch_size'] * 5 * optim['accumulative_counts'] == 10

        checkpoint = cfg['default_hooks']['checkpoint']
        assert checkpoint['by_epoch'] is True
        assert checkpoint['interval'] == 1
        assert checkpoint['max_keep_ckpts'] == -1
        assert checkpoint['save_best'] is None
        assert checkpoint['rule'] is None
        assert checkpoint['save_last'] is True
        assert cfg['custom_hooks'] == [
            {'type': 'D13NParentEvalModeHook', 'role': role}]

    assert control['work_dir'] != candidate['work_dir']
    assert control['d13n_audit_path'] != candidate['d13n_audit_path']
    assert _normalize_arm(control) == _normalize_arm(candidate)


def test_d13n_raw_configs_are_test_only_epoch12_mouths():
    control = _load(RAW_CONTROL)
    candidate = _load(RAW_CANDIDATE)
    training_arms = {
        'control': _load(CONTROL),
        'candidate': _load(CANDIDATE),
    }
    canonical_mouth = _load(E24_RAW_MOUTH)['test_dataloader']

    for cfg, role, hook_role, coefficient, port in (
            (control, 'raw-control', 'control', 0.0, 29843),
            (candidate, 'raw-candidate', 'candidate', 1.0, 29844)):
        assert tuple(cfg['physical_gpus']) == (5, 6, 7, 8, 9)
        assert cfg['selected_world_size'] == 5
        assert cfg['d13n_role'] == role
        assert cfg['d13n_mouth'] == 'raw13833'
        assert cfg['d13n_test_only'] is True
        assert cfg['d13n_master_port'] == port
        assert cfg['d13n_checkpoint_arg_required'] is True
        assert cfg['d13n_checkpoint_epoch'] == 12
        assert cfg['d13n_parent_sha256'] == PARENT_SHA256
        assert cfg['load_from'] == PARENT
        assert cfg['train_cfg'] is None
        assert cfg['train_dataloader'] is None
        assert cfg['optim_wrapper'] is None
        assert cfg['param_scheduler'] is None
        assert cfg['model']['train_cfg'] == training_arms[hook_role]['model'][
            'train_cfg']
        assert cfg['custom_hooks'] == [
            {'type': 'D13NParentEvalModeHook', 'role': hook_role}]

        model = cfg['model']
        assert model['num_queries'] == 600
        assert model['train_query_groups'] == 3
        # The configured training topology remains valid. Evaluation emits
        # the one fixed 600-query output group; setting this field to 1 would
        # make the D13-N head constructor fail closed.
        assert model['bbox_head']['matching_query_groups'] == 3
        assert cfg['d13n_inference_groups'] == 1
        assert cfg['d13n_output_queries'] == 600
        assert model['bbox_head']['existence_loss_weight'] == coefficient
        assert model['freeze_except_patterns'] == FREEZE_PATTERNS
        assert model['test_cfg'] == {}
        _assert_disabled(model)

        assert cfg['test_dataloader'] == cfg['val_dataloader']
        assert cfg['test_dataloader'] == canonical_mouth
        dataset = cfg['test_dataloader']['dataset']
        assert dataset['data_root'] == '/data1/zcy/datasets/DOTA2_1024_500/'
        assert dataset['ann_file'] == 'ss_val/annfiles/'
        assert dataset['data_prefix'] == {'img_path': 'ss_val/images/'}
        assert dataset['filter_cfg'] == {'filter_empty_gt': False}
        assert dataset['test_mode'] is True
        assert tuple(dataset['metainfo']['classes']) == CLASSES
        assert dataset['return_classes'] is True
        pipeline = dataset['pipeline']
        assert pipeline[0] == {'type': 'mmdet.LoadImageFromFile'}
        assert pipeline[1] == {
            'type': 'mmdet.Resize',
            'scale': (1024, 1024),
            'keep_ratio': True,
        }
        assert pipeline[2] == {
            'type': 'mmdet.LoadAnnotations',
            'with_bbox': True,
            'box_type': 'qbox',
        }
        assert pipeline[3] == {
            'type': 'ConvertBoxType',
            'box_type_mapping': {'gt_bboxes': 'rbox'},
        }
        assert pipeline[4]['type'] == 'mmdet.PackDetInputs'

    assert control['work_dir'] != candidate['work_dir']
    assert control['d13n_audit_path'] != candidate['d13n_audit_path']
    assert _normalize_arm(control) == _normalize_arm(candidate)


def test_d13n_each_raw_config_is_an_exact_test_only_training_arm_transform():
    canonical_raw_mouth = _load(E24_RAW_MOUTH)['test_dataloader']
    arm_specs = (
        (CONTROL, RAW_CONTROL, 'raw-control', 29843),
        (CANDIDATE, RAW_CANDIDATE, 'raw-candidate', 29844),
    )
    for training_name, raw_name, raw_role, raw_port in arm_specs:
        expected = copy.deepcopy(_load(training_name))
        expected.update({
            'd13n_role': raw_role,
            'd13n_mouth': 'raw13833',
            'd13n_test_only': True,
            'd13n_master_port': raw_port,
            'd13n_checkpoint_arg_required': True,
            'd13n_checkpoint_epoch': 12,
            'd13n_inference_groups': 1,
            'd13n_output_queries': 600,
            'd13n_audit_path': EXPECTED_PATHS[raw_name][
                'd13n_audit_path'],
            'work_dir': EXPECTED_PATHS[raw_name]['work_dir'],
            'val_dataloader': copy.deepcopy(canonical_raw_mouth),
            'test_dataloader': copy.deepcopy(canonical_raw_mouth),
            'train_cfg': None,
            'train_dataloader': None,
            'optim_wrapper': None,
            'param_scheduler': None,
        })
        actual = _load(raw_name)
        assert actual == expected
        assert actual['model']['train_cfg'] == _load(training_name)['model'][
            'train_cfg']


def test_d13n_all_eight_work_and_audit_paths_are_exact_and_distinct():
    configs = {
        name: _load(name)
        for name in (CONTROL, CANDIDATE, RAW_CONTROL, RAW_CANDIDATE)
    }
    for name, expected in EXPECTED_PATHS.items():
        assert configs[name]['work_dir'] == expected['work_dir']
        assert configs[name]['d13n_audit_path'] == expected[
            'd13n_audit_path']

    work_dirs = [cfg['work_dir'] for cfg in configs.values()]
    audit_paths = [cfg['d13n_audit_path'] for cfg in configs.values()]
    assert len(work_dirs) == len(set(work_dirs)) == 4
    assert len(audit_paths) == len(set(audit_paths)) == 4
    assert all('{epoch:02d}' in path for path in audit_paths)


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


def test_d13n_world5_sampler_exactly_covers_1600_without_padding():
    cfg = _load(CONTROL)
    dataset = _Dataset(1600)
    batch_cfg = cfg['train_dataloader']['batch_sampler']
    rank_batches = []
    reports = []
    for rank in range(5):
        # Task 7 owns audit_noreplace support. This Task 6 fixture passes only
        # arguments supported by the current sampler implementation.
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
    assert len({report['coverage_checksum'] for report in reports}) == 1


def test_d13n_real_proxy_manifest_has_only_preregistered_rare_repeats():
    cfg = _load(CONTROL)
    init_default_scope('mmrotate')
    dataset = DATASETS.build(cfg['train_dataloader']['dataset'])
    image_ids = [
        dataset.get_data_info(index)['img_id']
        for index in range(len(dataset))
    ]
    assert len(image_ids) == len(set(image_ids)) == 1600
    assert all(type(image_id) is str for image_id in image_ids)

    suffix = re.compile(r'__rare_repeat_(?:01|02|03)$')
    normalized = [suffix.sub('', image_id) for image_id in image_ids]
    counts = Counter(normalized)
    assert len(counts) == 1492
    assert sum(count - 1 for count in counts.values()) == 108
    assert sum(count == 4 for count in counts.values()) == 36
    assert all(count in (1, 4) for count in counts.values())


def test_d13n_both_real_proxy400_mouths_match_recipe_and_native_ids():
    canonical = _load(PROXY_RECIPE)
    init_default_scope('mmrotate')
    mouths = []
    for config_name in (CONTROL, CANDIDATE):
        cfg = _load(config_name)
        assert cfg['val_dataloader'] == canonical['val_dataloader']
        assert cfg['test_dataloader'] == canonical['test_dataloader']
        assert cfg['val_evaluator'] == canonical['val_evaluator']
        assert cfg['test_evaluator'] == canonical['test_evaluator']
        dataset = DATASETS.build(cfg['val_dataloader']['dataset'])
        image_ids = [
            dataset.get_data_info(index)['img_id']
            for index in range(len(dataset))
        ]
        assert len(image_ids) == len(set(image_ids)) == 400
        assert all(type(image_id) is str for image_id in image_ids)
        assert all('__rare_repeat_' not in image_id for image_id in image_ids)
        mouths.append(cfg['val_dataloader'])
    assert mouths[0] == mouths[1]


def test_d13n_both_real_raw_datasets_have_13833_unique_string_ids():
    init_default_scope('mmrotate')
    mouths = []
    for config_name in (RAW_CONTROL, RAW_CANDIDATE):
        cfg = _load(config_name)
        dataset = DATASETS.build(cfg['test_dataloader']['dataset'])
        image_ids = [
            dataset.get_data_info(index)['img_id']
            for index in range(len(dataset))
        ]
        assert len(image_ids) == 13_833
        assert len(set(image_ids)) == 13_833
        assert all(type(image_id) is str for image_id in image_ids)
        mouths.append(cfg['test_dataloader'])
    assert mouths[0] == mouths[1]


def test_resolved_d13n_matching_group_three_uses_one_q600_eval_output():
    register_all_modules()
    outputs = []
    for config_name, coefficient in ((CONTROL, 0.0), (CANDIDATE, 1.0)):
        cfg = _load(config_name)
        head_cfg = copy.deepcopy(cfg['model']['bbox_head'])
        head_cfg['train_cfg'] = copy.deepcopy(cfg['model']['train_cfg'])
        head = MODELS.build(head_cfg)
        assert head.matching_query_groups == 3
        assert head.existence_loss_weight == coefficient

        query_count = cfg['model']['num_queries']
        cls_scores = torch.empty(1, query_count, 2)
        cls_scores[..., 0] = torch.linspace(-3.0, 1.0, query_count)
        cls_scores[..., 1] = torch.linspace(1.0, -3.0, query_count)
        bbox_preds = torch.zeros(1, query_count, 5)
        bbox_preds[..., 0] = torch.linspace(0.1, 0.9, query_count)
        bbox_preds[..., 1:4] = 0.5

        def forward(self, *args, **kwargs):
            return cls_scores.unsqueeze(0), bbox_preds.unsqueeze(0)

        head.forward = types.MethodType(forward, head)
        hidden_states = torch.zeros(2, 1, query_count + 3, 256)
        sample = types.SimpleNamespace(
            metainfo=dict(
                img_shape=(100, 200), scale_factor=(1.0, 1.0)),
            token_positive_map={1: [0], 2: [1]})
        prediction = head.predict(
            hidden_states,
            references=None,
            memory_text=None,
            text_token_mask=torch.ones(1, 2, dtype=torch.bool),
            batch_data_samples=[sample])

        assert len(prediction) == 1
        assert len(prediction[0]) == query_count == 600
        assert prediction[0].scores.shape == (600, )
        assert prediction[0].labels.shape == (600, )
        assert prediction[0].bboxes.shape == (600, 5)
        assert head.last_existence_logits.shape == (1, 600)
        outputs.append(prediction[0])

    assert torch.equal(outputs[0].scores, outputs[1].scores)
    assert torch.equal(outputs[0].labels, outputs[1].labels)
    assert torch.equal(outputs[0].bboxes, outputs[1].bboxes)
