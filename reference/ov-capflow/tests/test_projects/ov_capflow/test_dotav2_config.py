import copy
from pathlib import Path

from mmengine import Config


CONFIG_DIR = Path('configs/ov_capflow/dotav2')
C0_NAME = 'ov_capflow_swin-t_dotav2_c0_native_1e.py'
C1_NAME = 'ov_capflow_swin-t_dotav2_c1_parent_preserving_1e.py'
EXPECTED_CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')


def _load(name):
    return Config.fromfile(CONFIG_DIR / name)


def _normalize_intervention(cfg):
    effective = copy.deepcopy(cfg.to_dict())
    layer_cfg = effective['model']['decoder']['layer_cfg']
    layer_cfg.pop('enable_semantic_fusion')
    layer_cfg.pop('semantic_fusion_cfg', None)
    for key in ('load_from', 'custom_hooks', 'work_dir'):
        effective.pop(key, None)
    return effective


def test_dotav2_configs_share_fixed_protocol():
    c0 = _load(C0_NAME)
    c1 = _load(C1_NAME)

    for cfg in (c0, c1):
        assert cfg.model.type == 'OVCapFlow'
        assert cfg.model.num_queries == 200
        assert cfg.model.encoder.num_cp == 0
        assert cfg.model_wrapper_cfg.find_unused_parameters is True
        assert cfg.model_wrapper_cfg.static_graph is True
        assert cfg.model_wrapper_cfg.broadcast_buffers is False
        assert cfg.train_cfg.type == 'EpochBasedTrainLoop'
        assert cfg.train_cfg.max_epochs == 1
        assert cfg.randomness.seed == 20260712
        assert cfg.train_dataloader.batch_size == 6
        assert cfg.train_dataloader.sampler.type == 'DefaultSampler'
        assert cfg.train_dataloader.sampler.round_up is False
        assert (cfg.train_dataloader.batch_sampler.type
                == 'DNQueryBudgetBatchSampler')
        assert cfg.train_dataloader.dataset.type == 'DOTAv2Dataset'
        assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
        assert cfg.val_dataloader.dataset.filter_cfg.filter_empty_gt is False
        assert cfg.train_dataloader.dataset.ann_file == 'ss_train/annfiles/'
        assert cfg.val_dataloader.dataset.ann_file == 'ss_val/annfiles/'
        assert tuple(cfg.val_dataloader.dataset.metainfo.classes) == (
            EXPECTED_CLASSES)
        assert cfg.val_evaluator.type == 'DOTAMetric'
        assert cfg.val_evaluator.iou_thrs == 0.5
        assert cfg.visualizer.vis_backends == [dict(type='LocalVisBackend')]


def test_dotav2_c1_is_parent_preserving_single_intervention():
    c0 = _load(C0_NAME)
    c1 = _load(C1_NAME)

    assert not c0.model.decoder.layer_cfg.enable_semantic_fusion
    assert c1.model.decoder.layer_cfg.enable_semantic_fusion
    assert c1.model.decoder.layer_cfg.semantic_fusion_cfg == dict(
        adapter_init='identity')
    assert not c1.model.bbox_head.balanced_cfg.enabled
    assert not c1.model.decoder.enable_null_reservoir
    assert not c1.model.decoder.layer_cfg.enable_density_capacity
    assert c0.work_dir.endswith('dotav2_c0_native_1e_recovery_static')
    assert c1.load_from.endswith(
        'dotav2_c0_native_1e_recovery_static/epoch_1.pth')
    assert c1.custom_hooks == [dict(
        type='FreezeExceptHook',
        trainable_patterns=[r'^decoder\.layers\.\d+\.semantic_fusion\.'])]
    assert _normalize_intervention(c1) == _normalize_intervention(c0)
