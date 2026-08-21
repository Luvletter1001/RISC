from pathlib import Path

import pytest
from mmengine import Config
from mmrotate.registry import DATASETS
from mmrotate.utils import register_all_modules


CONFIG_DIR = Path('configs/ov_capflow/hrsc')
BERT_DIR = Path('/data1/zcy/LAEDINO/weights/bert-base-uncased')


@pytest.mark.parametrize('name,model_type', [
    ('grounding_dino_swin-t_hrsc_parent_10e.py', 'RotatedGroundingDINO'),
    ('ov_capflow_swin-t_hrsc_c0_native_10e.py', 'OVCapFlow'),
    ('ov_capflow_swin-t_hrsc_c1_fusion_5e.py', 'OVCapFlow'),
    ('ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py', 'OVCapFlow'),
    ('ov_capflow_swin-t_hrsc_c2_balanced_5e.py', 'OVCapFlow'),
    ('ov_capflow_swin-t_hrsc_c3_null_5e.py', 'OVCapFlow'),
])
def test_hrsc_configs_share_fixed_protocol(name, model_type):
    cfg = Config.fromfile(CONFIG_DIR / name)
    assert cfg.model.type == model_type
    assert cfg.model.num_queries == 600
    assert cfg.model.encoder.num_cp == 0
    assert Path(cfg.model.language_model.name) == BERT_DIR
    assert (BERT_DIR / 'config.json').is_file()
    assert (BERT_DIR / 'tokenizer.json').is_file()
    assert (BERT_DIR / 'model.safetensors').is_file()
    assert cfg.train_dataloader.batch_size == 3
    assert cfg.train_dataloader.dataset.ann_file == 'ImageSets/trainval.txt'
    assert cfg.val_dataloader.dataset.ann_file == 'ImageSets/test.txt'
    assert cfg.val_evaluator.type == 'DOTAMetric'
    assert cfg.val_evaluator.iou_thrs == 0.5
    assert cfg.visualizer.vis_backends == [dict(type='LocalVisBackend')]
    assert cfg.randomness.seed == 20260712


def test_causal_switches_are_single_direction():
    c0 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c0_native_10e.py')
    c1 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c1_fusion_5e.py')
    repaired = Config.fromfile(
        CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py')
    c2 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c2_balanced_5e.py')
    c3 = Config.fromfile(CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c3_null_5e.py')
    assert not c0.model.decoder.layer_cfg.enable_semantic_fusion
    assert c1.model.decoder.layer_cfg.enable_semantic_fusion
    assert not c1.model.bbox_head.balanced_cfg.enabled
    assert repaired.model.decoder.layer_cfg.enable_semantic_fusion
    assert not repaired.model.decoder.layer_cfg.enable_density_capacity
    assert not repaired.model.bbox_head.balanced_cfg.enabled
    assert not repaired.model.decoder.enable_null_reservoir
    assert repaired.load_from.endswith(
        'hrsc_c0_native_10e/epoch_10.pth')
    assert repaired.work_dir.endswith(
        'hrsc_c1_parent_preserving_5e')
    assert c2.model.bbox_head.balanced_cfg.enabled
    assert not c2.model.decoder.enable_null_reservoir
    assert c3.model.decoder.enable_null_reservoir
    assert not c3.model.decoder.layer_cfg.enable_density_capacity


def test_parent_preserving_seed13_only_changes_seed_and_work_dir():
    seed12 = Config.fromfile(
        CONFIG_DIR / 'ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py')
    seed13 = Config.fromfile(
        CONFIG_DIR /
        'ov_capflow_swin-t_hrsc_c1_parent_preserving_seed13_5e.py')
    assert seed12.randomness.seed == 20260712
    assert seed13.randomness.seed == 20260713
    assert seed13.work_dir.endswith(
        'hrsc_c1_parent_preserving_seed13_5e')

    seed12_effective = seed12.to_dict()
    seed13_effective = seed13.to_dict()
    seed12_effective.pop('randomness')
    seed13_effective.pop('randomness')
    seed12_effective.pop('work_dir')
    seed13_effective.pop('work_dir')
    assert seed13_effective == seed12_effective


def test_hrsc_dataset_exposes_grounding_text_metadata():
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(
        CONFIG_DIR / 'grounding_dino_swin-t_hrsc_parent_10e.py')
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    data_info = dataset.get_data_info(0)
    assert data_info['text'] == ('ship', )
    assert data_info['custom_entities'] is True
