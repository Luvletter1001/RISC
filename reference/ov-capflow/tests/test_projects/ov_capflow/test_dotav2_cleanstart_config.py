from pathlib import Path

from mmengine import Config


CONFIG_DIR = Path('configs/ov_capflow/dotav2')
BASE = 'ov_capflow_swin-t_dotav2_cleanstart_q600_base.py'
S0 = 'ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py'
S1 = 'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py'
S1_GROUPED = 'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py'
S1_GROUPED_SCALE1024 = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024.py')
S1_GROUPED_SCALE1024_BATCH1 = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024_batch1.py')
S1_GROUPED_SCALE1024_BATCH1_RARE4X = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py')
S1_GROUPED_SCALE1024_BATCH1_RARE4X_POSITION_SUPERVISED = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x_position_supervised.py')
S1_GROUPED_SCALE1024_BATCH1_TEXTLR1E5 = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_textlr1e5.py')
S1_HAUSDORFF_DN = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_hausdorff_dn.py')
S1_CHAMFER = 'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_chamfer.py'
FULL24 = 'ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py'
FULL24_SCALE1024_RARE4X_GPU89_B2 = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_'
    'full24e_scale1024_rare4x_gpu89_batch2.py')
FULL24_SCALE1024_RARE4X_GPU2389_B2_E6_RESUME = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_'
    'full24e_scale1024_rare4x_gpu2389_batch2_e6_resume.py')
FULL24_SCALE1024_RARE4X_GPU89_B2_MAXQA20M = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_'
    'full24e_scale1024_rare4x_gpu89_batch2_maxqa20m.py')
EXPECTED_CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
EXPECTED_NOVEL = ('airport', 'container-crane', 'helipad', 'helicopter')
FILTERED_CHECKPOINT = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_compatible.pth')
MANIFEST_SHA256 = (
    'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e')


def _load(name):
    return Config.fromfile(CONFIG_DIR / name)


def _mapping_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _mapping_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _mapping_keys(item)


def test_cleanstart_base_is_q600_raw_and_checkpoint_auditable():
    cfg = _load(BASE)
    assert cfg.model.type == 'OVCapFlow'
    assert cfg.model.num_queries == 600
    assert cfg.model.train_query_groups == 1
    assert cfg.model.backbone.init_cfg is None
    assert cfg.model.encoder.num_cp == 0
    assert cfg.model.bbox_head.type == 'OVCapFlowHead'
    assert cfg.model.bbox_head.matching_query_groups == 1
    assert cfg.model.bbox_head.contrastive_cfg.max_text_len == 256
    assert cfg.model.test_cfg == {}
    assert cfg.load_from == FILTERED_CHECKPOINT
    assert cfg.resume is False
    assert cfg.cleanstart_source_sha256 == (
        '3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799')
    assert cfg.cleanstart_provenance.endswith(
        'groundingdino_swint_ogc_q600_provenance.json')
    assert cfg.model_wrapper_cfg.static_graph is True
    assert cfg.model_wrapper_cfg.find_unused_parameters is True
    assert not cfg.get('custom_hooks', [])

    assert cfg.train_dataloader.batch_size == 1
    assert cfg.train_dataloader.batch_sampler.type == 'DNQueryBudgetBatchSampler'
    assert cfg.train_dataloader.batch_sampler.num_matching_queries == 600
    assert cfg.train_dataloader.batch_sampler.num_dn_queries == 100
    assert cfg.train_dataloader.dataset.ann_file == 'ss_train/annfiles/'
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.val_dataloader.dataset.ann_file == 'ss_val/annfiles/'
    assert cfg.val_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.val_dataloader.dataset.test_mode is True
    assert tuple(cfg.val_dataloader.dataset.metainfo.classes) == EXPECTED_CLASSES
    assert cfg.val_evaluator.type == 'DOTAMetric'
    assert cfg.val_evaluator.iou_thrs == 0.5

    keys = set(_mapping_keys(cfg.to_dict()))
    assert not keys.intersection({
        'nms', 'topk', 'max_per_img', 'multiclass_nms', 'rotated_nms'})


def test_cleanstart_prompt_protocol_is_shape_independent():
    cfg = _load(BASE)
    assert tuple(cfg.classes) == EXPECTED_CLASSES
    assert tuple(cfg.novel_classes) == EXPECTED_NOVEL
    assert tuple(cfg.base_classes) == tuple(
        label for label in EXPECTED_CLASSES if label not in EXPECTED_NOVEL)
    assert cfg.model.bbox_head.num_classes == 18
    assert cfg.model.bbox_head.type == 'OVCapFlowHead'
    assert cfg.prompt_protocol.classification == 'text_token_similarity'
    assert cfg.prompt_protocol.fixed_classifier_width is False
    assert cfg.prompt_protocol.separator == '. '


def test_s0_uses_real_manifest_subset_for_exactly_50_updates():
    cfg = _load(S0)
    assert cfg.subset_manifest_sha256 == MANIFEST_SHA256
    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert cfg.train_cfg.max_iters == 50
    assert cfg.train_cfg.val_interval == 50
    assert cfg.train_dataloader.dataset.data_root.endswith(
        'work_dirs/dotav2_cleanstart/subsets/seed20260715/')
    assert cfg.train_dataloader.dataset.ann_file == 's0/annfiles/'
    assert cfg.train_dataloader.dataset.data_prefix.img_path == 's0/images/'
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.param_scheduler == []
    assert cfg.train_dataloader.batch_size == 3
    assert cfg.optim_wrapper.accumulative_counts == 1
    assert (cfg.train_dataloader.batch_size *
            cfg.optim_wrapper.accumulative_counts) == 3
    assert cfg.resume is False


def test_s1_control_uses_locked_1600_400_protocol():
    cfg = _load(S1)
    assert cfg.subset_manifest_sha256 == MANIFEST_SHA256
    assert cfg.randomness.seed == 20260715
    assert cfg.train_cfg.type == 'VariableBatchEpochBasedTrainLoop'
    assert cfg.train_cfg.max_epochs == 12
    assert cfg.train_cfg.val_interval == 1
    assert cfg.train_dataloader.dataset.ann_file == (
        's1_train_all18/annfiles/')
    assert cfg.train_dataloader.dataset.data_prefix.img_path == (
        's1_train_all18/images/')
    assert cfg.val_dataloader.dataset.ann_file == 's1_val_all18/annfiles/'
    assert cfg.val_dataloader.dataset.data_prefix.img_path == (
        's1_val_all18/images/')
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.val_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert [item.type for item in cfg.param_scheduler] == [
        'LinearLR', 'CosineAnnealingLR']
    assert cfg.optim_wrapper.type == 'OptimWrapper'
    assert cfg.optim_wrapper.optimizer.type == 'AdamW'
    assert cfg.train_dataloader.batch_size == 4
    assert cfg.optim_wrapper.accumulative_counts == 2
    assert (cfg.train_dataloader.batch_size *
            cfg.optim_wrapper.accumulative_counts) == 8
    assert cfg.resume is False


def test_disabled_group_settings_are_accepted_and_stored(monkeypatch):
    from projects.OVCapFlow.ov_capflow.ov_capflow import OVCapFlow
    from projects.OVCapFlow.ov_capflow.ov_capflow_head import OVCapFlowHead
    from projects.GroundingDINO.groundingdino.grounding_dino import (
        RotatedGroundingDINO)
    from projects.GroundingDINO.groundingdino.grounding_dino_head import (
        RotatedGroundingDINOHead)

    monkeypatch.setattr(
        RotatedGroundingDINO, '__init__', lambda self, **kwargs: None)
    monkeypatch.setattr(
        RotatedGroundingDINOHead, '__init__', lambda self, **kwargs: None)
    detector = OVCapFlow(train_query_groups=1)
    head = OVCapFlowHead(matching_query_groups=1)
    assert detector.train_query_groups == 1
    assert head.matching_query_groups == 1


def test_s1_grouped_changes_only_training_query_geometry():
    control = _load(S1).to_dict()
    grouped = _load(S1_GROUPED).to_dict()
    assert grouped['model']['train_query_groups'] == 3
    assert grouped['model']['bbox_head']['matching_query_groups'] == 3
    assert grouped['train_dataloader']['batch_sampler'][
        'num_matching_queries'] == 1800
    assert grouped['work_dir'].endswith('s1_grouped')

    for cfg in (control, grouped):
        cfg['model'].pop('train_query_groups')
        cfg['model']['bbox_head'].pop('matching_query_groups')
        cfg['train_dataloader']['batch_sampler'].pop('num_matching_queries')
        cfg.pop('work_dir')
    assert grouped == control


def test_s1_grouped_scale1024_is_two_gpu_matched_resolution_screen():
    control = _load(S1_GROUPED).to_dict()
    candidate = _load(S1_GROUPED_SCALE1024).to_dict()

    assert tuple(candidate['physical_gpus']) == (8, 9)
    assert candidate['selected_world_size'] == 2
    assert candidate['randomness']['seed'] == 20260716
    assert candidate['train_dataloader']['batch_size'] == 2
    assert candidate['optim_wrapper']['accumulative_counts'] == 2
    assert (candidate['train_dataloader']['batch_size'] *
            candidate['selected_world_size'] *
            candidate['optim_wrapper']['accumulative_counts']) == 8
    assert candidate['train_dataloader']['batch_sampler'][
        'update_count_multiple'] == 2
    assert candidate['model']['backbone']['with_cp'] is False
    assert candidate['model_wrapper_cfg']['static_graph'] is False
    assert candidate['load_from'] == control['load_from']
    assert candidate['resume'] is False

    def resize_scales(pipeline):
        return [
            tuple(item['scale']) for item in pipeline
            if item['type'] == 'mmdet.Resize'
        ]

    assert resize_scales(candidate['train_pipeline']) == [(1024, 1024)]
    assert resize_scales(candidate['val_pipeline']) == [(1024, 1024)]
    assert resize_scales(
        candidate['train_dataloader']['dataset']['pipeline']) == [
            (1024, 1024)
        ]
    for loader in ('val_dataloader', 'test_dataloader'):
        assert resize_scales(candidate[loader]['dataset']['pipeline']) == [
            (1024, 1024)
        ]

    candidate.pop('physical_gpus')
    candidate.pop('selected_world_size')
    candidate.pop('matched_control')
    candidate['randomness'] = control['randomness']
    candidate['train_dataloader']['batch_size'] = control[
        'train_dataloader']['batch_size']
    candidate['train_dataloader']['batch_sampler'].pop(
        'update_count_multiple')
    candidate['train_dataloader']['batch_sampler']['audit_path'] = control[
        'train_dataloader']['batch_sampler']['audit_path']
    candidate['model']['backbone']['with_cp'] = control['model'][
        'backbone']['with_cp']
    candidate['model_wrapper_cfg']['static_graph'] = control[
        'model_wrapper_cfg']['static_graph']
    candidate['work_dir'] = control['work_dir']

    for key in ('train_pipeline', 'val_pipeline'):
        for item in candidate[key]:
            if item['type'] == 'mmdet.Resize':
                item['scale'] = (800, 800)
    for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        for item in candidate[loader]['dataset']['pipeline']:
            if item['type'] == 'mmdet.Resize':
                item['scale'] = (800, 800)

    assert candidate == control


def test_s1_grouped_scale1024_batch1_preserves_effective_batch_and_protocol():
    parent = _load(S1_GROUPED_SCALE1024).to_dict()
    recovery = _load(S1_GROUPED_SCALE1024_BATCH1).to_dict()

    assert tuple(recovery['physical_gpus']) == (8, 9)
    assert recovery['selected_world_size'] == 2
    assert recovery['train_dataloader']['batch_size'] == 1
    assert recovery['optim_wrapper']['accumulative_counts'] == 4
    assert (recovery['train_dataloader']['batch_size'] *
            recovery['selected_world_size'] *
            recovery['optim_wrapper']['accumulative_counts']) == 8
    assert recovery['train_dataloader']['batch_sampler'][
        'update_count_multiple'] == 4
    assert recovery['load_from'] == parent['load_from']
    assert recovery['resume'] is False
    assert recovery['work_dir'].endswith(
        's1_grouped_scale1024_seed20260716_gpu89_batch1')

    recovery.pop('recovery_of')
    recovery['train_dataloader']['batch_size'] = parent[
        'train_dataloader']['batch_size']
    recovery['train_dataloader']['batch_sampler'][
        'update_count_multiple'] = parent['train_dataloader'][
            'batch_sampler']['update_count_multiple']
    recovery['train_dataloader']['batch_sampler']['audit_path'] = parent[
        'train_dataloader']['batch_sampler']['audit_path']
    recovery['optim_wrapper']['accumulative_counts'] = parent[
        'optim_wrapper']['accumulative_counts']
    recovery['work_dir'] = parent['work_dir']
    assert recovery == parent


def test_a1_position_supervision_changes_only_target_rule_and_paths():
    control = _load(S1_GROUPED_SCALE1024_BATCH1_RARE4X).to_dict()
    candidate = _load(
        S1_GROUPED_SCALE1024_BATCH1_RARE4X_POSITION_SUPERVISED).to_dict()

    assert candidate['a1_parent'] == '8-T6-R-E12'
    assert candidate['a1_only_scientific_delta'] == (
        'rotated_iou_positive_classification_target')
    assert candidate['model']['bbox_head']['position_supervised_cfg'] == {
        'enabled': True}
    assert candidate['model']['num_queries'] == 600
    assert candidate['model']['train_query_groups'] == 3
    assert candidate['model']['bbox_head']['matching_query_groups'] == 3
    assert candidate['model']['decoder']['num_layers'] == 6
    assert candidate['model']['bbox_head']['balanced_cfg'] == {
        'enabled': False}
    assert candidate['model']['bbox_head']['loss_cls'] == {
        'type': 'mmdet.FocalLoss',
        'use_sigmoid': True,
        'gamma': 2.0,
        'alpha': 0.25,
        'loss_weight': 1.0,
    }
    assert candidate['model']['bbox_head']['loss_bbox'] == {
        'type': 'mmdet.L1Loss',
        'loss_weight': 5.0,
    }
    assert candidate['model']['bbox_head']['loss_iou'] == {
        'type': 'GDLoss',
        'loss_type': 'kld',
        'fun': 'log1p',
        'tau': 1,
        'sqrt': False,
        'loss_weight': 2.0,
    }
    assert candidate['model']['train_cfg']['assigner']['match_costs'] == [
        {
            'type': 'mmdet.BinaryFocalLossCost',
            'weight': 2.0,
        },
        {
            'type': 'RBoxL1Cost',
            'weight': 5.0,
            'box_format': 'xywha',
        },
        {
            'type': 'GDCost',
            'loss_type': 'kld',
            'fun': 'log1p',
            'tau': 1,
            'sqrt': False,
            'weight': 2.0,
        },
    ]
    assert candidate['model']['type'] == 'OVCapFlow'
    assert candidate['resume'] is False
    assert candidate['randomness']['seed'] == 20260716
    assert candidate['train_cfg']['max_epochs'] == 12
    assert candidate['work_dir'] == (
        'work_dirs/dotav2_cleanstart/'
        's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_'
        'position_supervised')
    assert candidate['train_dataloader']['batch_sampler']['audit_path'] == (
        'work_dirs/dotav2_cleanstart/audits/'
        's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_'
        'position_supervised_sampler.json')
    assert candidate['work_dir'] != control['work_dir']
    assert candidate['train_dataloader']['batch_sampler'][
        'audit_path'] != control['train_dataloader']['batch_sampler'][
            'audit_path']

    candidate.pop('a1_parent')
    candidate.pop('a1_only_scientific_delta')
    candidate['model']['bbox_head'].pop('position_supervised_cfg')
    candidate['train_dataloader']['batch_sampler']['audit_path'] = control[
        'train_dataloader']['batch_sampler']['audit_path']
    candidate['work_dir'] = control['work_dir']
    assert candidate == control


def test_t4_textlr1e5_changes_only_language_model_lr_and_paths():
    control = _load(S1_GROUPED_SCALE1024_BATCH1).to_dict()
    candidate = _load(S1_GROUPED_SCALE1024_BATCH1_TEXTLR1E5).to_dict()

    assert candidate['t4_parent'] == '8-T2-R-B1-E12'
    assert candidate['optim_wrapper']['optimizer']['lr'] == 1e-4
    assert candidate['optim_wrapper']['paramwise_cfg']['custom_keys'][
        'language_model'] == {'lr_mult': 0.1}
    assert candidate['work_dir'].endswith(
        's1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5')

    candidate.pop('t4_parent')
    candidate.pop('t4_only_scientific_delta')
    candidate['optim_wrapper']['paramwise_cfg']['custom_keys'].pop(
        'language_model')
    candidate['train_dataloader']['batch_sampler']['audit_path'] = control[
        'train_dataloader']['batch_sampler']['audit_path']
    candidate['work_dir'] = control['work_dir']
    assert candidate == control


def test_s1_hausdorff_dn_changes_only_training_geometry_package():
    control = _load(S1).to_dict()
    candidate = _load(S1_HAUSDORFF_DN).to_dict()
    assert candidate['model']['train_query_groups'] == 1
    assert candidate['model']['bbox_head']['matching_query_groups'] == 1
    assert candidate['model']['bbox_head']['adaptive_dn_cfg'] == {
        'enabled': True}
    assert candidate['model']['dn_cfg']['angle_noise_scale'] == 0.25
    match_costs = candidate['model']['train_cfg']['assigner']['match_costs']
    dn_costs = candidate['model']['train_cfg']['dn_assigner']['match_costs']
    assert any(item['type'] == 'HausdorffCost' for item in match_costs)
    assert any(item['type'] == 'HausdorffCost' for item in dn_costs)
    assert candidate['work_dir'].endswith('s1_hausdorff_dn')

    candidate['model']['bbox_head'].pop('adaptive_dn_cfg')
    candidate['model']['dn_cfg'].pop('angle_noise_scale')
    candidate['model']['train_cfg'] = control['model']['train_cfg']
    candidate.pop('work_dir')
    control.pop('work_dir')
    assert candidate == control

    forbidden = {'teacher', 'student', 'distill', 'pseudo', 'rpn', 'roi'}
    assert not forbidden.intersection(set(_mapping_keys(candidate)))


def test_s1_chamfer_replaces_only_l1_matching_distance():
    control = _load(S1).to_dict()
    candidate = _load(S1_CHAMFER).to_dict()
    control_costs = control['model']['train_cfg']['assigner']['match_costs']
    candidate_costs = candidate['model']['train_cfg']['assigner'][
        'match_costs']

    assert [cost['type'] for cost in candidate_costs] == [
        'mmdet.BinaryFocalLossCost', 'ChamferCost', 'GDCost']
    assert candidate_costs[1]['weight'] == 5.0
    assert candidate_costs[1]['box_format'] == 'xywha'
    assert candidate['work_dir'].endswith('s1_chamfer')

    candidate_costs[1] = control_costs[1]
    candidate.pop('work_dir')
    control.pop('work_dir')
    assert candidate == control

    forbidden = {'teacher', 'student', 'distill', 'pseudo', 'rpn', 'roi'}
    assert not forbidden.intersection(set(_mapping_keys(candidate)))


def test_full24_promotes_grouped_recipe_and_raw_full_mouth_only():
    cfg = _load(FULL24)

    assert cfg.promoted_s1_winner == '8-S1-G'
    assert tuple(cfg.physical_gpus) == (4, 5, 6, 7)
    assert cfg.expected_train_tiles == 47294
    assert cfg.expected_raw_val_tiles == 13833
    assert cfg.raw_validation_filter_empty_gt is False

    assert cfg.model.train_query_groups == 3
    assert cfg.model.bbox_head.matching_query_groups == 3
    assert cfg.model.backbone.with_cp is False
    assert cfg.model_wrapper_cfg.static_graph is False
    assert cfg.train_dataloader.batch_sampler.num_matching_queries == 1800
    assert cfg.train_dataloader.batch_sampler.max_query_area == 50000000
    assert cfg.train_dataloader.batch_sampler.update_count_multiple == 2
    assert cfg.train_dataloader.batch_size == 4
    assert cfg.optim_wrapper.accumulative_counts == 2
    assert (cfg.train_dataloader.batch_size * len(cfg.physical_gpus) *
            cfg.optim_wrapper.accumulative_counts) == 32

    assert cfg.train_dataloader.dataset.ann_file == 'ss_train/annfiles/'
    assert cfg.train_dataloader.dataset.data_prefix.img_path == (
        'ss_train/images/')
    assert cfg.val_dataloader.dataset.ann_file == 'ss_val/annfiles/'
    assert cfg.val_dataloader.dataset.data_prefix.img_path == 'ss_val/images/'
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.val_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.val_dataloader.dataset.test_mode is True

    assert cfg.train_cfg.type == 'VariableBatchEpochBasedTrainLoop'
    assert cfg.train_cfg.max_epochs == 24
    assert cfg.train_cfg.val_interval == 1
    assert cfg.train_cfg.dynamic_intervals == [(3, 6)]
    assert cfg.param_scheduler[-1].type == 'CosineAnnealingLR'
    assert cfg.param_scheduler[-1].end == 24
    checkpoint = cfg.default_hooks.checkpoint
    assert checkpoint.type == 'MilestoneCheckpointHook'
    assert tuple(checkpoint.milestones) == (1, 6, 12, 18, 24)
    assert cfg.resume is False
    assert cfg.load_from == FILTERED_CHECKPOINT

    forbidden = {
        'teacher', 'student', 'distill', 'pseudo', 'rpn', 'roi', 'nms',
        'topk', 'max_per_img', 'multiclass_nms', 'rotated_nms'}
    assert not forbidden.intersection(set(_mapping_keys(cfg.to_dict())))


def test_full24_scale1024_rare4x_preserves_raw_protocol_and_effective_batch():
    cfg = _load(FULL24_SCALE1024_RARE4X_GPU89_B2)

    assert cfg.promoted_screen_winner == '8-T6-R-E12'
    assert cfg.promoted_scientific_deltas == (
        'scale_800_to_1024', 'fixed_size_rare_positive_exposure_4x')
    assert tuple(cfg.physical_gpus) == (8, 9)
    assert cfg.expected_train_tiles == 47294
    assert cfg.expected_raw_val_tiles == 13833
    assert cfg.raw_validation_filter_empty_gt is False
    assert cfg.rare_repeat_factor == 4
    assert cfg.rare_unique_train_images == 824
    assert cfg.rare_train_exposures == 3296
    assert cfg.replaced_empty_images == 2472
    assert cfg.full_train_empty_images == 20103

    assert cfg.train_pipeline[3].scale == (1024, 1024)
    assert cfg.val_pipeline[1].scale == (1024, 1024)
    assert cfg.train_dataloader.batch_size == 2
    assert cfg.train_dataloader.batch_sampler.num_matching_queries == 1800
    assert cfg.train_dataloader.batch_sampler.max_query_area == 50000000
    assert cfg.train_dataloader.batch_sampler.update_count_multiple == 8
    assert cfg.optim_wrapper.accumulative_counts == 8
    assert (cfg.train_dataloader.batch_size * len(cfg.physical_gpus) *
            cfg.optim_wrapper.accumulative_counts) == 32

    rare_root = (
        'work_dirs/dotav2_cleanstart/full_rare4x_seed20260718/train/')
    assert cfg.train_dataloader.dataset.data_root == rare_root
    assert cfg.train_dataloader.dataset.ann_file == 'annfiles/'
    assert cfg.train_dataloader.dataset.data_prefix.img_path == 'images/'
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.val_dataloader.dataset.data_root == (
        '/data1/zcy/datasets/DOTA2_1024_500/')
    assert cfg.val_dataloader.dataset.ann_file == 'ss_val/annfiles/'
    assert cfg.val_dataloader.dataset.data_prefix.img_path == 'ss_val/images/'
    assert cfg.val_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.val_dataloader.dataset.test_mode is True

    assert cfg.randomness.seed == 20260716
    assert cfg.train_cfg.max_epochs == 24
    assert cfg.train_cfg.dynamic_intervals == [(3, 6)]
    assert cfg.param_scheduler[-1].end == 24
    assert tuple(cfg.default_hooks.checkpoint.milestones) == (
        1, 6, 12, 18, 24)
    assert cfg.resume is False
    assert cfg.load_from == FILTERED_CHECKPOINT

    forbidden = {
        'teacher', 'student', 'distill', 'pseudo', 'rpn', 'roi', 'nms',
        'topk', 'max_per_img', 'multiclass_nms', 'rotated_nms'}
    assert not forbidden.intersection(set(_mapping_keys(cfg.to_dict())))


def test_full24_scale1024_rare4x_epoch6_resume_uses_four_gpu_matched_batch():
    parent = _load(FULL24_SCALE1024_RARE4X_GPU89_B2).to_dict()
    resume = _load(FULL24_SCALE1024_RARE4X_GPU2389_B2_E6_RESUME).to_dict()

    assert tuple(resume['physical_gpus']) == (2, 3, 8, 9)
    assert resume['selected_world_size'] == 4
    assert resume['continuation_from_epoch'] == 6
    assert resume['train_dataloader']['batch_size'] == 2
    assert resume['optim_wrapper']['accumulative_counts'] == 4
    assert resume['train_dataloader']['batch_sampler'][
        'update_count_multiple'] == 4
    assert (resume['train_dataloader']['batch_size'] *
            resume['selected_world_size'] *
            resume['optim_wrapper']['accumulative_counts']) == 32
    assert resume['load_from'].endswith('epoch_6_world4_iterrebased.pth')
    assert resume['resume'] is True
    assert resume['work_dir'] == parent['work_dir']
    assert resume['val_dataloader'] == parent['val_dataloader']
    assert resume['val_evaluator'] == parent['val_evaluator']
    assert resume['model'] == parent['model']


def test_full24_scale1024_rare4x_maxqa20m_is_resource_only_fallback():
    control = _load(FULL24_SCALE1024_RARE4X_GPU89_B2).to_dict()
    fallback = _load(FULL24_SCALE1024_RARE4X_GPU89_B2_MAXQA20M).to_dict()

    assert fallback['resource_fallback_of'] == '8-T7-F'
    assert fallback['train_dataloader']['batch_size'] == 2
    assert fallback['optim_wrapper']['accumulative_counts'] == 8
    assert fallback['train_dataloader']['batch_sampler'][
        'max_query_area'] == 20_000_000
    assert fallback['train_dataloader']['batch_sampler'][
        'update_count_multiple'] == 8
    assert (fallback['train_dataloader']['batch_size'] *
            len(fallback['physical_gpus']) *
            fallback['optim_wrapper']['accumulative_counts']) == 32

    fallback.pop('resource_fallback_of')
    fallback['train_dataloader']['batch_sampler']['max_query_area'] = control[
        'train_dataloader']['batch_sampler']['max_query_area']
    fallback['train_dataloader']['batch_sampler']['audit_path'] = control[
        'train_dataloader']['batch_sampler']['audit_path']
    fallback['work_dir'] = control['work_dir']
    assert fallback == control
