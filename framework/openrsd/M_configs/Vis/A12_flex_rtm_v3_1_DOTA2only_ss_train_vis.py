import os

_base_ = [
    '/data1/zcy/OpenRSD/results/MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train/A12_flex_rtm_v3_1_DOTA2only_ss_train.py'
]

vis_data_root = os.environ.get(
    'OPENRSD_VIS_DATA_ROOT',
    '/data1/zcy/OpenRSD/vis/P0092__1024__824___0/dataset')
vis_img_dir = os.environ.get('OPENRSD_VIS_IMG_DIR', 'images')
vis_ann_dir = os.environ.get('OPENRSD_VIS_ANN_DIR', 'annfiles')
vis_batch_size = int(os.environ.get('OPENRSD_VIS_BATCH_SIZE', '8'))
vis_num_workers = int(os.environ.get('OPENRSD_VIS_NUM_WORKERS', '4'))

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'M_AD.engine.runner.meta_remove_runer',
        'M_AD.engine.hooks.pred_only_visualization_hook',
        'M_AD.datasets.transforms.formatting',
        'M_AD.datasets.transforms.loading',
        'M_AD.datasets.transforms.transforms',
        'M_AD.datasets.dota_online_v1',
        'M_AD.datasets.samplers.one_task_sampler',
        'M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner',
        'M_AD.models.detectors.Flex_Rtmdet_v3_1_formal',
        'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1',
        'M_AD.models.roi_heads.CLIP_VP_head_v1',
        'M_AD.models.necks.promopt_cspnext_pafpn',
        'M_AD.evaluation.metrics.detail_dota_metric',
    ])

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='PredOnlyDetVisualizationHook'))

vis_metainfo = dict(
    classes=[
        'airport',
        'baseball-diamond',
        'basketball-court',
        'bridge',
        'container-crane',
        'ground-track-field',
        'harbor',
        'helicopter',
        'helipad',
        'large-vehicle',
        'plane',
        'roundabout',
        'ship',
        'small-vehicle',
        'soccer-ball-field',
        'storage-tank',
        'swimming-pool',
        'tennis-court',
    ],
    # Keep the finetuned visualization colors aligned with the legacy
    # /vis outputs from SimpleRun/step1_inference.py.
    palette=[
        (80, 220, 60),
        (80, 220, 60),
        (80, 220, 60),
        (80, 220, 60),
        (180, 160, 40),
        (220, 120, 40),
        (255, 90, 90),
        (90, 220, 220),
        (90, 220, 220),
        (255, 90, 90),
        (40, 190, 240),
        (230, 80, 180),
        (170, 90, 255),
        (170, 90, 255),
        (40, 190, 240),
        (255, 90, 90),
        (40, 190, 240),
        (70, 170, 120),
    ])

vis_test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
    dict(
        type='ConvertBoxTypeSafe',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputsMM',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]

test_dataloader = dict(
    batch_size=vis_batch_size,
    num_workers=vis_num_workers,
    persistent_workers=vis_num_workers > 0,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADatasetOnline',
        data_root=vis_data_root,
        ann_file=vis_ann_dir,
        data_prefix=dict(img_path=vis_img_dir),
        img_shape=(1024, 1024),
        metainfo=vis_metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        pipeline=vis_test_pipeline))

val_dataloader = test_dataloader
test_evaluator = []
val_evaluator = test_evaluator
