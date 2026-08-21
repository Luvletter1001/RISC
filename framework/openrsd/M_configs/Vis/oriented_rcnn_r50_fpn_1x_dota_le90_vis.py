import os

_base_ = ['../../mmrotate_configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py']

vis_data_root = os.environ.get(
    'OPENRSD_VIS_DATA_ROOT',
    '/data1/zcy/OpenRSD/vis/P0092__1024__824___0/dataset')
vis_img_dir = os.environ.get('OPENRSD_VIS_IMG_DIR', 'images')
vis_ann_dir = os.environ.get('OPENRSD_VIS_ANN_DIR', 'annfiles')
vis_batch_size = int(os.environ.get('OPENRSD_VIS_BATCH_SIZE', '4'))
vis_num_workers = int(os.environ.get('OPENRSD_VIS_NUM_WORKERS', '4'))

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'M_AD.engine.hooks.pred_only_visualization_hook',
        'M_AD.datasets.transforms.formatting',
        'M_AD.datasets.transforms.loading',
        'M_AD.datasets.transforms.transforms',
        'M_AD.datasets.dota_online_v1',
    ])

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PredOnlyDetVisualizationHook'))

model = dict(backbone=dict(init_cfg=None))

vis_metainfo = dict(
    classes=[
        'plane',
        'baseball-diamond',
        'bridge',
        'ground-track-field',
        'small-vehicle',
        'large-vehicle',
        'ship',
        'tennis-court',
        'basketball-court',
        'storage-tank',
        'soccer-ball-field',
        'roundabout',
        'harbor',
        'swimming-pool',
        'helicopter',
    ],
    palette=[
        (40, 190, 240),
        (80, 220, 60),
        (80, 220, 60),
        (220, 120, 40),
        (170, 90, 255),
        (255, 90, 90),
        (170, 90, 255),
        (70, 170, 120),
        (80, 220, 60),
        (255, 90, 90),
        (40, 190, 240),
        (230, 80, 180),
        (255, 90, 90),
        (40, 190, 240),
        (90, 220, 220),
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
