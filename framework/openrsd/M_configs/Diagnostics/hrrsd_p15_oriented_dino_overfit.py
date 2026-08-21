custom_imports = dict(
    imports=[
        'M_AD.models.detectors.p15_oriented_dino',
        'M_AD.models.dense_heads.p15_oriented_dino_head',
        'M_AD.models.layers.transformer.dinor_layersv2',
        'M_AD.models.layers.transformer.deformable_detr_layers',
    ],
    allow_failed_imports=False)

data_root = '/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/'
dataset_type = 'DOTADataset'
img_scale = (800, 800)
file_client_args = dict(backend='disk')
class_name = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
num_classes = len(class_name)

work_dir = 'work_dirs/p15_scod_hrrsd_20260623/overfit20'
default_scope = 'mmrotate'
log_level = 'INFO'
load_from = None
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

model = dict(
    type='P15SupportConditionedOrientedDINO',
    num_queries=300,
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=4,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='P15OrientedDINOSetHead',
        num_classes=num_classes,
        embed_dims=256,
        num_queries=300,
        support_query_scale=1.0,
        max_per_img=300),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=0.4,
        angle_noise_scale=0.2,
        group_cfg=dict(dynamic=True, num_dn_queries=100)),
    train_cfg=dict(),
    test_cfg=dict(max_per_img=300))

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        indices=20,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        img_shape=img_scale,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=20,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        img_shape=img_scale,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

max_epochs = 20
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = []
auto_scale_lr = dict(enable=False, base_batch_size=1)
