_base_ = [
    '../../../projects/GroundingDINO/configs/'
    'grounding_dino_swin-t_visdrone_base-set_adamw.py'
]

data_root = '/data1/zcy/datasets/DOTA2_1024_500/'
classes = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
metainfo = dict(classes=classes)
num_queries = 200
batch_size = 6

custom_imports = dict(
    imports=['projects.OVCapFlow.ov_capflow'], allow_failed_imports=False)
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True,
    static_graph=True)

model = dict(
    type='OVCapFlow',
    num_queries=num_queries,
    encoder=dict(num_cp=0),
    language_model=dict(
        name='/data1/zcy/LAEDINO/weights/bert-base-uncased'),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data/zcy/swin_tiny_patch4_window7_224.pth')),
    decoder=dict(
        enable_null_reservoir=False,
        layer_cfg=dict(
            enable_semantic_fusion=False,
            enable_density_capacity=False)),
    bbox_head=dict(
        type='OVCapFlowHead',
        num_classes=len(classes),
        balanced_cfg=dict(enabled=False),
        readout_cfg=dict(
            temperature=1.0, power=1.0, use_capacity=False)),
    density_loss_cfg=dict(weight=0.0),
    null_loss_cfg=dict(),
    test_cfg=dict(_delete_=True))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    batch_sampler=dict(
        type='DNQueryBudgetBatchSampler',
        num_matching_queries=num_queries,
        num_dn_queries=100,
        max_query_area=50000000,
        audit_path=(
            'work_dirs/ov_capflow_dotav2/audits/'
            'sampler_train_epoch.json')),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root=data_root,
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        return_classes=True))
val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root=data_root,
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        pipeline=val_pipeline,
        return_classes=True))
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = []
optim_wrapper = dict(accumulative_counts=1)

val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=0.5)
test_evaluator = val_evaluator
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=2,
        save_best='dota/mAP',
        rule='greater',
        save_last=True))
visualizer = dict(vis_backends=[dict(type='LocalVisBackend')])
log_processor = dict(by_epoch=True)
randomness = dict(seed=20260712, deterministic=False, diff_rank_seed=False)
work_dir = (
    'work_dirs/ov_capflow_dotav2/'
    'dotav2_c0_native_1e_recovery_static')
