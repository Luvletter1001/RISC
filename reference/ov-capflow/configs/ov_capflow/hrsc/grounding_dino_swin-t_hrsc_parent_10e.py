_base_ = [
    '../../../projects/GroundingDINO/configs/'
    'grounding_dino_swin-t_visdrone_base-set_adamw.py'
]

data_root = '/data/zcy/dataset/HRSC_unzip/'
num_queries = 600
batch_size = 3

model = dict(
    num_queries=num_queries,
    encoder=dict(num_cp=0),
    language_model=dict(
        name='/data1/zcy/LAEDINO/weights/bert-base-uncased'),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data/zcy/swin_tiny_patch4_window7_224.pth')),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(max_per_img=num_queries))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip', prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='HRSCDataset',
        data_root=data_root,
        ann_file='ImageSets/trainval.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        return_classes=True))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='HRSCDataset',
        data_root=data_root,
        ann_file='ImageSets/test.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True,
        pipeline=val_pipeline,
        return_classes=True))
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=10, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=10, by_epoch=True,
        milestones=[8], gamma=0.1),
]
optim_wrapper = dict(accumulative_counts=2)
val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=0.5)
test_evaluator = val_evaluator
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=5, max_keep_ckpts=3,
        save_best='dota/mAP', rule='greater', save_last=True))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)
log_processor = dict(by_epoch=True)
randomness = dict(seed=20260712, deterministic=False, diff_rank_seed=False)
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_p0_parent_10e'
