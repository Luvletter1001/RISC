_base_ = (
    '../../work_dirs/p16_generalization_hrrsd_20260625/'
    'p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5/'
    'hrrsd_p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5.py')

dota2_classes = [
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
]

data_root = '/data1/zcy/datasets/DOTA2_1024_500/'
num_classes = 18
metainfo = dict(classes=dota2_classes, palette=[(220, 20, 60)])
img_scale = (800, 800)
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs'),
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

work_dir = (
    'work_dirs/p21_strict_e2e_dota2_subset_20260626/'
    'p21a_train500_resnet50_b32x4_v1')

load_from = None
resume = False
max_epochs = 80

model = dict(
    bbox_head=dict(
        num_classes=num_classes,
        num_queries=200,
        max_per_img=200),
    num_queries=200,
    test_cfg=dict(max_per_img=200))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='dota/mAP',
        rule='greater',
        max_keep_ckpts=10,
        save_last=True))

train_dataloader = dict(
    _delete_=True,
    batch_sampler=None,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/'),
        metainfo=metainfo,
        img_shape=img_scale,
        indices=500,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/'),
        metainfo=metainfo,
        img_shape=img_scale,
        indices=500,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

auto_scale_lr = dict(enable=False, base_batch_size=128)
randomness = dict(seed=3407, deterministic=False)
