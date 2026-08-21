_base_ = [
    '/data1/zcy/OpenRSD/mmrotate_configs/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota.py'
]

data_root = '/data1/zcy/OpenRSD/data/DOTA1_1024_500/'

model = dict(
    crop_size=(256, 256),
    backbone=dict(init_cfg=None),
    bbox_head=dict(crop_size=(256, 256)))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(256, 256), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        indices=8,
        pipeline=train_pipeline))

val_cfg = None
val_dataloader = None
val_evaluator = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=999)

default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(interval=999))

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05))

work_dir = '/data1/zcy/OpenRSD/work_dirs/h2rbox_smoke_dota1'
