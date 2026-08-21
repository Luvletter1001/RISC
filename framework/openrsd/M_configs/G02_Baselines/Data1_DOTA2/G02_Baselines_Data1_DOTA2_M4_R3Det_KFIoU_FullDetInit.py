_base_ = './G02_Baselines_Data1_DOTA2_M4_R3Det_KFIoU.py'

# Use a DOTA-trained R3Det-KFIoU detector checkpoint. The checkpoint's
# 15-class classification tensors were removed because this DOTA2 config has
# 18 classes.
load_from = '/data1/zcy/OpenRSD/weights/r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d_mmrotate1xkeys_no_cls.pth'

img_scale = (1024, 1024)
train_batch_size = 2
max_epochs = 24
val_interval = 4
ckpt_interval = 4

file_client_args = dict(backend='disk')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[11, 15]),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
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
                   'scale_factor'))
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
default_hooks = dict(checkpoint=dict(interval=ckpt_interval))

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/'),
        img_shape=img_scale,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        img_shape=img_scale,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

work_dir = 'work_dirs/r3det_kfiou_r50_dotav2_full_det_init_1024_bs2_24e'
